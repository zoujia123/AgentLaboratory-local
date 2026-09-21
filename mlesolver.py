import ast
import hashlib
import json
import re
from pathlib import Path

from agent_runtime import AgentRuntime, ToolRegistry
from prompts import (
    experiment_reward_prompt,
    experiment_solver_prompt,
    supplementary_experiment_solver_prompt,
)
from tools import execute_code, execution_capabilities
from run_logging import current_prompt_recorder, current_run_logger


def _compact_solver_traceback(traceback_text):
    """Keep user frames and exception chains; shorten recognized library frame runs."""
    if len(traceback_text) <= 2500:
        return traceback_text
    lines = traceback_text.splitlines(keepends=True)
    frame = re.compile(r'^\s*File "([^"]+)", line \d+(?:, in .*)?\s*$')
    chunks = []
    index = 0
    while index < len(lines):
        match = frame.match(lines[index].rstrip("\r\n"))
        if match is None:
            chunks.append((False, lines[index]))
            index += 1
            continue
        start = index
        index += 1
        while index < len(lines):
            line = lines[index]
            if frame.match(line.rstrip("\r\n")) or not line.strip() or not line[0].isspace():
                break
            index += 1
        path = match.group(1).replace("\\", "/").lower()
        library = "/site-packages/" in path or "/dist-packages/" in path
        chunks.append((library, "".join(lines[start:index])))
    rendered = []
    index = 0
    omitted = 0
    while index < len(chunks):
        if not chunks[index][0]:
            rendered.append(chunks[index][1])
            index += 1
            continue
        start = index
        while index < len(chunks) and chunks[index][0]:
            index += 1
        # Retain the innermost library frame of each contiguous run.
        count = index - start - 1
        if count:
            omitted += count
            rendered.append(f"  ... {count} library frames omitted ...\n")
        rendered.append(chunks[index - 1][1])
    if not omitted:
        return traceback_text
    compact = "".join(rendered) + "\n[Full execution traceback recorded in workflow.log and prompts.jsonl.]"
    return compact if len(compact) < len(traceback_text) else traceback_text


def _solver_execution_feedback(result, phase):
    """Archive the original tool result before shortening its model-facing copy."""
    original = result.get("traceback") or ""
    compact = _compact_solver_traceback(original)
    if compact == original:
        return result
    current_run_logger().debug(
        f"[MLESolverAgent][{phase}] FULL EXECUTION DIAGNOSTIC\n"
        + json.dumps(result, ensure_ascii=False, indent=2, default=str)
    )
    current_prompt_recorder().record({
        "event": "solver_execution_diagnostic",
        "agent": "MLESolverAgent",
        "phase": phase,
        "code_hash": result.get("code_hash"),
        "result": result,
    })
    return {**result, "traceback": compact}


def _schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description):
    return {"type": "string", "description": description}


# 提前检测 LLM 生成的实验代码有没有语法问题
def _python_syntax_error(code):
    try:
        ast.parse(str(code))
    except SyntaxError as exc:
        return f"Generated experiment code is invalid Python: {exc}"
    return None


def _code_hash(code):
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()


def _execution_failure_category(output):
    text = str(output or "").casefold()
    patterns = (
        ("non_numeric_model_input", ("could not convert string to float",)),
        ("tensor_shape_mismatch", ("shapes cannot be multiplied", "size mismatch")),
        ("resource_exhaustion", ("not enough memory", "out of memory")),
        ("data_representation_error", ("cannot handle this data type",)),
        ("unavailable_model_asset", ("from_pretrained", "can't load tokenizer")),
        ("unavailable_dependency", ("no module named", "cannot import name")),
        ("seed_policy", ("seed policy violations",)),
        ("indexing_error", ("index out of range", "not found in axis")),
    )
    for category, fragments in patterns:
        if any(fragment in text for fragment in fragments):
            return category
    return "code_execution_error"


def _failure_repair_hint(category):
    return {
        "non_numeric_model_input": (
            "Build one consistent numeric feature representation for every required "
            "split. Encode or exclude categorical source columns before all estimators; "
            "do not patch only one model call."
        ),
        "tensor_shape_mismatch": (
            "Trace tensor shapes through every layer and replace the complete model "
            "definition when necessary."
        ),
        "resource_exhaustion": (
            "Reduce data, model, and intermediate tensor sizes before retrying; verify "
            "broadcasting shapes explicitly."
        ),
        "data_representation_error": (
            "Inspect the actual array layout and dtype, then rebuild the complete "
            "conversion pipeline consistently."
        ),
        "unavailable_model_asset": (
            "Use a model trained from scratch with available local modules; no "
            "pretrained assets are materialized."
        ),
        "unavailable_dependency": (
            "Rewrite with modules and public APIs listed in the execution environment."
        ),
        "seed_policy": "Use the exact symbolic fixed seed required by the seed policy.",
        "indexing_error": (
            "Inspect the authoritative schema and replace brittle positional or string "
            "splitting assumptions."
        ),
        "code_execution_error": (
            "Use the traceback to make a substantive correction before retrying."
        ),
        "timeout": "Reduce computation or data size; execution exceeded its time limit.",
        "execution_infrastructure_error": (
            "Execution did not return a reliable result. Inspect the environment or "
            "worker failure; this does not establish a Python syntax error."
        ),
    }[category]


def _record_execution_result(
    current, fingerprint, output, success, error_type=None, error=None, traceback_text="",
):
    current.update(output=output, success=success, code_hash=fingerprint)
    if success:
        current.update(
            last_failure_category=None,
            consecutive_failure_category_count=0,
            force_replace=False,
        )
        return None, None
    infrastructure_error = error_type in {
        "process_exit", "missing_result", "invalid_result", "result_transport_error",
        "execution_environment_error", "worker_setup_error", "result_collection_error",
    }
    if not infrastructure_error:
        current["failed_hashes"].add(fingerprint)
    diagnostic_text = "\n".join(text for text in (error, traceback_text) if text)
    category = (
        "execution_infrastructure_error" if infrastructure_error
        else "timeout" if error_type == "timeout"
        else _execution_failure_category(diagnostic_text)
    )
    if category == current.get("last_failure_category"):
        count = current.get("consecutive_failure_category_count", 0) + 1
    else:
        count = 1
    current.update(
        last_failure_category=category,
        consecutive_failure_category_count=count,
        force_replace=count >= 2 and category not in {
            "code_execution_error", "execution_infrastructure_error",
        },
    )
    return category, _failure_repair_hint(category)


class MLESolverAgent:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key

    def solve(self, state, workdir):
        current = {
            "code": "",
            "output": "",
            "success": None,
            "code_hash": None,
            "failed_hashes": set(),
            "last_failure_category": None,
            "consecutive_failure_category_count": 0,
            "force_replace": False,
        }

        def execute_current_experiment():
            if not current["code"]:
                return {"success": False, "error": "Create experiment code first"}
            fingerprint = _code_hash(current["code"])
            if fingerprint in current["failed_hashes"]:
                return {
                    "success": False,
                    "error": (
                        "Experiment code is unchanged since its previous failed "
                        "execution. Edit or replace it before retrying."
                    ),
                    "code_hash": fingerprint,
                }
            combined = state.dataset_code + "\n\n" + current["code"]
            execution = execute_code(
                combined,
                timeout=600,
                workdir=workdir,
                return_details=True,
            )
            output = execution["output"]
            success = execution["success"]
            category, repair_hint = _record_execution_result(
                current,
                fingerprint,
                output,
                success,
                execution["error_type"],
                error=execution["error"],
                traceback_text=execution["traceback"],
            )
            return _solver_execution_feedback({
                "success": success,
                "error_type": execution["error_type"],
                "error": execution["error"],
                "traceback": execution["traceback"],
                "exit_code": execution["exit_code"],
                "output": output,
                "code_hash": fingerprint,
                "failure_category": category,
                "repair_hint": repair_hint,
                "same_category_count": current[
                    "consecutive_failure_category_count"
                ],
                "full_replacement_required": current["force_replace"],
            }, "running experiments")

        def replace_experiment_code(code):
            syntax_error = _python_syntax_error(code)
            if syntax_error:
                return {"success": False, "error": syntax_error}
            fingerprint = _code_hash(code)
            if fingerprint in {
                current.get("code_hash"),
            } or fingerprint in current["failed_hashes"]:
                return {
                    "success": False,
                    "error": (
                        "Replacement experiment code is unchanged from the current "
                        "or previously failed draft. Make a substantive correction."
                    ),
                    "code_hash": fingerprint,
                }
            current.update(
                code=code,
                output="",
                success=None,
                code_hash=fingerprint,
                force_replace=False,
            )
            return {
                "success": True,
                "message": "Experiment code replaced; automatic execution is next",
                "code_hash": fingerprint,
            }

        def edit_experiment_code(old_text, new_text):
            if not current["code"]:
                return {"success": False, "error": "Create experiment code first"}
            occurrences = current["code"].count(old_text)
            if occurrences != 1:
                return {
                    "success": False,
                    "error": f"old_text must match exactly once; matched {occurrences}",
                }
            edited = current["code"].replace(old_text, new_text, 1)
            syntax_error = _python_syntax_error(edited)
            if syntax_error:
                return {"success": False, "error": syntax_error}
            fingerprint = _code_hash(edited)
            if fingerprint in {
                current.get("code_hash"),
            } or fingerprint in current["failed_hashes"]:
                return {
                    "success": False,
                    "error": (
                        "The edited experiment code matches the current or a "
                        "previously failed version"
                    ),
                    "code_hash": fingerprint,
                }
            current.update(
                code=edited,
                output="",
                success=None,
                code_hash=fingerprint,
            )
            return {
                "success": True,
                "message": "Experiment code edited; automatic execution is next",
                "code_hash": fingerprint,
            }

        def execute_experiment_code():
            return execute_current_experiment()

        def submit_experiment():
            if not current["code"]:
                return {"success": False, "error": "Create experiment code first"}
            if current["success"] is not True:
                return {
                    "success": False,
                    "error": "Execute experiment code successfully before submitting",
                }
            return {
                "success": True,
                "code": current["code"],
                "output": current["output"],
            }

        def available_tools():
            if not current["code"]:
                return {"replace_experiment_code"}
            if current["success"] is True:
                return {"submit_experiment"}
            if current["success"] is False:
                if current["last_failure_category"] == "execution_infrastructure_error":
                    return {"execute_experiment_code", "edit_experiment_code", "replace_experiment_code"}
                if current["force_replace"]:
                    return {"replace_experiment_code"}
                return {"edit_experiment_code", "replace_experiment_code"}
            return {"execute_experiment_code"}

        registry = ToolRegistry()
        registry.register(
            "replace_experiment_code",
            (
                "Create or completely replace the current experiment code. The "
                "framework automatically executes a valid replacement and returns "
                "its real execution result in the same tool response."
            ),
            _schema(
                {"code": _string("Complete experiment-only Python code")},
                ["code"],
            ),
            replace_experiment_code,
        ).register(
            "edit_experiment_code",
            (
                "Replace one exact fragment in the current experiment code. The "
                "framework automatically executes a valid edit and returns its real "
                "execution result in the same tool response."
            ),
            _schema(
                {
                    "old_text": _string("Exact unique code fragment to replace"),
                    "new_text": _string("Replacement code fragment"),
                },
                ["old_text", "new_text"],
            ),
            edit_experiment_code,
        ).register(
            "execute_experiment_code",
            (
                "Retry dataset preparation followed by the unchanged current "
                "experiment code after an execution-infrastructure failure."
            ),
            _schema({}, []),
            execute_experiment_code,
        ).register(
            "submit_experiment",
            "Submit the most recent successfully executed experiment.",
            _schema({}, []),
            submit_experiment,
        )
        runtime = AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name="MLESolverAgent",
            phase="running experiments",
            max_tool_steps=20,
        )
        result = runtime.run(
            system_prompt=experiment_solver_prompt(),
            user_prompt=(
                f"Topic: {state.topic}\nPlan: {state.plan}\n"
                f"Selected dataset: {state.dataset_selection}\n"
                f"Fixed numerical seeds: "
                f"{json.dumps(getattr(state, 'seeds', {}), sort_keys=True)}\n"
                f"Execution environment: "
                f"{json.dumps(execution_capabilities(), ensure_ascii=True)}\n"
                f"Prepared dataset manifest: "
                f"{json.dumps(getattr(state, 'dataset_manifest', {}), ensure_ascii=True)}\n"
                f"Dataset validation warnings: "
                f"{json.dumps(getattr(state, 'dataset_validation_warnings', []), ensure_ascii=True)}\n"
                f"Dataset code: {state.dataset_code}\n"
                f"Literature review: {state.literature_review}"
            ),
            registry=registry,
            terminal_tools={"submit_experiment"},
            tool_names_provider=available_tools,
            automatic_tools={"execute_experiment_code", "submit_experiment"},
        )
        return result.value

    def supplement(self, state, request, workdir, supplement_index):
        supplement_id = f"supplement_{int(supplement_index):02d}"
        supplement_workdir = Path(workdir).resolve() / "supplements" / supplement_id
        supplement_workdir.mkdir(parents=True, exist_ok=True)
        current = {
            "code": "",
            "output": "",
            "success": None,
            "code_hash": None,
            "failed_hashes": set(),
            "last_failure_category": None,
            "consecutive_failure_category_count": 0,
            "force_replace": False,
        }

        def execute_current_supplement():
            if not current["code"]:
                return {"success": False, "error": "Create supplement code first"}
            fingerprint = _code_hash(current["code"])
            if fingerprint in current["failed_hashes"]:
                return {
                    "success": False,
                    "error": (
                        "Supplement code is unchanged since its previous failed "
                        "execution. Edit or replace it before retrying."
                    ),
                    "code_hash": fingerprint,
                }
            combined = state.dataset_code + "\n\n" + current["code"]
            execution = execute_code(
                combined,
                timeout=600,
                workdir=str(supplement_workdir),
                return_details=True,
            )
            output = execution["output"]
            success = execution["success"]
            category, repair_hint = _record_execution_result(
                current,
                fingerprint,
                output,
                success,
                execution["error_type"],
                error=execution["error"],
                traceback_text=execution["traceback"],
            )
            return _solver_execution_feedback({
                "success": success,
                "error_type": execution["error_type"],
                "error": execution["error"],
                "traceback": execution["traceback"],
                "exit_code": execution["exit_code"],
                "output": output,
                "code_hash": fingerprint,
                "failure_category": category,
                "repair_hint": repair_hint,
                "same_category_count": current[
                    "consecutive_failure_category_count"
                ],
                "full_replacement_required": current["force_replace"],
            }, f"supplementary experiment {supplement_id}")

        def replace_supplement_code(code):
            syntax_error = _python_syntax_error(code)
            if syntax_error:
                return {"success": False, "error": syntax_error}
            fingerprint = _code_hash(code)
            if fingerprint in {
                current.get("code_hash"),
            } or fingerprint in current["failed_hashes"]:
                return {
                    "success": False,
                    "error": (
                        "Replacement supplement code is unchanged from the current "
                        "or previously failed draft. Make a substantive correction."
                    ),
                    "code_hash": fingerprint,
                }
            current.update(
                code=code,
                output="",
                success=None,
                code_hash=fingerprint,
                force_replace=False,
            )
            return {
                "success": True,
                "message": "Supplement code replaced; automatic execution is next",
                "code_hash": fingerprint,
            }

        def edit_supplement_code(old_text, new_text):
            if not current["code"]:
                return {"success": False, "error": "Create supplement code first"}
            occurrences = current["code"].count(old_text)
            if occurrences != 1:
                return {
                    "success": False,
                    "error": f"old_text must match exactly once; matched {occurrences}",
                }
            edited = current["code"].replace(old_text, new_text, 1)
            syntax_error = _python_syntax_error(edited)
            if syntax_error:
                return {"success": False, "error": syntax_error}
            fingerprint = _code_hash(edited)
            if fingerprint in {
                current.get("code_hash"),
            } or fingerprint in current["failed_hashes"]:
                return {
                    "success": False,
                    "error": (
                        "The edited supplement code matches the current or a "
                        "previously failed version"
                    ),
                    "code_hash": fingerprint,
                }
            current.update(
                code=edited,
                output="",
                success=None,
                code_hash=fingerprint,
            )
            return {
                "success": True,
                "message": "Supplement code edited; automatic execution is next",
                "code_hash": fingerprint,
            }

        def execute_supplement_code():
            return execute_current_supplement()

        def submit_supplementary_experiment():
            if not current["code"]:
                return {"success": False, "error": "Create supplement code first"}
            if current["success"] is not True:
                return {
                    "success": False,
                    "error": "Execute supplement code successfully before submitting",
                }
            return {
                "success": True,
                "supplement_id": supplement_id,
                "request": request,
                "code": current["code"],
                "output": current["output"],
            }

        def available_tools():
            if not current["code"]:
                return {"replace_supplement_code"}
            if current["success"] is True:
                return {"submit_supplementary_experiment"}
            if current["success"] is False:
                if current["last_failure_category"] == "execution_infrastructure_error":
                    return {"execute_supplement_code", "edit_supplement_code", "replace_supplement_code"}
                if current["force_replace"]:
                    return {"replace_supplement_code"}
                return {"edit_supplement_code", "replace_supplement_code"}
            return {"execute_supplement_code"}

        registry = ToolRegistry()
        registry.register(
            "replace_supplement_code",
            (
                "Create or completely replace separate supplementary experiment "
                "code. The framework automatically executes a valid replacement and "
                "returns its real execution result in the same tool response."
            ),
            _schema(
                {"code": _string("Complete supplement-only Python code")},
                ["code"],
            ),
            replace_supplement_code,
        ).register(
            "edit_supplement_code",
            (
                "Replace one exact fragment in the current supplementary code. The "
                "framework automatically executes a valid edit and returns its real "
                "execution result in the same tool response."
            ),
            _schema(
                {
                    "old_text": _string("Exact unique supplement code fragment"),
                    "new_text": _string("Replacement supplement code fragment"),
                },
                ["old_text", "new_text"],
            ),
            edit_supplement_code,
        ).register(
            "execute_supplement_code",
            (
                "Retry dataset preparation followed by the unchanged supplementary "
                "code after an execution-infrastructure failure."
            ),
            _schema({}, []),
            execute_supplement_code,
        ).register(
            "submit_supplementary_experiment",
            "Submit the most recent successfully executed supplement without modifying the original experiment.",
            _schema({}, []),
            submit_supplementary_experiment,
        )
        runtime = AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name="MLESolverAgent",
            phase="supplementary experiments",
            max_tool_steps=20,
        )
        result = runtime.run(
            system_prompt=supplementary_experiment_solver_prompt(supplement_id),
            user_prompt=(
                f"Topic: {state.topic}\nPlan: {state.plan}\n"
                f"Selected dataset: {state.dataset_selection}\n"
                f"Fixed numerical seeds: "
                f"{json.dumps(getattr(state, 'seeds', {}), sort_keys=True)}\n"
                f"Execution environment: "
                f"{json.dumps(execution_capabilities(), ensure_ascii=True)}\n"
                f"Prepared dataset manifest: "
                f"{json.dumps(getattr(state, 'dataset_manifest', {}), ensure_ascii=True)}\n"
                f"Dataset validation warnings: "
                f"{json.dumps(getattr(state, 'dataset_validation_warnings', []), ensure_ascii=True)}\n"
                f"Dataset code: {state.dataset_code}\n"
                f"Original experiment code (read only): {state.experiment_code}\n"
                f"Previous supplements: "
                f"{json.dumps(getattr(state, 'supplementary_experiments', []), ensure_ascii=True)}\n"
                f"Supplement request: {json.dumps(request, ensure_ascii=True)}"
            ),
            registry=registry,
            terminal_tools={"submit_supplementary_experiment"},
            tool_names_provider=available_tools,
            automatic_tools={"execute_supplement_code", "submit_supplementary_experiment"},
        )
        return result.value


class ExperimentRewardAgent:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key

    def score(self, state):
        def submit_experiment_score(score, rationale):
            score = float(score)
            if not 0.0 <= score <= 1.0:
                return {"success": False, "error": "score must be between 0 and 1"}
            return {"success": True, "score": score, "rationale": rationale}

        registry = ToolRegistry().register(
            "submit_experiment_score",
            "Submit the final experiment-quality score without changing the experiment.",
            _schema(
                {
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Plan adherence and evidence quality score",
                    },
                    "rationale": _string("Evidence-based scoring rationale"),
                },
                ["score", "rationale"],
            ),
            submit_experiment_score,
        )
        runtime = AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name="ExperimentRewardAgent",
            phase="experiment scoring",
            max_tool_steps=1,
        )
        return runtime.run_one(
            system_prompt=experiment_reward_prompt(),
            user_prompt=(
                f"Plan: {state.plan}\n"
                f"Fixed numerical seeds: "
                f"{json.dumps(getattr(state, 'seeds', {}), sort_keys=True)}\n"
                f"Code: {state.experiment_code}\n"
                f"Observed output: {state.experiment_output}\n"
                f"Supplementary experiments: "
                f"{json.dumps(getattr(state, 'supplementary_experiments', []), ensure_ascii=True)}"
            ),
            registry=registry,
        ).value
