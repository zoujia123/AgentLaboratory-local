import json
import re
from copy import deepcopy
from pathlib import Path

from agent_helpers import (
    _authoritative_execution_failure,
    _citation_catalog,
    _coalesce_experiment_edit_batches,
    _code_hash,
    _latest_experiment_failure_messages,
    _latex_escape_text,
    _object_schema,
    _preparation_feedback_for_prompt,
    _process_execution_result,
    _stale_diagnostic_marker,
    _string,
)
from agent_runtime import AgentRuntime, MAX_IDENTICAL_FAILED_TOOL_CALLS, ToolRegistry
from data_preparation import _run_preparation_code
from execution_diagnostics import python_syntax_error
from seeding import seed_prompt_text
from prompts import (
    LITERATURE_RETRY_RELEVANCE_RULE,
    experiment_solver_prompt,
    literature_review_prompt,
    ml_dataset_selection_prompt,
    paper_blueprint_prompt,
    paper_section_prompt,
    phd_interpretation_prompt,
    phd_plan_feedback_prompt,
    postdoc_interpretation_feedback_prompt,
    postdoc_plan_prompt,
    professor_readme_prompt,
    sw_data_preparation_prompt,
)
from tools import ArxivSearch, HFDataSearch, execute_code, execution_capabilities
from run_logging import current_run_logger
from utils import compile_latex


class BaseResearchAgent:
    role = "research agent"

    def __init__(self, model, api_key, name=None):
        self.model = model
        self.api_key = api_key
        self.name = name or self.__class__.__name__

    def runtime(self, phase, *, max_tokens=8192):
        return AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name=self.name,
            phase=phase,
            max_tokens=max_tokens,
        )


class PhDStudentAgent(BaseResearchAgent):
    role = "computer science PhD student"

    def conduct_literature_review(self, topic):
        """Read and optionally add papers in at most two screening rounds."""
        class RoundEnded(Exception):
            """Leave the runtime between tool calls."""

        search = ArxivSearch()
        logger = current_run_logger()
        for review_round in (1, 2):
            logger.info(f"[literature review] SCREENING ROUND START round={review_round}/2")
            user_prompt = f"Research topic: {topic}\nScreening round: {review_round}/2."
            if review_round == 2:
                user_prompt += f"\n{LITERATURE_RETRY_RELEVANCE_RULE}"
            papers = []
            readed_paper_id = set()  # 本轮已读论文 ID 集合。
            current_paper_id = None
            stage = "search"
            consecutive_empty_searches = 0
            max_empty_searches = 3

            def search_arxiv(query):
                nonlocal consecutive_empty_searches, stage, current_paper_id

                details = search.find_papers(query, excluded_ids=readed_paper_id.copy())
                candidate_ids = details["candidate_ids"]

                if candidate_ids:
                    current_paper_id = candidate_ids[0]
                    consecutive_empty_searches = 0
                    stage = "read"
                else:
                    consecutive_empty_searches += 1
                    if consecutive_empty_searches >= max_empty_searches:
                        stage = "ended"
                return {
                    "success": True,
                    "results": details["text"],
                    "candidate_ids": candidate_ids,
                    "consecutive_empty_searches": consecutive_empty_searches,
                }

            def read_arxiv_paper():
                nonlocal stage
                full_text = search.retrieve_full_paper_text(current_paper_id, MAX_LEN=4000)
                readed_paper_id.add(current_paper_id)
                stage = "decide"
                return {"success": True, "arxiv_id": current_paper_id, "full_text": full_text}

            def decide_add(add, summary):
                nonlocal stage, current_paper_id
                if add:
                    full_text = search.get_cached_full_paper_text(current_paper_id)
                    papers.append({"arxiv_id": current_paper_id, "summary": summary.strip(), "full_text": full_text})
                    stage = "submit"
                else:
                    current_paper_id = None
                    stage = "search"
                return {"success": True, "added": add, "paper_count": len(papers)}

            def submit_literature_review(review):
                return {"success": True, "review": review, "papers": papers}



            def available_tools():
                tools_by_stage = {
                    "search": {"search_arxiv"},
                    "read": {"read_arxiv_paper"},
                    "decide": {"decide_add"},
                    "submit": {"submit_literature_review"},
                }
                if stage == "ended":
                    raise RoundEnded
                return tools_by_stage[stage]

            registry = ToolRegistry()
            registry.register(
                "search_arxiv",
                "Search local arXiv papers with a short query",
                {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Short query about the research task or methods."
                        }
                    },
                    "required": ["query"]
                },
                search_arxiv,
            ).register(
                "read_arxiv_paper",
                "Read the full text of the single paper selected by the latest search. ",
                {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                read_arxiv_paper,
            ).register(
                "decide_add",
                "Decide whether to include the paper just read in the literature review.",
                {
                    "type": "object",
                    "properties": {
                        "add": {
                            "type": "boolean",
                            "description": "Whether to include the current paper."
                        },
                        "summary": {
                            "type": "string",
                            "description": "When add=true, summarize relevance, methods, findings, and limitations from the provided text. When add=False, use an empty string."
                        }
                    },
                    "required": ["add","summary"]
                },
                decide_add,
            ).register(
                "submit_literature_review",
                "Submit the complete English literature review to finish this stage.",
                {
                    "type": "object",
                    "properties": {
                        "review": {
                            "type": "string",
                            "description": "Review connecting the selected paper to the research topic."
                        }
                    },
                    "required": ["review"]
                },
                submit_literature_review,
            )
            try:
                result = self.runtime("literature review").run(
                    system_prompt=literature_review_prompt(),
                    user_prompt=user_prompt,
                    registry=registry,
                    terminal_tools={"submit_literature_review"},
                    tool_names_provider=available_tools,
                )
            except RoundEnded:
                logger.info(f"[literature review] SCREENING ROUND END round={review_round}/2")
                continue
            return result.value

    def plan_feedback(self, topic, literature_review, postdoc_message, *, planning_history=None, seeds=None):
        def review_plan(message, approved, required_changes):
            if approved == bool(required_changes) or any(not item.strip() for item in required_changes):
                return {"success": False, "error": "Approve only with no required_changes; otherwise list concrete necessary corrections."}
            return {"success": True, "message": message, "approved": approved, "required_changes": required_changes}

        registry = ToolRegistry().register(
            "send_dialogue",
            "Review whether the proposed plan answers the research question. Request necessary corrections only, or approve it if sufficient.",
            {
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean", "description": "Whether the plan is sufficient to answer the research question."},
                    "required_changes": {"type": "array", "items": {"type": "string"}, "description": "Necessary corrections only; empty when approved."},
                    "message": {
                        "type": "string",
                        "description": "Give concise feedback to PostdocAgent; preserve the requested scope and use literature as background, not a requirement to reproduce its methods."
                    }
                },
                "required": [
                    "message", "approved", "required_changes"
                ]
            },
            review_plan,
        )
        return self.runtime("plan formulation").run_one(
            system_prompt=phd_plan_feedback_prompt(),
            user_prompt=(
                f"Research topic: {topic}\nLiterature review: {literature_review}\n"
                f"Proposed plan: {postdoc_message}\n"
                f"Planning history: {json.dumps(planning_history or [], ensure_ascii=True)}\n"
                f"{seed_prompt_text(seeds)}"
            ),
            registry=registry,
        ).value

    def interpretation_turn(
        self, topic, plan, code, output, dialogue, literature_review="",
        supplementary_experiments=None, interpretation_history=None,
        *, stage="review", workflow_notice="",
    ):
        def send_dialogue(message):
            if not message.strip():
                return {"success": False, "error": "Provide a non-empty interpretation or question."}
            return {"success": True, "message": message}

        def submit_interpretation(interpretation):
            if not interpretation.strip():
                return {"success": False, "error": "Provide a non-empty interpretation."}
            return {"success": True, "interpretation": interpretation}

        def request_supplementary_experiment(objective, reason, required_outputs, constraints):
            if not objective.strip() or not required_outputs:
                return {"success": False, "error": "Provide a non-empty objective and required_outputs."}
            return {
                "success": True,
                "request": {
                    "objective": objective,
                    "reason": reason,
                    "required_outputs": required_outputs,
                    "constraints": constraints,
                },
            }

        def available_tools():
            return {
                "review": {"send_dialogue"},
                "supplement": {"request_supplementary_experiment"},
                "submit": {"submit_interpretation"},
                "decide": {"request_supplementary_experiment", "submit_interpretation"},
            }[stage]

        registry = ToolRegistry()
        registry.register(
            "send_dialogue",
            "Send the proposed result interpretation for review.",
            {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Proposed interpretation or focused question grounded in the observed results.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
            send_dialogue,
        ).register(
            "request_supplementary_experiment",
            "Request a bounded supplementary experiment.",
            {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Single evidence gap to resolve, including any observed values or anomalies needed to execute the experiment.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why the additional evidence is necessary for the interpretation.",
                    },
                    "required_outputs": {
                        "type": "array",
                        "items": {"type": "string", "description": "One required metric, comparison, or diagnostic."},
                        "description": "Exact results the supplementary experiment must produce.",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string", "description": "One limit on scope, resources, or allowed changes."},
                        "description": "Scope and resource limits that preserve the original experiment unchanged.",
                    },
                },
                "required": ["objective", "reason", "required_outputs", "constraints"],
                "additionalProperties": False,
            },
            request_supplementary_experiment,
        ).register(
            "submit_interpretation",
            "Submit the final result interpretation.",
            {
                "type": "object",
                "properties": {
                    "interpretation": {
                        "type": "string",
                        "description": "Complete English interpretation connecting the evidence to the literature and planned hypothesis: observed metrics, comparisons, anomalies, plausible explanations, claim support, and remaining limitations. Use only observed values without invented results or significance.",
                    },
                },
                "required": ["interpretation"],
                "additionalProperties": False,
            },
            submit_interpretation,
        )
        return self.runtime("results interpretation").run(
            system_prompt=phd_interpretation_prompt(),
            user_prompt=(
                f"Topic: {topic}\nPlan: {plan}\n"
                f"Literature review: {literature_review}\n"
                f"Original experiment code: {code}\n"
                f"Original experiment output: {output}\n"
                f"Supplementary experiments: "
                f"{json.dumps(supplementary_experiments or [], ensure_ascii=True)}\n"
                f"Interpretation history: "
                f"{json.dumps(interpretation_history or [], ensure_ascii=True)}\n"
                f"Current stage: {stage}\n"
                f"Latest Postdoc feedback: {json.dumps(dialogue, ensure_ascii=True)}\n"
                f"Workflow notice: {workflow_notice or 'None'}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            tool_names_provider=available_tools,
        )


class PostdocAgent(BaseResearchAgent):

    role = "computer science postdoctoral researcher"

    def plan_turn(self, topic, literature_review, dialogue, *, approved=False, previous_plan="", planning_history=None, round_number=1, remaining_reviews=0, seeds=None):

        def submit_plan(experiment_plan, data_plan):
            if not experiment_plan.strip() or not data_plan.strip():
                return {
                    "success": False,
                    "error": "Provide a non-empty experiment_plan and data_plan.",
                }
            return {
                "success": True,
                "plan": experiment_plan,
                "data_requirements": data_plan,
            }

        def available_tools():
            return {"submit_plan"} if approved else {"send_dialogue"}

        registry = ToolRegistry().register(
            "send_dialogue",
            (
                "Draft an experiment plan and send it to PhDStudentAgent for review."
            ),
            {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Describe a minimal, feasible plan preserving the requested model, comparisons, and evaluation conditions; use literature as background rather than expanding the task."
                    }
                },
                "required": [
                    "message"
                ]
            },
            lambda message: {"success": True, "message": message},
        ).register(
            "submit_plan",
            (
                "Submit the final experiment plan and its data-preparation plan."
            ),
            {
                "type": "object",
                "properties": {
                    "experiment_plan": {
                        "type": "string",
                        "description": "Describe the minimal experiment in English: question, methods, controls, evaluation, steps, and reproducibility; preserve the requested model and comparisons."
                    },
                    "data_plan": {
                        "type": "string",
                        "description": "Describe the required local data and preparation steps, excluding modeling and evaluation. Omit dataset IDs and schema-dependent field names."
                    }
                },
                "required": [
                    "experiment_plan",
                    "data_plan"
                ]
            },
            submit_plan,
        )
        return self.runtime("plan formulation").run(
            system_prompt=postdoc_plan_prompt(),
            user_prompt=(
                f"Research topic: {topic}\nLiterature review: {literature_review}\n"
                f"Latest PhD feedback: {dialogue if dialogue and str(dialogue).strip() else 'None'}\n"
                f"Previous complete plan: {previous_plan or 'None'}\n"
                f"Planning history: {json.dumps(planning_history or [], ensure_ascii=True)}\n"
                f"Review approved: {approved}\n"
                f"Round: {round_number}; remaining reviews: {remaining_reviews}\n"
                f"{seed_prompt_text(seeds)}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            tool_names_provider=available_tools,
        )

    def interpretation_feedback(
        self, topic, plan, experiment_output, phd_message, experiment_code="",
        literature_review="", supplementary_experiments=None, interpretation_history=None,
    ):
        def send_dialogue(message, supplement_required):
            if not message.strip():
                return {"success": False, "error": "Provide non-empty feedback."}
            if supplement_required is not None and not isinstance(supplement_required, bool):
                return {"success": False, "error": "supplement_required must be a boolean or null."}
            return {"success": True, "message": message, "supplement_required": supplement_required}

        registry = ToolRegistry().register(
            "send_dialogue",
            "Send feedback on the proposed result interpretation.",
            {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Evidence-based feedback on metrics, comparisons, anomalies, plausible explanations, limitations, and claim support in relation to the literature and planned hypothesis. Use only observed values; if additional evidence is necessary, specify its goal and expected outputs.",
                    },
                    "supplement_required": {
                        "type": ["boolean", "null"],
                        "description": "True when a bounded, executable experiment is necessary; false when existing evidence suffices with remaining limitations; null when either supplementing or submitting is acceptable.",
                    },
                },
                "required": ["message", "supplement_required"],
                "additionalProperties": False,
            },
            send_dialogue,
        )
        return self.runtime("results interpretation").run_one(
            system_prompt=postdoc_interpretation_feedback_prompt(),
            user_prompt=(
                f"Topic: {topic}\nPlan: {plan}\n"
                f"Literature review: {literature_review}\n"
                f"Original experiment code: {experiment_code}\n"
                f"Original results: {experiment_output}\n"
                f"Supplementary experiments: "
                f"{json.dumps(supplementary_experiments or [], ensure_ascii=True)}\n"
                f"Interpretation history: "
                f"{json.dumps(interpretation_history or [], ensure_ascii=True)}\n"
                f"PhD message: {phd_message}"
            ),
            registry=registry,
        ).value


class MLEngineerAgent(BaseResearchAgent):
    role = "machine learning engineer"

    @staticmethod
    def _select_first_complete_dataset(results):
        """Choose the first locally materialized candidate without using query-specific rules."""
        for item in results or []:
            dataset_id = item.get("dataset_id") or item.get("id")
            revision = item.get("revision")
            if not dataset_id or not revision:
                continue

            config_name = None
            split = None
            materialized = item.get("materialized") or {}
            if isinstance(materialized, dict):
                for candidate_config, split_map in materialized.items():
                    config_name = str(candidate_config)
                    split = str(next(iter(split_map)))
                    break
            if config_name is None:
                configs = item.get("configs") or []
                splits = item.get("splits") or []
                if configs and splits:
                    config_name = str(configs[0])
                    split = str(splits[0])
            if not config_name or not split:
                continue
            return {
                "dataset_id": str(dataset_id),
                "revision": str(revision),
                "config_name": config_name,
                "split": split,
                "selection_reason": "first complete local candidate returned by search",
            }
        return None

    def data_preparation_turn(self, topic, data_requirements, sw_message, conversation_history="", workdir=None, feedback=None, inspected_dataset=None, selected_dataset=None,latest_execution_failure=None, failed_trial_code_failures=None,*, code_execution_timeout_seconds,):
        search = HFDataSearch()
        def available_tools():
            if inspected_dataset:
                return {"write_and_run_dataset_code"}
            if selected_dataset:
                return {"inspect_selected_dataset"}
            return {"search_huggingface_datasets"}

        latest_execution_failure = _authoritative_execution_failure(latest_execution_failure)   # 是把“上一轮失败信息”过滤成可靠形式。
        if failed_trial_code_failures is None:  # 用来保存已经执行失败过的数据准备代码哈希, 避免重复执行相同的失败代码。
            failed_trial_code_failures = {}

        def search_huggingface_datasets(query, max_results):
            results = search.retrieve_ds(query, N=max(1, min(int(max_results), 10)))
            selected = self._select_first_complete_dataset(results)
            return {
                "success": True,
                "results": results,
                "selected_dataset": selected,
            }

        def inspect_selected_dataset():
            inspected = search.inspect_dataset(
                selected_dataset["dataset_id"],
                selected_dataset["revision"],
                selected_dataset["config_name"],
                selected_dataset["split"],
            )
            return {
                "success": True,
                **inspected,
            }

        def write_and_run_dataset_code(code):
            # A trailing newline or spaces do not make a failed preparation program a new repair.
            fingerprint = _code_hash(str(code).rstrip(" \t\r\n"))
            # 检查是否已经执行过相同的代码
            if fingerprint in failed_trial_code_failures:
                prior_failure = dict(failed_trial_code_failures.get(fingerprint) or {}) # 取出上次失败摘要，并复制一份。
                original_error = prior_failure.get("error") or "the prior failure"
                return {    # 直接返回失败结果
                    "success": False,
                    "error_type": "unchanged_failed_dataset_code",
                    "error": ("This exact data-preparation code already failed. It was not executed again."),
                    "repair_hint": (f"Make a substantive code change that resolves the original failure before retrying: {original_error}"),  # 提示模型必须真正修复此前错误，而不是原样重试。
                    "original_error_type": prior_failure.get("error_type"),
                    "original_error": prior_failure.get("error"),
                    "original_repair_hint": prior_failure.get("repair_hint"),
                    "code": code,
                    "code_hash": fingerprint,
                    "selected_dataset": selected_dataset,
                }
            result = {
                **_run_preparation_code(
                    code, workdir, inspected_dataset, code_execution_timeout_seconds,
                ),
                "code_hash": fingerprint,
                "selected_dataset": selected_dataset,
            }   # 它会真正执行代码
            if result.get("success") is False:  # 只要真实执行失败，就把该代码哈希加入失败表：
                failed_trial_code_failures[fingerprint] = {
                    key: result.get(key)
                    for key in ("error_type", "error", "repair_hint")
                    if result.get(key)
                }   # 只是把上次失败的诊断摘要存起来：下一次模型又提交同一份代码时，框架会取回它的失败原因
            return result

        registry = ToolRegistry().register(
            "search_huggingface_datasets",
            (
                "Search local datasets and return the first complete candidate."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short search terms describing the required data and task."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results, from 1 to 10."
                    }
                },
                "required": [
                    "query",
                    "max_results"
                ]
            },
            search_huggingface_datasets,
        ).register(
            "inspect_selected_dataset",
            "Inspect the selected dataset's structure and runtime sample types.",
            {
                "type": "object",
                "properties": {},
                "required": []
            },
            inspect_selected_dataset,
        ).register(
            "write_and_run_dataset_code",
            "Write and execute a complete data-preparation program.",
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python code that loads and prepares the selected local dataset "
                            "using datasets.load_dataset. Include imports and use the inspected "
                            "column names and data types. Keep the required splits and labels aligned, "
                            "and fit preprocessing on training data only. "
                            "Do not train or evaluate models. Print only a short summary."
                        )
                    }
                },
                "required": [
                    "code"
                ]
            },
            write_and_run_dataset_code,
        )
        return self.runtime("data preparation collaboration").run(
            system_prompt=ml_dataset_selection_prompt(),
            user_prompt=(
                f"Topic: {topic}\n"
                f"Data preparation requirements: {data_requirements}\n"
                f"Latest SWEngineer message: {sw_message}\n"
                f"Previous ML command result: "
                f"{json.dumps(_preparation_feedback_for_prompt(feedback, inspected_dataset, latest_execution_failure), ensure_ascii=True, default=str)}\n"
                f"Authoritative latest execution failure: "
                f"{json.dumps(latest_execution_failure, ensure_ascii=True, default=str)}\n"
                f"Authoritative inspected dataset: "
                f"{json.dumps(inspected_dataset, ensure_ascii=True, default=str)}\n"
                + (f" Selected local dataset: {json.dumps(selected_dataset, ensure_ascii=True, default=str)}\n"
                   if selected_dataset and not inspected_dataset else "")
                + f"Conversation so far:\n{conversation_history or 'None'}\n"
                f"Execution environment: {json.dumps(execution_capabilities(), ensure_ascii=True)}\n"
                f"Reproducibility: {seed_prompt_text()}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            return_after_tool=True,
            tool_names_provider=available_tools,
        )


class SWEngineerAgent(BaseResearchAgent):
    role = "software engineer"

    def data_preparation_turn(self, topic, data_requirements,conversation_history="", feedback=None, ml_result=None,inspected_dataset=None,latest_execution_failure=None,):
        registry = ToolRegistry().register(
            "send_dialogue",
            "Send data-preparation feedback or repair advice.",
            {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Concise requirements feedback or a concrete correction supported by execution results."
                    }
                },
                "required": [
                    "message"
                ],
                "additionalProperties": False
            },
            lambda message: {"success": True, "message": message},
        )
        return self.runtime("data preparation collaboration").run(
            system_prompt=sw_data_preparation_prompt(),
            user_prompt=(
                f"Topic: {topic}\n"
                f"Data preparation requirements: {data_requirements}\n"
                f"Previous SW command result: "
                f"{json.dumps(_stale_diagnostic_marker() if _authoritative_execution_failure(latest_execution_failure) else feedback, ensure_ascii=True, default=str)}\n"
                f"Latest ML command result: "
                f"{json.dumps(_preparation_feedback_for_prompt(ml_result, inspected_dataset, latest_execution_failure), ensure_ascii=True, default=str)}\n"
                f"Authoritative latest execution failure: "
                f"{json.dumps(_authoritative_execution_failure(latest_execution_failure), ensure_ascii=True, default=str)}\n"
                f"Authoritative inspected dataset: "
                f"{json.dumps(inspected_dataset, ensure_ascii=True, default=str)}\n"
                f"Conversation so far:\n{conversation_history or 'None'}\n"
                f"Execution environment: {json.dumps(execution_capabilities(), ensure_ascii=True)}\n"
                f"Reproducibility: {seed_prompt_text()}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            return_after_tool=True, # 无论成功还是失败，只要执行过一个工具，就把控制权交回外层协作流程
        )


class MLESolverAgent(BaseResearchAgent):
    role = "machine learning experiment solver"

    def __init__(self, model, api_key, code_execution_timeout_seconds):
        super().__init__(model, api_key)
        self.code_execution_timeout_seconds = code_execution_timeout_seconds

    def solve(self, state, workdir, *, supplement_request=None):
        """Execute the plan or a bounded supplement in the caller-provided workdir."""
        phase = "running experiments" if supplement_request is None else "supplementary experiments"
        current = {
            "code": "",     # 当前版本的实验代码，不包含前置的数据准备代码.write_experiment_code() 整体替换；edit_experiment_code() 局部修改
            "output": "",   # 当前程序最近一次执行的输出.替换、编辑代码时清空；_process_execution_result() 在执行后写入
            "success": None,    # 当前代码的执行状态：None 未执行、False 执行失败、True 执行成功
            "code_hash": None,  # 当前代码的指纹，用于判断代码文本是否相同.code_hash：现在是哪一份代码？
            "failed_hashes": set(), # 已经失败过的代码指纹集合。failed_hashes：这份代码以前失败过吗？
            "failed_failures": {},  # 每个失败代码指纹对应的错误详情。failed_failures：它以前为什么失败？
            "last_failure_category": None,  # 最近一次执行失败的类别
            "consecutive_failure_category_count": 0,    # 连续出现同一错误类别的次数
            "force_replace": False,  # 是否强制替换当前代码
            "latest_execution_failure": None,  # 最近一次执行失败的详情
        }

        def rejected_experiment_code(fingerprint, action):
            latest_failure = current["latest_execution_failure"]
            if fingerprint in current["failed_hashes"]:
                original_failure = current["failed_failures"][fingerprint]
                result = {
                    "success": False,
                    "error_type": "unchanged_failed_experiment_code",
                    "error": (
                        "This exact experiment code previously failed and was not executed again."
                    ),
                    "repair_hint": "Change the code to resolve its recorded execution failure.",
                    "original_error_type": original_failure.get("error_type"),
                    "original_error": original_failure.get("error"),
                    "original_failure_category": original_failure.get(
                        "failure_category"
                    ),
                    "original_repair_hint": original_failure.get("repair_hint"),
                    "code_hash": fingerprint,
                    "action": action,
                }
                result["original_execution_diagnostic"] = original_failure
                if latest_failure:
                    result["latest_execution_diagnostic"] = dict(latest_failure)
                return result
            return {
                "success": False,
                "error_type": "unchanged_current_experiment_code",
                "error": "The proposed experiment code is unchanged from the current draft.",
                "repair_hint": "Make a substantive code change before retrying.",
                "code_hash": fingerprint,
                "action": action,
            }

        def execute_experiment_code():
            fingerprint = current["code_hash"]
            if fingerprint in current["failed_hashes"]:
                return rejected_experiment_code(fingerprint, "execute_experiment_code")

            combined = state.dataset_code + "\n\n" + current["code"]
            execution = execute_code(combined,timeout=self.code_execution_timeout_seconds,workdir=workdir,return_details=True,expected_hf_dataset=state.dataset_selection or None,)
            return _process_execution_result(current, fingerprint, execution, phase)

        def write_experiment_code(code):
            syntax_error = python_syntax_error(code)
            if syntax_error:
                return {"success": False,"error_type": "syntax_error","error": syntax_error,}

            fingerprint = _code_hash(code)
            if fingerprint == current["code_hash"] or fingerprint in current["failed_hashes"]:
                return rejected_experiment_code(fingerprint,"write_experiment_code")

            current.update(code=code,output="",success=None,code_hash=fingerprint,force_replace=False,)
            return {"success": True,"message": "Experiment code saved.","code_hash": fingerprint,}

        def edit_experiment_code(edits):
            # 正常调用只有一个片段；兼容 runtime 合并多个编辑调用后的批量输入。
            if not isinstance(edits, list) or not edits:
                return {"success": False,"error": "Submit exactly one replacement object in edits.",}
            if len(edits) > 6:
                return {"success": False,"error": "Too many replacement items; at most 6 are supported.",}

            # 所有片段基于同一版本定位，通过全部校验后原子更新。
            source = current["code"]
            seen_old_text = set()
            planned_edits = []
            for index, edit in enumerate(edits, start=1):
                if not isinstance(edit, dict) or set(edit) != {"old_text", "new_text"}:
                    return {"success": False,"error": (f"edit {index} must contain only old_text and new_text"),}
                old_text = edit["old_text"]
                new_text = edit["new_text"]

                if not isinstance(old_text, str) or not isinstance(new_text, str):
                    return {"success": False,"error": f"edit {index} old_text and new_text must be strings",}
                if not old_text:
                    return {"success": False,"error": f"edit {index} old_text must not be empty",}
                if old_text in seen_old_text:
                    return {"success": False,"error": "Repeated old_text fragments.",}
                seen_old_text.add(old_text)

                occurrences = source.count(old_text)
                if occurrences != 1:
                    return {"success": False,
                        "error": ( f"edit {index} old_text must match exactly once; matched {occurrences}"),}

                start = source.index(old_text)
                planned_edits.append((start, start + len(old_text), new_text))

            planned_edits.sort(key=lambda item: item[0])
            for previous, following in zip(planned_edits, planned_edits[1:]):
                if following[0] < previous[1]:
                    return {"success": False,"error": "Replacement ranges overlap.",}

            edited = source
            for start, end, new_text in reversed(planned_edits):
                edited = edited[:start] + new_text + edited[end:]
            syntax_error = python_syntax_error(edited)
            if syntax_error:
                return {
                    "success": False,
                    "error_type": "syntax_error",
                    "error": syntax_error,
                }

            fingerprint = _code_hash(edited)
            if fingerprint == current["code_hash"] or fingerprint in current["failed_hashes"]:
                return rejected_experiment_code(fingerprint,"edit_experiment_code",)

            current.update(code=edited,output="",success=None,code_hash=fingerprint,)
            return {
                "success": True,
                "message": "Experiment code updated.",
                "code_hash": fingerprint,
                "edit_count": len(planned_edits),
            }

        def submit_experiment():
            return {"success": True,"code": current["code"],"output": current["output"],}

        def available_tools():
            if not current["code"] or current["force_replace"]:
                return {"write_experiment_code"}
            if current["success"] is True:
                return {"submit_experiment"}
            if current["success"] is False:
                if current["last_failure_category"] == "execution_infrastructure_error":    # 当前 _process_execution_result() 遇到子进程异常退出、结果传输失败等被归为基础设施故障的情况，不会把代码加入失败集合，以便允许重试。
                    return {"execute_experiment_code", "edit_experiment_code", "write_experiment_code"}
                return {"edit_experiment_code", "write_experiment_code"}
            return {"execute_experiment_code"}

        registry = ToolRegistry()
        registry.register(
            "write_experiment_code",
            "Write a complete experiment program.",
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete experiment-only Python source with imports, reusing prepared data variables without repeating data-preparation code or Markdown fences.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            write_experiment_code,
        ).register(
            "edit_experiment_code",
            "Replace an exact, unique fragment in the saved experiment code.",
            {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": "Exactly one replacement for a contiguous fragment of the current experiment code.",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {
                                    "type": "string",
                                    "description": "Non-empty code fragment occurring exactly once, including its indentation.",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "Replacement Python source with correct indentation; empty to delete the fragment.",
                                },
                            },
                            "required": ["old_text", "new_text"],
                        },
                    },
                },
                "required": ["edits"],
                "additionalProperties": False,
            },
            edit_experiment_code,
            # 仅作为模型同一响应返回多个编辑调用时的兜底，不扩大对外 schema。
            batch_coalescer=_coalesce_experiment_edit_batches,
        ).register(
            "execute_experiment_code",
            "Run the saved experiment with the data-preparation code.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            execute_experiment_code,
        ).register(
            "submit_experiment",
            "Submit the successfully executed experiment code and output.",
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            submit_experiment,
        )
        runtime = self.runtime(phase)
        system_prompt = experiment_solver_prompt()
        user_prompt = (
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
            f"Execution timeout: {self.code_execution_timeout_seconds} seconds.\n"
            "Execution constraints: Each run starts fresh with the dataset code prepended. "
            "Work offline using available modules; no network, subprocesses, package "
            "installation, LLM calls, or pretrained assets are available. Use a simple "
            "model trained from scratch when the plan leaves model choice open.\n"
            "Reproducibility: Use SKLEARN_SEED for scikit-learn random_state and "
            "Dataset.train_test_split(seed=...), and NUMPY_SEED for np.random.default_rng. "
            "The framework initializes Python, NumPy, and PyTorch random generators.\n"
            "Results: Print actual measured results and explain what each metric or "
            "comparison demonstrates; never fabricate output.\n"
        )
        if supplement_request is None:
            user_prompt += (
                f"Literature review: {state.literature_review}\n"
                "Experiment requirements: Cover every planned baseline, metric, control, "
                "and comparison, using literature as methodological background. Preserve "
                "the planned model, sample cap, splits, preprocessing, and training budget. "
                "For representation comparisons, vary only the requested representation; "
                "reuse aligned sample IDs, labels, train/test splits, model, budget, and metrics. "
                "For capped stratified sampling, select the capped sample before splitting. "
                "Keep train/test samples disjoint and fit learned transformations on training "
                "data only. Check nonempty splits, valid labels, aligned fields, and finite, "
                "plausible metrics. Use the observed data types and supported APIs.\n"
                "When useful, save result figures as Figure_1.png and Figure_2.png."
            )
        else:
            user_prompt += (
                f"Original experiment code (read only): {state.experiment_code}\n"
                f"Previous supplements: {json.dumps(state.supplementary_experiments, ensure_ascii=True)}\n"
                f"Supplement request: {json.dumps(supplement_request, ensure_ascii=True)}\n"
                "Supplement requirements: Resolve only the requested objective and produce "
                "the required outputs within its constraints. Use the request for necessary "
                "original result values; the full original output is not included. "
                "Reuse prepared data and recreate any needed variables or models from the "
                "reference experiment; its code is not executed automatically. Reload data "
                "or retrain only as needed, without repeating unrelated experiments. "
                "Preserve the original experiment files and figures. Keep the planned model, "
                "sample cap, preprocessing, splits, and training budget unless the request "
                "explicitly varies them. For capped stratified sampling, cap before splitting. "
                "Keep comparisons aligned, train/test samples disjoint, "
                "and learned transformations fitted on training data only. Check data types, "
                "nonempty splits, valid labels, aligned fields, and finite metrics.\n"
                f"Supplement ID: {Path(workdir).name}\n"
                f"Save any supplementary figure as {Path(workdir).name}_figure.png "
                "in the current working directory."
            )

        # 当实验执行失败时，调用 compact_experiment_failure 来生成压缩后的失败信息。减少发给模型的错误信息长度。
        # 一次工具调用失败后，如果手里有“当前实验代码”和“最近一次执行错误”，就重新组织下一轮发给模型的消息。
        def compact_experiment_failure(*, tool_call, arguments, value):
            # diagnostic：最近一次程序运行为什么失败。
            # value：刚才这次工具调用返回了什么。
            del arguments
            if tool_call.name not in {
                "execute_experiment_code", "write_experiment_code", "edit_experiment_code",
            }:
                return None
            repeated_count = value.get("identical_failed_call_count", 0)
            if repeated_count >= MAX_IDENTICAL_FAILED_TOOL_CALLS:
                current["force_replace"] = True
            diagnostic = current["latest_execution_failure"]    # 获取最近一次真实执行失败的诊断。
            if current["force_replace"]:
                value = {
                    **value,
                    "full_replacement_required": True,
                    "repair_hint": (
                        "Provide a new complete code version addressing the reported failure."
                    ),
                }
                # Keep cached execution diagnostics immutable; this is recovery state.
                diagnostic = {
                    **(diagnostic or {}), "full_replacement_required": True,
                }
            if not diagnostic:
                return None
            diagnostic = {**diagnostic, "identical_failed_call_count": repeated_count}
            return _latest_experiment_failure_messages(
                system_prompt,
                user_prompt,
                current["code"],
                diagnostic,
                action_rejection=(
                    None if tool_call.name == "execute_experiment_code" else value
                ),
            )

        result = runtime.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            registry=registry,
            terminal_tools={"submit_experiment"},
            tool_names_provider=available_tools,
            failure_history_compactor=compact_experiment_failure,
            stop_on_repeated_failure=False,
        )
        return result.value


class PaperSolverAgent(BaseResearchAgent):
    role = "research paper writer"

    def __init__(self, model, api_key, name=None, *, paper_writing):
        super().__init__(model, api_key, name=name)
        self.paper_config = deepcopy(paper_writing)

    def write(self, state, workdir, compile_pdf=False):
        settings = self.paper_config["sections"]
        sections = tuple(item["name"] for item in settings)
        generation_order = self.paper_config["generation_order"]
        word_targets = {item["name"]: item["word_target"] for item in settings}
        max_chars = {item["name"]: item["max_chars"] for item in settings}
        workdir = Path(workdir).resolve()
        available_figures = [name for name in ("Figure_1.png", "Figure_2.png") if Path(workdir, name).is_file() ]   # 检查有哪些图片
        citations = _citation_catalog(state)    # 根据研究状态中的文献记录，生成引用目录。
        supplements = getattr(state, "supplementary_experiments", []) or []
        authoritative_evidence = {
            "topic": state.topic,
            "literature_review": state.literature_review,
            "plan": state.plan,
            "dataset_selection": state.dataset_selection,
            "fixed_numerical_seeds": getattr(state, "seeds", {}),
            "experiment_output": state.experiment_output,
            "supplementary_experiments": [
                {key: value for key, value in item.items() if key != "code"}
                for item in supplements
            ],
            "interpretation": state.interpretation,
            "citation_catalog": citations,
            "available_figures": available_figures,
        }   # 把研究资料整理成一个字典：

        def submit_paper_blueprint(title, central_claim, section_plans):
            """
            title：论文标题。
            central_claim：论文最核心的结论。
            section_plans：每个章节准备写什么。
            """
            if set(section_plans) != set(sections):
                return {
                    "success": False,
                    "error": "The blueprint must plan every fixed paper section exactly once",
                }
            title = title.strip()
            central_claim = central_claim.strip()
            section_plans = {
                section: section_plans[section].strip()
                for section in sections
            }
            if not title or not central_claim or not all(section_plans.values()):
                return {
                    "success": False,
                    "error": "Provide a non-empty title, central_claim, and plan for every section.",
                }
            return {
                "success": True,
                "title": title,
                "central_claim": central_claim,
                "section_plans": section_plans,
            }

        blueprint_registry = ToolRegistry().register(
            "submit_paper_blueprint",
            "Submit the paper title, central claim, and section plans.",
            {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "A concise paper title.",
                        "maxLength": 180,
                    },
                    "central_claim": {
                        "type": "string",
                        "description": "One main claim supported by the supplied evidence.",
                        "maxLength": 600,
                    },
                    "section_plans": {
                        "type": "object",
                        "description": "A brief writing plan for each required section.",
                        "properties": {
                            section: {
                                "type": "string",
                                "description": f"A brief content plan for {section}.",
                                "maxLength": 500,
                            }
                            for section in sections
                        },
                        "required": list(sections),
                        "additionalProperties": False,
                    },
                },
                "required": ["title", "central_claim", "section_plans"],
                "additionalProperties": False,
            },
            submit_paper_blueprint,
        )
        blueprint = self.runtime(
            "report writing blueprint",
            max_tokens=3000,
        ).run_one(
            system_prompt=paper_blueprint_prompt(),
            user_prompt=(
                f"Required sections: {', '.join(sections)}\n"
                "Provide a restrained title, one evidence-grounded central claim, and "
                "a concise writing plan for every required section. Base the blueprint "
                "only on the supplied evidence; do not invent experiments, numbers, "
                "citations, significance tests, or claims. Available figures are listed "
                "in the evidence and inserted by the framework in Results.\n"
                "Authoritative evidence:\n"
                + json.dumps(
                    {
                        **authoritative_evidence,
                        "experiment_code": state.experiment_code,
                    },
                    ensure_ascii=True,
                    default=str,
                )
            ),
            registry=blueprint_registry,
        ).value  # 让模型真正生成提纲

        tex_dir = workdir / "tex"
        sections_dir = tex_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        (tex_dir / "paper_blueprint.json").write_text(
            json.dumps(blueprint, ensure_ascii=True, indent=2, default=str),
            encoding="utf-8",
        )

        # 接着准备逐节生成：
        section_bodies = {}
        allowed_citation_keys = {item["key"] for item in citations}
        # 定义章节正文不能包含的内容
        forbidden_section_fragments = (
            "\\documentclass",
            "\\begin{document}",
            "\\end{document}",
            "\\title{",
            "\\maketitle",
            "\\begin{abstract}",
            "\\begin{thebibliography}",
            "\\begin{figure}",
            "\\includegraphics",
        )

        # 开始依次生成章节：
        for section in generation_order:
            # 每一轮都会定义一个章节提交函数：
            def submit_paper_section(section_name, latex_body, expected=section):
                """
                section_name：模型声称正在提交的章节。
                latex_body：该章节的 LaTeX 正文。
                expected=section：锁定当前这一轮要求的章节。
                """
                if section_name != expected:
                    return {
                        "success": False,
                        "error": f"Submit only the requested section {expected!r}",
                    }
                body = latex_body.strip()
                if not body:
                    return {"success": False, "error": "Section body cannot be empty"}  # 正文为空就拒绝。
                forbidden = [fragment for fragment in forbidden_section_fragments if fragment in body]   # 检查禁用内容：
                if re.search(r"\\section\*?\s*\{", body):   # 正文不应该自己带一级章节标题，因为程序稍后会统一添加。
                    forbidden.append("main section command")
                if forbidden:
                    return {"success": False,"error": ("The framework owns document structure and figures; remove: "+ ", ".join(forbidden)),}

                used_citations = set()  # 建立集合，收集正文实际使用的引用标识。
                for group in re.findall(r"\\cite(?:t|p)?\{([^}]*)\}", body):
                    used_citations.update(
                        key.strip()
                        for key in group.split(",")
                        if key.strip()
                    )
                unknown_citations = sorted(used_citations - allowed_citation_keys)
                if unknown_citations:
                    return {
                        "success": False,
                        "error": "Unknown citation keys: " + ", ".join(unknown_citations),
                    }
                return {
                    "success": True,
                    "section_name": expected,
                    "latex_body": body,
                }

            section_registry = ToolRegistry().register(
                "submit_paper_section",
                "Submit the LaTeX body of the requested paper section.",
                {
                    "type": "object",
                    "properties": {
                        "section_name": {
                            "type": "string",
                            "enum": [section],
                            "description": "The requested section name.",
                        },
                        "latex_body": {
                            "type": "string",
                            "description": "The section body in LaTeX.",
                            "maxLength": max_chars[section],
                        },
                    },
                    "required": ["section_name", "latex_body"],
                    "additionalProperties": False,
                },
                submit_paper_section,
            )
            section_evidence = dict(authoritative_evidence)
            if section in {"Methods", "Experimental Setup"}:
                section_evidence["experiment_code"] = state.experiment_code
                section_evidence["supplementary_experiments"] = supplements
            section_result = self.runtime(
                f"report writing section {section}",
                max_tokens=4096,
            ).run_one(
                system_prompt=paper_section_prompt(),
                user_prompt=(
                    f"Current section: {section}\n"
                    f"Target length: approximately {word_targets[section]} words.\n"
                    "Write only the LaTeX body. The framework adds document structure, "
                    "title, main section headings, bibliography, and available figures "
                    "in Results. Refer to available figures in prose; omit figure environments "
                    "and image commands. Subsections, equations, and tables are allowed. "
                    "Record fixed seeds in Experimental Setup. Put limitations and "
                    "potential negative impacts in Discussion. Abstract must be one "
                    "focused paragraph. Preserve reported numbers exactly, use only "
                    "supplied citation keys, and do not invent evidence or significance.\n"
                    "Paper blueprint:\n"
                    + json.dumps(blueprint, ensure_ascii=True, default=str)
                    + "\nAuthoritative evidence:\n"
                    + json.dumps(
                        section_evidence,
                        ensure_ascii=True,
                        default=str,
                    )
                ),
                registry=section_registry,
            ).value
            section_bodies[section] = section_result["latex_body"]
            section_index = sections.index(section) + 1   # 查询该章节在论文最终排列顺序中的位置。
            section_filename = (f"{section_index:02d}_"+ section.lower().replace(" ", "_")+ ".tex")   # 然后构造文件名：
            (sections_dir / section_filename).write_text(section_bodies[section] + "\n",encoding="utf-8",)# 保存文件

        # 所有章节完成后，开始组装整篇 LaTeX：
        document = [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb,booktabs,graphicx}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\title{" + _latex_escape_text(blueprint["title"]) + "}",
            r"\author{Agent Laboratory}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
        ]
        for section in sections:
            document.extend(["", rf"\section{{{section}}}", section_bodies[section]])
            if section == "Results":
                for figure_index, figure in enumerate(available_figures, start=1):
                    document.extend([
                        "",
                        r"\begin{figure}[htbp]",
                        r"\centering",
                        rf"\includegraphics[width=0.82\linewidth]{{{figure}}}",
                        (
                            rf"\caption{{Observed experiment result visualization "
                            rf"{figure_index}.}}"
                        ),
                        r"\end{figure}",
                    ])
        if citations:
            document.extend(["", r"\begin{thebibliography}{99}"])
            for citation in citations:
                document.append(
                    rf"\bibitem{{{citation['key']}}} Reviewed work, "
                    rf"\texttt{{arXiv:{_latex_escape_text(citation['arxiv_id'])}}}."
                )
            document.append(r"\end{thebibliography}")
        document.extend(["", r"\end{document}", ""])
        latex = "\n".join(document)
        compile_result = compile_latex(
            latex,
            str(workdir),
            compile=bool(compile_pdf),
        )
        if "[CODE EXECUTION ERROR]" in compile_result:
            raise RuntimeError(compile_result)
        return {
            "success": True,
            "latex": latex,
            "compile_result": compile_result,
            "blueprint": blueprint,
            "sections": section_bodies,
        }


class ProfessorAgent(BaseResearchAgent):
    role = "computer science professor"

    def generate_readme(self, state):
        registry = ToolRegistry().register(
            "submit_readme",
            "Submit the research README.",
            {
                "type": "object",
                "properties": {
                    "markdown": {
                        "type": "string",
                        "description": (
                            "Complete Markdown README covering the research objective, "
                            "method, repository outputs, reproduction steps, dependencies, "
                            "fixed numerical seeds, observed metrics, and limitations, "
                            "using only supported information."
                        ),
                    },
                },
                "required": ["markdown"],
                "additionalProperties": False,
            },
            lambda markdown: {"success": True, "markdown": markdown},
        )
        return self.runtime("readme writing").run_one(
            system_prompt=professor_readme_prompt(),
            user_prompt=(
                f"Topic: {state.topic}\nPlan: {state.plan}\n"
                f"Dataset selection: {state.dataset_selection}\n"
                f"Fixed numerical seeds: "
                f"{json.dumps(getattr(state, 'seeds', {}), sort_keys=True)}\n"
                f"Experiment code: {state.experiment_code}\n"
                f"Experiment output: {state.experiment_output}\n"
                f"Supplementary experiments: "
                f"{json.dumps(getattr(state, 'supplementary_experiments', []), ensure_ascii=True)}\n"
                f"Interpretation: {state.interpretation}\nPaper: {state.paper}"
            ),
            registry=registry,
        ).value["markdown"]
