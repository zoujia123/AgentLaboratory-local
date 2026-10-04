import ast
import copy
import hashlib
import importlib
import json
import re
from collections import Counter

from agent_runtime import AgentRuntime, ToolRegistry
from execution_diagnostics import classify_runtime_data_failure, runtime_data_repair_hint
from local_replay import current_local_replay
from prompts import literature_review_prompt, ml_dataset_selection_prompt, phd_interpretation_feedback_prompt, phd_plan_feedback_prompt, postdoc_interpretation_prompt, postdoc_plan_prompt, professor_readme_prompt,sw_data_preparation_prompt
from tools import ArxivSearch, HFDataSearch, execute_code, execution_capabilities


def _object_schema(properties, required):
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _string(description):
    return {"type": "string", "description": description}


def _execution_capabilities():
    return execution_capabilities()


def _code_hash(code):
    # A trailing newline or spaces do not make a failed program a new repair.
    canonical_code = str(code).rstrip(" \t\r\n")
    return hashlib.sha256(canonical_code.encode("utf-8")).hexdigest()



def _assigned_name(target):
    return target.id if isinstance(target, ast.Name) else None


def _is_string_sequence(node, known_names):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str)
            for item in node.elts
        )
    if isinstance(node, ast.Name):
        return node.id in known_names
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_string_sequence(node.left, known_names) and _is_string_sequence(
            node.right, known_names
        )
    if isinstance(node, ast.ListComp):
        return _is_string_sequence(node.generators[0].iter, known_names)
    return False


def _is_load_dataset_call(node):
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "load_dataset"
    return isinstance(node.func, ast.Attribute) and node.func.attr == "load_dataset"


def _hf_load_dataset_split_list_line(tree):
    string_sequences = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = _assigned_name(node.targets[0])
            if (
                name
                and name not in string_sequences
                and _is_string_sequence(node.value, string_sequences)
            ):
                string_sequences.add(name)
                changed = True
    for node in ast.walk(tree):
        if not _is_load_dataset_call(node):
            continue
        split_keyword = next(
            (item for item in node.keywords if item.arg == "split"),
            None,
        )
        if split_keyword and _is_string_sequence(
            split_keyword.value,
            string_sequences,
        ):
            return node.lineno
    return None


def _hf_dataset_column_misuse(tree):
    string_sequences = set()
    dataset_objects = set()
    dataset_methods = {
        "filter",
        "map",
        "remove_columns",
        "select",
        "select_columns",
        "shuffle",
        "train_test_split",
    }
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = _assigned_name(node.targets[0])
            if not name:
                continue
            if name not in string_sequences and _is_string_sequence(
                node.value, string_sequences
            ):
                string_sequences.add(name)
                changed = True
            dataset_value = _is_load_dataset_call(node.value)
            if (
                isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in dataset_objects
            ):
                dataset_value = True
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr in dataset_methods
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id in dataset_objects
            ):
                dataset_value = True
            if dataset_value and name not in dataset_objects:
                dataset_objects.add(name)
                changed = True

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "select"
            and node.args
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in dataset_objects
            and _is_string_sequence(node.args[0], string_sequences)
        ):
            return "select", node.lineno
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in dataset_objects
            and _is_string_sequence(node.slice, string_sequences)
        ):
            return "multi_column_index", node.lineno
    return None, None



def _prepared_manifest_validation_issues(contract, manifest, artifacts):
    if not isinstance(manifest, dict):
        return [], [
            "The code did not define prepared_data_manifest for the selected "
            "dataset contract"
        ]
    if manifest.get("_manifest_error"):
        return [], [str(manifest["_manifest_error"])]

    def string_list(name):
        value = manifest.get(name)
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            return None
        return value

    errors = []
    warnings = []
    source_features = string_list("source_feature_columns")
    target_columns = string_list("target_columns")
    analysis_columns = string_list("analysis_columns")
    if source_features is None:
        errors.append("prepared_data_manifest.source_feature_columns must be a string array")
    elif set(source_features) != set(contract.get("feature_columns") or []):
        errors.append(
            "prepared_data_manifest source features do not match dataset_contract"
        )
    if target_columns is None:
        errors.append("prepared_data_manifest.target_columns must be a string array")
    elif set(target_columns) != set(contract.get("target_columns") or []):
        errors.append(
            "prepared_data_manifest target columns do not match dataset_contract"
        )
    if analysis_columns is None:
        warnings.append(
            "prepared_data_manifest.analysis_columns must be a string array"
        )
    elif set(analysis_columns) != set(contract.get("analysis_columns") or []):
        warnings.append(
            "prepared_data_manifest analysis columns do not match dataset_contract"
        )

    feature_artifacts = manifest.get("feature_artifacts")
    target_artifacts = manifest.get("target_artifacts")
    analysis_artifacts = manifest.get("analysis_artifacts")
    role_artifacts = manifest.get("role_artifacts")
    if not isinstance(feature_artifacts, dict):
        errors.append("prepared_data_manifest.feature_artifacts must be an object")
        feature_artifacts = {}
    if not isinstance(target_artifacts, dict):
        errors.append("prepared_data_manifest.target_artifacts must be an object")
        target_artifacts = {}
    if not isinstance(analysis_artifacts, dict):
        warnings.append(
            "prepared_data_manifest.analysis_artifacts must be an object"
        )
        analysis_artifacts = {}
    role_issues = (
        errors
        if contract.get("artifact_layout", "split_aligned") == "role_based"
        else warnings
    )
    if role_artifacts is not None and not isinstance(role_artifacts, dict):
        role_issues.append("prepared_data_manifest.role_artifacts must be an object")
        role_artifacts = {}
    elif not isinstance(role_artifacts, dict):
        role_artifacts = {}

    artifact_index = {
        str(artifact.get("name")): artifact
        for artifact in artifacts
        if artifact.get("name")
    }
    required_splits = contract.get("required_prepared_splits") or []
    feature_metadata = {}
    target_metadata = {}
    if contract.get("artifact_layout", "split_aligned") == "split_aligned":
        for split in required_splits:
            feature_name = feature_artifacts.get(split)
            if not isinstance(feature_name, str) or feature_name not in artifact_index:
                errors.append(
                    f"Missing prepared feature artifact for split {split!r}"
                )
            else:
                feature_metadata[split] = artifact_index[feature_name]
                if (
                    contract.get("feature_representation") == "numeric_matrix"
                    and artifact_index[feature_name].get("numeric") is not True
                ):
                    errors.append(
                        f"Feature artifact {feature_name!r} for split {split!r} "
                        "is not numeric"
                    )
            if contract.get("target_columns"):
                target_name = target_artifacts.get(split)
                if not isinstance(target_name, str) or target_name not in artifact_index:
                    errors.append(
                        f"Missing prepared target artifact for split {split!r}"
                    )
                else:
                    target_metadata[split] = artifact_index[target_name]
                    if (
                        contract.get("task_type") in {
                            "classification",
                            "regression",
                            "text_classification",
                            "image_classification",
                            "time_series_forecasting",
                            "retrieval",
                        }
                        and artifact_index[target_name].get("numeric") is not True
                    ):
                        errors.append(
                            f"Target artifact {target_name!r} for split {split!r} "
                            "is not numeric"
                        )

    feature_widths = set()
    for split, metadata in feature_metadata.items():
        shape = metadata.get("shape")
        if isinstance(shape, list) and len(shape) >= 2:
            feature_widths.add(shape[1])
        target = target_metadata.get(split)
        if target and metadata.get("length") != target.get("length"):
            errors.append(
                f"Feature and target lengths differ for split {split!r}"
            )
    if len(feature_widths) > 1:
        errors.append("Prepared feature dimensions differ across required splits")

    for column in contract.get("analysis_columns") or []:
        reference = analysis_artifacts.get(column)
        names = list(reference.values()) if isinstance(reference, dict) else [reference]
        valid_names = [name for name in names if isinstance(name, str)]
        if not valid_names or any(name not in artifact_index for name in valid_names):
            warnings.append(
                f"Missing prepared analysis artifact for column {column!r}"
            )

    expected_roles = {
        str(source.get("role")): source
        for source in contract.get("role_sources") or []
        if source.get("role")
    }
    unknown_roles = sorted(set(role_artifacts) - set(expected_roles))
    if unknown_roles:
        warnings.append(
            "prepared_data_manifest contains unknown role artifacts: "
            + str(unknown_roles)
        )
    required_role_artifacts = {
        role
        for role, source in expected_roles.items()
        if contract.get("artifact_layout") == "role_based"
        or source.get("source_type") == "derived"
    }
    for role, source in expected_roles.items():
        reference = role_artifacts.get(role)
        if reference is None and role not in required_role_artifacts:
            continue
        role_splits = source.get("required_prepared_splits") or []
        if role_splits:
            if not isinstance(reference, dict):
                role_issues.append(
                    f"Role artifact {role!r} must map prepared splits to variables"
                )
                continue
            missing_role_splits = [
                split for split in role_splits if split not in reference
            ]
            if missing_role_splits:
                role_issues.append(
                    f"Role artifact {role!r} is missing prepared splits: "
                    + str(missing_role_splits)
                )
            references = {
                split: reference.get(split)
                for split in role_splits
                if split in reference
            }
        elif isinstance(reference, str):
            references = {None: reference}
        elif isinstance(reference, dict) and reference:
            references = dict(reference)
        else:
            role_issues.append(f"Missing prepared role artifact for {role!r}")
            continue

        for split, artifact_name in references.items():
            if not isinstance(artifact_name, str) or artifact_name not in artifact_index:
                role_issues.append(
                    f"Role artifact {role!r} references missing variable "
                    f"{artifact_name!r}"
                )
                continue
            if source.get("alignment") != "sample" or split is None:
                continue
            feature = feature_metadata.get(split)
            role_metadata = artifact_index[artifact_name]
            if (
                feature
                and feature.get("length") is not None
                and role_metadata.get("length") != feature.get("length")
            ):
                role_issues.append(
                    f"Role artifact {role!r} length differs from prepared "
                    f"features for split {split!r}"
                )
    return errors, warnings


class BaseResearchAgent:
    role = "research agent"

    def __init__(self, model, api_key, name=None):
        self.model = model
        self.api_key = api_key
        self.name = name or self.__class__.__name__

    def runtime(self, phase, max_tool_steps=8):
        return AgentRuntime(
            model=self.model,
            api_key=self.api_key,
            agent_name=self.name,
            phase=phase,
            max_tool_steps=max_tool_steps,
        )


class PhDStudentAgent(BaseResearchAgent):
    role = "computer science PhD student"

    def conduct_literature_review(self, topic):
        search = ArxivSearch()

        papers = [] # 保存已纳入综述的论文：arxiv_id、摘要、全文。纳入一篇后即可提交。
        pending_read = {"arxiv_id": None}   # 记录“刚读完、但尚未决定添加还是拒绝”的论文 ID。
        rejected_papers = set() # 保存暂时拒绝的论文，避免下一次普通搜索又返回它；重新阅读并添加时会移除。
        seen_paper_ids = set()  # 保存本轮所有搜索曾返回过的 ID，允许从较早搜索批次重新阅读。
        candidate_batch = {"query": None, "ids": []}    # ids 是最新一次搜索中尚待处理的候选；决定当前应继续读还是继续搜索。

        def unavailable_paper_ids():
            return rejected_papers | {item["arxiv_id"] for item in papers}

        def search_arxiv(query):
            details = search.find_papers(
                query,
                excluded_ids=unavailable_paper_ids(),
            )
            candidate_batch["query"] = query
            candidate_batch["ids"] = list(details["candidate_ids"])
            seen_paper_ids.update(candidate_batch["ids"])
            return {
                "success": True,
                "results": details["text"],
                "candidate_ids": list(candidate_batch["ids"]),
                "candidate_count": len(candidate_batch["ids"]),
                "decision_required": bool(candidate_batch["ids"]),
                "reconsiderable_paper_ids": sorted(rejected_papers),
                "next_action_hint": (
                    "Read one returned paper. After reading it, add it or reject it "
                    "with the matching arXiv ID."
                    if candidate_batch["ids"]
                    else "No candidates were returned. Search once more with a new query."
                ),
            }

        def read_arxiv_paper(arxiv_id):
            if any(item["arxiv_id"] == arxiv_id for item in papers):
                return {
                    "success": False,
                    "error": f"Paper {arxiv_id!r} is already included in the review",
                }
            if arxiv_id not in seen_paper_ids:  # 它要求模型只能阅读：本轮文献综述中，之前由 search_arxiv 返回过的论文
                return {
                    "success": False,
                    "error": (
                        f"Paper {arxiv_id!r} has not been returned by search in this review"
                    ),
                    "available_paper_ids": sorted(seen_paper_ids),
                }
            if pending_read["arxiv_id"] is not None:
                return {
                    "success": False,
                    "error": (
                        "Resolve the previously read paper with add_paper or "
                        "reject_paper before reading another paper"
                    ),
                }
            full_text = search.retrieve_full_paper_text(arxiv_id, MAX_LEN=4000)
            pending_read["arxiv_id"] = arxiv_id
            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "decision_required": True,
                "next_action_hint": (
                    "Call add_paper if this paper is relevant, or reject_paper "
                    "if it is not relevant. Use the same arXiv ID."
                ),
            }

        def add_paper(arxiv_id, summary):
            if pending_read["arxiv_id"] != arxiv_id:
                return {
                    "success": False,
                    "error": (
                        "Add the currently pending read paper with arXiv ID "
                        f"{pending_read['arxiv_id']!r}"
                    ),
                }
            full_text = search.get_cached_full_paper_text(arxiv_id)
            if full_text is None:
                return {
                    "success": False,
                    "error": "Read the paper with read_arxiv_paper before adding it",
                }
            if not any(item["arxiv_id"] == arxiv_id for item in papers):
                papers.append({
                    "arxiv_id": arxiv_id,
                    "summary": summary,
                    "full_text": full_text,
                })
            pending_read["arxiv_id"] = None
            rejected_papers.discard(arxiv_id)
            candidate_batch["ids"] = [
                item for item in candidate_batch["ids"] if item != arxiv_id
            ]
            return {
                "success": True,
                "paper_count": len(papers),
                "required_papers": 1,
                "review_ready": True,
                "next_action_hint": (
                    "The required paper count is satisfied. "
                    "Call submit_literature_review next."
                ),
            }

        def reject_paper(arxiv_id, reason):
            if pending_read["arxiv_id"] != arxiv_id:
                return {
                    "success": False,
                    "error": (
                        "Reject the currently pending read paper with arXiv ID "
                        f"{pending_read['arxiv_id']!r}"
                    ),
                }
            rejected_papers.add(arxiv_id)
            pending_read["arxiv_id"] = None
            candidate_batch["ids"] = [
                item for item in candidate_batch["ids"] if item != arxiv_id
            ]
            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "reason": reason,
                "rejected_paper_count": len(rejected_papers),
                "next_action_hint": (
                    "The paper is set aside. Consider other candidates, or reread it "
                    "if you have a new reason to reconsider its relevance."
                ),
            }

        def submit_literature_review(review):
            if not papers:
                return {
                    "success": False,
                    "error": "Add at least 1 reviewed paper(s) first",
                }
            return {
                "success": True,
                "review": review,
                "papers": papers,
            }

        def available_tools():
            if pending_read["arxiv_id"] is not None:
                return {"add_paper", "reject_paper"}
            if papers:
                return {"submit_literature_review"}
            readable = seen_paper_ids - {item["arxiv_id"] for item in papers}   # 系统允许模型之后有新理由时重新阅读曾 reject 的论文，而不是永久禁止它。
            if candidate_batch["ids"]:
                return {"read_arxiv_paper"}
            return {"search_arxiv"} | ({"read_arxiv_paper"} if readable else set())

        registry = ToolRegistry()
        registry.register(
            "search_arxiv",
            (
                "Search this query's local arXiv papers with a short query, excluding "
                "already included or set-aside papers from new results."
            ),
            _object_schema(
                {
                    "query": _string("Short English paper search query"),
                },
                ["query"],
            ),
            search_arxiv,
        ).register(
            "read_arxiv_paper",
            (
                "Read a paper returned by any search in this review, including a "
                "previously rejected paper. Then call add_paper or reject_paper for "
                "the same arXiv ID."
            ),
            _object_schema(
                {"arxiv_id": _string("arXiv identifier returned by search_arxiv")},
                ["arxiv_id"],
            ),
            read_arxiv_paper,
        ).register(
            "add_paper",
            (
                "Include the currently read paper when it either directly supports the "
                "study or provides transferable methodological, comparative, evaluative "
                "or conceptual background. The dataset, domain, model, task and exact "
                "research question need not match. Summarize what transfers, findings, "
                "relevance and limitations using the same arXiv ID."
            ),
            _object_schema(
                {
                    "arxiv_id": _string("Identifier of the paper already read"),
                    "summary": _string("Accurate summary and relevant experimental findings"),
                },
                ["arxiv_id", "summary"],
            ),
            add_paper,
        ).register(
            "reject_paper",
            (
                "Set aside the currently read paper only when its full text has no "
                "substantive connection after considering transferable methods, "
                "comparisons, evaluations or concepts. A mismatch in dataset, domain, "
                "model, task, wording or exact research question alone is never enough. "
                "Give a concise reason for the same arXiv ID; the paper may be "
                "reconsidered later."
            ),
            _object_schema(
                {
                    "arxiv_id": _string("Identifier of the paper just read"),
                    "reason": _string("Concise evidence-based rejection reason"),
                },
                ["arxiv_id", "reason"],
            ),
            reject_paper,
        ).register(
            "submit_literature_review",
            "Submit the completed literature review.",
            _object_schema(
                {"review": _string("Integrated English literature review")},
                ["review"],
            ),
            submit_literature_review,
        )
        result = self.runtime("literature review", max_tool_steps=20).run(
            system_prompt=literature_review_prompt(),
            user_prompt=(
                f"Research topic: {topic}\n"
                "Required papers for this review: 1."
            ),
            registry=registry,
            terminal_tools={"submit_literature_review"},
            tool_names_provider=available_tools,
        )
        return result.value

    def plan_feedback(self, topic, literature_review, postdoc_message):
        registry = ToolRegistry().register(
            "send_dialogue",
            "Send planning feedback to the PostdocAgent.",
            _object_schema(
                {"message": _string("Concise planning feedback")},
                ["message"],
            ),
            lambda message: {"success": True, "message": message},
        )
        return self.runtime("plan formulation", max_tool_steps=1).run_one(
            system_prompt=phd_plan_feedback_prompt(),
            user_prompt=(
                f"Topic: {topic}\nLiterature review: {literature_review}\n"
                f"Postdoc message: {postdoc_message}"
            ),
            registry=registry,
        ).value["message"]

    def interpretation_feedback(self,topic,plan,experiment_output,postdoc_message,experiment_code="",literature_review="",supplementary_experiments=None,interpretation_history=None,):
        registry = ToolRegistry().register(
            "send_dialogue",
            (
                "Send results-analysis feedback to the PostdocAgent and state "
                "whether one bounded, executable supplementary experiment is "
                "necessary before final interpretation."
            ),
            _object_schema(
                {
                    "message": _string("Evidence-based interpretation feedback"),
                    "supplement_required": {
                        "type": "boolean",
                        "description": (
                            "True only when one specific bounded executable "
                            "supplement is necessary before final interpretation"
                        ),
                    },
                },
                ["message", "supplement_required"],
            ),
            lambda message, supplement_required: {
                "success": True,
                "message": message,
                "supplement_required": supplement_required,
            },
        )
        return self.runtime("results interpretation", max_tool_steps=1).run_one(
            system_prompt=phd_interpretation_feedback_prompt(),
            user_prompt=(
                f"Topic: {topic}\nPlan: {plan}\n"
                f"Literature review: {literature_review}\n"
                f"Original experiment code: {experiment_code}\n"
                f"Original results: {experiment_output}\n"
                f"Supplementary experiments: "
                f"{json.dumps(supplementary_experiments or [], ensure_ascii=True)}\n"
                f"Interpretation history: "
                f"{json.dumps(interpretation_history or [], ensure_ascii=True)}\n"
                f"Postdoc message: {postdoc_message}"
            ),
            registry=registry,
        ).value


class PostdocAgent(BaseResearchAgent):
    role = "computer science postdoctoral researcher"

    def plan_turn(self, topic, literature_review, dialogue):
        def submit_plan(plan, data_requirements):
            if not str(plan).strip() or not str(data_requirements).strip():
                return {
                    "success": False,
                    "error": "Both plan and data_requirements must be non-empty",
                }
            return {
                "success": True,
                "plan": plan,
                "data_requirements": data_requirements,
            }

        registry = ToolRegistry().register(
            "send_dialogue",
            (
                "Ask the PhDStudentAgent for focused planning feedback. This must be "
                "used before submit_plan when no PhD feedback has been received."
            ),
            _object_schema(
                {"message": _string("Question or proposed planning direction")},
                ["message"],
            ),
            lambda message: {"success": True, "message": message},
        ).register(
            "submit_plan",
            (
                "Submit the final experimental plan after receiving and incorporating "
                "PhDStudentAgent feedback, together with a separate concise abstract "
                "data-preparation requirements brief. Do not choose a concrete dataset."
            ),
            _object_schema(
                {
                    "plan": _string("Complete English experimental plan for MLESolver"),
                    "data_requirements": _string(
                        "Concise abstract data requirements for SWEngineerAgent and "
                        "MLEngineerAgent before schema inspection"
                    ),
                },
                ["plan", "data_requirements"],
            ),
            submit_plan,
        )
        return self.runtime("plan formulation", max_tool_steps=1).run_one(
            system_prompt=postdoc_plan_prompt(),
            user_prompt=(
                f"Topic: {topic}\nLiterature review: {literature_review}\n"
                f"Planning consultation status: "
                f"{'feedback_received' if dialogue else 'consultation_required'}\n"
                f"Latest PhD feedback: {dialogue or 'None'}"
            ),
            registry=registry,
        )

    def interpretation_turn(self,topic,plan,code,output,dialogue,literature_review="",supplementary_experiments=None,interpretation_history=None,phd_review_required=None,supplement_required=None,force_submit=False,workflow_notice="",):
        if phd_review_required is None:
            phd_review_required = not bool(dialogue)

        def request_supplementary_experiment(objective,reason,required_outputs,constraints,):
            return {
                "success": True,
                "request": {
                    "objective": objective,
                    "reason": reason,
                    "required_outputs": list(required_outputs),
                    "constraints": list(constraints),
                },
            }

        request_schema = _object_schema(
            {
                "objective": _string("One specific evidence gap to resolve"),
                "reason": _string("Why interpretation is unsafe without it"),
                "required_outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact metrics, comparisons, or diagnostics needed",
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Boundaries preserving the original experiment",
                },
            },
            ["objective", "reason", "required_outputs", "constraints"],
        )
        submit_schema = _object_schema(
            {"interpretation": _string("Final English interpretation")},
            ["interpretation"],
        )
        submit_handler = lambda interpretation: {
            "success": True,
            "interpretation": interpretation,
        }
        registry = ToolRegistry()
        if phd_review_required:
            registry.register(
                "send_dialogue",
                (
                    "Ask the PhDStudentAgent for focused interpretation feedback. This "
                    "must be used before any final interpretation decision."
                ),
                _object_schema(
                    {"message": _string("Question or proposed interpretation")},
                    ["message"],
                ),
                lambda message: {"success": True, "message": message},
            )
        elif supplement_required is True and not force_submit:
            registry.register(
                "request_supplementary_experiment",
                (
                    "Request the one bounded, concrete supplementary experiment "
                    "identified as necessary by PhD review. It must preserve the "
                    "original experiment and cannot replace or edit it."
                ),
                request_schema,
                request_supplementary_experiment,
            )
        elif supplement_required is None and not force_submit:
            registry.register(
                "request_supplementary_experiment",
                (
                    "Request one bounded, materially new supplementary experiment "
                    "only when it is necessary for a sound interpretation. It must "
                    "preserve the original experiment and cannot replace or edit it."
                ),
                request_schema,
                request_supplementary_experiment,
            ).register(
                "submit_interpretation",
                (
                    "Submit the final evidence-based result interpretation after "
                    "receiving and incorporating PhDStudentAgent feedback. State "
                    "unresolved evidence gaps as limitations rather than repeating "
                    "dialogue or replacing the original experiment."
                ),
                submit_schema,
                submit_handler,
            )
        else:
            registry.register(
                "submit_interpretation",
                (
                    "Submit the final evidence-based result interpretation after "
                    "receiving and incorporating PhDStudentAgent feedback. State "
                    "unresolved evidence gaps as limitations rather than repeating "
                    "dialogue or replacing the original experiment."
                ),
                submit_schema,
                submit_handler,
            )
        return self.runtime("results interpretation", max_tool_steps=1).run_one(
            system_prompt=postdoc_interpretation_prompt(),
            user_prompt=(
                f"Topic: {topic}\nPlan: {plan}\n"
                f"Literature review: {literature_review}\n"
                f"Original experiment code: {code}\n"
                f"Original experiment output: {output}\n"
                f"Supplementary experiments: "
                f"{json.dumps(supplementary_experiments or [], ensure_ascii=True)}\n"
                f"Interpretation history: "
                f"{json.dumps(interpretation_history or [], ensure_ascii=True)}\n"
                f"Latest evidence review status: "
                f"{'review_required' if phd_review_required else 'review_complete'}\n"
                f"Supplement decision from PhD review: "
                f"{'required' if supplement_required is True else ('not_required' if supplement_required is False else 'undecided')}\n"
                f"Final submission required: {bool(force_submit)}\n"
                f"Latest PhD feedback: {dialogue or 'None'}\n"
                f"Workflow notice: {workflow_notice or 'None'}"
            ),
            registry=registry,
        )

_DATA_PREPARATION_FORBIDDEN_MODULE_PREFIXES = (
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.neural_network",
    "sklearn.cluster",
    "sklearn.naive_bayes",
    "sklearn.discriminant_analysis",
    "sklearn.gaussian_process",
    "sklearn.metrics",
    "xgboost",
    "lightgbm",
    "catboost",
    "tensorflow",
    "keras",
    "torch.nn",
    "torch.optim",
)

_DATA_PREPARATION_FORBIDDEN_METHODS = {
    "predict",
    "predict_proba",
    "predict_log_proba",
    "fit_predict",
    "score",
    "evaluate",
    "backward",
}

_DATA_PREPARATION_FORBIDDEN_CALLS = {
    "GridSearchCV",
    "RandomizedSearchCV",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "validation_curve",
}

_DATA_PREPARATION_FORBIDDEN_METRIC_CALLS = {
    "accuracy_score",
    "balanced_accuracy_score",
    "classification_report",
    "confusion_matrix",
    "f1_score",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
}

_DATA_PREPARATION_FORBIDDEN_NAMESPACE_IMPORTS = {
    "sklearn": {
        "cluster",
        "discriminant_analysis",
        "ensemble",
        "gaussian_process",
        "linear_model",
        "metrics",
        "naive_bayes",
        "neighbors",
        "neural_network",
        "svm",
        "tree",
    },
    "torch": {"nn", "optim"},
}


def _dotted_ast_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_data_preparation_forbidden_module(module_name):
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in _DATA_PREPARATION_FORBIDDEN_MODULE_PREFIXES
    )


def _data_preparation_scope_violations(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # The execution layer already returns a precise syntax diagnostic.
        return []

    violations = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_data_preparation_forbidden_module(alias.name):
                    violations.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if _is_data_preparation_forbidden_module(module_name):
                violations.add(f"import from {module_name}")
            elif module_name in _DATA_PREPARATION_FORBIDDEN_NAMESPACE_IMPORTS:
                for alias in node.names:
                    if alias.name in _DATA_PREPARATION_FORBIDDEN_NAMESPACE_IMPORTS[module_name]:
                        violations.add(f"import {module_name}.{alias.name}")
        elif isinstance(node, ast.Call):
            dotted_name = _dotted_ast_name(node.func)
            call_name = dotted_name.rsplit(".", 1)[-1]
            if _is_data_preparation_forbidden_module(dotted_name):
                violations.add(f"call {dotted_name}()")
            if call_name in _DATA_PREPARATION_FORBIDDEN_METHODS:
                violations.add(f"call .{call_name}()")
            if call_name in _DATA_PREPARATION_FORBIDDEN_CALLS:
                violations.add(f"call {call_name}()")
            if call_name in _DATA_PREPARATION_FORBIDDEN_METRIC_CALLS:
                violations.add(f"call {call_name}()")
            if call_name.endswith(("Classifier", "Regressor")):
                violations.add(f"construct {call_name}()")
    return sorted(violations)


_KEY_ERROR_COLUMN_RE = re.compile(r"KeyError:\s*['\"]([^'\"]+)['\"]")
_GENERATED_CODE_LINE_RE = re.compile(r'File "<string>", line (\d+)')
_DATAFRAME_TRANSFORMATION_MARKERS = (
    ("pd.get_dummies", "one-hot encoding"),
    (".drop(", "column removal"),
    (".rename(", "column renaming"),
    (".set_index(", "index restructuring"),
    (".reset_index(", "index restructuring"),
    (".merge(", "dataframe merge"),
    (".join(", "dataframe join"),
    (".pivot(", "dataframe pivot"),
    (".melt(", "dataframe reshape"),
)


def _data_preparation_syntax_repair_hint(code):
    """Point a repair attempt at the local syntax error and its nearby code."""
    try:
        ast.parse(str(code))
    except SyntaxError as exc:
        if not exc.lineno:
            return None
        code_lines = str(code).splitlines()
        start = max(0, exc.lineno - 3)
        end = min(len(code_lines), exc.lineno + 2)
        context = "\n".join(
            f"{index + 1}: {code_lines[index]}"
            for index in range(start, end)
        )
        return (
            f"Python syntax validation failed at line {exc.lineno}: {exc.msg}. "
            "Correct the shown line and its surrounding indentation, parentheses, "
            "or commas before retrying.\nRelevant code:\n"
            + context
        )
    return None


def _data_preparation_key_error_repair_hint(
    code,
    error,
    traceback_text,
    inspected_dataset,
):
    match = _KEY_ERROR_COLUMN_RE.search(str(error or ""))
    if match is None:
        match = _KEY_ERROR_COLUMN_RE.search(str(traceback_text or ""))
    if match is None:
        return None
    missing_column = match.group(1)
    schema = (
        inspected_dataset.get("dataset_schema", {})
        if isinstance(inspected_dataset, dict)
        else {}
    )
    columns = schema.get("columns", {}) if isinstance(schema, dict) else {}
    if missing_column not in columns:
        return None

    code_lines = str(code).splitlines()
    generated_lines = _GENERATED_CODE_LINE_RE.findall(str(traceback_text or ""))
    if generated_lines:
        failure_line = int(generated_lines[-1])
        relevant_code = "\n".join(code_lines[:max(0, failure_line - 1)])
    else:
        relevant_code = str(code)
    transformations = [
        label
        for marker, label in _DATAFRAME_TRANSFORMATION_MARKERS
        if marker in relevant_code
    ]
    if not transformations:
        return None
    transformation_summary = ", ".join(dict.fromkeys(transformations))
    return (
        f"The missing column {missing_column!r} exists in the inspected raw schema. "
        "The KeyError occurred after a dataframe transformation "
        f"({transformation_summary}), so do not re-inspect the raw schema. "
        "Trace the transformed dataframe, preserve target and analysis-role columns "
        "before feature transformations, encode a separate feature dataframe, and "
        "do not access original role columns from the transformed dataframe."
    )


def _run_preparation_code(code, workdir, inspected_dataset=None):
    """Shared execution for ML trials and SW final submissions."""
    if not code.strip():
        return {"success": False, "code": code, "output": "", "error": "Provide Python code"}
    violations = _data_preparation_scope_violations(code)
    if violations:
        return {
            "success": False,
            "error_type": "data_preparation_scope_violation",
            "error": (
                "Data preparation code may only prepare reusable data. Move model "
                "training, prediction, and final evaluation to MLESolver: "
                + ", ".join(violations)
            ),
            "traceback": "",
            "exit_code": None,
            "code": code,
            "output": "",
            "selection": {},
            "artifacts": [],
            "prepared_data_manifest": None,
        }
    execution = execute_code(code, timeout=900, workdir=workdir, return_details=True)
    details = execution
    output = details["output"]
    selection = {}
    for event in details.get("replay_events", []):
        if event.get("tool") == "huggingface.load_dataset" and event.get("success") is True:
            request = event.get("request") or {}
            selection = {
                "dataset_id": request.get("dataset_id"),
                "config_name": request.get("config_name"),
                "split": request.get("split"),
                "revision": event.get("object_revision") or request.get("revision"),
            }
            break
    result = {
        "success": details["success"],
        "error_type": details["error_type"],
        "error": details["error"],
        "traceback": details["traceback"],
        "exit_code": details["exit_code"],
        "code": code,
        "output": output,
        "selection": selection,
        "artifacts": list(details.get("artifacts", [])),
        "prepared_data_manifest": details.get("prepared_data_manifest"),
    }
    if result["success"] is False:
        category = classify_runtime_data_failure(
            result["error"], result["traceback"]
        )
        if category is not None:
            result["failure_category"] = category
            repair_hint = runtime_data_repair_hint(category)
        else:
            repair_hint = _data_preparation_syntax_repair_hint(code)
        if repair_hint is None:
            repair_hint = _data_preparation_key_error_repair_hint(
                code,
                result["error"],
                result["traceback"],
                inspected_dataset,
            )
        if repair_hint:
            result["repair_hint"] = repair_hint
    return result


def _inspection_feedback_for_prompt(feedback, inspected_dataset):
    """Reference an identical inspection already included separately in this prompt."""
    if (
        not isinstance(feedback, dict)
        or feedback.get("tool") != "inspect_huggingface_dataset"
        or feedback.get("success") is not True
        or not isinstance(inspected_dataset, dict)
        or inspected_dataset.get("success", True) is not True
        or not inspected_dataset.get("dataset_schema")
    ):
        return feedback
    identity_fields = ("dataset_id", "revision", "config_name", "split")
    if any(not inspected_dataset.get(name) for name in identity_fields):
        return feedback
    feedback_details = {key: value for key, value in feedback.items() if key not in {"tool", "success"}}
    inspected_details = {key: value for key, value in inspected_dataset.items() if key not in {"tool", "success"}}
    if feedback_details != inspected_details:
        return feedback
    return {
        "tool": "inspect_huggingface_dataset",
        "success": True,
        "message": "Inspection succeeded; see Authoritative inspected dataset below for the complete result.",
    }


def _authoritative_execution_failure(value):
    if not isinstance(value, dict) or value.get("success") is not False:
        return None
    return dict(value)


def _stale_diagnostic_marker():
    return {
        "message": (
            "A newer execution failure is supplied below as the authoritative "
            "diagnostic. Do not rely on older error analysis or repair advice."
        )
    }


class MLEngineerAgent(BaseResearchAgent):
    role = "machine learning engineer"

    @staticmethod
    def _select_first_complete_dataset(results):
        """Choose the first locally materialized candidate without using query-specific rules."""
        for item in results or []:
            if not isinstance(item, dict):
                continue
            if item.get("snapshot_status") not in (None, "complete"):
                continue
            dataset_id = item.get("dataset_id") or item.get("id")
            revision = item.get("revision")
            if not dataset_id or not revision:
                continue

            config_name = None
            split = None
            materialized = item.get("materialized") or {}
            if isinstance(materialized, dict):
                for candidate_config, split_map in materialized.items():
                    if not isinstance(split_map, dict) or not split_map:
                        continue
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

    def data_preparation_turn(
        self, topic, data_requirements, literature_review, sw_message,
        conversation_history="", workdir=None, feedback=None,
        inspected_dataset=None,
        trial_pending=False,
        selected_dataset=None,
        latest_execution_failure=None,
        failed_trial_code_failures=None,
    ):
        search = HFDataSearch()
        must_execute = bool(inspected_dataset) and trial_pending
        if isinstance(selected_dataset, dict):
            selected_dataset = dict(selected_dataset)
        elif isinstance(feedback, dict) and isinstance(feedback.get("selected_dataset"), dict):
            selected_dataset = dict(feedback["selected_dataset"])
        else:
            selected_dataset = None
        latest_execution_failure = _authoritative_execution_failure(
            latest_execution_failure
        )
        if failed_trial_code_failures is None:
            failed_trial_code_failures = {}

        def search_huggingface(query, max_results):
            results = search.retrieve_ds(query, N=max(1, min(int(max_results), 10)))
            selected = self._select_first_complete_dataset(results)
            return {
                "success": True,
                "results": results,
                "selected_dataset": selected,
                "next_action_hint": (
                    "A complete local candidate was selected. Call "
                    "inspect_huggingface_dataset for that candidate; do not search again."
                    if selected
                    else "No complete local candidate was returned. Search with a different query."
                ),
            }

        def inspect_huggingface_dataset(
            dataset_id=None,
            revision=None,
            config_name=None,
            split=None,
        ):
            if selected_dataset:
                dataset_id = selected_dataset["dataset_id"]
                revision = selected_dataset["revision"]
                config_name = selected_dataset["config_name"]
                split = selected_dataset["split"]
            elif not all((dataset_id, revision, config_name, split)):
                return {
                    "success": False,
                    "error_type": "missing_dataset_selection",
                    "error": "Provide a complete dataset selection before inspection",
                }
            inspected = search.inspect_dataset(
                dataset_id,
                revision,
                config_name,
                split,
            )
            return {
                "success": True,
                **inspected,
                "selected_dataset": selected_dataset,
            }

        def execute_dataset_code(code):
            fingerprint = _code_hash(code)
            if fingerprint in failed_trial_code_failures:
                prior_failure = dict(
                    failed_trial_code_failures.get(fingerprint) or {}
                )
                original_error = prior_failure.get("error") or "the prior failure"
                return {
                    "success": False,
                    "error_type": "unchanged_failed_dataset_code",
                    "error": (
                        "This exact data-preparation code already failed. It was not "
                        "executed again."
                    ),
                    "repair_hint": (
                        "Make a substantive code change that resolves the original "
                        f"failure before retrying: {original_error}"
                    ),
                    "original_error_type": prior_failure.get("error_type"),
                    "original_error": prior_failure.get("error"),
                    "original_repair_hint": prior_failure.get("repair_hint"),
                    "code": code,
                    "code_hash": fingerprint,
                    "output": "",
                    "traceback": "",
                    "exit_code": None,
                    "selection": {},
                    "artifacts": [],
                    "prepared_data_manifest": None,
                    "selected_dataset": selected_dataset,
                }
            result = {
                **_run_preparation_code(code, workdir, inspected_dataset),
                "code_hash": fingerprint,
                "selected_dataset": selected_dataset,
            }
            if result.get("success") is False:
                failed_trial_code_failures[fingerprint] = {
                    key: result.get(key)
                    for key in ("error_type", "error", "repair_hint")
                    if result.get(key)
                }
            return result

        selection_pending = bool(selected_dataset) and not inspected_dataset
        selected_and_inspected = bool(selected_dataset) and bool(inspected_dataset)
        inspection_schema = (
            _object_schema({}, [])
            if selection_pending
            else _object_schema({
                "dataset_id": _string("Exact dataset ID returned by search_huggingface"),
                "revision": _string("Exact materialized revision returned by search"),
                "config_name": _string("Exact materialized configuration name"),
                "split": _string("Exact materialized split name"),
            }, ["dataset_id", "revision", "config_name", "split"])
        )
        inspection_description = (
            "Inspect the automatically selected complete local dataset. No arguments "
            "are needed; the framework supplies the dataset identity."
            if selection_pending
            else (
                "Read the authoritative schema of one exact local Hugging Face "
                "dataset revision, configuration, and split. Use this after search "
                "and before writing schema-dependent preparation code."
            )
        )
        registry = ToolRegistry().register(
            "search_huggingface",
            (
                "Search the current query's local Hugging Face datasets. When a "
                "complete candidate is returned, the framework selects the first one "
                "and exposes only inspect_huggingface_dataset next."
            ),
            _object_schema({
                "query": _string("Dataset search query"),
                "max_results": {"type": "integer", "description": "Number of results, 1 to 10"},
            }, ["query", "max_results"]),
            search_huggingface,
        ).register(
            "inspect_huggingface_dataset",
            inspection_description,
            inspection_schema,
            inspect_huggingface_dataset,
        ).register(
            "execute_dataset_code",
            "Run complete candidate data-preparation Python code and inspect its output. "
            "This is a trial; the SWEngineerAgent submits the final code.",
            _object_schema({"code": _string("Complete candidate Python code")}, ["code"]),
            execute_dataset_code,
        ).register(
            "send_dialogue",
            "Reply to the SWEngineerAgent with dataset choices, code suggestions or questions.",
            _object_schema({"message": _string("Message to the software engineer")}, ["message"]),
            lambda message: {"success": True, "message": message},
        )
        return self.runtime("data preparation collaboration", max_tool_steps=1).run(
            system_prompt=ml_dataset_selection_prompt(),
            user_prompt=(
                f"Topic: {topic}\n"
                f"Data preparation requirements: {data_requirements}\n"
                f"Latest SWEngineer message: "
                f"{'Repair the authoritative latest execution failure below.' if latest_execution_failure else sw_message}\n"
                f"Previous ML command result: "
                f"{json.dumps(_stale_diagnostic_marker() if latest_execution_failure else _inspection_feedback_for_prompt(feedback, inspected_dataset), ensure_ascii=True, default=str)}\n"
                f"Authoritative latest execution failure: "
                f"{json.dumps(latest_execution_failure, ensure_ascii=True, default=str)}\n"
                f"Authoritative inspected dataset: "
                f"{json.dumps(inspected_dataset, ensure_ascii=True, default=str)}\n"
                f"Automatically selected local dataset: "
                f"{json.dumps(selected_dataset, ensure_ascii=True, default=str)}\n"
                f"Next action: "
                f"{'Execute the complete preparation code now; schema inspection has succeeded.' if must_execute else ('Inspect the automatically selected local candidate; do not search again.' if selection_pending else ('Repair and execute preparation code for the selected, inspected dataset.' if selected_and_inspected else 'Choose the next useful action from the available tools.'))}\n"
                f"Conversation so far:\n{conversation_history or 'None'}\n"
                f"Execution environment: {json.dumps(_execution_capabilities(), ensure_ascii=True)}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            return_after_tool=True,
            tool_names_provider=(
                lambda: {"execute_dataset_code"}
                if must_execute or selected_and_inspected
                else ({"inspect_huggingface_dataset"} if selection_pending else registry.names)
            ),
        )


class SWEngineerAgent(BaseResearchAgent):
    role = "software engineer"

    def data_preparation_turn(
        self, topic, data_requirements, workdir, literature_review="",
        conversation_history="", feedback=None, ml_result=None,
        inspected_dataset=None,
        latest_execution_failure=None,
        verified_trial=None,
    ):
        def submit_dataset_code(code=None):
            if isinstance(verified_trial, dict):
                return dict(verified_trial)
            if not inspected_dataset:
                return {
                    "success": False,
                    "error_type": "schema_not_inspected",
                    "error": (
                        "Inspect the selected Hugging Face dataset schema before "
                        "submitting data-preparation code"
                    ),
                    "code": code,
                    "output": "",
                }
            return _run_preparation_code(code, workdir, inspected_dataset)

        # 据 verified_trial 是否存在，注册不同版本的工具。
        if isinstance(verified_trial, dict):    # MLE 已经验证成功,不传任何参数
            registry = ToolRegistry().register(
                "submit_dataset_code",
                (
                    "Promote the exact successful MLE trial code and execution result. "
                    "No arguments are accepted and the code is not executed again."
                ),
                _object_schema({}, []),
                submit_dataset_code,
            )
        else:   # 还没有验证成功,一整段新的数据准备代码
            registry = ToolRegistry().register(
                "send_dialogue",
                "Direct or consult the MLEngineerAgent about dataset search and preparation code.",
                _object_schema({"message": _string("Message to the ML engineer")}, ["message"]),
                lambda message: {"success": True, "message": message},
            ).register(
                "submit_dataset_code",
                "Execute the complete final data-preparation code. Successful execution "
                "submits it; on failure the error returns to the collaboration.",
                _object_schema({"code": _string("Complete final Python data-preparation code")}, ["code"]),
                submit_dataset_code,
            )
        return self.runtime("data preparation collaboration", max_tool_steps=1).run(
            system_prompt=sw_data_preparation_prompt(),
            user_prompt=(
                f"Topic: {topic}\n"
                f"Data preparation requirements: {data_requirements}\n"
                f"Previous SW command result: "
                f"{json.dumps(_stale_diagnostic_marker() if _authoritative_execution_failure(latest_execution_failure) else feedback, ensure_ascii=True, default=str)}\n"
                f"Latest ML command result: "
                f"{json.dumps(_stale_diagnostic_marker() if _authoritative_execution_failure(latest_execution_failure) else _inspection_feedback_for_prompt(ml_result, inspected_dataset), ensure_ascii=True, default=str)}\n"
                f"Authoritative latest execution failure: "
                f"{json.dumps(_authoritative_execution_failure(latest_execution_failure), ensure_ascii=True, default=str)}\n"
                f"Authoritative inspected dataset: "
                f"{json.dumps(inspected_dataset, ensure_ascii=True, default=str)}\n"
                f"Verified MLE trial ready for direct submission: "
                f"{isinstance(verified_trial, dict)}\n"
                f"Conversation so far:\n{conversation_history or 'None'}\n"
                f"Execution environment: {json.dumps(_execution_capabilities(), ensure_ascii=True)}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            return_after_tool=True, # 无论成功还是失败，只要执行过一个工具，就把控制权交回外层协作流程
            automatic_tools={"submit_dataset_code"} if isinstance(verified_trial, dict) else set(), # 框架要不要跳过模型，自己直接调用工具.
            # 若 verified_trial 存在：直接submit_dataset_code()，不会请求大模型。若不存在，automatic_tools = set()，需要调用模型让模型选择
        )


class ProfessorAgent(BaseResearchAgent):
    role = "computer science professor"

    def generate_readme(self, state):
        registry = ToolRegistry().register(
            "submit_readme",
            "Submit the final GitHub README in Markdown.",
            _object_schema(
                {"markdown": _string("Complete README Markdown")},
                ["markdown"],
            ),
            lambda markdown: {"success": True, "markdown": markdown},
        )
        return self.runtime("readme writing", max_tool_steps=1).run_one(
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
