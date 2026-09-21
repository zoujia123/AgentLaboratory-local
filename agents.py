import ast
import copy
import hashlib
import importlib
import json
import re
from collections import Counter

from agent_runtime import AgentRuntime, ToolRegistry
from local_replay import current_local_replay
from prompts import (
    literature_review_prompt,
    ml_dataset_selection_prompt,
    phd_interpretation_feedback_prompt,
    phd_plan_feedback_prompt,
    postdoc_interpretation_prompt,
    postdoc_plan_prompt,
    professor_readme_prompt,
    sw_data_preparation_prompt,
)
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
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()



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




DATASET_CODE_MAX_LINES = 100
DATASET_CODE_MAX_LINE_CHARS = 1000
DATASET_CODE_MAX_CHARS = 30000
DATASET_CODE_LIMIT_GUARD = DATASET_CODE_MAX_CHARS
DATASET_CODE_MAX_IMPORT_ITEMS = 40
DATASET_CODE_MAX_REPEATED_IMPORT = 5
DATASET_TASK_TYPES = (
    "classification",
    "regression",
    "text_classification",
    "image_classification",
    "time_series_forecasting",
    "retrieval",
    "clustering",
    "other",
)
FEATURE_REPRESENTATIONS = (
    "numeric_matrix",
    "text",
    "image",
    "sequence",
    "multi_table",
    "other",
)
PREPARED_SPLITS = ("train", "validation", "test")
ARTIFACT_LAYOUTS = ("split_aligned", "role_based")
ROLE_SOURCE_TYPES = ("column", "resource_membership", "derived")
ROLE_ALIGNMENTS = ("sample", "independent")



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

    def conduct_literature_review(self, topic, required_papers=1):
        search = ArxivSearch()
        papers = []
        pending_read = {"arxiv_id": None}
        rejected_papers = set()
        seen_paper_ids = set()
        candidate_batch = {"query": None, "ids": []}
        empty_search_count = {"value": 0}
        paper_pool_exhausted = {"value": False}

        def unavailable_paper_ids():
            return rejected_papers | {
                item["arxiv_id"]
                for item in papers
            }

        def search_arxiv(query, max_results=5):
            result_limit = max(1, min(int(max_results), 5))
            details = search.find_papers(
                query,
                N=result_limit,
                excluded_ids=unavailable_paper_ids(),
            )
            fallback_used = False
            if details["candidate_ids"]:
                empty_search_count["value"] = 0
            else:
                empty_search_count["value"] += 1
                if empty_search_count["value"] >= 2:
                    details = search.browse_remaining_papers(
                        N=result_limit,
                        excluded_ids=unavailable_paper_ids(),
                    )
                    fallback_used = True
                    if details["candidate_ids"]:
                        empty_search_count["value"] = 0
                    else:
                        paper_pool_exhausted["value"] = True
            candidate_batch["query"] = query
            candidate_batch["ids"] = list(details["candidate_ids"])
            seen_paper_ids.update(candidate_batch["ids"])
            return {
                "success": True,
                "results": details["text"],
                "candidate_ids": list(candidate_batch["ids"]),
                "candidate_count": len(candidate_batch["ids"]),
                "fallback_used": fallback_used,
                "consecutive_empty_searches": empty_search_count["value"],
                "decision_required": bool(candidate_batch["ids"]),
                "reconsiderable_paper_ids": sorted(rejected_papers),
                "next_action_hint": (
                    "Read one returned paper or reject the current candidate batch."
                    if candidate_batch["ids"]
                    else (
                        "No new candidates remain. Reconsider a previously returned paper "
                        "or report that none is suitable."
                        if paper_pool_exhausted["value"]
                        else "No candidates were returned. Search once more with a new query."
                    )
                ),
            }

        def read_arxiv_paper(arxiv_id):
            if any(item["arxiv_id"] == arxiv_id for item in papers):
                return {
                    "success": False,
                    "error": f"Paper {arxiv_id!r} is already included in the review",
                }
            if arxiv_id not in seen_paper_ids:
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
            full_text = search.retrieve_full_paper_text(arxiv_id, MAX_LEN=2000)
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
            paper_count = len(papers)
            review_ready = paper_count >= required_papers
            if review_ready:
                next_action_hint = (
                    "The required paper count is satisfied. "
                    "Call submit_literature_review next."
                )
            else:
                next_action_hint = (
                    f"Add {required_papers - paper_count} more reviewed paper(s) "
                    "before calling submit_literature_review."
                )
            return {
                "success": True,
                "paper_count": paper_count,
                "required_papers": required_papers,
                "review_ready": review_ready,
                "next_action_hint": next_action_hint,
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

        def reject_arxiv_candidates(reason):
            if not candidate_batch["ids"]:
                return {
                    "success": False,
                    "error": "There is no current arXiv candidate batch to reject",
                }
            rejected_ids = list(candidate_batch["ids"])
            rejected_papers.update(rejected_ids)
            rejected_query = candidate_batch["query"]
            candidate_batch["query"] = None
            candidate_batch["ids"] = []
            return {
                "success": True,
                "reason": reason,
                "rejected_query": rejected_query,
                "rejected_candidate_ids": rejected_ids,
                "next_action_hint": (
                    "Search with a different query, or read a previously returned "
                    "paper if reconsidering its relevance."
                ),
            }

        def submit_literature_review(review):
            if len(papers) < required_papers:
                return {
                    "success": False,
                    "error": f"Add at least {required_papers} reviewed paper(s) first",
                }
            return {
                "success": True,
                "review": review,
                "papers": papers,
            }

        def report_no_local_paper(reason):
            if not paper_pool_exhausted["value"]:
                return {
                    "success": False,
                    "error": "Unreviewed local arXiv candidates still remain",
                }
            return {
                "success": True,
                "reason": reason,
                "rejected_paper_ids": sorted(rejected_papers),
            }

        def available_tools():
            if pending_read["arxiv_id"] is not None:
                return {"add_paper", "reject_paper"}
            if len(papers) >= required_papers:
                return {"submit_literature_review"}
            readable = seen_paper_ids - {item["arxiv_id"] for item in papers}
            if paper_pool_exhausted["value"]:
                return {"report_no_local_paper"} | ({"read_arxiv_paper"} if readable else set())
            if candidate_batch["ids"]:
                return {"read_arxiv_paper", "reject_arxiv_candidates"}
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
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results from 1 to 5",
                    },
                },
                ["query", "max_results"],
            ),
            search_arxiv,
        ).register(
            "reject_arxiv_candidates",
            (
                "Set aside the current batch only if none offers direct evidence or "
                "useful methodological background. Read uncertain candidates first; "
                "set-aside papers remain readable."
            ),
            _object_schema(
                {"reason": _string("Why none of the current candidates are suitable")},
                ["reason"],
            ),
            reject_arxiv_candidates,
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
                "Include the currently read paper as direct evidence or meaningful "
                "methodological background. Summarize its findings, relevance and "
                "limitations using the same arXiv ID."
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
                "Set aside the currently read paper if it offers no useful evidence "
                "or methodological connection. Give a concise reason for the same "
                "arXiv ID; the paper may be reconsidered later."
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
            "report_no_local_paper",
            (
                "Report that every local arXiv candidate has been reviewed or "
                "rejected and no paper can satisfy the literature review."
            ),
            _object_schema(
                {"reason": _string("Why the local candidate pool cannot satisfy the review")},
                ["reason"],
            ),
            report_no_local_paper,
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
            system_prompt=literature_review_prompt(required_papers),
            user_prompt=(
                f"Research topic: {topic}\n"
                f"Required papers for this review: {required_papers}."
            ),
            registry=registry,
            terminal_tools={"submit_literature_review", "report_no_local_paper"},
            tool_names_provider=available_tools,
        )
        if getattr(result, "tool_name", None) == "report_no_local_paper":
            raise RuntimeError(
                "No suitable local arXiv paper remains: "
                + json.dumps(result.value, ensure_ascii=True, default=str)
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
            "Send results-analysis feedback to the PostdocAgent.",
            _object_schema(
                {"message": _string("Evidence-based interpretation feedback")},
                ["message"],
            ),
            lambda message: {"success": True, "message": message},
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
        ).value["message"]


class PostdocAgent(BaseResearchAgent):
    role = "computer science postdoctoral researcher"

    def plan_turn(self, topic, literature_review, dialogue):
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
                "PhDStudentAgent feedback, without choosing a concrete dataset."
            ),
            _object_schema(
                {"plan": _string("Complete English experimental plan")},
                ["plan"],
            ),
            lambda plan: {"success": True, "plan": plan},
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

    def interpretation_turn(self,topic,plan,code,output,dialogue,literature_review="",supplementary_experiments=None,interpretation_history=None,phd_review_required=None,workflow_notice="",):
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

        registry = ToolRegistry()
        registry.register(
            "send_dialogue",
            (
                "Ask the PhDStudentAgent for focused interpretation feedback. This must "
                "be used before submit_interpretation when no PhD feedback exists."
            ),
            _object_schema(
                {"message": _string("Question or proposed interpretation")},
                ["message"],
            ),
            lambda message: {"success": True, "message": message},
        )
        if not phd_review_required:
            registry.register(
                "request_supplementary_experiment",
                (
                    "Request one bounded additional experiment from MLESolverAgent "
                    "after PhD feedback, without replacing or editing the original "
                    "experiment."
                ),
                _object_schema(
                    {
                        "objective": _string("One specific evidence gap to resolve"),
                        "reason": _string("Why interpretation is unsafe without it"),
                        "required_outputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Exact metrics, comparisons, or diagnostics needed"
                            ),
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Boundaries preserving the original experiment"
                            ),
                        },
                    },
                    ["objective", "reason", "required_outputs", "constraints"],
                ),
                request_supplementary_experiment,
            ).register(
                "submit_interpretation",
                (
                    "Submit the final evidence-based result interpretation after "
                    "receiving and incorporating PhDStudentAgent feedback."
                ),
                _object_schema(
                    {"interpretation": _string("Final English interpretation")},
                    ["interpretation"],
                ),
                lambda interpretation: {
                    "success": True,
                    "interpretation": interpretation,
                },
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
                f"Latest PhD feedback: {dialogue or 'None'}\n"
                f"Workflow notice: {workflow_notice or 'None'}"
            ),
            registry=registry,
        )


def _run_preparation_code(code, workdir):
    """Shared execution for ML trials and SW final submissions."""
    if not code.strip():
        return {"success": False, "code": code, "output": "", "error": "Provide Python code"}
    execution = execute_code(code, timeout=600, workdir=workdir, return_details=True)
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
    return {
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


class MLEngineerAgent(BaseResearchAgent):
    role = "machine learning engineer"

    def data_preparation_turn(
        self, topic, plan, literature_review, sw_message,
        conversation_history="", workdir=None, feedback=None,
        inspected_dataset=None,
        trial_pending=False,
    ):
        search = HFDataSearch()
        must_execute = bool(inspected_dataset) and trial_pending

        def search_huggingface(query, max_results):
            results = search.retrieve_ds(query, N=max(1, min(int(max_results), 10)))
            return {"success": True, "results": results}

        def inspect_huggingface_dataset(
            dataset_id,
            revision,
            config_name,
            split,
        ):
            inspected = search.inspect_dataset(
                dataset_id,
                revision,
                config_name,
                split,
            )
            return {"success": True, **inspected}

        def execute_dataset_code(code):
            return _run_preparation_code(code, workdir)

        registry = ToolRegistry().register(
            "search_huggingface",
            "Search the current query's local Hugging Face datasets.",
            _object_schema({
                "query": _string("Dataset search query"),
                "max_results": {"type": "integer", "description": "Number of results, 1 to 10"},
            }, ["query", "max_results"]),
            search_huggingface,
        ).register(
            "inspect_huggingface_dataset",
            (
                "Read the authoritative schema of one exact local Hugging Face "
                "dataset revision, configuration, and split. Use this after search "
                "and before writing schema-dependent preparation code."
            ),
            _object_schema({
                "dataset_id": _string("Exact dataset ID returned by search_huggingface"),
                "revision": _string("Exact materialized revision returned by search"),
                "config_name": _string("Exact materialized configuration name"),
                "split": _string("Exact materialized split name"),
            }, ["dataset_id", "revision", "config_name", "split"]),
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
                f"Topic: {topic}\nPlan: {plan}\nLiterature review: {literature_review}\n"
                f"Latest SWEngineer message: {sw_message}\n"
                f"Previous ML command result: "
                f"{json.dumps(_inspection_feedback_for_prompt(feedback, inspected_dataset), ensure_ascii=True, default=str)}\n"
                f"Authoritative inspected dataset: "
                f"{json.dumps(inspected_dataset, ensure_ascii=True, default=str)}\n"
                f"Next action: {'Execute the complete preparation code now; schema inspection has succeeded.' if must_execute else 'Choose the next useful action from the available tools.'}\n"
                f"Conversation so far:\n{conversation_history or 'None'}\n"
                f"Execution environment: {json.dumps(_execution_capabilities(), ensure_ascii=True)}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            return_after_tool=True,
            tool_names_provider=lambda: {"execute_dataset_code"} if must_execute else registry.names,
        )


class SWEngineerAgent(BaseResearchAgent):
    role = "software engineer"

    def data_preparation_turn(
        self, topic, plan, workdir, literature_review="",
        conversation_history="", feedback=None, ml_result=None,
        inspected_dataset=None,
    ):
        def submit_dataset_code(code):
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
            return _run_preparation_code(code, workdir)

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
                f"Topic: {topic}\nPlan: {plan}\nLiterature review: {literature_review}\n"
                f"Previous SW command result: {json.dumps(feedback, ensure_ascii=True, default=str)}\n"
                f"Latest ML command result: "
                f"{json.dumps(_inspection_feedback_for_prompt(ml_result, inspected_dataset), ensure_ascii=True, default=str)}\n"
                f"Authoritative inspected dataset: "
                f"{json.dumps(inspected_dataset, ensure_ascii=True, default=str)}\n"
                f"Conversation so far:\n{conversation_history or 'None'}\n"
                f"Execution environment: {json.dumps(_execution_capabilities(), ensure_ascii=True)}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            return_after_tool=True,
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
