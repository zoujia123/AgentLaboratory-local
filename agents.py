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
from agent_runtime import AgentRuntime, ToolRegistry
from data_preparation import _run_preparation_code
from execution_diagnostics import python_syntax_error
from prompts import (
    experiment_solver_prompt,
    literature_review_prompt,
    ml_dataset_selection_prompt,
    paper_blueprint_prompt,
    paper_section_prompt,
    phd_interpretation_feedback_prompt,
    phd_plan_feedback_prompt,
    postdoc_interpretation_prompt,
    postdoc_plan_prompt,
    professor_readme_prompt,
    supplementary_experiment_solver_prompt,
    sw_data_preparation_prompt,
)
from tools import ArxivSearch, HFDataSearch, execute_code, execution_capabilities
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
        search = ArxivSearch()

        papers = [] # 保存已纳入综述的论文：arxiv_id、摘要、全文。纳入一篇后即可提交。
        pending_read = {"arxiv_id": None}   # 记录“刚读完、但尚未决定添加还是拒绝”的论文 ID。
        rejected_papers = set() # 本轮拒绝过的论文不再返回或重新阅读。
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
            return {
                "success": True,
                "results": details["text"],
                "candidate_ids": list(candidate_batch["ids"]),
                "candidate_count": len(candidate_batch["ids"]),
                "decision_required": bool(candidate_batch["ids"]),
                "next_action_hint": (
                    "Read a candidate from this search, then call decide_paper once with the "
                    "matching arXiv ID and decision 'add' or 'reject'."
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
            if arxiv_id not in candidate_batch["ids"] or arxiv_id in rejected_papers:
                return {
                    "success": False,
                    "error": (
                        f"Paper {arxiv_id!r} is not an unread candidate from the latest search"
                    ),
                    "available_paper_ids": [
                        item for item in candidate_batch["ids"] if item not in rejected_papers
                    ],
                }
            if pending_read["arxiv_id"] is not None:
                return {
                    "success": False,
                    "error": (
                        "Resolve the previously read paper with decide_paper "
                        "before reading another paper"
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
                    "Call decide_paper exactly once for this arXiv ID. Choose "
                    "decision 'add' for useful direct or methodological relevance, "
                    "or 'reject' when neither applies, and provide an explanation."
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
            candidate_batch["ids"] = []  # 拒绝后结束当前批次，下一步必须重新搜索。
            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "reason": reason,
                "rejected_paper_count": len(rejected_papers),
                "next_action_hint": (
                    "The paper was rejected. Call search_arxiv next to find a new "
                    "candidate; rejected papers cannot be reread in this review."
                ),
            }

        def decide_paper(arxiv_id, decision, explanation):
            if not isinstance(arxiv_id, str) or not arxiv_id.strip():
                return {"success": False, "error": "arxiv_id must be a non-empty string"}
            if not isinstance(decision, str) or decision not in {"add", "reject"}:
                return {"success": False, "error": "decision must be exactly 'add' or 'reject'"}
            if not isinstance(explanation, str) or not explanation.strip():
                return {"success": False, "error": "explanation must be a non-empty string"}
            if pending_read["arxiv_id"] is None:
                return {"success": False, "error": "Read a paper before calling decide_paper"}
            if arxiv_id != pending_read["arxiv_id"]:
                return {
                    "success": False,
                    "error": (
                        "Decide only the currently read paper with arXiv ID "
                        f"{pending_read['arxiv_id']!r}"
                    ),
                }
            if decision == "add":
                return add_paper(arxiv_id, summary=explanation.strip())
            return reject_paper(arxiv_id, reason=explanation.strip())

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
                return {"decide_paper"}
            if papers:
                return {"submit_literature_review"}
            if candidate_batch["ids"]:
                return {"read_arxiv_paper"}
            return {"search_arxiv"}

        registry = ToolRegistry()
        registry.register(
            "search_arxiv",
            (
                "Search local arXiv papers for query. Return candidate IDs and search "
                "results, excluding papers already added or rejected in this review."
            ),
            _object_schema(
                {
                    "query": _string("Write a short English query for papers relevant to the research topic."),
                },
                ["query"],
            ),
            search_arxiv,
        ).register(
            "read_arxiv_paper",
            (
                "Read one unreviewed candidate from the latest search_arxiv result. "
                "Rejected papers cannot be read again in this review. Then "
                "call decide_paper once for the same arxiv_id."
            ),
            _object_schema(
                {"arxiv_id": _string("Use the exact candidate ID returned by the latest search_arxiv call.")},
                ["arxiv_id"],
            ),
            read_arxiv_paper,
        ).register(
            "decide_paper",
            (
                "Make exactly one decision for the paper just read. "
                "Choose 'add' for either direct evidence OR useful methodological "
                "or conceptual background. Background relevance is sufficient "
                "when you can identify a concrete contribution to the current "
                "study's hypothesis, feature or model choice, experimental design, "
                "evaluation, or interpretation. "
                "The paper need not answer the entire research question or use "
                "the same dataset, model, settings, or exact comparison. "
                "Missing the exact ablation or baseline comparison is not by "
                "itself a reason to reject. "
                "Shared keywords or a broad topic alone are insufficient. "
                "Choose 'reject' only when no substantive direct or background "
                "connection can be identified. "
                "Do not infer that the paper lacks a method or analysis merely "
                "because it is absent from a truncated excerpt."
            ),
            _object_schema(
                {
                    "arxiv_id": _string("Use the exact ID of the paper just read."),
                    "decision": {
                        "type": "string",
                        "enum": ["add", "reject"],
                        "description": "Choose exactly one: add the paper or reject it.",
                    },
                    "explanation": _string(
                        "For add, identify the concrete contribution to this study, "
                        "summarize the supporting findings, and state limitations. "
                        "For reject, explain why neither direct evidence nor useful "
                        "background applies; dataset or setting differences alone "
                        "are insufficient."
                    ),
                },
                ["arxiv_id", "decision", "explanation"],
            ),
            decide_paper,
        ).register(
            "submit_literature_review",
            (
                "Write the complete English literature review in review and submit it. "
                "Add at least one read paper first; this completes the literature-review stage."
            ),
            _object_schema(
                {"review": _string("Write the complete review, connecting the included papers to the research topic.")},
                ["review"],
            ),
            submit_literature_review,
        )
        result = self.runtime("literature review").run(
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
            "Write concise planning feedback in message and send it to PostdocAgent.",
            _object_schema(
                {"message": _string("Write feedback or corrections for the proposed experimental plan.")},
                ["message"],
            ),
            lambda message: {"success": True, "message": message},
        )
        return self.runtime("plan formulation").run_one(
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
                "Write evidence-based feedback in message for PostdocAgent. Set "
                "supplement_required to say whether one specific additional experiment "
                "is needed before final interpretation."
            ),
            _object_schema(
                {
                    "message": _string("Write feedback on the results; if a supplement is needed, state its goal and expected outputs."),
                    "supplement_required": {
                        "type": "boolean",
                        "description": (
                            "True: a specific supplement is required. False: finalize "
                            "interpretation using existing evidence without another experiment."
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
        return self.runtime("results interpretation").run_one(
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
                "Write a proposed plan or focused question in message for PhDStudentAgent. "
                "Request feedback before submitting the plan."
            ),
            _object_schema(
                {"message": _string("Write the proposed plan or planning question for PhDStudentAgent.")},
                ["message"],
            ),
            lambda message: {"success": True, "message": message},
        ).register(
            "submit_plan",
            (
                "Write the final experiment plan in plan and separate data-preparation "
                "needs in data_requirements. Submit after PhD feedback; leave concrete "
                "dataset selection to the engineers."
            ),
            _object_schema(
                {
                    "plan": _string("Write the complete English experiment plan for MLESolverAgent."),
                    "data_requirements": _string(
                        "Write abstract data requirements only; omit dataset IDs and "
                        "schema-dependent field names."
                    ),
                },
                ["plan", "data_requirements"],
            ),
            submit_plan,
        )
        return self.runtime("plan formulation").run_one(
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

        def available_tools():
            # 先完成审阅，再根据补实验决定或结束要求开放工具。
            if phd_review_required:
                return {"send_dialogue"}
            if force_submit:
                return {"submit_interpretation"}
            if supplement_required is True:
                return {"request_supplementary_experiment"}
            if supplement_required is None:
                return {"request_supplementary_experiment", "submit_interpretation"}
            return {"submit_interpretation"}

        request_schema = _object_schema(
            {
                "objective": _string("Write the single evidence gap this supplement should resolve."),
                "reason": _string("Write why this extra evidence is needed for the final interpretation."),
                "required_outputs": {
                    "type": "array",
                    "items": _string("One required metric, comparison, or diagnostic."),
                    "description": "List the exact results the supplement must produce.",
                },
                "constraints": {
                    "type": "array",
                    "items": _string("One limit on scope, resources, or allowed changes."),
                    "description": "List limits for the supplement; keep the original experiment unchanged.",
                },
            },
            ["objective", "reason", "required_outputs", "constraints"],
        )
        submit_schema = _object_schema(
            {"interpretation": _string("Write the complete final English interpretation, including remaining evidence limits.")},
            ["interpretation"],
        )
        submit_handler = lambda interpretation: {
            "success": True,
            "interpretation": interpretation,
        }
        if supplement_required is True: # PhD 要求补充。评审完成且未强制收尾时，只开放申请补充实验的工具。
            request_description = (
                "Write objective, reason, required_outputs, and constraints for the "
                "supplement required by PhD review. "
                "At most ONE supplementary experiment is allowed across the entire "
                "workflow for the original experiment. If one already exists, "
                "do not request another; report remaining gaps as limitations. "
                "This submits a request; the workflow runs it. "
                "Preserve the original experiment."
            )
        else:   # else 虽然同时覆盖 None 和 False，但在 False 时，这段工具描述不会实际提供给模型，因为申请工具被隐藏了。 它主要用于 None 时让 Postdoc 自行判断。
            request_description = (
                "Write objective, reason, required_outputs, and constraints for one "
                "necessary new supplement. "
                "At most ONE supplementary experiment is allowed across the entire "
                "workflow for the original experiment. If one already exists, "
                "do not request another; report remaining gaps as limitations. "
                "This submits a request; the workflow runs it. "
                "Preserve the original experiment."
            )

        registry = ToolRegistry()
        registry.register(
            "send_dialogue",
            (
                "Write a result interpretation or focused question in message for "
                "PhDStudentAgent. Get feedback before requesting a supplement or "
                "submitting the interpretation."
            ),
            _object_schema(
                {"message": _string("Write the interpretation or question that PhDStudentAgent should review.")},
                ["message"],
            ),
            lambda message: {"success": True, "message": message},
        ).register(
            "request_supplementary_experiment",
            request_description,
            request_schema,
            request_supplementary_experiment,
        ).register(
            "submit_interpretation",
            (
                "Write the final English interpretation in interpretation using the "
                "observed results and PhD feedback. Submit it to finish this stage; "
                "state unresolved gaps as limitations."
            ),
            submit_schema,
            submit_handler,
        )
        return self.runtime("results interpretation").run(
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
            terminal_tools=registry.names,
            tool_names_provider=available_tools,
        )


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

    def data_preparation_turn(self, topic, data_requirements, literature_review, sw_message,conversation_history="", workdir=None, feedback=None,inspected_dataset=None,selected_dataset=None,latest_execution_failure=None,failed_trial_code_failures=None,*,code_execution_timeout_seconds):
        search = HFDataSearch()
        if inspected_dataset:   # 若已经完成 inspection，inspection 的身份信息最可靠，直接从中重新构造选中数据集。
            selected_dataset = {
                key: inspected_dataset.get(key)
                for key in ("dataset_id", "revision", "config_name", "split")
            }
        elif isinstance(selected_dataset, dict):    # 表示还没 inspection，但外层工作流已经明确传来了选定数据集。这里复制一份，避免后面修改局部变量时影响外层原来的字典。
            selected_dataset = dict(selected_dataset)
        elif isinstance(feedback, dict) and isinstance(feedback.get("selected_dataset"), dict): # 若参数 selected_dataset 没有传入，但上一轮工具反馈里带着：就从反馈中把选择恢复出来。
            selected_dataset = dict(feedback["selected_dataset"])
        else:   # 表示当前还没有选定任何数据集。因此后面只应允许搜索，不能直接 inspection 或执行代码。
            selected_dataset = None
        latest_execution_failure = _authoritative_execution_failure(
            latest_execution_failure
        )   # 是把“上一轮失败信息”过滤成可靠形式。
        if failed_trial_code_failures is None:  # 用来保存已经执行失败过的数据准备代码哈希, 避免重复执行相同的失败代码。
            failed_trial_code_failures = {}

        def search_huggingface_datasets(query, max_results):
            if selected_dataset:
                return {
                    "success": False,
                    "error_type": "dataset_selection_locked",
                    "error": "The dataset is already selected; inspect or prepare that dataset",
                    "selected_dataset": selected_dataset,
                }
            results = search.retrieve_ds(query, N=max(1, min(int(max_results), 10)))
            selected = self._select_first_complete_dataset(results)
            return {
                "success": True,
                "results": results,
                "selected_dataset": selected,
                "next_action_hint": (
                    "A complete local candidate was selected. Call "
                    "inspect_selected_dataset for that candidate; do not search again."
                    if selected
                    else "No complete local candidate was returned. Search with a different query."
                ),
            }

        def inspect_selected_dataset():
            if not selected_dataset:
                return {
                    "success": False,
                    "error_type": "missing_dataset_selection",
                    "error": "Select a local Hugging Face dataset before inspection",
                }
            inspected = search.inspect_dataset(
                selected_dataset["dataset_id"],
                selected_dataset["revision"],
                selected_dataset["config_name"],
                selected_dataset["split"],
            )
            return {
                "success": True,
                **inspected,
                "selected_dataset": selected_dataset,
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
                    "output": "",
                    "traceback": "",
                    "exit_code": None,
                    "selection": {},
                    "artifacts": [],
                    "prepared_data_manifest": None,
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

        selection_pending = bool(selected_dataset) and not inspected_dataset    # 表示已经选择数据集，但还没有完成 inspection。
        selected_and_inspected = bool(selected_dataset) and bool(inspected_dataset)    # 表示已经选择数据集，并且已经完成 inspection。
        if selection_pending:
            next_action = "Inspect the automatically selected local candidate; do not search again."
        elif selected_and_inspected and latest_execution_failure:
            next_action = (
                "Repair the complete preparation program using the authoritative execution "
                "failure and repair_hint, then call write_and_run_dataset_code. "
                "Do not resubmit unchanged failed code."
            )
        elif selected_and_inspected:
            next_action = "Generate the complete preparation program and call write_and_run_dataset_code."
        else:
            next_action = "Search for a local dataset or discuss the preparation requirements."
        registry = ToolRegistry().register(
            "search_huggingface_datasets",
            (
                "Find local Hugging Face datasets for your query. The framework "
                "selects the first complete result for preparation."
            ),
            _object_schema({
                "query": _string("Short search terms describing the required data and task."),
                "max_results": {"type": "integer", "description": "Maximum number of dataset results to return, from 1 to 10."},
            }, ["query", "max_results"]),
            search_huggingface_datasets,
        ).register(
            "inspect_selected_dataset",
            "Read the selected dataset's columns, types, row counts, and sample types. "
            "No arguments; the dataset is already selected.",
            _object_schema({}, []),
            inspect_selected_dataset,
        ).register(
            "write_and_run_dataset_code",
            "Write the complete data-preparation program in code. The tool checks and "
            "runs it. Success finishes data preparation; failure returns an error to fix.",
            _object_schema({"code": _string(
                "Complete Python source with imports for the inspected dataset. "
                "Data preparation only; no model training or evaluation. No Markdown fences."
            )}, ["code"]),
            write_and_run_dataset_code,
        ).register(
            "send_dialogue",
            "Send SWEngineerAgent a dataset suggestion or a question about the data requirements.",
            _object_schema({"message": _string(
                "Your concrete question or suggestion for preparing the required data."
            )}, ["message"]),
            lambda message: {"success": True, "message": message},
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
                f"Automatically selected local dataset: "
                f"{json.dumps(selected_dataset, ensure_ascii=True, default=str)}\n"
                f"Next action: {next_action}\n"
                f"Conversation so far:\n{conversation_history or 'None'}\n"
                f"Execution environment: {json.dumps(execution_capabilities(), ensure_ascii=True)}"
            ),
            registry=registry,
            terminal_tools=registry.names,
            return_after_tool=True,
            tool_names_provider=(
                lambda: {"write_and_run_dataset_code"}
                if selected_and_inspected
                else ({"inspect_selected_dataset"} if selection_pending else {"search_huggingface_datasets", "send_dialogue"})
            ),
        )


class SWEngineerAgent(BaseResearchAgent):
    role = "software engineer"

    def data_preparation_turn(
        self, topic, data_requirements, workdir, literature_review="",
        conversation_history="", feedback=None, ml_result=None,
        inspected_dataset=None,
        latest_execution_failure=None,
        *, code_execution_timeout_seconds,
    ):
        registry = ToolRegistry().register(
            "send_dialogue",
            "Send MLEngineerAgent your data requirements analysis or a specific code-fix instruction.",
            _object_schema({"message": _string(
                "Your analysis and next requested action. For a failure, identify the failing code and the fix."
            )}, ["message"]),
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
                f"Execution environment: {json.dumps(execution_capabilities(), ensure_ascii=True)}"
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

    def solve(self, state, workdir):
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

        # 当模型提交了一份不应再次执行的实验代码时，生成拒绝结果，并把之前的报错信息返回给模型，帮助它继续修复。
        def rejected_experiment_code(fingerprint, action):
            """
                fingerprint：这次提交代码的哈希值，用来识别代码版本。
                action：这次操作的名字，例如 write_experiment_code 或 edit_experiment_code。
                current：外层函数保存的实验状态，包括当前代码、失败过的代码版本，以及对应的错误信息。
            """
            previously_failed = fingerprint in current["failed_hashes"] # 在曾经执行失败的集合中，previously_failed 就是 True，否则是 False。
            latest_failure = current.get("latest_execution_failure")    # # 取出最近一次实验执行的错误信息。如果不存在这个键，则得到 None。
            if previously_failed:  # 如果这份代码以前失败过，进入这个分支。
                original_failure = dict(current.get("failed_failures", {}).get(fingerprint) or {})  # 找到这份代码自己对应的历史失败诊断。original_failure 是整份错误详情，是一个字典。
                original_error = original_failure.get("error") or "the original failure"    # 取出这份代码当时的错误描述。失败记录里的具体错误描述
                result = {
                    "success": False,
                    "error_type": "unchanged_failed_experiment_code",
                    "error": (
                        "This exact experiment code previously failed and was not executed again."
                    ),
                    "repair_hint": (
                        "Do not submit this code hash again. Apply atomic edits or "
                        "write a materially different replacement that resolves the "
                        f"original failure: {original_error}"
                    ),
                    "original_error_type": original_failure.get("error_type"),
                    "original_error": original_failure.get("error"),
                    "original_failure_category": original_failure.get(
                        "failure_category"
                    ),
                    "original_repair_hint": original_failure.get("repair_hint"),
                    "code_hash": fingerprint,
                    "action": action,
                }   # 创建一个新的返回字典，同时把旧记录里的字段复制过去，换一个名字
                if original_failure:    #  # 如果找到了这份代码自己的历史失败诊断，把它加入结果中。
                    result["original_execution_diagnostic"] = original_failure
                if latest_failure:   # 如果当前状态中存在最近一次执行的失败诊断
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
            if not current["code"]:
                return {"success": False, "error": "Create experiment code first"}
            fingerprint = _code_hash(current["code"])
            if fingerprint in current["failed_hashes"]:
                return rejected_experiment_code(fingerprint, "execute_experiment_code")

            combined = state.dataset_code + "\n\n" + current["code"]    # 合并数据准备代码和当前实验代码
            execution = execute_code(combined,timeout=self.code_execution_timeout_seconds,workdir=workdir,return_details=True,expected_hf_dataset=state.dataset_selection or None,)
            return _process_execution_result(current, fingerprint, execution, "running experiments",)

        def write_experiment_code(code):
            syntax_error = python_syntax_error(code)   # 检查代码能否被 Python 正常解析。
            if syntax_error:
                return {"success": False,"error_type": "syntax_error","error": syntax_error,}

            fingerprint = _code_hash(code)  # 计算代码指纹，检查是否重复
            if fingerprint in {current.get("code_hash"),} or fingerprint in current["failed_hashes"]:   # 如果代码指纹与当前代码指纹相同，或者代码指纹在失败集合中，都拒绝替换。
                return rejected_experiment_code(fingerprint,"write_experiment_code")

            current.update(code=code,output="",success=None,code_hash=fingerprint,force_replace=False,) # 通过检查后，接受新代码
            return {"success": True,"message": "Experiment code replaced. Call execute_experiment_code to run the saved draft.","code_hash": fingerprint,}

        def edit_experiment_code(edits):
            if not current["code"]: # 如果目前还没有代码，就无法修改，直接返回失败。
                return {"success": False, "error": "Create experiment code first"}
            if not isinstance(edits, list) or not edits:    # edits 必须是列表。列表不能为空。列表不能为空。列表不能为空。edit：当前修改字典，例如 {"old_text": "...", "new_text": "..."}。
                return {"success": False,"error": "edits must contain at least one edit object",}
            if len(edits) > 6:  # 限制一次最多提交 6 处修改。
                return {"success": False,"error": "edits may contain at most 6 changes",}

            source = current["code"]    # 把当前代码取出来，保存为 source。这一批修改都以这份原代码为匹配依据。
            seen_old_text = set()    # 建立集合，记录本批次已经出现过哪些 old_text，用于检查重复。
            planned_edits = []  # 建立列表，保存已经检查通过的修改计划。每处修改的起始位置、结束位置和新内容。此时还没有真正替换代码。
            for index, edit in enumerate(edits, start=1):   # 逐个检查修改对象
                if not isinstance(edit, dict) or set(edit) != {"old_text", "new_text"}:
                    return {"success": False,"error": (f"edit {index} must contain only old_text and new_text"),}
                old_text = edit["old_text"]
                new_text = edit["new_text"] # 取出要查找的旧片段，以及准备替换进去的新片段。

                if not isinstance(old_text, str) or not isinstance(new_text, str):  # 要求两者都是字符串。
                    return {"success": False,"error": f"edit {index} old_text and new_text must be strings",}
                if not old_text:    # 旧片段不能为空，否则无法明确定位修改位置。
                    return {"success": False,"error": f"edit {index} old_text must not be empty",}
                if old_text in seen_old_text:   # 如果本批次已经处理过相同的旧片段，就拒绝重复修改。
                    return {"success": False,"error": "Each edit old_text must be unique within one batch",}
                seen_old_text.add(old_text)

                occurrences = source.count(old_text)    # 统计旧片段在原代码中的匹配次数。
                if occurrences != 1:     # 如果匹配次数不是 1，就拒绝修改。0 次：原代码中找不到，可能写错了内容或空格。多于 1 次：无法确定想修改哪一处。
                    return {"success": False,
                        "error": ( f"edit {index} old_text must match exactly once; matched {occurrences}"),}

                start = source.index(old_text)  # 找到旧片段在原代码中开始的位置。
                planned_edits.append((start, start + len(old_text), new_text))  # 把修改计划保存为三元组：(起始位置, 结束位置, 新内容)

            planned_edits.sort(key=lambda item: item[0])    # 按照每处修改的起始位置排序。
            for previous, following in zip(planned_edits, planned_edits[1:]):
                if following[0] < previous[1]:    # 如果后续修改的起始位置在前一个修改的结束位置之前，说明重叠，就拒绝修改。
                    return {"success": False,"error": "Batch edits overlap in the current experiment code",}

            edited = source
            for start, end, new_text in reversed(planned_edits):    # 按位置从后往前处理修改。
                edited = edited[:start] + new_text + edited[end:]   # 换掉原来的 edited[start:end]
            syntax_error = python_syntax_error(edited)  #@ 检查修改后的 Python 语法，例如括号、缩进是否有问题。
            if syntax_error:    # 发现语法错误就拒绝更新。语法检查通过，只能说明语法合法，不代表代码运行一定正确。
                return {
                    "success": False,
                    "error_type": "syntax_error",
                    "error": syntax_error,
                }

            fingerprint = _code_hash(edited)    # 为修改后的代码计算哈希，用来识别代码版本。
            if fingerprint in {current.get("code_hash"),} or fingerprint in current["failed_hashes"]:
                return rejected_experiment_code(fingerprint,"edit_experiment_code",)

            current.update(code=edited,output="",success=None,code_hash=fingerprint,)
            return {
                "success": True,
                "message": (
                    f"Applied {len(planned_edits)} experiment edits atomically; "
                    "Call execute_experiment_code to run the saved draft."
                ),
                "code_hash": fingerprint,
                "edit_count": len(planned_edits),
            }

        def submit_experiment():
            if not current["code"]:
                return {"success": False, "error": "Create experiment code first"}
            if current["success"] is not True:
                return {"success": False,"error": "Execute experiment code successfully before submitting",}
            return {"success": True,"code": current["code"],"output": current["output"],}

        def available_tools():
            if not current["code"]:
                return {"write_experiment_code"}
            if current["success"] is True:
                return {"submit_experiment"}
            if current["success"] is False:
                if current["last_failure_category"] == "execution_infrastructure_error":    # 当前 _process_execution_result() 遇到子进程异常退出、结果传输失败等被归为基础设施故障的情况，不会把代码加入失败集合，以便允许重试。
                    return {"execute_experiment_code", "edit_experiment_code", "write_experiment_code"}
                if current["force_replace"]:
                    return {"write_experiment_code"}
                return {"edit_experiment_code", "write_experiment_code"}
            return {"execute_experiment_code"}

        registry = ToolRegistry()
        registry.register(
            "write_experiment_code",
            (
                "Save the full experiment program from code, replacing the current draft. "
                "Call execute_experiment_code next, then submit_experiment after successful execution."
            ),
            _object_schema(
                {"code": _string(
                    "Complete experiment-only Python source with imports. Reuse prepared data variables; "
                    "do not repeat data-preparation code. No Markdown fences."
                )},
                ["code"],
            ),
            write_experiment_code,
        ).register(
            "edit_experiment_code",
            (
                "Provide 1 exact replacements in edits. The tool applies all edits "
                "or none. Call execute_experiment_code to run the updated experiment."
            ),
            _object_schema(
                {
                    "edits": {
                        "type": "array",
                        "description": "All changes to the current experiment code in one call. The matched fragments must not overlap.",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {
                            "type": "object",
                            "description": "One exact text replacement in the current experiment code.",
                            "properties": {
                                "old_text": _string(
                                    "Copy a non-empty fragment exactly from the current code, including indentation. It must occur once."
                                ),
                                "new_text": _string(
                                    "Python text to replace old_text; keep correct indentation. Use an empty string to delete it."
                                ),
                            },
                            "required": ["old_text", "new_text"],
                            "additionalProperties": False,
                        },
                    },
                },
                ["edits"],
            ),
            edit_experiment_code,
            batch_coalescer=_coalesce_experiment_edit_batches,
        ).register(
            "execute_experiment_code",
            (
                "Execute the saved experiment with the saved data-preparation code. "
                "Used for new drafts and retries after an execution environment failure. No arguments."
            ),
            _object_schema({}, []),
            execute_experiment_code,
        ).register(
            "submit_experiment",
            "Return the last successful code and output as the final experiment result. No arguments.",
            _object_schema({}, []),
            submit_experiment,
        )
        runtime = self.runtime("running experiments")
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
            f"Literature review: {state.literature_review}"
        )

        # 当实验执行失败时，调用 compact_experiment_failure 来生成压缩后的失败信息。减少发给模型的错误信息长度。
        # 一次工具调用失败后，如果手里有“当前实验代码”和“最近一次执行错误”，就重新组织下一轮发给模型的消息。
        def compact_experiment_failure(*, tool_call, arguments, value):
            # diagnostic：最近一次程序运行为什么失败。
            # value：刚才这次工具调用返回了什么。
            del arguments
            if not current.get("code") or tool_call.name not in {
                "execute_experiment_code", "write_experiment_code", "edit_experiment_code",
            }:
                return None
            diagnostic = current.get("latest_execution_failure")    # 获取最近一次真实执行失败的诊断。
            if not diagnostic:
                return None
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
            "failed_failures": {},
            "last_failure_category": None,
            "consecutive_failure_category_count": 0,
            "force_replace": False,
        }

        def rejected_supplement_code(fingerprint, action):
            original_failure = dict(
                current.get("failed_failures", {}).get(fingerprint) or {}
            )
            original_error = original_failure.get("error") or "the original failure"
            result = {
                "success": False,
                "error_type": "unchanged_failed_supplement_code",
                "error": (
                    "This exact supplement code previously failed and was not "
                    "executed again."
                ),
                "repair_hint": (
                    "Do not submit this code hash again. Make a substantive "
                    f"change that resolves the original failure: {original_error}"
                ),
                "original_error_type": original_failure.get("error_type"),
                "original_error": original_failure.get("error"),
                "original_failure_category": original_failure.get(
                    "failure_category"
                ),
                "original_repair_hint": original_failure.get("repair_hint"),
                "code_hash": fingerprint,
                "action": action,
            }
            if original_failure:
                result["original_execution_diagnostic"] = original_failure
            return result

        def execute_supplement_code():
            if not current["code"]:
                return {"success": False, "error": "Create supplement code first"}
            fingerprint = _code_hash(current["code"])
            if fingerprint in current["failed_hashes"]:
                return rejected_supplement_code(
                    fingerprint,
                    "execute_supplement_code",
                )
            combined = state.dataset_code + "\n\n" + current["code"]
            execution = execute_code(
                combined,
                timeout=self.code_execution_timeout_seconds,
                workdir=str(supplement_workdir),
                return_details=True,
                expected_hf_dataset=state.dataset_selection or None,
            )
            return _process_execution_result(
                current, fingerprint, execution, f"supplementary experiment {supplement_id}",
            )

        def write_supplement_code(code):
            syntax_error = python_syntax_error(code)
            if syntax_error:
                return {
                    "success": False,
                    "error_type": "syntax_error",
                    "error": syntax_error,
                }
            fingerprint = _code_hash(code)
            if fingerprint in current["failed_hashes"]:
                return rejected_supplement_code(
                    fingerprint,
                    "write_supplement_code",
                )
            if fingerprint == current.get("code_hash"):
                return {
                    "success": False,
                    "error_type": "unchanged_current_supplement_code",
                    "error": "Replacement supplement code is unchanged from the current draft.",
                    "repair_hint": "Make a substantive code change before retrying.",
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
                "message": "Supplement code replaced. Call execute_supplement_code to run the saved draft.",
                "code_hash": fingerprint,
            }

        def edit_supplement_code(old_text, new_text):
            if not current["code"]: # 如果目前还没有代码，就无法修改，直接返回失败。
                return {"success": False, "error": "Create supplement code first"}

            occurrences = current["code"].count(old_text)
            if occurrences != 1:
                return {
                    "success": False,
                    "error": f"old_text must match exactly once; matched {occurrences}",
                }
            edited = current["code"].replace(old_text, new_text, 1)
            syntax_error = python_syntax_error(edited)
            if syntax_error:
                return {
                    "success": False,
                    "error_type": "syntax_error",
                    "error": syntax_error,
                }
            fingerprint = _code_hash(edited)
            if fingerprint in current["failed_hashes"]:
                return rejected_supplement_code(
                    fingerprint,
                    "edit_supplement_code",
                )
            if fingerprint == current.get("code_hash"):
                return {
                    "success": False,
                    "error_type": "unchanged_current_supplement_code",
                    "error": "The edited supplement code is unchanged from the current draft.",
                    "repair_hint": "Make a substantive code change before retrying.",
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
                "message": "Supplement code edited. Call execute_supplement_code to run the saved draft.",
                "code_hash": fingerprint,
            }

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
                return {"write_supplement_code"}
            if current["success"] is True:
                return {"submit_supplementary_experiment"}
            if current["success"] is False:
                if current["last_failure_category"] == "execution_infrastructure_error":
                    return {"execute_supplement_code", "edit_supplement_code", "write_supplement_code"}
                if current["force_replace"]:
                    return {"write_supplement_code"}
                return {"edit_supplement_code", "write_supplement_code"}
            return {"execute_supplement_code"}

        registry = ToolRegistry()
        registry.register(
            "write_supplement_code",
            (
                "Save the full supplementary experiment from code, replacing the current draft. "
                "Call execute_supplement_code next, then submit_supplementary_experiment after successful execution."
            ),
            _object_schema(
                {"code": _string(
                    "Complete supplement-only Python source with imports. Reuse prepared data and "
                    "leave the original experiment unchanged. No Markdown fences."
                )},
                ["code"],
            ),
            write_supplement_code,
        ).register(
            "edit_supplement_code",
            (
                "Provide one exact text replacement for the saved supplement. "
                "The tool applies it. Call execute_supplement_code to run the updated supplement."
            ),
            _object_schema(
                {
                    "old_text": _string(
                        "Copy a non-empty fragment exactly from the saved supplement, including indentation. It must occur once."
                    ),
                    "new_text": _string(
                        "Python text to replace old_text; keep correct indentation. Use an empty string to delete it."
                    ),
                },
                ["old_text", "new_text"],
            ),
            edit_supplement_code,
        ).register(
            "execute_supplement_code",
            (
                "Execute the saved supplement with the saved data-preparation code. "
                "Used for new drafts and retries after an execution environment failure. No arguments."
            ),
            _object_schema({}, []),
            execute_supplement_code,
        ).register(
            "submit_supplementary_experiment",
            "Return the last successful supplement code and output with its request and ID. No arguments.",
            _object_schema({}, []),
            submit_supplementary_experiment,
        )
        runtime = self.runtime("supplementary experiments")
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
            return {
                "success": True,
                "title": title.strip(),
                "central_claim": central_claim.strip(),
                "section_plans": {
                    section: str(section_plans[section]).strip()
                    for section in sections
                },
            }

        blueprint_registry = ToolRegistry().register(
            "submit_paper_blueprint",
            (
                "Write a title in title, one evidence-based claim in central_claim, "
                "and a plan for every required section in section_plans. Submit this "
                "outline before writing sections."
            ),
            _object_schema(
                {
                    "title": {
                        "type": "string",
                        "description": "Write a concise paper title.",
                        "maxLength": 180,
                    },
                    "central_claim": {
                        "type": "string",
                        "description": "Write one main claim supported by the supplied evidence.",
                        "maxLength": 600,
                    },
                    "section_plans": {
                        "type": "object",
                        "description": "Map every required section name to its brief writing plan.",
                        "properties": {
                            section: {
                                "type": "string",
                                "description": f"Write a brief content plan for {section}.",
                                "maxLength": 500,
                            }
                            for section in sections
                        },
                        "required": list(sections),
                        "additionalProperties": False,
                    },
                },
                ["title", "central_claim", "section_plans"],
            ),
            submit_paper_blueprint,
        )
        blueprint = self.runtime(
            "report writing blueprint",
            max_tokens=3000,
        ).run_one(
            system_prompt=paper_blueprint_prompt(sections, available_figures),
            user_prompt=(
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
                body = str(latex_body).strip()  # 把正文转成字符串，去掉两端空白。
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
                (
                        f"Submit only the LaTeX body for {section}, with its exact "
                        "name in section_name. The framework adds section headings, "
                        "figures, and the bibliography automatically. "
                        "Do not insert images or figure environments. "
                        "If submission is rejected, correct the reported violations "
                        "before resubmitting."
                ),
                _object_schema(
                    {
                        "section_name": {
                            "type": "string",
                            "enum": [section],
                            "description": f"Use exactly {section!r}.",
                        },
                        "latex_body": {
                            "type": "string",
                            "description": (
                                "Write section body only. Subsections, equations, and tables "
                                "are allowed. Do not include a main section heading, document "
                                "wrapper, or bibliography. "
                                "Refer to available figures in prose, such as 'Figure 1'; "
                                "the framework inserts the images automatically. "
                                "Never include \\begin{figure}, \\end{figure}, or \\includegraphics. "
                                "If rejected for figure commands, remove the entire figure "
                                "environment, including its image, caption, and label, "
                                "while preserving the surrounding results and discussion. "
                                "Use only supplied citation keys."
                            ),
                            "maxLength": max_chars[section],
                        },
                    },
                    ["section_name", "latex_body"],
                ),
                submit_paper_section,
            )
            section_evidence = dict(authoritative_evidence)
            if section in {"Methods", "Experimental Setup"}:
                section_evidence["experiment_code"] = state.experiment_code
                section_evidence["supplementary_experiments"] = [
                    dict(item) for item in supplements
                ]
            section_result = self.runtime(
                f"report writing section {section}",
                max_tokens=4096,
            ).run_one(
                system_prompt=paper_section_prompt(
                    section,
                    word_targets[section],
                    available_figures,
                ),
                user_prompt=(
                    "Approved paper blueprint:\n"
                    + json.dumps(blueprint, ensure_ascii=True, default=str)
                    + "\nCurrent section plan:\n"
                    + blueprint["section_plans"][section]
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
            "Write the complete README in markdown and submit it for the workflow to save.",
            _object_schema(
                {"markdown": _string("Write the complete README in Markdown.")},
                ["markdown"],
            ),
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
