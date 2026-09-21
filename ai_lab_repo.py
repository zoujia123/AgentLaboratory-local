import argparse
import hashlib
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from inference import llm_http_attempt_budget_scope
from agents import (
    MLEngineerAgent,
    PhDStudentAgent,
    PostdocAgent,
    ProfessorAgent,
    SWEngineerAgent,
)
from local_replay import LocalReplaySession, local_replay_scope
from mlesolver import ExperimentRewardAgent, MLESolverAgent
from papersolver import PaperRewardAgent, PaperSolverAgent, ReviewerAgent
from research_state import ResearchState
from seeding import SeedConfig, seed_scope
from run_logging import (
    PromptRecorder,
    RunLogger,
    current_run_logger,
    phase_log,
    run_log_scope,
)
from telemetry import TraceRecorder, trace_scope


load_dotenv()

DEFAULT_CONFIG = "experiment_configs/BATCH_LOCAL_V2_CONCURRENT.yaml"
DEFAULT_OUTPUT_ROOT = "research_runs_v2"
QUERY_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
COLLABORATION_HISTORY_LIMIT = 6
COLLABORATION_DIAGNOSTIC_HEAD_CHARS = 1000
COLLABORATION_DIAGNOSTIC_TAIL_CHARS = 2000
COLLABORATION_CODE_PREVIEW_CHARS = 500


def _normalize_request_text(value):
    return " ".join(str(value or "").strip().lower().split())


def _supplement_request_fingerprint(request):
    request = dict(request or {})
    normalized = {
        "objective": _normalize_request_text(request.get("objective")),
        "reason": _normalize_request_text(request.get("reason")),
        "required_outputs": sorted(
            _normalize_request_text(item)
            for item in request.get("required_outputs", [])
        ),
        "constraints": sorted(
            _normalize_request_text(item)
            for item in request.get("constraints", [])
        ),
    }
    serialized = json.dumps(normalized, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _find_duplicate_supplement(supplements, request_fingerprint):
    for supplement in supplements:
        existing_fingerprint = supplement.get("request_fingerprint")
        if not existing_fingerprint:
            existing_fingerprint = _supplement_request_fingerprint(
                supplement.get("request", {})
            )
        if existing_fingerprint == request_fingerprint:
            return supplement
    return None


def _bounded_collaboration_text(
    value,
    head=COLLABORATION_DIAGNOSTIC_HEAD_CHARS,
    tail=COLLABORATION_DIAGNOSTIC_TAIL_CHARS,
):
    text = str(value or "")
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    suffix = text[-tail:] if tail else ""
    return text[:head] + f"\n...[omitted {omitted} chars]...\n" + suffix


def _collaboration_code_summary(code):
    if not isinstance(code, str) or not code:
        return None
    preview_chars = COLLABORATION_CODE_PREVIEW_CHARS
    if len(code) <= preview_chars:
        preview = code
    else:
        head = preview_chars // 2
        tail = preview_chars - head
        preview = (
            code[:head]
            + f"\n...[omitted {len(code) - preview_chars} code chars]...\n"
            + code[-tail:]
        )
    return {
        "code_chars": len(code),
        "code_lines": len(code.splitlines()),
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "code_preview": preview,
    }


def _compact_dataset_results(results):
    compacted = []
    for item in list(results or []):
        if not isinstance(item, dict):
            compacted.append(_bounded_collaboration_text(item, head=500, tail=0))
            continue
        compact = {}
        for key in (
            "id",
            "configs",
            "splits",
            "revision",
            "gated",
            "private",
            "train_element_size",
            "test_element_size",
        ):
            if item.get(key) is not None:
                compact[key] = item[key]
        if item.get("description"):
            compact["description"] = _bounded_collaboration_text(
                item["description"],
                head=500,
                tail=0,
            )
        compacted.append(compact)
    return compacted


def _compact_collaboration_result(sender, tool_name, value, round_index):
    value = value if isinstance(value, dict) else {"value": value}
    compact = {
        "round": round_index,
        "sender": sender,
        "tool": tool_name,
        "success": value.get("success"),
    }
    for key in (
        "error_type",
        "exit_code",
        "selection",
        "prepared_data_manifest",
        "validation_warnings",
    ):
        if value.get(key) is not None:
            compact[key] = value[key]

    diagnostic_values = set()
    if value.get("error"):
        error_text = str(value["error"])
        compact["error"] = _bounded_collaboration_text(error_text)
        diagnostic_values.add(error_text)

    code_summary = _collaboration_code_summary(value.get("code"))
    if code_summary is not None:
        compact.update(code_summary)

    for key in ("traceback", "output"):
        if value.get(key):
            diagnostic = str(value[key])
            if diagnostic not in diagnostic_values:
                compact[key] = _bounded_collaboration_text(diagnostic)
                diagnostic_values.add(diagnostic)

    if tool_name == "send_dialogue" and value.get("message") is not None:
        compact["message"] = _bounded_collaboration_text(
            value["message"],
            head=1500,
            tail=500,
        )
    if tool_name == "search_huggingface":
        compact["results"] = _compact_dataset_results(value.get("results"))
    return compact


def _render_collaboration_history(
    conversation,
    omit_latest_senders=(),
    limit=COLLABORATION_HISTORY_LIMIT,
    summarize_tool_history=False,
):
    omitted_indexes = set()
    for sender in omit_latest_senders:
        for index in range(len(conversation) - 1, -1, -1):
            if (
                index not in omitted_indexes
                and conversation[index].get("sender") == sender
            ):
                omitted_indexes.add(index)
                break
    visible = [
        item
        for index, item in enumerate(conversation)
        if index not in omitted_indexes
    ]
    limit = max(0, int(limit))
    visible = visible[-limit:] if limit else []
    if summarize_tool_history:
        summarized = []
        for item in visible:
            tool = item.get("tool")
            if tool == "search_huggingface" and item.get("success") is True:
                summary = {
                    key: item[key]
                    for key in ("round", "sender", "tool", "success")
                    if key in item
                }
                summary["message"] = (
                    "Dataset search completed; use the authoritative inspected "
                    "dataset instead of the earlier candidate list."
                )
                summarized.append(summary)
            elif (
                tool in {"execute_dataset_code", "submit_dataset_code"}
                and item.get("success") is False
            ):
                summary = {
                    key: item[key]
                    for key in (
                        "round", "sender", "tool", "success", "error_type",
                        "code_sha256", "selection", "validation_warnings",
                    )
                    if key in item
                }
                summary["message"] = (
                    "An earlier candidate failed; the latest candidate executed "
                    "successfully. This does not establish data validity."
                )
                if item.get("error"):
                    summary["error"] = _bounded_collaboration_text(
                        item["error"], head=200, tail=100,
                    )
                summarized.append(summary)
            else:
                summarized.append(item)
        visible = summarized
    return "\n".join(
        json.dumps(item, ensure_ascii=True, sort_keys=True, default=str)
        for item in visible
    )


def _render_sw_collaboration_history(conversation, ml_feedback, inspected_dataset):
    trial = ml_feedback or {}
    inspected = inspected_dataset or {}
    selection = trial.get("selection") or {}
    matches = (
        trial.get("tool") == "execute_dataset_code"
        and trial.get("success") is True
        and isinstance(trial.get("code"), str)
        and bool(trial["code"].strip())
        and bool(inspected.get("dataset_schema"))
        and all(
            inspected.get(key) is not None
            and selection.get(key) == inspected[key]
            for key in ("dataset_id", "revision", "split")
        )
    )
    if matches:
        actual_config = selection.get("config_name")
        inspected_config = inspected.get("config_name")
        # An omitted config is unambiguous only when exactly one is materialized.
        matches = inspected_config is not None and (
            actual_config == inspected_config
            or (
                actual_config is None
                and inspected.get("available_configs") == [inspected_config]
            )
        )
    return _render_collaboration_history(
        conversation,
        omit_latest_senders={"SWEngineerAgent", "MLEngineerAgent"},
        summarize_tool_history=matches,
    )


@dataclass(frozen=True)
class QueryJob:
    query_id: str
    recording_id: str = "latest"


@dataclass(frozen=True)
class BatchConfig:
    material_root: str
    max_concurrent_queries: int
    queries: tuple
    output_root: str = DEFAULT_OUTPUT_ROOT
    console_log: bool = False
    compile_latex: bool = False
    max_llm_http_attempts_per_query: int = 100
    seeds: SeedConfig = field(default_factory=SeedConfig)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Agent Laboratory local batch workflow")
    parser.add_argument(
        "--yaml-location",
        default=DEFAULT_CONFIG,
        help="Batch YAML configuration path",
    )
    return parser.parse_args()


def parse_yaml(path):
    with open(path, "r", encoding="utf-8") as source:
        payload = yaml.safe_load(source) or {}
    material_root = payload.get("material-root")
    output_root = payload.get("output-root", DEFAULT_OUTPUT_ROOT)
    concurrency = int(payload.get("max-concurrent-queries", 5))
    http_attempt_limit = int(payload.get("max-llm-http-attempts-per-query", 100))
    raw_queries = payload.get("queries")

    jobs = []
    for raw in raw_queries:
        query_id = raw.get("query-id")
        recording_id = raw.get("recording-id", "latest")
        jobs.append(QueryJob(query_id=query_id, recording_id=recording_id))
        
    return BatchConfig(
        material_root=str(Path(material_root).resolve()),
        output_root=str(Path(output_root).resolve()),
        max_concurrent_queries=concurrency,
        queries=tuple(jobs),
        console_log=bool(payload.get("console-log", False)),
        compile_latex=bool(payload.get("compile-latex", False)),
        max_llm_http_attempts_per_query=http_attempt_limit,
        seeds=SeedConfig.from_mapping(payload.get("seeds")),
    )


def load_runtime_environment():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    return api_key, model


class LaboratoryWorkflow:
    def __init__(self, state, model, api_key, workdir, compile_latex=False):
        self.state = state
        self.model = model
        self.api_key = api_key
        self.workdir = Path(workdir).resolve()
        self.compile_latex = bool(compile_latex)
        self.phd = PhDStudentAgent(model, api_key)
        self.postdoc = PostdocAgent(model, api_key)
        self.ml_engineer = MLEngineerAgent(model, api_key)
        self.sw_engineer = SWEngineerAgent(model, api_key)
        self.ml_solver = MLESolverAgent(model, api_key)
        self.experiment_reward = ExperimentRewardAgent(model, api_key)
        self.paper_solver = PaperSolverAgent(model, api_key)
        self.professor = ProfessorAgent(model, api_key)
        self.reviewers = [
            ReviewerAgent(model,api_key,1,"Expect sound experiments and insights aligned with the research topic."),
            ReviewerAgent(model,api_key,2,"Focus on technical quality, evidence, and potential impact."),
            ReviewerAgent(model,api_key,3,"Focus on originality, clarity, limitations, and reproducibility."),
        ]
        self.paper_reward = PaperRewardAgent(model, api_key)

    def _save_text(self, relative_path, content):
        path = self.workdir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")

    def save_state(self):
        path = self.workdir / "state.json"
        path.write_text(
            json.dumps(self.state.to_dict(), ensure_ascii=True, indent=2, default=str),
            encoding="utf-8",
        )

    def _formulate_plan(self):
        dialogue = ""
        for _ in range(8):
            postdoc_result = self.postdoc.plan_turn(self.state.topic,self.state.literature_review,dialogue,)
            if postdoc_result.tool_name == "submit_plan":
                return postdoc_result.value["plan"]
            dialogue = self.phd.plan_feedback(
                self.state.topic,
                self.state.literature_review,
                postdoc_result.value["message"],
            )
        raise RuntimeError("Plan formulation exceeded its dialogue budget")

    def _interpret_results(self):
        latest_phd_feedback = ""    # 保存PhD最近一次对实验结果的反馈
        phd_review_required = True  # 表示当前证据是否必须先经过PhD审核。初始实验结果还没审核，所以是True
        workflow_notice = ""    # Workflow给Postdoc的临时通知，例如“你刚才请求的补充实验已经做过了”。
        history = self.state.interpretation_history
        for round_index in range(8):
            current_run_logger().info(
                f"[results interpretation][round={round_index}] CALL PostdocAgent"
            )
            postdoc_result = self.postdoc.interpretation_turn(
                self.state.topic,
                self.state.plan,
                self.state.experiment_code,
                self.state.experiment_output,
                latest_phd_feedback,
                literature_review=self.state.literature_review,
                supplementary_experiments=self.state.supplementary_experiments,
                interpretation_history=history,
                phd_review_required=phd_review_required,
                workflow_notice=workflow_notice,
            )
            workflow_notice = ""
            if phd_review_required and postdoc_result.tool_name != "send_dialogue": # 强制PhD审核
                raise RuntimeError(
                    "PostdocAgent must obtain PhD review for the latest evidence"
                )
            if postdoc_result.tool_name == "submit_interpretation":
                interpretation = postdoc_result.value["interpretation"]
                history.append({
                    "round": round_index,
                    "sender": "PostdocAgent",
                    "type": "final_interpretation",
                    "interpretation": interpretation,
                })
                self.save_state()
                return interpretation
            if postdoc_result.tool_name == "send_dialogue":
                postdoc_message = postdoc_result.value["message"]
                history.append({
                    "round": round_index,
                    "sender": "PostdocAgent",
                    "type": "interpretation_question",
                    "message": postdoc_message,
                })
                self.save_state()
                current_run_logger().info(
                    f"[results interpretation][round={round_index}] "
                    "HANDOFF PostdocAgent -> PhDStudentAgent"
                )
                latest_phd_feedback = self.phd.interpretation_feedback(
                    self.state.topic,
                    self.state.plan,
                    self.state.experiment_output,
                    postdoc_message,
                    experiment_code=self.state.experiment_code,
                    literature_review=self.state.literature_review,
                    supplementary_experiments=self.state.supplementary_experiments,
                    interpretation_history=history,
                )
                history.append({
                    "round": round_index,
                    "sender": "PhDStudentAgent",
                    "type": "phd_feedback",
                    "message": latest_phd_feedback,
                })
                self.save_state()
                phd_review_required = False
                continue
            if postdoc_result.tool_name == "request_supplementary_experiment":
                supplement_request = postdoc_result.value["request"]
                request_fingerprint = _supplement_request_fingerprint(supplement_request)   # 计算request_fingerprint，用于判断相同补充实验是否已经做过
                duplicate = _find_duplicate_supplement(
                    self.state.supplementary_experiments,
                    request_fingerprint,
                )
                if duplicate is not None:
                    duplicate_id = duplicate.get("supplement_id", "existing supplement")
                    workflow_notice = (
                        "Duplicate supplementary request rejected. Reuse the results "
                        f"already available in {duplicate_id}, or request a materially "
                        "different experiment."
                    )
                    history.append({
                        "round": round_index,
                        "sender": "LaboratoryWorkflow",
                        "type": "duplicate_supplement_rejected",
                        "supplement_id": duplicate_id,
                        "request_fingerprint": request_fingerprint,
                        "request": supplement_request,
                    })
                    self.save_state()
                    continue
                history.append({
                    "round": round_index,
                    "sender": "PostdocAgent",
                    "type": "supplement_request",
                    "request_fingerprint": request_fingerprint,
                    "request": supplement_request,
                })
                self.save_state()
                supplement_index = len(self.state.supplementary_experiments) + 1
                supplement_id = f"supplement_{supplement_index:02d}"
                current_run_logger().info(
                    f"[results interpretation][round={round_index}] "
                    f"HANDOFF PostdocAgent -> MLESolverAgent supplement={supplement_id}"
                )
                with phase_log(f"supplementary experiment {supplement_id}"):
                    supplement = self.ml_solver.supplement(
                        self.state,
                        supplement_request,
                        self.workdir,
                        supplement_index,
                    )
                supplement["request_fingerprint"] = request_fingerprint
                self.state.supplementary_experiments.append(supplement)
                self._save_text(f"src/{supplement_id}.py",supplement["code"],)
                self._save_text(f"src/{supplement_id}_output.log",supplement["output"])
                history.append({
                    "round": round_index,
                    "sender": "MLESolverAgent",
                    "type": "supplement_result",
                    "supplement_id": supplement["supplement_id"],
                    "request_fingerprint": request_fingerprint,
                    "request": supplement_request,
                })
                self.save_state()
                latest_phd_feedback = ""
                phd_review_required = True
                continue
            raise RuntimeError(
                "PostdocAgent must send dialogue, request a supplementary "
                "experiment, or submit the interpretation"
            )
        raise RuntimeError("Results interpretation exceeded its dialogue budget")

    def _prepare_dataset_collaboratively(self):
        logger = current_run_logger()
        conversation = []
        sw_feedback = None
        ml_feedback = None
        inspected_dataset = None
        trial_pending = False
        for round_index in range(8):
            logger.info(f"[data preparation collaboration][round={round_index}] CALL SWEngineerAgent")
            sw_result = self.sw_engineer.data_preparation_turn(
                self.state.topic, self.state.plan, self.workdir,
                literature_review=self.state.literature_review,
                conversation_history=_render_sw_collaboration_history(
                    conversation,
                    ml_feedback,
                    inspected_dataset,
                ),
                feedback=sw_feedback,
                ml_result=ml_feedback,
                inspected_dataset=inspected_dataset,
            )
            sw_feedback = sw_result.value
            if (
                sw_result.tool_name == "submit_dataset_code"
                and not inspected_dataset
            ):
                sw_feedback = {
                    **sw_feedback,
                    "success": False,
                    "error_type": "schema_not_inspected",
                    "error": (
                        "Inspect the selected Hugging Face dataset schema before "
                        "submitting data-preparation code"
                    ),
                }
            if sw_result.tool_name == "submit_dataset_code":
                if sw_feedback.get("success"):
                    selection = dict(sw_feedback.get("selection") or {})
                    mismatched_fields = [
                        name
                        for name in ("dataset_id", "revision", "config_name", "split")
                        if selection.get(name) is not None
                        and inspected_dataset.get(name) is not None
                        and selection[name] != inspected_dataset[name]
                    ]
                    if mismatched_fields:
                        sw_feedback = {
                            **sw_feedback,
                            "success": False,
                            "error_type": "inspected_dataset_mismatch",
                            "error": (
                                "Submitted code loaded a different dataset than the "
                                "inspected selection for: "
                                + ", ".join(mismatched_fields)
                            ),
                        }
                    else:
                        for name in (
                            "dataset_id",
                            "revision",
                            "config_name",
                            "split",
                            "available_configs",
                            "available_splits",
                            "dataset_schema",
                            "materialized_resources",
                            "materialized_schemas",
                        ):
                            if inspected_dataset.get(name) is not None:
                                selection[name] = inspected_dataset[name]
                        sw_feedback = {**sw_feedback, "selection": selection}
                conversation.append(
                    _compact_collaboration_result(
                        "SWEngineerAgent",
                        sw_result.tool_name,
                        sw_feedback,
                        round_index,
                    )
                )
                if sw_feedback.get("success"):
                    logger.info("[data preparation collaboration] SWEngineerAgent SUBMITTED DATASET CODE")
                    return sw_feedback.get("selection") or {}, sw_feedback
                sw_message = "Final submission failed. Help repair this code:\n" + json.dumps(
                    sw_feedback, ensure_ascii=True, default=str
                )
            elif sw_result.tool_name == "send_dialogue":
                conversation.append(
                    _compact_collaboration_result(
                        "SWEngineerAgent",
                        sw_result.tool_name,
                        sw_feedback,
                        round_index,
                    )
                )
                sw_message = sw_feedback["message"]
            else:
                raise RuntimeError("SWEngineerAgent must send dialogue or submit dataset code")

            logger.info(f"[data preparation collaboration][round={round_index}] HANDOFF SWEngineerAgent -> MLEngineerAgent")
            ml_result = self.ml_engineer.data_preparation_turn(
                self.state.topic, self.state.plan, self.state.literature_review,
                sw_message,
                conversation_history=_render_collaboration_history(
                    conversation,
                    omit_latest_senders={"SWEngineerAgent", "MLEngineerAgent"},
                ),
                workdir=self.workdir,
                feedback=ml_feedback,
                inspected_dataset=inspected_dataset,
                trial_pending=trial_pending,
            )
            if (
                ml_result.tool_name == "inspect_huggingface_dataset"
                and isinstance(ml_result.value, dict)
                and ml_result.value.get("success") is True
            ):
                inspected_dataset = dict(ml_result.value)
                trial_pending = True
            elif ml_result.tool_name == "execute_dataset_code":
                # Runtime returns after one tool, including an execution failure.
                # Empty-code rejection is not a trial; all actual attempts unlock discussion.
                if isinstance(ml_result.value.get("code"), str) and ml_result.value["code"].strip():
                    trial_pending = False
            ml_feedback = {"tool": ml_result.tool_name, **ml_result.value}
            conversation.append(
                _compact_collaboration_result(
                    "MLEngineerAgent",
                    ml_result.tool_name,
                    ml_result.value,
                    round_index,
                )
            )
            logger.info(f"[data preparation collaboration][round={round_index}] HANDOFF MLEngineerAgent -> SWEngineerAgent")
        raise RuntimeError("Data preparation exceeded its collaboration budget")

    def perform_research(self):
        with phase_log("literature review"):
            literature = self.phd.conduct_literature_review(self.state.topic,required_papers=1,)
        self.state.literature_review = literature["review"]
        self.state.literature_papers = literature["papers"]
        self.save_state()

        with phase_log("plan formulation"):
            self.state.plan = self._formulate_plan()
        self.save_state()

        with phase_log("data preparation collaboration"):
            self.state.dataset_selection, dataset = (
                self._prepare_dataset_collaboratively()
            )
        self.state.dataset_code = dataset["code"]   # 保存数据准备代码及其执行输出
        self.state.dataset_output = dataset["output"]   
        self.state.dataset_manifest = dataset.get("prepared_data_manifest") or {}   # 保存数据产物清单，告诉后续实验阶段哪些变量是特征、目标和分析数据
        self.state.dataset_validation_warnings = list(  # 保存不阻止运行、但需要关注的数据验证警告。
            dataset.get("validation_warnings") or []
        )
        self._save_text("src/load_data.py", self.state.dataset_code)    # 把最终数据加载代码写入 src/load_data.py
        self.save_state()

        with phase_log("running experiments"):
            experiment = self.ml_solver.solve(self.state, self.workdir)
        self.state.experiment_code = experiment["code"]
        self.state.experiment_output = experiment["output"]
        self._save_text("src/run_experiments.py", self.state.experiment_code)
        self._save_text("src/experiment_output.log", self.state.experiment_output)
        self.save_state()

        with phase_log("results interpretation"):
            self.state.interpretation = self._interpret_results()
        with phase_log("experiment scoring"):
            self.state.experiment_score = self.experiment_reward.score(self.state)
        self.save_state()

        with phase_log("report writing"):
            paper = self.paper_solver.write(
                self.state,
                self.workdir,
                compile_pdf=self.compile_latex,
            )
        self.state.paper = paper["latex"]
        self._save_text("report.tex", self.state.paper)
        with phase_log("readme writing"):
            self.state.readme = self.professor.generate_readme(self.state)
        self._save_text("readme.md", self.state.readme)
        self.save_state()

        self.state.reviews = []
        for index, reviewer in enumerate(self.reviewers, start=1):
            with phase_log(f"paper review reviewer={index}"):
                self.state.reviews.append(reviewer.review(self.state))
        with phase_log("paper scoring"):
            self.state.paper_score = self.paper_reward.score(self.state)
        self._save_text(
            "reviews.json",
            json.dumps(self.state.reviews, ensure_ascii=True, indent=2),
        )
        self.save_state()
        return self.state


def _new_job_id():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def run_query_job(batch, job, model, api_key, output_root=DEFAULT_OUTPUT_ROOT):
    replay = LocalReplaySession.from_config({
        "enabled": True,
        "root": batch.material_root,
        "query-id": job.query_id,
        "recording-id": job.recording_id,
        "arxiv-search-mode": "local-bm25",
        "strict-offline": True,
        "latency-mode": "recorded",
        "latency-scale": 1.0,
    })
    job_id = _new_job_id()
    workdir = Path(output_root).resolve() / job.query_id / job_id
    for relative in ("src", "tex", "traces"):
        (workdir / relative).mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(workdir,replay.query_id,job_id,console=batch.console_log,)  # 创建给人读的日志器
    prompt_recorder = PromptRecorder(workdir, replay.query_id, job_id)  # 保存LLM消息、工具schema、返回工具调用和工具结果
    recorder = TraceRecorder(str(workdir / "traces"))   # 记录工作流执行轨迹
    state = ResearchState(
        query_id=replay.query_id,
        recording_id=replay.recording_id,
        topic=replay.user_query,
        seeds=batch.seeds.to_dict(),
    )   # 初始化全局研究状态
    workflow = LaboratoryWorkflow(  
        state=state,
        model=model,
        api_key=api_key,
        workdir=workdir,
        compile_latex=batch.compile_latex,
    )# 创建Workflow和所有Agent
    error = None
    job_summary = None
    try:
        with run_log_scope(run_logger, prompt_recorder), seed_scope(batch.seeds), llm_http_attempt_budget_scope(batch.max_llm_http_attempts_per_query) as llm_http_budget:
            run_logger.info(f"JOB START job_id={job_id} recording_id={replay.recording_id} model={model}")
            run_logger.info(f"TOPIC {replay.user_query}")
            run_logger.info("SEEDS " + json.dumps(batch.seeds.to_dict(), ensure_ascii=True, sort_keys=True))
            run_logger.info(f"LLM HTTP ATTEMPT BUDGET " + f"limit={batch.max_llm_http_attempts_per_query}")

            try:
                with trace_scope(recorder=recorder,agent="LaboratoryWorkflow",phase="workflow",trigger="framework_automatic"):
                    with local_replay_scope(replay):
                        workflow.perform_research()
            except BaseException as exc:
                error = exc
                run_logger.exception(f"JOB FAILED error={exc}")
                raise
            finally:
                workflow.save_state()
                replay_summary = replay.write_trace(recorder.run_dir)
                trace_summary = recorder.finalize()
                job_summary = {
                    "job_id": job_id,
                    "query_id": replay.query_id,
                    "recording_id": replay.recording_id,
                    "topic": replay.user_query,
                    "seeds": batch.seeds.to_dict(),
                    "console_log": batch.console_log,
                    "success": error is None,
                    "error": str(error) if error else None,
                    "workdir": str(workdir),
                    "workflow_log": str(run_logger.path),
                    "prompt_log": str(prompt_recorder.path),
                    "llm_http_attempt_budget": llm_http_budget.snapshot(),
                    "trace": trace_summary,
                    "replay": replay_summary,
                }
                (workdir / "job_summary.json").write_text(json.dumps(job_summary,ensure_ascii=True,indent=2,default=str,),encoding="utf-8",)
                run_logger.info(f"JOB END success={error is None} workflow_log={run_logger.path} prompts={prompt_recorder.path}")
    finally:
        run_logger.close()
    return job_summary


# 批量任务调度器
def run_batch(config, model, api_key):
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []    # 用于保存成功任务返回的结果
    failures = []    # 用于保存失败任务的信息
    with ThreadPoolExecutor(max_workers=config.max_concurrent_queries) as executor:
        futures = {
            executor.submit(run_query_job, config, job, model, api_key, config.output_root,): job for job in config.queries
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"query_id": job.query_id, "error": str(exc)})
    batch_summary = {
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    summary_path = output_root / "latest_batch_summary.json"
    summary_path.write_text(
        json.dumps(batch_summary, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )
    return batch_summary


def main():
    arguments = parse_arguments()
    config = parse_yaml(arguments.yaml_location)
    api_key, model = load_runtime_environment()
    summary = run_batch(config, model, api_key)
    if config.console_log:
        print(
            f"Completed {summary['success_count']} query job(s); "
            f"failures={summary['failure_count']}"
        )
    if summary["failure_count"]:
        summary_path = Path(config.output_root) / "latest_batch_summary.json"
        raise SystemExit(
            f"{summary['failure_count']} query job(s) failed; see {summary_path}"
        )


if __name__ == "__main__":
    main()
