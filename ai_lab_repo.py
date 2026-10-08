import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from inference import llm_http_attempt_budget_scope, llm_sampling_scope

from local_replay import LocalReplaySession, local_replay_scope
from research_state import ResearchState
from seeding import SeedConfig, seed_scope
from run_logging import PromptRecorder, RunLogger, run_log_scope
from telemetry import TraceRecorder, trace_scope
from workflows.laboratory import LaboratoryWorkflow


load_dotenv()   # 用这个包load_dotenv()来load environment  from .env file

def load_paper_writing_config(config=None):
    """Validate batch paper settings, using the main batch YAML for defaults."""
    if config is None:
        path = Path(__file__).resolve().parent / "experiment_configs" / "BATCH_LOCAL_V2_CONCURRENT.yaml"
        with path.open(encoding="utf-8") as source:
            batch_config = yaml.safe_load(source)
        config = batch_config.get("paper_writing") if isinstance(batch_config, dict) else None
    if not isinstance(config, dict):
        raise ValueError("paper_writing must be a mapping")
    raw_sections = config.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("Paper writing sections must be a non-empty list")

    sections = []
    word_targets = {}
    max_chars = {}
    for section in raw_sections:
        if not isinstance(section, dict):
            raise ValueError("Each paper section must be a mapping")
        name = section.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Each paper section needs a non-empty name without surrounding whitespace")
        if name in word_targets:
            raise ValueError(f"Duplicate paper section: {name}")
        for key in ("word_target", "max_chars"):
            value = section.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Paper section {name!r} {key} must be a positive integer")
        sections.append(name)
        word_targets[name] = section["word_target"]
        max_chars[name] = section["max_chars"]

    generation_order = config.get("generation_order")
    if (
        not isinstance(generation_order, list)
        or not all(isinstance(name, str) for name in generation_order)
        or len(generation_order) != len(sections)
        or set(generation_order) != set(sections)
    ):
        raise ValueError("Paper generation_order must contain every section exactly once")
    return {
        "sections": [
            {"name": name, "word_target": word_targets[name], "max_chars": max_chars[name]}
            for name in sections
        ],
        "generation_order": list(generation_order),
    }


# 描述一条任务
@dataclass(frozen=True)
class QueryJob:
    query_id: str
    recording_id: str = "latest"


# 描述整批任务的公共配置
@dataclass(frozen=True)
class BatchConfig:
    material_root: str
    max_concurrent_queries: int
    queries: tuple
    output_root: str
    collaboration_rounds: dict
    code_execution_timeout_seconds: float
    console_log: bool = False
    compile_latex: bool = False
    max_llm_http_attempts_per_query: int = 100
    seeds: SeedConfig = field(default_factory=SeedConfig)
    llm_sampling: dict | None = None
    paper_writing: dict = field(default_factory=load_paper_writing_config)


def run_query_job(batch, job, model, api_key, output_root):
    # 创建本地回放会话，把一次 query 绑定到本地录制的数据与离线回放。
    replay = LocalReplaySession.from_config({
        "root": batch.material_root,
        "query-id": job.query_id,
        "recording-id": job.recording_id,       # 选择该 query 的哪次录制快照，默认latest最新
        "latency-mode": "recorded",  # 按录制时的工具耗时模拟延迟# 
        "latency-scale": 0.2,   # 延迟倍率；1.0 原样、0.5 一半、0 不等待
    })
    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
    workdir = Path(output_root).resolve() / job.query_id / job_id
    for relative in ("src", "tex", "traces"):
        (workdir / relative).mkdir(parents=True, exist_ok=True)

    run_logger = RunLogger(workdir,replay.query_id,job_id,console=batch.console_log,)  # 创建给人读的日志器
    prompt_recorder = PromptRecorder(workdir, replay.query_id, job_id)  # 保存LLM消息、工具schema、返回工具调用和工具结果
    recorder = TraceRecorder(str(workdir / "traces"))   # 记录工作流执行轨迹

    # 初始化全局研究状态
    state = ResearchState(
        query_id=replay.query_id,
        recording_id=replay.recording_id,
        topic=replay.user_query,
        seeds=batch.seeds.to_dict(),
    )   
    workflow = LaboratoryWorkflow(  
        state=state,
        model=model,
        api_key=api_key,
        workdir=workdir,
        collaboration_rounds=batch.collaboration_rounds,
        code_execution_timeout_seconds=batch.code_execution_timeout_seconds,
        compile_latex=batch.compile_latex,
        paper_writing=batch.paper_writing,
    )# 创建Workflow和所有Agent

    error = None
    job_summary = None
    try:
        with (
            run_log_scope(run_logger, prompt_recorder),
            seed_scope(batch.seeds),
            llm_sampling_scope(batch.llm_sampling),
            llm_http_attempt_budget_scope(batch.max_llm_http_attempts_per_query) as llm_http_budget,
        ):
            run_logger.info(f"JOB START job_id={job_id} recording_id={replay.recording_id} model={model}")
            run_logger.info(f"TOPIC {replay.user_query}")
            run_logger.info("SEEDS " + json.dumps(batch.seeds.to_dict(), ensure_ascii=True, sort_keys=True))
            run_logger.info("COLLABORATION ROUNDS " + json.dumps(batch.collaboration_rounds, sort_keys=True))
            run_logger.info(f"CODE EXECUTION TIMEOUT seconds={batch.code_execution_timeout_seconds}")
            run_logger.info(f"LLM HTTP ATTEMPT BUDGET " + f"limit={batch.max_llm_http_attempts_per_query}")

            try:
                with trace_scope(recorder=recorder,agent="LaboratoryWorkflow",phase="workflow",trigger="framework_automatic"):
                    with local_replay_scope(replay):    # 设置当前 query 的本地物料会话。
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
                    "llm_sampling": batch.llm_sampling,
                    "collaboration_rounds": batch.collaboration_rounds,
                    "paper_writing": batch.paper_writing,
                    "code_execution_timeout_seconds": batch.code_execution_timeout_seconds,
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

def parse_yaml(path):
    with open(path, "r", encoding="utf-8") as source:
        payload = yaml.safe_load(source) or {}
    material_root = payload["material-root"]
    output_root = payload["output-root"]
    concurrency = int(payload.get("max-concurrent-queries", 5))
    http_attempt_limit = int(payload.get("max-llm-http-attempts-per-query", 100))
    llm_sampling = payload["llm_sampling"]
    collaboration_rounds = payload["collaboration_rounds"]
    code_execution_timeout_seconds = payload["code_execution_timeout_seconds"]
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
        llm_sampling=llm_sampling,
        collaboration_rounds=collaboration_rounds,
        code_execution_timeout_seconds=code_execution_timeout_seconds,
        paper_writing=load_paper_writing_config(payload.get("paper_writing")),
    )

def load_runtime_environment():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    return api_key, model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Agent Laboratory local batch workflow")
    parser.add_argument(
        "--yaml-location",
        default="experiment_configs/BATCH_LOCAL_V2_CONCURRENT.yaml",
        help="Batch YAML configuration path",
    )
    return parser.parse_args()



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

if __name__ == "__main__":
    main()
