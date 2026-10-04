import ast
import importlib
import importlib.util
import io
import json
import multiprocessing
import os
import queue
import re
import socket
import sys
import time
import traceback
import types
from pathlib import Path

import matplotlib

from local_replay import current_local_replay, install_worker_hf_replay
from seeding import (
    SeedConfig,
    current_seed_config,
    execution_seed_globals,
    validate_seed_usage,
)
from telemetry import traced_call


class OfflineExecutionError(RuntimeError):
    pass


def execution_capabilities():
    modules = (
        "datasets",
        "numpy",
        "pandas",
        "PIL",
        "sklearn",
        "torch",
        "torchvision",
        "transformers",
    )   # 要检查的模块名单
    available = []
    unavailable = []
    # find_spec() 检查当前执行环境是否包含指定的模块
    for module in modules:
        target = (
            available
            if importlib.util.find_spec(module) is not None
            else unavailable
        )
        target.append(module)
    # 真正 import datasets。因为后面需要读取版本号。
    datasets_module = (
        importlib.import_module("datasets")
        if "datasets" in available
        else None
    )
    return {
        "available_modules": available,
        "unavailable_modules": unavailable,
        "datasets_version": getattr(datasets_module, "__version__", None),
        "required_dataset_loader": "datasets.load_dataset",         # 必须用 datasets.load_dataset
        "pretrained_model_assets": [],  
        "pretrained_model_loading": False,  # 没有预训练权重
        "network": False,   # 禁止网络和安装包。
        "package_installation": False,
    }


class HFDataSearch:
    @traced_call("tool", "huggingface.index.initialize")
    def __init__(self):
        self.replay = current_local_replay(required=True)

    @traced_call(
        "tool",
        "huggingface.dataset.search",
        metadata_factory=lambda args, kwargs: {
            "top_n": kwargs.get("N", args[2] if len(args) > 2 else 10)
        },
    )
    def retrieve_ds(self, query, N=10):
        return self.replay.replay_hf_search(query, N)

    @traced_call("tool", "huggingface.dataset.inspect")
    def inspect_dataset(self, dataset_id, revision, config_name, split):
        return self.replay.select_hf_dataset(
            dataset_id,
            config_name,
            split,
            revision=revision,
        )

    @staticmethod
    def results_str(results):
        rendered = []
        for result in results:
            rendered.append(
                "\n".join([
                    f"Dataset ID: {result.get('id', '')}",
                    f"Description: {result.get('description', '')}",
                    f"Configs: {result.get('configs', [])}",
                    f"Splits: {result.get('splits', [])}",
                    f"Likes: {result.get('likes', 0)}",
                    f"Downloads: {result.get('downloads', 0)}",
                ])
            )
        return rendered


class ArxivSearch:
    def __init__(self):
        self.replay = current_local_replay(required=True)   # 当前 query 的本地回放会话对象，一个 LocalReplaySession 对象
        self._full_text_cache = {}  # 本轮文献综述期间临时保存“已读取论文全文”的内存字典。

    @staticmethod
    def _paper_cache_keys(query):
        exact = str(query).strip().casefold()
        base = re.sub(r"v\d+$", "", exact, flags=re.IGNORECASE)
        return exact, base

    def get_cached_full_paper_text(self, query):
        exact, base = self._paper_cache_keys(query)
        return self._full_text_cache.get(exact) or self._full_text_cache.get(base)

    @staticmethod
    def _process_query(query):
        words = str(query).split()
        output = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > 300:
                break
            output.append(word)
            current_length += len(word) + 1
        return " ".join(output)

    @traced_call(
        "tool",
        "arxiv.search",
        metadata_factory=lambda args, kwargs: {
            "max_results": kwargs.get("N", args[2] if len(args) > 2 else 20)
        },
        success_predicate=lambda result: result is not None,
    )
    def find_papers_by_str(self, query, N=20):
        return self.replay.replay_arxiv_search(
            query,
            N,
            self._process_query(query),
        )

    @traced_call(
        "tool",
        "arxiv.search",
        metadata_factory=lambda args, kwargs: {"max_results": 1},
        success_predicate=lambda result: result is not None,
    )
    def find_papers(self, query, excluded_ids=None):
        return self.replay.replay_arxiv_search(
            query,
            max_results=1,
            processed_query=self._process_query(query),
            excluded_ids=excluded_ids,
            return_details=True,
        )

    @traced_call("tool", "arxiv.full_text")
    def retrieve_full_paper_text(self, query, MAX_LEN=50000):
        text = self.replay.replay_arxiv_full_text(query, MAX_LEN)
        exact, base = self._paper_cache_keys(query)
        self._full_text_cache[exact] = text
        self._full_text_cache.setdefault(base, text)
        return text


BLOCKED_IMPORTS = {
    "aiohttp",
    "arxiv",
    "ftplib",
    "httpx",
    "huggingface_hub",
    "requests",
    "semanticscholar",
    "socket",
    "subprocess",
    "urllib",
}
BLOCKED_CALLS = {
    "InferenceClient",
    "OpenAI",
    "query_deepseekv3",
    "query_gemini",
    "query_gemini2p0",
    "query_gpt4o",
    "query_gpt4omini",
    "query_model",
    "query_qwen",
}


def validate_offline_code(code_str, seeds=None):
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        raise OfflineExecutionError(f"Generated code is invalid Python: {exc}") from exc
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                root = name.name.split(".", 1)[0]
                if root in BLOCKED_IMPORTS:
                    violations.append(f"import {name.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in BLOCKED_IMPORTS:
                violations.append(f"from {node.module} import ...")
            if root == "os" and any(
                name.name in {"system", "popen", "spawnl", "spawnv"}
                for name in node.names
            ):
                violations.append("import process execution from os")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
                violations.append(f"call {node.func.id}()")
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "from_pretrained":
                    violations.append(
                        "call from_pretrained() without a materialized local model"
                    )
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "system"
                ):
                    violations.append("call os.system()")
    if violations:
        raise OfflineExecutionError(
            "Generated code violates strict offline execution: "
            + ", ".join(sorted(set(violations)))
        )
    try:
        validate_seed_usage(tree, SeedConfig.from_mapping(seeds or current_seed_config()))
    except ValueError as exc:
        raise OfflineExecutionError(str(exc)) from exc


def _install_network_guard():
    def blocked(*args, **kwargs):
        del args, kwargs
        raise OfflineExecutionError("Network and subprocess access are disabled")

    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    os.system = blocked


matplotlib.use("Agg")


def _artifact_inventory(globals_dict, initial_names):
    artifacts = []
    for name, value in sorted(globals_dict.items()):
        if name in initial_names or name.startswith("_"):
            continue
        if isinstance(value, types.ModuleType) or callable(value):
            continue
        if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
            continue

        artifact = {
            "name": name,
            "type": f"{type(value).__module__}.{type(value).__name__}",
        }
        try:
            shape = getattr(value, "shape", None)
        except Exception:
            shape = None
        if shape is not None:
            try:
                artifact["shape"] = [int(item) for item in shape]
            except (TypeError, ValueError):
                artifact["shape"] = str(shape)
        try:
            dtype = getattr(value, "dtype", None)
        except Exception:
            dtype = None
        if dtype is not None:
            artifact["dtype"] = str(dtype)
            kind = getattr(dtype, "kind", None)
            artifact["numeric"] = (
                kind in {"b", "i", "u", "f", "c"}
                if kind is not None
                else str(dtype).startswith((
                    "torch.bool",
                    "torch.int",
                    "torch.uint",
                    "torch.float",
                    "torch.complex",
                ))
            )
        try:
            dtypes = getattr(value, "dtypes", None)
        except Exception:
            dtypes = None
        if dtypes is not None and hasattr(dtypes, "items"):
            rendered_dtypes = {
                str(column): str(column_dtype)
                for column, column_dtype in dtypes.items()
            }
            artifact["dtypes"] = rendered_dtypes
            artifact["numeric"] = all(
                getattr(column_dtype, "kind", None)
                in {"b", "i", "u", "f", "c"}
                for _, column_dtype in dtypes.items()
            )
            artifact["columns"] = list(rendered_dtypes)
        try:
            artifact["length"] = int(len(value))
        except (TypeError, ValueError, OverflowError):
            pass
        if "shape" not in artifact and "length" not in artifact:
            continue
        artifacts.append(artifact)
        if len(artifacts) >= 50:
            break
    return artifacts


def _prepared_manifest_for_transport(value):
    if not isinstance(value, dict):
        return None
    try:
        serialized = json.dumps(value, ensure_ascii=True)
    except (TypeError, ValueError):
        return {
            "_manifest_error": (
                "prepared_data_manifest must contain JSON metadata and variable "
                "names, not arrays or model objects"
            )
        }
    if len(serialized) > 20000:
        return {
            "_manifest_error": "prepared_data_manifest exceeds 20000 characters"
        }
    return json.loads(serialized)


class _BoundedCapture(io.StringIO):
    """Capture a bounded prefix without retaining unlimited generated output."""
    def __init__(self, limit):
        super().__init__()
        self.limit = max(0, int(limit))
        self.truncated = False

    def write(self, text):
        remaining = max(0, self.limit - self.tell())
        self.truncated |= len(text) > remaining
        super().write(text[:remaining])
        return len(text)


def _execution_failure(error_type, error, traceback_text="", exit_code=None):
    return {
        "success": False,
        "output": "",
        "error_type": error_type,
        "error": str(error)[:2000],
        "traceback": traceback_text[-12000:],
        "exit_code": exit_code,
        "replay_events": [],
        "artifacts": [],
        "prepared_data_manifest": None,
        "output_truncated": False,
    }


def _format_execution_result(result, max_len):
    """Bound stdout/stderr while keeping structured diagnostics separate."""
    result = dict(result)
    output = str(result.get("output", ""))
    limit = max(0, int(max_len))
    result["output_truncated"] = bool(result.get("output_truncated")) or len(output) > limit
    result["output"] = output[:limit]
    return result


def format_execution_text(result, max_len=50000):
    """Render diagnostics only for callers of the legacy text interface."""
    output = str(result.get("output", ""))
    limit = max(0, int(max_len))
    if result["success"]:
        return output[:limit]
    summary = f"[CODE EXECUTION ERROR]: {result['error_type']}: {result['error']}"
    traceback_text = result.get("traceback") or ""
    text = summary + "\n"
    if traceback_text:
        text += traceback_text + "\n"
    text += output
    rendered = text[:limit]
    # Preserve the old failure-marker fallback for limits too small to hold it.
    if "[CODE EXECUTION ERROR]" not in rendered:
        return summary
    return rendered


def worker_run_code(
    code_str,
    output_queue,
    replay_context=None,
    workdir=None,
    seeds=None,
    max_output=50000,
):
    stdout = _BoundedCapture(max_output)
    stderr = _BoundedCapture(max_output)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout, stderr
    worker_replay = None
    globals_dict = {}
    initial_names = set()
    stage = "worker_setup_error"
    result = _execution_failure(stage, "Worker did not complete")
    try:
        if workdir:
            os.chdir(workdir)
        if replay_context is not None:
            worker_replay = install_worker_hf_replay(replay_context)
        _install_network_guard()
        globals_dict = {
            "__name__": "__main__",
            **execution_seed_globals(code_str, seeds),
        }
        initial_names = set(globals_dict)
        stage = "python_exception"
        exec(code_str, globals_dict)
        stage = "result_collection_error"
        result.update(
            artifacts=_artifact_inventory(globals_dict, initial_names),
            prepared_data_manifest=_prepared_manifest_for_transport(
                globals_dict.get("prepared_data_manifest")
            ),
            success=True,
            error_type=None,
            error=None,
            traceback="",
        )
    except BaseException as exc:
        # Only the child catches SystemExit/KeyboardInterrupt. The parent remains interruptible.
        kind = "worker_interrupted" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else stage
        result.update(_execution_failure(
            kind, f"{type(exc).__name__}: {exc}", traceback.format_exc()
        ))
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
    result["output"] = stdout.getvalue()
    if stderr.getvalue():
        result["output"] += "\n[stderr]\n" + stderr.getvalue()
    result["output_truncated"] = stdout.truncated or stderr.truncated
    result["replay_events"] = worker_replay.events if worker_replay is not None else []
    output_queue.put(result)


def _wait_for_execution(process, output_queue, deadline):
    # Drain the queue while the worker is alive: joining first can block its feeder
    # thread when the serialized result is larger than the OS pipe buffer.
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _execution_failure("timeout", "Code execution exceeded its time limit")
        try:
            payload = output_queue.get(timeout=min(0.1, remaining))
            break
        except queue.Empty:
            if process.exitcode is not None:
                process.join(timeout=0)
                # A final message may have arrived between the timed get and exit.
                try:
                    payload = output_queue.get(timeout=min(0.1, max(0, deadline - time.monotonic())))
                    break
                except queue.Empty:
                    pass
                if process.exitcode != 0:
                    return _execution_failure(
                        "process_exit", f"Worker exited with code {process.exitcode}",
                        exit_code=process.exitcode,
                    )
                return _execution_failure(
                    "missing_result", "Worker exited without returning an execution result",
                    exit_code=process.exitcode,
                )
        except (EOFError, OSError, ValueError) as exc:
            return _execution_failure("result_transport_error", f"{type(exc).__name__}: {exc}")

    process.join(timeout=max(0, deadline - time.monotonic()))
    if process.is_alive():
        return _execution_failure("timeout", "Worker returned a result but did not exit before the deadline")
    if process.exitcode != 0:
        return _execution_failure(
            "process_exit", f"Worker exited with code {process.exitcode}",
            exit_code=process.exitcode,
        )
    if (
        not isinstance(payload, dict)
        or not {"success", "output", "error_type", "error", "traceback",
                "replay_events", "artifacts", "prepared_data_manifest"}.issubset(payload)
        or type(payload.get("success")) is not bool
        or not isinstance(payload.get("output"), str)
        or not isinstance(payload.get("traceback"), str)
        or not isinstance(payload.get("replay_events"), list)
        or not isinstance(payload.get("artifacts"), list)
        or (payload["success"] is False and not payload.get("error"))
    ):
        return _execution_failure("invalid_result", "Worker returned an invalid execution result")
    payload["exit_code"] = process.exitcode
    return payload


@traced_call(
    "tool",
    "python.execute_code",
    metadata_factory=lambda args, kwargs: {
        "timeout_seconds": kwargs.get("timeout", args[1] if len(args) > 1 else 600),
        "workdir": kwargs.get("workdir"),
        "seeds": SeedConfig.from_mapping(
            kwargs.get("seeds") or current_seed_config()
        ).to_dict(),
    },
    success_predicate=lambda result: result["success"] is True,
)
def _execute_code_details(code_str, timeout=600, MAX_LEN=50000, workdir=None, seeds=None):
    process = None
    output_queue = None
    replay = None
    result = None
    try:
        resolved_seeds = SeedConfig.from_mapping(seeds or current_seed_config())
        try:
            validate_offline_code(code_str, resolved_seeds)
        except OfflineExecutionError as exc:
            return _format_execution_result(
                _execution_failure("validation_error", exc), MAX_LEN
            )
        replay = current_local_replay(required=False)
        replay_context = replay.worker_context() if replay is not None else None
        resolved_workdir = None
        if workdir is not None:
            resolved_workdir = str(Path(workdir).resolve())
            os.makedirs(resolved_workdir, exist_ok=True)
        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=worker_run_code,
            args=(code_str, output_queue, replay_context, resolved_workdir,
                  resolved_seeds.to_dict(), MAX_LEN),
        )
        deadline = time.monotonic() + float(timeout)
        process.start()
        result = _wait_for_execution(process, output_queue, deadline)
        if result["error_type"] == "timeout":
            result["error"] += f" ({timeout} seconds)"
    except Exception as exc:
        result = _execution_failure(
            "execution_environment_error", f"{type(exc).__name__}: {exc}", traceback.format_exc()
        )
    finally:
        # Also clean up on a parent KeyboardInterrupt, then let it propagate.
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            if result is not None:
                result["exit_code"] = process.exitcode
            if not process.is_alive():
                process.close()
        if output_queue is not None:
            output_queue.close()
    if replay is not None:
        # Recording errors are framework errors, not generated-code failures.
        replay.record_worker_events(result.get("replay_events", []))
    return _format_execution_result(result, MAX_LEN)


def execute_code(
    code_str, timeout=600, MAX_LEN=50000, workdir=None, seeds=None, return_details=False,
):
    result = _execute_code_details(
        code_str, timeout=timeout, MAX_LEN=MAX_LEN, workdir=workdir, seeds=seeds,
    )
    if return_details:
        return result
    return format_execution_text(result, MAX_LEN)
