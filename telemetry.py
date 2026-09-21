import contextlib
import contextvars
import csv
import functools
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone


_recorder_var = contextvars.ContextVar("agentlab_trace_recorder", default=None)
_agent_var = contextvars.ContextVar("agentlab_trace_agent", default="LaboratoryWorkflow")
_phase_var = contextvars.ContextVar("agentlab_trace_phase", default="unknown")
_step_var = contextvars.ContextVar("agentlab_trace_step", default=None)
_trigger_var = contextvars.ContextVar("agentlab_trace_trigger", default="framework_automatic")
_parent_call_var = contextvars.ContextVar("agentlab_trace_parent_call", default=None)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


class TraceRecorder:
    def __init__(self, base_dir, run_id=None):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = run_id or f"{timestamp}_{uuid.uuid4().hex[:8]}"
        self.base_dir = os.path.abspath(base_dir)
        self.run_dir = os.path.join(self.base_dir, self.run_id)
        self.started_at = _utc_now()
        self.started_perf_ns = time.perf_counter_ns()
        self.events = []
        self.lock = threading.Lock()
        os.makedirs(self.run_dir, exist_ok=True)
        with open(os.path.join(self.base_dir, "latest_run.txt"), "w", encoding="utf-8") as output:
            output.write(self.run_id)

    def record(self, event):
        event = dict(event)
        event["run_id"] = self.run_id
        with self.lock:
            self.events.append(event)
            self._append_jsonl("trace.jsonl", event)
            if event["category"] == "tool":
                self._append_jsonl("tool_trace.jsonl", event)
            elif event["category"] == "llm":
                self._append_jsonl("llm_trace.jsonl", event)

    def _append_jsonl(self, filename, event):
        path = os.path.join(self.run_dir, filename)
        with open(path, "a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")

    def finalize(self):
        with self.lock:
            events = list(self.events)
        self._write_summary("tool", "tool_summary.csv", events)
        self._write_summary("llm", "llm_summary.csv", events)
        elapsed_ms = (time.perf_counter_ns() - self.started_perf_ns) / 1_000_000
        run_summary = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": _utc_now(),
            "workflow_elapsed_ms": round(elapsed_ms, 3),
            "event_count": len(events),
            "tool_call_count": sum(event["category"] == "tool" for event in events),
            "llm_call_count": sum(event["category"] == "llm" for event in events),
            "tool_inclusive_duration_ms": round(sum(
                event["duration_ms"] for event in events if event["category"] == "tool"
            ), 3),
            "llm_duration_ms": round(sum(
                event["duration_ms"] for event in events if event["category"] == "llm"
            ), 3),
            "note": "Tool durations are inclusive; use parent_call_id to avoid double-counting nested calls.",
        }
        with open(os.path.join(self.run_dir, "run_summary.json"), "w", encoding="utf-8") as output:
            json.dump(run_summary, output, ensure_ascii=True, indent=2)
        return run_summary

    def _write_summary(self, category, filename, events):
        grouped = {}
        for event in events:
            if event["category"] != category:
                continue
            key = (
                event["agent"],
                event["phase"],
                event["name"],
                event["trigger"],
            )
            grouped.setdefault(key, []).append(event)

        columns = [
            "agent", "phase", "name", "trigger", "count", "success_count",
            "failure_count", "total_ms", "avg_ms", "p50_ms", "p95_ms", "max_ms",
        ]
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for key in sorted(grouped):
                group = grouped[key]
                durations = [event["duration_ms"] for event in group]
                success_count = sum(bool(event["success"]) for event in group)
                total_ms = sum(durations)
                writer.writerow({
                    "agent": key[0],
                    "phase": key[1],
                    "name": key[2],
                    "trigger": key[3],
                    "count": len(group),
                    "success_count": success_count,
                    "failure_count": len(group) - success_count,
                    "total_ms": round(total_ms, 3),
                    "avg_ms": round(total_ms / len(group), 3),
                    "p50_ms": round(_percentile(durations, 0.50), 3),
                    "p95_ms": round(_percentile(durations, 0.95), 3),
                    "max_ms": round(max(durations), 3),
                })


@contextlib.contextmanager
def trace_scope(recorder=None, agent=None, phase=None, step=None, trigger=None):
    updates = []
    values = (
        (_recorder_var, recorder),
        (_agent_var, agent),
        (_phase_var, phase),
        (_step_var, step),
        (_trigger_var, trigger),
    )
    for variable, value in values:
        if value is not None:
            updates.append((variable, variable.set(value)))
    try:
        yield
    finally:
        for variable, token in reversed(updates):
            variable.reset(token)


def current_trace_context():
    return {
        "agent": _agent_var.get(),
        "phase": _phase_var.get(),
        "step": _step_var.get(),
        "trigger": _trigger_var.get(),
        "trace_call_id": _parent_call_var.get(),
    }


def record_trace_event(category, name, success=True, metadata=None):
    recorder = _recorder_var.get()
    if recorder is None:
        return
    context = current_trace_context()
    now = _utc_now()
    recorder.record({
        "call_id": uuid.uuid4().hex,
        "parent_call_id": context["trace_call_id"],
        "category": category,
        "agent": context["agent"],
        "phase": context["phase"],
        "step": context["step"],
        "name": name,
        "trigger": context["trigger"],
        "started_at": now,
        "ended_at": now,
        "duration_ms": 0.0,
        "success": bool(success),
        "error_type": None,
        "error_message": None,
        "metadata": metadata or {},
    })


def traced_call(category, name=None, metadata_factory=None, success_predicate=None):
    def decorator(function):
        event_name = name or f"{function.__module__}.{function.__qualname__}"

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            recorder = _recorder_var.get()
            if recorder is None:
                return function(*args, **kwargs)

            call_id = uuid.uuid4().hex
            parent_call_id = _parent_call_var.get()
            parent_token = _parent_call_var.set(call_id)
            started_at = _utc_now()
            started_perf_ns = time.perf_counter_ns()
            success = False
            error_type = None
            error_message = None
            result = None
            try:
                result = function(*args, **kwargs)
                success = True if success_predicate is None else bool(success_predicate(result))
                return result
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)[:500]
                raise
            finally:
                duration_ms = (time.perf_counter_ns() - started_perf_ns) / 1_000_000
                metadata = {}
                if metadata_factory is not None:
                    try:
                        metadata = metadata_factory(args, kwargs) or {}
                    except Exception as exc:
                        metadata = {"metadata_error": str(exc)[:200]}
                try:
                    recorder.record({
                        "call_id": call_id,
                        "parent_call_id": parent_call_id,
                        "category": category,
                        "agent": _agent_var.get(),
                        "phase": _phase_var.get(),
                        "step": _step_var.get(),
                        "name": event_name,
                        "trigger": _trigger_var.get(),
                        "started_at": started_at,
                        "ended_at": _utc_now(),
                        "duration_ms": round(duration_ms, 3),
                        "success": success,
                        "error_type": error_type,
                        "error_message": error_message,
                        "metadata": metadata,
                    })
                except Exception:
                    pass
                finally:
                    _parent_call_var.reset(parent_token)

        return wrapper
    return decorator
