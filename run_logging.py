import contextlib
import contextvars
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


_run_logger_var = contextvars.ContextVar("agentlab_run_logger", default=None)
_prompt_recorder_var = contextvars.ContextVar(
    "agentlab_prompt_recorder", default=None
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class NullRunLogger:
    def debug(self, message):
        del message

    def info(self, message):
        del message

    def error(self, message):
        del message

    def exception(self, message):
        del message

    def close(self):
        return None


class NullPromptRecorder:
    def record(self, payload):
        del payload


class RunLogger:
    def __init__(self, workdir, query_id, job_id, console=True):
        self.query_id = str(query_id)
        self.job_id = str(job_id)
        self.path = Path(workdir).resolve() / "workflow.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"agentlab.{self.query_id}.{self.job_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(self.path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(query_id)s] %(message)s"
        ))
        self.logger.addHandler(file_handler)

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(query_id)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            self.logger.addHandler(console_handler)

    def _write(self, level, message, exc_info=False):
        self.logger.log(
            level,
            str(message),
            extra={"query_id": self.query_id},
            exc_info=exc_info,
        )

    def debug(self, message):
        self._write(logging.DEBUG, message)

    def info(self, message):
        self._write(logging.INFO, message)

    def error(self, message):
        self._write(logging.ERROR, message)

    def exception(self, message):
        self._write(logging.ERROR, message, exc_info=True)

    def close(self):
        handlers = list(self.logger.handlers)
        self.logger.handlers.clear()
        for handler in handlers:
            handler.flush()
            handler.close()


class PromptRecorder:
    def __init__(self, workdir, query_id, job_id):
        self.query_id = str(query_id)
        self.job_id = str(job_id)
        self.path = Path(workdir).resolve() / "prompts.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def record(self, payload):
        event = {
            "recorded_at": _utc_now(),
            "query_id": self.query_id,
            "job_id": self.job_id,
            **dict(payload),
        }
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as output:
                output.write(
                    json.dumps(event, ensure_ascii=True, default=str) + "\n"
                )


_NULL_LOGGER = NullRunLogger()
_NULL_PROMPT_RECORDER = NullPromptRecorder()


def current_run_logger():
    return _run_logger_var.get() or _NULL_LOGGER


def current_prompt_recorder():
    return _prompt_recorder_var.get() or _NULL_PROMPT_RECORDER


@contextlib.contextmanager
def run_log_scope(run_logger, prompt_recorder):
    logger_token = _run_logger_var.set(run_logger)
    prompt_token = _prompt_recorder_var.set(prompt_recorder)
    try:
        yield
    finally:
        _prompt_recorder_var.reset(prompt_token)
        _run_logger_var.reset(logger_token)


@contextlib.contextmanager
def phase_log(name):
    logger = current_run_logger()
    started = time.perf_counter()
    logger.info(f"BEGIN {name}")
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception(f"FAILED {name} elapsed={elapsed:.2f}s")
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info(f"END {name} elapsed={elapsed:.2f}s")
