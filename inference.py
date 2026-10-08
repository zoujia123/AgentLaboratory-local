import contextlib
import contextvars
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from openai import OpenAI

from run_logging import current_run_logger
from telemetry import traced_call


_http_attempt_budget_var = contextvars.ContextVar(
    "agentlab_llm_http_attempt_budget",
    default=None,
)

_llm_sampling_var = contextvars.ContextVar(
    "agentlab_llm_sampling",
    default=None,
)


class LLMHTTPAttemptLimitError(RuntimeError):
    pass


class ToolResponseError(RuntimeError):
    pass


class ToolResponseLengthError(ToolResponseError):
    pass


@dataclass
class LLMHTTPAttemptBudget:
    limit: int
    used: int = 0
    request_blocked: bool = False

    def __post_init__(self):
        self.limit = int(self.limit)
        if self.limit < 1:
            raise ValueError("LLM HTTP attempt limit must be at least 1")

    @property
    def remaining(self):
        return max(0, self.limit - self.used)

    def consume(self):
        if self.used >= self.limit:
            self.request_blocked = True
            raise LLMHTTPAttemptLimitError(
                f"LLM HTTP request limit reached: {self.used}/{self.limit} attempts used"
            )
        self.used += 1
        return self.used

    def snapshot(self):
        return {
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "limit_reached": self.used >= self.limit,
            "request_blocked": self.request_blocked,
        }


@contextlib.contextmanager
def llm_http_attempt_budget_scope(limit):
    budget = LLMHTTPAttemptBudget(limit)
    token = _http_attempt_budget_var.set(budget)
    try:
        yield budget
    finally:
        _http_attempt_budget_var.reset(token)


def current_llm_http_attempt_budget():
    return _http_attempt_budget_var.get()


@contextlib.contextmanager
def llm_sampling_scope(config):
    """Apply a task's sampling parameters in its executing thread."""
    token = _llm_sampling_var.set(config)
    try:
        yield config
    finally:
        _llm_sampling_var.reset(token)


def _effective_llm_sampling(temperature, max_tokens):
    config = _llm_sampling_var.get()
    if config is not None:
        # Task configuration takes precedence over per-agent token limits.
        return config.copy()
    return {"temperature": temperature, "max_tokens": max_tokens}


@dataclass
class ModelToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass
class ToolModelResponse:
    content: str
    tool_calls: list
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    reasoning_content_chars: int = 0
    response_id: str | None = None


def _single_tool_choice(tools):
    """如果只有一个工具，返回它的函数调用信息，必须调用它，如果有多个工具则返回"required"表示需要用户选择工具"""
    if len(tools) == 1:
        tool = tools[0]
        if not isinstance(tool, dict):
            return "required"
        function = tool.get("function")
        if tool.get("type") == "function" and isinstance(function, dict):
            name = function.get("name")
            if name:
                return {
                    "type": "function",
                    "function": {"name": name},
                }
    return "required"


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_choice_and_message(completion):
    choices = _field(completion, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
        raise ToolResponseError("Model response choices must be a non-empty sequence")
    if not choices:
        raise ToolResponseError("Model response has no choices")
    choice = choices[0]
    missing = object()
    message = _field(choice, "message", missing)
    if message is missing:
        raise ToolResponseError("Model response first choice has no message")
    message_fields = ("role", "content", "tool_calls", "refusal")
    if message is None or not any(
        _field(message, name, missing) is not missing for name in message_fields
    ):
        raise ToolResponseError("Model response first choice has an invalid message")
    content = _field(message, "content")
    if content is not None and not isinstance(content, str):
        raise ToolResponseError("Model response message content must be a string or null")
    return choice, message


def _response_structure_summary(completion):
    """Record bounded response metadata, never message bodies or request credentials."""
    summary = {"response_type": type(completion).__name__}
    try:
        response_id = _field(completion, "id")
        if isinstance(response_id, str):
            summary["response_id"] = response_id[:120]
        choices = _field(completion, "choices")
        summary["choices_type"] = type(choices).__name__
        if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
            summary["choices_count"] = len(choices)
            if choices:
                choice = choices[0]
                summary["first_choice_type"] = type(choice).__name__
                finish_reason = _field(choice, "finish_reason")
                if isinstance(finish_reason, str):
                    summary["finish_reason"] = finish_reason[:80]
                message = _field(choice, "message")
                summary["message_type"] = type(message).__name__
                content = _field(message, "content")
                summary["content_type"] = type(content).__name__
                if isinstance(content, str):
                    summary["content_chars"] = len(content)
                tool_calls = _field(message, "tool_calls")
                summary["tool_calls_type"] = type(tool_calls).__name__
                if isinstance(tool_calls, (list, tuple)):
                    summary["tool_calls_count"] = len(tool_calls)
        summary["error_field_type"] = type(_field(completion, "error")).__name__
    except Exception as exc:
        # Diagnostics must not hide the original request/response failure.
        summary["summary_error_type"] = type(exc).__name__
    return json.dumps(summary, ensure_ascii=True)[:2000]


def _response_usage(completion):
    usage = _field(completion, "usage")
    if usage is None:
        return {}
    completion_details = _field(usage, "completion_tokens_details")
    prompt_details = _field(usage, "prompt_tokens_details")
    values = {
        "prompt_tokens": _field(usage, "prompt_tokens"),
        "completion_tokens": _field(usage, "completion_tokens"),
        "total_tokens": _field(usage, "total_tokens"),
        "reasoning_tokens": _field(completion_details, "reasoning_tokens"),
        "cached_tokens": _field(prompt_details, "cached_tokens"),
    }
    return {name: value for name, value in values.items() if value is not None}


def _reasoning_content(message):
    value = _field(message, "reasoning_content")
    if value is None:
        value = _field(_field(message, "model_extra", {}), "reasoning_content")
    return "" if value is None else str(value)


def _tool_query_metadata(args, kwargs):
    requested_model = kwargs.get("model_str") or (args[0] if args else None)
    tools = kwargs.get("tools") or (args[2] if len(args) > 2 else [])
    budget = current_llm_http_attempt_budget()
    return {
        "requested_model": requested_model,
        "effective_model": os.getenv("OPENAI_MODEL") or requested_model,
        "tool_count": len(tools),
        "max_attempts": kwargs.get("tries", 3),
        "request_timeout_seconds": kwargs.get("request_timeout", 1800.0),
        "retry_delay_seconds": kwargs.get("retry_delay", 5.0),
        "sdk_max_retries": 0,
        "tool_choice": _single_tool_choice(tools),
        "workflow_http_attempt_budget": budget.snapshot() if budget else None,
        "llm_sampling": _effective_llm_sampling(
            kwargs.get("temperature", args[7] if len(args) > 7 else 0.2),
            kwargs.get("max_tokens", args[8] if len(args) > 8 else 8192),
        ),
    }


def _length_repair_messages(messages, tools):
    has_code_argument = any(
        "code" in (
            ((tool.get("function") or {}).get("parameters") or {}).get(
                "properties", {}
            )
        )
        for tool in tools
        if isinstance(tool, dict)
    )
    repair = (
        "Your previous response reached the output limit before completing a "
        "tool call. Return exactly one concise tool call now. Do not repeat the "
        "unchanged malformed output."
    )
    if has_code_argument:
        repair += (
            " Keep code under 6000 characters. Do not repeat imports, comments, "
            "or unchanged code."
        )
    repaired = [dict(message) for message in messages]
    if repaired and repaired[-1].get("role") == "user":
        repaired[-1]["content"] = (
            str(repaired[-1].get("content") or "") + "\n\n" + repair
        )
    else:
        repaired.append({"role": "user", "content": repair})
    return repaired


@traced_call("llm", "llm.query_model_with_tools", metadata_factory=_tool_query_metadata)
def query_model_with_tools(
    model_str,
    messages,
    tools,
    openai_api_key=None,
    tries=3,
    request_timeout=1800.0,
    retry_delay=50.0,
    temperature=0.2,
    max_tokens=8192 ,
    return_on_no_tool_call=False,
):
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    effective_model = os.getenv("OPENAI_MODEL") or model_str
    base_url = os.getenv("OPENAI_BASE_URL") or None
    sampling = _effective_llm_sampling(temperature, max_tokens)
    logger = current_run_logger()
    last_error = None
    tool_choice = _single_tool_choice(tools)
    request_messages = list(messages)
    length_repair_applied = False
    attempts_made = 0
    for attempt in range(1, tries + 1):
        attempts_made = attempt
        budget = current_llm_http_attempt_budget()
        try:
            workflow_attempt = budget.consume() if budget else None
        except LLMHTTPAttemptLimitError:
            logger.error(f"[LLM HTTP][attempt={attempt}/{tries}] BLOCKED request_limit={budget.used}/{budget.limit}")
            raise
        started = time.perf_counter()
        budget_suffix = (
            f" workflow_http_requests={workflow_attempt}/{budget.limit}"
            if budget else ""
        )
        logger.info(f"[LLM HTTP][attempt={attempt}/{tries}] BEGIN model={effective_model} timeout={request_timeout}s{budget_suffix} llm_sampling={json.dumps(sampling, sort_keys=True)}")
        completion = None
        failure_stage = "request"
        try:
            client_kwargs = {
                "api_key": api_key,
                "max_retries": 0,   # 表示关闭SDK自己的重试，因为外层已经实现了明确的3次重试，避免出现两层重试叠加。
                "timeout": request_timeout,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            completion = client.chat.completions.create(
                model=effective_model,
                messages=request_messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,  # 要求模型不要并行调用多个工具，尽量一次返回一个
                **sampling,
                extra_body={"enable_thinking": False},
            )
            failure_stage = "parse_response"
            choice, message = _response_choice_and_message(completion)
            finish_reason = _field(choice, "finish_reason")
            usage = _response_usage(completion)
            reasoning_content = _reasoning_content(message)
            response_id = _field(completion, "id")
            content = _field(message, "content") or ""
            returned_tool_calls = list(_field(message, "tool_calls", []) or [])
            logger.info(f"[LLM HTTP][attempt={attempt}/{tries}] RESPONSE response_id={response_id or 'None'} finish_reason={finish_reason or 'None'} prompt_tokens={usage.get('prompt_tokens', 'None')} completion_tokens={usage.get('completion_tokens', 'None')} total_tokens={usage.get('total_tokens', 'None')} reasoning_tokens={usage.get('reasoning_tokens', 'None')} cached_tokens={usage.get('cached_tokens', 'None')} reasoning_content_chars={len(reasoning_content)} content_chars={len(content)} tool_calls={len(returned_tool_calls)}")
            if not returned_tool_calls and not return_on_no_tool_call:
                error_type = (
                    ToolResponseLengthError
                    if finish_reason == "length"
                    else ToolResponseError
                )
                raise error_type(
                    "Model returned no tool calls "
                    f"(finish_reason={finish_reason or 'None'}, "
                    f"content_chars={len(content)}, "
                    f"reasoning_content_chars={len(reasoning_content)})"
                )
            response = ToolModelResponse(
                content=content,
                tool_calls=[
                    ModelToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                    )
                    for call in returned_tool_calls
                ],
                finish_reason=finish_reason,
                usage=usage,
                reasoning_content_chars=len(reasoning_content),
                response_id=response_id,
            )
            elapsed = time.perf_counter() - started
            logger.info(f"[LLM HTTP][attempt={attempt}/{tries}] SUCCESS elapsed={elapsed:.2f}s")
            return response
        except Exception as exc:
            elapsed = time.perf_counter() - started
            last_error = exc
            logger.exception(f"[LLM HTTP][attempt={attempt}/{tries}] FAILED error_type={type(exc).__name__} elapsed={elapsed:.2f}s stage={failure_stage} error={exc}")
            if failure_stage == "parse_response":
                summary = _response_structure_summary(completion)
                if api_key:
                    summary = summary.replace(api_key, "[REDACTED]")
                logger.error(f"[LLM HTTP][attempt={attempt}/{tries}] RESPONSE STRUCTURE {summary}")
            if attempt < tries:
                if isinstance(exc, ToolResponseLengthError):
                    if length_repair_applied:
                        logger.error(f"[LLM HTTP][attempt={attempt}/{tries}] LENGTH REPAIR EXHAUSTED")
                        break
                    if budget is not None and budget.remaining == 0:
                        break
                    length_repair_applied = True
                    request_messages = _length_repair_messages(messages, tools)
                    logger.info(f"[LLM HTTP][attempt={attempt}/{tries}] RETRYING mode=length_repair")
                    continue
                if budget is not None and budget.remaining == 0:
                    continue
                logger.info(f"[LLM HTTP][attempt={attempt}/{tries}] RETRYING after={retry_delay:g}s")
                time.sleep(retry_delay)
    raise RuntimeError(
        f"Tool-calling model request failed after {attempts_made} attempts: {last_error}"
    ) from last_error
