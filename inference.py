import contextlib
import contextvars
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI

from run_logging import current_run_logger
from telemetry import traced_call


_http_attempt_budget_var = contextvars.ContextVar(
    "agentlab_llm_http_attempt_budget",
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
    retry_delay=15.0,
    temperature=0.2,
    max_tokens=8192 ,
    return_on_no_tool_call=False,
):
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    effective_model = os.getenv("OPENAI_MODEL") or model_str
    base_url = os.getenv("OPENAI_BASE_URL") or None
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
            logger.error(
                f"[LLM HTTP][attempt={attempt}/{tries}] BLOCKED "
                f"request_limit={budget.used}/{budget.limit}"
            )
            raise
        started = time.perf_counter()
        budget_suffix = (
            f" workflow_http_requests={workflow_attempt}/{budget.limit}"
            if budget else ""
        )
        logger.info(
            f"[LLM HTTP][attempt={attempt}/{tries}] BEGIN "
            f"model={effective_model} timeout={request_timeout}s{budget_suffix}"
        )
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
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": False},
            )
            choice = completion.choices[0]  # 代码只读取第一个候选答案
            message = choice.message
            finish_reason = _field(choice, "finish_reason")
            usage = _response_usage(completion)
            reasoning_content = _reasoning_content(message)
            response_id = _field(completion, "id")
            content = _field(message, "content") or ""
            returned_tool_calls = list(_field(message, "tool_calls", []) or [])
            logger.info(
                f"[LLM HTTP][attempt={attempt}/{tries}] RESPONSE "
                f"response_id={response_id or 'None'} "
                f"finish_reason={finish_reason or 'None'} "
                f"prompt_tokens={usage.get('prompt_tokens', 'None')} "
                f"completion_tokens={usage.get('completion_tokens', 'None')} "
                f"total_tokens={usage.get('total_tokens', 'None')} "
                f"reasoning_tokens={usage.get('reasoning_tokens', 'None')} "
                f"cached_tokens={usage.get('cached_tokens', 'None')} "
                f"reasoning_content_chars={len(reasoning_content)} "
                f"content_chars={len(content)} "
                f"tool_calls={len(returned_tool_calls)}"
            )
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
            logger.info(
                f"[LLM HTTP][attempt={attempt}/{tries}] SUCCESS "
                f"elapsed={elapsed:.2f}s"
            )
            return response
        except Exception as exc:
            elapsed = time.perf_counter() - started
            last_error = exc
            logger.error(
                f"[LLM HTTP][attempt={attempt}/{tries}] FAILED "
                f"error_type={type(exc).__name__} elapsed={elapsed:.2f}s "
                f"error={exc}"
            )
            if attempt < tries:
                if isinstance(exc, ToolResponseLengthError):
                    if length_repair_applied:
                        logger.error(
                            f"[LLM HTTP][attempt={attempt}/{tries}] "
                            "LENGTH REPAIR EXHAUSTED"
                        )
                        break
                    if budget is not None and budget.remaining == 0:
                        break
                    length_repair_applied = True
                    request_messages = _length_repair_messages(messages, tools)
                    logger.info(
                        f"[LLM HTTP][attempt={attempt}/{tries}] RETRYING "
                        "mode=length_repair"
                    )
                    continue
                if budget is not None and budget.remaining == 0:
                    continue
                logger.info(
                    f"[LLM HTTP][attempt={attempt}/{tries}] RETRYING "
                    f"after={retry_delay:g}s"
                )
                time.sleep(retry_delay)
    raise RuntimeError(
        f"Tool-calling model request failed after {attempts_made} attempts: {last_error}"
    )
