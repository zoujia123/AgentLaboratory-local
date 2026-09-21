import hashlib
import json
import time
from dataclasses import dataclass

from inference import ModelToolCall, query_model_with_tools
from run_logging import current_prompt_recorder, current_run_logger
from telemetry import record_trace_event, trace_scope


class ToolProtocolError(RuntimeError):
    pass


FAILED_ARGUMENT_COMPACT_THRESHOLD = 2000
MAX_IDENTICAL_FAILED_TOOL_CALLS = 3


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: object
    allow_multiple: bool = False

    # strict: bool = True 请求服务端严格遵守工具参数结构
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentRunResult:
    tool_name: str
    arguments: dict
    value: object
    messages: list


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(
        self,
        name,
        description,
        parameters,
        handler,
        allow_multiple=False,
    ):
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")
        if parameters.get("type") != "object":
            raise ValueError(f"Tool {name!r} parameters must be an object schema")
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            allow_multiple=bool(allow_multiple),
        )
        return self

    @property
    def schemas(self):
        return self.schemas_for(self.names)

    @property
    def names(self):
        return set(self._tools)

    def schemas_for(self, names):
        names = set(names)
        unknown = sorted(names - self.names)
        if unknown:
            raise ToolProtocolError(
                "Cannot expose unknown tools: " + ", ".join(unknown)
            )
        return [
            tool.schema
            for name, tool in self._tools.items()
            if name in names
        ]

    def allows_multiple(self, name):
        try:
            return self._tools[name].allow_multiple
        except KeyError as exc:
            raise ToolProtocolError(f"Model requested unknown tool {name!r}") from exc

    def missing_required_arguments(self, name, arguments):
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise ToolProtocolError(f"Model requested unknown tool {name!r}") from exc
        if not isinstance(arguments, dict):
            raise ToolProtocolError(f"Arguments for {name!r} must be a JSON object")
        required = set(tool.parameters.get("required", []))
        return sorted(required - set(arguments))

    def execute(self, name, arguments):
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise ToolProtocolError(f"Model requested unknown tool {name!r}") from exc
        missing = self.missing_required_arguments(name, arguments)
        properties = set(tool.parameters.get("properties", {}))
        if missing:
            raise ToolProtocolError(
                f"Tool {name!r} is missing required arguments: {', '.join(missing)}"
            )
        if tool.parameters.get("additionalProperties") is False:
            unexpected = sorted(set(arguments) - properties)
            if unexpected:
                raise ToolProtocolError(
                    f"Tool {name!r} received unexpected arguments: {', '.join(unexpected)}"
                )
        try:
            return tool.handler(**arguments)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Tool {name!r} failed: {exc}",
            }


class AgentRuntime:
    def __init__(self,model,api_key,agent_name,phase,max_tool_steps=8,max_tokens=8192,max_protocol_repairs=2,max_rounds=None,):
        self.model = model
        self.api_key = api_key
        self.agent_name = agent_name
        self.phase = phase
        self.max_tool_steps = max(1, int(max_tool_steps))
        self.max_tokens = max(256, int(max_tokens))
        self.max_protocol_repairs = max(0, int(max_protocol_repairs))
        default_max_rounds = self.max_tool_steps + self.max_protocol_repairs
        self.max_rounds = max(
            1,
            int(default_max_rounds if max_rounds is None else max_rounds),
        )

    @staticmethod
    def _tool_output(value):
        return json.dumps(value, ensure_ascii=True, default=str)

    @staticmethod
    def _assistant_tool_call(tool_call):
        return {
            "id": tool_call.call_id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        }

    @staticmethod
    def _failed_call_signature(tool_name, arguments, value):
        error = value.get("error") if isinstance(value, dict) else str(value)
        code_hash = value.get("code_hash") if isinstance(value, dict) else None
        payload = json.dumps(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "error": error,
                "code_hash": code_hash,
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _compact_failed_arguments(arguments):
        serialized = json.dumps(
            arguments,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        if len(serialized) <= FAILED_ARGUMENT_COMPACT_THRESHOLD:
            return None

        compacted = {}
        for name, value in arguments.items():
            value_text = json.dumps(value, ensure_ascii=True, default=str)
            if len(value_text) <= 500:
                compacted[name] = value
                continue
            digest = hashlib.sha256(value_text.encode("utf-8")).hexdigest()[:12]
            if isinstance(value, str):
                marker = (
                    f"[rejected {name} omitted: sha256={digest}, "
                    f"chars={len(value)}, lines={len(value.splitlines())}]"
                )
                compacted[name] = marker
            elif isinstance(value, list):
                marker = (
                    f"[rejected {name} omitted: sha256={digest}, "
                    f"items={len(value)}, serialized_chars={len(value_text)}]"
                )
                compacted[name] = [marker]
            else:
                compacted[name] = {
                    "rejected_value_omitted": True,
                    "sha256": digest,
                    "serialized_chars": len(value_text),
                }
        return compacted

    def _parse_arguments(self, tool_call):
        try:
            return json.loads(tool_call.arguments)
        except json.JSONDecodeError as exc:
            raise ToolProtocolError(
                f"{self.agent_name} returned invalid JSON arguments for "
                f"{tool_call.name!r}: {exc}"
            ) from exc

    def _execute_tool(self, registry, tool_call, arguments, step):
        logger = current_run_logger()
        recorder = current_prompt_recorder()
        logger.info(
            f"[{self.agent_name}][{self.phase}][step={step}] "
            f"TOOL BEGIN {tool_call.name}"
        )
        logger.debug(
            f"[{self.agent_name}][{self.phase}][step={step}] "
            f"TOOL ARGUMENTS {tool_call.name}\n"
            + json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
        )
        started = time.perf_counter()
        value = registry.execute(tool_call.name, arguments)
        elapsed = time.perf_counter() - started
        success = not (
            isinstance(value, dict) and value.get("success") is False
        )
        logger.info(
            f"[{self.agent_name}][{self.phase}][step={step}] "
            f"TOOL END {tool_call.name} success={success} elapsed={elapsed:.2f}s"
        )
        logger.debug(
            f"[{self.agent_name}][{self.phase}][step={step}] "
            f"TOOL RESULT {tool_call.name}\n"
            + json.dumps(value, ensure_ascii=False, indent=2, default=str)
        )
        recorder.record({
            "event": "tool_execution",
            "agent": self.agent_name,
            "phase": self.phase,
            "step": step,
            "tool_call_id": tool_call.call_id,
            "tool_name": tool_call.name,
            "arguments": arguments,
            "result": value,
            "success": success,
            "duration_seconds": round(elapsed, 6),
        })
        return value

    def _append_no_tool_call_rejection(
        self,
        messages,
        response,
        active_tool_names,
        step,
        round_index,
    ):
        logger = current_run_logger()
        recorder = current_prompt_recorder()
        available_tools = sorted(active_tool_names)
        messages.append({
            "role": "assistant",
            "content": response.content or "",
        })
        messages.append({
            "role": "user",
            "content": (
                "Tool protocol error: the previous response did not call a tool. "
                "Call exactly one currently available tool in the next response. "
                "Available tools: " + ", ".join(available_tools)
            ),
        })
        logger.info(
            f"[{self.agent_name}][{self.phase}][round={round_index}]"
            f"[step={step}] PROTOCOL RECOVERY reason=no_tool_calls "
            f"available_tools={available_tools} executed_tools=0"
        )
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
            "round": round_index,
            "step": step,
            "reason": "no_tool_calls",
            "available_tools": available_tools,
            "executed_tools": [],
        })
        with trace_scope(
            agent=self.agent_name,
            phase=self.phase,
            step=step,
            trigger="framework_automatic",
        ):
            record_trace_event(
                category="tool_protocol",
                name="agent.rejected_missing_tool_call",
                success=True,
                metadata={
                    "agent": self.agent_name,
                    "phase": self.phase,
                    "round": round_index,
                    "available_tools": available_tools,
                    "executed_tools": [],
                },
            )

    def _append_unavailable_tool_rejection(
        self,
        messages,
        response,
        active_tool_names,
        step,
        round_index,
    ):
        logger = current_run_logger()
        recorder = current_prompt_recorder()
        available_tools = sorted(active_tool_names)
        requested_tools = [call.name for call in response.tool_calls]
        unavailable_tools = sorted(set(requested_tools) - set(active_tool_names))
        messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                self._assistant_tool_call(call)
                for call in response.tool_calls
            ],
        })
        for call in response.tool_calls:
            is_unavailable = call.name not in active_tool_names
            error_type = (
                "tool_not_available"
                if is_unavailable
                else "batch_rejected_due_to_unavailable_tool"
            )
            error = (
                f"Tool {call.name!r} is not available in the current state. "
                f"Choose one of the currently available tools: {available_tools}."
                if is_unavailable
                else (
                    "This tool was not executed because another call in the same "
                    "batch requested an unavailable tool. Retry using only currently "
                    f"available tools: {available_tools}."
                )
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": self._tool_output({
                    "success": False,
                    "error_type": error_type,
                    "error": error,
                    "tool_name": call.name,
                    "available_tools": available_tools,
                    "retryable": True,
                }),
            })
        logger.info(
            f"[{self.agent_name}][{self.phase}][round={round_index}]"
            f"[step={step}] PROTOCOL RECOVERY reason=tool_not_available "
            f"requested_tools={requested_tools} "
            f"unavailable_tools={unavailable_tools} executed_tools=0"
        )
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
            "round": round_index,
            "step": step,
            "reason": "tool_not_available",
            "requested_tools": requested_tools,
            "unavailable_tools": unavailable_tools,
            "available_tools": available_tools,
            "tool_call_ids": [call.call_id for call in response.tool_calls],
            "executed_tools": [],
        })
        with trace_scope(
            agent=self.agent_name,
            phase=self.phase,
            step=step,
            trigger="framework_automatic",
        ):
            record_trace_event(
                category="tool_protocol",
                name="agent.rejected_unavailable_tool_call",
                success=True,
                metadata={
                    "agent": self.agent_name,
                    "phase": self.phase,
                    "round": round_index,
                    "requested_tools": requested_tools,
                    "unavailable_tools": unavailable_tools,
                    "available_tools": available_tools,
                    "executed_tools": [],
                },
            )

    def _append_invalid_json_rejection(
        self,
        messages,
        response,
        invalid_json_by_call_id,
        step,
        round_index,
    ):
        logger = current_run_logger()
        recorder = current_prompt_recorder()
        requested_tools = [call.name for call in response.tool_calls]
        messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                self._assistant_tool_call(call)
                for call in response.tool_calls
            ],
        })
        for call in response.tool_calls:
            parse_error = invalid_json_by_call_id.get(call.call_id)
            if parse_error:
                error_type = "invalid_json_arguments"
                error = (
                    f"Tool {call.name!r} received invalid JSON arguments: "
                    f"{parse_error}. Retry with one valid JSON object containing "
                    "all required arguments. No tool was executed."
                )
            else:
                error_type = "batch_rejected_due_to_invalid_json_arguments"
                error = (
                    "This tool was not executed because another call in the same "
                    "batch contained invalid JSON arguments. Retry the corrected "
                    "batch."
                )
            messages.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": self._tool_output({
                    "success": False,
                    "error_type": error_type,
                    "error": error,
                    "tool_name": call.name,
                    "retryable": True,
                }),
            })
        logger.info(
            f"[{self.agent_name}][{self.phase}][round={round_index}]"
            f"[step={step}] PROTOCOL RECOVERY reason=invalid_json_arguments "
            f"requested_tools={requested_tools} executed_tools=0"
        )
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
            "round": round_index,
            "step": step,
            "reason": "invalid_json_arguments",
            "requested_tools": requested_tools,
            "invalid_json_by_call_id": dict(invalid_json_by_call_id),
            "tool_call_ids": [call.call_id for call in response.tool_calls],
            "executed_tools": [],
        })
        with trace_scope(
            agent=self.agent_name,
            phase=self.phase,
            step=step,
            trigger="framework_automatic",
        ):
            record_trace_event(
                category="tool_protocol",
                name="agent.rejected_invalid_json_arguments",
                success=True,
                metadata={
                    "agent": self.agent_name,
                    "phase": self.phase,
                    "round": round_index,
                    "requested_tools": requested_tools,
                    "invalid_json_by_call_id": dict(invalid_json_by_call_id),
                    "executed_tools": [],
                },
            )

    def _append_non_batchable_batch_rejection(self,messages,response,disallowed,step,repair_attempt,):
        logger = current_run_logger()
        recorder = current_prompt_recorder()
        requested_tools = [call.name for call in response.tool_calls]
        error = (
            "This entire tool-call batch was rejected. No tool was executed and "
            "the current state is unchanged. Call exactly one tool in the next "
            "response, then inspect its result before choosing another tool."
        )
        messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                self._assistant_tool_call(call)
                for call in response.tool_calls
            ],
        })
        for call in response.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": self._tool_output({
                    "success": False,
                    "error_type": "non_batchable_tool_batch",
                    "error": error,
                    "tool_name": call.name,
                    "requested_tools": requested_tools,
                }),
            })
        logger.info(
            f"[{self.agent_name}][{self.phase}][step={step}] "
            f"PROTOCOL RECOVERY attempt={repair_attempt}/"
            f"{self.max_protocol_repairs} requested_tools={requested_tools} "
            "executed_tools=0"
        )
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
            "step": step,
            "reason": "non_batchable_tool_batch",
            "repair_attempt": repair_attempt,
            "max_protocol_repairs": self.max_protocol_repairs,
            "requested_tools": requested_tools,
            "disallowed_tools": list(disallowed),
            "tool_call_ids": [
                call.call_id for call in response.tool_calls
            ],
            "executed_tools": [],
        })
        with trace_scope(
            agent=self.agent_name,
            phase=self.phase,
            step=step,
            trigger="framework_automatic",
        ):
            record_trace_event(
                category="tool_protocol",
                name="agent.rejected_non_batchable_tool_batch",
                success=True,
                metadata={
                    "agent": self.agent_name,
                    "phase": self.phase,
                    "repair_attempt": repair_attempt,
                    "requested_tools": requested_tools,
                    "disallowed_tools": list(disallowed),
                    "executed_tools": [],
                },
            )

    def _append_missing_argument_rejection(
        self,
        messages,
        response,
        missing_by_call_id,
        step,
        repair_attempt,
    ):
        logger = current_run_logger()
        recorder = current_prompt_recorder()
        requested_tools = [call.name for call in response.tool_calls]
        recovery_guidance = (
            "Do not repeat the unchanged malformed tool call. "
            "Review the previous successful tool result and choose the next "
            "appropriate tool."
        )
        messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                self._assistant_tool_call(call)
                for call in response.tool_calls
            ],
        })
        for call in response.tool_calls:
            missing = missing_by_call_id.get(call.call_id, [])
            if missing:
                error_type = "missing_required_arguments"
                error = (
                    f"Tool {call.name!r} is missing required arguments: "
                    f"{', '.join(missing)}. Retry with all required arguments or "
                    "choose a different appropriate tool. No tool was executed. "
                    + recovery_guidance
                )
            else:
                error_type = "batch_rejected_due_to_missing_arguments"
                error = (
                    "This tool was not executed because another call in the same "
                    "batch was missing required arguments. Retry the corrected batch. "
                    + recovery_guidance
                )
            messages.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": self._tool_output({
                    "success": False,
                    "error_type": error_type,
                    "error": error,
                    "tool_name": call.name,
                    "missing_arguments": missing,
                    "retryable": True,
                }),
            })
        logger.info(
            f"[{self.agent_name}][{self.phase}][step={step}] "
            f"PROTOCOL RECOVERY attempt={repair_attempt}/"
            f"{self.max_protocol_repairs} reason=missing_required_arguments "
            f"requested_tools={requested_tools} executed_tools=0"
        )
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
            "step": step,
            "reason": "missing_required_arguments",
            "repair_attempt": repair_attempt,
            "max_protocol_repairs": self.max_protocol_repairs,
            "requested_tools": requested_tools,
            "missing_by_call_id": dict(missing_by_call_id),
            "tool_call_ids": [call.call_id for call in response.tool_calls],
            "executed_tools": [],
        })
        with trace_scope(
            agent=self.agent_name,
            phase=self.phase,
            step=step,
            trigger="framework_automatic",
        ):
            record_trace_event(
                category="tool_protocol",
                name="agent.rejected_missing_tool_arguments",
                success=True,
                metadata={
                    "agent": self.agent_name,
                    "phase": self.phase,
                    "repair_attempt": repair_attempt,
                    "requested_tools": requested_tools,
                    "missing_by_call_id": dict(missing_by_call_id),
                    "executed_tools": [],
                },
            )

    def run(self,system_prompt,user_prompt,registry,terminal_tools,tool_names_provider=None,return_after_tool=False,automatic_tools=(),):
        # Alternating collaborators hand back each tool result, including failures.
        # Other agents retain the default loop-until-success behavior.

        logger = current_run_logger()   
        prompt_recorder = current_prompt_recorder()
        logger.info(f"[{self.agent_name}][{self.phase}] START "f"max_steps={self.max_tool_steps} "f"max_rounds={self.max_rounds} "f"max_protocol_repairs={self.max_protocol_repairs}")
        logger.debug(f"[{self.agent_name}][{self.phase}] SYSTEM PROMPT\n{system_prompt}")
        logger.debug(f"[{self.agent_name}][{self.phase}] USER PROMPT\n{user_prompt}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]   # messages会保存完整执行上下文,这里是初始化对话历史
        step = 0    # 已经执行的工具轮次
        round_index = 0    # 已经发起的LLM轮次，包括协议纠错轮次
        protocol_repairs = 0    # LLM违反多工具调用规则协议后，已经修复几次
        last_failed_call_signature = None   # 用于检测模型是否反复调用同一个失败工具
        identical_failed_call_count = 0
        terminal_tools = set(terminal_tools)
        automatic_tools = set(automatic_tools)
        unknown_automatic_tools = automatic_tools - registry.names
        if unknown_automatic_tools:
            raise ToolProtocolError(
                "Cannot automatically execute unknown tools: "
                + ", ".join(sorted(unknown_automatic_tools))
            )

        while step < self.max_tool_steps:
            current_round = round_index
            # 计算当前可用工具
            active_tool_names = set(    
                tool_names_provider()
                if tool_names_provider is not None
                else registry.names
            )   
            if not active_tool_names:
                raise ToolProtocolError(
                    f"{self.agent_name} has no tools available in phase {self.phase!r}"
                )
            tool_schemas = registry.schemas_for(active_tool_names)  # 从完整注册表里取出当前工具的名称、描述和参数Schema

            automatic_candidates = active_tool_names & automatic_tools
            if len(active_tool_names) == 1 and len(automatic_candidates) == 1:
                tool_name = next(iter(automatic_candidates))
                if not registry.missing_required_arguments(tool_name, {}):
                    tool_call = ModelToolCall(
                        call_id=(
                            f"automatic-{self.agent_name}-{self.phase}-"
                            f"{current_round}-{step}"
                        ),
                        name=tool_name,
                        arguments="{}",
                    )
                    logger.info(
                        f"[{self.agent_name}][{self.phase}][round={current_round}]"
                        f"[step={step}] AUTOMATIC TOOL {tool_name}"
                    )
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [self._assistant_tool_call(tool_call)],
                    })
                    value = self._execute_tool(registry, tool_call, {}, step)
                    if isinstance(value, dict):
                        value = {**value, "automatic_execution": True}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "content": self._tool_output(value),
                    })
                    terminal_success = not (
                        isinstance(value, dict) and value.get("success") is False
                    )
                    if return_after_tool or (
                        tool_name in terminal_tools and terminal_success
                    ):
                        return AgentRunResult(
                            tool_name=tool_name,
                            arguments={},
                            value=value,
                            messages=messages,
                        )
                    step += 1
                    continue

            if round_index >= self.max_rounds:
                break
            round_index += 1

            logger.info(
                f"[{self.agent_name}][{self.phase}][round={current_round}]"
                f"[step={step}] "
                f"LLM REQUEST tools={len(tool_schemas)} "
                f"names={sorted(active_tool_names)}"
            )
            logger.debug(   # 记录当前步骤和可用工具数量。
                f"[{self.agent_name}][{self.phase}][step={step}] "
                "LLM REQUEST PAYLOAD\n"
                + json.dumps(
                    {"messages": messages, "tools": tool_schemas},
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            request_messages = list(messages)

            started = time.perf_counter()
            # 调用 LLM
            try:
                with trace_scope(agent=self.agent_name,phase=self.phase,step=step,trigger="agent_requested",):
                    response = query_model_with_tools(  # 同步HTTP调用。当前线程会等待LLM返回或超时
                        model_str=self.model,
                        messages=messages,
                        tools=tool_schemas,
                        openai_api_key=self.api_key,
                        max_tokens=self.max_tokens,
                        return_on_no_tool_call=True,
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                logger.exception(f"[{self.agent_name}][{self.phase}][step={step}] "f"LLM FAILED elapsed={elapsed:.2f}s")
                prompt_recorder.record({
                    "event": "llm_exchange",
                    "agent": self.agent_name,
                    "phase": self.phase,
                    "round": current_round,
                    "step": step,
                    "messages": request_messages,
                    "tool_schemas": tool_schemas,
                    "error": str(exc),
                    "duration_seconds": round(elapsed, 6),
                })
                raise   # run()在这里终止

            elapsed = time.perf_counter() - started
            returned_calls = [
                {
                    "id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in response.tool_calls
            ]   # 将模型返回的工具调用对象转换成便于记录的字典
            logger.info(
                f"[{self.agent_name}][{self.phase}][step={step}] "
                f"LLM RESPONSE tool_calls={len(returned_calls)} "
                f"names={[call['name'] for call in returned_calls]} "
                f"finish_reason={response.finish_reason or 'None'} "
                f"reasoning_content_chars={response.reasoning_content_chars} "
                f"elapsed={elapsed:.2f}s"
            )
            logger.debug(
                f"[{self.agent_name}][{self.phase}][step={step}] "
                "LLM RESPONSE PAYLOAD\n"
                + json.dumps(
                    {
                        "content": response.content,
                        "tool_calls": returned_calls,
                        "response_metadata": {
                            "response_id": response.response_id,
                            "finish_reason": response.finish_reason,
                            "usage": response.usage,
                            "reasoning_content_chars": response.reasoning_content_chars,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            prompt_recorder.record({
                "event": "llm_exchange",
                "agent": self.agent_name,
                "phase": self.phase,
                "round": current_round,
                "step": step,
                "messages": request_messages,
                "tool_schemas": tool_schemas,
                "response_content": response.content,
                "returned_tool_calls": returned_calls,
                "response_metadata": {
                    "response_id": response.response_id,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                    "reasoning_content_chars": response.reasoning_content_chars,
                },
                "duration_seconds": round(elapsed, 6),
            })

            # 检查是否返回工具调用
            if not response.tool_calls:
                self._append_no_tool_call_rejection(
                    messages,
                    response,
                    active_tool_names,
                    step,
                    current_round,
                )
                continue
            unavailable = sorted({
                call.name
                for call in response.tool_calls
                if call.name not in active_tool_names
            })  # 检查工具当前是否开放
            if unavailable:
                self._append_unavailable_tool_rejection(
                    messages,
                    response,
                    active_tool_names,
                    step,
                    current_round,
                )
                continue
            if len(response.tool_calls) > 1:
                disallowed = [  # 检查这批工具是否都允许批量调用
                    call.name
                    for call in response.tool_calls
                    if not registry.allows_multiple(call.name)
                ]
                if disallowed:  # 如果有工具不允许批量调用
                    protocol_repairs += 1
                    if protocol_repairs > self.max_protocol_repairs:
                        logger.error(
                            f"[{self.agent_name}][{self.phase}][step={step}] "
                            "PROTOCOL ERROR repair budget exhausted "
                            f"non-batchable tools={disallowed}"
                        )
                        raise ToolProtocolError(
                            f"{self.agent_name} exceeded its "
                            f"{self.max_protocol_repairs} protocol-repair budget "
                            f"after repeatedly returning multiple calls containing "
                            f"non-batchable tools: {disallowed}"
                        )
                    self._append_non_batchable_batch_rejection(
                        messages,
                        response,
                        disallowed,
                        step,
                        protocol_repairs,
                    )   # 向 messages 写入“本批被拒绝，请下一次只调用一个工具”的反馈。
                    continue


            # 解析和校验参数
            parsed_calls = []
            missing_by_call_id = {}
            invalid_json_by_call_id = {}
            for call in response.tool_calls:
                try:
                    arguments = self._parse_arguments(call)
                except ToolProtocolError as exc:
                    invalid_json_by_call_id[call.call_id] = str(exc)
                    continue
                parsed_calls.append((call, arguments))  # 保存原始工具调用 + 已解析参数
                missing = registry.missing_required_arguments(call.name, arguments)# 根据工具 Schema 检查必填参数是否齐全
                if missing:
                    missing_by_call_id[call.call_id] = missing  # 按调用ID保存缺失参数

            if invalid_json_by_call_id:
                self._append_invalid_json_rejection(
                    messages,
                    response,
                    invalid_json_by_call_id,
                    step,
                    current_round,
                )
                continue

            # 修复缺少参数的调用
            if missing_by_call_id:
                protocol_repairs += 1
                # 超过修复预算就抛出 ToolProtocolError
                if protocol_repairs > self.max_protocol_repairs:
                    logger.error(
                        f"[{self.agent_name}][{self.phase}][step={step}] "
                        "PROTOCOL ERROR repair budget exhausted "
                        f"missing_arguments={missing_by_call_id}"
                    )
                    raise ToolProtocolError(
                        f"{self.agent_name} exceeded its "
                        f"{self.max_protocol_repairs} protocol-repair budget after "
                        "repeatedly returning tool calls with missing required "
                        f"arguments: {missing_by_call_id}"
                    )
                # 把错误反馈加入对话，告诉模型缺少哪些参数。本批工具一个都不会执行，然后重新请求模型。
                self._append_missing_argument_rejection(
                    messages,
                    response,
                    missing_by_call_id,
                    step,
                    protocol_repairs,
                )
                continue


            protocol_repairs = 0    # 本轮协议校验全部通过，因此清零连续修复计数
            # 处理多个工具调用
            if len(response.tool_calls) > 1:
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        self._assistant_tool_call(call)
                        for call in response.tool_calls
                    ],
                })# 把模型发出的整批工具调用加入对话历史。
                batch_results = []  # 准备保存本批执行结果
                logger.info(
                    f"[{self.agent_name}][{self.phase}][step={step}] "
                    f"TOOL BATCH count={len(response.tool_calls)} mode=sequential"
                )
                for call, arguments in parsed_calls:
                    value = self._execute_tool(registry, call, arguments, step)
                    batch_results.append({
                        "tool_name": call.name,
                        "arguments": arguments,
                        "value": value,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": self._tool_output(value),
                    })  # 随后把每个结果加入 batch_results，并以 role="tool" 追加到对话历史。
                with trace_scope(
                    agent=self.agent_name,
                    phase=self.phase,
                    step=step,
                    trigger="framework_automatic",
                ):
                    record_trace_event(
                        category="tool_protocol",
                        name="agent.executed_multi_tool_batch",
                        success=True,
                        metadata={
                            "agent": self.agent_name,
                            "phase": self.phase,
                            "tool_count": len(response.tool_calls),
                            "tools": [
                                result["tool_name"] for result in batch_results
                            ],
                            "execution_mode": "sequential",
                        },
                    )
                step += 1
                continue

            # 处理单个工具调用
            tool_call, arguments = parsed_calls[0]  # 取出唯一工具调用和参数
            assistant_message = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [self._assistant_tool_call(tool_call)],
            }   
            messages.append(assistant_message)  # 把模型发出的单个工具调用加入对话历史。

            value = self._execute_tool(
                registry, tool_call, arguments, step
            )

            tool_failed = isinstance(value, dict) and value.get("success") is False # 只有返回字典且明确包含 "success": False 才算失败
            if tool_failed:
                compacted = self._compact_failed_arguments(arguments)   # 失败时检查参数是否过长。超过 2000 个序列化字符时，会把大段代码、列表等替换成摘要和哈希。避免失败的大参数在后续对话里重复占用上下文。
                if compacted is not None:
                    assistant_message["tool_calls"][0]["function"]["arguments"] = (
                        json.dumps(compacted, ensure_ascii=True, default=str)
                    )
                    logger.info(
                        f"[{self.agent_name}][{self.phase}][step={step}] "
                        f"HISTORY COMPACTED failed tool={tool_call.name} "
                        f"original_argument_chars="
                        f"{len(json.dumps(arguments, ensure_ascii=True, default=str))}"
                    )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": self._tool_output(value),
            })  # 把工具结果加入对话

            # 如果连续发生完全相同的失败，就累加次数；否则重新计数
            if tool_failed:
                failed_signature = self._failed_call_signature(
                    tool_call.name,
                    arguments,
                    value,
                )
                if failed_signature == last_failed_call_signature:
                    identical_failed_call_count += 1
                else:
                    last_failed_call_signature = failed_signature
                    identical_failed_call_count = 1
                if identical_failed_call_count >= MAX_IDENTICAL_FAILED_TOOL_CALLS:
                    logger.error(
                        f"[{self.agent_name}][{self.phase}][step={step}] "
                        "FAILED identical failed tool-call limit "
                        f"tool={tool_call.name} count={identical_failed_call_count} "
                        f"signature={failed_signature[:12]}"
                    )
                    raise ToolProtocolError(
                        f"{self.agent_name} repeated the same failed tool call "
                        f"{tool_call.name!r} {identical_failed_call_count} times "
                        f"(signature={failed_signature[:12]})"
                    )
            else:
                last_failed_call_signature = None
                identical_failed_call_count = 0
            # 检查工具调用是否成功
            terminal_success = not (
                isinstance(value, dict) and value.get("success") is False
            )

            # 当前工具属于终止工具
            if return_after_tool or (tool_call.name in terminal_tools and terminal_success):
                logger.info(
                    f"[{self.agent_name}][{self.phase}] "
                    f"{'TURN END' if return_after_tool else 'COMPLETE'} "
                    f"terminal_tool={tool_call.name}"
                )
                return AgentRunResult(
                    tool_name=tool_call.name,
                    arguments=arguments,
                    value=value,
                    messages=messages,
                )   
            step += 1
        if round_index >= self.max_rounds:
            logger.error(
                f"[{self.agent_name}][{self.phase}] FAILED LLM round budget exhausted"
            )
            raise ToolProtocolError(
                f"{self.agent_name} exceeded its {self.max_rounds} LLM-round budget"
            )
        logger.error(f"[{self.agent_name}][{self.phase}] FAILED tool budget exhausted")
        raise ToolProtocolError(
            f"{self.agent_name} exceeded its {self.max_tool_steps} tool-call budget"
        )

    def run_one(self, system_prompt, user_prompt, registry):
        return self.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            registry=registry,
            terminal_tools=registry.names,
        )
