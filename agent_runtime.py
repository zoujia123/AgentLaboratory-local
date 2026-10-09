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
    batch_coalescer: object = None

    # strict: bool = True 请求服务端严格遵守工具参数结构
    @property
    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
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
        batch_coalescer=None,
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
            batch_coalescer=batch_coalescer,
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

    def batch_coalescer_for(self, name):
        try:
            return self._tools[name].batch_coalescer
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
    def __init__(self,model,api_key,agent_name,phase,*,max_tokens=8192,max_protocol_repairs=3,):
        self.model = model
        self.api_key = api_key
        self.agent_name = agent_name
        self.phase = phase
        self.max_tokens = max(256, int(max_tokens))
        self.max_protocol_repairs = max(0, int(max_protocol_repairs))

    # @staticmethod
    def _tool_output(self, value):
        raw = json.dumps(value, ensure_ascii=True, default=str)
        if self.phase != "literature review" or len(raw) <= 2000:
            return raw

        compact = {
            "success": value.get("success"),
            "arxiv_id": value.get("arxiv_id"),
            "candidate_ids": value.get("candidate_ids"),
            "candidate_count": value.get("candidate_count"),
            "decision_required": value.get("decision_required"),
            "next_action_hint": value.get("next_action_hint"),
            "history_truncated": True,
            "original_chars": len(raw),
            "preview": raw[:4000],
        }
        return json.dumps(compact, ensure_ascii=True, default=str)

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

    def _apply_failure_history_compactor(
        self,
        messages,
        failure_history_compactor,
        tool_call,
        arguments,
        value,
        step,
    ):
        if failure_history_compactor is None:
            return False
        if not isinstance(value, dict) or value.get("success") is not False:
            return False
        compacted_messages = failure_history_compactor(
            tool_call=tool_call,
            arguments=arguments,
            value=value,
        )
        if compacted_messages is None:
            return False
        if (
            not isinstance(compacted_messages, list)
            or not all(isinstance(message, dict) for message in compacted_messages)
        ):
            raise ToolProtocolError(
                "failure_history_compactor must return a list of message objects"
            )
        previous_count = len(messages)
        messages[:] = compacted_messages
        current_run_logger().info(f"[{self.agent_name}][{self.phase}][step={step}] MODEL HISTORY COMPACTED failure_tool={tool_call.name} messages={previous_count}->{len(messages)}")
        current_prompt_recorder().record({
            "event": "failure_history_compacted",
            "agent": self.agent_name,
            "phase": self.phase,
            "step": step,
            "failure_tool": tool_call.name,
            "error_type": value.get("error_type"),
            "code_hash": value.get("code_hash"),
            "previous_message_count": previous_count,
            "message_count": len(messages),
        })
        return True

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

    def _coalesce_tool_calls(self, registry, tool_calls):
        """Return one safe synthetic call when a registered atomic tool can merge it."""
        if len(tool_calls) < 2:
            return None
        names = {call.name for call in tool_calls}
        if len(names) != 1:
            return None
        name = next(iter(names))
        coalescer = registry.batch_coalescer_for(name)
        if coalescer is None:
            return None
        try:
            argument_batches = [self._parse_arguments(call) for call in tool_calls]
            arguments = coalescer(argument_batches)
        except (ToolProtocolError, TypeError, ValueError):
            return None
        if not isinstance(arguments, dict):
            return None
        return ModelToolCall(
            call_id=tool_calls[0].call_id,
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=True, default=str),
        )
    # 记录日志、执行工具、统计耗时、记录结构化事件，再把结果返回给外层运行时。
    def _execute_tool(self, registry, tool_call, arguments, step):
        logger = current_run_logger()   # 取得当前任务的日志器。后续 logger.info 和 logger.debug 会写入该 query 的运行日志。
        recorder = current_prompt_recorder()    # 取得结构化记录器。它会把本次工具执行以 JSON 事件形式保存，供后续查看 prompts.jsonl、复盘工作流或分析失败原因。
        logger.info(f"[{self.agent_name}][{self.phase}][step={step}] TOOL BEGIN {tool_call.name}")   # 工具开始执行
        logger.debug(f"[{self.agent_name}][{self.phase}][step={step}] TOOL ARGUMENTS {tool_call.name}\n"+ json.dumps(arguments, ensure_ascii=False, indent=2, default=str))
        
        started = time.perf_counter()
        value = registry.execute(tool_call.name, arguments)
        elapsed = time.perf_counter() - started
        success = not (
            isinstance(value, dict) and value.get("success") is False
        )
        logger.info(f"[{self.agent_name}][{self.phase}][step={step}] TOOL END {tool_call.name} success={success} elapsed={elapsed:.2f}s")
        logger.debug(f"[{self.agent_name}][{self.phase}][step={step}] TOOL RESULT {tool_call.name}\n" + json.dumps(value, ensure_ascii=False, indent=2, default=str))
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
        logger.info(f"[{self.agent_name}][{self.phase}][step={step}] PROTOCOL RECOVERY reason=no_tool_calls available_tools={available_tools} executed_tools=0")
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
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
        logger.info(f"[{self.agent_name}][{self.phase}][step={step}] PROTOCOL RECOVERY reason=tool_not_available requested_tools={requested_tools} unavailable_tools={unavailable_tools} executed_tools=0")
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
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
            parse_error = invalid_json_by_call_id[call.call_id]
            error = (
                f"Tool {call.name!r} received invalid JSON arguments: "
                f"{parse_error}. Retry with one valid JSON object containing "
                "all required arguments. No tool was executed."
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": self._tool_output({
                    "success": False,
                    "error_type": "invalid_json_arguments",
                    "error": error,
                    "tool_name": call.name,
                    "retryable": True,
                }),
            })
        logger.info(f"[{self.agent_name}][{self.phase}][step={step}] PROTOCOL RECOVERY reason=invalid_json_arguments requested_tools={requested_tools} executed_tools=0")
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
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
                    "requested_tools": requested_tools,
                    "invalid_json_by_call_id": dict(invalid_json_by_call_id),
                    "executed_tools": [],
                },
            )

    def _append_multiple_tool_call_rejection(
        self, messages, response, step, repair_attempt,
    ):
        logger = current_run_logger()
        recorder = current_prompt_recorder()
        requested_tools = [call.name for call in response.tool_calls]
        error = (
            "Multiple tool calls were rejected. No tool was executed and "
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
                    "error_type": "multiple_tool_calls",
                    "error": error,
                    "tool_name": call.name,
                    "requested_tools": requested_tools,
                }),
            })
        logger.info(f"[{self.agent_name}][{self.phase}][step={step}] PROTOCOL RECOVERY attempt={repair_attempt}/{self.max_protocol_repairs} requested_tools={requested_tools} executed_tools=0")
        recorder.record({
            "event": "tool_protocol_recovery",
            "agent": self.agent_name,
            "phase": self.phase,
            "step": step,
            "reason": "multiple_tool_calls",
            "repair_attempt": repair_attempt,
            "max_protocol_repairs": self.max_protocol_repairs,
            "requested_tools": requested_tools,
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
                name="agent.rejected_multiple_tool_calls",
                success=True,
                metadata={
                    "agent": self.agent_name,
                    "phase": self.phase,
                    "repair_attempt": repair_attempt,
                    "requested_tools": requested_tools,
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
            missing = missing_by_call_id[call.call_id]
            error = (
                f"Tool {call.name!r} is missing required arguments: "
                f"{', '.join(missing)}. Retry with all required arguments or "
                "choose a different appropriate tool. No tool was executed. "
                + recovery_guidance
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": self._tool_output({
                    "success": False,
                    "error_type": "missing_required_arguments",
                    "error": error,
                    "tool_name": call.name,
                    "missing_arguments": missing,
                    "retryable": True,
                }),
            })
        logger.info(f"[{self.agent_name}][{self.phase}][step={step}] PROTOCOL RECOVERY attempt={repair_attempt}/{self.max_protocol_repairs} reason=missing_required_arguments requested_tools={requested_tools} executed_tools=0")
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

    def run(self,system_prompt,user_prompt,registry,terminal_tools,tool_names_provider=None,return_after_tool=False,failure_history_compactor=None,stop_on_repeated_failure=True,):
        # Alternating collaborators hand back each tool result, including failures.
        # Other agents retain the default loop-until-success behavior.

        logger = current_run_logger()   
        prompt_recorder = current_prompt_recorder()
        logger.info(f"[{self.agent_name}][{self.phase}] START "f"max_protocol_repairs={self.max_protocol_repairs}")
        logger.debug(f"[{self.agent_name}][{self.phase}] SYSTEM PROMPT\n{system_prompt}")
        logger.debug(f"[{self.agent_name}][{self.phase}] USER PROMPT\n{user_prompt}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]   # messages会保存完整执行上下文,这里是初始化对话历史

        step = 0    # 已经执行的工具轮次
        protocol_repairs = 0    # LLM违反多工具调用规则协议后，已经修复几次
        last_failed_call_signature = None   # 用于检测模型是否反复调用同一个失败工具
        identical_failed_call_count = 0

        terminal_tools = set(terminal_tools)

        while True:
            # 计算当前可用工具
            active_tool_names = set(tool_names_provider() if tool_names_provider is not None else registry.names)   # 当前状态机允许什么工具
            tool_schemas = registry.schemas_for(active_tool_names)  # 从完整注册表里取出当前工具的名称、描述和参数Schema

            logger.info(f"[{self.agent_name}][{self.phase}][step={step}] "f"LLM REQUEST tools={len(tool_schemas)} "f"names={sorted(active_tool_names)}")
            logger.debug(f"[{self.agent_name}][{self.phase}][step={step}] ""LLM REQUEST PAYLOAD\n"+ json.dumps({"messages": messages, "tools": tool_schemas},ensure_ascii=False,indent=2,default=str,))
            
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
                prompt_recorder.record({"event": "llm_exchange","agent": self.agent_name,"phase": self.phase,"step": step,"messages": request_messages,"tool_schemas": tool_schemas,"error": str(exc),"duration_seconds": round(elapsed, 6),})
                
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

            logger.info(f"[{self.agent_name}][{self.phase}][step={step}] "f"LLM RESPONSE tool_calls={len(returned_calls)} "f"names={[call['name'] for call in returned_calls]} "f"finish_reason={response.finish_reason or 'None'} "f"reasoning_content_chars={response.reasoning_content_chars} "f"elapsed={elapsed:.2f}s")
            logger.debug(f"[{self.agent_name}][{self.phase}][step={step}] LLM RESPONSE PAYLOAD\n" + json.dumps({"content": response.content,"tool_calls": returned_calls,"response_metadata": {"response_id": response.response_id,"finish_reason": response.finish_reason,"usage": response.usage,"reasoning_content_chars": response.reasoning_content_chars}}, ensure_ascii=False, indent=2, default=str))
            prompt_recorder.record({"event": "llm_exchange","agent": self.agent_name,"phase": self.phase,"step": step,"messages": request_messages, "tool_schemas": tool_schemas,"response_content": response.content,"returned_tool_calls": returned_calls,"response_metadata": { "response_id": response.response_id,"finish_reason": response.finish_reason, "usage": response.usage,"reasoning_content_chars": response.reasoning_content_chars,},"duration_seconds": round(elapsed, 6),})

            # 检查是否返回工具调用
            if not response.tool_calls:
                self._append_no_tool_call_rejection(messages, response, active_tool_names, step,)
                continue

            unavailable = sorted({call.name for call in response.tool_calls if call.name not in active_tool_names})  # 检查工具当前是否开放
            if unavailable:
                self._append_unavailable_tool_rejection(
                    messages, response, active_tool_names, step,
                )
                continue

            if len(response.tool_calls) > 1:
                original_calls = list(response.tool_calls)
                coalesced_call = self._coalesce_tool_calls(registry,original_calls,)
                if coalesced_call is not None:
                    response.tool_calls = [coalesced_call]
                    logger.info(f"[{self.agent_name}][{self.phase}][step={step}] TOOL CALLS COALESCED tool={coalesced_call.name} count={len(original_calls)}")
                    prompt_recorder.record({"event": "tool_calls_coalesced","agent": self.agent_name,"phase": self.phase,"step": step,"tool_name": coalesced_call.name,"source_tool_call_ids": [call.call_id for call in original_calls],"coalesced_tool_call_id": coalesced_call.call_id,"source_call_count": len(original_calls),})
            
            if len(response.tool_calls) > 1:
                requested_tools = [call.name for call in response.tool_calls]
                protocol_repairs += 1
                if protocol_repairs > self.max_protocol_repairs:
                    logger.error(f"[{self.agent_name}][{self.phase}][step={step}] " "PROTOCOL ERROR repair budget exhausted "f"multiple tool calls={requested_tools}")
                    raise ToolProtocolError(f"{self.agent_name} exceeded its "f"{self.max_protocol_repairs} protocol-repair budget " f"after repeatedly returning multiple tool calls: {requested_tools}")
                self._append_multiple_tool_call_rejection(messages, response, step, protocol_repairs,)
                continue

            # 合并和数量检查通过后，只解析唯一工具调用的参数。
            tool_call = response.tool_calls[0]
            try:
                arguments = self._parse_arguments(tool_call)
            except ToolProtocolError as exc:
                self._append_invalid_json_rejection(messages,response,{tool_call.call_id: str(exc)},step,)
                continue

            # 修复缺少参数的调用
            missing = registry.missing_required_arguments(tool_call.name, arguments)
            if missing:
                missing_by_call_id = {tool_call.call_id: missing}
                protocol_repairs += 1
                # 超过修复预算就抛出 ToolProtocolError
                if protocol_repairs > self.max_protocol_repairs:
                    logger.error(f"[{self.agent_name}][{self.phase}][step={step}] PROTOCOL ERROR repair budget exhausted missing_arguments={missing_by_call_id}")
                    raise ToolProtocolError(f"{self.agent_name} exceeded its {self.max_protocol_repairs} protocol-repair budget after repeatedly returning tool calls with missing required arguments: {missing_by_call_id}")
                # 告诉模型缺少哪些参数，不执行该工具，重新请求模型。
                self._append_missing_argument_rejection( messages,response,missing_by_call_id,step,protocol_repairs,)
                continue


            protocol_repairs = 0    # 本轮协议校验全部通过，因此清零连续修复计数

            # 处理单个工具调用
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
            # Compute the unchanged signature before adding recovery feedback.
            if tool_failed:
                failed_signature = self._failed_call_signature(tool_call.name, arguments, value)
                if failed_signature == last_failed_call_signature:
                    identical_failed_call_count += 1
                else:
                    last_failed_call_signature = failed_signature
                    identical_failed_call_count = 1
                if not stop_on_repeated_failure:
                    value = {**value, "identical_failed_call_count": identical_failed_call_count}
            else:
                last_failed_call_signature = None
                identical_failed_call_count = 0

            if tool_failed:
                compacted = self._compact_failed_arguments(arguments)   # 失败时检查参数是否过长。超过 2000 个序列化字符时，会把大段代码、列表等替换成摘要和哈希。避免失败的大参数在后续对话里重复占用上下文。
                if compacted is not None:
                    assistant_message["tool_calls"][0]["function"]["arguments"] = (
                        json.dumps(compacted, ensure_ascii=True, default=str)
                    )
                    logger.info(f"[{self.agent_name}][{self.phase}][step={step}] HISTORY COMPACTED failed tool={tool_call.name} original_argument_chars={len(json.dumps(arguments, ensure_ascii=True, default=str))}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": self._tool_output(value),
            })  # 把工具结果加入对话
            self._apply_failure_history_compactor(
                messages,
                failure_history_compactor,
                tool_call,
                arguments,
                value,
                step,
            )

            if tool_failed and identical_failed_call_count >= MAX_IDENTICAL_FAILED_TOOL_CALLS:
                if stop_on_repeated_failure:
                    logger.error(f"[{self.agent_name}][{self.phase}][step={step}] FAILED identical failed tool-call limit tool={tool_call.name} count={identical_failed_call_count} signature={failed_signature[:12]}")
                    raise ToolProtocolError(
                        f"{self.agent_name} repeated the same failed tool call "
                        f"{tool_call.name!r} {identical_failed_call_count} times "
                        f"(signature={failed_signature[:12]})"
                    )
                logger.info(
                    f"[{self.agent_name}][{self.phase}][step={step}] "
                    f"CONTINUE repeated failed tool call tool={tool_call.name} "
                    f"count={identical_failed_call_count} signature={failed_signature[:12]}"
                )
            # 检查工具调用是否成功
            terminal_success = not (
                isinstance(value, dict) and value.get("success") is False
            )

            # 当前工具属于终止工具
            if return_after_tool or (tool_call.name in terminal_tools and terminal_success):
                logger.info(f"[{self.agent_name}][{self.phase}] {'TURN END' if return_after_tool else 'COMPLETE'} terminal_tool={tool_call.name}")
                return AgentRunResult(
                    tool_name=tool_call.name,
                    arguments=arguments,
                    value=value,
                    messages=messages,
                )   
            step += 1

    def run_one(self, system_prompt, user_prompt, registry):
        return self.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            registry=registry,
            terminal_tools=registry.names,
        )
