import json
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from agent_runtime import AgentRuntime, ToolProtocolError, ToolRegistry
from inference import ModelToolCall, ToolModelResponse


def schema(properties=None, required=None):
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


class AgentRuntimeTest(unittest.TestCase):
    def test_compacts_large_failed_tool_arguments_in_next_llm_request(self):
        observed_messages = []
        large_code = "x" * 8000
        registry = ToolRegistry().register(
            "replace",
            "Replace",
            schema({"code": {"type": "string"}}, ["code"]),
            lambda code: {"success": False, "error": "source rejected"},
        ).register(
            "submit",
            "Submit",
            schema(),
            lambda: {"success": True},
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall("call-1", "replace", json.dumps({"code": large_code}))],
            ),
            ToolModelResponse("", [ModelToolCall("call-2", "submit", "{}")]),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=2)
        recorder = MagicMock()
        with patch("agent_runtime.query_model_with_tools", side_effect=respond), patch(
            "agent_runtime.current_prompt_recorder", return_value=recorder
        ):
            result = runtime.run("system", "user", registry, {"submit"})

        self.assertTrue(result.value["success"])
        history_arguments = json.loads(
            observed_messages[1][-2]["tool_calls"][0]["function"]["arguments"]
        )
        self.assertIn("rejected code omitted", history_arguments["code"])
        self.assertNotIn(large_code, json.dumps(observed_messages[1]))
        tool_events = [
            call.args[0]
            for call in recorder.record.call_args_list
            if call.args[0].get("event") == "tool_execution"
        ]
        self.assertEqual(tool_events[0]["arguments"]["code"], large_code)

    def test_stops_after_three_identical_failed_tool_calls(self):
        registry = ToolRegistry().register(
            "replace",
            "Replace",
            schema({"code": {"type": "string"}}, ["code"]),
            lambda code: {"success": False, "error": "source rejected"},
        )
        failed = ToolModelResponse(
            "",
            [ModelToolCall("call", "replace", '{"code":"same"}')],
        )
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=8)
        with patch(
            "agent_runtime.query_model_with_tools",
            side_effect=[failed, failed, failed],
        ) as query:
            with self.assertRaisesRegex(
                ToolProtocolError,
                "repeated the same failed tool call 'replace' 3 times",
            ):
                runtime.run("system", "user", registry, set())

        self.assertEqual(query.call_count, 3)

    def test_changed_code_hash_counts_as_failed_tool_progress(self):
        code_hashes = iter(["version-1", "version-2", "version-3"])
        registry = ToolRegistry().register(
            "repair",
            "Repair",
            schema({}, []),
            lambda: {
                "success": False,
                "error": "source still invalid",
                "code_hash": next(code_hashes),
            },
        ).register(
            "submit",
            "Submit",
            schema({}, []),
            lambda: {"success": True},
        )
        failed = ToolModelResponse("", [ModelToolCall("repair", "repair", "{}")])
        responses = [
            failed,
            failed,
            failed,
            ToolModelResponse("", [ModelToolCall("submit", "submit", "{}")]),
        ]
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=4)

        with patch("agent_runtime.query_model_with_tools", side_effect=responses):
            result = runtime.run("system", "user", registry, {"submit"})

        self.assertTrue(result.value["success"])

    def test_tool_schemas_enable_strict_argument_generation(self):
        registry = ToolRegistry().register(
            "search",
            "Search",
            schema({"query": {"type": "string"}}, ["query"]),
            lambda query: {"success": True, "query": query},
        )

        tool_schema = registry.schemas[0]["function"]

        self.assertIs(tool_schema["strict"], True)
        self.assertEqual(tool_schema["parameters"]["required"], ["query"])

    def test_executes_one_tool_per_round_until_submit(self):
        calls = []
        registry = ToolRegistry()
        registry.register(
            "search",
            "Search",
            schema({"query": {"type": "string"}}, ["query"]),
            lambda query: calls.append(("search", query)) or {"results": ["item"]},
        ).register(
            "submit",
            "Submit",
            schema({"value": {"type": "string"}}, ["value"]),
            lambda value: calls.append(("submit", value)) or {"success": True, "value": value},
        )
        responses = [
            ToolModelResponse(
                content="",
                tool_calls=[ModelToolCall("call-1", "search", json.dumps({"query": "iris"}))],
            ),
            ToolModelResponse(
                content="",
                tool_calls=[ModelToolCall("call-2", "submit", json.dumps({"value": "done"}))],
            ),
        ]
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=2)
        with patch("agent_runtime.query_model_with_tools", side_effect=responses):
            result = runtime.run("system", "user", registry, {"submit"})
        self.assertEqual(calls, [("search", "iris"), ("submit", "done")])
        self.assertEqual(result.value["value"], "done")
        self.assertEqual(result.messages[-1]["role"], "tool")

    def test_dynamic_tool_provider_changes_tools_on_next_request(self):
        ready = {"value": False}
        observed_tools = []
        observed_messages = []

        def add():
            ready["value"] = True
            return {"success": True}

        registry = ToolRegistry().register(
            "search",
            "Search",
            schema({"query": {"type": "string"}}, ["query"]),
            lambda query: {"success": True, "query": query},
        ).register(
            "add",
            "Add",
            schema(),
            add,
        ).register(
            "submit",
            "Submit",
            schema({"review": {"type": "string"}}, ["review"]),
            lambda review: {"success": True, "review": review},
        )
        responses = [
            ToolModelResponse("", [ModelToolCall("call-1", "add", "{}")]),
            ToolModelResponse(
                "",
                [ModelToolCall("call-2", "submit", '{"review":"done"}')],
            ),
        ]

        def respond(**kwargs):
            observed_tools.append([
                item["function"]["name"] for item in kwargs["tools"]
            ])
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_tools) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=2)
        with patch("agent_runtime.query_model_with_tools", side_effect=respond):
            result = runtime.run(
                "system",
                "user",
                registry,
                {"submit"},
                tool_names_provider=lambda: (
                    {"submit"} if ready["value"] else registry.names
                ),
            )

        self.assertEqual(observed_tools[0], ["search", "add", "submit"])
        self.assertEqual(observed_tools[1], ["submit"])
        self.assertEqual(observed_messages[1][-2]["role"], "assistant")
        self.assertEqual(observed_messages[1][-1]["role"], "tool")
        self.assertEqual(result.value["review"], "done")

    def test_automatically_executes_unique_zero_argument_tool(self):
        stage = {"value": "replace"}
        actions = []
        observed_messages = []

        def replace(code):
            actions.append(("replace", code))
            stage["value"] = "execute"
            return {"success": True}

        def execute():
            actions.append(("execute", None))
            stage["value"] = "submit"
            return {"success": True, "output": "ok"}

        registry = ToolRegistry().register(
            "replace",
            "Replace",
            schema({"code": {"type": "string"}}, ["code"]),
            replace,
        ).register(
            "execute",
            "Execute",
            schema(),
            execute,
        ).register(
            "submit",
            "Submit",
            schema(),
            lambda: actions.append(("submit", None)) or {"success": True},
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall("replace-call", "replace", '{"code":"x = 1"}')],
            ),
            ToolModelResponse("", [ModelToolCall("submit-call", "submit", "{}")]),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=3)
        with patch("agent_runtime.query_model_with_tools", side_effect=respond) as query:
            result = runtime.run(
                "system",
                "user",
                registry,
                {"submit"},
                tool_names_provider=lambda: {stage["value"]},
                automatic_tools={"execute"},
            )

        self.assertTrue(result.value["success"])
        self.assertEqual(
            actions,
            [("replace", "x = 1"), ("execute", None), ("submit", None)],
        )
        self.assertEqual(query.call_count, 2)
        automatic_result = json.loads(observed_messages[1][-1]["content"])
        self.assertTrue(automatic_result["automatic_execution"])
        self.assertEqual(observed_messages[1][-2]["tool_calls"][0]["function"]["name"], "execute")

    def test_recovers_from_tool_not_available_in_current_turn(self):
        actions = []
        observed_messages = []
        registry = ToolRegistry().register(
            "search",
            "Search",
            schema(),
            lambda: actions.append("search") or {"success": True},
        ).register(
            "submit",
            "Submit",
            schema(),
            lambda: actions.append("submit") or {"success": True},
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall("call-1", "search", "{}")],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall("call-2", "submit", "{}")],
            ),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)

        with patch(
            "agent_runtime.query_model_with_tools",
            side_effect=respond,
        ):
            result = runtime.run(
                "system",
                "user",
                registry,
                {"submit"},
                tool_names_provider=lambda: {"submit"},
            )

        self.assertTrue(result.value["success"])
        self.assertEqual(actions, ["submit"])
        rejected = json.loads(observed_messages[1][-1]["content"])
        self.assertEqual(rejected["error_type"], "tool_not_available")
        self.assertEqual(rejected["available_tools"], ["submit"])
        self.assertTrue(rejected["retryable"])

    def test_rejects_all_search_calls_then_recovers_with_single_calls(self):
        calls = []
        observed_messages = []
        registry = ToolRegistry()
        registry.register(
            "search",
            "Search",
            schema({"query": {"type": "string"}}, ["query"]),
            lambda query: calls.append(query) or {"results": [query]},
        ).register(
            "submit",
            "Submit",
            schema(),
            lambda: {"success": True},
        )
        responses = [
            ToolModelResponse(
                "",
                [
                    ModelToolCall("call-1", "search", '{"query":"first"}'),
                    ModelToolCall("call-2", "search", '{"query":"second"}'),
                    ModelToolCall("call-3", "search", '{"query":"third"}'),
                ],
            ),
            ToolModelResponse(
                "", [ModelToolCall("call-4", "search", '{"query":"corrected"}')]
            ),
            ToolModelResponse("", [ModelToolCall("call-5", "submit", "{}")]),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=2)
        with patch("agent_runtime.query_model_with_tools", side_effect=respond), patch(
            "agent_runtime.record_trace_event"
        ) as record:
            result = runtime.run("system", "user", registry, {"submit"})

        self.assertTrue(result.value["success"])
        self.assertEqual(calls, ["corrected"])
        second_turn_messages = observed_messages[1]
        assistant_message = second_turn_messages[-4]
        self.assertEqual(len(assistant_message["tool_calls"]), 3)
        self.assertEqual(
            [message["tool_call_id"] for message in second_turn_messages[-3:]],
            ["call-1", "call-2", "call-3"],
        )
        for message in second_turn_messages[-3:]:
            rejected = json.loads(message["content"])
            self.assertFalse(rejected["success"])
            self.assertEqual(rejected["error_type"], "multiple_tool_calls")
        rejection = next(
            call.kwargs
            for call in record.call_args_list
            if call.kwargs.get("name") == "agent.rejected_multiple_tool_calls"
        )
        self.assertEqual(rejection["metadata"]["requested_tools"], ["search"] * 3)
        self.assertEqual(rejection["metadata"]["executed_tools"], [])
        self.assertEqual(len(observed_messages), 3)

    def test_rejects_mixed_tool_calls_without_partially_executing_them(self):
        actions = []
        observed_messages = []
        registry = ToolRegistry()
        registry.register(
            "replace",
            "Replace",
            schema({"code": {"type": "string"}}, ["code"]),
            lambda code: actions.append(("replace", code)) or {"success": True},
        ).register(
            "execute",
            "Execute",
            schema(),
            lambda: actions.append(("execute", None)) or {"success": True},
        ).register(
            "submit",
            "Submit",
            schema(),
            lambda: actions.append(("submit", None)) or {"success": True},
        )
        responses = [
            ToolModelResponse(
                "",
                [
                    ModelToolCall("call-1", "replace", '{"code":"new"}'),
                    ModelToolCall("call-2", "execute", "{}"),
                    ModelToolCall("call-3", "submit", "{}"),
                ],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall("call-4", "submit", "{}")],
            ),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)
        recorder = MagicMock()
        with patch("agent_runtime.query_model_with_tools", side_effect=respond), patch(
            "agent_runtime.current_prompt_recorder", return_value=recorder
        ):
            result = runtime.run("system", "user", registry, {"submit"})

        self.assertTrue(result.value["success"])
        self.assertEqual(actions, [("submit", None)])
        correction_messages = observed_messages[1][-4:]
        self.assertEqual(correction_messages[0]["role"], "assistant")
        self.assertEqual(
            [message["tool_call_id"] for message in correction_messages[1:]],
            ["call-1", "call-2", "call-3"],
        )
        for message in correction_messages[1:]:
            payload = json.loads(message["content"])
            self.assertFalse(payload["success"])
            self.assertEqual(
                payload["error_type"],
                "multiple_tool_calls",
            )
            self.assertIn("No tool was executed", payload["error"])
        recorded_events = [call.args[0] for call in recorder.record.call_args_list]
        recovery = next(
            event
            for event in recorded_events
            if event["event"] == "tool_protocol_recovery"
        )
        self.assertEqual(recovery["repair_attempt"], 1)
        self.assertEqual(recovery["reason"], "multiple_tool_calls")
        self.assertEqual(recovery["executed_tools"], [])

    def test_coalesces_registered_atomic_edit_calls(self):
        actions = []

        def coalesce_edits(argument_batches):
            edits = []
            for arguments in argument_batches:
                if set(arguments) != {"edits"}:
                    return None
                edits.extend(arguments["edits"])
            return {"edits": edits} if len(edits) <= 6 else None

        registry = ToolRegistry().register(
            "edit",
            "Atomically edit code",
            schema({"edits": {"type": "array"}}, ["edits"]),
            lambda edits: actions.append(edits) or {"success": True},
            batch_coalescer=coalesce_edits,
        )
        response = ToolModelResponse(
            "",
            [
                ModelToolCall(
                    f"call-{index}",
                    "edit",
                    json.dumps({"edits": [{
                        "old_text": f"old-{index}",
                        "new_text": f"new-{index}",
                    }]}),
                )
                for index in range(1, 6)
            ],
        )
        recorder = MagicMock()
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)
        with patch(
            "agent_runtime.query_model_with_tools",
            return_value=response,
        ), patch("agent_runtime.current_prompt_recorder", return_value=recorder):
            result = runtime.run("system", "user", registry, {"edit"})

        self.assertTrue(result.value["success"])
        self.assertEqual(result.tool_name, "edit")
        self.assertEqual(len(actions), 1)
        self.assertEqual(
            [edit["old_text"] for edit in actions[0]],
            [f"old-{index}" for index in range(1, 6)],
        )
        events = [call.args[0] for call in recorder.record.call_args_list]
        coalesced = next(event for event in events if event["event"] == "tool_calls_coalesced")
        self.assertEqual(coalesced["source_call_count"], 5)
        self.assertFalse(any(event["event"] == "tool_protocol_recovery" for event in events))

    def test_rejects_multiple_calls_when_registered_coalescer_declines(self):
        actions = []
        coalescer = MagicMock(return_value=None)
        registry = ToolRegistry().register(
            "edit",
            "Atomically edit code",
            schema({"edits": {"type": "array"}}, ["edits"]),
            lambda edits: actions.append(edits) or {"success": True},
            batch_coalescer=coalescer,
        )
        responses = [
            ToolModelResponse(
                "",
                [
                    ModelToolCall("call-1", "edit", '{"edits":["first"]}'),
                    ModelToolCall("call-2", "edit", '{"edits":["second"]}'),
                ],
            ),
            ToolModelResponse(
                "", [ModelToolCall("call-3", "edit", '{"edits":["corrected"]}')]
            ),
        ]
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)
        with patch("agent_runtime.query_model_with_tools", side_effect=responses) as query:
            result = runtime.run_one("system", "user", registry)

        self.assertTrue(result.value["success"])
        self.assertEqual(actions, [["corrected"]])
        self.assertEqual(query.call_count, 2)
        coalescer.assert_called_once_with([
            {"edits": ["first"]}, {"edits": ["second"]},
        ])

    def test_repeated_multiple_calls_exhaust_protocol_repair_budget(self):
        actions = []
        registry = ToolRegistry().register(
            "submit",
            "Submit",
            schema(),
            lambda: actions.append("submit") or {"success": True},
        )
        invalid = ToolModelResponse(
            "",
            [
                ModelToolCall("call-1", "submit", "{}"),
                ModelToolCall("call-2", "submit", "{}"),
            ],
        )
        runtime = AgentRuntime(
            "model",
            "key",
            "TestAgent",
            "test",
            max_tool_steps=1,
            max_protocol_repairs=1,
        )
        with patch(
            "agent_runtime.query_model_with_tools",
            side_effect=[invalid, invalid],
        ) as query:
            with self.assertRaisesRegex(
                ToolProtocolError,
                "exceeded its 1 protocol-repair budget",
            ):
                runtime.run("system", "user", registry, {"submit"})

        self.assertEqual(query.call_count, 2)
        self.assertEqual(actions, [])

    def test_recovers_from_missing_required_arguments_without_using_tool_step(self):
        actions = []
        observed_messages = []
        registry = ToolRegistry().register(
            "submit",
            "Submit",
            schema({"value": {"type": "string"}}, ["value"]),
            lambda value: actions.append(value) or {"success": True, "value": value},
        )
        responses = [
            ToolModelResponse("", [ModelToolCall("call-1", "submit", "{}")]),
            ToolModelResponse(
                "",
                [ModelToolCall("call-2", "submit", '{"value":"done"}')],
            ),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)
        with patch("agent_runtime.query_model_with_tools", side_effect=respond):
            result = runtime.run("system", "user", registry, {"submit"})

        self.assertEqual(actions, ["done"])
        self.assertEqual(result.value["value"], "done")
        recovery = json.loads(observed_messages[1][-1]["content"])
        self.assertEqual(recovery["error_type"], "missing_required_arguments")
        self.assertEqual(recovery["missing_arguments"], ["value"])
        self.assertTrue(recovery["retryable"])
        self.assertIn(
            "Do not repeat the unchanged malformed tool call.",
            recovery["error"],
        )
        self.assertIn(
            "Review the previous successful tool result and choose the next "
            "appropriate tool.",
            recovery["error"],
        )

    def test_missing_required_arguments_exhaust_protocol_repair_budget(self):
        actions = []
        registry = ToolRegistry().register(
            "submit",
            "Submit",
            schema({"value": {"type": "string"}}, ["value"]),
            lambda value: actions.append(value) or {"success": True},
        )
        invalid = ToolModelResponse(
            "",
            [ModelToolCall("call-1", "submit", "{}")],
        )
        runtime = AgentRuntime(
            "model",
            "key",
            "TestAgent",
            "test",
            max_tool_steps=1,
            max_protocol_repairs=1,
        )

        with patch(
            "agent_runtime.query_model_with_tools",
            side_effect=[invalid, invalid],
        ) as query:
            with self.assertRaisesRegex(
                ToolProtocolError,
                "exceeded its 1 protocol-repair budget",
            ):
                runtime.run("system", "user", registry, {"submit"})

        self.assertEqual(query.call_count, 2)
        self.assertEqual(actions, [])

    def test_missing_search_arguments_are_rejected_before_execution(self):
        actions = []
        observed_messages = []
        registry = ToolRegistry().register(
            "search",
            "Search",
            schema({"query": {"type": "string"}}, ["query"]),
            lambda query: actions.append(query) or {"success": True},
        ).register(
            "submit",
            "Submit",
            schema(),
            lambda: actions.append("submit") or {"success": True},
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall("call-1", "search", "{}")],
            ),
            ToolModelResponse(
                "", [ModelToolCall("call-2", "search", '{"query":"corrected"}')]
            ),
            ToolModelResponse("", [ModelToolCall("call-3", "submit", "{}")]),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=2)
        with patch("agent_runtime.query_model_with_tools", side_effect=respond):
            result = runtime.run("system", "user", registry, {"submit"})

        self.assertTrue(result.value["success"])
        self.assertEqual(actions, ["corrected", "submit"])
        recovery_messages = observed_messages[1][-2:]
        self.assertEqual(recovery_messages[0]["role"], "assistant")
        missing_call_result = json.loads(recovery_messages[1]["content"])
        self.assertEqual(
            missing_call_result["error_type"],
            "missing_required_arguments",
        )
        self.assertEqual(missing_call_result["missing_arguments"], ["query"])

    def test_return_after_tool_rejects_multiple_calls_before_executing_one(self):
        actions = []
        registry = ToolRegistry()
        registry.register(
            "read",
            "Read",
            schema({"paper_id": {"type": "string"}}, ["paper_id"]),
            lambda paper_id: actions.append(("read", paper_id)) or {"text": paper_id},
        ).register(
            "add",
            "Add",
            schema({"paper_id": {"type": "string"}}, ["paper_id"]),
            lambda paper_id: actions.append(("add", paper_id)) or {"success": True},
        )
        responses = [
            ToolModelResponse(
                "",
                [
                    ModelToolCall("read-1", "read", '{"paper_id":"p1"}'),
                    ModelToolCall("add-1", "add", '{"paper_id":"p1"}'),
                ],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall("read-2", "read", '{"paper_id":"corrected"}')],
            ),
        ]
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)
        with patch("agent_runtime.query_model_with_tools", side_effect=responses) as query:
            result = runtime.run(
                "system", "user", registry, set(), return_after_tool=True,
            )
        self.assertEqual(result.tool_name, "read")
        self.assertEqual(result.value["text"], "corrected")
        self.assertEqual(actions, [("read", "corrected")])
        self.assertEqual(query.call_count, 2)

    def test_recovers_from_response_without_tool_calls(self):
        observed_messages = []
        registry = ToolRegistry().register(
            "submit", "Submit", schema(), lambda: {"success": True}
        )
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)

        responses = [
            ToolModelResponse("plain text", []),
            ToolModelResponse("", [ModelToolCall("call-1", "submit", "{}")]),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        with patch(
            "agent_runtime.query_model_with_tools",
            side_effect=respond,
        ):
            result = runtime.run_one("system", "user", registry)

        self.assertTrue(result.value["success"])
        recovery_messages = observed_messages[1][-2:]
        self.assertEqual(recovery_messages[0], {
            "role": "assistant",
            "content": "plain text",
        })
        self.assertEqual(recovery_messages[1]["role"], "user")
        self.assertIn("did not call a tool", recovery_messages[1]["content"])
        self.assertIn("submit", recovery_messages[1]["content"])

    def test_recovers_from_invalid_json_arguments(self):
        actions = []
        observed_messages = []
        registry = ToolRegistry().register(
            "submit",
            "Submit",
            schema({"value": {"type": "string"}}, ["value"]),
            lambda value: actions.append(value) or {"success": True, "value": value},
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall("call-1", "submit", '{"value":')],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall("call-2", "submit", '{"value":"done"}')],
            ),
        ]

        def respond(**kwargs):
            observed_messages.append(deepcopy(kwargs["messages"]))
            return responses[len(observed_messages) - 1]

        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=1)
        with patch("agent_runtime.query_model_with_tools", side_effect=respond):
            result = runtime.run_one("system", "user", registry)

        self.assertEqual(actions, ["done"])
        self.assertEqual(result.value["value"], "done")
        rejected = json.loads(observed_messages[1][-1]["content"])
        self.assertEqual(rejected["error_type"], "invalid_json_arguments")
        self.assertTrue(rejected["retryable"])
        self.assertIn("invalid JSON arguments", rejected["error"])

    def test_protocol_recovery_rounds_are_bounded(self):
        registry = ToolRegistry().register(
            "submit", "Submit", schema(), lambda: {"success": True}
        )
        runtime = AgentRuntime(
            "model",
            "key",
            "TestAgent",
            "test",
            max_tool_steps=1,
            max_rounds=2,
        )
        empty = ToolModelResponse("plain text", [])

        with patch(
            "agent_runtime.query_model_with_tools",
            side_effect=[empty, empty],
        ) as query:
            with self.assertRaisesRegex(
                ToolProtocolError,
                "exceeded its 2 LLM-round budget",
            ):
                runtime.run_one("system", "user", registry)

        self.assertEqual(query.call_count, 2)

    def test_failed_terminal_tool_allows_correction(self):
        attempts = []

        def submit(value):
            attempts.append(value)
            return {"success": value == "valid", "value": value}

        registry = ToolRegistry().register(
            "submit",
            "Submit",
            schema({"value": {"type": "string"}}, ["value"]),
            submit,
        )
        responses = [
            ToolModelResponse("", [ModelToolCall("one", "submit", '{"value":"bad"}')]),
            ToolModelResponse("", [ModelToolCall("two", "submit", '{"value":"valid"}')]),
        ]
        runtime = AgentRuntime("model", "key", "TestAgent", "test", max_tool_steps=2)
        with patch("agent_runtime.query_model_with_tools", side_effect=responses):
            result = runtime.run("system", "user", registry, {"submit"})
        self.assertEqual(attempts, ["bad", "valid"])
        self.assertTrue(result.value["success"])


if __name__ == "__main__":
    unittest.main()
