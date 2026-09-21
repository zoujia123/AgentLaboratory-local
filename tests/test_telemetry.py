import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from inference import query_model_with_tools
from telemetry import TraceRecorder, trace_scope, traced_call
from tools import execute_code


class TelemetryIntegrationTest(unittest.TestCase):
    def test_records_nested_tool_and_llm_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(temp_dir, run_id="test_run")

            @traced_call("tool", "test.parent_tool")
            def parent_tool():
                return execute_code("print(2 + 3)", timeout=10)

            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="",
                tool_calls=[SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(name="submit", arguments='{"value":"ok"}'),
                )],
            ))])
            client = MagicMock()
            client.chat.completions.create.return_value = response

            with trace_scope(
                recorder=recorder,
                agent="TestAgent",
                phase="test phase",
                step=2,
                trigger="agent_requested",
            ):
                self.assertEqual(parent_tool().strip(), "5")
                with patch.dict(os.environ, {
                    "OPENAI_BASE_URL": "https://example.invalid/v1",
                    "OPENAI_MODEL": "test-model",
                }), patch("inference.OpenAI", return_value=client):
                    tool_response = query_model_with_tools(
                            model_str="requested-model",
                            messages=[{"role": "user", "content": "prompt"}],
                            tools=[{
                                "type": "function",
                                "function": {
                                    "name": "submit",
                                    "description": "Submit",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                        "required": ["value"],
                                        "additionalProperties": False,
                                    },
                                },
                            }],
                            openai_api_key="test-key",
                            tries=1,
                        )
                    self.assertEqual(tool_response.tool_calls[0].name, "submit")
                    self.assertFalse(
                        client.chat.completions.create.call_args.kwargs[
                            "parallel_tool_calls"
                        ]
                    )
                    self.assertEqual(
                        client.chat.completions.create.call_args.kwargs[
                            "tool_choice"
                        ],
                        {
                            "type": "function",
                            "function": {"name": "submit"},
                        },
                    )

            summary = recorder.finalize()
            self.assertEqual(summary["tool_call_count"], 2)
            self.assertEqual(summary["llm_call_count"], 1)

            trace_path = os.path.join(recorder.run_dir, "trace.jsonl")
            with open(trace_path, "r", encoding="utf-8") as trace_file:
                events = [json.loads(line) for line in trace_file]

            by_name = {event["name"]: event for event in events}
            parent_event = by_name["test.parent_tool"]
            execute_event = by_name["python.execute_code"]
            llm_event = by_name["llm.query_model_with_tools"]

            self.assertEqual(execute_event["parent_call_id"], parent_event["call_id"])
            self.assertEqual(llm_event["agent"], "TestAgent")
            self.assertEqual(llm_event["phase"], "test phase")
            self.assertEqual(llm_event["metadata"]["effective_model"], "test-model")
            self.assertNotIn("prompt", llm_event["metadata"])
            self.assertTrue(os.path.exists(os.path.join(recorder.run_dir, "tool_summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(recorder.run_dir, "llm_summary.csv")))


if __name__ == "__main__":
    unittest.main()
