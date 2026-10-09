import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_runtime import AgentRuntime, ToolRegistry
from inference import ModelToolCall, ToolModelResponse
from run_logging import PromptRecorder, RunLogger, phase_log, run_log_scope


class RunLoggingTest(unittest.TestCase):
    def test_writes_phase_and_prompt_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts):
                logger.info("JOB START")
                with phase_log("literature review"):
                    logger.debug("full prompt")
                prompts.record({"event": "example", "prompt": "complete prompt"})
            logger.close()

            workflow_log = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
            self.assertIn("JOB START", workflow_log)
            self.assertIn("BEGIN literature review", workflow_log)
            self.assertIn("END literature review", workflow_log)
            self.assertIn("full prompt", workflow_log)
            prompt_event = json.loads(
                Path(temp_dir, "prompts.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(prompt_event["query_id"], "query_001")
            self.assertEqual(prompt_event["prompt"], "complete prompt")

    def test_agent_runtime_logs_full_exchange_and_tool_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            registry = ToolRegistry().register(
                "submit",
                "Submit",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                lambda value: {"success": True, "value": value},
            )
            response = ToolModelResponse(
                "",
                [ModelToolCall("call-1", "submit", '{"value":"done"}')],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                reasoning_content_chars=7,
                response_id="response-1",
            )
            runtime = AgentRuntime("model", "key", "TestAgent", "test")
            with run_log_scope(logger, prompts), patch(
                "agent_runtime.query_model_with_tools", return_value=response
            ):
                runtime.run("system prompt", "user prompt", registry, {"submit"})
            logger.close()

            workflow_log = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
            self.assertIn("SYSTEM PROMPT\nsystem prompt", workflow_log)
            self.assertIn("USER PROMPT\nuser prompt", workflow_log)
            self.assertIn("LLM RESPONSE tool_calls=1", workflow_log)
            self.assertIn("TOOL BEGIN submit", workflow_log)
            self.assertIn("TOOL END submit success=True", workflow_log)
            events = [
                json.loads(line)
                for line in Path(temp_dir, "prompts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([event["event"] for event in events], [
                "llm_exchange",
                "tool_execution",
            ])
            self.assertEqual(events[0]["messages"][0]["content"], "system prompt")
            self.assertEqual(events[0]["returned_tool_calls"][0]["name"], "submit")
            self.assertEqual(
                events[0]["response_metadata"],
                {
                    "response_id": "response-1",
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "reasoning_content_chars": 7,
                },
            )


if __name__ == "__main__":
    unittest.main()
