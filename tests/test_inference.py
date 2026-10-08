import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from inference import (
    LLMHTTPAttemptLimitError,
    ToolResponseError,
    llm_http_attempt_budget_scope,
    query_model_with_tools,
)
from run_logging import PromptRecorder, RunLogger, run_log_scope


def successful_completion():
    return SimpleNamespace(
        id="completion-1",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
            prompt_tokens_details=SimpleNamespace(cached_tokens=10),
        ),
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(
                content="",
                reasoning_content="internal reasoning",
                tool_calls=[SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="submit",
                        arguments='{"value":"ok"}',
                    ),
                )],
            ),
        )],
    )


def empty_tool_completion():
    return SimpleNamespace(
        id="completion-empty",
        usage=SimpleNamespace(
            prompt_tokens=90,
            completion_tokens=30,
            total_tokens=120,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
            prompt_tokens_details=None,
        ),
        choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(
                content="",
                reasoning_content="unfinished reasoning",
                tool_calls=[],
            ),
        )],
    )


class InferenceRetryTest(unittest.TestCase):
    def setUp(self):
        self.tools = [{
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
        }]

    def test_empty_choices_retry_then_success_without_changing_request(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(id="empty-choices", choices=[]),
            successful_completion(),
        ]
        messages = [{"role": "user", "content": "prompt"}]
        with llm_http_attempt_budget_scope(2) as budget, patch(
            "inference.OpenAI", return_value=client
        ), patch("inference.time.sleep") as sleep:
            response = query_model_with_tools(
                "model", messages, self.tools, openai_api_key="key",
                tries=2, retry_delay=0.25,
            )
        self.assertEqual(response.tool_calls[0].name, "submit")
        self.assertEqual(budget.used, 2)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        sleep.assert_called_once_with(0.25)
        for call in client.chat.completions.create.call_args_list:
            self.assertEqual(call.kwargs["messages"], messages)

    def test_malformed_choices_and_messages_are_not_valid_empty_tool_responses(self):
        malformed = [
            (SimpleNamespace(), "choices must be"),
            (SimpleNamespace(choices=None), "choices must be"),
            (SimpleNamespace(choices={}), "choices must be"),
            (SimpleNamespace(choices="invalid"), "choices must be"),
            (SimpleNamespace(choices=[]), "has no choices"),
            (SimpleNamespace(choices=[None]), "first choice has no message"),
            (SimpleNamespace(choices=[{}]), "first choice has no message"),
            (SimpleNamespace(choices=[{"message": None}]), "invalid message"),
            (SimpleNamespace(choices=[{"message": []}]), "invalid message"),
            (SimpleNamespace(choices=[{"message": {}}]), "invalid message"),
            (SimpleNamespace(choices=[{"message": {"content": []}}]), "content must be"),
        ]
        for completion, error in malformed:
            with self.subTest(completion=completion):
                client = MagicMock()
                client.chat.completions.create.return_value = completion
                with patch("inference.OpenAI", return_value=client), patch(
                    "inference.time.sleep"
                ) as sleep:
                    with self.assertRaisesRegex(RuntimeError, error) as caught:
                        query_model_with_tools(
                            "model", [], self.tools, openai_api_key="key",
                            tries=1, return_on_no_tool_call=True,
                        )
                self.assertIsInstance(caught.exception.__cause__, ToolResponseError)
                sleep.assert_not_called()

    def test_tuple_choices_and_valid_text_message_preserve_no_tool_handling(self):
        client = MagicMock()
        client.chat.completions.create.return_value = {
            "id": "text-only",
            "choices": ({
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "finished", "tool_calls": []},
            },),
        }
        with patch("inference.OpenAI", return_value=client), patch(
            "inference.time.sleep"
        ) as sleep:
            response = query_model_with_tools(
                "model", [], self.tools, openai_api_key="key",
                return_on_no_tool_call=True,
            )
        self.assertEqual(response.content, "finished")
        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        sleep.assert_not_called()

    def test_repeated_empty_choices_stop_at_retry_limit_with_cause_and_traceback(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(id="empty-choices", choices=[])
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_018", "job_018", console=False)
            prompts = PromptRecorder(temp_dir, "query_018", "job_018")
            with run_log_scope(logger, prompts), llm_http_attempt_budget_scope(10) as budget, patch(
                "inference.OpenAI", return_value=client
            ), patch("inference.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "failed after 3 attempts: Model response has no choices") as caught:
                    query_model_with_tools("model", [], self.tools, openai_api_key="key")
            logger.close()
            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(budget.used, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertIsInstance(caught.exception.__cause__, ToolResponseError)
        self.assertIn("stage=parse_response", log_text)
        self.assertIn("Traceback (most recent call last)", log_text)
        self.assertIn('"choices_count": 0', log_text)

    def test_sdk_index_error_is_logged_as_request_failure_without_stale_response(self):
        client = MagicMock()
        sdk_error = IndexError("SDK list index out of range")
        client.chat.completions.create.side_effect = [
            SimpleNamespace(id="earlier-malformed", choices=[]), sdk_error,
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_018", "job_018", console=False)
            prompts = PromptRecorder(temp_dir, "query_018", "job_018")
            with run_log_scope(logger, prompts), patch(
                "inference.OpenAI", return_value=client
            ), patch("inference.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "SDK list index out of range") as caught:
                    query_model_with_tools("model", [], self.tools, openai_api_key="key", tries=2)
            logger.close()
            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
        self.assertIs(caught.exception.__cause__, sdk_error)
        self.assertIn("stage=request error=SDK list index out of range", log_text)
        self.assertIn("IndexError: SDK list index out of range", log_text)
        self.assertNotIn("[attempt=2/2] RESPONSE STRUCTURE", log_text)

    def test_response_structure_summary_is_bounded_and_omits_payload_and_credentials(self):
        secret = "sensitive-api-key"
        payload = "private model response " * 1000
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            id=secret + "x" * 4000,
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=[payload], tool_calls=[]),
                finish_reason="stop",
            )],
            headers={"Authorization": secret},
            error={"message": payload},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_018", "job_018", console=False)
            prompts = PromptRecorder(temp_dir, "query_018", "job_018")
            with run_log_scope(logger, prompts), patch("inference.OpenAI", return_value=client):
                with self.assertRaises(RuntimeError):
                    query_model_with_tools("model", [], self.tools, openai_api_key=secret, tries=1)
            logger.close()
            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
        summary_line = next(line for line in log_text.splitlines() if "RESPONSE STRUCTURE " in line)
        summary_text = summary_line.split("RESPONSE STRUCTURE ", 1)[1]
        self.assertLessEqual(len(summary_text), 2000)
        summary = json.loads(summary_text)
        self.assertEqual(summary["choices_count"], 1)
        self.assertEqual(summary["content_type"], "list")
        self.assertNotIn(secret, log_text)
        self.assertNotIn(payload, log_text)
        self.assertNotIn("Authorization", log_text)

    def test_malformed_response_retry_respects_http_budget(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[])
        with llm_http_attempt_budget_scope(1) as budget, patch(
            "inference.OpenAI", return_value=client
        ), patch("inference.time.sleep") as sleep:
            with self.assertRaises(LLMHTTPAttemptLimitError):
                query_model_with_tools("model", [], self.tools, openai_api_key="key", tries=3)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(budget.used, 1)
        self.assertTrue(budget.request_blocked)
        sleep.assert_not_called()

    def test_logs_failed_attempt_then_successful_retry(self):
        first_client = MagicMock()
        first_client.chat.completions.create.side_effect = TimeoutError("slow provider")
        second_client = MagicMock()
        second_client.chat.completions.create.return_value = successful_completion()

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts), patch(
                "inference.OpenAI", side_effect=[first_client, second_client]
            ) as openai, patch("inference.time.sleep") as sleep:
                response = query_model_with_tools(
                    "model",
                    [{"role": "user", "content": "prompt"}],
                    self.tools,
                    openai_api_key="key",
                    tries=2,
                    request_timeout=12.0,
                    retry_delay=0.25,
                )
            logger.close()

            self.assertEqual(response.tool_calls[0].name, "submit")
            self.assertEqual(response.finish_reason, "tool_calls")
            self.assertEqual(response.response_id, "completion-1")
            self.assertEqual(response.usage["prompt_tokens"], 100)
            self.assertEqual(response.usage["reasoning_tokens"], 5)
            self.assertEqual(response.reasoning_content_chars, 18)
            self.assertEqual(openai.call_count, 2)
            for openai_call in openai.call_args_list:
                self.assertEqual(openai_call.kwargs["max_retries"], 0)
                self.assertEqual(openai_call.kwargs["timeout"], 12.0)
            expected_choice = {
                "type": "function",
                "function": {"name": "submit"},
            }
            self.assertEqual(
                first_client.chat.completions.create.call_args.kwargs["tool_choice"],
                expected_choice,
            )
            self.assertEqual(
                second_client.chat.completions.create.call_args.kwargs["tool_choice"],
                expected_choice,
            )
            sleep.assert_called_once_with(0.25)
            for client in (first_client, second_client):
                self.assertEqual(
                    client.chat.completions.create.call_args.kwargs["extra_body"],
                    {"enable_thinking": False},
                )
            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
            self.assertIn("[LLM HTTP][attempt=1/2] BEGIN", log_text)
            self.assertIn(
                "[LLM HTTP][attempt=1/2] FAILED error_type=TimeoutError",
                log_text,
            )
            self.assertIn("error=slow provider", log_text)
            self.assertIn("[LLM HTTP][attempt=1/2] RETRYING after=0.25s", log_text)
            self.assertIn("[LLM HTTP][attempt=2/2] BEGIN", log_text)
            self.assertIn("[LLM HTTP][attempt=2/2] SUCCESS", log_text)
            self.assertIn("finish_reason=tool_calls", log_text)
            self.assertIn("reasoning_content_chars=18", log_text)

    def test_empty_tool_response_retries_then_succeeds(self):
        first_client = MagicMock()
        first_client.chat.completions.create.return_value = empty_tool_completion()
        second_client = MagicMock()
        second_client.chat.completions.create.return_value = successful_completion()

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts), llm_http_attempt_budget_scope(
                2
            ) as budget, patch(
                "inference.OpenAI", side_effect=[first_client, second_client]
            ) as openai, patch("inference.time.sleep") as sleep:
                response = query_model_with_tools(
                    "model",
                    [{"role": "user", "content": "prompt"}],
                    self.tools,
                    openai_api_key="key",
                    tries=2,
                    retry_delay=0.25,
                )
            logger.close()

            self.assertEqual(response.tool_calls[0].name, "submit")
            self.assertEqual(openai.call_count, 2)
            self.assertEqual(budget.used, 2)
            sleep.assert_not_called()
            repaired_messages = (
                second_client.chat.completions.create.call_args.kwargs["messages"]
            )
            self.assertIn(
                "Return exactly one concise tool call",
                repaired_messages[-1]["content"],
            )
            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
            self.assertIn("finish_reason=length", log_text)
            self.assertIn("reasoning_tokens=30", log_text)
            self.assertIn("reasoning_content_chars=20", log_text)
            self.assertIn("FAILED error_type=ToolResponseLengthError", log_text)
            self.assertIn("RETRYING mode=length_repair", log_text)
            self.assertIn("Model returned no tool calls", log_text)

    def test_empty_tool_response_can_be_returned_to_agent_runtime(self):
        client = MagicMock()
        client.chat.completions.create.return_value = empty_tool_completion()

        with patch("inference.OpenAI", return_value=client) as openai, patch(
            "inference.time.sleep"
        ) as sleep:
            response = query_model_with_tools(
                "model",
                [{"role": "user", "content": "prompt"}],
                self.tools,
                openai_api_key="key",
                tries=3,
                retry_delay=0.25,
                return_on_no_tool_call=True,
            )

        self.assertEqual(response.tool_calls, [])
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(openai.call_count, 1)
        sleep.assert_not_called()

    def test_empty_tool_response_fails_after_all_attempts(self):
        clients = [MagicMock(), MagicMock()]
        for client in clients:
            client.chat.completions.create.return_value = empty_tool_completion()

        with patch("inference.OpenAI", side_effect=clients) as openai, patch(
            "inference.time.sleep"
        ) as sleep:
            with self.assertRaisesRegex(
                RuntimeError,
                "failed after 2 attempts: Model returned no tool calls",
            ):
                query_model_with_tools(
                    "model",
                    [{"role": "user", "content": "prompt"}],
                    self.tools,
                    openai_api_key="key",
                    tries=3,
                    retry_delay=0.0,
                )
        self.assertEqual(openai.call_count, 2)
        sleep.assert_not_called()

    def test_multiple_tools_keep_required_tool_choice(self):
        client = MagicMock()
        client.chat.completions.create.return_value = successful_completion()
        tools = self.tools + [{
            "type": "function",
            "function": {
                "name": "other",
                "description": "Other",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }]

        with patch("inference.OpenAI", return_value=client):
            query_model_with_tools(
                "model",
                [{"role": "user", "content": "prompt"}],
                tools,
                openai_api_key="key",
                tries=1,
            )

        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["tool_choice"],
            "required",
        )

    def test_logs_every_failure_before_raising(self):
        first_client = MagicMock()
        first_client.chat.completions.create.side_effect = ConnectionError("first")
        second_client = MagicMock()
        second_client.chat.completions.create.side_effect = TimeoutError("second")

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts), patch(
                "inference.OpenAI", side_effect=[first_client, second_client]
            ), patch("inference.time.sleep"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "failed after 2 attempts: second",
                ):
                    query_model_with_tools(
                        "model",
                        [{"role": "user", "content": "prompt"}],
                        self.tools,
                        openai_api_key="key",
                        tries=2,
                        request_timeout=12.0,
                        retry_delay=0.0,
                    )
            logger.close()

            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
            self.assertIn(
                "[LLM HTTP][attempt=1/2] FAILED error_type=ConnectionError",
                log_text,
            )
            self.assertIn(
                "[LLM HTTP][attempt=2/2] FAILED error_type=TimeoutError",
                log_text,
            )

    def test_http_attempt_budget_counts_retries(self):
        first_client = MagicMock()
        first_client.chat.completions.create.side_effect = TimeoutError("first")
        second_client = MagicMock()
        second_client.chat.completions.create.return_value = successful_completion()

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts), llm_http_attempt_budget_scope(
                2
            ) as budget, patch(
                "inference.OpenAI", side_effect=[first_client, second_client]
            ), patch("inference.time.sleep"):
                response = query_model_with_tools(
                    "model",
                    [{"role": "user", "content": "prompt"}],
                    self.tools,
                    openai_api_key="key",
                    tries=2,
                    retry_delay=0.0,
                )
            logger.close()

            self.assertEqual(response.tool_calls[0].name, "submit")
            self.assertEqual(budget.used, 2)
            self.assertEqual(budget.remaining, 0)
            self.assertFalse(budget.request_blocked)

    def test_http_attempt_budget_blocks_retry_beyond_limit(self):
        first_client = MagicMock()
        first_client.chat.completions.create.side_effect = TimeoutError("first")
        second_client = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts), llm_http_attempt_budget_scope(
                1
            ) as budget, patch(
                "inference.OpenAI", side_effect=[first_client, second_client]
            ) as openai, patch("inference.time.sleep") as sleep:
                with self.assertRaisesRegex(
                    LLMHTTPAttemptLimitError,
                    "request limit reached: 1/1",
                ):
                    query_model_with_tools(
                        "model",
                        [{"role": "user", "content": "prompt"}],
                        self.tools,
                        openai_api_key="key",
                        tries=2,
                        retry_delay=5.0,
                    )
            logger.close()

            self.assertEqual(openai.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(budget.used, 1)
            self.assertTrue(budget.request_blocked)
            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
            self.assertIn(
                "[LLM HTTP][attempt=2/2] BLOCKED request_limit=1/1",
                log_text,
            )


if __name__ == "__main__":
    unittest.main()
