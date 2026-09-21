import json
import multiprocessing
import queue
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools import execute_code, _wait_for_execution, format_execution_text
from telemetry import TraceRecorder, trace_scope


class ExecutionStatusTest(unittest.TestCase):
    def run_code(self, code, **kwargs):
        return execute_code(code, timeout=kwargs.pop("timeout", 30),
                            return_details=True, **kwargs)

    def test_marker_print_and_stderr_are_not_failures(self):
        result = self.run_code(
            "import sys\nprint('[CODE EXECUTION ERROR] is literal text')\n"
            "print('warning only', file=sys.stderr)"
        )
        self.assertTrue(result["success"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("warning only", result["output"])

    def test_long_output_then_exception_preserves_error_and_status(self):
        result = self.run_code("print('x' * 100000)\nraise KeyError('label')", MAX_LEN=1000)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "python_exception")
        self.assertIn("KeyError", result["error"])
        self.assertEqual(result["output"], "x" * 1000)
        self.assertNotIn("Traceback", result["output"])
        self.assertIn('File "<string>", line 2', result["traceback"])
        self.assertTrue(result["output_truncated"])
        self.assertLessEqual(len(result["output"]), 1000)

    def test_large_result_drained_before_worker_join(self):
        result = self.run_code("print('x' * 400000)", MAX_LEN=400001)
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(len(result["output"]), 400001)

    def test_structured_preparation_result_and_legacy_text_are_separate(self):
        from agents import _run_preparation_code
        code = "print('before error')\nraise KeyError('label')"
        result = _run_preparation_code(code, None)
        self.assertFalse(result["success"])
        self.assertEqual(result["output"], "before error\n")
        self.assertIn("KeyError", result["error"])
        self.assertIn('File "<string>", line 2', result["traceback"])
        legacy = execute_code(code, timeout=30)
        self.assertTrue(legacy.startswith("[CODE EXECUTION ERROR]"))
        self.assertEqual(legacy.count("Traceback (most recent call last)"), 1)
        self.assertTrue(legacy.endswith("before error\n"))
        limited = format_execution_text(result, 100)
        self.assertLessEqual(len(limited), 100)
        self.assertIn("KeyError", limited)
        self.assertEqual(result["output"], "before error\n")

    def test_program_printed_diagnostics_are_preserved(self):
        result = self.run_code("print('Traceback user text')\nraise ValueError('bad')")
        self.assertEqual(result["output"], "Traceback user text\n")

    def test_text_without_traceback_preserves_failure_summary(self):
        result = {"success": False, "error_type": "timeout", "error": "limit exceeded",
                  "traceback": "", "output": "partial output"}
        self.assertEqual(format_execution_text(result),
                         "[CODE EXECUTION ERROR]: timeout: limit exceeded\npartial output")

    def test_validation_failure_and_empty_output_limit(self):
        result = self.run_code("if :", MAX_LEN=0)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "validation_error")
        self.assertEqual(result["output"], "")
        self.assertTrue(result["error"])
        self.assertIn("[CODE EXECUTION ERROR]", execute_code("if :", MAX_LEN=0))

    def test_system_exit_is_failed_code_even_with_zero_exit_status(self):
        result = self.run_code("raise SystemExit(0)")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "worker_interrupted")
        self.assertIn("SystemExit", result["error"])

    def test_hard_exit_and_missing_result(self):
        for code, expected in ((7, "process_exit"), (0, "missing_result")):
            with self.subTest(code=code):
                result = self.run_code(f"import os\nos._exit({code})")
                self.assertFalse(result["success"])
                self.assertEqual(result["error_type"], expected)
                self.assertEqual(result["exit_code"], code)

    def test_timeout_cleans_up_child(self):
        before = {p.pid for p in multiprocessing.active_children()}
        result = self.run_code("while True: pass", timeout=0.5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "timeout")
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_invalid_queue_payload_and_transport_failure(self):
        process = MagicMock()
        process.exitcode = 0
        process.is_alive.return_value = False
        for payload in ({}, {"success": True, "output": ""}, None):
            received = MagicMock()
            received.get.return_value = payload
            result = _wait_for_execution(process, received, time.monotonic() + 2)
            self.assertFalse(result["success"])
            self.assertEqual(result["error_type"], "invalid_result")
        received = MagicMock()
        received.get.side_effect = EOFError("closed")
        result = _wait_for_execution(process, received, time.monotonic() + 2)
        self.assertEqual(result["error_type"], "result_transport_error")

    def test_parent_interrupt_propagates_and_cleans_up(self):
        process = MagicMock()
        process.pid = 123
        process.is_alive.side_effect = [True, False, False]
        output_queue = MagicMock()
        with patch("tools.multiprocessing.Process", return_value=process), patch(
            "tools.multiprocessing.Queue", return_value=output_queue
        ), patch("tools._wait_for_execution", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_code("pass")
        process.terminate.assert_called_once()
        output_queue.close.assert_called_once()

    def test_start_failure_is_not_success(self):
        process = MagicMock()
        process.pid = None
        process.start.side_effect = OSError("cannot start")
        with patch("tools.multiprocessing.Process", return_value=process):
            result = self.run_code("pass")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "execution_environment_error")

    def test_trace_uses_status_even_for_legacy_string_return(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = TraceRecorder(folder, run_id="status")
            with trace_scope(recorder=recorder, agent="test", phase="test"):
                output = execute_code("print('[CODE EXECUTION ERROR] literal')", timeout=30)
            recorder.finalize()
            self.assertIsInstance(output, str)
            events = [json.loads(line) for line in
                      Path(recorder.run_dir, "trace.jsonl").read_text().splitlines()]
            event = next(item for item in events if item.get("name") == "python.execute_code")
            self.assertTrue(event["success"])


if __name__ == "__main__":
    unittest.main()
