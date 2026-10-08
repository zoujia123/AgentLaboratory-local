import json
import multiprocessing
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools import execute_code, _wait_for_execution, format_execution_text
from telemetry import TraceRecorder, trace_scope


INSPECTED_DATASET = {
    "dataset_id": "owner/data", "revision": "r1",
    "config_name": "default", "split": "train",
    "dataset_schema": {"columns": {"label": "int64"}},
}


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

    def test_failure_context_explains_dataset_dict_unpacking(self):
        setup = (
            "from datasets import Dataset\n"
            "dataset = Dataset.from_dict({'text': ['one', 'two', 'three', 'four']})\n"
            "splits = dataset.train_test_split(test_size=0.5, seed=SKLEARN_SEED)\n"
        )
        failing_line = "print(train_dataset[0]['text'])"
        code = setup + "train_dataset, held_out_dataset = splits\n" + failing_line

        result = self.run_code(code)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "python_exception")
        self.assertIn("string indices must be integers", result["error"])
        context = result["failure_context"]
        self.assertEqual(context["line"], code.splitlines().index(failing_line) + 1)
        self.assertIn(failing_line, context["source"])
        self.assertEqual(context["variables"]["train_dataset"], {"type": "str", "length": 5})
        self.assertNotIn("dataset", context["variables"])
        self.assertNotIn("splits", context["variables"])

        repaired = self.run_code(
            setup + "train_dataset = splits['train']\n"
            "held_out_dataset = splits['test']\n"
            "assert len(train_dataset) == len(held_out_dataset) == 2\n"
            "assert isinstance(train_dataset[0]['text'], str)\n"
            "print('split access repaired')"
        )
        self.assertTrue(repaired["success"], repaired["error"])
        self.assertIn("split access repaired", repaired["output"])
        self.assertNotIn("failure_context", repaired)

    def test_failure_context_uses_innermost_dataset_callback_frame(self):
        code = textwrap.dedent("""\
            import re
            from datasets import Dataset
            dataset = Dataset.from_dict({'text': ['one', 'two']})
            def remove_urls(text):
                return re.sub(r'https?://\\S+', '', text)
            dataset.map(lambda batch: {'clean': remove_urls(batch['text'])}, batched=True)
        """)

        result = self.run_code(code)

        self.assertFalse(result["success"])
        self.assertIn("expected string or bytes-like object", result["error"])
        context = result["failure_context"]
        self.assertEqual(context["line"], 5)
        self.assertIn("return re.sub", context["source"])
        self.assertEqual(context["variables"]["text"], {"type": "list", "length": 2})
        self.assertNotIn("dataset", context["variables"])
        self.assertNotIn("batch", context["variables"])

    def test_failure_context_uses_innermost_lambda_frame(self):
        code = (
            "import re\n"
            "clean = lambda text: re.sub(r'https?://\\S+', '', text)\n"
            "clean(['one', 'two'])\n"
        )

        result = self.run_code(code)

        self.assertFalse(result["success"])
        context = result["failure_context"]
        self.assertEqual(context["line"], 2)
        self.assertIn("lambda text", context["source"])
        self.assertEqual(context["variables"]["text"], {"type": "list", "length": 2})

    def test_failure_context_does_not_inspect_objects_or_expand_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory, "unexpected_introspection.txt")
            code = textwrap.dedent(f"""\
                from pathlib import Path
                def record_access(name):
                    with open({str(marker)!r}, 'a', encoding='utf-8') as marker_file:
                        marker_file.write(name + '\\n')
                class Guard:
                    def __repr__(self):
                        record_access('repr')
                        return 'guard'
                    def __len__(self):
                        record_access('len')
                        return 99
                    @property
                    def shape(self):
                        record_access('shape')
                        return (99,)
                    @property
                    def dtype(self):
                        record_access('dtype')
                        return 'object'
                guard = Guard()
                huge_rows = ['PRIVATE_ROW_CONTENT'] * 100000
                unrelated_secret = 'UNRELATED_NAMESPACE_CONTENT'
                (guard, huge_rows)[2]
            """)

            result = self.run_code(code)

            self.assertFalse(result["success"])
            self.assertIn("IndexError", result["error"])
            context = result["failure_context"]
            self.assertEqual(set(context["variables"]), {"guard", "huge_rows"})
            self.assertIn("Guard", context["variables"]["guard"]["type"])
            self.assertEqual(set(context["variables"]["guard"]), {"type"})
            self.assertEqual(context["variables"]["huge_rows"], {"type": "list", "length": 100000})
            serialized = json.dumps(context)
            self.assertNotIn("PRIVATE_ROW_CONTENT", serialized)
            self.assertNotIn("UNRELATED_NAMESPACE_CONTENT", serialized)
            self.assertLess(len(serialized), 1000)
            self.assertFalse(marker.exists(), "Collecting context called a user object method")

    def test_failure_context_reads_variables_from_multiline_expression(self):
        code = textwrap.dedent("""\
            train_dataset = 'train'
            field = 'text'
            email = (
                train_dataset[0][
                    field
                ]
            )
        """)

        result = self.run_code(code)

        self.assertFalse(result["success"])
        self.assertIn("string indices must be integers", result["error"])
        context = result["failure_context"]
        self.assertEqual(context["line"], 4)
        self.assertIn("train_dataset[0]", context["source"])
        self.assertIn("field", context["source"])
        self.assertEqual(context["variables"]["train_dataset"], {"type": "str", "length": 5})
        self.assertEqual(context["variables"]["field"], {"type": "str", "length": 4})
        self.assertNotIn("email", context["variables"])

    def test_large_result_drained_before_worker_join(self):
        result = self.run_code("print('x' * 400000)", MAX_LEN=400001)
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(len(result["output"]), 400001)

    def test_structured_preparation_result_and_legacy_text_are_separate(self):
        from data_preparation import _run_preparation_code
        code = "print('before error')\nraise KeyError('label')"
        execution = self.run_code(code)
        with patch("data_preparation.execute_code", return_value=execution):
            result = _run_preparation_code(code, None, INSPECTED_DATASET, code_execution_timeout_seconds=17.5)
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

    def test_preparation_scope_rejects_model_training_before_execution(self):
        from data_preparation import _run_preparation_code

        code = (
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.metrics import accuracy_score\n"
            "model = RandomForestClassifier()\n"
            "model.fit(X_train, y_train)\n"
            "accuracy_score(y_test, model.predict(X_test))\n"
        )
        with patch("data_preparation.execute_code") as execute:
            result = _run_preparation_code(code, ".", INSPECTED_DATASET, code_execution_timeout_seconds=17.5)

        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "data_preparation_scope_violation")
        self.assertIn("RandomForestClassifier", result["error"])
        self.assertIn("accuracy_score", result["error"])
        execute.assert_not_called()

    def test_preparation_scope_allows_preprocessing(self):
        from data_preparation import _run_preparation_code

        details = {
            "success": True,
            "error_type": None,
            "error": None,
            "traceback": "",
            "exit_code": 0,
            "output": "prepared\n",
            "replay_events": [],
            "artifacts": [],
            "prepared_data_manifest": None,
            "executed_dataset": {
                "dataset_id": "owner/data", "revision": "r1",
                "resources": [{"config_name": "default", "split": "train"}],
            },
        }
        code = (
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
            "X_train = scaler.fit_transform(X_train)\n"
        )
        with patch("data_preparation.execute_code", return_value=details) as execute:
            result = _run_preparation_code(code, ".", INSPECTED_DATASET, code_execution_timeout_seconds=17.5)

        self.assertTrue(result["success"])
        execute.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["expected_hf_dataset"], INSPECTED_DATASET)
        self.assertEqual(execute.call_args.kwargs["timeout"], 17.5)

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
