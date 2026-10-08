import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_runtime import AgentRuntime
from agent_helpers import _compact_solver_traceback, _process_execution_result
from agents import MLESolverAgent
from data_preparation import _run_preparation_code
from execution_diagnostics import runtime_data_repair_hint
from inference import ModelToolCall, ToolModelResponse
from run_logging import PromptRecorder, RunLogger, run_log_scope


def long_traceback():
    library = "".join(
        f'  File "C:/env/Lib/site-packages/joblib/frame_{i}.py", line {i + 1}, in dispatch\n'
        f'    call_{i}()\n'
        for i in range(35)
    )
    return (
        "joblib.externals.loky.process_executor._RemoteTraceback:\n"
        "Traceback (most recent call last):\n"
        + library
        + "ValueError: original remote cause\n\n"
        "The above exception was the direct cause of the following exception:\n\n"
        "Traceback (most recent call last):\n"
        '  File "C:/project/tools.py", line 445, in worker_run_code\n'
        "    exec(code_str, globals_dict)\n"
        '  File "<string>", line 66, in <module>\n'
        "    grid.fit(X_train, y_train)\n"
        + library
        + "ValueError: Invalid parameter 'C'. Valid parameters are: ['alpha', 'solver'].\n"
    )


class SolverDiagnosticTest(unittest.TestCase):
    def test_compaction_keeps_user_frames_inner_frames_and_causes(self):
        original = long_traceback()
        compact = _compact_solver_traceback(original)
        self.assertLess(len(compact), len(original) // 2)
        for evidence in (
            'File "<string>", line 66', "grid.fit(X_train, y_train)",
            "C:/project/tools.py", "frame_34.py", "original remote cause",
            "direct cause", "Valid parameters are: ['alpha', 'solver']",
        ):
            self.assertIn(evidence, compact)
        self.assertNotIn("frame_0.py", compact)
        self.assertIn("library frames omitted", compact)

    def test_short_and_unrecognized_diagnostics_are_untouched(self):
        for trace in ("", "ValueError: bad", "custom stack format\n" * 200):
            self.assertEqual(_compact_solver_traceback(trace), trace)

    def test_main_and_supplement_archive_before_returning_compact_feedback(self):
        state = SimpleNamespace(
            topic="topic", plan="plan", literature_review="review",
            dataset_selection={}, seeds={}, dataset_code="data = [1]",
            experiment_code="print(data)", experiment_output="1",
            supplementary_experiments=[],
        )
        execution = {
            "success": False, "error_type": "python_exception",
            "error": "could not convert string to float: 'tcp'",
            "traceback": long_traceback(), "exit_code": 0, "output": "printed output",
        }
        original = deepcopy(execution)
        for supplement in (False, True):
            with self.subTest(supplement=supplement), tempfile.TemporaryDirectory() as directory:
                logger = RunLogger(directory, "query", "job", console=False)
                recorder = PromptRecorder(directory, "query", "job")
                try:
                    with run_log_scope(logger, recorder), patch.object(
                        AgentRuntime, "run", return_value=SimpleNamespace(value={})
                    ) as run, patch("agents.execute_code", return_value=execution), patch(
                        "agent_helpers._compact_solver_traceback", wraps=_compact_solver_traceback,
                    ) as compact:
                        agent = MLESolverAgent("model", "key", code_execution_timeout_seconds=123)
                        if supplement:
                            agent.supplement(state, {
                                "objective": "check", "reason": "check",
                                "required_outputs": [], "constraints": [],
                            }, directory, 1)
                        else:
                            agent.solve(state, directory)
                        registry = run.call_args.kwargs["registry"]
                        replace = "write_supplement_code" if supplement else "write_experiment_code"
                        execute = "execute_supplement_code" if supplement else "execute_experiment_code"
                        registry.execute(replace, {"code": "print(data)"})
                        feedback = registry.execute(execute, {})
                        compact.assert_called_once_with(original["traceback"])
                        # Files already contain original diagnosis when the tool returns.
                        records = [json.loads(line) for line in
                                   Path(recorder.path).read_text().splitlines()]
                        diagnostic_events = [
                            row for row in records if row["event"] == "solver_execution_diagnostic"
                        ]
                        self.assertEqual(len(diagnostic_events), 1)
                        event = diagnostic_events[0]
                        self.assertEqual(event["result"]["traceback"], original["traceback"])
                        self.assertEqual(event["code_hash"], feedback["code_hash"])
                        log = Path(logger.path).read_text(encoding="utf-8")
                        self.assertIn("frame_0.py", log)
                        self.assertIn("FULL EXECUTION DIAGNOSTIC", log)
                finally:
                    logger.close()
                self.assertEqual(feedback["failure_category"], "non_numeric_model_input")
                self.assertFalse(feedback["success"])
                self.assertLess(len(feedback["traceback"]), len(original["traceback"]))
                self.assertEqual(feedback["error"], original["error"])
                self.assertEqual(feedback["output"], original["output"])
                self.assertEqual(execution, original)

    def test_data_preparation_retains_original_traceback(self):
        execution = {
            "success": False, "error_type": "python_exception", "error": "bad",
            "traceback": long_traceback(), "exit_code": 0, "output": "",
        }
        with patch("data_preparation.execute_code", return_value=execution):
            result = _run_preparation_code("raise ValueError('bad')", ".", {
                "dataset_id": "owner/data", "revision": "r1",
                "config_name": "default", "split": "train",
                "dataset_schema": {"columns": {"value": "int64"}},
            }, code_execution_timeout_seconds=123)
        self.assertEqual(result["traceback"], execution["traceback"])

    def test_preparation_uses_shared_categories_and_preserves_syntax_fallback(self):
        cases = (
            ("import unavailable_encoder", "python_exception",
             "ModuleNotFoundError: No module named 'unavailable_encoder'",
             "unavailable_dependency"),
            ("pass", "python_exception",
             "AttributeError: 'TiffImageFile' object has no attribute 'shape'",
             "image_object_api_error"),
            ("pass", "python_exception", "KeyError: 'derived_feature'",
             "missing_key_or_column"),
            ("pass", "python_exception", "IndexError: positional indexers are out-of-bounds",
             "pandas_positional_indexing_error"),
            ("pass", "timeout", "ValueError: Input X contains NaN", "timeout"),
            ("pass", "missing_result", "ValueError: Input X contains NaN",
             "execution_infrastructure_error"),
            ("if True\n    pass", "validation_error", "SyntaxError: expected ':'", None),
            ("raise RuntimeError('custom failure')", "python_exception",
             "RuntimeError: custom failure", None),
        )
        for code, error_type, error, category in cases:
            with self.subTest(error_type=error_type, code=code):
                execution = {
                    "success": False, "error_type": error_type, "error": error,
                    "traceback": error, "exit_code": 1, "output": "",
                }
                with patch("data_preparation.execute_code", return_value=execution):
                    result = _run_preparation_code(
                        code, ".", {"dataset_id": "owner/data"},
                        code_execution_timeout_seconds=123,
                    )
                self.assertFalse(result["success"])
                self.assertEqual(result["error"], error)
                if category is not None:
                    self.assertEqual(result["failure_category"], category)
                    self.assertEqual(result["repair_hint"], runtime_data_repair_hint(category))
                else:
                    self.assertNotIn("failure_category", result)
                    if error_type == "validation_error":
                        self.assertIn("Syntax source context:", result["repair_hint"])
                        self.assertIn("if True", result["repair_hint"])
                    else:
                        self.assertNotIn("repair_hint", result)

    def test_image_and_pandas_diagnostics_reach_solver_and_bound_repeated_failures(self):
        cases = (
            ("image_object_api_error", (
                "AttributeError: 'TiffImageFile' object has no attribute 'shape'",
                "AttributeError: 'PngImageFile' object has no attribute 'dtype'",
            )),
            ("missing_key_or_column", (
                "KeyError: 'derived_feature'", "KeyError: 'held_out_feature'",
            )),
            ("pandas_positional_indexing_error", (
                "IndexError: positional indexers are out-of-bounds",
                "IndexError: single positional indexer is out-of-bounds",
            )),
        )
        state = SimpleNamespace(
            topic="topic", plan="plan", literature_review="review",
            dataset_selection={}, seeds={}, dataset_code="data = [1]",
        )
        for category, errors in cases:
            with self.subTest(category=category):
                codes = ["draft = 1", "draft = 2", "fixed = 3"]
                requested_messages = []
                state_snapshots = []
                executions = [{
                    "success": False, "error_type": "python_exception", "error": error,
                    "traceback": error, "exit_code": 0, "output": "",
                } for error in errors] + [{
                    "success": True, "error_type": None, "error": None,
                    "traceback": "", "exit_code": 0, "output": "completed",
                }]

                def respond(**kwargs):
                    requested_messages.append(deepcopy(kwargs["messages"]))
                    index = len(requested_messages) - 1
                    if index == 2:
                        self.assertEqual(
                            {tool["function"]["name"] for tool in kwargs["tools"]},
                            {"write_experiment_code"},
                        )
                    return ToolModelResponse("", [ModelToolCall(
                        f"call-{index}", "write_experiment_code",
                        json.dumps({"code": codes[index]}),
                    )])

                def record_processed_state(current, *args, **kwargs):
                    feedback = _process_execution_result(current, *args, **kwargs)
                    state_snapshots.append(deepcopy(current))
                    return feedback

                with tempfile.TemporaryDirectory() as directory, patch(
                    "agent_runtime.query_model_with_tools", side_effect=respond,
                ), patch(
                    "agents.execute_code", side_effect=executions,
                ) as execute, patch(
                    "agents._process_execution_result", side_effect=record_processed_state,
                ):
                    result = MLESolverAgent(
                        "model", "key", code_execution_timeout_seconds=123,
                    ).solve(state, directory)

                self.assertTrue(result["success"])
                self.assertEqual(result["code"], codes[-1])
                self.assertEqual(execute.call_count, 3)
                self.assertEqual(len(requested_messages), 3)
                hint = runtime_data_repair_hint(category)
                self.assertTrue(hint)
                for index, messages in enumerate(requested_messages[1:], start=1):
                    recovery = messages[-1]["content"]
                    diagnostic = json.loads(recovery.split("Latest execution diagnostic:\n", 1)[1])
                    self.assertEqual(diagnostic["failure_category"], category)
                    self.assertEqual(diagnostic["repair_hint"], hint)
                    self.assertEqual(diagnostic["error"], errors[index - 1])
                    self.assertEqual(diagnostic["same_category_count"], index)
                    self.assertEqual(diagnostic["full_replacement_required"], index == 2)
                self.assertEqual(
                    [snapshot["consecutive_failure_category_count"] for snapshot in state_snapshots],
                    [1, 2, 0],
                )
                self.assertEqual(
                    [snapshot["force_replace"] for snapshot in state_snapshots],
                    [False, True, False],
                )
                self.assertIsNone(state_snapshots[-1]["last_failure_category"])
                self.assertIsNone(state_snapshots[-1]["latest_execution_failure"])

    def test_tensor_repair_hint_survives_unmatched_edit_in_solver_requests(self):
        state = SimpleNamespace(
            topic="topic", plan="plan", literature_review="review",
            dataset_selection={}, seeds={}, dataset_code="images = [[1.0]]",
        )
        draft = "images = images.to('cpu')"
        replacement = "import torch\nimages = torch.as_tensor(images).to('cpu')"
        error = "AttributeError: 'list' object has no attribute 'to'"
        failure = {
            "success": False, "error_type": "python_exception", "error": error,
            "traceback": (
                'Traceback (most recent call last):\n'
                '  File "<string>", line 3, in <module>\n'
                "    images = images.to('cpu')\n" + error
            ),
            "exit_code": 0, "output": "",
        }
        success = {
            "success": True, "error_type": None, "error": None,
            "traceback": "", "exit_code": 0, "output": "completed",
        }
        actions = [
            ("write_experiment_code", {"code": draft}),
            ("edit_experiment_code", {"edits": [{
                "old_text": "images = missing_old_version",
                "new_text": replacement,
            }]}),
            ("write_experiment_code", {"code": replacement}),
        ]
        requested_messages = []

        def respond(**kwargs):
            requested_messages.append(deepcopy(kwargs["messages"]))
            index = len(requested_messages) - 1
            name, arguments = actions[index]
            return ToolModelResponse("", [ModelToolCall(
                f"call-{index}", name, json.dumps(arguments),
            )])

        with tempfile.TemporaryDirectory() as directory, patch(
            "agent_runtime.query_model_with_tools", side_effect=respond,
        ), patch(
            "agents.execute_code", side_effect=[failure, success],
        ) as execute:
            result = MLESolverAgent("model", "key", code_execution_timeout_seconds=123).solve(state, directory)

        self.assertTrue(result["success"])
        self.assertEqual(result["code"], replacement)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(len(requested_messages), 3)
        hint = runtime_data_repair_hint("tensor_device_transfer_error")
        self.assertTrue(hint)
        for messages in requested_messages[1:]:
            recovery = messages[-1]["content"]
            self.assertIn("Current experiment code:\n```python\n" + draft, recovery)
            diagnostic_text = recovery.split("Latest execution diagnostic:\n", 1)[1]
            diagnostic = json.loads(diagnostic_text.split("\n\nLatest code action rejection:", 1)[0])
            self.assertEqual(diagnostic["failure_category"], "tensor_device_transfer_error")
            self.assertEqual(diagnostic["repair_hint"], hint)
            self.assertEqual(diagnostic["error"], error)
        self.assertIn("matched 0", requested_messages[2][-1]["content"])

    def test_old_failed_candidate_does_not_replace_current_execution_diagnostic(self):
        state = SimpleNamespace(
            topic="Classify traffic using numerical network-flow features",
            plan="Fit a Random Forest", literature_review="review",
            dataset_selection={}, seeds={},
            dataset_code="rows = [{'srcip': '59.166.0.6', 'dur': 0.5}]\ny = [0]",
        )
        prefix = (
            "import numpy as np\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
        )
        suffix = "\nRandomForestClassifier().fit(X, y)"
        mapping_code = prefix + "X = np.array(rows)" + suffix
        string_code = prefix + "X = np.array([list(row.values()) for row in rows])" + suffix
        fixed_code = prefix + "X = np.array([[row['dur']] for row in rows], dtype=float)" + suffix
        mapping_error = "TypeError: float() argument must be a string or a real number, not 'dict'"
        string_error = "ValueError: could not convert string to float: '59.166.0.6'"

        def failed_execution(error):
            return {
                "success": False, "error_type": "python_exception", "error": error,
                "traceback": (
                    'Traceback (most recent call last):\n'
                    '  File "<string>", line 6, in <module>\n'
                    "    RandomForestClassifier().fit(X, y)\n" + error
                ),
                "exit_code": 0, "output": "",
            }

        success = {
            "success": True, "error_type": None, "error": None,
            "traceback": "", "exit_code": 0, "output": "completed",
        }
        proposed_codes = [mapping_code, string_code, mapping_code, fixed_code]
        requested_messages = []

        def respond(**kwargs):
            requested_messages.append(deepcopy(kwargs["messages"]))
            index = len(requested_messages) - 1
            return ToolModelResponse("", [ModelToolCall(
                f"call-{index}", "write_experiment_code",
                json.dumps({"code": proposed_codes[index]}),
            )])

        with tempfile.TemporaryDirectory() as directory, patch(
            "agent_runtime.query_model_with_tools", side_effect=respond,
        ), patch(
            "agents.execute_code", side_effect=[
                failed_execution(mapping_error), failed_execution(string_error), success,
            ],
        ) as execute:
            result = MLESolverAgent("model", "key", code_execution_timeout_seconds=123).solve(state, directory)

        self.assertTrue(result["success"])
        self.assertEqual(result["code"], fixed_code)
        self.assertEqual(len(requested_messages), 4)
        self.assertEqual(
            [call.args[0] for call in execute.call_args_list],
            [state.dataset_code + "\n\n" + code for code in (
                mapping_code, string_code, fixed_code,
            )],
        )
        for messages, code, category, error in zip(
            requested_messages[1:],
            (mapping_code, string_code, string_code),
            ("mapping_numeric_input", "non_numeric_model_input", "non_numeric_model_input"),
            (mapping_error, string_error, string_error),
        ):
            recovery = messages[-1]["content"]
            self.assertIn("Current experiment code:\n```python\n" + code, recovery)
            diagnostic_text = recovery.split("Latest execution diagnostic:\n", 1)[1]
            diagnostic = json.loads(diagnostic_text.split("\n\nLatest code action rejection:", 1)[0])
            self.assertEqual(diagnostic["failure_category"], category)
            hint = runtime_data_repair_hint(category)
            self.assertTrue(hint)
            self.assertEqual(diagnostic["repair_hint"], hint)
            self.assertEqual(diagnostic["error"], error)
        rejection = json.loads(requested_messages[3][-1]["content"].split(
            "\n\nLatest code action rejection:\n", 1,
        )[1])
        self.assertEqual(rejection["error_type"], "unchanged_failed_experiment_code")
        self.assertEqual(rejection["original_error"], mapping_error)
        self.assertEqual(
            rejection["original_repair_hint"],
            runtime_data_repair_hint("mapping_numeric_input"),
        )
        self.assertEqual(
            rejection["original_execution_diagnostic"]["code_hash"],
            rejection["code_hash"],
        )
        self.assertNotEqual(diagnostic["code_hash"], rejection["code_hash"])


if __name__ == "__main__":
    unittest.main()
