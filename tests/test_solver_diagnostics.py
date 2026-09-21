import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_runtime import AgentRuntime
from agents import _run_preparation_code
from mlesolver import MLESolverAgent, _compact_solver_traceback
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
                    ) as run, patch("mlesolver.execute_code", return_value=execution):
                        agent = MLESolverAgent("model", "key")
                        if supplement:
                            agent.supplement(state, {
                                "objective": "check", "reason": "check",
                                "required_outputs": [], "constraints": [],
                            }, directory, 1)
                        else:
                            agent.solve(state, directory)
                        registry = run.call_args.kwargs["registry"]
                        replace = "replace_supplement_code" if supplement else "replace_experiment_code"
                        execute = "execute_supplement_code" if supplement else "execute_experiment_code"
                        registry.execute(replace, {"code": "print(data)"})
                        feedback = registry.execute(execute, {})
                        # Files already contain original diagnosis when the tool returns.
                        records = [json.loads(line) for line in
                                   Path(recorder.path).read_text().splitlines()]
                        event = next(row for row in records if row["event"] == "solver_execution_diagnostic")
                        self.assertEqual(event["result"]["traceback"], original["traceback"])
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
        with patch("agents.execute_code", return_value=execution):
            result = _run_preparation_code("raise ValueError('bad')", ".")
        self.assertEqual(result["traceback"], execution["traceback"])


if __name__ == "__main__":
    unittest.main()
