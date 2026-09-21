import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_runtime import AgentRuntime
from mlesolver import MLESolverAgent


def execution_result(output, success=True, error_type=None):
    return {"output": output, "success": success, "error_type": error_type,
            "error": output if not success else None, "traceback": "", "exit_code": 0}


class MLESolverSolveTest(unittest.TestCase):
    def test_main_and_supplement_auto_submit_without_another_model_request(self):
        import json
        from unittest.mock import MagicMock
        from inference import ModelToolCall, ToolModelResponse

        state = SimpleNamespace(
            topic="topic", plan="plan", dataset_selection={}, seeds={},
            dataset_code="data = [1]", literature_review="review",
            experiment_code="original_code", experiment_output="original_output",
            supplementary_experiments=[],
        )
        request = {"objective": "check", "reason": "check",
                   "required_outputs": [], "constraints": []}
        for supplement in (False, True):
            for fail_first in (False, True):
                with self.subTest(supplement=supplement, fail_first=fail_first), tempfile.TemporaryDirectory() as folder:
                    replace = "replace_supplement_code" if supplement else "replace_experiment_code"
                    edit = "edit_supplement_code" if supplement else "edit_experiment_code"
                    submit = "submit_supplementary_experiment" if supplement else "submit_experiment"
                    calls = []
                    responses = [
                        ToolModelResponse("", [ModelToolCall("write", replace, json.dumps({"code": "value = 1"}))]),
                        ToolModelResponse("", [ModelToolCall("repair", edit, json.dumps({
                            "old_text": "value = 1", "new_text": "value = 2",
                        }))]),
                    ]
                    def respond(**kwargs):
                        names = {tool["function"]["name"] for tool in kwargs["tools"]}
                        self.assertNotIn(submit, names)
                        if calls:
                            self.assertTrue(fail_first)
                            self.assertIn(edit, names)
                            self.assertIn("broken", json.dumps(kwargs["messages"]))
                        calls.append(kwargs)
                        self.assertLessEqual(len(calls), 2 if fail_first else 1)
                        return responses[len(calls) - 1]

                    outputs = (
                        [execution_result("broken", False, "python_exception")]
                        if fail_first else []
                    ) + [execution_result("final_output")]
                    recorder = MagicMock()
                    with patch("agent_runtime.query_model_with_tools", side_effect=respond), patch(
                        "agent_runtime.current_prompt_recorder", return_value=recorder
                    ), patch("mlesolver.execute_code", side_effect=outputs) as execute:
                        agent = MLESolverAgent("model", "key")
                        result = (
                            agent.supplement(state, request, folder, 1)
                            if supplement else agent.solve(state, folder)
                        )
                    self.assertEqual(len(calls), 2 if fail_first else 1)
                    self.assertEqual(execute.call_count, len(outputs))
                    self.assertTrue(result["success"])
                    self.assertTrue(result["automatic_execution"])
                    self.assertEqual(result["code"], "value = 2" if fail_first else "value = 1")
                    self.assertEqual(result["output"], "final_output")
                    events = [call.args[0] for call in recorder.record.call_args_list
                              if call.args[0].get("event") == "tool_execution"]
                    submissions = [event for event in events if event["tool_name"] == submit]
                    self.assertEqual(len(submissions), 1)
                    self.assertEqual(events[-1]["tool_name"], submit)
                    self.assertTrue(submissions[0]["success"])
                    if supplement:
                        self.assertEqual(result["supplement_id"], "supplement_01")
                        self.assertEqual(result["request"], request)
                    self.assertEqual(state.experiment_code, "original_code")
                    self.assertEqual(state.experiment_output, "original_output")

    def test_main_and_supplement_classify_separate_diagnostics(self):
        state = SimpleNamespace(
            topic="topic", plan="plan", dataset_selection={}, seeds={},
            dataset_code="data = [1]", literature_review="review",
            experiment_code="print(data)", experiment_output="1",
            supplementary_experiments=[],
        )
        for supplement in (False, True):
            for diagnostic_field in ("error", "traceback"):
                with self.subTest(supplement=supplement, field=diagnostic_field), tempfile.TemporaryDirectory() as folder:
                    failure = {
                        "success": False, "output": "out of memory is just printed text",
                        "error_type": "python_exception", "error": "ValueError",
                        "traceback": "", "exit_code": 0,
                    }
                    failure[diagnostic_field] = "ValueError: could not convert string to float: 'tcp'"
                    agent = MLESolverAgent("model", "key")
                    with patch.object(AgentRuntime, "run", return_value=SimpleNamespace(value={})) as run, patch(
                        "mlesolver.execute_code", side_effect=[
                            failure, failure, execution_result("accuracy=0.9"),
                        ],
                    ):
                        if supplement:
                            agent.supplement(state, {
                                "objective": "check", "reason": "check",
                                "required_outputs": [], "constraints": [],
                            }, folder, 1)
                        else:
                            agent.solve(state, folder)
                        registry = run.call_args.kwargs["registry"]
                        provider = run.call_args.kwargs["tool_names_provider"]
                        replace = "replace_supplement_code" if supplement else "replace_experiment_code"
                        execute = "execute_supplement_code" if supplement else "execute_experiment_code"
                        submit = "submit_supplementary_experiment" if supplement else "submit_experiment"
                        for version in (1, 2):
                            registry.execute(replace, {"code": f"value = {version}"})
                            failed = registry.execute(execute, {})
                            self.assertEqual(failed["failure_category"], "non_numeric_model_input")
                            self.assertEqual(failed["output"], failure["output"])
                            self.assertEqual(failed[diagnostic_field], failure[diagnostic_field])
                            self.assertEqual(failed["same_category_count"], version)
                            self.assertFalse(registry.execute(submit, {})["success"])
                        self.assertEqual(provider(), {replace})
                        self.assertIn("numeric feature", failed["repair_hint"])
                        registry.execute(replace, {"code": "value = 3"})
                        registry.execute(execute, {})
                        self.assertEqual(provider(), {submit})
                        self.assertEqual(registry.execute(submit, {})["output"], "accuracy=0.9")

    def test_main_and_supplement_use_status_and_allow_infrastructure_retry(self):
        state = SimpleNamespace(
            topic="topic", plan="plan", dataset_selection={}, seeds={},
            dataset_code="data = [1]", literature_review="review",
            experiment_code="print(data)", experiment_output="1",
            supplementary_experiments=[],
        )
        for supplement in (False, True):
            with self.subTest(supplement=supplement), tempfile.TemporaryDirectory() as folder:
                agent = MLESolverAgent("model", "key")
                with patch.object(AgentRuntime, "run", return_value=SimpleNamespace(value={} )) as run, patch(
                    "mlesolver.execute_code", side_effect=[
                        execution_result("", False, "missing_result"),
                        execution_result("[CODE EXECUTION ERROR] literal text"),
                    ],
                ) as execute:
                    if supplement:
                        agent.supplement(state, {
                            "objective": "check", "reason": "check",
                            "required_outputs": [], "constraints": [],
                        }, folder, 1)
                    else:
                        agent.solve(state, folder)
                    registry = run.call_args.kwargs["registry"]
                    provider = run.call_args.kwargs["tool_names_provider"]
                    replace = "replace_supplement_code" if supplement else "replace_experiment_code"
                    execute_name = "execute_supplement_code" if supplement else "execute_experiment_code"
                    submit = "submit_supplementary_experiment" if supplement else "submit_experiment"
                    registry.execute(replace, {"code": "print(data)"})
                    failed = registry.execute(execute_name, {})
                    self.assertFalse(failed["success"])
                    self.assertEqual(failed["failure_category"], "execution_infrastructure_error")
                    self.assertFalse(registry.execute(submit, {})["success"])
                    self.assertIn(execute_name, provider())
                    successful = registry.execute(execute_name, {})
                    self.assertTrue(successful["success"])
                    self.assertEqual(provider(), {submit})
                    self.assertTrue(registry.execute(submit, {})["success"])
                    self.assertTrue(execute.call_args.kwargs["return_details"])

    def test_solver_escalates_repeated_error_category_to_full_replacement(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="dataset_code",
            literature_review="review",
        )
        agent = MLESolverAgent("model", "key")
        completed = SimpleNamespace(value={"success": True})
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run, patch(
            "mlesolver.execute_code",
            side_effect=[
                execution_result("could not convert string to float: 'tcp'", False, "python_exception"),
                execution_result("could not convert string to float: 'udp'", False, "python_exception"),
            ],
        ):
            agent.solve(state, temp_dir)
            registry = run.call_args.kwargs["registry"]
            provider = run.call_args.kwargs["tool_names_provider"]
            registry.execute("replace_experiment_code", {"code": "version_a = 1"})
            first = registry.execute("execute_experiment_code", {})
            registry.execute(
                "edit_experiment_code",
                {"old_text": "version_a = 1", "new_text": "version_b = 1"},
            )
            second = registry.execute("execute_experiment_code", {})

        self.assertEqual(first["failure_category"], "non_numeric_model_input")
        self.assertEqual(second["same_category_count"], 2)
        self.assertTrue(second["full_replacement_required"])
        self.assertEqual(provider(), {"replace_experiment_code"})
        self.assertIn("do not patch only one model call", second["repair_hint"])

    def test_solver_rejects_any_previously_failed_code_version(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="dataset_code",
            literature_review="review",
        )
        agent = MLESolverAgent("model", "key")
        completed = SimpleNamespace(value={"success": True})
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run, patch(
            "mlesolver.execute_code",
            side_effect=[
                execution_result("first failure", False, "python_exception"),
                execution_result("list index out of range", False, "python_exception"),
            ],
        ):
            agent.solve(state, temp_dir)
            registry = run.call_args.kwargs["registry"]
            registry.execute("replace_experiment_code", {"code": "version_a = 1"})
            registry.execute("execute_experiment_code", {})
            registry.execute(
                "edit_experiment_code",
                {"old_text": "version_a = 1", "new_text": "version_b = 1"},
            )
            registry.execute("execute_experiment_code", {})
            repeated = registry.execute(
                "edit_experiment_code",
                {"old_text": "version_b = 1", "new_text": "version_a = 1"},
            )

        self.assertFalse(repeated["success"])
        self.assertIn("previously failed version", repeated["error"])

    def test_solver_rejects_invalid_python_before_execution(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="data = [1, 2]",
            literature_review="literature_review",
        )
        completed = SimpleNamespace(value={"success": True})
        agent = MLESolverAgent("model", "key")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run, patch("mlesolver.execute_code") as execute_code:
            agent.solve(state, temp_dir)
            registry = run.call_args.kwargs["registry"]
            provider = run.call_args.kwargs["tool_names_provider"]

            invalid = registry.execute(
                "replace_experiment_code",
                {"code": "print('unterminated)"},
            )
            self.assertFalse(invalid["success"])
            self.assertIn("invalid Python", invalid["error"])
            self.assertEqual(provider(), {"replace_experiment_code"})

            registry.execute(
                "replace_experiment_code",
                {"code": "result = {'accuracy': 1.0}"},
            )
            invalid_edit = registry.execute(
                "edit_experiment_code",
                {
                    "old_text": "result = {'accuracy': 1.0}",
                    "new_text": "print(f'{result['accuracy']}')",
                },
            )

        self.assertFalse(invalid_edit["success"])
        self.assertIn("invalid Python", invalid_edit["error"])
        execute_code.assert_not_called()

    def test_solver_exposes_tools_in_valid_execution_order(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="dataset_code",
            dataset_manifest={"feature_artifacts": {"train": "X_train"}},
            dataset_validation_warnings=["analysis metadata incomplete"],
            literature_review="literature_review",
        )
        completed = SimpleNamespace(value={"success": True})
        agent = MLESolverAgent("model", "key")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime,
            "run",
            return_value=completed,
        ) as run, patch(
            "mlesolver.execute_code",
            side_effect=[execution_result("broken", False, "python_exception"), execution_result("experiment_output")],
        ) as execute_code:
            agent.solve(state, temp_dir)
            kwargs = run.call_args.kwargs
            registry = kwargs["registry"]
            available_tools = kwargs["tool_names_provider"]
            self.assertIn("X_train", kwargs["user_prompt"])
            self.assertIn("analysis metadata incomplete", kwargs["user_prompt"])
            self.assertEqual(
                kwargs["automatic_tools"],
                {"execute_experiment_code", "submit_experiment"},
            )

            self.assertEqual(available_tools(), {"replace_experiment_code"})

            registry.execute("replace_experiment_code", {"code": "broken_code"})
            self.assertEqual(
                available_tools(),
                {"execute_experiment_code"},
            )
            unchanged = registry.execute(
                "replace_experiment_code",
                {"code": "broken_code"},
            )
            self.assertFalse(unchanged["success"])
            self.assertIn("unchanged", unchanged["error"])

            failed = registry.execute("execute_experiment_code", {})
            self.assertFalse(failed["success"])
            self.assertEqual(
                available_tools(),
                {"edit_experiment_code", "replace_experiment_code"},
            )

            registry.execute(
                "edit_experiment_code",
                {"old_text": "broken_code", "new_text": "fixed_code"},
            )
            self.assertEqual(available_tools(), {"execute_experiment_code"})

            succeeded = registry.execute("execute_experiment_code", {})
            self.assertTrue(succeeded["success"])
            self.assertEqual(available_tools(), {"submit_experiment"})

            submitted = registry.execute("submit_experiment", {})
            self.assertTrue(submitted["success"])
            self.assertEqual(submitted["code"], "fixed_code")
            self.assertEqual(submitted["output"], "experiment_output")
            self.assertEqual(execute_code.call_count, 2)


class MLESolverSupplementTest(unittest.TestCase):
    def test_supplement_tools_preserve_original_experiment(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="dataset_code",
            experiment_code="original_experiment_code",
            experiment_output="original_experiment_output",
            supplementary_experiments=[],
        )
        request = {
            "objective": "check robustness",
            "reason": "Observed accuracy was 0.93; one split is insufficient",
            "required_outputs": ["mean F1"],
            "constraints": ["preserve original experiment"],
        }
        completed = SimpleNamespace(value={"success": True})
        agent = MLESolverAgent("model", "key")

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime,
            "run",
            return_value=completed,
        ) as run, patch(
            "mlesolver.execute_code",
            return_value=execution_result("supplement_output"),
        ) as execute_code:
            agent.supplement(state, request, temp_dir, 1)
            kwargs = run.call_args.kwargs
            self.assertNotIn("original_experiment_output", kwargs["user_prompt"])
            self.assertNotIn("Original experiment output (read only):", kwargs["user_prompt"])
            self.assertIn("original_experiment_code", kwargs["user_prompt"])
            self.assertIn("Dataset code: dataset_code", kwargs["user_prompt"])
            self.assertIn(request["reason"], kwargs["user_prompt"])
            registry = kwargs["registry"]
            available_tools = kwargs["tool_names_provider"]
            self.assertEqual(
                registry.names,
                {
                    "replace_supplement_code",
                    "edit_supplement_code",
                    "execute_supplement_code",
                    "submit_supplementary_experiment",
                },
            )
            self.assertEqual(
                kwargs["terminal_tools"],
                {"submit_supplementary_experiment"},
            )
            self.assertEqual(available_tools(), {"replace_supplement_code"})
            registry.execute(
                "replace_supplement_code",
                {"code": "print('supplement')"},
            )
            self.assertEqual(available_tools(), {"execute_supplement_code"})
            execution = registry.execute("execute_supplement_code", {})
            self.assertEqual(
                available_tools(),
                {"submit_supplementary_experiment"},
            )
            submitted = registry.execute("submit_supplementary_experiment", {})

            expected_workdir = str(
                Path(temp_dir).resolve() / "supplements" / "supplement_01"
            )
            execute_code.assert_called_with(
                "dataset_code\n\nprint('supplement')",
                timeout=600,
                workdir=expected_workdir,
                return_details=True,
            )
            self.assertEqual(execute_code.call_count, 1)
            self.assertTrue(execution["success"])
            self.assertTrue(submitted["success"])
            self.assertEqual(submitted["supplement_id"], "supplement_01")
            self.assertEqual(submitted["request"], request)
            self.assertEqual(submitted["code"], "print('supplement')")
            self.assertEqual(submitted["output"], "supplement_output")
            self.assertEqual(state.experiment_code, "original_experiment_code")
            self.assertEqual(state.experiment_output, "original_experiment_output")
            self.assertNotIn("original_experiment_code", execute_code.call_args.args[0])

    def test_supplement_rejects_unchanged_replacement(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="dataset_code",
            experiment_code="original",
            experiment_output="output",
            supplementary_experiments=[],
        )
        completed = SimpleNamespace(value={"success": True})
        agent = MLESolverAgent("model", "key")
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.supplement(state, {"objective": "check"}, temp_dir, 1)
            registry = run.call_args.kwargs["registry"]
            registry.execute(
                "replace_supplement_code",
                {"code": "print('same')"},
            )
            unchanged = registry.execute(
                "replace_supplement_code",
                {"code": "print('same')"},
            )

        self.assertFalse(unchanged["success"])
        self.assertIn("unchanged", unchanged["error"])
        self.assertIn("pretrained_model_assets", run.call_args.kwargs["user_prompt"])


if __name__ == "__main__":
    unittest.main()
