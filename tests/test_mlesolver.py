import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_runtime import AgentRuntime
from mlesolver import (
    EXPERIMENT_EXECUTION_TIMEOUT_SECONDS,
    MLESolverAgent,
    _execution_failure_category,
    _failure_repair_hint,
    _python_syntax_error,
)


def execution_result(output, success=True, error_type=None):
    return {"output": output, "success": success, "error_type": error_type,
            "error": output if not success else None, "traceback": "", "exit_code": 0}


class MLESolverSolveTest(unittest.TestCase):
    def test_main_solver_uses_latest_execution_failure_context(self):
        import json
        from unittest.mock import MagicMock
        from inference import ModelToolCall, ToolModelResponse

        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={},
            seeds={},
            dataset_code="data = [1]",
            literature_review="review",
            experiment_code="",
            experiment_output="",
            supplementary_experiments=[],
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall(
                    "write",
                    "replace_experiment_code",
                    json.dumps({"code": "draft = 1"}),
                )],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall(
                    "rewrite",
                    "replace_experiment_code",
                    json.dumps({"code": "fixed = 1"}),
                )],
            ),
        ]
        requested_messages = []

        def respond(**kwargs):
            requested_messages.append(
                json.loads(json.dumps(kwargs["messages"]))
            )
            return responses[len(requested_messages) - 1]

        failure = {
            "success": False,
            "output": "OLD_OUTPUT_MUST_NOT_REAPPEAR",
            "error_type": "python_exception",
            "error": "LATEST_EXECUTION_ERROR",
            "traceback": "LATEST_TRACEBACK",
            "exit_code": 0,
        }
        recorder = MagicMock()
        with tempfile.TemporaryDirectory() as folder, patch(
            "agent_runtime.query_model_with_tools",
            side_effect=respond,
        ), patch(
            "agent_runtime.current_prompt_recorder",
            return_value=recorder,
        ), patch(
            "mlesolver.execute_code",
            side_effect=[failure, execution_result("final_output")],
        ):
            result = MLESolverAgent("model", "key").solve(state, folder)

        self.assertTrue(result["success"])
        self.assertEqual(result["code"], "fixed = 1")
        self.assertEqual(len(requested_messages), 2)
        recovery_messages = requested_messages[1]
        self.assertEqual(
            [message["role"] for message in recovery_messages],
            ["system", "user", "user"],
        )
        recovery_text = json.dumps(recovery_messages)
        self.assertIn("draft = 1", recovery_text)
        self.assertIn("LATEST_EXECUTION_ERROR", recovery_text)
        self.assertIn("LATEST_TRACEBACK", recovery_text)
        self.assertIn("Earlier execution failures are superseded", recovery_text)
        self.assertNotIn("OLD_OUTPUT_MUST_NOT_REAPPEAR", recovery_text)
        compactions = [
            call.args[0]
            for call in recorder.record.call_args_list
            if call.args[0].get("event") == "failure_history_compacted"
        ]
        self.assertEqual(len(compactions), 1)
        self.assertEqual(
            compactions[0]["failure_tool"],
            "execute_experiment_code",
        )

    def test_duplicate_failed_replacement_keeps_latest_execution_diagnostic(self):
        import json
        from unittest.mock import MagicMock
        from inference import ModelToolCall, ToolModelResponse

        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={},
            seeds={},
            dataset_code="data = [1]",
            literature_review="review",
            experiment_code="",
            experiment_output="",
            supplementary_experiments=[],
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall(
                    "write",
                    "replace_experiment_code",
                    json.dumps({"code": "draft = 1"}),
                )],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall(
                    "repeat",
                    "replace_experiment_code",
                    json.dumps({"code": "draft = 1"}),
                )],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall(
                    "rewrite",
                    "replace_experiment_code",
                    json.dumps({"code": "fixed = 1"}),
                )],
            ),
        ]
        requested_messages = []

        def respond(**kwargs):
            requested_messages.append(
                json.loads(json.dumps(kwargs["messages"]))
            )
            return responses[len(requested_messages) - 1]

        failure = {
            "success": False,
            "output": "OLD_OUTPUT_MUST_NOT_REAPPEAR",
            "error_type": "python_exception",
            "error": (
                "ValueError: Found input variables with inconsistent numbers of "
                "samples: [3, 4]"
            ),
            "traceback": "LATEST_TRACEBACK",
            "exit_code": 0,
        }
        recorder = MagicMock()
        with tempfile.TemporaryDirectory() as folder, patch(
            "agent_runtime.query_model_with_tools",
            side_effect=respond,
        ), patch(
            "agent_runtime.current_prompt_recorder",
            return_value=recorder,
        ), patch(
            "mlesolver.execute_code",
            side_effect=[failure, execution_result("final_output")],
        ):
            result = MLESolverAgent("model", "key").solve(state, folder)

        self.assertTrue(result["success"])
        self.assertEqual(len(requested_messages), 3)
        duplicate_recovery = json.dumps(requested_messages[2])
        self.assertIn("unchanged_failed_experiment_code", duplicate_recovery)
        self.assertIn("original_error", duplicate_recovery)
        self.assertIn("LATEST_TRACEBACK", duplicate_recovery)
        self.assertIn("one boolean mask", duplicate_recovery)
        self.assertNotIn("OLD_OUTPUT_MUST_NOT_REAPPEAR", duplicate_recovery)

    def test_solver_classifies_evaluation_subset_alignment(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={},
            seeds={},
            dataset_code="data = [1]",
            literature_review="review",
        )
        completed = SimpleNamespace(value={"success": True})
        failure = {
            "success": False,
            "output": "",
            "error_type": "python_exception",
            "error": (
                "ValueError: Found input variables with inconsistent numbers of "
                "samples: [3, 4]"
            ),
            "traceback": "",
            "exit_code": 0,
        }
        with tempfile.TemporaryDirectory() as folder, patch.object(
            AgentRuntime,
            "run",
            return_value=completed,
        ) as run, patch(
            "mlesolver.execute_code",
            return_value=failure,
        ):
            MLESolverAgent("model", "key").solve(state, folder)
            registry = run.call_args.kwargs["registry"]
            registry.execute("replace_experiment_code", {"code": "draft = 1"})
            result = registry.execute("execute_experiment_code", {})

        self.assertEqual(result["failure_category"], "evaluation_subset_alignment")
        self.assertIn("one boolean mask", result["repair_hint"])

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
                    edit_arguments = (
                        {
                            "old_text": "value = 1",
                            "new_text": "value = 2",
                        }
                        if supplement
                        else {
                            "edits": [{
                                "old_text": "value = 1",
                                "new_text": "value = 2",
                            }],
                        }
                    )
                    calls = []
                    responses = [
                        ToolModelResponse("", [ModelToolCall("write", replace, json.dumps({"code": "value = 1"}))]),
                        ToolModelResponse("", [ModelToolCall("repair", edit, json.dumps(edit_arguments))]),
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
                {"edits": [{
                    "old_text": "version_a = 1",
                    "new_text": "version_b = 1",
                }]},
            )
            second = registry.execute("execute_experiment_code", {})

        self.assertEqual(first["failure_category"], "non_numeric_model_input")
        self.assertEqual(second["same_category_count"], 2)
        self.assertTrue(second["full_replacement_required"])
        self.assertEqual(provider(), {"replace_experiment_code"})
        self.assertIn("do not patch only one model call", second["repair_hint"])

    def test_solver_escalates_repeated_ragged_input_to_full_replacement(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="dataset_code",
            literature_review="review",
        )
        failure = "ValueError: expected sequence of length 3 at dim 3 (got 4)"
        agent = MLESolverAgent("model", "key")
        completed = SimpleNamespace(value={"success": True})
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run, patch(
            "mlesolver.execute_code",
            side_effect=[
                execution_result(failure, False, "python_exception"),
                execution_result(failure, False, "python_exception"),
            ],
        ):
            agent.solve(state, temp_dir)
            registry = run.call_args.kwargs["registry"]
            provider = run.call_args.kwargs["tool_names_provider"]
            registry.execute("replace_experiment_code", {"code": "version_a = 1"})
            first = registry.execute("execute_experiment_code", {})
            registry.execute(
                "edit_experiment_code",
                {"edits": [{
                    "old_text": "version_a = 1",
                    "new_text": "version_b = 1",
                }]},
            )
            second = registry.execute("execute_experiment_code", {})

        self.assertEqual(first["failure_category"], "ragged_array_input")
        self.assertEqual(second["same_category_count"], 2)
        self.assertTrue(second["full_replacement_required"])
        self.assertEqual(provider(), {"replace_experiment_code"})
        self.assertIn("np.stack", second["repair_hint"])

    def test_runtime_data_categories_have_specific_repair_hints(self):
        cases = {
            "ValueError: Input X contains infinity": "non_finite_numeric_input",
            "TypeError: list indices must be integers or slices, not str": (
                "list_of_records_access_error"
            ),
        }
        for error, category in cases.items():
            with self.subTest(error=error):
                self.assertEqual(_execution_failure_category(error), category)
                self.assertNotIn(
                    "Use the traceback", _failure_repair_hint(category)
                )

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
                {"edits": [{
                    "old_text": "version_a = 1",
                    "new_text": "version_b = 1",
                }]},
            )
            registry.execute("execute_experiment_code", {})
            repeated = registry.execute(
                "edit_experiment_code",
                {"edits": [{
                    "old_text": "version_b = 1",
                    "new_text": "version_a = 1",
                }]},
            )

        self.assertFalse(repeated["success"])
        self.assertEqual(
            repeated["error_type"],
            "unchanged_failed_experiment_code",
        )
        self.assertEqual(repeated["original_error"], "first failure")
        self.assertEqual(
            repeated["original_execution_diagnostic"]["error"],
            "first failure",
        )
        self.assertEqual(
            repeated["latest_execution_diagnostic"]["error"],
            "list index out of range",
        )
        self.assertIn("latest_execution_diagnostic", repeated)

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
            self.assertEqual(invalid["error_type"], "syntax_error")
            self.assertIn("invalid Python", invalid["error"])
            self.assertIn("Syntax source context:", invalid["error"])
            self.assertIn("-> 1: print('unterminated)", invalid["error"])
            self.assertIn("^", invalid["error"])
            self.assertEqual(provider(), {"replace_experiment_code"})

            registry.execute(
                "replace_experiment_code",
                {"code": "result = {'accuracy': 1.0}"},
            )
            invalid_edit = registry.execute(
                "edit_experiment_code",
                {
                    "edits": [{
                        "old_text": "result = {'accuracy': 1.0}",
                        "new_text": "print(f'{result['accuracy']}')",
                    }],
                },
            )

        self.assertFalse(invalid_edit["success"])
        self.assertEqual(invalid_edit["error_type"], "syntax_error")
        self.assertIn("invalid Python", invalid_edit["error"])
        execute_code.assert_not_called()

    def test_syntax_error_context_is_bounded_and_includes_following_lines(self):
        source = (
            "first = 1\n"
            "print(f'Entry:\n"
            "{text}\\n')\n"
            + "x" * 500
            + "\nnearby = 2\nFAR_AWAY_SENTINEL = True\n"
        )

        feedback = _python_syntax_error(source)

        self.assertIn("Syntax source context:", feedback)
        self.assertIn("-> 2: print(f'Entry:", feedback)
        self.assertIn("3: {text}\\n')", feedback)
        self.assertIn("^", feedback)
        self.assertNotIn("FAR_AWAY_SENTINEL", feedback)
        self.assertLess(len(feedback), 1_200)

    def test_solver_keeps_syntax_context_after_compacting_failed_code(self):
        import json
        from inference import ModelToolCall, ToolModelResponse

        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={},
            seeds={},
            dataset_code="data = [1]",
            literature_review="review",
        )
        invalid_code = (
            "first = 1\n"
            "print(f'Entry:\n"
            "{text}\\n')\n"
            + "x" * 2_500
            + "\nFAR_AWAY_SENTINEL = True\n"
        )
        responses = [
            ToolModelResponse(
                "",
                [ModelToolCall(
                    "invalid",
                    "replace_experiment_code",
                    json.dumps({"code": invalid_code}),
                )],
            ),
            ToolModelResponse(
                "",
                [ModelToolCall(
                    "valid",
                    "replace_experiment_code",
                    json.dumps({"code": "result = 1"}),
                )],
            ),
        ]
        requested_messages = []

        def respond(**kwargs):
            requested_messages.append(
                json.loads(json.dumps(kwargs["messages"]))
            )
            return responses[len(requested_messages) - 1]

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "agent_runtime.query_model_with_tools",
            side_effect=respond,
        ), patch(
            "mlesolver.execute_code",
            return_value=execution_result("final output"),
        ) as execute_code:
            result = MLESolverAgent("model", "key").solve(state, temp_dir)

        recovery = json.dumps(requested_messages[1])
        self.assertTrue(result["success"])
        self.assertIn("[rejected code omitted:", recovery)
        self.assertIn("Syntax source context:", recovery)
        self.assertIn("-> 2: print(f'Entry:", recovery)
        self.assertIn("^", recovery)
        self.assertNotIn("FAR_AWAY_SENTINEL", recovery)
        execute_code.assert_called_once()

    def test_main_solver_applies_batch_edits_atomically(self):
        state = SimpleNamespace(
            topic="topic",
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"sklearn": 4001},
            dataset_code="dataset_code",
            literature_review="literature_review",
        )
        completed = SimpleNamespace(value={"success": True})
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            AgentRuntime,
            "run",
            return_value=completed,
        ) as run, patch(
            "mlesolver.execute_code",
            return_value=execution_result("experiment_output"),
        ) as execute_code:
            MLESolverAgent("model", "key").solve(state, temp_dir)
            registry = run.call_args.kwargs["registry"]
            registry.execute(
                "replace_experiment_code",
                {"code": "alpha = 1\nbeta = 2\n"},
            )
            edited = registry.execute(
                "edit_experiment_code",
                {"edits": [
                    {"old_text": "alpha = 1", "new_text": "alpha = 10"},
                    {"old_text": "beta = 2", "new_text": "beta = 20"},
                ]},
            )
            failed = registry.execute(
                "edit_experiment_code",
                {"edits": [
                    {"old_text": "alpha = 10", "new_text": "alpha = 30"},
                    {"old_text": "missing = 1", "new_text": "missing = 2"},
                ]},
            )
            executed = registry.execute("execute_experiment_code", {})

        self.assertTrue(edited["success"])
        self.assertEqual(edited["edit_count"], 2)
        self.assertFalse(failed["success"])
        self.assertIn("edit 2", failed["error"])
        self.assertTrue(executed["success"])
        self.assertFalse(registry.allows_multiple("edit_experiment_code"))
        coalescer = registry.batch_coalescer_for("edit_experiment_code")
        self.assertIsNotNone(coalescer)
        self.assertEqual(
            coalescer([
                {"edits": [{"old_text": "alpha = 10", "new_text": "alpha = 30"}]},
                {"edits": [{"old_text": "beta = 20", "new_text": "beta = 30"}]},
            ]),
            {
                "edits": [
                    {"old_text": "alpha = 10", "new_text": "alpha = 30"},
                    {"old_text": "beta = 20", "new_text": "beta = 30"},
                ]
            },
        )
        schema = next(
            item["function"]
            for item in registry.schemas
            if item["function"]["name"] == "edit_experiment_code"
        )
        self.assertEqual(schema["parameters"]["required"], ["edits"])
        self.assertEqual(schema["parameters"]["properties"]["edits"]["maxItems"], 6)
        combined = execute_code.call_args.args[0]
        self.assertIn("alpha = 10", combined)
        self.assertIn("beta = 20", combined)
        self.assertNotIn("alpha = 30", combined)

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
                {"edits": [{
                    "old_text": "broken_code",
                    "new_text": "fixed_code",
                }]},
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
                timeout=EXPERIMENT_EXECUTION_TIMEOUT_SECONDS,
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
