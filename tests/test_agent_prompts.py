import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from agent_runtime import AgentRuntime, ToolProtocolError
from agents import (
    MLEngineerAgent,
    PhDStudentAgent,
    PostdocAgent,
    SWEngineerAgent,
)
from data_preparation import _run_preparation_code
from execution_diagnostics import runtime_data_repair_hint
from prompts import (
    ml_dataset_selection_prompt,
    sw_data_preparation_prompt,
)
from tools import ArxivSearch

with patch("dotenv.load_dotenv"):
    from ai_lab_repo import load_paper_writing_config


COLLABORATION_ROUNDS = {
    "plan_max_rounds": 8,
    "data_preparation_max_rounds": 12,
}


def arxiv_search_mock(*candidate_ids):
    search = MagicMock(spec=ArxivSearch)
    search.remaining_candidate_count.side_effect = lambda excluded_ids=None: len(
        set(candidate_ids) - set(excluded_ids or ())
    )
    return search


def dataset_contract(**overrides):
    value = {
        "task_type": "classification",
        "feature_representation": "numeric_matrix",
        "artifact_layout": "split_aligned",
        "target_columns": ["label"],
        "feature_columns": ["value"],
        "analysis_columns": [],
        "excluded_columns": [],
        "categorical_columns": [],
        "required_prepared_splits": ["train", "validation", "test"],
    }
    value.update(overrides)
    value.setdefault("role_sources", [{
        "role": "prepared_samples",
        "purpose": "Provide the model inputs, targets, and analysis attributes",
        "source_type": "column",
        "source_columns": list(dict.fromkeys(
            value["feature_columns"]
            + value["target_columns"]
            + value["analysis_columns"]
        )),
        "resources": [],
        "derivation": "",
        "alignment": "sample",
        "required_prepared_splits": value["required_prepared_splits"],
    }])
    return value


def inspect_and_select(registry, arguments):
    inspected = registry.execute("inspect_selected_dataset", arguments)
    if not inspected.get("success"):
        raise AssertionError(inspected)
    return registry.execute("select_inspected_huggingface_dataset", {})


def dataset_code_arguments(code):
    return {"lines": str(code).splitlines()}


def dataset_inspection(**overrides):
    return {
        "dataset_id": "owner/data",
        "revision": "r1",
        "config_name": "default",
        "split": "train",
        "dataset_schema": {"row_count": 2, "columns": {"value": "int64"}},
        **overrides,
    }


def executed_dataset(inspection, resources=None):
    return {
        "dataset_id": inspection["dataset_id"],
        "revision": inspection["revision"],
        "resources": resources or [{
            "config_name": inspection["config_name"], "split": inspection["split"],
        }],
    }


class AgentPromptTest(unittest.TestCase):
    def test_workflow_promotes_matching_successful_trial_before_next_ml_turn(self):
        import tempfile
        from workflows.laboratory import LaboratoryWorkflow
        from research_state import ResearchState

        selection = {
            "dataset_id": "owner/data",
            "revision": "r1",
            "config_name": "default",
            "split": "train",
        }
        inspected = {
            "success": True,
            **selection,
            "dataset_schema": {"columns": {"value": "int64"}},
        }
        actions = [
            ("search_huggingface_datasets", {
                "success": True,
                "results": [],
                "selected_dataset": dict(selection),
            }),
            ("inspect_selected_dataset", {
                **inspected,
                "selected_dataset": dict(selection),
            }),
            ("write_and_run_dataset_code", {
                "success": True,
                "code": "data = [1, 2]",
                "output": "prepared",
                "selection": dict(selection),
                "artifacts": [],
                "prepared_data_manifest": None,
            }),
        ]
        ml_calls = []
        def ml_turn(*args, **kwargs):
            ml_calls.append(kwargs)
            name, value = actions[len(ml_calls) - 1]
            return SimpleNamespace(tool_name=name, value=value)

        def sw_turn(*args, **kwargs):
            return SimpleNamespace(
                tool_name="send_dialogue",
                value={"success": True, "message": "Continue."},
            )

        with tempfile.TemporaryDirectory() as directory:
            workflow = LaboratoryWorkflow(
                ResearchState("query", "recording", "topic"),
                "model",
                "key",
                directory,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            with patch.object(
                workflow.ml_engineer,
                "data_preparation_turn",
                side_effect=ml_turn,
            ), patch.object(
                workflow.sw_engineer,
                "data_preparation_turn",
                side_effect=sw_turn,
            ):
                result_selection, submitted = workflow._prepare_dataset_collaboratively()

        self.assertEqual(len(ml_calls), 3)
        self.assertEqual(submitted["code"], "data = [1, 2]")
        self.assertEqual(result_selection["dataset_id"], "owner/data")

    def test_workflow_preserves_complete_inspection_for_successful_multisplit_trial(self):
        import tempfile
        from workflows.laboratory import LaboratoryWorkflow
        from research_state import ResearchState

        inspected_selection = {
            "dataset_id": "owner/data",
            "revision": "r1",
            "config_name": "default",
            "split": "train",
        }
        inspected = {
            "success": True,
            **inspected_selection,
            "dataset_schema": {"columns": {"value": "int64"}},
            "materialized_resources": [
                {"config_name": "default", "split": "train"},
                {"config_name": "default", "split": "test"},
            ],
            "selected_dataset": dict(inspected_selection),
        }
        actions = [
            ("search_huggingface_datasets", {
                "success": True,
                "results": [],
                "selected_dataset": dict(inspected_selection),
            }),
            ("inspect_selected_dataset", {
                **inspected,
                "selected_dataset": dict(inspected_selection),
            }),
            ("write_and_run_dataset_code", {
                "success": True,
                "code": "data = {'train': [1], 'test': [2]}",
                "output": "prepared both splits",
                "selection": deepcopy(inspected),
                "executed_dataset": executed_dataset(
                    inspected, inspected["materialized_resources"],
                ),
                "artifacts": [],
                "prepared_data_manifest": None,
            }),
        ]
        ml_calls = []
        def ml_turn(*args, **kwargs):
            ml_calls.append(kwargs)
            name, value = actions[len(ml_calls) - 1]
            return SimpleNamespace(tool_name=name, value=value)

        def sw_turn(*args, **kwargs):
            return SimpleNamespace(
                tool_name="send_dialogue",
                value={"success": True, "message": "Continue."},
            )

        with tempfile.TemporaryDirectory() as directory:
            workflow = LaboratoryWorkflow(
                ResearchState("query", "recording", "topic"),
                "model",
                "key",
                directory,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            with patch.object(
                workflow.ml_engineer,
                "data_preparation_turn",
                side_effect=ml_turn,
            ), patch.object(
                workflow.sw_engineer,
                "data_preparation_turn",
                side_effect=sw_turn,
            ):
                result_selection, submitted = workflow._prepare_dataset_collaboratively()

        self.assertEqual(len(ml_calls), 3)
        self.assertTrue(submitted["success"])
        self.assertEqual(result_selection, inspected)
        self.assertEqual(submitted["selection"], inspected)
        self.assertEqual(
            submitted["executed_dataset"]["resources"],
            inspected["materialized_resources"],
        )

    def test_workflow_submits_trial_that_succeeds_in_final_round(self):
        import tempfile
        from workflows.laboratory import LaboratoryWorkflow
        from research_state import ResearchState

        selection = {
            "dataset_id": "owner/data",
            "revision": "r1",
            "config_name": "default",
            "split": "train",
        }
        inspected = {
            "success": True,
            **selection,
            "dataset_schema": {"columns": {"value": "int64"}},
        }
        ml_calls = []
        sw_calls = []

        def ml_turn(*args, **kwargs):
            ml_calls.append(kwargs)
            index = len(ml_calls) - 1
            if index == 0:
                return SimpleNamespace(
                    tool_name="inspect_selected_dataset",
                    value={**inspected, "selected_dataset": dict(selection)},
                )
            if index == 11:
                return SimpleNamespace(
                    tool_name="write_and_run_dataset_code",
                    value={
                        "success": True,
                        "code": "data = [1, 2]",
                        "output": "prepared",
                        "selection": dict(selection),
                        "artifacts": [],
                        "prepared_data_manifest": None,
                    },
                )
            return SimpleNamespace(
                tool_name="send_dialogue",
                value={"success": True, "message": "Continue."},
            )

        def sw_turn(*args, **kwargs):
            sw_calls.append(kwargs)
            return SimpleNamespace(
                tool_name="send_dialogue",
                value={"success": True, "message": "Continue."},
            )

        with tempfile.TemporaryDirectory() as directory:
            workflow = LaboratoryWorkflow(
                ResearchState("query", "recording", "topic"),
                "model",
                "key",
                directory,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            with patch.object(
                workflow.ml_engineer,
                "data_preparation_turn",
                side_effect=ml_turn,
            ), patch.object(
                workflow.sw_engineer,
                "data_preparation_turn",
                side_effect=sw_turn,
            ):
                result_selection, submitted = workflow._prepare_dataset_collaboratively()

        self.assertEqual(len(ml_calls), 12)
        self.assertEqual(len(sw_calls), 12)
        self.assertEqual(result_selection["dataset_id"], "owner/data")
        self.assertTrue(submitted["success"])
        self.assertEqual(submitted["code"], "data = [1, 2]")

    def test_workflow_keeps_original_diagnostic_after_duplicate_trial(self):
        import tempfile
        from workflows.laboratory import LaboratoryWorkflow
        from research_state import ResearchState

        selected = {
            "dataset_id": "owner/data",
            "revision": "r1",
            "config_name": "default",
            "split": "train",
        }
        inspected = {
            "success": True,
            **selected,
            "dataset_schema": {"columns": {"value": "int64"}},
        }
        original_failure = {
            "success": False,
            "error_type": "python_exception",
            "error": "TypeError: string indices must be integers, not 'str'",
            "repair_hint": "Read the train and test entries from the split mapping.",
            "code": (
                "train_dataset, held_out_dataset = sampled_dataset.train_test_split()\n"
                "print(train_dataset[0]['text'])"
            ),
            "code_hash": "failed-code-hash",
            "output": "",
            "traceback": "ORIGINAL_TRACEBACK_SENTINEL",
            "failure_context": {
                "line": 2,
                "source": "print(train_dataset[0]['text'])",
                "variables": {"train_dataset": {"type": "str", "length": 5}},
            },
        }
        actions = [
            ("search_huggingface_datasets", {
                "success": True,
                "results": [],
                "selected_dataset": dict(selected),
            }),
            ("inspect_selected_dataset", {
                **inspected,
                "selected_dataset": dict(selected),
            }),
            ("write_and_run_dataset_code", original_failure),
            ("write_and_run_dataset_code", {
                "success": False,
                "error_type": "unchanged_failed_dataset_code",
                "error": "This exact data-preparation code already failed.",
                "original_error_type": original_failure["error_type"],
                "original_error": original_failure["error"],
                "original_repair_hint": original_failure["repair_hint"],
                "repair_hint": "Correct the original failure before retrying.",
                "code": original_failure["code"],
                "code_hash": original_failure["code_hash"],
                "output": "",
                "traceback": "",
            }),
            ("write_and_run_dataset_code", {
                "success": True,
                "code": "data = [1]",
                "output": "prepared",
                "selection": dict(selected),
                "artifacts": [],
                "prepared_data_manifest": None,
            }),
        ]
        ml_calls = []
        ml_prompts = []
        sw_prompts = []

        def ml_turn(*args, **kwargs):
            ml_calls.append(deepcopy(kwargs))
            name, value = actions[len(ml_calls) - 1]
            with patch.object(
                AgentRuntime, "run", return_value=SimpleNamespace(tool_name=name, value=value),
            ) as run:
                result = MLEngineerAgent.data_preparation_turn(workflow.ml_engineer, *args, **kwargs)
            ml_prompts.append(run.call_args.kwargs["user_prompt"])
            return result

        def sw_turn(*args, **kwargs):
            with patch.object(
                AgentRuntime, "run", return_value=SimpleNamespace(
                    tool_name="send_dialogue", value={"success": True, "message": "Continue."},
                ),
            ) as run:
                result = SWEngineerAgent.data_preparation_turn(workflow.sw_engineer, *args, **kwargs)
            sw_prompts.append(run.call_args.kwargs["user_prompt"])
            return result

        with tempfile.TemporaryDirectory() as directory:
            workflow = LaboratoryWorkflow(
                ResearchState("query", "recording", "topic"),
                "model",
                "key",
                directory,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            with patch.object(
                workflow.ml_engineer,
                "data_preparation_turn",
                side_effect=ml_turn,
            ), patch.object(
                workflow.sw_engineer,
                "data_preparation_turn",
                side_effect=sw_turn,
            ), patch("agents.HFDataSearch"), patch("agents.execution_capabilities", return_value={}):
                _, submitted = workflow._prepare_dataset_collaboratively()

        self.assertTrue(submitted["success"])
        self.assertEqual(
            ml_calls[3]["latest_execution_failure"]["error"],
            original_failure["error"],
        )
        self.assertEqual(
            ml_calls[4]["latest_execution_failure"]["error"],
            original_failure["error"],
        )
        for prompt, prefix in (
            (ml_prompts[4], "Previous ML command result: "),
            (sw_prompts[4], "Latest ML command result: "),
        ):
            with self.subTest(prefix=prefix):
                feedback = json.loads(next(
                    line[len(prefix):] for line in prompt.splitlines() if line.startswith(prefix)
                ))
                self.assertEqual(feedback["error_type"], "unchanged_failed_dataset_code")
                self.assertEqual(feedback["code_hash"], original_failure["code_hash"])
                self.assertIn("not executed", feedback["error"])
                self.assertIn("complete corrected program", feedback["repair_hint"])
                self.assertNotIn("code", feedback)
                self.assertNotIn("traceback", feedback)
                self.assertNotIn("output", feedback)
                self.assertNotIn("failure_context", feedback)
                self.assertEqual(prompt.count(json.dumps(original_failure["code"])), 1)
                self.assertEqual(prompt.count(original_failure["traceback"]), 1)
                failure_prefix = "Authoritative latest execution failure: "
                retained = json.loads(next(
                    line[len(failure_prefix):] for line in prompt.splitlines()
                    if line.startswith(failure_prefix)
                ))
                self.assertEqual(retained, {"tool": "write_and_run_dataset_code", **original_failure})
                self.assertEqual(retained["failure_context"], original_failure["failure_context"])
                self.assertEqual(prompt.count('"failure_context"'), 1)
        for prompt in ml_prompts[3:]:
            self.assertNotIn("Next action:", prompt)
            self.assertNotIn("Automatically selected local dataset:", prompt)

    def test_ml_tool_availability_follows_selection_and_inspection(self):
        selected = {
            "dataset_id": "owner/data",
            "revision": "r1",
            "config_name": "default",
            "split": "train",
        }
        cases = (
            (None, None, {"search_huggingface_datasets", "send_dialogue"}),
            (None, selected, {"inspect_selected_dataset"}),
            (dataset_inspection(), selected, {"write_and_run_dataset_code"}),
        )
        for inspected, selected_dataset, expected in cases:
            with self.subTest(inspected=inspected, selected=selected_dataset), patch(
                "agents.HFDataSearch"
            ), patch.object(AgentRuntime, "run", return_value=SimpleNamespace(value={})) as run:
                MLEngineerAgent("model", "key").data_preparation_turn(
                    "topic", "plan",  "Please proceed",
                    inspected_dataset=inspected,
                    selected_dataset=selected_dataset,
                    code_execution_timeout_seconds=17.5,
                )
                args = run.call_args.kwargs
                self.assertEqual(
                    args["tool_names_provider"](),
                    expected,
                )

    def test_ml_selection_comes_from_workflow_not_previous_feedback(self):
        stale = {"dataset_id": "old/data", "revision": "old", "config_name": "default", "split": "train"}
        selected = {**stale, "dataset_id": "current/data", "revision": "current"}
        for explicit in (None, selected):
            with self.subTest(selected=explicit), patch("agents.HFDataSearch") as search, patch.object(
                AgentRuntime, "run", return_value=SimpleNamespace(value={})
            ) as run:
                MLEngineerAgent("model", "key").data_preparation_turn(
                    "topic", "plan",  "proceed",
                    feedback={"selected_dataset": stale}, selected_dataset=explicit,
                    code_execution_timeout_seconds=17.5,
                )
                kwargs = run.call_args.kwargs
                if explicit is None:
                    self.assertEqual(kwargs["tool_names_provider"](), {"search_huggingface_datasets", "send_dialogue"})
                else:
                    self.assertEqual(kwargs["tool_names_provider"](), {"inspect_selected_dataset"})
                    kwargs["registry"].execute("inspect_selected_dataset", {})
                    search.return_value.inspect_dataset.assert_called_once_with(
                        "current/data", "current", "default", "train",
                    )

    def test_workflow_preserves_selected_dataset_after_failed_trial(self):
        import tempfile
        from workflows.laboratory import LaboratoryWorkflow
        from research_state import ResearchState
        inspected = {
            "success": True, "dataset_id": "owner/data", "revision": "r1",
            "config_name": "default", "split": "train",
            "dataset_schema": {"columns": {"value": "int64"}},
        }
        selected = {
            "dataset_id": "owner/data", "revision": "r1",
            "config_name": "default", "split": "train",
        }
        actions = [
            ("search_huggingface_datasets", {"success": True, "results": [], "selected_dataset": selected}),
            ("inspect_selected_dataset", {**inspected, "selected_dataset": selected}),
            ("write_and_run_dataset_code", {"success": False, "code": "", "error": "Provide Python code"}),
            ("write_and_run_dataset_code", {"success": False, "code": "raise ValueError()", "error": "bad"}),
            ("write_and_run_dataset_code", {
                "success": True, "code": "data = [1]", "output": "ok",
                "selection": dict(selected),
            }),
        ]
        observed_selected = []

        def ml_turn(*args, **kwargs):
            observed_selected.append(kwargs["selected_dataset"])
            name, value = actions[len(observed_selected) - 1]
            return SimpleNamespace(tool_name=name, value=value)

        def sw_turn(*args, **kwargs):
            return SimpleNamespace(
                tool_name="send_dialogue",
                value={"success": True, "message": "Continue."},
            )

        with tempfile.TemporaryDirectory() as directory:
            workflow = LaboratoryWorkflow(
                ResearchState("query", "recording", "topic"), "model", "key", directory,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            with patch.object(workflow.ml_engineer, "data_preparation_turn", side_effect=ml_turn), patch.object(
                workflow.sw_engineer, "data_preparation_turn", side_effect=sw_turn
            ):
                selection, result = workflow._prepare_dataset_collaboratively()
        self.assertEqual(observed_selected, [None, selected, selected, selected, selected])
        self.assertTrue(result["success"])
        self.assertEqual(selection["revision"], "r1")


    def test_inspection_feedback_is_deduplicated_only_when_identical(self):
        inspected = {
            "success": True, "dataset_id": "owner/data", "revision": "r1",
            "config_name": "default", "split": "train",
            "dataset_schema": {"row_count": 2, "columns": {"SCHEMA_SENTINEL": "int64"}},
            "materialized_resources": [],
        }
        matching = {"tool": "inspect_selected_dataset", **deepcopy(inspected)}
        cases = [
            ("identical", matching, inspected, True),
            ("failed", {**matching, "success": False, "error": "inspection failed"}, inspected, False),
            ("missing", matching, None, False),
            ("other_tool", {**matching, "tool": "write_and_run_dataset_code"}, inspected, False),
            ("extra_feedback", {**matching, "warning": "retain this"}, inspected, False),
            ("different_schema", {**matching, "dataset_schema": {"columns": {"other": "float64"}}}, inspected, False),
            ("failed_snapshot", matching, {**inspected, "success": False}, False),
        ]
        for field in ("dataset_id", "revision", "config_name", "split"):
            cases.append((field, {**matching, field: "different"}, inspected, False))
        for role in ("ml", "sw"):
            for label, feedback, snapshot, compact in cases:
                with self.subTest(role=role, case=label):
                    original = deepcopy((feedback, snapshot))
                    with patch("agents.HFDataSearch"), patch("agents.execution_capabilities", return_value={}), patch.object(
                        AgentRuntime, "run", return_value=SimpleNamespace(value={})
                    ) as run:
                        if role == "ml":
                            MLEngineerAgent("model", "key").data_preparation_turn(
                                "topic", "plan",  "continue",
                                feedback=feedback, inspected_dataset=snapshot,
                                code_execution_timeout_seconds=17.5,
                            )
                            prefix = "Previous ML command result: "
                        else:
                            SWEngineerAgent("model", "key").data_preparation_turn(
                                "topic", "plan",  ml_result=feedback,
                                inspected_dataset=snapshot,

                            )
                            prefix = "Latest ML command result: "
                    prompt = run.call_args.kwargs["user_prompt"]
                    rendered = json.loads(next(line[len(prefix):] for line in prompt.splitlines() if line.startswith(prefix)))
                    if compact:
                        self.assertNotIn("dataset_schema", rendered)
                        self.assertTrue(rendered["success"])
                        self.assertIn("Authoritative inspected dataset", rendered["message"])
                        self.assertEqual(prompt.count("SCHEMA_SENTINEL"), 1)
                    else:
                        self.assertEqual(rendered, feedback)
                    schema_prefix = "Authoritative inspected dataset: "
                    rendered_snapshot = json.loads(next(line[len(schema_prefix):] for line in prompt.splitlines() if line.startswith(schema_prefix)))
                    self.assertEqual(rendered_snapshot, snapshot)
                    self.assertEqual((feedback, snapshot), original)

    def test_data_preparation_system_prompts_only_describe_roles_and_tasks(self):
        self.assertIn("machine learning engineer", ml_dataset_selection_prompt())
        self.assertIn("software engineer", sw_data_preparation_prompt())
        for prompt in (ml_dataset_selection_prompt(), sw_data_preparation_prompt()):
            self.assertLess(len(prompt), 250)
            self.assertNotIn("write_and_run_dataset_code", prompt)
            self.assertNotIn("PYTHON_SEED", prompt)

    def test_data_preparation_prompts_do_not_receive_full_plan(self):
        requirements = "DATA_REQUIREMENTS_SENTINEL"
        full_plan = "FULL_PLAN_SENTINEL RandomForestClassifier metrics baselines"
        completed = SimpleNamespace(value={})
        with patch("agents.HFDataSearch"), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            MLEngineerAgent("model", "key").data_preparation_turn(
                "topic", requirements,  "message",
                code_execution_timeout_seconds=17.5,
            )
            ml_prompt = run.call_args.kwargs["user_prompt"]
            code_schema = run.call_args.kwargs["registry"].schemas_for({"write_and_run_dataset_code"})[0]
            description = code_schema["function"]["parameters"]["properties"]["code"]["description"]
            self.assertIn("datasets.load_dataset", description)
            self.assertIn("fit preprocessing on training data only", description)
            self.assertIn("Do not train or evaluate models", description)
            SWEngineerAgent("model", "key").data_preparation_turn(
                "topic", requirements,

            )
            sw_prompt = run.call_args.kwargs["user_prompt"]

        for prompt in (ml_prompt, sw_prompt):
            self.assertIn(requirements, prompt)
            self.assertNotIn(full_plan, prompt)
            self.assertNotIn("Preparation constraints:", prompt)
            self.assertNotIn("fit preprocessing on training data only", prompt)
            self.assertIn("PYTHON_SEED=", prompt)
            self.assertIn("SKLEARN_SEED=", prompt)

    def test_latest_execution_failure_keeps_current_sw_message_and_hides_stale_feedback(self):
        latest_failure = {
            "tool": "write_and_run_dataset_code",
            "success": False,
            "error_type": "python_exception",
            "error": "CURRENT_ERROR",
            "traceback": "CURRENT_TRACEBACK",
            "code": "CURRENT_CODE",
            "output": "CURRENT_OUTPUT",
            "repair_hint": "CURRENT_REPAIR_HINT",
        }
        stale_feedback = {
            "tool": "write_and_run_dataset_code",
            "success": False,
            "error": "OLD_ERROR",
            "traceback": "OLD_TRACEBACK",
        }
        completed = SimpleNamespace(value={})
        with patch("agents.HFDataSearch"), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            MLEngineerAgent("model", "key").data_preparation_turn(
                "topic", "plan",  "CURRENT_SW_REPAIR_ADVICE",
                feedback=stale_feedback,
                latest_execution_failure=latest_failure,
                code_execution_timeout_seconds=17.5,
            )
            ml_prompt = run.call_args.kwargs["user_prompt"]
            SWEngineerAgent("model", "key").data_preparation_turn(
                "topic", "plan",
                feedback={"message": "OLD_SW_RESULT"},
                ml_result=stale_feedback,
                latest_execution_failure=latest_failure,

            )
            sw_prompt = run.call_args.kwargs["user_prompt"]

        for prompt in (ml_prompt, sw_prompt):
            self.assertIn("CURRENT_ERROR", prompt)
            self.assertIn("CURRENT_TRACEBACK", prompt)
            self.assertIn("CURRENT_CODE", prompt)
            self.assertIn("CURRENT_REPAIR_HINT", prompt)
            self.assertNotIn("OLD_ERROR", prompt)
            self.assertNotIn("OLD_TRACEBACK", prompt)
        self.assertIn("CURRENT_SW_REPAIR_ADVICE", ml_prompt)
        self.assertNotIn("OLD_SW_RESULT", sw_prompt)

    def test_preparation_key_errors_receive_shared_diagnostic_hint(self):
        cases = (
            (
                "transformed_dataframe",
                "df = pd.get_dummies(df, columns=['group_label'])\n"
                "value = df['group_label']\n",
                {"group_label": "string"},
            ),
            (
                "no_transformation",
                "value = df['group_label']\n",
                {"group_label": "string"},
            ),
            (
                "column_absent_from_raw_schema",
                "value = df['group_label']\n",
                {"actual_label": "string"},
            ),
            (
                "ordinary_dictionary",
                "options = {'mode': 'train'}\nvalue = options['group_label']\n",
                {"value": "int64"},
            ),
        )
        for name, code, columns in cases:
            with self.subTest(case=name):
                execution = {
                    "success": False,
                    "output": "",
                    "error_type": "python_exception",
                    "error": "KeyError: 'group_label'",
                    "traceback": (
                        'Traceback\n  File "<string>", line 2, in <module>\n'
                        "KeyError: 'group_label'\n"
                    ),
                    "exit_code": 0,
                    "artifacts": [],
                    "replay_events": [],
                    "prepared_data_manifest": None,
                }
                inspected_dataset = dataset_inspection(
                    success=True,
                    dataset_schema={"columns": columns},
                )
                with patch("data_preparation.execute_code", return_value=execution):
                    result = _run_preparation_code(code, ".", inspected_dataset, code_execution_timeout_seconds=17.5)

                self.assertFalse(result["success"])
                self.assertEqual(result["failure_category"], "missing_key_or_column")
                self.assertEqual(
                    result["repair_hint"],
                    runtime_data_repair_hint("missing_key_or_column"),
                )
                self.assertEqual(result["code"], code)
                self.assertEqual(result["error"], execution["error"])
                self.assertEqual(result["traceback"], execution["traceback"])

    def test_preparation_preserves_execution_failure_context(self):
        code = "train_dataset = 'train'\nprint(train_dataset[0]['text'])"
        failure_context = {
            "line": 2,
            "source": "print(train_dataset[0]['text'])",
            "variables": {"train_dataset": {"type": "str", "length": 5}},
        }
        execution = {
            "success": False,
            "output": "",
            "error_type": "python_exception",
            "error": "TypeError: string indices must be integers, not 'str'",
            "traceback": "original traceback",
            "exit_code": 0,
            "failure_context": failure_context,
        }
        with patch("data_preparation.execute_code", return_value=execution):
            result = _run_preparation_code(code, ".", dataset_inspection(), code_execution_timeout_seconds=17.5)

        self.assertFalse(result["success"])
        self.assertEqual(result["failure_context"], failure_context)
        self.assertIsNot(result["failure_context"], failure_context)
        self.assertEqual(result["code"], code)
        self.assertEqual(result["error"], execution["error"])
        self.assertEqual(result["traceback"], execution["traceback"])

    def test_preparation_syntax_error_has_local_repair_hint(self):
        code = "first = 1\n second = 2\nthird = 3\n"
        execution = {
            "success": False,
            "output": "",
            "error_type": "validation_error",
            "error": "Generated code is invalid Python: unexpected indent",
            "traceback": "",
            "exit_code": None,
            "artifacts": [],
            "replay_events": [],
            "prepared_data_manifest": None,
        }
        with patch("data_preparation.execute_code", return_value=execution):
            result = _run_preparation_code(code, ".", dataset_inspection(), code_execution_timeout_seconds=17.5)

        self.assertFalse(result["success"])
        self.assertIn("line 2", result["repair_hint"])
        self.assertIn("2:  second = 2", result["repair_hint"])

    def test_preparation_runtime_data_errors_have_specific_repair_hints(self):
        errors = {
            "ValueError: Input X contains infinity": "non_finite_numeric_input",
            "ValueError: expected sequence of length 3 at dim 3 (got 4)": (
                "ragged_array_input"
            ),
            "TypeError: list indices must be integers or slices, not str": (
                "list_of_records_access_error"
            ),
        }
        for error, category in errors.items():
            with self.subTest(error=error):
                details = {
                    "success": False,
                    "output": "",
                    "error_type": "python_exception",
                    "error": error,
                    "traceback": "",
                    "exit_code": 0,
                    "artifacts": [],
                    "replay_events": [],
                    "prepared_data_manifest": None,
                }
                with patch("data_preparation.execute_code", return_value=details):
                    result = _run_preparation_code("print('trial')", ".", dataset_inspection(), code_execution_timeout_seconds=17.5)
                self.assertEqual(result["failure_category"], category)
                self.assertNotIn("Use the traceback", result["repair_hint"])

    def test_ml_engineer_exposes_structured_dataset_inspection(self):
        completed = SimpleNamespace(value={})
        search = MagicMock()
        search.inspect_dataset.return_value = {
            "dataset_id": "owner/data",
            "revision": "revision",
            "config_name": "default",
            "split": "train",
            "dataset_schema": {
                "row_count": 2,
                "columns": {"value": "int64"},
            },
        }
        with patch("agents.HFDataSearch", return_value=search), patch.object(
            AgentRuntime,
            "run",
            return_value=completed,
        ) as run:
            MLEngineerAgent("model", "key").data_preparation_turn(
                "topic",
                "plan",

                "inspect data",
                selected_dataset={
                    "dataset_id": "owner/data", "revision": "revision",
                    "config_name": "default", "split": "train",
                },
                code_execution_timeout_seconds=17.5,
            )

        registry = run.call_args.kwargs["registry"]
        inspected = registry.execute(
            "inspect_selected_dataset",
            {},
        )
        self.assertTrue(inspected["success"])
        self.assertEqual(inspected["dataset_schema"]["row_count"], 2)
        search.inspect_dataset.assert_called_once_with(
            "owner/data",
            "revision",
            "default",
            "train",
        )

    def test_sw_only_exposes_dialogue_in_every_preparation_state(self):
        inspected = dataset_inspection()
        failure = {
            "tool": "write_and_run_dataset_code", "success": False,
            "code": "print(missing)", "error": "NameError: missing",
        }
        cases = (
            ("initial", {}),
            ("selected", {"ml_result": {
                "tool": "search_huggingface_datasets", "success": True,
                "selected_dataset": inspected,
            }}),
            ("inspected", {"inspected_dataset": inspected}),
            ("failed", {
                "inspected_dataset": inspected, "ml_result": failure,
                "latest_execution_failure": failure,
            }),
            ("successful", {
                "inspected_dataset": inspected,
                "ml_result": {
                    "tool": "write_and_run_dataset_code", "success": True,
                    "code": "data = [1]", "selection": inspected,
                },
            }),
        )
        for state, kwargs in cases:
            with self.subTest(state=state), patch.object(
                AgentRuntime, "run", return_value=SimpleNamespace(value={}),
            ) as run, patch("agents._run_preparation_code") as prepare, patch(
                "data_preparation.execute_code",
            ) as execute:
                SWEngineerAgent("model", "key").data_preparation_turn(
                    "topic", "requirements",
                     **kwargs,
                )
                options = run.call_args.kwargs
                registry = options["registry"]
                self.assertEqual(registry.names, {"send_dialogue"})
                self.assertEqual(options["terminal_tools"], {"send_dialogue"})
                self.assertTrue(options["return_after_tool"])
                advice = "Use the inspected field and fix the failing access."
                self.assertEqual(
                    registry.execute("send_dialogue", {"message": advice}),
                    {"success": True, "message": advice},
                )
                prepare.assert_not_called()
                execute.assert_not_called()

    def test_sw_rejects_code_tools_without_execution_even_after_inspection(self):
        from inference import ModelToolCall, ToolModelResponse

        for inspected in (None, dataset_inspection()):
            for tool_name in ("submit_dataset_code", "write_and_run_dataset_code"):
                with self.subTest(inspected=bool(inspected), tool=tool_name):
                    responses = [
                        ToolModelResponse("", [ModelToolCall(
                            "unavailable", tool_name,
                            json.dumps({"code": "print('data')"}),
                        )]),
                        ToolModelResponse("", [ModelToolCall(
                            "dialogue", "send_dialogue",
                            json.dumps({"message": "Please correct the failing code."}),
                        )]),
                    ]
                    with patch(
                        "agent_runtime.query_model_with_tools", side_effect=responses,
                    ) as query, patch("agents._run_preparation_code") as prepare, patch(
                        "data_preparation.execute_code",
                    ) as execute:
                        result = SWEngineerAgent("model", "key").data_preparation_turn(
                            "topic", "requirements",  inspected_dataset=inspected,

                        )
                    self.assertEqual(result.tool_name, "send_dialogue")
                    self.assertEqual(result.value, {
                        "success": True, "message": "Please correct the failing code.",
                    })
                    self.assertEqual(query.call_count, 2)
                    for model_call in query.call_args_list:
                        self.assertEqual(
                            {tool["function"]["name"] for tool in model_call.kwargs["tools"]},
                            {"send_dialogue"},
                        )
                    prepare.assert_not_called()
                    execute.assert_not_called()

    def test_ml_preparation_forwards_configured_execution_timeout(self):
        from inference import ModelToolCall, ToolModelResponse

        code = "print('prepared')"
        inspected = dataset_inspection()
        execution = {
            "success": True, "output": "prepared", "error_type": None,
            "error": None, "traceback": "", "exit_code": 0,
            "executed_dataset": executed_dataset(inspected),
        }
        for timeout in (17.5, 123):
            with self.subTest(timeout=timeout):
                response = ToolModelResponse("", [ModelToolCall(
                    "call", "write_and_run_dataset_code", json.dumps({"code": code}),
                )])
                with patch("agents.HFDataSearch"), patch(
                    "agent_runtime.query_model_with_tools", return_value=response,
                ), patch(
                    "data_preparation.execute_code", return_value=execution,
                ) as execute:
                    result = MLEngineerAgent("model", "key").data_preparation_turn(
                        "topic", "requirements",  "Prepare data.",
                        workdir=".", inspected_dataset=inspected,
                        code_execution_timeout_seconds=timeout,
                    )
                self.assertTrue(result.value["success"])
                execute.assert_called_once_with(
                    code, timeout=timeout, workdir=".", return_details=True,
                    expected_hf_dataset=inspected,
                )

    def test_preparation_alternates_agents_and_successful_ml_execution_finishes(self):
        from inference import ModelToolCall, ToolModelResponse
        from workflows.laboratory import LaboratoryWorkflow
        from research_state import ResearchState
        import json
        import tempfile

        def response(name, arguments):
            return ToolModelResponse("", [ModelToolCall(
                "call", name, json.dumps(arguments)
            )])

        fixed = "data = [1, 2]\nprint(len(data))"
        responses = [
            response("send_dialogue", {"message": "Find data for the plan."}),
            response("search_huggingface_datasets", {"query": "data", "max_results": 5}),
            response("send_dialogue", {"message": "Inspect the selected dataset."}),
            response("inspect_selected_dataset", {}),
            response("send_dialogue", {"message": "Test a preparation program."}),
            response("write_and_run_dataset_code", {"code": fixed}),
        ]
        calls = []
        search = MagicMock()
        search.retrieve_ds.return_value = [{
            "id": "owner/data",
            "revision": "r1",
            "snapshot_status": "complete",
            "configs": ["default"],
            "splits": ["train"],
            "materialized": {"default": {"train": {"rows": 2}}},
        }]
        search.inspect_dataset.return_value = {
            "dataset_id": "owner/data",
            "config_name": "default",
            "split": "train",
            "revision": "r1",
            "available_configs": ["default"],
            "available_splits": ["train"],
            "dataset_schema": {
                "row_count": 2,
                "columns": {"value": "int64"},
            },
            "materialized_resources": [],
            "materialized_schemas": {},
        }
        details = {
            "success": True, "error_type": None, "error": None, "traceback": "", "exit_code": 0,
            "output": "2", "artifacts": [{"name": "data"}],
            "executed_dataset": executed_dataset(search.inspect_dataset.return_value),
            "replay_events": [{
                "tool": "huggingface.load_dataset", "success": True,
                "request": {"dataset_id": "owner/data", "config_name": "default", "split": "train"},
                "object_revision": "r1",
            }],
        }

        def respond(**kwargs):
            calls.append(kwargs)
            if len(calls) > len(responses):
                raise AssertionError(
                    "Unexpected LLM call with tools: "
                    + repr([item["function"]["name"] for item in kwargs["tools"]])
                )
            return responses[len(calls) - 1]

        with tempfile.TemporaryDirectory() as workdir, patch(
            "agents.HFDataSearch", return_value=search
        ), patch("agent_runtime.query_model_with_tools", side_effect=respond), patch(
            "data_preparation.execute_code",
            return_value=details,
        ) as execute:
            workflow = LaboratoryWorkflow(
                ResearchState("query_001", "recording", "topic", plan="plan"),
                "model", "key", workdir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            selection, submitted = workflow._prepare_dataset_collaboratively()
        self.assertEqual(submitted["code"], fixed)
        self.assertTrue(submitted["success"])
        self.assertEqual(selection["dataset_id"], "owner/data")
        self.assertEqual(selection["revision"], "r1")
        self.assertEqual(execute.call_count, 1)
        for call in execute.call_args_list:
            self.assertEqual(str(call.kwargs["workdir"]), workdir)
            self.assertEqual(call.kwargs["timeout"], 17.5)
            self.assertEqual(call.kwargs["expected_hf_dataset"], selection)
        expected_sw = {"send_dialogue"}
        expected_ml = {
            "search_huggingface_datasets",
            "send_dialogue",
        }
        expected_tools = [
            expected_sw,
            expected_ml,
            expected_sw,
            {"inspect_selected_dataset"},
            expected_sw,
            {"write_and_run_dataset_code"},
        ]
        self.assertEqual(len(calls), len(expected_tools))
        for index, call in enumerate(calls):
            names = {item["function"]["name"] for item in call["tools"]}
            self.assertEqual(names, expected_tools[index])
            if index == 3:
                inspect_schema = next(
                    item["function"]
                    for item in call["tools"]
                    if item["function"]["name"] == "inspect_selected_dataset"
                )
                self.assertEqual(inspect_schema["parameters"]["required"], [])
        self.assertIn("owner/data", calls[2]["messages"][1]["content"])
        self.assertIn("row_count", calls[4]["messages"][1]["content"])

    def test_ml_trial_failure_returns_to_workflow_without_internal_retry(self):
        from inference import ModelToolCall, ToolModelResponse
        import json
        code = "raise RuntimeError('failure')"
        with patch("agents.HFDataSearch"), patch(
            "data_preparation.execute_code", return_value={"success": False, "output": "failure",
                "error_type": "python_exception", "error": "failure", "traceback": "", "exit_code": 0}
        ), patch("agent_runtime.query_model_with_tools", return_value=ToolModelResponse(
            "", [ModelToolCall("call", "write_and_run_dataset_code", json.dumps({"code": code}))]
        )) as query:
            result = MLEngineerAgent("model", "key").data_preparation_turn(
                "topic", "plan",  "test code", workdir=".",
                selected_dataset=dataset_inspection(),
                inspected_dataset=dataset_inspection(),
                code_execution_timeout_seconds=17.5,
            )
        self.assertFalse(result.value["success"])
        self.assertEqual(result.value["code"], code)
        query.assert_called_once()

    def test_ml_engineer_rejects_unchanged_failed_trial_without_execution(self):
        from inference import ModelToolCall, ToolModelResponse

        first_code = "raise RuntimeError('failure')\n"
        repeated_code = "raise RuntimeError('failure')\n\n"
        first_response = ToolModelResponse(
            "",
            [ModelToolCall(
                "call",
                "write_and_run_dataset_code",
                json.dumps({"code": first_code}),
            )],
        )
        repeated_response = ToolModelResponse(
            "",
            [ModelToolCall(
                "call",
                "write_and_run_dataset_code",
                json.dumps({"code": repeated_code}),
            )],
        )
        execution = {
            "success": False,
            "output": "failure output",
            "error_type": "python_exception",
            "error": "RuntimeError: failure",
            "traceback": "Traceback: failure",
            "exit_code": 0,
            "artifacts": [],
            "replay_events": [],
            "prepared_data_manifest": None,
        }
        failed_failures = {}
        with patch("agents.HFDataSearch"), patch(
            "data_preparation.execute_code",
            return_value=execution,
        ) as execute_code, patch(
            "agent_runtime.query_model_with_tools",
            side_effect=[first_response, repeated_response],
        ):
            first = MLEngineerAgent("model", "key").data_preparation_turn(
                "topic",
                "requirements",

                "try code",
                workdir=".",
                selected_dataset=dataset_inspection(),
                inspected_dataset=dataset_inspection(),
                failed_trial_code_failures=failed_failures,
                code_execution_timeout_seconds=17.5,
            )
            repeated = MLEngineerAgent("model", "key").data_preparation_turn(
                "topic",
                "requirements",

                "try code",
                workdir=".",
                selected_dataset=dataset_inspection(),
                inspected_dataset=dataset_inspection(),
                failed_trial_code_failures=failed_failures,
                code_execution_timeout_seconds=17.5,
            )

        self.assertFalse(first.value["success"])
        self.assertEqual(execute_code.call_count, 1)
        self.assertFalse(repeated.value["success"])
        self.assertEqual(
            repeated.value["error_type"],
            "unchanged_failed_dataset_code",
        )
        self.assertIn("substantive code change", repeated.value["repair_hint"])
        self.assertIn("RuntimeError: failure", repeated.value["repair_hint"])
        self.assertEqual(repeated.value["original_error_type"], "python_exception")
        self.assertEqual(repeated.value["original_error"], "RuntimeError: failure")
        self.assertEqual(repeated.value["code"], repeated_code)













    def test_postdoc_plan_prompt_requires_phd_consultation_and_full_plan(self):
        agent = PostdocAgent("model", "key")
        completed = SimpleNamespace(value={"message": "question"})
        with patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.plan_turn("topic", "literature", "")

        prompt = run.call_args.kwargs["system_prompt"]
        user_prompt = run.call_args.kwargs["user_prompt"]
        self.assertIn("computer science postdoctoral researcher", prompt)
        self.assertIn("minimal, feasible experiment plan", prompt)
        self.assertIn("Research topic: topic", user_prompt)
        self.assertIn("Latest PhD feedback: None", user_prompt)
        self.assertNotIn("Planning consultation status", user_prompt)
        self.assertEqual(run.call_args.kwargs["tool_names_provider"](), {"send_dialogue"})

        registry = run.call_args.kwargs["registry"]
        schemas = {item["function"]["name"]: item["function"] for item in registry.schemas}
        self.assertIn("Draft an experiment plan", schemas["send_dialogue"]["description"])
        self.assertEqual(
            schemas["submit_plan"]["parameters"]["required"],
            ["experiment_plan", "data_plan"],
        )
        self.assertIn("data-preparation plan", schemas["submit_plan"]["description"])
        self.assertNotIn("submit_plan", run.call_args.kwargs["tool_names_provider"]())

    def test_postdoc_plan_prompt_marks_received_feedback(self):
        agent = PostdocAgent("model", "key")
        completed = SimpleNamespace(value={"experiment_plan": "plan"})
        with patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.plan_turn("topic", "literature", "PhD feedback")
        self.assertIn("Latest PhD feedback: PhD feedback", run.call_args.kwargs["user_prompt"])
        self.assertEqual(
            run.call_args.kwargs["tool_names_provider"](),
            {"send_dialogue"},
        )
        registry = run.call_args.kwargs["registry"]
        self.assertFalse(registry.execute(
            "submit_plan", {"experiment_plan": "full plan", "data_plan": " "},
        )["success"])
        self.assertFalse(registry.execute(
            "submit_plan", {"experiment_plan": " ", "data_plan": "abstract requirements"},
        )["success"])
        self.assertTrue(registry.execute(
            "submit_plan", {"experiment_plan": "full plan", "data_plan": "abstract requirements"},
        )["success"])

    def test_postdoc_plan_does_not_unlock_submission_without_nonempty_phd_feedback(self):
        for dialogue in ("", " \n\t "):
            with self.subTest(dialogue=dialogue), patch.object(AgentRuntime, "run") as run:
                PostdocAgent("model", "key").plan_turn("topic", "literature", dialogue)

                kwargs = run.call_args.kwargs
                registry = kwargs["registry"]
                self.assertIn("Latest PhD feedback: None", kwargs["user_prompt"])
                self.assertEqual(kwargs["tool_names_provider"](), {"send_dialogue"})
                registry.execute("send_dialogue", {"message": "Please review the plan"})
                self.assertEqual(kwargs["tool_names_provider"](), {"send_dialogue"})

    def test_postdoc_plan_runtime_waits_for_phd_feedback_before_submission(self):
        from agent_runtime import ToolRegistry
        from inference import ModelToolCall, ToolModelResponse

        plan = {"experiment_plan": "reviewed plan", "data_plan": "abstract requirements"}
        feedback = "Keep the classifier and evaluation split fixed."

        def tool_call(call_id, name, arguments):
            return ModelToolCall(call_id, name, json.dumps(arguments))

        responses = [
            ToolModelResponse("", [tool_call("premature", "submit_plan", plan)]),
            ToolModelResponse("", [
                tool_call("mixed_dialogue", "send_dialogue", {"message": "Premature consultation"}),
                tool_call("mixed_submit", "submit_plan", plan),
            ]),
            ToolModelResponse("", [tool_call(
                "consult", "send_dialogue", {"message": "Please review this plan"},
            )]),
            ToolModelResponse("", [tool_call(
                "feedback", "send_dialogue", {"message": feedback, "approved": True, "required_changes": []},
            )]),
            ToolModelResponse("", [tool_call("submit", "submit_plan", plan)]),
        ]
        executed_tools = []
        original_execute = ToolRegistry.execute

        def execute(registry, name, arguments):
            executed_tools.append(name)
            return original_execute(registry, name, arguments)

        with patch(
            "agent_runtime.query_model_with_tools", side_effect=responses,
        ) as query, patch.object(ToolRegistry, "execute", new=execute):
            postdoc = PostdocAgent("model", "key")
            consultation = postdoc.plan_turn("topic", "literature", "")
            self.assertEqual(consultation.tool_name, "send_dialogue")
            self.assertEqual(executed_tools, ["send_dialogue"])
            actual_feedback = PhDStudentAgent("model", "key").plan_feedback(
                "topic", "literature", consultation.value["message"],
            )
            result = postdoc.plan_turn("topic", "literature", actual_feedback, approved=actual_feedback["approved"])

        self.assertEqual(result.tool_name, "submit_plan")
        self.assertEqual(result.value, {"success": True, "plan": plan["experiment_plan"], "data_requirements": plan["data_plan"]})
        self.assertEqual(executed_tools, ["send_dialogue", "send_dialogue", "submit_plan"])
        exposed_tools = [
            {tool["function"]["name"] for tool in invocation.kwargs["tools"]}
            for invocation in query.call_args_list
        ]
        self.assertEqual(exposed_tools, [
            {"send_dialogue"}, {"send_dialogue"}, {"send_dialogue"},
            {"send_dialogue"}, {"submit_plan"},
        ])
        self.assertIn(feedback, query.call_args_list[-1].kwargs["messages"][1]["content"])

    def test_phd_plan_feedback_restores_research_planning_role(self):
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"message": "feedback"})
        with patch.object(
            AgentRuntime, "run_one", return_value=completed
        ) as run:
            agent.plan_feedback("topic", "literature", "proposal")
        prompt = run.call_args.kwargs["system_prompt"]
        self.assertIn("computer science PhD student", prompt)
        self.assertIn("only changes necessary", prompt)
        self.assertIn("Proposed plan: proposal", run.call_args.kwargs["user_prompt"])
        schema = run.call_args.kwargs["registry"].schemas[0]["function"]
        self.assertIn("approve it if sufficient", schema["description"])
        self.assertIn("use literature as background", schema["parameters"]["properties"]["message"]["description"])
        self.assertEqual(schema["parameters"]["required"], ["message", "approved", "required_changes"])

    def test_planning_preserves_context_and_requires_consistent_approval(self):
        history = [{"round": 1, "plan": "CNN with 32 filters", "feedback": {"approved": False}}]
        with patch.object(AgentRuntime, "run") as run:
            PostdocAgent("model", "key").plan_turn(
                "topic", "literature", {"approved": True}, approved=True,
                previous_plan="CNN with 32 filters", planning_history=history,
                round_number=9, remaining_reviews=0, seeds={"sklearn": 123},
            )
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["tool_names_provider"](), {"submit_plan"})
        self.assertIn("Previous complete plan: CNN with 32 filters", kwargs["user_prompt"])
        self.assertIn(json.dumps(history), kwargs["user_prompt"])
        self.assertIn("SKLEARN_SEED=123", kwargs["user_prompt"])
        self.assertIn("remaining reviews: 0", kwargs["user_prompt"])
        with patch.object(AgentRuntime, "run_one") as review:
            PhDStudentAgent("model", "key").plan_feedback(
                "topic", "literature", "plan", planning_history=history,
            )
        registry = review.call_args.kwargs["registry"]
        for approved, changes, valid in (
            (True, [], True), (False, ["Specify split"], True),
            (True, ["Specify split"], False), (False, [], False),
            (False, [" "], False),
        ):
            self.assertEqual(registry.execute("send_dialogue", {
                "message": "Review", "approved": approved, "required_changes": changes,
            })["success"], valid)

    def test_interpretation_tools_preserve_contracts_and_evidence_requirements(self):
        phd = PhDStudentAgent("model", "key")
        postdoc = PostdocAgent("model", "key")
        completed = SimpleNamespace(value={"message": "feedback"})
        with patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            phd.interpretation_turn("topic", "plan", "code", "output", "")
            phd_prompt = run.call_args.kwargs["system_prompt"]
            phd_user = run.call_args.kwargs["user_prompt"]
            self.assertIn("PhD student", phd_prompt)
            self.assertLess(len(phd_prompt), 250)
            self.assertNotIn("send_dialogue", phd_prompt)
            self.assertNotIn("at most one", phd_user)
            self.assertIn("Current stage: review", phd_user)
            self.assertIn("Interpretation history:", phd_user)
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(run.call_args.kwargs["tool_names_provider"](), {"send_dialogue"})

            phd.interpretation_turn(
                "topic",
                "plan",
                "code",
                "output",
                "Postdoc feedback",
                stage="supplement",
                workflow_notice="Use existing evidence.",
            )
            reviewed_user = run.call_args.kwargs["user_prompt"]
            self.assertIn("Current stage: supplement", reviewed_user)
            self.assertIn("Use existing evidence.", reviewed_user)
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(
                run.call_args.kwargs["tool_names_provider"](),
                {"request_supplementary_experiment"},
            )
            request_tool = registry.schemas_for({"request_supplementary_experiment"})[0]["function"]
            request_schema = request_tool["parameters"]
            self.assertEqual(set(request_schema["properties"]), {
                "objective", "reason", "required_outputs", "constraints",
            })
            self.assertEqual(set(request_schema["required"]), set(request_schema["properties"]))
            self.assertFalse(request_schema["additionalProperties"])
            self.assertIn("observed values", request_schema["properties"]["objective"]["description"])
            self.assertIn("original experiment unchanged", request_schema["properties"]["constraints"]["description"])
            final_schema = registry.schemas_for({"submit_interpretation"})[0]["function"]["parameters"]
            final_description = final_schema["properties"]["interpretation"]["description"]
            for requirement in ("English", "metrics", "comparisons", "anomalies", "limitations", "hypothesis", "literature", "significance"):
                self.assertIn(requirement, final_description)
            self.assertEqual(registry.execute("submit_interpretation", {"interpretation": "Observed conclusion"}), {
                "success": True, "interpretation": "Observed conclusion",
            })
            request = registry.execute(
                "request_supplementary_experiment",
                {
                    "objective": "check robustness",
                    "reason": "one split is insufficient",
                    "required_outputs": ["mean F1"],
                    "constraints": ["preserve original experiment"],
                },
            )
            self.assertTrue(request["success"])
            self.assertEqual(request["request"]["objective"], "check robustness")

            phd.interpretation_turn(
                "topic",
                "plan",
                "code",
                "output",
                "No further experiment is needed.",
                stage="submit",
            )
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(run.call_args.kwargs["tool_names_provider"](), {"submit_interpretation"})

            phd.interpretation_turn(
                "topic",
                "plan",
                "code",
                "output",
                "One diagnostic is required.",
                stage="supplement",
            )
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(
                run.call_args.kwargs["tool_names_provider"](),
                {"request_supplementary_experiment"},
            )
            self.assertEqual(
                registry.schemas_for({"request_supplementary_experiment"})[0]["function"],
                request_tool,
            )

            postdoc.interpretation_feedback("topic", "plan", "output", "proposal")
            postdoc_prompt = run.call_args.kwargs["system_prompt"]
            postdoc_user = run.call_args.kwargs["user_prompt"]
            self.assertIn("postdoctoral researcher", postdoc_prompt)
            self.assertLess(len(postdoc_prompt), 250)
            self.assertNotIn("send_dialogue", postdoc_prompt)
            self.assertIn("Interpretation history:", postdoc_user)
            registry = run.call_args.kwargs["registry"]
            parameters = registry.schemas_for({"send_dialogue"})[0]["function"]["parameters"]
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(set(parameters["required"]), {"message", "supplement_required"})
            self.assertEqual(parameters["properties"]["supplement_required"]["type"], ["boolean", "null"])
            self.assertIn("limitations", parameters["properties"]["supplement_required"]["description"])
            self.assertIn("observed values", parameters["properties"]["message"]["description"])
            feedback = registry.execute(
                "send_dialogue",
                {"message": "ready", "supplement_required": False},
            )
            self.assertFalse(feedback["supplement_required"])


    def test_interpretation_tool_provider_preserves_state_priority(self):
        all_tools = {
            "send_dialogue", "request_supplementary_experiment", "submit_interpretation",
        }
        cases = [
            ("", {}, {"send_dialogue"}),
            ("feedback", {}, {"send_dialogue"}),
            ("feedback", {"stage": "review"}, {"send_dialogue"}),
            ("feedback", {"stage": "supplement"}, {"request_supplementary_experiment"}),
            ("feedback", {"stage": "submit"}, {"submit_interpretation"}),
            ("feedback", {"stage": "decide"}, {"request_supplementary_experiment", "submit_interpretation"}),
        ]
        phd = PhDStudentAgent("model", "key")
        with patch.object(AgentRuntime, "run", return_value=SimpleNamespace(value={})) as run:
            for dialogue, status, expected in cases:
                with self.subTest(dialogue=dialogue, **status):
                    phd.interpretation_turn(
                        "topic", "plan", "code", "output", dialogue, **status,
                    )
                    kwargs = run.call_args.kwargs
                    registry = kwargs["registry"]
                    self.assertEqual(registry.names, all_tools)
                    self.assertEqual(kwargs["tool_names_provider"](), expected)
                    self.assertEqual(kwargs["terminal_tools"], all_tools)
                    self.assertFalse(kwargs.get("return_after_tool", False))

    def test_interpretation_runtime_returns_each_action_and_rejects_unavailable_tool(self):
        from agent_runtime import ToolRegistry
        from inference import ModelToolCall, ToolModelResponse

        request = {
            "objective": "check robustness", "reason": "resolve one evidence gap",
            "required_outputs": ["mean F1"], "constraints": ["preserve original experiment"],
        }
        cases = [
            ({"stage": "review"},
             "send_dialogue", {"message": "Review the latest results"}, True),
            ({"stage": "supplement"},
             "request_supplementary_experiment", request, True),
            ({"stage": "submit"},
             "submit_interpretation", {"interpretation": "Final evidence-based conclusion"}, False),
        ]
        original_execute = ToolRegistry.execute
        for status, tool_name, arguments, reject_first in cases:
            with self.subTest(tool_name=tool_name):
                executed_tools = []

                def execute(registry, name, tool_arguments):
                    executed_tools.append(name)
                    return original_execute(registry, name, tool_arguments)

                responses = []
                if reject_first:
                    responses.append(ToolModelResponse("", [ModelToolCall(
                        "unavailable", "submit_interpretation",
                        json.dumps({"interpretation": "Must not execute"}),
                    )]))
                responses.append(ToolModelResponse("", [ModelToolCall(
                    "allowed", tool_name, json.dumps(arguments),
                )]))
                with patch(
                    "agent_runtime.query_model_with_tools", side_effect=responses,
                ) as query, patch.object(ToolRegistry, "execute", new=execute):
                    result = PhDStudentAgent("model", "key").interpretation_turn(
                        "topic", "plan", "code", "output", "feedback", **status,
                    )
                self.assertEqual(result.tool_name, tool_name)
                self.assertTrue(result.value["success"])
                self.assertEqual(executed_tools, [tool_name])
                self.assertEqual(query.call_count, 2 if reject_first else 1)
                for model_call in query.call_args_list:
                    self.assertEqual(
                        {tool["function"]["name"] for tool in model_call.kwargs["tools"]},
                        {tool_name},
                    )
                if tool_name == "request_supplementary_experiment":
                    self.assertEqual(result.value["request"], request)
                else:
                    self.assertEqual(result.value, {"success": True, **arguments})

    def test_interpretation_invalid_inputs_retry_without_advancing(self):
        from inference import ModelToolCall, ToolModelResponse

        request = {"objective": "Check robustness", "reason": "Evidence gap",
                   "required_outputs": ["accuracy"], "constraints": []}
        cases = [
            ("review", "send_dialogue", {"message": " "}, {"message": "Review results"}),
            ("submit", "submit_interpretation", {"interpretation": " "}, {"interpretation": "Observed conclusion"}),
            ("supplement", "request_supplementary_experiment", {**request, "objective": " "}, request),
            ("supplement", "request_supplementary_experiment", {**request, "required_outputs": []}, request),
            ("postdoc", "send_dialogue", {"message": " ", "supplement_required": False},
             {"message": "Evidence sufficient", "supplement_required": False}),
        ]
        cases.append(("postdoc", "send_dialogue", {"message": "Feedback"},
                      {"message": "Evidence sufficient", "supplement_required": False}))
        for invalid in ("false", 0, 1, [], {}):
            cases.append(("postdoc", "send_dialogue", {"message": "Feedback", "supplement_required": invalid},
                          {"message": "Evidence sufficient", "supplement_required": False}))
        for stage, tool, invalid, valid in cases:
            with self.subTest(stage=stage, invalid=invalid), patch(
                "agent_runtime.query_model_with_tools", side_effect=[
                    ToolModelResponse("", [ModelToolCall(str(i), tool, json.dumps(arguments))])
                    for i, arguments in enumerate((invalid, valid))
                ],
            ) as query:
                if stage == "postdoc":
                    value = PostdocAgent("model", "key").interpretation_feedback("topic", "plan", "output", "proposal")
                else:
                    value = PhDStudentAgent("model", "key").interpretation_turn(
                        "topic", "plan", "code", "output", None, stage=stage,
                    ).value
                self.assertTrue(value["success"])
                self.assertEqual(query.call_count, 2)
                for call in query.call_args_list:
                    self.assertEqual({item["function"]["name"] for item in call.kwargs["tools"]}, {tool})
                tool_results = [json.loads(message["content"]) for message in query.call_args_list[-1].kwargs["messages"]
                                if message["role"] == "tool"]
                self.assertFalse(tool_results[0]["success"])
                self.assertEqual(value, {"success": True, "request": valid} if stage == "supplement" else {"success": True, **valid})



if __name__ == "__main__":
    unittest.main()
