import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from agent_runtime import AgentRuntime
from agents import (
    MLEngineerAgent,
    PhDStudentAgent,
    PostdocAgent,
    SWEngineerAgent,
    _prepared_manifest_validation_issues,
    _run_preparation_code,
)
from prompts import ml_dataset_selection_prompt, sw_data_preparation_prompt
from tools import ArxivSearch


COLLABORATION_ROUNDS = {
    "plan_max_rounds": 8,
    "data_preparation_max_rounds": 12,
    "interpretation_max_phd_reviews": 8,
}


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
    inspected = registry.execute("inspect_huggingface_dataset", arguments)
    if not inspected.get("success"):
        raise AssertionError(inspected)
    return registry.execute("select_inspected_huggingface_dataset", {})


def dataset_code_arguments(code):
    return {"lines": str(code).splitlines()}


class AgentPromptTest(unittest.TestCase):
    def test_sw_automatically_promotes_verified_trial_without_reexecution(self):
        selection = {
            "dataset_id": "owner/data",
            "revision": "r1",
            "config_name": "default",
            "split": "train",
        }
        verified_trial = {
            "success": True,
            "code": "data = [1, 2]",
            "output": "prepared",
            "selection": dict(selection),
            "artifacts": [],
            "prepared_data_manifest": None,
        }
        with patch("agent_runtime.query_model_with_tools") as query, patch(
            "agents.execute_code"
        ) as execute:
            result = SWEngineerAgent("model", "key").data_preparation_turn(
                "topic",
                "requirements",
                ".",
                inspected_dataset={"success": True, **selection},
                verified_trial=verified_trial,
            )

        self.assertEqual(result.tool_name, "submit_dataset_code")
        self.assertTrue(result.value.pop("automatic_execution"))
        self.assertEqual(result.value, verified_trial)
        query.assert_not_called()
        execute.assert_not_called()

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
            ("search_huggingface", {
                "success": True,
                "results": [],
                "selected_dataset": dict(selection),
            }),
            ("inspect_huggingface_dataset", {
                **inspected,
                "selected_dataset": dict(selection),
            }),
            ("execute_dataset_code", {
                "success": True,
                "code": "data = [1, 2]",
                "output": "prepared",
                "selection": dict(selection),
                "artifacts": [],
                "prepared_data_manifest": None,
            }),
        ]
        ml_calls = []
        sw_verified_trials = []

        def ml_turn(*args, **kwargs):
            ml_calls.append(kwargs)
            name, value = actions[len(ml_calls) - 1]
            return SimpleNamespace(tool_name=name, value=value)

        def sw_turn(*args, **kwargs):
            verified_trial = kwargs.get("verified_trial")
            sw_verified_trials.append(verified_trial)
            if verified_trial is not None:
                return SimpleNamespace(
                    tool_name="submit_dataset_code",
                    value=dict(verified_trial),
                )
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
        self.assertTrue(all(trial is None for trial in sw_verified_trials))

    def test_workflow_promotes_successful_multisplit_trial_with_provenance_warnings(self):
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
        }
        trial_selection = {
            "dataset_id": "owner/data",
            "revision": "r2",
            "config_name": "default",
            "split": None,
        }
        actions = [
            ("search_huggingface", {
                "success": True,
                "results": [],
                "selected_dataset": dict(inspected_selection),
            }),
            ("inspect_huggingface_dataset", {
                **inspected,
                "selected_dataset": dict(inspected_selection),
            }),
            ("execute_dataset_code", {
                "success": True,
                "code": "data = {'train': [1], 'test': [2]}",
                "output": "prepared both splits",
                "selection": dict(trial_selection),
                "artifacts": [],
                "prepared_data_manifest": None,
            }),
        ]
        ml_calls = []
        verified_trials = []

        def ml_turn(*args, **kwargs):
            ml_calls.append(kwargs)
            name, value = actions[len(ml_calls) - 1]
            return SimpleNamespace(tool_name=name, value=value)

        def sw_turn(*args, **kwargs):
            verified_trial = kwargs.get("verified_trial")
            verified_trials.append(verified_trial)
            if verified_trial is not None:
                return SimpleNamespace(
                    tool_name="submit_dataset_code",
                    value=dict(verified_trial),
                )
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
        self.assertIsNone(result_selection["split"])
        self.assertEqual(result_selection["revision"], "r2")
        warnings = submitted["validation_warnings"]
        self.assertTrue(any("explicit split" in warning for warning in warnings))
        self.assertTrue(any("revision" in warning for warning in warnings))
        self.assertTrue(all(trial is None for trial in verified_trials))

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
                    tool_name="inspect_huggingface_dataset",
                    value={**inspected, "selected_dataset": dict(selection)},
                )
            if index == 11:
                return SimpleNamespace(
                    tool_name="execute_dataset_code",
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
            "error_type": "validation_error",
            "error": "Generated code is invalid Python: unexpected indent (line 2)",
            "repair_hint": "Correct line 2 before retrying.",
            "code": "first = 1\n second = 2",
            "output": "",
            "traceback": "",
        }
        actions = [
            ("search_huggingface", {
                "success": True,
                "results": [],
                "selected_dataset": dict(selected),
            }),
            ("inspect_huggingface_dataset", {
                **inspected,
                "selected_dataset": dict(selected),
            }),
            ("execute_dataset_code", original_failure),
            ("execute_dataset_code", {
                "success": False,
                "error_type": "unchanged_failed_dataset_code",
                "error": "This exact data-preparation code already failed.",
                "original_error_type": original_failure["error_type"],
                "original_error": original_failure["error"],
                "original_repair_hint": original_failure["repair_hint"],
                "repair_hint": "Correct the original failure before retrying.",
                "code": original_failure["code"],
                "output": "",
                "traceback": "",
            }),
            ("execute_dataset_code", {
                "success": True,
                "code": "data = [1]",
                "output": "prepared",
                "selection": dict(selected),
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
            verified_trial = kwargs.get("verified_trial")
            if verified_trial is not None:
                return SimpleNamespace(
                    tool_name="submit_dataset_code",
                    value=dict(verified_trial),
                )
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

    def test_ml_selected_and_inspected_candidate_only_exposes_execution(self):
        selected = {
            "dataset_id": "owner/data",
            "revision": "r1",
            "config_name": "default",
            "split": "train",
        }
        cases = (
            (None, False, None, None),
            ({"dataset_id": "owner/data"}, True, selected, {"execute_dataset_code"}),
            ({"dataset_id": "owner/data"}, False, selected, {"execute_dataset_code"}),
            ({"dataset_id": "owner/data"}, False, None, None),
        )
        for inspected, pending, selected_dataset, expected in cases:
            with self.subTest(inspected=inspected, pending=pending, selected=selected_dataset), patch(
                "agents.HFDataSearch"
            ), patch.object(AgentRuntime, "run", return_value=SimpleNamespace(value={})) as run:
                MLEngineerAgent("model", "key").data_preparation_turn(
                    "topic", "plan", "review", "Please proceed",
                    inspected_dataset=inspected,
                    trial_pending=pending,
                    selected_dataset=selected_dataset,
                )
                args = run.call_args.kwargs
                self.assertEqual(
                    args["tool_names_provider"](),
                    expected if expected is not None else args["registry"].names,
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
            ("search_huggingface", {"success": True, "results": [], "selected_dataset": selected}),
            ("inspect_huggingface_dataset", {**inspected, "selected_dataset": selected}),
            ("execute_dataset_code", {"success": False, "code": "", "error": "Provide Python code"}),
            ("execute_dataset_code", {"success": False, "code": "raise ValueError()", "error": "bad"}),
            ("execute_dataset_code", {"success": True, "code": "data = [1]", "output": "ok"}),
        ]
        observed = []
        observed_selected = []

        def ml_turn(*args, **kwargs):
            observed.append(kwargs["trial_pending"])
            observed_selected.append(kwargs["selected_dataset"])
            name, value = actions[len(observed) - 1]
            return SimpleNamespace(tool_name=name, value=value)

        def sw_turn(*args, **kwargs):
            if len(observed) < len(actions):
                return SimpleNamespace(tool_name="send_dialogue", value={"success": True, "message": "Continue."})
            return SimpleNamespace(tool_name="submit_dataset_code", value={
                "success": True, "code": "data = [1]", "output": "ok",
                "selection": {key: value for key, value in inspected.items() if key != "success"},
            })

        with tempfile.TemporaryDirectory() as directory:
            workflow = LaboratoryWorkflow(
                ResearchState("query", "recording", "topic"), "model", "key", directory,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
            )
            with patch.object(workflow.ml_engineer, "data_preparation_turn", side_effect=ml_turn), patch.object(
                workflow.sw_engineer, "data_preparation_turn", side_effect=sw_turn
            ):
                selection, result = workflow._prepare_dataset_collaboratively()
        self.assertEqual(observed, [False, False, True, True, False])
        self.assertEqual(observed_selected, [None, selected, selected, selected, selected])
        self.assertTrue(result["success"])
        self.assertEqual(selection["revision"], "r1")

    def test_reconsider_prior_batch_after_empty_searches(self):
        search = MagicMock()
        search.find_papers.side_effect = [
            {"text": "paper A", "candidate_ids": ["A"]},
            {"text": "paper B", "candidate_ids": ["B"]},
            {"text": "", "candidate_ids": []},
            {"text": "", "candidate_ids": []},
        ]
        search.retrieve_full_paper_text.return_value = "paper text"
        search.get_cached_full_paper_text.return_value = "paper text"
        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=SimpleNamespace(value={})
        ) as run:
            PhDStudentAgent("model", "key").conduct_literature_review("topic")
        registry = run.call_args.kwargs["registry"]
        provider = run.call_args.kwargs["tool_names_provider"]
        registry.execute("search_arxiv", {"query": "first"})
        self.assertEqual(provider(), {"read_arxiv_paper"})
        unknown = registry.execute("read_arxiv_paper", {"arxiv_id": "never-returned"})
        self.assertFalse(unknown["success"])
        search.retrieve_full_paper_text.assert_not_called()
        self.assertTrue(registry.execute("read_arxiv_paper", {"arxiv_id": "A"})["success"])
        registry.execute("reject_paper", {"arxiv_id": "A", "reason": "initial assessment"})
        registry.execute("search_arxiv", {"query": "second"})
        self.assertEqual(search.find_papers.call_args.kwargs["excluded_ids"], {"A"})
        self.assertTrue(registry.execute("read_arxiv_paper", {"arxiv_id": "A"})["success"])
        self.assertEqual(provider(), {"add_paper", "reject_paper"})
        registry.execute("reject_paper", {"arxiv_id": "A", "reason": "set aside again"})
        self.assertTrue(registry.execute("read_arxiv_paper", {"arxiv_id": "B"})["success"])
        registry.execute("reject_paper", {"arxiv_id": "B", "reason": "B unrelated"})
        for query in ("third", "fourth"):
            result = registry.execute("search_arxiv", {"query": query})
        self.assertEqual(result["reconsiderable_paper_ids"], ["A", "B"])
        self.assertEqual(provider(), {"search_arxiv", "read_arxiv_paper"})
        self.assertTrue(registry.execute("read_arxiv_paper", {"arxiv_id": "A"})["success"])
        added = registry.execute("add_paper", {"arxiv_id": "A", "summary": "useful background"})
        self.assertTrue(added["review_ready"])
        self.assertEqual(provider(), {"submit_literature_review"})
        self.assertFalse(registry.execute("read_arxiv_paper", {"arxiv_id": "A"})["success"])
        review = registry.execute("submit_literature_review", {"review": "review"})
        self.assertEqual([paper["arxiv_id"] for paper in review["papers"]], ["A"])

    def test_inspection_feedback_is_deduplicated_only_when_identical(self):
        inspected = {
            "success": True, "dataset_id": "owner/data", "revision": "r1",
            "config_name": "default", "split": "train",
            "dataset_schema": {"row_count": 2, "columns": {"SCHEMA_SENTINEL": "int64"}},
            "materialized_resources": [],
        }
        matching = {"tool": "inspect_huggingface_dataset", **deepcopy(inspected)}
        cases = [
            ("identical", matching, inspected, True),
            ("failed", {**matching, "success": False, "error": "inspection failed"}, inspected, False),
            ("missing", matching, None, False),
            ("other_tool", {**matching, "tool": "execute_dataset_code"}, inspected, False),
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
                    with patch("agents.HFDataSearch"), patch("agents._execution_capabilities", return_value={}), patch.object(
                        AgentRuntime, "run", return_value=SimpleNamespace(value={})
                    ) as run:
                        if role == "ml":
                            MLEngineerAgent("model", "key").data_preparation_turn(
                                "topic", "plan", "review", "continue",
                                feedback=feedback, inspected_dataset=snapshot,
                            )
                            prefix = "Previous ML command result: "
                        else:
                            SWEngineerAgent("model", "key").data_preparation_turn(
                                "topic", "plan", ".", ml_result=feedback,
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

    def test_data_preparation_prompts_require_schema_evidence(self):
        ml_prompt = ml_dataset_selection_prompt()
        sw_prompt = sw_data_preparation_prompt()

        self.assertIn("not authoritative schema evidence", ml_prompt)
        self.assertIn("exact column names", ml_prompt)
        self.assertIn("stored representation", ml_prompt)
        self.assertIn("runtime evidence for Python access", ml_prompt)
        self.assertIn("bracket and attribute access", ml_prompt)
        self.assertIn("complete candidate preparation program", ml_prompt)
        self.assertIn("not authoritative schema evidence", sw_prompt)
        self.assertIn("stored representation", sw_prompt)
        self.assertIn("runtime evidence", sw_prompt)
        self.assertIn("bracket and attribute access", sw_prompt)
        self.assertIn("MUST NOT call submit_dataset_code", sw_prompt)
        self.assertIn("successful candidate execution", sw_prompt)
        self.assertIn("framework selects the first such candidate", ml_prompt)
        self.assertIn("do not request repeated searches", sw_prompt)
        self.assertIn("must not instantiate, train, tune, predict with", ml_prompt)
        self.assertIn("must not instantiate, train, tune, predict with", sw_prompt)
        self.assertIn("full experimental plan is intentionally unavailable", ml_prompt)
        self.assertIn("full experimental plan is intentionally unavailable", sw_prompt)
        self.assertIn("analysis role for subgroup or per-group analysis", ml_prompt)
        self.assertIn("analysis role for subgroup or per-group analysis", sw_prompt)
        self.assertIn("explicitly import every module", ml_prompt)
        self.assertIn("explicitly import every module", sw_prompt)

    def test_data_preparation_prompts_do_not_receive_full_plan(self):
        requirements = "DATA_REQUIREMENTS_SENTINEL"
        full_plan = "FULL_PLAN_SENTINEL RandomForestClassifier metrics baselines"
        completed = SimpleNamespace(value={})
        with patch("agents.HFDataSearch"), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            MLEngineerAgent("model", "key").data_preparation_turn(
                "topic", requirements, full_plan, "message"
            )
            ml_prompt = run.call_args.kwargs["user_prompt"]
            SWEngineerAgent("model", "key").data_preparation_turn(
                "topic", requirements, ".", literature_review=full_plan
            )
            sw_prompt = run.call_args.kwargs["user_prompt"]

        for prompt in (ml_prompt, sw_prompt):
            self.assertIn(requirements, prompt)
            self.assertNotIn(full_plan, prompt)

    def test_latest_execution_failure_replaces_stale_data_preparation_feedback(self):
        latest_failure = {
            "tool": "execute_dataset_code",
            "success": False,
            "error_type": "python_exception",
            "error": "CURRENT_ERROR",
            "traceback": "CURRENT_TRACEBACK",
            "code": "CURRENT_CODE",
            "output": "CURRENT_OUTPUT",
            "repair_hint": "CURRENT_REPAIR_HINT",
        }
        stale_feedback = {
            "tool": "execute_dataset_code",
            "success": False,
            "error": "OLD_ERROR",
            "traceback": "OLD_TRACEBACK",
        }
        completed = SimpleNamespace(value={})
        with patch("agents.HFDataSearch"), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            MLEngineerAgent("model", "key").data_preparation_turn(
                "topic", "plan", "review", "OLD_SW_REPAIR_ADVICE",
                feedback=stale_feedback,
                latest_execution_failure=latest_failure,
            )
            ml_prompt = run.call_args.kwargs["user_prompt"]
            SWEngineerAgent("model", "key").data_preparation_turn(
                "topic", "plan", ".",
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
        self.assertNotIn("OLD_SW_REPAIR_ADVICE", ml_prompt)
        self.assertNotIn("OLD_SW_RESULT", sw_prompt)

    def test_preparation_key_error_after_dataframe_transform_has_repair_hint(self):
        code = (
            "df = pd.get_dummies(df, columns=['group_label'])\n"
            "value = df['group_label']\n"
        )
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
        inspected_dataset = {
            "success": True,
            "dataset_schema": {"columns": {"group_label": "string"}},
        }
        with patch("agents.execute_code", return_value=execution):
            result = _run_preparation_code(
                code,
                ".",
                inspected_dataset,
            )

        self.assertFalse(result["success"])
        self.assertIn("group_label", result["repair_hint"])
        self.assertIn("inspected raw schema", result["repair_hint"])
        self.assertIn("separate feature dataframe", result["repair_hint"])

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
        with patch("agents.execute_code", return_value=execution):
            result = _run_preparation_code(code, ".")

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
                with patch("agents.execute_code", return_value=details):
                    result = _run_preparation_code("print('trial')", ".")
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
                "review",
                "inspect data",
            )

        registry = run.call_args.kwargs["registry"]
        inspected = registry.execute(
            "inspect_huggingface_dataset",
            {
                "dataset_id": "owner/data",
                "revision": "revision",
                "config_name": "default",
                "split": "train",
            },
        )
        self.assertTrue(inspected["success"])
        self.assertEqual(inspected["dataset_schema"]["row_count"], 2)
        search.inspect_dataset.assert_called_once_with(
            "owner/data",
            "revision",
            "default",
            "train",
        )

    def test_sw_submission_requires_inspected_schema(self):
        from inference import ModelToolCall, ToolModelResponse

        response = ToolModelResponse(
            "",
            [ModelToolCall(
                "call",
                "submit_dataset_code",
                json.dumps({"code": "print('data')"}),
            )],
        )
        with patch(
            "agent_runtime.query_model_with_tools",
            return_value=response,
        ), patch("agents.execute_code") as execute:
            result = SWEngineerAgent("model", "key").data_preparation_turn(
                "topic",
                "plan",
                ".",
            )

        self.assertFalse(result.value["success"])
        self.assertEqual(result.value["error_type"], "schema_not_inspected")
        execute.assert_not_called()

    def test_missing_prepared_manifest_is_a_warning(self):
        errors, warnings = _prepared_manifest_validation_issues(
            dataset_contract(required_prepared_splits=["train"]),
            None,
            [],
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("did not define prepared_data_manifest", warnings[0])

    def test_preparation_alternates_agents_and_only_sw_submission_finishes(self):
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
            response("search_huggingface", {"query": "data", "max_results": 5}),
            response("send_dialogue", {"message": "Inspect the selected dataset."}),
            response("inspect_huggingface_dataset", {}),
            response("send_dialogue", {"message": "Test a preparation program."}),
            response("execute_dataset_code", {"code": fixed}),
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
            "agents.execute_code",
            return_value=details,
        ) as execute:
            workflow = LaboratoryWorkflow(
                ResearchState("query_001", "recording", "topic", plan="plan"),
                "model", "key", workdir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
            )
            selection, submitted = workflow._prepare_dataset_collaboratively()
        self.assertEqual(submitted["code"], fixed)
        self.assertTrue(submitted["success"])
        self.assertEqual(selection["dataset_id"], "owner/data")
        self.assertEqual(selection["revision"], "r1")
        self.assertEqual(execute.call_count, 1)
        for call in execute.call_args_list:
            self.assertEqual(str(call.kwargs["workdir"]), workdir)
            self.assertEqual(call.kwargs["timeout"], 900)
        expected_sw = {"send_dialogue", "submit_dataset_code"}
        expected_ml = {
            "search_huggingface",
            "inspect_huggingface_dataset",
            "execute_dataset_code",
            "send_dialogue",
        }
        expected_tools = [
            expected_sw,
            expected_ml,
            expected_sw,
            {"inspect_huggingface_dataset"},
            expected_sw,
            {"execute_dataset_code"},
        ]
        self.assertEqual(len(calls), len(expected_tools))
        for index, call in enumerate(calls):
            names = {item["function"]["name"] for item in call["tools"]}
            self.assertEqual(names, expected_tools[index])
            if index == 3:
                inspect_schema = next(
                    item["function"]
                    for item in call["tools"]
                    if item["function"]["name"] == "inspect_huggingface_dataset"
                )
                self.assertEqual(inspect_schema["parameters"]["required"], [])
        self.assertIn("owner/data", calls[2]["messages"][1]["content"])
        self.assertIn("row_count", calls[4]["messages"][1]["content"])

    def test_ml_trial_failure_returns_to_workflow_without_internal_retry(self):
        from inference import ModelToolCall, ToolModelResponse
        import json
        code = "raise RuntimeError('failure')"
        with patch("agents.HFDataSearch"), patch(
            "agents.execute_code", return_value={"success": False, "output": "failure",
                "error_type": "python_exception", "error": "failure", "traceback": "", "exit_code": 0}
        ), patch("agent_runtime.query_model_with_tools", return_value=ToolModelResponse(
            "", [ModelToolCall("call", "execute_dataset_code", json.dumps({"code": code}))]
        )) as query:
            result = MLEngineerAgent("model", "key").data_preparation_turn(
                "topic", "plan", "review", "test code", workdir="."
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
                "execute_dataset_code",
                json.dumps({"code": first_code}),
            )],
        )
        repeated_response = ToolModelResponse(
            "",
            [ModelToolCall(
                "call",
                "execute_dataset_code",
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
            "agents.execute_code",
            return_value=execution,
        ) as execute_code, patch(
            "agent_runtime.query_model_with_tools",
            side_effect=[first_response, repeated_response],
        ):
            first = MLEngineerAgent("model", "key").data_preparation_turn(
                "topic",
                "requirements",
                "review",
                "try code",
                workdir=".",
                failed_trial_code_failures=failed_failures,
            )
            repeated = MLEngineerAgent("model", "key").data_preparation_turn(
                "topic",
                "requirements",
                "review",
                "try code",
                workdir=".",
                failed_trial_code_failures=failed_failures,
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

    def test_literature_prompt_restores_old_progression_guidance(self):
        search = MagicMock()
        search.find_papers.return_value = {
            "text": "papers",
            "candidate_ids": ["1234.5678"],
        }
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"review": "review", "papers": []})

        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.conduct_literature_review("topic")

        prompt = run.call_args.kwargs["system_prompt"]
        self.assertIn("exactly 1 relevant paper", prompt)
        self.assertIn("read before rejecting", prompt)
        self.assertIn("add_paper", prompt)
        self.assertIn("reject_paper", prompt)
        self.assertIn("submit without further searches", prompt)
        self.assertIn("meaningful methodological background", prompt)
        self.assertIn("dataset identity is not a requirement", prompt)
        self.assertIn("Do not reject it solely because these details differ", prompt)
        self.assertIn("prefer adding it as methodological background", prompt)
        self.assertIn("Reserve rejection for papers with no meaningful connection", prompt)
        self.assertIn("Reading a paper does not imply acceptance", prompt)
        self.assertIn("Never add a paper merely to satisfy the required paper count", prompt)
        self.assertIn("return exactly one tool call total", prompt)
        self.assertIn("currently read arXiv ID only", prompt)
        self.assertIn("including rejected ones", prompt)

        registry = run.call_args.kwargs["registry"]
        tool_names_provider = run.call_args.kwargs["tool_names_provider"]
        self.assertEqual(
            tool_names_provider(),
            {"search_arxiv"},
        )
        schemas = {
            item["function"]["name"]: item["function"]
            for item in registry.schemas
        }
        self.assertIn(
            "local arXiv",
            schemas["search_arxiv"]["description"],
        )
        self.assertEqual(
            set(schemas["search_arxiv"]["parameters"]["properties"]),
            {"query"},
        )
        self.assertEqual(
            schemas["search_arxiv"]["parameters"]["required"],
            ["query"],
        )
        self.assertIn(
            "add_paper or reject_paper",
            schemas["read_arxiv_paper"]["description"],
        )
        self.assertIn(
            "transferable methodological",
            schemas["add_paper"]["description"],
        )
        self.assertIn(
            "may be reconsidered later",
            schemas["reject_paper"]["description"],
        )
        self.assertNotIn("reject_arxiv_candidates", schemas)
        self.assertNotIn("reject_arxiv_candidates", registry.names)
        self.assertIn("dataset, domain, model, task", schemas["add_paper"]["description"])
        self.assertIn("mismatch in dataset, domain, model, task", schemas["reject_paper"]["description"])
        premature = registry.execute("submit_literature_review", {"review": "review"})
        self.assertFalse(premature["success"])
        registry.execute("search_arxiv", {"query": "short"})
        search.find_papers.assert_called_once_with(
            "short",
            excluded_ids=set(),
        )
        self.assertEqual(
            tool_names_provider(),
            {"read_arxiv_paper"},
        )
        search.retrieve_full_paper_text.return_value = "paper text"
        search.get_cached_full_paper_text.return_value = "paper text"
        read = registry.execute(
            "read_arxiv_paper",
            {"arxiv_id": "1234.5678"},
        )
        self.assertTrue(read["decision_required"])
        self.assertEqual(
            tool_names_provider(),
            {"add_paper", "reject_paper"},
        )
        added = registry.execute(
            "add_paper",
            {"arxiv_id": "1234.5678", "summary": "summary"},
        )
        self.assertEqual(added["paper_count"], 1)
        self.assertEqual(added["required_papers"], 1)
        self.assertTrue(added["review_ready"])
        self.assertEqual(
            added["next_action_hint"],
            "The required paper count is satisfied. "
            "Call submit_literature_review next.",
        )
        self.assertEqual(
            tool_names_provider(),
            {"submit_literature_review"},
        )

    def test_literature_runtime_submits_after_one_paper(self):
        from inference import ModelToolCall, ToolModelResponse

        search = MagicMock(spec=ArxivSearch)
        search.find_papers.return_value = {
            "text": "matching paper",
            "candidate_ids": ["1234.5678"],
        }
        search.retrieve_full_paper_text.return_value = "paper text"
        search.get_cached_full_paper_text.return_value = "paper text"
        actions = [
            ("search_arxiv", {"query": "topic"}),
            ("read_arxiv_paper", {"arxiv_id": "1234.5678"}),
            ("add_paper", {"arxiv_id": "1234.5678", "summary": "summary"}),
            ("submit_literature_review", {"review": "review"}),
        ]
        responses = [
            ToolModelResponse("", [ModelToolCall(
                f"call_{index}", name, json.dumps(arguments),
            )])
            for index, (name, arguments) in enumerate(actions)
        ]

        with patch("agents.ArxivSearch", return_value=search), patch(
            "agent_runtime.query_model_with_tools", side_effect=responses,
        ) as query_model:
            review = PhDStudentAgent("model", "key").conduct_literature_review("topic")

        self.assertTrue(review["success"])
        self.assertEqual(review["review"], "review")
        self.assertEqual(review["papers"], [{
            "arxiv_id": "1234.5678", "summary": "summary", "full_text": "paper text",
        }])
        search.find_papers.assert_called_once_with("topic", excluded_ids=set())
        search.retrieve_full_paper_text.assert_called_once_with("1234.5678", MAX_LEN=4000)
        self.assertEqual(query_model.call_count, 4)
        exposed_tools = [
            {tool["function"]["name"] for tool in invocation.kwargs["tools"]}
            for invocation in query_model.call_args_list
        ]
        self.assertEqual(exposed_tools, [
            {"search_arxiv"},
            {"read_arxiv_paper"},
            {"add_paper", "reject_paper"},
            {"submit_literature_review"},
        ])
        search_parameters = query_model.call_args_list[0].kwargs["tools"][0]["function"]["parameters"]
        self.assertEqual(set(search_parameters["properties"]), {"query"})
        self.assertEqual(search_parameters["required"], ["query"])

    def test_rejecting_read_paper_reopens_literature_search(self):
        search = MagicMock()
        search.find_papers.return_value = {
            "text": "papers",
            "candidate_ids": ["1234.5678"],
        }
        search.retrieve_full_paper_text.return_value = "paper text"
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"review": "review", "papers": []})

        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.conduct_literature_review("topic")

        registry = run.call_args.kwargs["registry"]
        tool_names_provider = run.call_args.kwargs["tool_names_provider"]
        registry.execute(
            "search_arxiv",
            {"query": "short"},
        )
        registry.execute("read_arxiv_paper", {"arxiv_id": "1234.5678"})
        rejected = registry.execute(
            "reject_paper",
            {
                "arxiv_id": "1234.5678",
                "reason": "The full text studies a different task.",
            },
        )

        self.assertTrue(rejected["success"])
        self.assertEqual(
            tool_names_provider(),
            {"search_arxiv", "read_arxiv_paper"},
        )
        repeated_read = registry.execute(
            "read_arxiv_paper",
            {"arxiv_id": "1234.5678"},
        )
        self.assertTrue(repeated_read["success"])
        self.assertEqual(tool_names_provider(), {"add_paper", "reject_paper"})

    def test_empty_arxiv_searches_allow_a_new_query_to_find_candidates(self):
        search = MagicMock(spec=ArxivSearch)
        search.find_papers.side_effect = [
            {"text": "", "candidate_ids": []},
            {"text": "", "candidate_ids": []},
            {"text": "matching paper", "candidate_ids": ["3456.7890"]},
        ]
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"review": "review", "papers": []})
        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.conduct_literature_review("topic")

        registry = run.call_args.kwargs["registry"]
        provider = run.call_args.kwargs["tool_names_provider"]
        for query in ("first", "second"):
            result = registry.execute(
                "search_arxiv", {"query": query}
            )
            self.assertEqual(result["candidate_ids"], [])
            self.assertEqual(provider(), {"search_arxiv"})
        third = registry.execute(
            "search_arxiv", {"query": "third"}
        )

        self.assertEqual(third["candidate_ids"], ["3456.7890"])
        self.assertEqual(third["results"], "matching paper")
        self.assertEqual(
            provider(),
            {"read_arxiv_paper"},
        )
        self.assertEqual(
            search.mock_calls,
            [
                call.find_papers(query, excluded_ids=set())
                for query in ("first", "second", "third")
            ],
        )

    def test_repeated_empty_arxiv_searches_do_not_create_candidates(self):
        search = MagicMock(spec=ArxivSearch)
        search.find_papers.return_value = {"text": "", "candidate_ids": []}
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"review": "review", "papers": []})
        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.conduct_literature_review("topic")

        registry = run.call_args.kwargs["registry"]
        provider = run.call_args.kwargs["tool_names_provider"]
        for query in ("first", "second", "third"):
            result = registry.execute(
                "search_arxiv", {"query": query}
            )
            self.assertEqual(result["candidate_ids"], [])
            self.assertEqual(result["candidate_count"], 0)
            self.assertFalse(result["decision_required"])
            self.assertEqual(provider(), {"search_arxiv"})
        self.assertEqual(
            search.mock_calls,
            [
                call.find_papers(query, excluded_ids=set())
                for query in ("first", "second", "third")
            ],
        )

    def test_postdoc_plan_prompt_requires_phd_consultation_and_full_plan(self):
        agent = PostdocAgent("model", "key")
        completed = SimpleNamespace(value={"message": "question"})
        with patch.object(
            AgentRuntime, "run_one", return_value=completed
        ) as run:
            agent.plan_turn("topic", "literature", "")

        prompt = run.call_args.kwargs["system_prompt"]
        user_prompt = run.call_args.kwargs["user_prompt"]
        self.assertIn("directing a PhD student", prompt)
        self.assertIn("MUST call send_dialogue", prompt)
        self.assertIn("MUST NOT call submit_plan", prompt)
        self.assertIn("simple, feasible experiment", prompt)
        self.assertIn("minimal valid experiment sufficient to answer the user's research question", prompt)
        self.assertNotIn("one innovative idea", prompt)
        self.assertIn("minimum data capabilities", prompt)
        self.assertIn("only when needed for a valid answer", prompt)
        self.assertIn("Do not turn examples into hard requirements", prompt)
        self.assertIn("operations explicitly requested by the topic", prompt)
        self.assertIn("a method mentioned only in the literature", prompt)
        self.assertIn("do not propose BERT", prompt)
        self.assertIn("simplest method trainable from scratch", prompt)
        self.assertIn("baselines", prompt)
        self.assertIn("evaluation metrics", prompt)
        self.assertIn("reproducibility details", prompt)
        self.assertIn("Hugging Face dataset", prompt)
        self.assertIn("subgroup, cohort, category, source, time-period", prompt)
        self.assertIn("do not open another consultation solely to confirm", prompt)
        self.assertIn("Do not ask the PhDStudentAgent to confirm field names", prompt)
        self.assertIn("consultation_required", user_prompt)

        registry = run.call_args.kwargs["registry"]
        schemas = {item["function"]["name"]: item["function"] for item in registry.schemas}
        self.assertIn("must be used before submit_plan", schemas["send_dialogue"]["description"])
        self.assertIn("after receiving", schemas["submit_plan"]["description"])
        self.assertEqual(
            schemas["submit_plan"]["parameters"]["required"],
            ["plan", "data_requirements"],
        )
        self.assertIn("data-preparation requirements", schemas["submit_plan"]["description"])
        self.assertFalse(
            registry.execute(
                "submit_plan",
                {"plan": "full plan", "data_requirements": " "},
            )["success"]
        )
        self.assertTrue(
            registry.execute(
                "submit_plan",
                {"plan": "full plan", "data_requirements": "abstract requirements"},
            )["success"]
        )

    def test_postdoc_plan_prompt_marks_received_feedback(self):
        agent = PostdocAgent("model", "key")
        completed = SimpleNamespace(value={"plan": "plan"})
        with patch.object(
            AgentRuntime, "run_one", return_value=completed
        ) as run:
            agent.plan_turn("topic", "literature", "PhD feedback")
        self.assertIn("feedback_received", run.call_args.kwargs["user_prompt"])

    def test_phd_plan_feedback_restores_research_planning_role(self):
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"message": "feedback"})
        with patch.object(
            AgentRuntime, "run_one", return_value=completed
        ) as run:
            agent.plan_feedback("topic", "literature", "proposal")
        prompt = run.call_args.kwargs["system_prompt"]
        self.assertIn("being directed by a postdoctoral researcher", prompt)
        self.assertIn("simple, feasible experiment", prompt)
        self.assertIn("minimal valid experiment sufficient to answer the user's research question", prompt)
        self.assertNotIn("novelty", prompt)
        self.assertIn("not an instruction to reproduce its algorithms", prompt)
        self.assertIn("approve it without inventing enhancements", prompt)
        self.assertIn("Remove out-of-scope requirements", prompt)
        self.assertIn("one deterministic held-out evaluation", prompt)
        self.assertIn("A method described in the literature is not", prompt)
        self.assertIn("reject a plan that proposes BERT", prompt)
        self.assertIn("local assets are explicitly supplied", prompt)
        self.assertIn("evaluation metrics", prompt)
        self.assertIn("exact experimental procedure", prompt)
        self.assertIn("Do not select or name a concrete dataset ID", prompt)
        self.assertIn("cannot block a valid abstract plan", prompt)

    def test_interpretation_prompts_require_postdoc_phd_dialogue(self):
        postdoc = PostdocAgent("model", "key")
        phd = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"message": "feedback"})
        with patch.object(
            AgentRuntime, "run_one", return_value=completed
        ) as run:
            postdoc.interpretation_turn("topic", "plan", "code", "output", "")
            postdoc_prompt = run.call_args.kwargs["system_prompt"]
            postdoc_user = run.call_args.kwargs["user_prompt"]
            self.assertIn("If PhD review is required, call send_dialogue", postdoc_prompt)
            self.assertIn("follow the tool availability", postdoc_prompt)
            self.assertIn("review_required", postdoc_user)
            self.assertIn("Interpretation history:", postdoc_user)
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(registry.names, {"send_dialogue"})

            postdoc.interpretation_turn(
                "topic",
                "plan",
                "code",
                "output",
                "PhD feedback",
                phd_review_required=False,
                workflow_notice="Use existing evidence.",
            )
            reviewed_user = run.call_args.kwargs["user_prompt"]
            self.assertIn("review_complete", reviewed_user)
            self.assertIn("Use existing evidence.", reviewed_user)
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(
                registry.names,
                {
                    "request_supplementary_experiment",
                    "submit_interpretation",
                },
            )
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

            postdoc.interpretation_turn(
                "topic",
                "plan",
                "code",
                "output",
                "No further experiment is needed.",
                phd_review_required=False,
                supplement_required=False,
            )
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(registry.names, {"submit_interpretation"})

            postdoc.interpretation_turn(
                "topic",
                "plan",
                "code",
                "output",
                "One diagnostic is required.",
                phd_review_required=False,
                supplement_required=True,
            )
            registry = run.call_args.kwargs["registry"]
            self.assertEqual(registry.names, {"request_supplementary_experiment"})

            phd.interpretation_feedback("topic", "plan", "output", "proposal")
            phd_prompt = run.call_args.kwargs["system_prompt"]
            phd_user = run.call_args.kwargs["user_prompt"]
            self.assertIn("being directed by a postdoctoral researcher", phd_prompt)
            self.assertIn("limitations", phd_prompt)
            self.assertIn("do not invent evidence", phd_prompt)
            self.assertIn("Interpretation history:", phd_user)
            registry = run.call_args.kwargs["registry"]
            feedback = registry.execute(
                "send_dialogue",
                {"message": "ready", "supplement_required": False},
            )
            self.assertFalse(feedback["supplement_required"])


if __name__ == "__main__":
    unittest.main()
