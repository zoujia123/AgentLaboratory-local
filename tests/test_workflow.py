import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent_runtime import AgentRunResult
from workflows.collaboration import (
    _compact_collaboration_result,
    _render_collaboration_history,
)
from workflows.laboratory import LaboratoryWorkflow
from research_state import ResearchState
from run_logging import PromptRecorder, RunLogger, run_log_scope

with patch("dotenv.load_dotenv"):
    from ai_lab_repo import load_paper_writing_config


COLLABORATION_ROUNDS = {
    "plan_max_rounds": 8,
    "data_preparation_max_rounds": 12,
    "interpretation_max_phd_reviews": 8,
}


def result(tool_name, value):
    return AgentRunResult(tool_name, {}, value, [])


class CollaborationHistoryTest(unittest.TestCase):
    def test_latest_execution_failure_hides_prior_failure_history(self):
        latest_failure = {
            "tool": "write_and_run_dataset_code",
            "success": False,
            "error": "CURRENT_ERROR",
        }
        conversation = [
            {"sender": "MLEngineerAgent", "tool": "write_and_run_dataset_code",
             "success": False, "error": "OLD_ERROR", "traceback": "OLD_TRACEBACK"},
            {"sender": "SWEngineerAgent", "tool": "send_dialogue",
             "success": True, "message": "CURRENT_SW_MESSAGE"},
            {"sender": "MLEngineerAgent", **latest_failure},
        ]
        original = deepcopy(conversation)
        history = _render_collaboration_history(
            conversation,
            omit_latest_senders={"SWEngineerAgent", "MLEngineerAgent"},
            latest_execution_failure=latest_failure,
        )
        self.assertEqual(history, "")
        self.assertEqual(conversation, original)


class WorkflowTest(unittest.TestCase):
    def test_compacts_code_and_diagnostics_for_collaboration_history(self):
        code = "prefix = 1\n" + ("value = 2\n" * 800) + "raise RuntimeError('tail')"
        output = "output-head\n" + ("x" * 5000) + "\noutput-tail"

        compact = _compact_collaboration_result(
            "MLEngineerAgent",
            "write_and_run_dataset_code",
            {
                "success": False,
                "error_type": "python_exception",
                "error": "failed",
                "code": code,
                "output": output,
            },
            round_index=2,
        )

        self.assertNotIn("code", compact)
        self.assertEqual(compact["code_chars"], len(code))
        self.assertEqual(compact["code_lines"], len(code.splitlines()))
        self.assertEqual(
            compact["code_sha256"],
            hashlib.sha256(code.encode("utf-8")).hexdigest(),
        )
        self.assertLess(len(compact["code_preview"]), len(code))
        self.assertIn("prefix = 1", compact["code_preview"])
        self.assertIn("raise RuntimeError", compact["code_preview"])
        self.assertLess(len(compact["output"]), len(output))
        self.assertIn("output-head", compact["output"])
        self.assertIn("output-tail", compact["output"])

    def test_collaboration_history_omits_latest_dedicated_results(self):
        conversation = [
            {"sender": "SWEngineerAgent", "message": "sw-old"},
            {"sender": "MLEngineerAgent", "message": "ml-old"},
            {"sender": "SWEngineerAgent", "message": "sw-current"},
            {"sender": "MLEngineerAgent", "message": "ml-current"},
        ]

        history = _render_collaboration_history(
            conversation,
            omit_latest_senders={"SWEngineerAgent", "MLEngineerAgent"},
        )

        self.assertIn("sw-old", history)
        self.assertIn("ml-old", history)
        self.assertNotIn("sw-current", history)
        self.assertNotIn("ml-current", history)

    def test_data_preparation_rejects_unknown_sw_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchState("query_001", "recording_001", "topic")
            state.plan = "plan"
            state.literature_review = "review"
            workflow = LaboratoryWorkflow(
                state, "model", "key", temp_dir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            workflow.sw_engineer.data_preparation_turn = MagicMock(return_value=result(
                "unknown_command",
                {"success": True, "message": "Please select a dataset."},
            ))
            workflow.ml_engineer.data_preparation_turn = MagicMock(return_value=result(
                "report_no_local_dataset",
                {
                    "success": True,
                    "reason": "All candidates were rejected.",
                    "rejected_dataset_ids": ["owner/data"],
                },
            ))
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "SWEngineerAgent must send dialogue during data preparation",
                ):
                    workflow._prepare_dataset_collaboratively()
            logger.close()
            workflow.ml_engineer.data_preparation_turn.assert_not_called()

    def test_sw_feedback_reaches_ml_and_dialogue_preserves_ml_execution_failure(self):
        inspected = {
            "success": True,
            "dataset_id": "owner/data",
            "revision": "revision",
            "config_name": "default",
            "split": "train",
            "dataset_schema": {"columns": {"value": "int64"}},
        }
        first_message = {"success": True, "message": "请检查选定的数据集。"}
        execution_message = {"success": True, "message": "请按照检查结果准备并执行代码。"}
        failure = {
            "success": False,
            "code": "value = data['missing']",
            "error_type": "python_exception",
            "error": "KeyError: 'missing'",
            "traceback": "execution traceback",
            "output": "loaded selected data",
        }
        repair_message = {
            "success": True,
            "message": "请使用已经检查过的 value 列修复代码。",
        }
        retry_message = {"success": True, "message": "请执行完整的修复代码。"}
        trial = {
            "success": True,
            "code": "value = data['value']",
            "output": "prepared",
            "executed_dataset": {
                "dataset_id": "owner/data",
                "revision": "revision",
                "resources": [{"config_name": "default", "split": "train"}],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            workflow = LaboratoryWorkflow(
                ResearchState("query_001", "recording_001", "topic"),
                "model", "key", temp_dir,
                collaboration_rounds={
                    **COLLABORATION_ROUNDS, "data_preparation_max_rounds": 4,
                },
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            workflow.sw_engineer.data_preparation_turn = MagicMock(side_effect=[
                result("send_dialogue", first_message),
                result("send_dialogue", execution_message),
                result("send_dialogue", repair_message),
                result("send_dialogue", retry_message),
            ])
            workflow.ml_engineer.data_preparation_turn = MagicMock(side_effect=[
                result("inspect_selected_dataset", inspected),
                result("write_and_run_dataset_code", failure),
                result("send_dialogue", {"success": True, "message": "I will repair it."}),
                result("write_and_run_dataset_code", trial),
            ])

            selected, submitted = workflow._prepare_dataset_collaboratively()

        self.assertEqual(selected, inspected)
        self.assertEqual(submitted["code"], trial["code"])
        calls = workflow.ml_engineer.data_preparation_turn.call_args_list
        self.assertEqual(len(calls), 4)
        for agent in (workflow.sw_engineer, workflow.ml_engineer):
            for preparation_call in agent.data_preparation_turn.call_args_list:
                self.assertEqual(
                    preparation_call.kwargs["code_execution_timeout_seconds"], 17.5,
                )
        for ml_call, feedback in zip(
            calls, (first_message, execution_message, repair_message, retry_message),
        ):
            self.assertEqual(json.loads(ml_call.args[3]), feedback)
        sw_calls = workflow.sw_engineer.data_preparation_turn.call_args_list
        self.assertEqual(sw_calls[2].kwargs["ml_result"], {
            "tool": "write_and_run_dataset_code", **failure,
        })
        for agent_calls in (sw_calls, calls):
            for preparation_call in agent_calls[:2]:
                self.assertIsNone(preparation_call.kwargs["latest_execution_failure"])
            for preparation_call in agent_calls[2:]:
                self.assertEqual(
                    preparation_call.kwargs["latest_execution_failure"],
                    {"tool": "write_and_run_dataset_code", **failure},
                )
                self.assertEqual(preparation_call.kwargs["inspected_dataset"], inspected)

    def test_data_preparation_passes_only_three_recent_dialogue_rounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchState("query_001", "recording_001", "topic")
            state.plan = "plan"
            state.literature_review = "review"
            workflow = LaboratoryWorkflow(
                state, "model", "key", temp_dir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            selection = {"dataset_id": "owner/data", "revision": "revision"}
            observed_sw_calls = []

            def sw_turn(*args, **kwargs):
                observed_sw_calls.append(kwargs)
                index = len(observed_sw_calls) - 1
                return result(
                    "send_dialogue",
                    {"success": True, "message": f"sw-message-{index}"},
                )

            def ml_turn(*args, **kwargs):
                index = len(observed_sw_calls) - 1
                return result(
                    "send_dialogue",
                    {
                        "success": True,
                        "message": f"ml-message-{index}",
                        "selection": selection,
                    },
                )

            workflow.sw_engineer.data_preparation_turn = MagicMock(
                side_effect=sw_turn
            )
            workflow.ml_engineer.data_preparation_turn = MagicMock(
                side_effect=ml_turn
            )
            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Data preparation exceeded its collaboration budget",
                ):
                    workflow._prepare_dataset_collaboratively()
            logger.close()

            self.assertEqual(len(observed_sw_calls), 12)
            latest_history = observed_sw_calls[-1]["conversation_history"]
            self.assertLessEqual(
                len(latest_history.splitlines()),
                6,
            )
            self.assertNotIn("sw-message-0", latest_history)
            self.assertIn("ml-message-9", latest_history)
            self.assertNotIn("ml-message-10", latest_history)
            self.assertEqual(
                observed_sw_calls[-1]["feedback"]["message"],
                "sw-message-10",
            )
            self.assertEqual(
                observed_sw_calls[-1]["ml_result"]["message"],
                "ml-message-10",
            )

    def test_research_finishes_with_paper_and_readme(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchState("query_001", "recording_001", "topic")
            workflow = LaboratoryWorkflow(
                state, "model", "key", temp_dir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            workflow.phd.conduct_literature_review = MagicMock(return_value={
                "review": "review",
                "papers": [{"arxiv_id": "1", "summary": "paper", "full_text": "text"}],
            })
            workflow.postdoc.plan_turn = MagicMock(
                return_value=result(
                    "submit_plan",
                    {
                        "success": True,
                        "plan": "plan",
                        "data_requirements": "requirements",
                    },
                )
            )
            selection = {
                "dataset_id": "owner/data",
                "config_name": "default",
                "split": "train",
                "revision": "revision",
                "dataset_contract": {
                    "artifact_layout": "split_aligned",
                    "role_sources": [{
                        "role": "model_input",
                        "source_type": "column",
                    }],
                },
            }
            workflow.sw_engineer.data_preparation_turn = MagicMock(side_effect=[
                result("send_dialogue", {
                    "success": True,
                    "message": "Please select a suitable dataset.",
                }),
                result("send_dialogue", {
                    "success": True,
                    "message": "Please prepare and execute the code using the inspected schema.",
                }),
            ])
            workflow.ml_engineer.data_preparation_turn = MagicMock(side_effect=[result(
                "inspect_selected_dataset",
                {
                    "success": True,
                    "dataset_id": "owner/data",
                    "config_name": "default",
                    "split": "train",
                    "revision": "revision",
                    "dataset_contract": selection["dataset_contract"],
                    "available_configs": ["default"],
                    "available_splits": ["train"],
                    "dataset_schema": {
                        "row_count": 10,
                        "columns": {"value": "double", "label": "int64"},
                    },
                    "materialized_resources": [],
                    "materialized_schemas": {},
                },
            ), result("write_and_run_dataset_code", {
                "success": True,
                "selection": selection,
                "code": "dataset_code",
                "output": "dataset_output",
                "prepared_data_manifest": {"feature_artifacts": {}},
                "validation_warnings": ["analysis metadata incomplete"],
            })])
            workflow.ml_solver.solve = MagicMock(return_value={
                "success": True,
                "code": "experiment_code",
                "output": "accuracy=0.9",
            })
            workflow.postdoc.interpretation_turn = MagicMock(side_effect=[
                result(
                    "send_dialogue",
                    {"success": True, "message": "review results"},
                ),
                result(
                    "submit_interpretation",
                    {"success": True, "interpretation": "interpretation"},
                ),
            ])
            workflow.phd.interpretation_feedback = MagicMock(
                return_value="PhD feedback"
            )
            workflow.paper_solver.write = MagicMock(
                return_value={"success": True, "latex": "paper"}
            )
            workflow.professor.generate_readme = MagicMock(return_value="readme")

            logger = RunLogger(temp_dir, "query_001", "job_001", console=False)
            prompts = PromptRecorder(temp_dir, "query_001", "job_001")
            with run_log_scope(logger, prompts):
                completed = workflow.perform_research()
            logger.close()

            self.assertEqual(completed.paper, "paper")
            self.assertEqual(completed.readme, "readme")
            self.assertEqual(completed.interpretation, "interpretation")
            workflow.phd.interpretation_feedback.assert_called_once()
            self.assertEqual(completed.data_requirements, "requirements")
            self.assertEqual(
                completed.dataset_selection["dataset_id"],
                selection["dataset_id"],
            )
            self.assertEqual(
                completed.dataset_selection["dataset_contract"],
                selection["dataset_contract"],
            )
            self.assertEqual(
                completed.dataset_selection["dataset_schema"],
                {
                    "row_count": 10,
                    "columns": {"value": "double", "label": "int64"},
                },
            )
            self.assertEqual(
                completed.dataset_manifest,
                {"feature_artifacts": {}},
            )
            self.assertEqual(
                completed.dataset_validation_warnings,
                ["analysis metadata incomplete"],
            )
            self.assertTrue(Path(temp_dir, "readme.md").is_file())
            self.assertTrue(Path(temp_dir, "state.json").is_file())
            self.assertEqual(
                Path(temp_dir, "report.tex").read_text(encoding="utf-8"),
                "paper",
            )
            self.assertEqual(
                {path.name for path in Path(temp_dir).glob("*.json")},
                {"state.json"},
            )
            self.assertEqual(
                json.loads(Path(temp_dir, "state.json").read_text(encoding="utf-8")),
                completed.to_dict(),
            )
            workflow.professor.generate_readme.assert_called_once()
            self.assertEqual(
                workflow.sw_engineer.data_preparation_turn.call_count,
                2,
            )
            self.assertEqual(workflow.ml_engineer.data_preparation_turn.call_count, 2)
            log_text = Path(temp_dir, "workflow.log").read_text(encoding="utf-8")
            expected_phases = [
                "literature review",
                "plan formulation",
                "data preparation collaboration",
                "running experiments",
                "results interpretation",
                "report writing",
                "readme writing",
            ]
            for event in ("BEGIN", "END"):
                marker = f"] {event} "
                self.assertEqual(
                    [line.split(marker, 1)[1].split(" elapsed=", 1)[0]
                     for line in log_text.splitlines() if marker in line],
                    expected_phases,
                )
            self.assertIn("HANDOFF SWEngineerAgent -> MLEngineerAgent", log_text)
            self.assertIn("HANDOFF MLEngineerAgent -> SWEngineerAgent", log_text)
            self.assertNotIn("DATASET CONTRACT REVIEW", log_text)

    def test_interpretation_appends_supplement_and_requires_new_phd_feedback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchState(
                "query_001",
                "recording_001",
                "topic",
                literature_review="review",
                plan="plan",
                dataset_code="dataset_code",
                experiment_code="original_code",
                experiment_output="original_output",
            )
            workflow = LaboratoryWorkflow(
                state, "model", "key", temp_dir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            request = {
                "objective": "check robustness",
                "reason": "one split is insufficient",
                "required_outputs": ["mean F1"],
                "constraints": ["preserve original experiment"],
            }
            workflow.postdoc.interpretation_turn = MagicMock(side_effect=[
                result("send_dialogue", {"success": True, "message": "review results"}),
                result(
                    "request_supplementary_experiment",
                    {"success": True, "request": request},
                ),
                result(
                    "send_dialogue",
                    {"success": True, "message": "review supplement"},
                ),
                result(
                    "submit_interpretation",
                    {"success": True, "interpretation": "final interpretation"},
                ),
            ])
            workflow.phd.interpretation_feedback = MagicMock(
                side_effect=["initial feedback", "supplement feedback"]
            )
            supplement = {
                "success": True,
                "supplement_id": "supplement_01",
                "request": request,
                "code": "supplement_code",
                "output": "supplement_output",
            }
            workflow.ml_solver.supplement = MagicMock(return_value=supplement)

            interpretation = workflow._interpret_results()

            self.assertEqual(interpretation, "final interpretation")
            self.assertEqual(state.experiment_code, "original_code")
            self.assertEqual(state.experiment_output, "original_output")
            self.assertEqual(state.supplementary_experiments, [supplement])
            workflow.ml_solver.supplement.assert_called_once_with(
                state,
                request,
                workflow.workdir,
                1,
            )
            self.assertEqual(workflow.phd.interpretation_feedback.call_count, 2)
            second_phd_call = workflow.phd.interpretation_feedback.call_args_list[1]
            self.assertEqual(
                second_phd_call.kwargs["supplementary_experiments"],
                [supplement],
            )
            third_postdoc_call = workflow.postdoc.interpretation_turn.call_args_list[2]
            self.assertEqual(
                third_postdoc_call.kwargs["supplementary_experiments"],
                [supplement],
            )
            self.assertEqual(third_postdoc_call.args[4], "")
            self.assertTrue(
                workflow.postdoc.interpretation_turn.call_args_list[0].kwargs[
                    "phd_review_required"
                ]
            )
            self.assertFalse(
                workflow.postdoc.interpretation_turn.call_args_list[1].kwargs[
                    "phd_review_required"
                ]
            )
            self.assertTrue(third_postdoc_call.kwargs["phd_review_required"])
            self.assertFalse(
                workflow.postdoc.interpretation_turn.call_args_list[3].kwargs[
                    "phd_review_required"
                ]
            )
            self.assertEqual(
                [event["type"] for event in state.interpretation_history],
                [
                    "interpretation_question",
                    "phd_feedback",
                    "supplement_request",
                    "supplement_result",
                    "interpretation_question",
                    "phd_feedback",
                    "final_interpretation",
                ],
            )
            self.assertEqual(
                third_postdoc_call.kwargs["interpretation_history"],
                state.interpretation_history,
            )
            self.assertEqual(
                second_phd_call.kwargs["interpretation_history"],
                state.interpretation_history,
            )
            self.assertTrue(Path(temp_dir, "src", "supplement_01.py").is_file())
            self.assertTrue(
                Path(temp_dir, "src", "supplement_01_output.log").is_file()
            )
            saved_state = Path(temp_dir, "state.json").read_text(encoding="utf-8")
            self.assertIn('"interpretation_history"', saved_state)
            self.assertIn('"supplement_result"', saved_state)

    def test_interpretation_rejects_duplicate_supplement_without_solver_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            request = {
                "objective": "check robustness",
                "reason": "one split is insufficient",
                "required_outputs": ["mean F1"],
                "constraints": ["preserve original experiment"],
            }
            state = ResearchState(
                "query_001",
                "recording_001",
                "topic",
                supplementary_experiments=[{
                    "success": True,
                    "supplement_id": "supplement_01",
                    "request": request,
                    "code": "supplement_code",
                    "output": "supplement_output",
                }],
            )
            workflow = LaboratoryWorkflow(
                state, "model", "key", temp_dir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            workflow.postdoc.interpretation_turn = MagicMock(side_effect=[
                result(
                    "send_dialogue",
                    {"success": True, "message": "review results"},
                ),
                result(
                    "request_supplementary_experiment",
                    {"success": True, "request": request},
                ),
                result(
                    "submit_interpretation",
                    {"success": True, "interpretation": "use existing evidence"},
                ),
            ])
            workflow.phd.interpretation_feedback = MagicMock(return_value="feedback")
            workflow.ml_solver.supplement = MagicMock()

            interpretation = workflow._interpret_results()

            self.assertEqual(interpretation, "use existing evidence")
            workflow.ml_solver.supplement.assert_not_called()
            self.assertEqual(len(state.supplementary_experiments), 1)
            self.assertEqual(
                [event["type"] for event in state.interpretation_history],
                [
                    "interpretation_question",
                    "phd_feedback",
                    "duplicate_supplement_rejected",
                    "final_interpretation",
                ],
            )
            final_turn = workflow.postdoc.interpretation_turn.call_args_list[2]
            self.assertFalse(final_turn.kwargs["phd_review_required"])
            self.assertTrue(final_turn.kwargs["force_submit"])
            self.assertIn(
                "Duplicate supplementary request rejected",
                final_turn.kwargs["workflow_notice"],
            )

    def test_interpretation_rejects_repeated_dialogue_after_phd_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchState("query_001", "recording_001", "topic")
            workflow = LaboratoryWorkflow(
                state, "model", "key", temp_dir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            workflow.postdoc.interpretation_turn = MagicMock(side_effect=[
                result("send_dialogue", {"success": True, "message": "review"}),
                result("send_dialogue", {"success": True, "message": "repeat"}),
            ])
            workflow.phd.interpretation_feedback = MagicMock(return_value={
                "message": "No supplement is needed.",
                "supplement_required": False,
            })

            with self.assertRaisesRegex(
                RuntimeError,
                "must submit the interpretation",
            ):
                workflow._interpret_results()

            self.assertEqual(workflow.postdoc.interpretation_turn.call_count, 2)
            self.assertEqual(workflow.phd.interpretation_feedback.call_count, 1)

    def test_interpretation_reserves_final_submit_after_review_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = ResearchState("query_001", "recording_001", "topic")
            workflow = LaboratoryWorkflow(
                state, "model", "key", temp_dir,
                collaboration_rounds=COLLABORATION_ROUNDS.copy(),
                code_execution_timeout_seconds=17.5,
                paper_writing=load_paper_writing_config(),
            )
            workflow.collaboration_rounds = {
                **workflow.collaboration_rounds,
                "interpretation_max_phd_reviews": 1,
            }
            workflow.postdoc.interpretation_turn = MagicMock(side_effect=[
                result("send_dialogue", {"success": True, "message": "review"}),
                result(
                    "submit_interpretation",
                    {"success": True, "interpretation": "cautious final result"},
                ),
            ])
            workflow.phd.interpretation_feedback = MagicMock(return_value={
                "message": "One more diagnostic would be useful.",
                "supplement_required": True,
            })

            interpretation = workflow._interpret_results()

            self.assertEqual(interpretation, "cautious final result")
            final_turn = workflow.postdoc.interpretation_turn.call_args_list[1]
            self.assertTrue(final_turn.kwargs["force_submit"])
            self.assertEqual(
                final_turn.kwargs["supplement_required"],
                True,
            )


if __name__ == "__main__":
    unittest.main()
