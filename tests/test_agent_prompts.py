import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent_runtime import AgentRuntime
from agents import (
    MLEngineerAgent,
    PhDStudentAgent,
    PostdocAgent,
    SWEngineerAgent,
    _prepared_manifest_validation_issues,
)
from prompts import ml_dataset_selection_prompt, sw_data_preparation_prompt


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
    def test_ml_inspected_candidate_requires_first_trial_only(self):
        for inspected, pending in ((None, False), ({"dataset_id": "owner/data"}, True),
                                   ({"dataset_id": "owner/data"}, False)):
            with self.subTest(inspected=inspected, pending=pending), patch(
                "agents.HFDataSearch"
            ), patch.object(AgentRuntime, "run", return_value=SimpleNamespace(value={})) as run:
                MLEngineerAgent("model", "key").data_preparation_turn(
                    "topic", "plan", "review", "Please proceed",
                    inspected_dataset=inspected, trial_pending=pending,
                )
                args = run.call_args.kwargs
                expected = {"execute_dataset_code"} if pending else args["registry"].names
                self.assertEqual(args["tool_names_provider"](), expected)

    def test_workflow_unlocks_ml_after_failed_trial_and_rearms_after_inspection(self):
        import tempfile
        from ai_lab_repo import LaboratoryWorkflow
        from research_state import ResearchState
        inspected = {
            "success": True, "dataset_id": "owner/data", "revision": "r1",
            "config_name": "default", "split": "train",
            "dataset_schema": {"columns": {"value": "int64"}},
        }
        actions = [
            ("inspect_huggingface_dataset", inspected),
            ("execute_dataset_code", {"success": False, "code": "", "error": "Provide Python code"}),
            ("execute_dataset_code", {"success": False, "code": "raise ValueError()", "error": "bad"}),
            ("send_dialogue", {"success": True, "message": "Use a revised preparation method."}),
            ("inspect_huggingface_dataset", {**inspected, "revision": "r2"}),
            ("execute_dataset_code", {"success": True, "code": "data = [1]", "output": "ok"}),
        ]
        observed = []

        def ml_turn(*args, **kwargs):
            observed.append(kwargs["trial_pending"])
            name, value = actions[len(observed) - 1]
            return SimpleNamespace(tool_name=name, value=value)

        def sw_turn(*args, **kwargs):
            if len(observed) < len(actions):
                return SimpleNamespace(tool_name="send_dialogue", value={"success": True, "message": "Continue."})
            return SimpleNamespace(tool_name="submit_dataset_code", value={
                "success": True, "code": "data = [1]", "output": "ok",
                "selection": {key: value for key, value in inspected.items() if key != "success"} | {"revision": "r2"},
            })

        with tempfile.TemporaryDirectory() as directory:
            workflow = LaboratoryWorkflow(ResearchState("query", "recording", "topic"), "model", "key", directory)
            with patch.object(workflow.ml_engineer, "data_preparation_turn", side_effect=ml_turn), patch.object(
                workflow.sw_engineer, "data_preparation_turn", side_effect=sw_turn
            ):
                selection, result = workflow._prepare_dataset_collaboratively()
        self.assertEqual(observed, [False, True, True, False, False, True])
        self.assertTrue(result["success"])
        self.assertEqual(selection["revision"], "r2")

    def test_reconsider_prior_batch_and_exhausted_candidates(self):
        search = MagicMock()
        search.find_papers.side_effect = [
            {"text": "paper A", "candidate_ids": ["A"]},
            {"text": "paper B", "candidate_ids": ["B"]},
            {"text": "", "candidate_ids": []},
            {"text": "", "candidate_ids": []},
        ]
        search.browse_remaining_papers.return_value = {"text": "", "candidate_ids": []}
        search.retrieve_full_paper_text.return_value = "paper text"
        search.get_cached_full_paper_text.return_value = "paper text"
        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=SimpleNamespace(value={})
        ) as run:
            PhDStudentAgent("model", "key").conduct_literature_review("topic")
        registry = run.call_args.kwargs["registry"]
        provider = run.call_args.kwargs["tool_names_provider"]
        registry.execute("search_arxiv", {"query": "first", "max_results": 2})
        registry.execute("reject_arxiv_candidates", {"reason": "initial assessment"})
        registry.execute("search_arxiv", {"query": "second", "max_results": 2})
        self.assertEqual(search.find_papers.call_args.kwargs["excluded_ids"], {"A"})
        unknown = registry.execute("read_arxiv_paper", {"arxiv_id": "never-returned"})
        self.assertFalse(unknown["success"])
        search.retrieve_full_paper_text.assert_not_called()
        self.assertTrue(registry.execute("read_arxiv_paper", {"arxiv_id": "A"})["success"])
        self.assertEqual(provider(), {"add_paper", "reject_paper"})
        self.assertFalse(registry.execute("read_arxiv_paper", {"arxiv_id": "B"})["success"])
        registry.execute("reject_paper", {"arxiv_id": "A", "reason": "set aside again"})
        registry.execute("reject_arxiv_candidates", {"reason": "B unrelated"})
        for query in ("third", "fourth"):
            result = registry.execute("search_arxiv", {"query": query, "max_results": 2})
        self.assertEqual(result["reconsiderable_paper_ids"], ["A", "B"])
        self.assertEqual(provider(), {"read_arxiv_paper", "report_no_local_paper"})
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
        self.assertIn("complete candidate preparation program", ml_prompt)
        self.assertIn("not authoritative schema evidence", sw_prompt)
        self.assertIn("MUST NOT call submit_dataset_code", sw_prompt)
        self.assertIn("successful candidate execution", sw_prompt)

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
        from ai_lab_repo import LaboratoryWorkflow
        from research_state import ResearchState
        import json
        import tempfile

        def response(name, arguments):
            return ToolModelResponse("", [ModelToolCall(
                "call", name, json.dumps(arguments)
            )])

        bad = "raise ValueError('bad preparation')"
        fixed = "data = [1, 2]\nprint(len(data))"
        responses = [
            response("send_dialogue", {"message": "Find data for the plan."}),
            response("search_huggingface", {"query": "data", "max_results": 5}),
            response("send_dialogue", {"message": "Inspect the selected dataset."}),
            response("inspect_huggingface_dataset", {
                "dataset_id": "owner/data",
                "revision": "r1",
                "config_name": "default",
                "split": "train",
            }),
            response("send_dialogue", {"message": "Test a preparation program."}),
            response("execute_dataset_code", {"code": fixed}),
            response("submit_dataset_code", {"code": bad}),
            response("send_dialogue", {"message": "Replace the failing code with data = [1, 2]."}),
            response("submit_dataset_code", {"code": fixed}),
        ]
        calls = []
        search = MagicMock()
        search.retrieve_ds.return_value = [{"id": "owner/data"}]
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
            return responses[len(calls) - 1]

        with tempfile.TemporaryDirectory() as workdir, patch(
            "agents.HFDataSearch", return_value=search
        ), patch("agent_runtime.query_model_with_tools", side_effect=respond), patch(
            "agents.execute_code",
            side_effect=[details, {"success": False, "output": "bad preparation",
                                  "error_type": "python_exception", "error": "bad preparation",
                                  "traceback": "", "exit_code": 0}, details],
        ) as execute:
            workflow = LaboratoryWorkflow(
                ResearchState("query_001", "recording", "topic", plan="plan"),
                "model", "key", workdir,
            )
            selection, submitted = workflow._prepare_dataset_collaboratively()
        self.assertEqual(submitted["code"], fixed)
        self.assertTrue(submitted["success"])
        self.assertEqual(selection["dataset_id"], "owner/data")
        self.assertEqual(selection["revision"], "r1")
        self.assertEqual(execute.call_count, 3)
        for call in execute.call_args_list:
            self.assertEqual(str(call.kwargs["workdir"]), workdir)
            self.assertEqual(call.kwargs["timeout"], 600)
        expected_sw = {"send_dialogue", "submit_dataset_code"}
        expected_ml = {
            "search_huggingface",
            "inspect_huggingface_dataset",
            "execute_dataset_code",
            "send_dialogue",
        }
        for index, call in enumerate(calls):
            names = {item["function"]["name"] for item in call["tools"]}
            expected = expected_sw if index % 2 == 0 else expected_ml
            if index == 5:
                expected = {"execute_dataset_code"}
            self.assertEqual(names, expected)
        self.assertIn("owner/data", calls[2]["messages"][1]["content"])
        self.assertIn("row_count", calls[4]["messages"][1]["content"])
        self.assertIn(json.dumps(fixed)[1:-1], calls[6]["messages"][1]["content"])
        self.assertIn("bad preparation", calls[7]["messages"][1]["content"])
        self.assertIn("bad preparation", calls[8]["messages"][1]["content"])

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
            agent.conduct_literature_review("topic", required_papers=1)

        prompt = run.call_args.kwargs["system_prompt"]
        self.assertIn("exactly 1 relevant paper", prompt)
        self.assertIn("read before rejecting", prompt)
        self.assertIn("add_paper", prompt)
        self.assertIn("reject_paper", prompt)
        self.assertIn("submit without further searches", prompt)
        self.assertIn("meaningful methodological background", prompt)
        self.assertIn("need not use the same dataset", prompt)
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
        self.assertIn(
            "add_paper or reject_paper",
            schemas["read_arxiv_paper"]["description"],
        )
        self.assertIn(
            "meaningful methodological background",
            schemas["add_paper"]["description"],
        )
        self.assertIn(
            "may be reconsidered later",
            schemas["reject_paper"]["description"],
        )
        self.assertIn(
            "Read uncertain candidates first",
            schemas["reject_arxiv_candidates"]["description"],
        )
        registry.execute("search_arxiv", {"query": "short", "max_results": 10})
        search.find_papers.assert_called_once_with(
            "short",
            N=5,
            excluded_ids=set(),
        )
        self.assertEqual(
            tool_names_provider(),
            {"read_arxiv_paper", "reject_arxiv_candidates"},
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
            agent.conduct_literature_review("topic", required_papers=1)

        registry = run.call_args.kwargs["registry"]
        tool_names_provider = run.call_args.kwargs["tool_names_provider"]
        registry.execute(
            "search_arxiv",
            {"query": "short", "max_results": 5},
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

    def test_rejecting_arxiv_candidate_batch_reopens_search(self):
        search = MagicMock()
        search.find_papers.return_value = {
            "text": "papers",
            "candidate_ids": ["1234.5678", "2345.6789"],
        }
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"review": "review", "papers": []})
        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.conduct_literature_review("topic", required_papers=1)

        registry = run.call_args.kwargs["registry"]
        provider = run.call_args.kwargs["tool_names_provider"]
        registry.execute(
            "search_arxiv", {"query": "first", "max_results": 2}
        )
        rejected = registry.execute(
            "reject_arxiv_candidates", {"reason": "The abstracts are off-topic."}
        )

        self.assertTrue(rejected["success"])
        self.assertEqual(provider(), {"search_arxiv", "read_arxiv_paper"})
        registry.execute(
            "search_arxiv", {"query": "second", "max_results": 2}
        )
        self.assertEqual(
            search.find_papers.call_args.kwargs["excluded_ids"],
            {"1234.5678", "2345.6789"},
        )

    def test_two_empty_arxiv_searches_browse_remaining_candidates(self):
        search = MagicMock()
        search.find_papers.return_value = {"text": "", "candidate_ids": []}
        search.browse_remaining_papers.return_value = {
            "text": "fallback papers",
            "candidate_ids": ["3456.7890"],
        }
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"review": "review", "papers": []})
        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.conduct_literature_review("topic", required_papers=1)

        registry = run.call_args.kwargs["registry"]
        provider = run.call_args.kwargs["tool_names_provider"]
        first = registry.execute(
            "search_arxiv", {"query": "first", "max_results": 3}
        )
        self.assertFalse(first["fallback_used"])
        self.assertEqual(provider(), {"search_arxiv"})
        second = registry.execute(
            "search_arxiv", {"query": "second", "max_results": 3}
        )

        self.assertTrue(second["fallback_used"])
        self.assertEqual(second["candidate_ids"], ["3456.7890"])
        self.assertEqual(
            provider(),
            {"read_arxiv_paper", "reject_arxiv_candidates"},
        )
        search.browse_remaining_papers.assert_called_once_with(
            N=3,
            excluded_ids=set(),
        )

    def test_exhausted_arxiv_pool_exposes_terminal_report(self):
        search = MagicMock()
        search.find_papers.return_value = {"text": "", "candidate_ids": []}
        search.browse_remaining_papers.return_value = {
            "text": "",
            "candidate_ids": [],
        }
        agent = PhDStudentAgent("model", "key")
        completed = SimpleNamespace(value={"review": "review", "papers": []})
        with patch("agents.ArxivSearch", return_value=search), patch.object(
            AgentRuntime, "run", return_value=completed
        ) as run:
            agent.conduct_literature_review("topic", required_papers=1)

        registry = run.call_args.kwargs["registry"]
        provider = run.call_args.kwargs["tool_names_provider"]
        registry.execute("search_arxiv", {"query": "first", "max_results": 3})
        exhausted = registry.execute(
            "search_arxiv", {"query": "second", "max_results": 3}
        )

        self.assertEqual(exhausted["candidate_count"], 0)
        self.assertEqual(provider(), {"report_no_local_paper"})
        report = registry.execute(
            "report_no_local_paper",
            {"reason": "No remaining candidate supports the topic."},
        )
        self.assertTrue(report["success"])

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
        self.assertIn("baselines", prompt)
        self.assertIn("evaluation metrics", prompt)
        self.assertIn("reproducibility details", prompt)
        self.assertIn("Hugging Face dataset", prompt)
        self.assertIn("consultation_required", user_prompt)

        registry = run.call_args.kwargs["registry"]
        schemas = {item["function"]["name"]: item["function"] for item in registry.schemas}
        self.assertIn("must be used before submit_plan", schemas["send_dialogue"]["description"])
        self.assertIn("after receiving", schemas["submit_plan"]["description"])

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
        self.assertIn("evaluation metrics", prompt)
        self.assertIn("exact experimental procedure", prompt)
        self.assertIn("Do not select or name a concrete dataset ID", prompt)

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
            self.assertIn("MUST call send_dialogue", postdoc_prompt)
            self.assertIn("MUST NOT call submit_interpretation", postdoc_prompt)
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
                    "send_dialogue",
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

            phd.interpretation_feedback("topic", "plan", "output", "proposal")
            phd_prompt = run.call_args.kwargs["system_prompt"]
            phd_user = run.call_args.kwargs["user_prompt"]
            self.assertIn("being directed by a postdoctoral researcher", phd_prompt)
            self.assertIn("limitations", phd_prompt)
            self.assertIn("do not invent evidence", phd_prompt)
            self.assertIn("Interpretation history:", phd_user)


if __name__ == "__main__":
    unittest.main()
