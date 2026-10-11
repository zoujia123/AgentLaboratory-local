import tempfile
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from agent_runtime import AgentRuntime
from agents import PaperSolverAgent
from inference import ModelToolCall, ToolModelResponse

with patch("dotenv.load_dotenv"):
    from ai_lab_repo import load_paper_writing_config, parse_yaml


class PaperSolverTest(unittest.TestCase):
    def test_writes_blueprint_then_bounded_sections_and_assembles_latex(self):
        agent = PaperSolverAgent("model", "key", paper_writing=load_paper_writing_config())
        sections = [section["name"] for section in agent.paper_config["sections"]]
        generation_order = agent.paper_config["generation_order"]
        state = SimpleNamespace(
            topic="topic",
            literature_review="review",
            literature_papers=[{"arxiv_id": "1234.5678v1"}],
            plan="plan",
            dataset_selection={"dataset_id": "owner/data"},
            seeds={"python": 2001},
            experiment_code="print('experiment')",
            experiment_output="accuracy=0.9",
            supplementary_experiments=[
                {
                    "supplement_id": "supplement_01", "success": True,
                    "request": {"objective": "Compare models", "constraints": ["Same split"]},
                    "code": "print('SUPPLEMENT_ONE_SOURCE')",
                    "output": "model accuracy=0.95", "request_fingerprint": "fingerprint-1",
                },
                {
                    "supplement_id": "supplement_02", "success": True,
                    "request": {"objective": "Check stability"},
                    "code": "print('SUPPLEMENT_TWO_SOURCE')",
                    "output": "CI=[0.89, 0.97]",
                },
            ],
            interpretation="interpretation",
        )
        phases = []
        invalid_results = []
        original_state = deepcopy(vars(state))

        def fake_run_one(runtime, system_prompt, user_prompt, registry):
            self.assertLess(len(system_prompt), 200)
            self.assertNotIn("Figure_1.png", system_prompt)
            self.assertIn("Figure_1.png", user_prompt)
            self.assertNotIn("Approved paper blueprint", user_prompt)
            if runtime.phase != "report writing blueprint":
                for requirement in (
                    "fixed seeds", "potential negative impacts", "focused paragraph",
                    "Preserve reported numbers exactly", "supplied citation keys",
                ):
                    self.assertIn(requirement, user_prompt)
            evidence = json.loads(user_prompt.split("Authoritative evidence:\n", 1)[1])
            include_code = runtime.phase in {
                "report writing section Methods", "report writing section Experimental Setup",
            }
            expected = [
                {key: value for key, value in item.items() if include_code or key != "code"}
                for item in original_state["supplementary_experiments"]
            ]
            self.assertEqual(evidence["supplementary_experiments"], expected)
            self.assertEqual(evidence["experiment_output"], state.experiment_output)
            main_code_expected = include_code or runtime.phase == "report writing blueprint"
            self.assertEqual("experiment_code" in evidence, main_code_expected)
            for marker in ("SUPPLEMENT_ONE_SOURCE", "SUPPLEMENT_TWO_SOURCE"):
                self.assertEqual(marker in user_prompt, include_code)
            phases.append((runtime.phase, runtime.max_tokens))
            if runtime.phase == "report writing blueprint":
                value = registry.execute(
                    "submit_paper_blueprint",
                    {
                        "title": "A & B",
                        "central_claim": "The observed result supports the comparison.",
                        "section_plans": {
                            section: f"Plan for {section}"
                            for section in sections
                        },
                    },
                )
                return SimpleNamespace(value=value)

            schema = registry.schemas[0]["function"]["parameters"]
            section = schema["properties"]["section_name"]["enum"][0]
            if section == "Introduction":
                invalid_results.append(registry.execute(
                    "submit_paper_section",
                    {
                        "section_name": section,
                        "latex_body": r"\section{Introduction} invalid wrapper",
                    },
                ))
            if section == "Related Work":
                invalid_results.append(registry.execute(
                    "submit_paper_section",
                    {
                        "section_name": section,
                        "latex_body": r"Unknown citation \cite{invented}.",
                    },
                ))
                body = r"The reviewed work provides context \cite{paper_1}."
            else:
                body = f"Evidence-grounded body for {section}."
            value = registry.execute(
                "submit_paper_section",
                {"section_name": section, "latex_body": body},
            )
            return SimpleNamespace(value=value)

        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "Figure_1.png").write_bytes(b"figure")
            with patch.object(
                AgentRuntime,
                "run_one",
                autospec=True,
                side_effect=fake_run_one,
            ), patch(
                "agents.compile_latex",
                return_value="Compilation disabled; LaTeX source saved",
            ) as compile_latex:
                result = agent.write(state, temp_dir, compile_pdf=False)

            self.assertEqual(vars(state), original_state)

            self.assertEqual(
                [phase for phase, _ in phases],
                ["report writing blueprint"]
                + [f"report writing section {section}" for section in generation_order],
            )
            self.assertEqual(phases[0][1], 3000)
            self.assertTrue(all(tokens <= 4096 for _, tokens in phases[1:]))
            self.assertTrue(all(not item["success"] for item in invalid_results))
            self.assertTrue(result["success"])
            self.assertEqual(set(result["sections"]), set(sections))
            latex = result["latex"]
            self.assertIn(r"\title{A \& B}", latex)
            self.assertEqual(latex.count(r"\section{Abstract}"), 1)
            self.assertLess(
                latex.index(r"\section{Abstract}"),
                latex.index(r"\section{Introduction}"),
            )
            self.assertEqual(latex.count(r"\includegraphics"), 1)
            self.assertIn(r"\bibitem{paper_1}", latex)
            compile_latex.assert_called_once()
            self.assertTrue(Path(temp_dir, "tex", "paper_blueprint.json").is_file())
            for index, section in enumerate(sections, start=1):
                filename = f"{index:02d}_{section.lower().replace(' ', '_')}.tex"
                self.assertTrue(Path(temp_dir, "tex", "sections", filename).is_file())

    def test_runtime_keeps_failed_submissions_in_the_current_step(self):
        config = {
            "sections": [
                {"name": "Methods", "word_target": 100, "max_chars": 1000},
                {"name": "Results", "word_target": 100, "max_chars": 1000},
            ],
            "generation_order": ["Methods", "Results"],
        }
        state = SimpleNamespace(
            topic="topic", literature_review="review", plan="plan",
            dataset_selection={}, experiment_code="print(1)",
            experiment_output="1", interpretation="interpretation",
        )
        blueprint = {
            "title": "Paper", "central_claim": "Observed evidence",
            "section_plans": {"Methods": "Method plan", "Results": "Result plan"},
        }
        attempts = [
            # An unavailable tool cannot bypass the blueprint.
            ("submit_paper_section", {"section_name": "Methods", "latex_body": "Early"}),
            ("submit_paper_blueprint", {**blueprint, "section_plans": {"Methods": "Plan"}}),
            ("submit_paper_blueprint", {**blueprint, "section_plans": {"Methods": 7, "Results": "Plan"}}),
            *[
                ("submit_paper_blueprint", {**blueprint, field: empty})
                for field in ("title", "central_claim")
                for empty in ("", " \t\n")
            ],
            *[
                ("submit_paper_blueprint", {
                    **blueprint,
                    "section_plans": {**blueprint["section_plans"], section: empty},
                })
                for section in ("Methods", "Results")
                for empty in ("", " \t\n")
            ],
            ("submit_paper_blueprint", {
                "title": "  Paper  ", "central_claim": "  Observed evidence  ",
                "section_plans": {"Methods": "  Method plan  ", "Results": "  Result plan  "},
            }),
            ("submit_paper_section", {"section_name": "Results", "latex_body": "Wrong section"}),
            ("submit_paper_section", {"section_name": "Methods", "latex_body": 7}),
            ("submit_paper_section", {"section_name": "Methods", "latex_body": "  "}),
            ("submit_paper_section", {"section_name": "Methods", "latex_body": "Method body"}),
            ("submit_paper_section", {"section_name": "Results", "latex_body": "Result body"}),
        ]
        observed = []

        def respond(**kwargs):
            tool = kwargs["tools"][0]["function"]
            self.assertEqual(len(kwargs["tools"]), 1)
            current_section = tool["parameters"]["properties"].get("section_name", {}).get("enum")
            observed.append((tool["name"], current_section))
            name, arguments = attempts[len(observed) - 1]
            return ToolModelResponse("", [ModelToolCall(
                str(len(observed)), name, json.dumps(arguments),
            )])

        with tempfile.TemporaryDirectory() as directory:
            with patch("agent_runtime.query_model_with_tools", side_effect=respond), patch(
                "agents.compile_latex", return_value="Compilation disabled",
            ):
                result = PaperSolverAgent("model", "key", paper_writing=config).write(state, directory)
            self.assertEqual(result["blueprint"], {"success": True, **blueprint})
            saved_blueprint = json.loads(Path(directory, "tex", "paper_blueprint.json").read_text())
            self.assertEqual(saved_blueprint, result["blueprint"])
            self.assertEqual(result["sections"], {"Methods": "Method body", "Results": "Result body"})
            self.assertEqual(Path(directory, "tex", "sections", "01_methods.tex").read_text(), "Method body\n")
        self.assertEqual(observed, [
            *[("submit_paper_blueprint", None)] * 12,
            *[("submit_paper_section", ["Methods"])] * 4,
            ("submit_paper_section", ["Results"]),
        ])

    def test_custom_config_controls_generation_prompts_schemas_and_document_order(self):
        config = {
            "sections": [
                {"name": "Summary", "word_target": 125, "max_chars": 1200},
                {"name": "Methods", "word_target": 320, "max_chars": 3300},
                {"name": "Results", "word_target": 210, "max_chars": 2200},
            ],
            "generation_order": ["Results", "Summary", "Methods"],
        }
        sections = tuple(item["name"] for item in config["sections"])
        by_name = {item["name"]: item for item in config["sections"]}
        state = SimpleNamespace(
            topic="topic", literature_review="review", plan="plan",
            dataset_selection={}, experiment_code="print('experiment')",
            experiment_output="observed result", interpretation="interpretation",
        )
        generated = []

        def fake_run_one(runtime, system_prompt, user_prompt, registry):
            schema = registry.schemas[0]["function"]["parameters"]
            if runtime.phase == "report writing blueprint":
                self.assertIn(", ".join(sections), user_prompt)
                self.assertNotIn(", ".join(sections), system_prompt)
                self.assertEqual(
                    schema["properties"]["section_plans"]["required"], list(sections),
                )
                value = registry.execute("submit_paper_blueprint", {
                    "title": "Configured paper", "central_claim": "Observed evidence",
                    "section_plans": {section: f"Plan for {section}" for section in sections},
                })
                return SimpleNamespace(value=value)

            section = schema["properties"]["section_name"]["enum"][0]
            generated.append(section)
            self.assertEqual(runtime.phase, f"report writing section {section}")
            self.assertIn(f"approximately {by_name[section]['word_target']} words", user_prompt)
            self.assertNotIn("approximately", system_prompt)
            self.assertIn(f"Current section: {section}", user_prompt)
            self.assertEqual(
                schema["properties"]["latex_body"]["maxLength"],
                by_name[section]["max_chars"],
            )
            blueprint_text = user_prompt.split("Paper blueprint:\n", 1)[1].split(
                "\nAuthoritative evidence:\n", 1,
            )[0]
            self.assertEqual(json.loads(blueprint_text)["section_plans"][section], f"Plan for {section}")
            self.assertNotIn("Current section plan:", user_prompt)
            return SimpleNamespace(value=registry.execute("submit_paper_section", {
                "section_name": section, "latex_body": f"Body for {section}.",
            }))

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, "batch.yaml")
            config_path.write_text(yaml.safe_dump({
                "material-root": directory,
                "output-root": directory,
                "queries": [{"query-id": "query_001"}],
                "llm_sampling": {"temperature": 0.1, "top_p": 1, "max_tokens": 8192},
                "collaboration_rounds": {
                    "plan_max_rounds": 2, "data_preparation_max_rounds": 3,
                },
                "code_execution_timeout_seconds": 17.5,
                "paper_writing": config,
            }), encoding="utf-8")
            batch = parse_yaml(config_path)
            self.assertEqual(batch.paper_writing, config)
            agent = PaperSolverAgent("model", "key", paper_writing=batch.paper_writing)
            self.assertEqual(agent.paper_config, config)
            self.assertIsNot(agent.paper_config, batch.paper_writing)
            self.assertIsNot(agent.paper_config["sections"], batch.paper_writing["sections"])
            self.assertIsNot(agent.paper_config["sections"][0], batch.paper_writing["sections"][0])
            self.assertIsNot(agent.paper_config["generation_order"], batch.paper_writing["generation_order"])
            with patch.object(
                AgentRuntime, "run_one", autospec=True, side_effect=fake_run_one,
            ), patch("agents.compile_latex", return_value="Compilation disabled") as compile_latex:
                result = agent.write(state, directory)

            self.assertTrue(result["success"])
            self.assertEqual(generated, config["generation_order"])
            positions = [result["latex"].index(r"\section{" + section + "}") for section in sections]
            self.assertEqual(positions, sorted(positions))
            expected_files = ["01_summary.tex", "02_methods.tex", "03_results.tex"]
            sections_dir = Path(directory, "tex", "sections")
            self.assertEqual(sorted(path.name for path in sections_dir.iterdir()), expected_files)
            for section, filename in zip(sections, expected_files):
                self.assertEqual(
                    (sections_dir / filename).read_text(encoding="utf-8"),
                    f"Body for {section}.\n",
                )
            compile_latex.assert_called_once_with(result["latex"], directory, compile=False)

    def test_entrypoint_rejects_invalid_config_before_requesting_a_model(self):
        valid = {
            "sections": [
                {"name": "Summary", "word_target": 125, "max_chars": 1200},
                {"name": "Results", "word_target": 210, "max_chars": 2200},
            ],
            "generation_order": ["Results", "Summary"],
        }
        duplicate_names = deepcopy(valid)
        duplicate_names["sections"][1]["name"] = "Summary"
        cases = [
            ("empty sections", {"sections": [], "generation_order": []}),
            ("duplicate section names", duplicate_names),
            ("missing generation section", {**valid, "generation_order": ["Summary"]}),
            ("duplicate generation section", {**valid, "generation_order": ["Summary", "Summary"]}),
            ("unknown generation section", {**valid, "generation_order": ["Summary", "Other"]}),
        ]
        for field in ("word_target", "max_chars"):
            for value in (0, -1, 2.5, "125", True):
                invalid = deepcopy(valid)
                invalid["sections"][0][field] = value
                cases.append((f"{field}={value!r}", invalid))

        for label, config in cases:
            with self.subTest(config=label):
                with patch("agent_runtime.query_model_with_tools") as query:
                    with self.assertRaises(ValueError):
                        load_paper_writing_config(config)
                    query.assert_not_called()

    def test_entrypoint_loads_default_config_from_another_working_directory(self):
        expected = load_paper_writing_config()
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                loaded = load_paper_writing_config()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(loaded, expected)

    def test_agent_constructs_from_provided_config_without_reading_files(self):
        config = load_paper_writing_config()
        with patch.object(Path, "open", side_effect=AssertionError("Agent must not read files")) as path_open, \
                patch("builtins.open", side_effect=AssertionError("Agent must not read files")) as builtin_open:
            agent = PaperSolverAgent("model", "key", paper_writing=config)

        path_open.assert_not_called()
        builtin_open.assert_not_called()
        self.assertEqual(agent.paper_config, config)
        self.assertIsNot(agent.paper_config, config)
        self.assertIsNot(agent.paper_config["sections"][0], config["sections"][0])
        self.assertIsNot(agent.paper_config["generation_order"], config["generation_order"])

    def test_agent_requires_explicit_config_without_reading_a_default(self):
        with patch.object(Path, "open", side_effect=AssertionError("Agent must not read files")) as path_open, \
                patch("builtins.open", side_effect=AssertionError("Agent must not read files")) as builtin_open:
            with self.assertRaisesRegex(TypeError, "paper_writing"):
                PaperSolverAgent("model", "key")

        path_open.assert_not_called()
        builtin_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
