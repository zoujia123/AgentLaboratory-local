import tempfile
import json
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_runtime import AgentRuntime
from papersolver import PAPER_GENERATION_ORDER, PAPER_SECTIONS, PaperSolverAgent


class PaperSolverTest(unittest.TestCase):
    def test_writes_blueprint_then_bounded_sections_and_assembles_latex(self):
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
            experiment_score={"score": 0.9},
            interpretation="interpretation",
        )
        phases = []
        invalid_results = []
        original_state = deepcopy(vars(state))

        def fake_run_one(runtime, system_prompt, user_prompt, registry):
            del system_prompt
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
                            for section in PAPER_SECTIONS
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

        agent = PaperSolverAgent("model", "key")
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "Figure_1.png").write_bytes(b"figure")
            with patch.object(
                AgentRuntime,
                "run_one",
                autospec=True,
                side_effect=fake_run_one,
            ), patch(
                "papersolver.compile_latex",
                return_value="Compilation disabled; LaTeX source saved",
            ) as compile_latex:
                result = agent.write(state, temp_dir, compile_pdf=False)

            self.assertEqual(vars(state), original_state)

            self.assertEqual(
                [phase for phase, _ in phases],
                ["report writing blueprint"]
                + [f"report writing section {section}" for section in PAPER_GENERATION_ORDER],
            )
            self.assertEqual(phases[0][1], 3000)
            self.assertTrue(all(tokens <= 4096 for _, tokens in phases[1:]))
            self.assertTrue(all(not item["success"] for item in invalid_results))
            self.assertTrue(result["success"])
            self.assertEqual(set(result["sections"]), set(PAPER_SECTIONS))
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
            for index, section in enumerate(PAPER_SECTIONS, start=1):
                filename = f"{index:02d}_{section.lower().replace(' ', '_')}.tex"
                self.assertTrue(Path(temp_dir, "tex", "sections", filename).is_file())


if __name__ == "__main__":
    unittest.main()
