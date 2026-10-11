import unittest
from prompts import (
    experiment_solver_prompt,
    ml_dataset_selection_prompt,
    paper_blueprint_prompt,
    paper_section_prompt,
    professor_readme_prompt,
    sw_data_preparation_prompt,
)


class PromptCatalogTest(unittest.TestCase):
    def test_preparation_prompts_describe_roles_and_tasks(self):
        ml = ml_dataset_selection_prompt()
        sw = sw_data_preparation_prompt()
        self.assertIn("Prepare reusable data", ml)
        self.assertIn("provided local datasets", ml)
        self.assertIn("Review data preparation and suggest necessary fixes", sw)
        self.assertIn("without adding model training or evaluation", sw)
        for prompt in (ml, sw):
            self.assertNotIn("Runtime object discipline", prompt)
            self.assertNotIn("PYTHON_SEED", prompt)

    def test_main_experiment_prompt_only_describes_role_and_task(self):
        prompt = experiment_solver_prompt()
        self.assertIn("machine learning engineer", prompt)
        self.assertIn("current experiment task", prompt)
        self.assertIn("prepared data", prompt)
        self.assertIn("reproducible, observed results", prompt)
        self.assertLess(len(prompt), 250)
        self.assertNotIn("write_experiment_code", prompt)
        self.assertNotIn("Runtime object discipline", prompt)
        self.assertNotIn("SKLEARN_SEED", prompt)

    def test_paper_system_prompts_only_describe_role_and_task(self):
        for prompt in (paper_blueprint_prompt(), paper_section_prompt()):
            self.assertIn("research paper writer", prompt)
            self.assertIn("English", prompt)
            self.assertIn("evidence", prompt)
            self.assertLess(len(prompt), 200)
            self.assertNotIn("submit_paper", prompt)
            self.assertNotIn("Figure_", prompt)
            self.assertNotIn("approximately", prompt)
            self.assertNotIn("approved", prompt.lower())

    def test_professor_readme_prompt_only_describes_role_and_task(self):
        professor = professor_readme_prompt()
        self.assertIn("computer science professor", professor)
        self.assertIn("English research README", professor)
        self.assertIn("evidence", professor)
        self.assertLess(len(professor), 200)
        self.assertNotIn("submit_readme", professor)

    def test_readme_content_requirements_are_in_markdown_parameter(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from agents import ProfessorAgent
        from research_state import ResearchState

        agent = ProfessorAgent("test-model", "test-key")
        with patch.object(agent, "runtime") as runtime:
            runtime.return_value.run_one.return_value = SimpleNamespace(
                value={"success": True, "markdown": "# Research"}
            )
            agent.generate_readme(ResearchState("query", "recording", "topic"))

        registry = runtime.return_value.run_one.call_args.kwargs["registry"]
        self.assertEqual(registry.names, {"submit_readme"})
        parameters = registry.schemas[0]["function"]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["required"], ["markdown"])
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(set(parameters["properties"]), {"markdown"})
        self.assertEqual(parameters["properties"]["markdown"]["type"], "string")
        description = parameters["properties"]["markdown"]["description"]
        for requirement in (
            "Markdown", "research objective", "method", "repository outputs",
            "reproduction steps", "dependencies", "fixed numerical seeds",
            "observed metrics", "limitations", "only supported information",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, description)


if __name__ == "__main__":
    unittest.main()
