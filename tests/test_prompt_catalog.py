import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_runtime import AgentRuntime
from papersolver import ReviewerAgent
from prompts import (
    experiment_reward_prompt,
    experiment_solver_prompt,
    ml_dataset_selection_prompt,
    paper_blueprint_prompt,
    paper_reward_prompt,
    paper_section_prompt,
    paper_writer_prompt,
    professor_readme_prompt,
    reviewer_prompt,
    supplementary_experiment_solver_prompt,
    sw_data_preparation_prompt,
)


class PromptCatalogTest(unittest.TestCase):
    def test_preparation_prompts_describe_main_roles_and_commands(self):
        ml = ml_dataset_selection_prompt()
        sw = sw_data_preparation_prompt()
        self.assertIn("directed by the SWEngineerAgent", ml)
        for tool in ("search_huggingface", "execute_dataset_code", "send_dialogue"):
            self.assertIn(tool, ml)
        self.assertIn("You act first", sw)
        self.assertIn("send_dialogue", sw)
        self.assertIn("submit_dataset_code", sw)
        self.assertNotIn("replace_dataset_code", sw)
        for prompt in (ml, sw):
            self.assertNotIn("configure_dataset_contract", prompt)
            self.assertIn("seed", prompt.lower())
            self.assertIn("Runtime object discipline", prompt)
            self.assertIn("Dataset.map(batched=True)", prompt)
            self.assertIn("runtime_sample_types", prompt)
            self.assertIn("stored_schema", prompt)

    def test_experiment_prompts_preserve_old_solver_quality_constraints(self):
        prompt = experiment_solver_prompt()
        self.assertIn("Implement every experiment", prompt)
        self.assertIn("literature review", prompt)
        self.assertIn("dataset code is automatically prepended", prompt)
        self.assertIn("do not choose BERT", prompt)
        self.assertIn("scikit-learn-compatible model trainable from scratch", prompt)
        self.assertIn("comparing input representations", prompt)
        self.assertIn("Figure_1.png", prompt)
        self.assertIn("Figure_2.png", prompt)
        self.assertIn("0% accuracy", prompt)
        self.assertIn("explain what the metric", prompt)
        self.assertIn("reflect on errors", prompt)
        self.assertIn("PIL.Image.Image", prompt)
        self.assertIn("list of records", prompt)

        supplement = supplementary_experiment_solver_prompt("supplement_01")
        self.assertIn("supplementary experiment", supplement)
        self.assertIn("read-only evidence", supplement)
        self.assertIn("never replace", supplement)
        self.assertIn("supplement_01_figure.png", supplement)
        self.assertIn("Never fabricate", supplement)
        self.assertIn("Runtime object discipline", supplement)

        reward = experiment_reward_prompt()
        self.assertIn("research plan", reward)
        self.assertIn("final experiment code", reward)
        self.assertIn("observed output", reward)
        self.assertIn("0-1 score", reward)
        self.assertIn("Penalize missing experiments", reward)

    def test_paper_prompts_use_blueprint_and_bounded_sections(self):
        prompt = paper_writer_prompt(["Figure_1.png", "Figure_2.png"])
        self.assertIn("paper blueprint", prompt)
        self.assertIn("do not write any section body", prompt)
        self.assertIn("Background", prompt)
        self.assertIn("submit_paper_blueprint", prompt)
        self.assertIn("Figure_1.png", prompt)
        self.assertIn("Figure_2.png", prompt)

        blueprint = paper_blueprint_prompt(
            ["Abstract", "Introduction"],
            ["Figure_1.png"],
        )
        self.assertIn("Abstract, Introduction", blueprint)
        section = paper_section_prompt("Results", 600, ["Figure_1.png"])
        self.assertIn("fixed section 'Results'", section)
        self.assertIn("approximately 600 words", section)
        self.assertIn("submit_paper_section", section)
        self.assertIn("framework owns", section)

    def test_professor_and_paper_reward_prompts_are_complete(self):
        professor = professor_readme_prompt()
        self.assertIn("research topic", professor)
        self.assertIn("experiment code", professor)
        self.assertIn("reproduction steps", professor)
        self.assertIn("observed metrics", professor)

        reward = paper_reward_prompt()
        self.assertIn("three independent reviews", reward)
        self.assertIn("contribution", reward)
        self.assertIn("presentation", reward)
        self.assertIn("soundness", reward)
        self.assertIn("reproducibility", reward)

    def test_reviewer_prompt_and_tool_schema_restore_full_rubric(self):
        prompt = reviewer_prompt(1, "profile")
        for term in (
            "originality",
            "technical quality",
            "clarity",
            "significance",
            "soundness",
            "presentation",
            "contribution",
            "reproducibility",
            "limitations",
            "ethical concerns",
        ):
            self.assertIn(term, prompt)

        agent = ReviewerAgent("model", "key", 1, "profile")
        state = SimpleNamespace(plan="plan", experiment_output="output", paper="paper")
        completed = SimpleNamespace(value={"review": "review"})
        with patch.object(
            AgentRuntime, "run_one", return_value=completed
        ) as run:
            agent.review(state)
        registry = run.call_args.kwargs["registry"]
        properties = registry.schemas[0]["function"]["parameters"]["properties"]
        expected = {
            "summary",
            "strengths",
            "weaknesses",
            "originality",
            "quality",
            "clarity",
            "significance",
            "questions",
            "limitations",
            "ethical_concerns",
            "soundness",
            "presentation",
            "contribution",
            "overall_score",
            "confidence",
            "decision",
        }
        self.assertEqual(set(properties), expected)


if __name__ == "__main__":
    unittest.main()
