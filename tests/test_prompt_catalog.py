import unittest
from unittest.mock import patch

with patch("dotenv.load_dotenv"):
    from ai_lab_repo import load_paper_writing_config
from prompts import (
    COMMON_CODE_GUIDANCE,
    experiment_solver_prompt,
    ml_dataset_selection_prompt,
    paper_blueprint_prompt,
    paper_section_prompt,
    professor_readme_prompt,
    supplementary_experiment_solver_prompt,
    sw_data_preparation_prompt,
)


class PromptCatalogTest(unittest.TestCase):
    def test_common_code_guidance_documents_container_boundaries(self):
        for fragment in (
            "PIL.Image.Image",
            "Dataset.map(batched=True)",
            "train_test_split uses seed",
            "list of records",
        ):
            self.assertIn(fragment, COMMON_CODE_GUIDANCE)

    def test_preparation_prompts_describe_main_roles_and_commands(self):
        ml = ml_dataset_selection_prompt()
        sw = sw_data_preparation_prompt()
        self.assertIn("directed by the SWEngineerAgent", ml)
        for tool in ("search_huggingface_datasets", "write_and_run_dataset_code", "send_dialogue"):
            self.assertIn(tool, ml)
        self.assertIn("You act first", sw)
        self.assertIn("send_dialogue", sw)
        self.assertIn("Your only tool is send_dialogue", sw)
        self.assertIn("You cannot execute or submit data-preparation code", sw)
        self.assertNotIn("submit_dataset_code", sw)
        self.assertNotIn("replace_dataset_code", sw)
        for prompt in (ml, sw):
            self.assertNotIn("configure_dataset_contract", prompt)
            self.assertIn("seed", prompt.lower())
            self.assertIn("Runtime object discipline", prompt)
            self.assertIn("Dataset.map(batched=True)", prompt)
            self.assertIn("runtime_sample_types", prompt)
            self.assertIn("stored_schema", prompt)

    def test_experiment_prompts_preserve_solver_quality_constraints(self):
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
        self.assertIn("write_experiment_code's code argument", prompt)
        self.assertIn("edit_experiment_code", prompt)
        self.assertIn("all changes in its edits array", prompt)
        self.assertIn("execute_experiment_code", prompt)
        self.assertIn("no arguments", prompt)

        supplement = supplementary_experiment_solver_prompt("supplement_01")
        self.assertIn("supplementary experiment", supplement)
        self.assertIn("read-only evidence", supplement)
        self.assertIn("never replace", supplement)
        self.assertIn("supplement_01_figure.png", supplement)
        self.assertIn("Never fabricate", supplement)
        self.assertIn("Runtime object discipline", supplement)
        self.assertIn("write_supplement_code's code argument", supplement)
        self.assertIn("edit_supplement_code", supplement)
        self.assertIn("old_text", supplement)
        self.assertIn("new_text", supplement)
        self.assertNotIn("edits array", supplement)
        self.assertIn("execute_supplement_code", supplement)
        self.assertIn("no arguments", supplement)

    def test_paper_prompts_use_blueprint_and_bounded_sections(self):
        prompt = paper_blueprint_prompt(
            [section["name"] for section in load_paper_writing_config()["sections"]],
            ["Figure_1.png", "Figure_2.png"],
        )
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

    def test_professor_readme_prompt_is_complete(self):
        professor = professor_readme_prompt()
        self.assertIn("research topic", professor)
        self.assertIn("experiment code", professor)
        self.assertIn("reproduction steps", professor)
        self.assertIn("observed metrics", professor)


if __name__ == "__main__":
    unittest.main()
