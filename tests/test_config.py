import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab_repo import load_runtime_environment, parse_yaml
from seeding import SeedConfig


class ConfigTest(unittest.TestCase):
    def write_config(self, content):
        config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        )
        config_file.write(
            "llm_sampling:\n"
            "  temperature: 0.1\n"
            "  top_p: 1\n"
            "  max_tokens: 8192\n"
            "collaboration_rounds:\n"
            "  plan_max_rounds: 8\n"
            "  data_preparation_max_rounds: 12\n"
            "  interpretation_max_phd_reviews: 8\n"
            + content
        )
        config_file.close()
        self.addCleanup(os.remove, config_file.name)
        return config_file.name

    def test_batch_config_uses_latest_recording_by_default(self):
        config = parse_yaml(self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results\n"
            "max-concurrent-queries: 8\n"
            "queries:\n"
            "  - query-id: query_001\n"
        ))
        self.assertEqual(config.max_concurrent_queries, 8)
        self.assertEqual(config.queries[0].query_id, "query_001")
        self.assertEqual(config.queries[0].recording_id, "latest")
        self.assertFalse(config.compile_latex)
        self.assertFalse(config.console_log)
        self.assertEqual(config.max_llm_http_attempts_per_query, 100)

    def test_parses_console_log_flag(self):
        config = parse_yaml(self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results\n"
            "console-log: true\n"
            "queries:\n"
            "  - query-id: query_001\n"
        ))
        self.assertTrue(config.console_log)

    def test_v2_default_config_targets_isolated_materials_and_outputs(self):
        from ai_lab_repo import DEFAULT_CONFIG

        config = parse_yaml(DEFAULT_CONFIG)

        self.assertEqual(Path(config.material_root).name, "MASLab_v2")
        self.assertEqual(Path(config.output_root).name, "research_runs_v2")
        self.assertEqual(config.max_concurrent_queries, 2)
        self.assertEqual(len(config.queries), 20)
        self.assertEqual(config.queries[0].query_id, "query_001")
        self.assertEqual(config.queries[-1].query_id, "query_020")

    def test_parses_explicit_output_root(self):
        config = parse_yaml(self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results-v2\n"
            "queries:\n"
            "  - query-id: query_001\n"
        ))
        self.assertEqual(config.output_root, str(Path("D:/results-v2").resolve()))

    def test_parses_llm_http_attempt_limit(self):
        config = parse_yaml(self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results\n"
            "max-llm-http-attempts-per-query: 75\n"
            "queries:\n"
            "  - query-id: query_001\n"
        ))
        self.assertEqual(config.max_llm_http_attempts_per_query, 75)

    def test_llm_http_attempt_limit_must_be_positive(self):
        path = self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results\n"
            "max-llm-http-attempts-per-query: 0\n"
            "queries:\n"
            "  - query-id: query_001\n"
        )
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            parse_yaml(path)

    def test_concurrency_is_limited_to_ten(self):
        path = self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results\n"
            "max-concurrent-queries: 11\n"
            "queries:\n"
            "  - query-id: query_001\n"
        )
        with self.assertRaisesRegex(ValueError, "between 1 and 10"):
            parse_yaml(path)

    def test_parses_fixed_numerical_seeds(self):
        config = parse_yaml(self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results\n"
            "queries:\n"
            "  - query-id: query_001\n"
            "seeds:\n"
            "  strategy: fixed\n"
            "  python: 2001\n"
            "  numpy: 3001\n"
            "  sklearn: 4001\n"
            "  pytorch: 5001\n"
            "  pytorch-deterministic: true\n"
        ))
        self.assertEqual(
            config.seeds,
            SeedConfig(
                strategy="fixed",
                python=2001,
                numpy=3001,
                sklearn=4001,
                pytorch=5001,
                pytorch_deterministic=True,
            ),
        )

    def test_rejects_invalid_seed_configuration(self):
        cases = (
            ("  strategy: per-query\n", "strategy"),
            ("  python: true\n", "integer"),
            ("  numpy: -1\n", "between"),
            ("  sklearn: 4294967296\n", "between"),
            ("  pytorch-deterministic: warn\n", "true or false"),
            ("  unknown-seed: 1\n", "Unsupported seed fields"),
        )
        for seed_yaml, expected_error in cases:
            with self.subTest(seed_yaml=seed_yaml):
                path = self.write_config(
                    "material-root: D:/materials\n"
                    "output-root: D:/results\n"
                    "queries:\n"
                    "  - query-id: query_001\n"
                    "seeds:\n"
                    + seed_yaml
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    parse_yaml(path)

    def test_legacy_fields_are_rejected(self):
        path = self.write_config(
            "material-root: D:/materials\n"
            "output-root: D:/results\n"
            "research-topic: Test topic\n"
            "copilot-mode: false\n"
            "queries:\n"
            "  - query-id: query_001\n"
        )
        with self.assertRaisesRegex(ValueError, "Unsupported YAML fields"):
            parse_yaml(path)

    def test_model_and_key_come_from_environment(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "test-model",
        }, clear=True):
            self.assertEqual(load_runtime_environment(), ("test-key", "test-model"))

    def test_environment_requires_model_and_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                load_runtime_environment()


if __name__ == "__main__":
    unittest.main()
