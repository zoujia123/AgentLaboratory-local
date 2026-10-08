import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_lab_repo import BatchConfig, QueryJob, run_query_job
from research_state import ResearchState
from seeding import SeedConfig, current_seed_config, seed_scope
from tools import OfflineExecutionError, execute_code, validate_offline_code


class SeedPolicyTest(unittest.TestCase):
    def setUp(self):
        self.seeds = SeedConfig(
            python=21,
            numpy=22,
            sklearn=23,
            pytorch=24,
            pytorch_deterministic=True,
        )

    def test_rejects_missing_or_wrong_sklearn_random_state(self):
        missing = (
            "from sklearn.model_selection import train_test_split\n"
            "train_test_split([1, 2, 3], test_size=0.2)\n"
        )
        wrong = (
            "from sklearn.linear_model import LogisticRegression\n"
            "LogisticRegression(random_state=999)\n"
        )
        with self.assertRaisesRegex(OfflineExecutionError, "missing random_state"):
            validate_offline_code(missing, self.seeds)
        with self.assertRaisesRegex(OfflineExecutionError, "SKLEARN_SEED"):
            validate_offline_code(wrong, self.seeds)

    def test_accepts_sklearn_seed_constant(self):
        code = (
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "train_test_split([1, 2, 3], random_state=SKLEARN_SEED)\n"
            "LogisticRegression(random_state=AGENTLAB_SEEDS['sklearn'])\n"
        )
        validate_offline_code(code, self.seeds)

    def test_distinguishes_datasets_and_sklearn_train_test_split_seeds(self):
        datasets_code = (
            "from datasets import Dataset\n"
            "dataset = Dataset.from_dict({'text': ['a', 'b', 'c']})\n"
            "parts = dataset.train_test_split(test_size=0.2, seed=SKLEARN_SEED)\n"
        )
        sklearn_alias_code = (
            "from sklearn.model_selection import train_test_split as split\n"
            "split([1, 2, 3], random_state=SKLEARN_SEED)\n"
        )
        validate_offline_code(datasets_code, self.seeds)
        validate_offline_code(sklearn_alias_code, self.seeds)

        with self.assertRaisesRegex(OfflineExecutionError, "missing random_state"):
            validate_offline_code(
                "from sklearn.model_selection import train_test_split as split\n"
                "split([1, 2, 3])\n",
                self.seeds,
            )

        with self.assertRaisesRegex(OfflineExecutionError, "must use seed"):
            validate_offline_code(
                "from datasets import Dataset\n"
                "dataset = Dataset.from_dict({'text': ['a', 'b', 'c']})\n"
                "dataset.train_test_split(test_size=0.2, random_state=SKLEARN_SEED)\n",
                self.seeds,
            )
        with self.assertRaisesRegex(OfflineExecutionError, "missing seed"):
            validate_offline_code(
                "from datasets import Dataset\n"
                "dataset = Dataset.from_dict({'text': ['a', 'b', 'c']})\n"
                "dataset.train_test_split(test_size=0.2)\n",
                self.seeds,
            )

    def test_requires_numpy_seed_for_default_rng(self):
        with self.assertRaisesRegex(OfflineExecutionError, "default_rng"):
            validate_offline_code(
                "import numpy as np\nrng = np.random.default_rng()\n",
                self.seeds,
            )
        validate_offline_code(
            "import numpy as np\nrng = np.random.default_rng(NUMPY_SEED)\n",
            self.seeds,
        )
        validate_offline_code(
            "import numpy as np\nrng = np.random.default_rng(seed=NUMPY_SEED)\n",
            self.seeds,
        )
        with self.assertRaisesRegex(OfflineExecutionError, "default_rng"):
            validate_offline_code(
                "import numpy as np\nrng = np.random.default_rng(seed=999)\n",
                self.seeds,
            )

    def test_worker_output_is_repeatable_and_constants_are_injected(self):
        code = (
            "import random\n"
            "import numpy as np\n"
            "print(PYTHON_SEED, NUMPY_SEED, SKLEARN_SEED, PYTORCH_SEED)\n"
            "print(AGENTLAB_SEEDS['sklearn'])\n"
            "print(f'{random.random():.12f}')\n"
            "print(f'{np.random.random():.12f}')\n"
        )
        first = execute_code(code, seeds=self.seeds)
        second = execute_code(code, seeds=self.seeds)
        self.assertNotIn("[CODE EXECUTION ERROR]", first)
        self.assertEqual(first, second)
        self.assertIn("21 22 23 24", first)
        self.assertIn("\n23\n", first)

    def test_seed_scopes_are_isolated_between_concurrent_queries(self):
        barrier = threading.Barrier(2)
        first = SeedConfig(python=101, numpy=102, sklearn=103, pytorch=104)
        second = SeedConfig(python=201, numpy=202, sklearn=203, pytorch=204)

        def read_scoped_seeds(config):
            with seed_scope(config):
                barrier.wait(timeout=5)
                return current_seed_config().to_dict()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(read_scoped_seeds, first),
                executor.submit(read_scoped_seeds, second),
            ]
            observed = [future.result() for future in futures]
        self.assertEqual(observed, [first.to_dict(), second.to_dict()])

    def test_state_and_job_summary_record_seeds(self):
        state = ResearchState(
            "query_001",
            "recording_001",
            "topic",
            seeds=self.seeds.to_dict(),
        )
        self.assertEqual(
            json.loads(json.dumps(state.to_dict()))["seeds"],
            self.seeds.to_dict(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            replay = SimpleNamespace(
                query_id="query_001",
                recording_id="recording_001",
                user_query="topic",
                write_trace=MagicMock(return_value={"success": True}),
            )
            batch = BatchConfig(
                material_root=temp_dir,
                max_concurrent_queries=1,
                queries=(QueryJob("query_001", "recording_001"),),
                output_root=temp_dir,
                collaboration_rounds={
                    "plan_max_rounds": 8,
                    "data_preparation_max_rounds": 12,
                    "interpretation_max_phd_reviews": 8,
                },
                code_execution_timeout_seconds=123,
                seeds=self.seeds,
            )
            workflow = MagicMock()
            with patch(
                "ai_lab_repo.LocalReplaySession.from_config",
                return_value=replay,
            ), patch("ai_lab_repo.LaboratoryWorkflow", return_value=workflow):
                summary = run_query_job(
                    batch,
                    batch.queries[0],
                    "model",
                    "key",
                    output_root=temp_dir,
                )
            self.assertEqual(summary["seeds"], self.seeds.to_dict())
            saved_summary = json.loads(
                Path(summary["workdir"], "job_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved_summary["seeds"], self.seeds.to_dict())
            self.assertEqual(
                saved_summary["llm_http_attempt_budget"],
                {
                    "limit": 100,
                    "used": 0,
                    "remaining": 100,
                    "limit_reached": False,
                    "request_blocked": False,
                },
            )


if __name__ == "__main__":
    unittest.main()
