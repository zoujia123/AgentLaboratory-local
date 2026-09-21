import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_lab_repo import BatchConfig, QueryJob, main, run_batch


class BatchRunnerTest(unittest.TestCase):
    def test_limits_concurrency_while_running_all_queries(self):
        jobs = tuple(QueryJob(f"query_{index:03d}") for index in range(6))
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_run_query(batch, job, model, api_key, output_root):
            del batch, model, api_key, output_root
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return {"query_id": job.query_id}

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_lab_repo.run_query_job", side_effect=fake_run_query
        ):
            config = BatchConfig(
                "D:/materials",
                3,
                jobs,
                output_root=str(Path(temp_dir, "runs")),
            )
            summary = run_batch(config, "model", "key")

        self.assertEqual(summary["success_count"], 6)
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(max_active, 3)

    def test_returns_and_persists_summary_when_a_query_fails(self):
        jobs = (QueryJob("query_001"), QueryJob("query_002"))

        def fake_run_query(batch, job, model, api_key, output_root):
            del batch, model, api_key, output_root
            if job.query_id == "query_002":
                raise ValueError("query failed")
            return {"query_id": job.query_id}

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "ai_lab_repo.run_query_job", side_effect=fake_run_query
        ):
            output_root = Path(temp_dir, "runs")
            config = BatchConfig(
                "D:/materials",
                2,
                jobs,
                output_root=str(output_root),
            )
            summary = run_batch(config, "model", "key")
            saved_summary = json.loads(
                Path(output_root, "latest_batch_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(
            summary["failures"],
            [{"query_id": "query_002", "error": "query failed"}],
        )
        self.assertEqual(saved_summary, summary)

    def test_main_exits_nonzero_after_partial_failure(self):
        config = BatchConfig(
            "D:/materials",
            1,
            (QueryJob("query_001"),),
            output_root="D:/results",
        )
        summary = {
            "success_count": 0,
            "failure_count": 1,
            "results": [],
            "failures": [{"query_id": "query_001", "error": "query failed"}],
        }

        with patch(
            "ai_lab_repo.parse_arguments",
            return_value=SimpleNamespace(yaml_location="config.yaml"),
        ), patch("ai_lab_repo.parse_yaml", return_value=config), patch(
            "ai_lab_repo.load_runtime_environment", return_value=("key", "model")
        ), patch("ai_lab_repo.run_batch", return_value=summary):
            with self.assertRaisesRegex(SystemExit, "1 query job.*failed") as caught:
                main()

        self.assertNotEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
