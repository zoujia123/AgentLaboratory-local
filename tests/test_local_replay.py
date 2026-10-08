import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image as PILImage

from local_replay import (
    LocalReplayMismatch,
    LocalReplaySession,
    _runtime_sample_types,
    local_replay_scope,
)
from tools import ArxivSearch, HFDataSearch, execute_code


class LocalReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.query_id = "query_001"
        self.recording_id = "recording_001"
        self.recording_dir = (
            self.root / "queries" / self.query_id / "recordings" / self.recording_id
        )
        self._build_fixture()

    def _write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _build_fixture(self):
        self._write_json(self.root / "schema.json", {"schema_version": 1})
        self._write_json(
            self.recording_dir / "recording_manifest.json",
            {
                "schema_version": 1,
                "query_id": self.query_id,
                "recording_id": self.recording_id,
                "user_query": "test query",
                "status": "complete",
                "materialization_status": "complete",
            },
        )
        arxiv_search_dir = self.recording_dir / "arxiv" / "searches"
        self._write_json(
            arxiv_search_dir / "call_000001.json",
            {
                "schema_version": 1,
                "call_index": 1,
                "query": "first query",
                "processed_query": "first query",
                "max_results": 1,
                "latency_ms": 100.0,
                "success": True,
                "results": [{
                    "rank": 1,
                    "arxiv_id": "1111.00001v1",
                    "title": "First Paper",
                    "abstract": "First abstract",
                    "published": "2024-01-02T03:04:05+00:00",
                }],
            },
        )
        self._write_json(
            arxiv_search_dir / "call_000002.json",
            {
                "schema_version": 1,
                "call_index": 2,
                "query": "second query",
                "processed_query": "second query",
                "max_results": 1,
                "latency_ms": 20.0,
                "success": True,
                "results": [{
                    "rank": 1,
                    "arxiv_id": "2222.00002v2",
                    "title": "Second Paper",
                    "abstract": "Second abstract",
                    "published": "2025-02-03T00:00:00+00:00",
                }, {
                    "rank": 2,
                    "arxiv_id": "3333.00003v1",
                    "title": "Third Paper",
                    "abstract": "Quantum systems and optimization",
                    "published": "2023-03-04T00:00:00+00:00",
                }],
            },
        )

        self._write_arxiv_object("1111.00001v1", "first candidate text")
        self._write_arxiv_object("2222.00002v2", "fallback candidate text")
        self._write_arxiv_object("3333.00003v1", "unrecorded candidate text")

        hf_search_dir = self.recording_dir / "huggingface" / "searches"
        self._write_json(
            hf_search_dir / "call_000001.json",
            {
                "schema_version": 1,
                "call_index": 1,
                "query": "small dataset",
                "max_results": 2,
                "latency_ms": 15.0,
                "success": True,
                "results": [{
                    "id": "owner/recorded",
                    "description": "recorded dataset",
                    "likes": 10,
                    "downloads": 100,
                    "has_test_set": False,
                    "has_train_set": True,
                    "test_download_size": None,
                    "test_element_size": None,
                    "train_download_size": "1 KiB",
                    "train_element_size": 2,
                }, {
                    "id": "owner/fallback",
                    "description": "auxiliary records",
                }],
            },
        )
        self._write_hf_object("owner/recorded", "revision-one", [1, 2])
        self._write_hf_object("owner/fallback", "revision-two", [3, 4, 5])

    def _write_arxiv_object(self, arxiv_id, text):
        object_dir = self.root / "objects" / "arxiv" / arxiv_id
        object_dir.mkdir(parents=True, exist_ok=True)
        text_path = object_dir / "text.txt"
        text_path.write_text(text, encoding="utf-8")
        self._write_json(
            object_dir / "metadata.json",
            {
                "arxiv_id": arxiv_id,
                "snapshot_status": "complete",
                "extraction_status": "complete",
                "text_path": str(text_path.relative_to(self.root)),
                "download_latency_ms": 10.0,
                "extraction_latency_ms": 5.0,
            },
        )

    def _write_hf_object(self, dataset_id, revision, values):
        safe_id = dataset_id.replace("/", "--")
        object_dir = self.root / "objects" / "huggingface" / safe_id / revision
        parquet_path = object_dir / "materialized" / "default" / "train.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_dict({"value": values}).to_parquet(str(parquet_path))
        self._write_json(
            object_dir / "manifest.json",
            {
                "dataset_id": dataset_id,
                "revision": revision,
                "snapshot_status": "complete",
                "materialization_latency_ms": 25.0,
                "materialized": {
                    "default": {
                        "train": {
                            "path": str(parquet_path.relative_to(self.root)),
                            "rows": len(values),
                        }
                    }
                },
            },
        )

    def make_session(self, **overrides):
        config = {
            "enabled": True,
            "root": str(self.root),
            "query-id": self.query_id,
            "recording-id": self.recording_id,
            "match-mode": "hybrid",
            "arxiv-search-mode": "local-bm25",
            "strict-offline": True,
            "latency-mode": "none",
        }
        config.update(overrides)
        return LocalReplaySession.from_config(config)

    def test_latest_selects_the_complete_recording(self):
        session = self.make_session(**{"recording-id": "latest"})
        self.assertEqual(session.recording_id, self.recording_id)
        self.assertEqual(session.user_query, "test query")

    def test_arxiv_search_ranks_current_query_without_network(self):
        session = self.make_session()
        with local_replay_scope(session), patch(
            "socket.create_connection",
            side_effect=AssertionError("network access is forbidden"),
        ):
            search = ArxivSearch()
            second = search.find_papers("  SECOND   QUERY ")
            first = search.find_papers("first query")

        self.assertIn("Second Paper", second["text"])
        self.assertIn("Publication Date: 2025-02-03", second["text"])
        self.assertIn("First Paper", first["text"])
        self.assertEqual(second["candidate_ids"], ["2222.00002v2"])
        self.assertEqual(first["candidate_ids"], ["1111.00001v1"])
        self.assertEqual(
            [event["match_kind"] for event in session.events],
            ["local_bm25", "local_bm25"],
        )
        self.assertEqual(
            [event["call_index"] for event in session.events],
            [1, 2],
        )

    def test_unrecorded_search_text_uses_local_candidate_index(self):
        session = self.make_session()
        with local_replay_scope(session):
            result = ArxivSearch().find_papers("second abstract topic")
        self.assertIn("Second Paper", result["text"])
        self.assertEqual(session.events[-1]["match_kind"], "local_bm25")

    def test_arxiv_search_uses_deterministic_synthetic_latency_when_exhausted(self):
        session = self.make_session()
        with local_replay_scope(session):
            search = ArxivSearch()
            search.find_papers("first")
            search.find_papers("second")
            search.find_papers("third")
        self.assertEqual(session.events[-1]["match_kind"], "local_bm25")
        self.assertEqual(session.events[-1]["latency_match_kind"], "synthetic_median")
        self.assertEqual(session.events[-1]["reservation_index"], 3)

    def test_arxiv_full_text_reads_candidate_objects_directly(self):
        session = self.make_session()
        with local_replay_scope(session):
            search = ArxivSearch()
            self.assertEqual(
                search.retrieve_full_paper_text("1111.00001v1"),
                "first candidate text",
            )
            self.assertEqual(
                search.retrieve_full_paper_text("2222.00002"),
                "fallback candidate text",
            )
            self.assertEqual(
                search.retrieve_full_paper_text("3333.00003v1"),
                "unrecorded candidate text",
            )
        self.assertEqual(
            [event["match_kind"] for event in session.events],
            ["object_direct", "object_direct", "object_direct"],
        )

    def test_arxiv_metadata_text_fallback_is_replayable(self):
        metadata_path = (
            self.root / "objects" / "arxiv" / "3333.00003v1" / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["extraction_status"] = "abstract_fallback"
        self._write_json(metadata_path, metadata)

        session = self.make_session()
        with local_replay_scope(session):
            text = ArxivSearch().retrieve_full_paper_text("3333.00003v1")

        self.assertEqual(text, "unrecorded candidate text")
        self.assertEqual(session.events[-1]["match_kind"], "object_direct")

    def test_excluded_arxiv_candidate_is_not_searchable_or_readable(self):
        excluded_dir = self.recording_dir / "arxiv" / "excluded"
        excluded_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            excluded_dir / "3333.00003v1.json",
            {
                "arxiv_id": "3333.00003v1",
                "reason": "http_404",
                "status": "excluded",
            },
        )

        session = self.make_session()
        self.assertNotIn("3333.00003v1", session.arxiv_candidates)
        with local_replay_scope(session):
            with self.assertRaisesRegex(LocalReplayMismatch, "was excluded"):
                ArxivSearch().retrieve_full_paper_text("3333.00003v1")

    def test_hf_search_replays_recorded_candidates(self):
        session = self.make_session()
        with local_replay_scope(session):
            search = HFDataSearch()
            results = search.retrieve_ds("small dataset", N=1)
        self.assertEqual(results[0]["id"], "owner/recorded")
        self.assertEqual(results[0]["description"], "recorded dataset")

    def test_excluded_hf_candidates_are_not_searchable_or_selectable(self):
        excluded_dir = self.recording_dir / "huggingface" / "excluded"
        for dataset_id in ("owner/recorded", "owner/fallback"):
            self._write_json(
                excluded_dir / f"{dataset_id.replace('/', '--')}.json",
                {
                    "dataset_id": dataset_id,
                    "reason": "irrelevant_auxiliary_candidate",
                    "status": "excluded",
                },
            )

        session = self.make_session()
        self.assertEqual(session.hf_candidates, set())
        with local_replay_scope(session):
            self.assertEqual(HFDataSearch().retrieve_ds("dataset", N=2), [])
        with self.assertRaisesRegex(LocalReplayMismatch, "was excluded"):
            session.select_hf_dataset("owner/recorded", "default", "train")

    def test_hf_search_has_no_candidates_without_recorded_search_calls(self):
        search_dir = self.recording_dir / "huggingface" / "searches"
        for path in search_dir.glob("*.json"):
            path.unlink()
        session = self.make_session()
        with local_replay_scope(session):
            results = HFDataSearch().retrieve_ds("fallback", N=2)
        self.assertEqual(results, [])
        self.assertEqual(session.hf_candidates, set())
        self.assertEqual(session.events[-1]["latency_match_kind"], "synthetic_median")

    def test_generated_code_loads_candidate_objects_directly(self):
        session = self.make_session()
        code = """
from datasets import load_dataset
recorded = load_dataset('owner/recorded', split='train')
fallback = load_dataset('owner/fallback', split='train')
print(len(recorded), len(fallback), sum(recorded['value']), sum(fallback['value']))
"""
        with local_replay_scope(session):
            output = execute_code(code, timeout=30)
        self.assertEqual(output.strip(), "2 3 3 12")
        hf_events = [
            event for event in session.events
            if event.get("tool") == "huggingface.load_dataset"
        ]
        self.assertEqual(
            [event["match_kind"] for event in hf_events],
            ["object_direct", "object_direct"],
        )

    def test_execute_code_details_report_local_load_and_reusable_artifacts(self):
        session = self.make_session()
        code = (
            "from datasets import load_dataset\n"
            "data = load_dataset('owner/recorded', split='train', "
            "revision='revision-one')\n"
        )
        with local_replay_scope(session):
            details = execute_code(code, timeout=30, return_details=True)

        self.assertEqual(details["output"], "")
        self.assertEqual(
            details["replay_events"][0]["request"]["dataset_id"],
            "owner/recorded",
        )
        artifacts = {item["name"]: item for item in details["artifacts"]}
        self.assertEqual(artifacts["data"]["length"], 2)
        self.assertIn("datasets", artifacts["data"]["type"])

    def test_preparation_execution_captures_dataset_for_workflow(self):
        from data_preparation import _run_preparation_code
        session = self.make_session()
        inspected = session.select_hf_dataset(
            "owner/recorded", revision="revision-one", config_name="default", split="train",
        )
        code = (
            "from datasets import load_dataset\n"
            "data = load_dataset('owner/recorded', split='train')\n"
            "print(len(data))\n"
        )
        with local_replay_scope(session):
            result = _run_preparation_code(code, str(self.root), inspected, code_execution_timeout_seconds=30)
        self.assertTrue(result["success"], result["output"])
        self.assertEqual(result["code"], code)
        self.assertEqual(result["output"].strip(), "2")
        self.assertEqual(result["selection"]["dataset_id"], "owner/recorded")
        self.assertEqual(result["selection"]["split"], "train")
        self.assertEqual(result["selection"], inspected)
        self.assertEqual(result["executed_dataset"], {
            "dataset_id": "owner/recorded", "revision": "revision-one",
            "resources": [{"config_name": "default", "split": "train"}],
        })

    def test_split_none_returns_dataset_dict_and_records_resolved_resources(self):
        object_dir = (
            self.root
            / "objects"
            / "huggingface"
            / "owner--recorded"
            / "revision-one"
        )
        test_path = object_dir / "materialized" / "default" / "test.parquet"
        Dataset.from_dict({"value": [3]}).to_parquet(str(test_path))
        manifest_path = object_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["materialized"]["default"]["test"] = {
            "path": str(test_path.relative_to(self.root)),
            "rows": 1,
        }
        self._write_json(manifest_path, manifest)

        session = self.make_session()
        code = (
            "from datasets import load_dataset\n"
            "data = load_dataset('owner/recorded', revision='revision-one')\n"
            "print(sorted(data), len(data['train']), len(data['test']))\n"
        )
        with local_replay_scope(session):
            details = execute_code(code, timeout=30, return_details=True)

        self.assertEqual(details["output"].strip(), "['test', 'train'] 2 1")
        self.assertEqual(
            details["replay_events"][0]["resolved_resources"],
            [
                {"config_name": "default", "split": "test"},
                {"config_name": "default", "split": "train"},
            ],
        )

    def test_execute_code_reports_numeric_artifacts_and_prepared_manifest(self):
        code = (
            "import numpy as np\n"
            "X_train = np.array([[1.0], [2.0]])\n"
            "y_train = np.array([0, 1])\n"
            "prepared_data_manifest = {\n"
            "    'source_feature_columns': ['value'],\n"
            "    'target_columns': ['label'],\n"
            "    'analysis_columns': [],\n"
            "    'feature_artifacts': {'train': 'X_train'},\n"
            "    'target_artifacts': {'train': 'y_train'},\n"
            "    'analysis_artifacts': {},\n"
            "}\n"
        )
        details = execute_code(code, timeout=30, return_details=True)
        artifacts = {item["name"]: item for item in details["artifacts"]}

        self.assertTrue(artifacts["X_train"]["numeric"])
        self.assertEqual(artifacts["X_train"]["dtype"], "float64")
        self.assertEqual(
            details["prepared_data_manifest"]["feature_artifacts"]["train"],
            "X_train",
        )

    def test_hf_load_rejects_dataset_outside_current_query_candidates(self):
        session = self.make_session()
        with local_replay_scope(session):
            output = execute_code(
                "from datasets import load_dataset\n"
                "load_dataset('owner/not-a-candidate', split='train')\n",
                timeout=30,
            )
        self.assertIn("does not belong to query", output)

    def test_hf_load_uses_first_sorted_revision_when_unpinned(self):
        self._write_hf_object("owner/recorded", "000-first", [9])
        session = self.make_session()

        selection = session.select_hf_dataset(
            "owner/recorded",
            "default",
            "train",
        )
        self.assertEqual(selection["revision"], "000-first")
        self.assertEqual(
            selection["dataset_schema"],
            {"row_count": 1, "columns": {"value": "int64"}},
        )
        self.assertEqual(selection["stored_schema"], selection["dataset_schema"])
        self.assertEqual(selection["runtime_sample_types"], {"value": "int"})
        with local_replay_scope(session):
            output = execute_code(
                "from datasets import load_dataset\n"
                "data = load_dataset('owner/recorded', split='train')\n"
                "print(len(data), sum(data['value']))\n",
                timeout=30,
            )
        self.assertEqual(output.strip(), "1 9")

        pinned = session.select_hf_dataset(
            "owner/recorded",
            "default",
            "train",
            revision="revision-one",
        )
        self.assertEqual(pinned["revision"], "revision-one")
        self.assertEqual(
            pinned["dataset_schema"],
            {"row_count": 2, "columns": {"value": "int64"}},
        )
        self.assertEqual(pinned["runtime_sample_types"], {"value": "int"})

    def test_hf_inspection_reports_decoded_image_runtime_type(self):
        object_dir = (
            self.root
            / "objects"
            / "huggingface"
            / "owner--image"
            / "revision-image"
        )
        parquet_path = object_dir / "materialized" / "default" / "train.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        image_bytes = io.BytesIO()
        PILImage.new("RGB", (2, 2), (1, 2, 3)).save(image_bytes, format="PNG")
        dataset = Dataset.from_dict(
            {
                "img": [{"bytes": image_bytes.getvalue(), "path": "sample.png"}],
                "label": [1],
            },
            features=Features({"img": HFImage(), "label": Value("int64")}),
        )
        dataset.to_parquet(str(parquet_path))
        self._write_json(
            object_dir / "manifest.json",
            {
                "dataset_id": "owner/image",
                "revision": "revision-image",
                "snapshot_status": "complete",
                "materialized": {
                    "default": {
                        "train": {
                            "path": str(parquet_path.relative_to(self.root)),
                            "rows": 1,
                        }
                    }
                },
            },
        )
        session = self.make_session()
        session.hf_candidates.add("owner/image")
        selection = session.select_hf_dataset(
            "owner/image",
            "default",
            "train",
            revision="revision-image",
        )
        self.assertEqual(
            selection["runtime_sample_types"],
            {"img": "PngImageFile", "label": "int"},
        )

    def test_runtime_sample_probe_handles_empty_and_broken_inputs(self):
        class EmptyParquet:
            class Metadata:
                num_rows = 0

            metadata = Metadata()

        self.assertEqual(_runtime_sample_types(EmptyParquet(), None), ({}, None))

        class BrokenParquet:
            class Metadata:
                num_rows = 1

            metadata = Metadata()

            def iter_batches(self, **kwargs):
                raise RuntimeError("probe failed")

        types, error = _runtime_sample_types(BrokenParquet(), None)
        self.assertEqual(types, {})
        self.assertEqual(error, "RuntimeError: probe failed")

    def test_generated_code_loads_additional_materialized_config(self):
        object_dir = (
            self.root
            / "objects"
            / "huggingface"
            / "owner--recorded"
            / "revision-one"
        )
        parquet_path = object_dir / "materialized" / "corpus" / "corpus.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_dict({"value": [7, 8]}).to_parquet(str(parquet_path))
        manifest_path = object_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["materialized"]["corpus"] = {
            "corpus": {
                "path": str(parquet_path.relative_to(self.root)),
                "rows": 2,
            }
        }
        self._write_json(manifest_path, manifest)

        session = self.make_session()
        selection = session.select_hf_dataset(
            "owner/recorded",
            "corpus",
            "corpus",
        )
        self.assertIn("corpus", selection["available_configs"])
        resources = {
            (item["config_name"], item["split"])
            for item in selection["materialized_resources"]
        }
        self.assertEqual(
            resources,
            {("default", "train"), ("corpus", "corpus")},
        )
        with local_replay_scope(session):
            output = execute_code(
                "from datasets import load_dataset\n"
                "data = load_dataset('owner/recorded', name='corpus', split='corpus')\n"
                "print(len(data), sum(data['value']))\n",
                timeout=30,
            )
        self.assertEqual(output.strip(), "2 15")

    def test_generated_code_rejects_network_imports(self):
        output = execute_code("import requests\nprint('network')", timeout=10)
        self.assertIn("strict offline", output)

    def test_generated_code_rejects_unmaterialized_pretrained_models(self):
        output = execute_code(
            "AutoTokenizer.from_pretrained('bert-base-uncased')",
            timeout=10,
        )
        self.assertIn("strict offline", output)
        self.assertIn("from_pretrained", output)

    def test_recorded_latency_sleeps_only_for_remaining_duration(self):
        session = self.make_session(**{"latency-mode": "recorded"})
        with local_replay_scope(session), patch("local_replay.time.sleep") as sleep:
            ArxivSearch().find_papers("first query")
        sleep.assert_called_once()
        self.assertGreater(sleep.call_args.args[0], 0.05)
        self.assertLessEqual(sleep.call_args.args[0], 0.1)

    def test_structured_arxiv_search_returns_one_non_excluded_candidate(self):
        session = self.make_session()
        with local_replay_scope(session):
            details = ArxivSearch().find_papers(
                "abstract quantum",
                excluded_ids={"1111.00001v1"},
            )

        self.assertNotIn("1111.00001v1", details["candidate_ids"])
        self.assertEqual(details["candidate_ids"], ["3333.00003v1"])

    def test_trace_summary_is_written_outside_material_root(self):
        session = self.make_session()
        with local_replay_scope(session):
            ArxivSearch().find_papers("first query")
        with tempfile.TemporaryDirectory() as trace_temp:
            trace_dir = Path(trace_temp)
            summary = session.write_trace(trace_dir)
            self.assertEqual(summary["event_count"], 1)
            self.assertEqual(
                set(summary["unused_calls"]),
                {"arxiv.search", "huggingface.search"},
            )
            self.assertTrue((trace_dir / "local_replay_trace.jsonl").exists())
            self.assertTrue((trace_dir / "local_replay_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
