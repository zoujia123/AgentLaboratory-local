import contextlib
import contextvars
import copy
import json
import os
import re
import sqlite3
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from telemetry import current_trace_context, record_trace_event


DEFAULT_REPLAY_ROOT = r"D:\work\HUST\MASLab_v2"

_session_var = contextvars.ContextVar("agentlab_local_replay", default=None)


class LocalReplayError(RuntimeError):
    pass


class LocalReplayMismatch(LocalReplayError):
    pass


class LocalReplayIntegrityError(LocalReplayError):
    pass


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "--", str(value)).strip(".-")
    return value or "unknown"


def _arxiv_base_id(value):
    return re.sub(r"v\d+$", "", str(value).strip(), flags=re.IGNORECASE)


def _json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, ValueError) as exc:
        raise LocalReplayIntegrityError(f"Cannot read replay JSON {path}: {exc}") from exc


def _load_records(directory):
    directory = Path(directory)
    records = []
    if not directory.exists():
        return records
    for path in directory.glob("*.json"):
        payload = _json_file(path)
        payload["_source_path"] = str(path)
        records.append(payload)
    records.sort(key=lambda item: (
        item.get("call_index") is None,
        item.get("call_index", 0),
        item.get("recorded_at", ""),
        item.get("_source_path", ""),
    ))
    for index, record in enumerate(records, start=1):
        record["_sequence_index"] = index
    return records


def _sleep_to_target(started_ns, source_latency_ms, mode, scale):
    if mode != "recorded" or source_latency_ms is None:
        return 0.0
    target_ms = max(0.0, float(source_latency_ms) * float(scale))
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    remaining_ms = max(0.0, target_ms - elapsed_ms)
    if remaining_ms:
        time.sleep(remaining_ms / 1000.0)
    return remaining_ms


@contextlib.contextmanager
def local_replay_scope(session):
    token = _session_var.set(session)
    try:
        yield session
    finally:
        _session_var.reset(token)


def current_local_replay(required=False):
    session = _session_var.get()
    if required and session is None:
        raise LocalReplayError(
            "Local replay is not configured. Run with a YAML file containing local-replay settings."
        )
    return session


class LocalReplaySession:
    def __init__(self, root, query_id, recording_id, config=None):
        self.config = dict(config or {})
        self.root = Path(root).resolve()
        self.query_id = str(query_id)
        self.recording_id = str(recording_id)
        self.latency_mode = self.config.get("latency-mode", "recorded")
        self.latency_scale = float(self.config.get("latency-scale", 1.0))
        self.arxiv_search_mode = self.config.get("arxiv-search-mode", "local-bm25")

        self.recording_dir = (
            self.root / "queries" / self.query_id / "recordings" / self.recording_id
        )
        self.manifest_path = self.recording_dir / "recording_manifest.json"
        self.manifest = _json_file(self.manifest_path)
        self.user_query = str(self.manifest.get("user_query") or "").strip()
        # 加载 arXiv/HF 记录
        self.records = {
            "arxiv.search": _load_records(self.recording_dir / "arxiv" / "searches"),
            "huggingface.search": _load_records(self.recording_dir / "huggingface" / "searches"),
        }
        self._consumed = {
            name: [False] * len(records)
            for name, records in self.records.items()
        }
        self._match_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self.events = []
        self._reservation_counters = Counter()

        # 构建本地搜索索引
        self.arxiv_documents = {}
        self.arxiv_excluded = set()
        self.hf_excluded = set()
        self.hf_candidates = set()
        excluded_dir = self.recording_dir / "arxiv" / "excluded"
        for path in excluded_dir.glob("*.json"):
            excluded = _json_file(path)
            arxiv_id = excluded.get("arxiv_id") or excluded.get("object_id")
            if arxiv_id:
                self.arxiv_excluded.add(str(arxiv_id))
        for record in self.records["arxiv.search"]:
            for result in record.get("results", []):
                arxiv_id = result.get("arxiv_id")
                if (
                    arxiv_id
                    and arxiv_id not in self.arxiv_excluded
                    and arxiv_id not in self.arxiv_documents
                ):
                    self.arxiv_documents[arxiv_id] = copy.deepcopy(result)
        self.arxiv_candidates = set(self.arxiv_documents)
        self._initialize_arxiv_search()

        hf_excluded_dir = self.recording_dir / "huggingface" / "excluded"
        for path in hf_excluded_dir.glob("*.json"):
            excluded = _json_file(path)
            dataset_id = excluded.get("dataset_id") or excluded.get("object_id")
            if dataset_id:
                self.hf_excluded.add(str(dataset_id))
        for record in self.records["huggingface.search"]:
            self.hf_candidates.update(
                result.get("id")
                for result in record.get("results", [])
                if result.get("id") and result.get("id") not in self.hf_excluded
            )
        self._load_event_candidates()
        self._initialize_hf_search()

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    @classmethod
    def from_config(cls, config):
        query_id = config.get("query-id")
        recording_id = config.get("recording-id")

        root = config.get("root", DEFAULT_REPLAY_ROOT)
        if recording_id == "latest":
            recording_id = cls.latest_complete_recording(root, query_id)
        return cls(
            root=root,
            query_id=query_id,
            recording_id=recording_id,
            config=config,
        )

    @staticmethod
    def latest_complete_recording(root, query_id):
        recordings_dir = Path(root).resolve() / "queries" / str(query_id) / "recordings"
        candidates = []
        for manifest_path in recordings_dir.glob("*/recording_manifest.json"):
            manifest = _json_file(manifest_path)
            if manifest.get("status") != "complete":
                continue
            candidates.append((
                str(manifest.get("ended_at") or manifest.get("workflow_ended_at") or ""),
                manifest_path.parent.name,
            ))
        return max(candidates)[1]

    def _load_event_candidates(self):
        events_path = self.recording_dir / "events.jsonl"
        if not events_path.exists():
            return
        try:
            with open(events_path, "r", encoding="utf-8") as source:
                for line in source:
                    event = json.loads(line)
                    if event.get("event_type") != "huggingface.candidate":
                        continue
                    candidate = event.get("candidate") or {}
                    if candidate.get("source") == "tool_index":
                        continue
                    dataset_id = candidate.get("id")
                    if dataset_id and dataset_id not in self.hf_excluded:
                        self.hf_candidates.add(dataset_id)
        except (OSError, ValueError) as exc:
            raise LocalReplayIntegrityError(f"Cannot read {events_path}: {exc}") from exc

    def _initialize_arxiv_search(self):
        self._arxiv_db_lock = threading.Lock()
        self._arxiv_db = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            self._arxiv_db.execute(
                "CREATE VIRTUAL TABLE arxiv_abstracts USING "
                "fts5(arxiv_id UNINDEXED, abstract, tokenize=porter)"
            )
        except sqlite3.OperationalError as exc:
            raise LocalReplayIntegrityError(
                "This Python build does not provide SQLite FTS5 for local arXiv search"
            ) from exc
        rows = [
            (arxiv_id, str(document.get("abstract") or ""))
            for arxiv_id, document in sorted(self.arxiv_documents.items())
        ]
        self._arxiv_db.executemany(
            "INSERT INTO arxiv_abstracts(arxiv_id, abstract) VALUES (?, ?)", rows
        )
        self._arxiv_db.commit()

    def _initialize_hf_search(self):
        details = {}
        for record in self.records["huggingface.search"]:
            for result in record.get("results", []):
                dataset_id = result.get("id")
                if dataset_id and dataset_id in self.hf_candidates:
                    details.setdefault(dataset_id, {}).update(copy.deepcopy(result))
        for dataset_id in sorted(self.hf_candidates):
            entry = details.setdefault(dataset_id, {"id": dataset_id})
            dataset_dir = self.root / "objects" / "huggingface" / _safe_name(dataset_id)
            manifests = []
            for path in dataset_dir.glob("*/manifest.json"):
                manifest = _json_file(path)
                if manifest.get("snapshot_status") == "complete":
                    manifests.append(manifest)
            if manifests:
                manifest = min(
                    manifests,
                    key=lambda item: str(item.get("revision") or ""),
                )
                entry.setdefault("revision", manifest.get("revision"))
                materialized = manifest.get("materialized") or {}
                entry.setdefault("configs", sorted(materialized))
                entry.setdefault(
                    "splits",
                    sorted({
                        split
                        for config in materialized.values()
                        for split in config
                    }),
                )
            entry.setdefault("description", "")
            entry.setdefault("likes", 0)
            entry.setdefault("downloads", 0)
            entry.setdefault("has_test_set", "test" in entry.get("splits", []))
            entry.setdefault("has_train_set", "train" in entry.get("splits", []))
            entry.setdefault("test_download_size", None)
            entry.setdefault("test_element_size", None)
            entry.setdefault("train_download_size", None)
            entry.setdefault("train_element_size", None)
        self.hf_documents = details
        self._hf_db_lock = threading.Lock()
        self._hf_db = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            self._hf_db.execute(
                "CREATE VIRTUAL TABLE hf_datasets USING "
                "fts5(dataset_id UNINDEXED, searchable, tokenize=porter)"
            )
        except sqlite3.OperationalError as exc:
            raise LocalReplayIntegrityError(
                "This Python build does not provide SQLite FTS5 for local Hugging Face search"
            ) from exc
        rows = []
        for dataset_id, document in sorted(self.hf_documents.items()):
            searchable = " ".join([
                dataset_id.replace("/", " ").replace("-", " "),
                str(document.get("description") or ""),
                " ".join(document.get("configs") or []),
                " ".join(document.get("splits") or []),
            ])
            rows.append((dataset_id, searchable))
        self._hf_db.executemany(
            "INSERT INTO hf_datasets(dataset_id, searchable) VALUES (?, ?)", rows
        )
        self._hf_db.commit()

    def _reserve_latency(self, stream, purpose):
        records = self.records[stream]
        consumed = self._consumed[stream]
        with self._match_lock:
            self._reservation_counters[stream] += 1
            reservation_index = self._reservation_counters[stream]
            index = next(
                (item for item, used in enumerate(consumed) if not used), None
            )
            if index is not None:
                consumed[index] = True
                return records[index], "recorded", reservation_index
            latencies = [
                float(record["latency_ms"])
                for record in records
                if record.get("latency_ms") is not None
            ]
            synthetic_latency = statistics.median(latencies) if latencies else 0.0
            return {
                "call_index": None,
                "latency_ms": synthetic_latency,
                "success": True,
                "purpose": purpose,
            }, "synthetic_median", reservation_index

    def _consume_next(self, stream, purpose):
        records = self.records[stream]
        consumed = self._consumed[stream]
        with self._match_lock:
            index = next(
                (item for item, used in enumerate(consumed) if not used), None
            )
            if index is not None:
                consumed[index] = True
                return records[index]
        raise LocalReplayMismatch(
            f"No unused {stream} recording remains for {purpose}. "
            f"query_id={self.query_id}, recording_id={self.recording_id}"
        )

    def _path_from_root(self, relative_path):
        if not relative_path:
            raise LocalReplayIntegrityError("Recorded material path is missing")
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise LocalReplayIntegrityError(
                f"Recorded path escapes material root: {relative_path}"
            ) from exc
        if not candidate.is_file():
            raise LocalReplayIntegrityError(f"Recorded material file is missing: {candidate}")
        return candidate

    def _record_event(self, **event):
        trace_context = current_trace_context()
        payload = {"recorded_at": _utc_now(), **trace_context, **event}
        with self._event_lock:
            self.events.append(payload)
        record_trace_event(
            category="replay",
            name=f"replay.{event.get('tool', 'unknown')}",
            success=event.get("success", True),
            metadata=payload,
        )

    def replay_arxiv_search(
        self,
        query,
        max_results,
        processed_query=None,
        excluded_ids=None,
        return_details=False,
    ):
        started = time.perf_counter_ns()
        try:
            latency_record, latency_kind, reservation_index = self._reserve_latency(
                "arxiv.search", purpose=f"query={query!r}, max_results={max_results}"
            )
            ranked = self._search_arxiv_abstracts(
                processed_query or query,
                int(max_results),
                excluded_ids=excluded_ids,
            )
        except Exception as exc:
            self._record_event(
                tool="arxiv.search", match_kind="mismatch", success=False,
                request={"query": query, "max_results": max_results}, error=str(exc),
            )
            raise
        latency = latency_record.get("latency_ms")
        results = [item["document"] for item in ranked]
        result = self._format_arxiv_results(results)
        _sleep_to_target(started, latency, self.latency_mode, self.latency_scale)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self._record_event(
            tool="arxiv.search", match_kind="local_bm25", success=True,
            call_index=latency_record.get("call_index"), source_latency_ms=latency,
            latency_match_kind=latency_kind, reservation_index=reservation_index,
            actual_duration_ms=round(elapsed_ms, 3),
            candidate_count=len(self.arxiv_documents),
            ranked_results=[
                {"rank": index, "arxiv_id": item["document"].get("arxiv_id"),
                 "bm25_score": round(item["score"], 6)}
                for index, item in enumerate(ranked, start=1)
            ],
            request={"query": query, "processed_query": processed_query or query,
                     "max_results": max_results},
            error=None,
        )
        if return_details:
            return {
                "text": result,
                "candidate_ids": [
                    item["document"].get("arxiv_id")
                    for item in ranked
                    if item["document"].get("arxiv_id")
                ],
            }
        return result

    def _search_arxiv_abstracts(self, query, max_results, excluded_ids=None):
        if max_results <= 0:
            return []
        tokens = []
        seen = set()
        for token in re.findall(r"\w+", str(query).casefold(), flags=re.UNICODE):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        if not tokens:
            return []
        fts_query = " OR ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
        excluded_ids = {str(item) for item in (excluded_ids or [])}
        fetch_limit = min(
            len(self.arxiv_documents),
            max_results + len(excluded_ids),
        )
        with self._arxiv_db_lock:
            rows = self._arxiv_db.execute(
                "SELECT arxiv_id, bm25(arxiv_abstracts) AS score "
                "FROM arxiv_abstracts WHERE arxiv_abstracts MATCH ? "
                "ORDER BY score ASC, arxiv_id ASC LIMIT ?",
                (fts_query, fetch_limit),
            ).fetchall()
        return [
            {"document": copy.deepcopy(self.arxiv_documents[arxiv_id]),
             "score": float(score)}
            for arxiv_id, score in rows
            if arxiv_id not in excluded_ids
        ][:max_results]

    def replay_remaining_arxiv_candidates(self, max_results, excluded_ids=None):
        started = time.perf_counter_ns()
        excluded_ids = {str(item) for item in (excluded_ids or [])}
        documents = [
            copy.deepcopy(document)
            for arxiv_id, document in self.arxiv_documents.items()
            if arxiv_id not in excluded_ids
        ][:max(0, int(max_results))]
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        candidate_ids = [
            document.get("arxiv_id")
            for document in documents
            if document.get("arxiv_id")
        ]
        self._record_event(
            tool="arxiv.candidate_browse",
            match_kind="local_candidate_browse",
            success=True,
            actual_duration_ms=round(elapsed_ms, 3),
            candidate_ids=candidate_ids,
            excluded_count=len(excluded_ids),
            request={"max_results": max_results},
            error=None,
        )
        return {
            "text": self._format_arxiv_results(documents),
            "candidate_ids": candidate_ids,
        }

    @staticmethod
    def _format_arxiv_results(results):
        formatted = []
        for result in results:
            published = str(result.get("published") or "")
            publication_date = published.split("T", 1)[0].split(" ", 1)[0]
            text = f"Title: {result.get('title', '')}\n"
            text += f"Summary: {result.get('abstract', '')}\n"
            text += f"Publication Date: {publication_date}\n"
            text += f"arXiv paper ID: {result.get('arxiv_id', '')}\n"
            formatted.append(text)
        return "\n".join(formatted)

    def replay_hf_search(self, query, max_results):
        started = time.perf_counter_ns()
        try:
            record, latency_kind, reservation_index = self._reserve_latency(
                "huggingface.search", purpose=f"query={query!r}, max_results={max_results}"
            )
            results = self._search_hf_candidates(query, int(max_results))
        except Exception as exc:
            self._record_event(
                tool="huggingface.search", match_kind="mismatch", success=False,
                request={"query": query, "max_results": max_results}, error=str(exc),
            )
            raise
        latency = record.get("latency_ms")
        _sleep_to_target(started, latency, self.latency_mode, self.latency_scale)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self._record_event(
            tool="huggingface.search", match_kind="local_bm25", success=True,
            call_index=record.get("call_index"), source_latency_ms=latency,
            latency_match_kind=latency_kind, reservation_index=reservation_index,
            actual_duration_ms=round(elapsed_ms, 3),
            request={"query": query, "max_results": max_results}, error=None,
        )
        return results

    def _search_hf_candidates(self, query, max_results):
        if max_results <= 0 or not self.hf_documents:
            return []
        tokens = []
        seen = set()
        for token in re.findall(r"\w+", str(query).casefold(), flags=re.UNICODE):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
        rows = []
        if tokens:
            fts_query = " OR ".join(
                f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
            )
            with self._hf_db_lock:
                rows = self._hf_db.execute(
                    "SELECT dataset_id, bm25(hf_datasets) AS score "
                    "FROM hf_datasets WHERE hf_datasets MATCH ? "
                    "ORDER BY score ASC, dataset_id ASC LIMIT ?",
                    (fts_query, max_results),
                ).fetchall()
        dataset_ids = [row[0] for row in rows]
        if not dataset_ids:
            dataset_ids = sorted(self.hf_documents)[:max_results]
        return [copy.deepcopy(self.hf_documents[item]) for item in dataset_ids]

    def _matching_arxiv_candidate(self, requested_id):
        requested_id = str(requested_id).strip()
        if requested_id in self.arxiv_candidates:
            return requested_id
        requested_base = _arxiv_base_id(requested_id)
        matches = sorted(
            candidate
            for candidate in self.arxiv_candidates
            if _arxiv_base_id(candidate) == requested_base
        )
        return matches[0] if len(matches) == 1 else None

    def replay_arxiv_full_text(self, arxiv_id, max_len=50000):
        started = time.perf_counter_ns()
        requested_id = str(arxiv_id).strip()
        requested_base = _arxiv_base_id(requested_id)
        if any(
            excluded == requested_id or _arxiv_base_id(excluded) == requested_base
            for excluded in self.arxiv_excluded
        ):
            raise LocalReplayMismatch(
                f"arXiv paper {requested_id!r} was excluded from recording "
                f"{self.recording_id!r}"
            )
        candidate_id = self._matching_arxiv_candidate(requested_id)
        if candidate_id is None:
            error = (
                f"arXiv paper {requested_id!r} does not belong to query "
                f"{self.query_id!r}"
            )
            self._record_event(
                tool="arxiv.full_text",
                match_kind="mismatch",
                success=False,
                request={"arxiv_id": requested_id, "max_len": max_len},
                error=error,
            )
            raise LocalReplayMismatch(error)
        return self._read_arxiv_object(
            candidate_id,
            requested_id,
            max_len,
            started,
        )

    def _read_arxiv_object(self, candidate_id, requested_id, max_len, started):
        metadata_path = (
            self.root / "objects" / "arxiv" / _safe_name(candidate_id) / "metadata.json"
        )
        metadata = _json_file(metadata_path)
        if metadata.get("snapshot_status") != "complete":
            raise LocalReplayIntegrityError(f"arXiv candidate is incomplete: {candidate_id}")
        if metadata.get("extraction_status") not in {
            "complete", "partial", "abstract_fallback"
        }:
            raise LocalReplayIntegrityError(
                f"arXiv candidate text extraction is incomplete: {candidate_id}"
            )
        text_path = self._path_from_root(metadata.get("text_path"))
        text = text_path.read_text(encoding="utf-8")[: int(max_len)]
        latency = float(metadata.get("download_latency_ms") or 0.0) + float(
            metadata.get("extraction_latency_ms") or 0.0
        )
        _sleep_to_target(started, latency, self.latency_mode, self.latency_scale)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self._record_event(
            tool="arxiv.full_text", match_kind="object_direct", success=True,
            candidate_id=candidate_id, source_latency_ms=round(latency, 3),
            actual_duration_ms=round(elapsed_ms, 3),
            request={"arxiv_id": requested_id, "max_len": max_len}, error=None,
        )
        return text

    def worker_context(self):
        return {
            "root": str(self.root),
            "query_id": self.query_id,
            "recording_id": self.recording_id,
            "latency_mode": self.latency_mode,
            "latency_scale": self.latency_scale,
            "hf_candidates": sorted(self.hf_candidates),
        }

    def select_hf_dataset(
        self,
        dataset_id,
        config_name,
        split,
        revision=None,
    ):
        if dataset_id in self.hf_excluded:
            raise LocalReplayMismatch(
                f"Dataset {dataset_id!r} was excluded from query {self.query_id!r}"
            )
        if dataset_id not in self.hf_candidates:
            raise LocalReplayMismatch(
                f"Dataset {dataset_id!r} does not belong to query {self.query_id!r}"
            )
        manifest = _find_hf_manifest(self.root, dataset_id, revision)
        materialized = manifest.get("materialized") or {}
        if config_name == "default" and "default" not in materialized and len(materialized) == 1:
            config_name = next(iter(materialized))
        config_key = _safe_name(config_name)
        if config_key not in materialized:
            raise LocalReplayMismatch(
                f"Config {config_name!r} is not materialized for {dataset_id!r}"
            )
        if split not in materialized[config_key]:
            raise LocalReplayMismatch(
                f"Split {split!r} is not materialized for {dataset_id!r}/{config_name!r}"
            )
        import pyarrow.parquet as parquet

        def resource_metadata(resource_config, resource_split, split_manifest):
            parquet_path = (self.root / split_manifest["path"]).resolve()
            try:
                parquet_path.relative_to(self.root)
            except ValueError as exc:
                raise LocalReplayIntegrityError(
                    f"Parquet path escapes material root: {parquet_path}"
                ) from exc
            if not parquet_path.is_file():
                raise LocalReplayIntegrityError(
                    f"Missing local Parquet: {parquet_path}"
                )
            parquet_file = parquet.ParquetFile(str(parquet_path))
            arrow_schema = parquet_file.schema_arrow
            return {
                "config_name": resource_config,
                "split": resource_split,
                "dataset_schema": {
                    "row_count": int(parquet_file.metadata.num_rows),
                    "columns": {
                        field.name: str(field.type)
                        for field in arrow_schema
                    },
                },
            }

        config_names = {
            _safe_name(name): str(name)
            for name in (manifest.get("configs") or [])
        }
        materialized_resources = [
            resource_metadata(
                config_names.get(resource_config, resource_config),
                resource_split,
                split_manifest,
            )
            for resource_config, splits in sorted(materialized.items())
            for resource_split, split_manifest in sorted(splits.items())
        ]
        primary = next(
            item
            for item in materialized_resources
            if _safe_name(item["config_name"]) == config_key
            and item["split"] == split
        )
        schema_keys = {}
        materialized_schemas = {}
        compact_resources = []
        for resource in materialized_resources:
            columns = resource["dataset_schema"]["columns"]
            signature = json.dumps(columns, ensure_ascii=True, sort_keys=True)
            schema_key = schema_keys.get(signature)
            if schema_key is None:
                schema_key = f"schema_{len(schema_keys) + 1:03d}"
                schema_keys[signature] = schema_key
                materialized_schemas[schema_key] = {"columns": columns}
            compact_resources.append({
                "config_name": resource["config_name"],
                "split": resource["split"],
                "row_count": resource["dataset_schema"]["row_count"],
                "schema_key": schema_key,
            })
        return {
            "dataset_id": dataset_id,
            "config_name": config_name,
            "split": split,
            "revision": manifest.get("revision"),
            "available_configs": sorted(materialized),
            "available_splits": sorted(materialized[config_key]),
            "dataset_schema": primary["dataset_schema"],
            "materialized_resources": compact_resources,
            "materialized_schemas": materialized_schemas,
        }

    def record_worker_events(self, events):
        trace_context = current_trace_context()
        enriched = [
            {**trace_context, **event}
            for event in (events or [])
        ]
        with self._event_lock:
            self.events.extend(enriched)
        for event in enriched:
            record_trace_event(
                category="replay",
                name=f"replay.{event.get('tool', 'unknown')}",
                success=event.get("success", True),
                metadata=event,
            )

    def write_trace(self, trace_dir):
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        with self._event_lock:
            events = list(self.events)
        trace_path = trace_dir / "local_replay_trace.jsonl"
        with open(trace_path, "w", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")
        matches = Counter(event.get("match_kind", "unknown") for event in events)
        summary = {
            "query_id": self.query_id,
            "recording_id": self.recording_id,
            "material_root": str(self.root),
            "event_count": len(events),
            "match_counts": dict(matches),
            "strict_offline": True,
            "latency_mode": self.latency_mode,
            "latency_scale": self.latency_scale,
            "unused_calls": self.unused_calls(),
        }
        with open(trace_dir / "local_replay_summary.json", "w", encoding="utf-8") as output:
            json.dump(summary, output, ensure_ascii=True, indent=2)
        return summary

    def unused_calls(self):
        return {
            stream: sum(not value for value in values)
            for stream, values in self._consumed.items()
        }


def _find_hf_manifest(root, dataset_id, revision=None):
    dataset_dir = Path(root) / "objects" / "huggingface" / _safe_name(dataset_id)
    manifests = []
    if dataset_dir.exists():
        for path in dataset_dir.glob("*/manifest.json"):
            manifest = _json_file(path)
            if manifest.get("dataset_id") != dataset_id:
                continue
            if manifest.get("snapshot_status") != "complete":
                continue
            if revision and manifest.get("revision") != revision:
                continue
            manifest["_manifest_path"] = str(path)
            manifests.append(manifest)
    if not manifests:
        raise LocalReplayIntegrityError(
            f"No complete local Hugging Face object for {dataset_id}@{revision or '*'}"
        )
    manifests.sort(key=lambda item: (
        str(item.get("revision") or ""),
        str(item.get("_manifest_path") or ""),
    ))
    return manifests[0]


class WorkerDatasetReplay:
    LOCAL_BUILDERS = {"csv", "json", "parquet", "text", "imagefolder", "audiofolder"}

    def __init__(self, context, original_load_dataset):
        self.context = context or {}
        self.original_load_dataset = original_load_dataset
        self.root = Path(self.context.get("root") or DEFAULT_REPLAY_ROOT).resolve()
        self.events = []

    def _event(self, **event):
        self.events.append({"recorded_at": _utc_now(), **event})

    @staticmethod
    def _config_name(args, kwargs):
        if kwargs.get("name") is not None:
            return kwargs.get("name")
        if args and isinstance(args[0], str):
            return args[0]
        return None

    @staticmethod
    def _local_data_files(value):
        if value is None:
            return False
        if isinstance(value, dict):
            return all(WorkerDatasetReplay._local_data_files(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(WorkerDatasetReplay._local_data_files(item) for item in value)
        text = str(value)
        if text.startswith(("http://", "https://", "hf://")):
            return False
        return Path(text).exists()

    def load_dataset(self, path, *args, **kwargs):
        dataset_id = str(path)
        if dataset_id in self.LOCAL_BUILDERS and self._local_data_files(
            kwargs.get("data_files")
        ):
            return self.original_load_dataset(path, *args, **kwargs)
        if Path(dataset_id).exists():
            return self.original_load_dataset(path, *args, **kwargs)

        started = time.perf_counter_ns()
        config_name = self._config_name(args, kwargs)
        split = kwargs.get("split")
        revision = kwargs.get("revision")
        allowed = dataset_id in set(self.context.get("hf_candidates") or [])
        if not allowed:
            error = (
                f"Hugging Face dataset {dataset_id!r} does not belong to query "
                f"{self.context.get('query_id')!r}"
            )
            self._event(
                tool="huggingface.load_dataset", match_kind="mismatch", success=False,
                request={"dataset_id": dataset_id, "config_name": config_name,
                         "split": split, "revision": revision}, error=error,
            )
            raise LocalReplayMismatch(error)

        manifest = _find_hf_manifest(self.root, dataset_id, revision)
        loaded, resolved_config, resolved_splits = self._load_materialized(
            manifest,
            config_name,
            split,
            kwargs,
        )
        latency = manifest.get("materialization_latency_ms")
        _sleep_to_target(
            started,
            latency,
            self.context.get("latency_mode", "recorded"),
            self.context.get("latency_scale", 1.0),
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        self._event(
            tool="huggingface.load_dataset", match_kind="object_direct", success=True,
            source_latency_ms=latency, actual_duration_ms=round(elapsed_ms, 3),
            object_revision=manifest.get("revision"),
            request={"dataset_id": dataset_id, "config_name": config_name,
                     "split": split, "revision": revision}, error=None,
            resolved_resources=[
                {"config_name": resolved_config, "split": split_name}
                for split_name in resolved_splits
            ],
        )
        return loaded

    def _load_materialized(self, manifest, config_name, split, kwargs):
        if kwargs.get("streaming"):
            raise LocalReplayError("streaming=True is not supported by local material replay")
        materialized = manifest.get("materialized") or {}
        if config_name is None:
            if "default" in materialized:
                config_key = "default"
            elif len(materialized) == 1:
                config_key = next(iter(materialized))
            else:
                raise LocalReplayMismatch(
                    f"Dataset {manifest.get('dataset_id')} requires a config name"
                )
        else:
            config_key = _safe_name(config_name)
        split_manifests = materialized.get(config_key)
        if not split_manifests:
            raise LocalReplayMismatch(
                f"Config {config_name or config_key!r} is not materialized for "
                f"{manifest.get('dataset_id')}"
            )
        config_names = {
            _safe_name(name): str(name)
            for name in (manifest.get("configs") or [])
        }
        resolved_config = config_names.get(config_key, config_key)

        import pyarrow.parquet as parquet
        from datasets import Dataset, DatasetDict

        datasets_by_split = {}
        for split_name, split_manifest in split_manifests.items():
            parquet_path = (self.root / split_manifest["path"]).resolve()
            try:
                parquet_path.relative_to(self.root)
            except ValueError as exc:
                raise LocalReplayIntegrityError(
                    f"Parquet path escapes material root: {parquet_path}"
                ) from exc
            if not parquet_path.is_file():
                raise LocalReplayIntegrityError(f"Missing local Parquet: {parquet_path}")
            datasets_by_split[split_name] = Dataset(parquet.read_table(str(parquet_path)))
        if split is None:
            return DatasetDict(datasets_by_split), resolved_config, sorted(datasets_by_split)
        if not isinstance(split, str) or split not in datasets_by_split:
            raise LocalReplayMismatch(
                f"Split expressions are not replayable; available splits: "
                f"{sorted(datasets_by_split)}"
            )
        return datasets_by_split[split], resolved_config, [split]


def install_worker_hf_replay(context):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import datasets

    original_load_dataset = datasets.load_dataset
    replay = WorkerDatasetReplay(context, original_load_dataset)
    datasets.load_dataset = replay.load_dataset

    def unavailable_builder(*args, **kwargs):
        del args, kwargs
        raise LocalReplayError(
            "load_dataset_builder is disabled in strict local replay; use recorded search metadata"
        )

    datasets.load_dataset_builder = unavailable_builder
    return replay
