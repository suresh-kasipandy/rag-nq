"""Milestone 2.2: Qdrant sparse vectors from streaming BM25-style weights.

Avoids materializing a full-corpus token list in RAM.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.config.settings import Settings
from src.ingestion.models import (
    SPARSE_INDEX_MANIFEST_VERSION,
    IndexChunk,
    Passage,
    SparseIndexManifest,
)
from src.observability.progress import ProgressTicker
from src.retrieval.sparse_index import (
    SPARSE_ANALYZER_VERSION,
    SparseAnalyzerName,
    _tokenize_passage_text,
)

LOGGER = logging.getLogger(__name__)

_SPARSE_UPSERT_RETRY_MAX_ATTEMPTS = 3
_SPARSE_UPSERT_RETRY_BASE_SLEEP_SECONDS = 1.0
_SPARSE_CHECKPOINT_SCHEMA_VERSION = "1"
_SPARSE_PASS1_ARTIFACT_SCHEMA_VERSION = "1"


class SparseQdrantClientLike(Protocol):
    """Qdrant operations needed for sparse vector attachment."""

    def get_collection(self, collection_name: str) -> Any: ...

    def update_vectors(self, collection_name: str, points: Any) -> None: ...


@dataclass(slots=True)
class SparseQdrantBuildResult:
    """Sparse Qdrant index build metadata."""

    document_count: int
    vocabulary_size: int
    points_updated: int


@dataclass(slots=True)
class _Pass1State:
    """Corpus statistics after a single streaming pass over silver."""

    document_count: int
    total_tokens: int
    term_to_id: dict[str, int]
    nd: dict[str, int]


@dataclass(slots=True, frozen=True)
class SparsePass1Data:
    """Persisted sparse pass-1 data required by query-time sparse encoding."""

    document_count: int
    total_tokens: int
    term_to_id: dict[str, int]
    silver_path_resolved: str
    max_passages: int | None
    sparse_analyzer: SparseAnalyzerName = "regex_stem_stop"
    sparse_analyzer_version: str = SPARSE_ANALYZER_VERSION


@dataclass(slots=True)
class _SparseCheckpoint:
    indexed_count: int


@dataclass(slots=True)
class _PendingWrite:
    future: Future[None]
    non_empty_count: int
    point_count: int


def _record_from_json(line: str) -> IndexChunk | Passage:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("JSONL line must decode to an object.")
    if "chunk_id" in payload:
        return IndexChunk.model_validate(payload)
    return Passage.model_validate(payload)


def _record_id(record: IndexChunk | Passage) -> str:
    return record.chunk_id if isinstance(record, IndexChunk) else record.passage_id


def _pass1_scan(
    jsonl_path: Path,
    *,
    max_passages: int | None,
    analyzer: SparseAnalyzerName = "regex_stem_stop",
    ticker: ProgressTicker | None = None,
) -> _Pass1State:
    """Stream silver once: document frequencies and stable term → id map (first-seen order)."""

    term_to_id: dict[str, int] = {}
    nd: dict[str, int] = {}
    doc_count = 0
    total_tokens = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if max_passages is not None and doc_count >= max_passages:
                break
            line = raw.strip()
            if not line:
                continue
            passage = _record_from_json(line)
            tokens = _tokenize_passage_text(passage.text, analyzer=analyzer)
            doc_count += 1
            total_tokens += len(tokens)
            for term in set(tokens):
                if term not in term_to_id:
                    term_to_id[term] = len(term_to_id)
                nd[term] = nd.get(term, 0) + 1
            if ticker is not None:
                ticker.tick(doc_count, total_tokens=total_tokens, vocabulary_size=len(term_to_id))

    return _Pass1State(
        document_count=doc_count,
        total_tokens=total_tokens,
        term_to_id=term_to_id,
        nd=nd,
    )


def _scan_document_frequencies(
    jsonl_path: Path,
    *,
    max_passages: int | None,
    analyzer: SparseAnalyzerName = "regex_stem_stop",
    ticker: ProgressTicker | None = None,
) -> tuple[int, int, dict[str, int]]:
    """Stream silver once and compute doc_count/total_tokens/nd without rebuilding term ids."""

    nd: dict[str, int] = {}
    doc_count = 0
    total_tokens = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if max_passages is not None and doc_count >= max_passages:
                break
            line = raw.strip()
            if not line:
                continue
            passage = _record_from_json(line)
            tokens = _tokenize_passage_text(passage.text, analyzer=analyzer)
            doc_count += 1
            total_tokens += len(tokens)
            for term in set(tokens):
                nd[term] = nd.get(term, 0) + 1
            if ticker is not None:
                ticker.tick(doc_count, total_tokens=total_tokens, vocabulary_size=len(nd))
    return doc_count, total_tokens, nd


def compute_okapi_idf(
    nd: dict[str, int],
    corpus_size: int,
    *,
    epsilon: float,
) -> dict[str, float]:
    """BM25Okapi-style idf with epsilon floor on negative idfs (matches ``rank_bm25``)."""

    idf: dict[str, float] = {}
    idf_sum = 0.0
    negative: list[str] = []
    for word, freq in nd.items():
        idf_val = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
        idf[word] = idf_val
        idf_sum += idf_val
        if idf_val < 0:
            negative.append(word)
    average_idf = idf_sum / len(idf) if idf else 0.0
    eps = epsilon * average_idf
    for word in negative:
        idf[word] = eps
    return idf


def _doc_term_weights_okapi(
    *,
    term_freqs: Counter[str],
    doc_len: int,
    avgdl: float,
    idf: dict[str, float],
    k1: float,
    b: float,
) -> dict[str, float]:
    """Per-term BM25Okapi contribution (matches one inner term of ``get_scores``)."""

    out: dict[str, float] = {}
    denom_len = k1 * (1.0 - b + b * (doc_len / avgdl)) if avgdl > 0 else k1 * (1.0 - b)
    for term, tf in term_freqs.items():
        idf_t = idf.get(term) or 0.0
        wt = idf_t * (tf * (k1 + 1.0)) / (tf + denom_len)
        if wt != 0.0:
            out[term] = wt
    return out


def _sparse_vector_for_passage(
    text: str,
    *,
    term_to_id: dict[str, int],
    idf: dict[str, float],
    corpus_size: int,
    total_tokens: int,
    k1: float,
    b: float,
    analyzer: SparseAnalyzerName = "regex_stem_stop",
) -> tuple[list[int], list[float]]:
    """Return Qdrant sparse indices (term ids) and values (BM25 weights) for one passage."""

    tokens = _tokenize_passage_text(text, analyzer=analyzer)
    doc_len = len(tokens)
    if doc_len == 0 or corpus_size == 0:
        return [], []
    avgdl = total_tokens / corpus_size
    weights = _doc_term_weights_okapi(
        term_freqs=Counter(tokens),
        doc_len=doc_len,
        avgdl=avgdl,
        idf=idf,
        k1=k1,
        b=b,
    )
    pairs = sorted(
        ((term_to_id[t], w) for t, w in weights.items() if t in term_to_id),
        key=lambda x: x[0],
    )
    indices = [i for i, _ in pairs]
    values = [float(w) for _, w in pairs]
    return indices, values


def encode_query_sparse_vector(
    query: str,
    *,
    term_to_id: dict[str, int],
    analyzer: SparseAnalyzerName = "regex_stem_stop",
) -> tuple[list[int], list[float]]:
    """Query sparse vector: index = term id, value = raw term frequency in the query string."""

    tokens = _tokenize_passage_text(query, analyzer=analyzer)
    counts = Counter(tokens)
    pairs: list[tuple[int, float]] = []
    for term, c in counts.items():
        tid = term_to_id.get(term)
        if tid is not None and c:
            pairs.append((tid, float(c)))
    pairs.sort(key=lambda x: x[0])
    return [i for i, _ in pairs], [w for _, w in pairs]


def load_sparse_pass1_artifact(
    *,
    path: Path,
    silver_path: Path | None = None,
    max_passages: int | None = None,
    sparse_analyzer: SparseAnalyzerName = "regex_stem_stop",
) -> SparsePass1Data:
    """Load persisted sparse pass-1 stats used by query-time sparse encoding."""

    if not path.is_file():
        raise RuntimeError(
            f"Sparse pass-1 artifact missing at {path}. "
            "Run `python -m src.scripts.index_sparse` first."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Sparse pass-1 artifact at {path} is unreadable or invalid JSON."
        ) from exc

    if payload.get("schema_version") != _SPARSE_PASS1_ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Sparse pass-1 artifact schema mismatch in {path} "
            f"(expected v{_SPARSE_PASS1_ARTIFACT_SCHEMA_VERSION})."
        )
    if silver_path is not None and payload.get("silver_path") != str(silver_path.resolve()):
        raise RuntimeError(
            "Sparse pass-1 artifact does not match the configured silver path. "
            "Rebuild sparse index to refresh vocabulary mapping."
        )
    if payload.get("max_passages") != max_passages:
        raise RuntimeError(
            "Sparse pass-1 artifact max_passages does not match current settings. "
            "Rebuild sparse index (or align RAG_MAX_PASSAGES)."
        )
    if payload.get("sparse_analyzer", "whitespace") != sparse_analyzer:
        raise RuntimeError(
            "Sparse pass-1 artifact analyzer does not match current settings. "
            "Rebuild sparse index to refresh vocabulary mapping."
        )
    if payload.get("sparse_analyzer_version", SPARSE_ANALYZER_VERSION) != SPARSE_ANALYZER_VERSION:
        raise RuntimeError(
            "Sparse pass-1 artifact analyzer version does not match current code. "
            "Rebuild sparse index to refresh vocabulary mapping."
        )

    document_count = payload.get("document_count")
    total_tokens = payload.get("total_tokens")
    term_to_id_raw = payload.get("term_to_id")
    if not isinstance(document_count, int) or document_count < 0:
        raise RuntimeError(f"Invalid `document_count` in sparse pass-1 artifact {path}.")
    if not isinstance(total_tokens, int) or total_tokens < 0:
        raise RuntimeError(f"Invalid `total_tokens` in sparse pass-1 artifact {path}.")
    if not isinstance(term_to_id_raw, dict):
        raise RuntimeError(f"Invalid `term_to_id` in sparse pass-1 artifact {path}.")

    term_to_id: dict[str, int] = {}
    for term, index in term_to_id_raw.items():
        if not isinstance(term, str) or not isinstance(index, int) or index < 0:
            raise RuntimeError(f"Invalid term mapping in sparse pass-1 artifact {path}.")
        term_to_id[term] = index

    return SparsePass1Data(
        document_count=document_count,
        total_tokens=total_tokens,
        term_to_id=term_to_id,
        silver_path_resolved=payload.get("silver_path", ""),
        max_passages=payload.get("max_passages"),
        sparse_analyzer=payload.get("sparse_analyzer", sparse_analyzer),
        sparse_analyzer_version=payload.get("sparse_analyzer_version", SPARSE_ANALYZER_VERSION),
    )


def _collection_has_sparse_vector(client: Any, collection_name: str, sparse_name: str) -> bool:
    info = client.get_collection(collection_name)
    params = info.config.params
    sparse_cfg = getattr(params, "sparse_vectors", None)
    if not sparse_cfg:
        return False
    return sparse_name in sparse_cfg


def _update_vectors_with_retry(
    *,
    client: SparseQdrantClientLike,
    collection_name: str,
    points: list[Any],
) -> None:
    delay = _SPARSE_UPSERT_RETRY_BASE_SLEEP_SECONDS
    for attempt in range(1, _SPARSE_UPSERT_RETRY_MAX_ATTEMPTS + 1):
        try:
            client.update_vectors(collection_name=collection_name, points=points)
            return
        except Exception:
            if attempt >= _SPARSE_UPSERT_RETRY_MAX_ATTEMPTS:
                raise
            LOGGER.warning(
                "sparse update_vectors failed (attempt %s/%s); retrying in %.1fs",
                attempt,
                _SPARSE_UPSERT_RETRY_MAX_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            delay *= 2.0


def _build_default_qdrant_client(url: str) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "qdrant-client is required for sparse Qdrant indexing. "
            "Install project dependencies first."
        ) from exc

    try:
        return QdrantClient(url=url, check_compatibility=False)
    except TypeError:
        return QdrantClient(url=url)


def _load_sparse_checkpoint(
    *,
    path: Path,
    silver_path: Path,
    collection_name: str,
    sparse_vector_name: str,
    max_passages: int | None,
    bm25_k1: float,
    bm25_b: float,
    bm25_epsilon: float,
    sparse_upsert_batch_size: int,
    sparse_analyzer: SparseAnalyzerName,
) -> _SparseCheckpoint:
    if not path.is_file():
        return _SparseCheckpoint(indexed_count=0)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return _SparseCheckpoint(indexed_count=0)

    expected = {
        "schema_version": _SPARSE_CHECKPOINT_SCHEMA_VERSION,
        "silver_path": str(silver_path.resolve()),
        "collection_name": collection_name,
        "sparse_vector_name": sparse_vector_name,
        "max_passages": max_passages,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_epsilon": bm25_epsilon,
        "sparse_upsert_batch_size": sparse_upsert_batch_size,
        "sparse_analyzer": sparse_analyzer,
        "sparse_analyzer_version": SPARSE_ANALYZER_VERSION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return _SparseCheckpoint(indexed_count=0)
    indexed_count = payload.get("indexed_count")
    if not isinstance(indexed_count, int) or indexed_count < 0:
        return _SparseCheckpoint(indexed_count=0)
    return _SparseCheckpoint(indexed_count=indexed_count)


def _write_sparse_checkpoint(
    *,
    path: Path,
    silver_path: Path,
    collection_name: str,
    sparse_vector_name: str,
    max_passages: int | None,
    bm25_k1: float,
    bm25_b: float,
    bm25_epsilon: float,
    sparse_upsert_batch_size: int,
    sparse_analyzer: SparseAnalyzerName,
    indexed_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": _SPARSE_CHECKPOINT_SCHEMA_VERSION,
        "silver_path": str(silver_path.resolve()),
        "collection_name": collection_name,
        "sparse_vector_name": sparse_vector_name,
        "max_passages": max_passages,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "bm25_epsilon": bm25_epsilon,
        "sparse_upsert_batch_size": sparse_upsert_batch_size,
        "sparse_analyzer": sparse_analyzer,
        "sparse_analyzer_version": SPARSE_ANALYZER_VERSION,
        "indexed_count": indexed_count,
        "updated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _remove_sparse_checkpoint(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _write_sparse_pass1_artifact(
    *,
    path: Path,
    silver_path: Path,
    max_passages: int | None,
    sparse_analyzer: SparseAnalyzerName,
    pass1: _Pass1State,
) -> None:
    """Persist sparse pass-1 vocabulary/stats for query-time sparse vector encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": _SPARSE_PASS1_ARTIFACT_SCHEMA_VERSION,
        "silver_path": str(silver_path.resolve()),
        "max_passages": max_passages,
        "document_count": pass1.document_count,
        "total_tokens": pass1.total_tokens,
        "term_to_id": pass1.term_to_id,
        "sparse_analyzer": sparse_analyzer,
        "sparse_analyzer_version": SPARSE_ANALYZER_VERSION,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class SparseQdrantIndexer:
    """Attach BM25-style sparse vectors to dense points."""

    def __init__(self, settings: Settings, client: SparseQdrantClientLike | None = None) -> None:
        self._settings = settings
        self._client = client or _build_default_qdrant_client(settings.qdrant_url)

    def assert_collection_sparse_ready(self) -> None:
        """Fail fast when the collection is missing or has no configured sparse vector name."""

        name = self._settings.qdrant_collection
        sparse_name = self._settings.qdrant_sparse_vector_name
        try:
            ok = _collection_has_sparse_vector(self._client, name, sparse_name)
        except Exception as exc:  # noqa: BLE001 — surface as configuration error
            raise RuntimeError(
                f"Cannot read Qdrant collection {name!r} at {self._settings.qdrant_url!r}. "
                "Start Qdrant (see README) and run dense indexing before sparse."
            ) from exc
        if not ok:
            raise RuntimeError(
                f"Collection {name!r} has no sparse vector {sparse_name!r}. "
                "Recreate the collection (drop + run index_dense) so sparse_vectors_config "
                "is present, or use a collection created with Milestone 2.2 dense indexing."
            )

    def build_from_jsonl(
        self,
        jsonl_path: Path,
        *,
        max_passages: int | None = None,
    ) -> SparseQdrantBuildResult:
        """Two-pass streaming sparse attach: global stats, then batched ``update_vectors``."""

        self.assert_collection_sparse_ready()
        p1: _Pass1State
        try:
            pass1_data = load_sparse_pass1_artifact(
                path=self._settings.sparse_pass1_path,
                silver_path=jsonl_path,
                max_passages=max_passages,
                sparse_analyzer=self._settings.sparse_analyzer,
            )
        except RuntimeError:
            pass1_ticker = ProgressTicker(
                logger=LOGGER,
                stage="sparse_pass1",
                label="records",
                total=max_passages,
                every_items=self._settings.progress_log_every_records,
                every_seconds=self._settings.progress_log_every_seconds,
            )
            pass1_ticker.start(
                input=jsonl_path,
                analyzer=self._settings.sparse_analyzer,
                analyzer_version=SPARSE_ANALYZER_VERSION,
            )
            p1 = _pass1_scan(
                jsonl_path,
                max_passages=max_passages,
                analyzer=self._settings.sparse_analyzer,
                ticker=pass1_ticker,
            )
            pass1_ticker.finish(
                p1.document_count,
                total_tokens=p1.total_tokens,
                vocabulary_size=len(p1.term_to_id),
            )
            _write_sparse_pass1_artifact(
                path=self._settings.sparse_pass1_path,
                silver_path=jsonl_path,
                max_passages=max_passages,
                sparse_analyzer=self._settings.sparse_analyzer,
                pass1=p1,
            )
        else:
            LOGGER.info(
                "reusing sparse pass1 artifact at %s",
                self._settings.sparse_pass1_path,
            )
            pass1_ticker = ProgressTicker(
                logger=LOGGER,
                stage="sparse_pass1",
                label="records",
                total=pass1_data.document_count,
                every_items=self._settings.progress_log_every_records,
                every_seconds=self._settings.progress_log_every_seconds,
            )
            pass1_ticker.start(
                input=jsonl_path,
                analyzer=self._settings.sparse_analyzer,
                analyzer_version=SPARSE_ANALYZER_VERSION,
                reused_artifact=self._settings.sparse_pass1_path,
            )
            doc_count, total_tokens, nd = _scan_document_frequencies(
                jsonl_path,
                max_passages=max_passages,
                analyzer=self._settings.sparse_analyzer,
                ticker=pass1_ticker,
            )
            pass1_ticker.finish(
                doc_count,
                total_tokens=total_tokens,
                vocabulary_size=len(pass1_data.term_to_id),
                reused_artifact=self._settings.sparse_pass1_path,
            )
            if doc_count != pass1_data.document_count or total_tokens != pass1_data.total_tokens:
                raise RuntimeError(
                    "Sparse pass-1 artifact does not match current silver stats. "
                    "Delete sparse artifacts/checkpoint and rebuild sparse indexing."
                )
            unknown_terms = [term for term in nd if term not in pass1_data.term_to_id]
            if unknown_terms:
                raise RuntimeError(
                    "Sparse pass-1 artifact is missing terms from current silver data. "
                    "Delete sparse artifacts/checkpoint and rebuild sparse indexing."
                )
            p1 = _Pass1State(
                document_count=pass1_data.document_count,
                total_tokens=pass1_data.total_tokens,
                term_to_id=pass1_data.term_to_id,
                nd=nd,
            )
        if p1.document_count == 0:
            result = SparseQdrantBuildResult(
                document_count=0,
                vocabulary_size=0,
                points_updated=0,
            )
            self.write_manifest(jsonl_path=jsonl_path, pass1=p1, idf={}, result=result)
            return result

        idf = compute_okapi_idf(
            p1.nd,
            p1.document_count,
            epsilon=self._settings.bm25_epsilon,
        )
        checkpoint_path = self._settings.sparse_checkpoint_path
        checkpoint = _load_sparse_checkpoint(
            path=checkpoint_path,
            silver_path=jsonl_path,
            collection_name=self._settings.qdrant_collection,
            sparse_vector_name=self._settings.qdrant_sparse_vector_name,
            max_passages=max_passages,
            bm25_k1=self._settings.bm25_k1,
            bm25_b=self._settings.bm25_b,
            bm25_epsilon=self._settings.bm25_epsilon,
            sparse_upsert_batch_size=self._settings.sparse_upsert_batch_size,
            sparse_analyzer=self._settings.sparse_analyzer,
        )
        if checkpoint.indexed_count:
            LOGGER.info(
                "resuming sparse from checkpoint at %s non-empty passages",
                checkpoint.indexed_count,
            )

        from qdrant_client.http import models as qm

        sparse_name = self._settings.qdrant_sparse_vector_name
        points_updated = 0
        completed_non_empty = checkpoint.indexed_count
        seen_non_empty = 0
        empty_sparse_vectors = 0
        batches_completed = 0
        total_records = p1.document_count
        remaining_records = max(total_records - checkpoint.indexed_count, 0)
        total_batches = (
            math.ceil(remaining_records / self._settings.sparse_upsert_batch_size)
            if remaining_records
            else 0
        )
        pass2_ticker = ProgressTicker(
            logger=LOGGER,
            stage="sparse_pass2",
            label="batches",
            total=total_batches,
            every_items=self._settings.progress_log_every_batches,
            every_seconds=self._settings.progress_log_every_seconds,
        )
        pass2_ticker.start(
            input=jsonl_path,
            collection=self._settings.qdrant_collection,
            sparse_vector_name=sparse_name,
            batch_size=self._settings.sparse_upsert_batch_size,
            workers=self._settings.sparse_workers,
            write_concurrency=self._settings.sparse_write_concurrency,
            resume_count=checkpoint.indexed_count,
            records_total=total_records,
        )
        pending_writes: list[_PendingWrite] = []

        def _compute_point(passage: IndexChunk | Passage) -> Any | None:
            indices, values = _sparse_vector_for_passage(
                passage.text,
                term_to_id=p1.term_to_id,
                idf=idf,
                corpus_size=p1.document_count,
                total_tokens=p1.total_tokens,
                k1=self._settings.bm25_k1,
                b=self._settings.bm25_b,
                analyzer=self._settings.sparse_analyzer,
            )
            if not indices:
                return None
            return qm.PointVectors(
                id=_record_id(passage),
                vector={sparse_name: qm.SparseVector(indices=indices, values=values)},
            )

        def _wait_one_pending() -> None:
            nonlocal batches_completed, completed_non_empty, points_updated
            pending = pending_writes.pop(0)
            pending.future.result()
            completed_non_empty += pending.non_empty_count
            points_updated += pending.point_count
            batches_completed += 1
            _write_sparse_checkpoint(
                path=checkpoint_path,
                silver_path=jsonl_path,
                collection_name=self._settings.qdrant_collection,
                sparse_vector_name=self._settings.qdrant_sparse_vector_name,
                max_passages=max_passages,
                bm25_k1=self._settings.bm25_k1,
                bm25_b=self._settings.bm25_b,
                bm25_epsilon=self._settings.bm25_epsilon,
                sparse_upsert_batch_size=self._settings.sparse_upsert_batch_size,
                sparse_analyzer=self._settings.sparse_analyzer,
                indexed_count=completed_non_empty,
            )
            pass2_ticker.tick(
                batches_completed,
                records_indexed=completed_non_empty,
                records_total=total_records,
                points_updated=points_updated,
                empty_sparse_vectors=empty_sparse_vectors,
            )

        def _submit_batch(
            passages: list[IndexChunk | Passage],
            *,
            compute_executor: ThreadPoolExecutor,
            writer_executor: ThreadPoolExecutor,
        ) -> None:
            nonlocal empty_sparse_vectors
            if not passages:
                return
            points = [p for p in compute_executor.map(_compute_point, passages) if p is not None]
            empty_sparse_vectors += len(passages) - len(points)

            def _writer_job() -> None:
                if not points:
                    return
                _update_vectors_with_retry(
                    client=self._client,
                    collection_name=self._settings.qdrant_collection,
                    points=points,
                )

            pending_writes.append(
                _PendingWrite(
                    future=writer_executor.submit(_writer_job),
                    non_empty_count=len(passages),
                    point_count=len(points),
                )
            )
            if len(pending_writes) >= self._settings.sparse_write_concurrency:
                _wait_one_pending()

        with (
            ThreadPoolExecutor(max_workers=self._settings.sparse_workers) as compute_executor,
            ThreadPoolExecutor(
                max_workers=self._settings.sparse_write_concurrency
            ) as writer_executor,
        ):
            passages_batch: list[IndexChunk | Passage] = []
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    if max_passages is not None and seen_non_empty >= max_passages:
                        break
                    line = raw.strip()
                    if not line:
                        continue
                    seen_non_empty += 1
                    if seen_non_empty <= checkpoint.indexed_count:
                        continue
                    passages_batch.append(_record_from_json(line))
                    if len(passages_batch) >= self._settings.sparse_upsert_batch_size:
                        _submit_batch(
                            passages_batch,
                            compute_executor=compute_executor,
                            writer_executor=writer_executor,
                        )
                        passages_batch = []
                _submit_batch(
                    passages_batch,
                    compute_executor=compute_executor,
                    writer_executor=writer_executor,
                )
            while pending_writes:
                _wait_one_pending()

        _remove_sparse_checkpoint(checkpoint_path)

        result = SparseQdrantBuildResult(
            document_count=p1.document_count,
            vocabulary_size=len(p1.term_to_id),
            points_updated=points_updated,
        )
        self.write_manifest(jsonl_path=jsonl_path, pass1=p1, idf=idf, result=result)
        pass2_ticker.finish(
            batches_completed,
            records_indexed=completed_non_empty,
            records_total=total_records,
            points_updated=points_updated,
            empty_sparse_vectors=empty_sparse_vectors,
            manifest=self._settings.sparse_manifest_path,
            pass1_artifact=self._settings.sparse_pass1_path,
        )
        return result

    def write_manifest(
        self,
        *,
        jsonl_path: Path,
        pass1: _Pass1State,
        idf: dict[str, float],
        result: SparseQdrantBuildResult,
    ) -> None:
        """Persist sparse index manifest (atomic replace)."""

        avgdl = (pass1.total_tokens / pass1.document_count) if pass1.document_count else 0.0
        manifest = SparseIndexManifest(
            schema_version=SPARSE_INDEX_MANIFEST_VERSION,
            silver_path_resolved=str(jsonl_path.resolve()),
            qdrant_url=self._settings.qdrant_url,
            qdrant_collection=self._settings.qdrant_collection,
            qdrant_sparse_vector_name=self._settings.qdrant_sparse_vector_name,
            bm25_k1=self._settings.bm25_k1,
            bm25_b=self._settings.bm25_b,
            bm25_epsilon=self._settings.bm25_epsilon,
            sparse_upsert_batch_size=self._settings.sparse_upsert_batch_size,
            sparse_analyzer=self._settings.sparse_analyzer,
            sparse_analyzer_version=SPARSE_ANALYZER_VERSION,
            document_count=result.document_count,
            vocabulary_size=result.vocabulary_size,
            points_updated=result.points_updated,
            avg_doc_len=avgdl,
            idf_term_count=len(idf),
            created_at_utc=datetime.now(UTC).replace(microsecond=0).isoformat(),
        )
        path = self._settings.sparse_manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(manifest.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
