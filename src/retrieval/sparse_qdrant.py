"""Milestone 2.2: Qdrant sparse vectors from streaming BM25-style weights.

Avoids materializing a full-corpus token list in RAM.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.config.settings import Settings
from src.ingestion.models import SPARSE_INDEX_MANIFEST_VERSION, Passage, SparseIndexManifest
from src.retrieval.sparse_index import _tokenize_passage_text

LOGGER = logging.getLogger(__name__)

_SPARSE_UPSERT_RETRY_MAX_ATTEMPTS = 3
_SPARSE_UPSERT_RETRY_BASE_SLEEP_SECONDS = 1.0


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


def _pass1_scan(jsonl_path: Path, *, max_passages: int | None) -> _Pass1State:
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
            passage = Passage.model_validate_json(line)
            tokens = _tokenize_passage_text(passage.text)
            doc_count += 1
            total_tokens += len(tokens)
            for term in set(tokens):
                if term not in term_to_id:
                    term_to_id[term] = len(term_to_id)
                nd[term] = nd.get(term, 0) + 1

    return _Pass1State(
        document_count=doc_count,
        total_tokens=total_tokens,
        term_to_id=term_to_id,
        nd=nd,
    )


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
) -> tuple[list[int], list[float]]:
    """Return Qdrant sparse indices (term ids) and values (BM25 weights) for one passage."""

    tokens = _tokenize_passage_text(text)
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
) -> tuple[list[int], list[float]]:
    """Query sparse vector: index = term id, value = raw term frequency in the query string."""

    tokens = _tokenize_passage_text(query)
    counts = Counter(tokens)
    pairs: list[tuple[int, float]] = []
    for term, c in counts.items():
        tid = term_to_id.get(term)
        if tid is not None and c:
            pairs.append((tid, float(c)))
    pairs.sort(key=lambda x: x[0])
    return [i for i, _ in pairs], [w for _, w in pairs]


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


class SparseQdrantIndexer:
    """Attach BM25-style sparse vectors to dense points (``passage_id`` is the Qdrant point id)."""

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
        p1 = _pass1_scan(jsonl_path, max_passages=max_passages)
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

        from qdrant_client.http import models as qm

        sparse_name = self._settings.qdrant_sparse_vector_name
        batch: list[Any] = []
        points_updated = 0
        doc_ix = 0

        def flush() -> None:
            nonlocal batch, points_updated
            if not batch:
                return
            _update_vectors_with_retry(
                client=self._client,
                collection_name=self._settings.qdrant_collection,
                points=batch,
            )
            points_updated += len(batch)
            batch = []

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if max_passages is not None and doc_ix >= max_passages:
                    break
                line = raw.strip()
                if not line:
                    continue
                passage = Passage.model_validate_json(line)
                indices, values = _sparse_vector_for_passage(
                    passage.text,
                    term_to_id=p1.term_to_id,
                    idf=idf,
                    corpus_size=p1.document_count,
                    total_tokens=p1.total_tokens,
                    k1=self._settings.bm25_k1,
                    b=self._settings.bm25_b,
                )
                if not indices:
                    doc_ix += 1
                    continue
                batch.append(
                    qm.PointVectors(
                        id=passage.passage_id,
                        vector={
                            sparse_name: qm.SparseVector(indices=indices, values=values),
                        },
                    )
                )
                doc_ix += 1
                if len(batch) >= self._settings.sparse_upsert_batch_size:
                    flush()
                    LOGGER.info(
                        "sparse_qdrant_index updated=%s batch_flushed",
                        points_updated,
                        extra={"stage": "sparse_qdrant_index"},
                    )
        flush()

        result = SparseQdrantBuildResult(
            document_count=p1.document_count,
            vocabulary_size=len(p1.term_to_id),
            points_updated=points_updated,
        )
        self.write_manifest(jsonl_path=jsonl_path, pass1=p1, idf=idf, result=result)
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
