"""Core ingestion data models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

# Bump when schema / ingest semantics change (skip-if-fresh gate).
RAW_DATASET_SCHEMA_VERSION = "1"
SILVER_SCHEMA_VERSION = "1"
INDEX_CHUNK_SCHEMA_VERSION = "1"
CHUNK_MANIFEST_SCHEMA_VERSION = "1"
SPARSE_INDEX_MANIFEST_VERSION = "1"


def _drop_none(data: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in data.items() if v is not None}


@dataclass(slots=True)
class _DataclassModel:
    """Tiny compatibility layer for existing ``model_dump*`` call sites."""

    def model_dump(
        self,
        *,
        exclude_none: bool = False,
        mode: str | None = None,
    ) -> dict[str, object]:
        del mode
        payload = asdict(self)
        return _drop_none(payload) if exclude_none else payload

    def model_dump_json(self, *, indent: int | None = None, exclude_none: bool = False) -> str:
        return json.dumps(self.model_dump(exclude_none=exclude_none), indent=indent)

    @classmethod
    def model_validate(cls, payload: dict[str, object]) -> _DataclassModel:
        return cls(**payload)

    @classmethod
    def model_validate_json(cls, raw: str) -> _DataclassModel:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"{cls.__name__} JSON must decode to object.")
        return cls.model_validate(payload)


@dataclass(slots=True)
class Passage(_DataclassModel):
    """Dataset-native passage representation."""

    passage_id: str
    text: str
    source: str | None = None
    title: str | None = None
    question: str | None = None
    passage_type: str | None = None
    document_url: str | None = None
    long_answers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.passage_id.strip():
            raise ValueError("passage_id must be non-empty.")
        if not self.text.strip():
            raise ValueError("text must be non-empty.")


@dataclass(slots=True)
class IndexChunk(_DataclassModel):
    """Indexable retrieval unit produced from raw NQ candidates."""

    chunk_id: str
    group_id: str
    text: str
    context_text: str
    source_row_ordinal: int
    start_candidate_idx: int
    end_candidate_idx: int
    passage_types: list[str | None] = field(default_factory=list)
    title: str | None = None
    source: str | None = None
    question: str | None = None
    document_url: str | None = None
    parent_candidate_idx: int | None = None
    chunk_kind: str = "chunk"
    token_count: int = 0
    context_token_count: int = 0
    long_answers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty.")
        if not self.group_id.strip():
            raise ValueError("group_id must be non-empty.")
        if not self.text.strip():
            raise ValueError("text must be non-empty.")
        if not self.context_text.strip():
            raise ValueError("context_text must be non-empty.")
        if self.source_row_ordinal < 0:
            raise ValueError("source_row_ordinal must be >= 0.")
        if self.start_candidate_idx < 0 or self.end_candidate_idx < self.start_candidate_idx:
            raise ValueError("candidate span is invalid.")


@dataclass(slots=True)
class ChunkManifest(_DataclassModel):
    """Metadata for the chunked ``index_chunks.jsonl`` artifact."""

    schema_version: str
    chunk_schema_version: str
    dataset_name: str
    dataset_split: str
    line_count: int
    raw_schema_version: str
    raw_row_count: int
    created_at_utc: str
    max_passages: int | None = None
    min_tokens_soft: int = 60
    min_tokens_hard: int = 20
    target_tokens: int = 160
    max_tokens: int = 300
    context_text_token_cap: int = 400
    candidate_count: int = 0
    boilerplate_count: int = 0
    duplicate_child_count: int = 0
    tiny_candidate_count: int = 0
    tiny_suppressed_count: int = 0
    short_fact_count: int = 0
    context_expanded_count: int = 0
    role_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.line_count < 0:
            raise ValueError("line_count must be >= 0.")
        if self.raw_row_count < 0:
            raise ValueError("raw_row_count must be >= 0.")


@dataclass(slots=True)
class RawDatasetManifest(_DataclassModel):
    """Metadata for local raw row snapshot sourced from Hugging Face."""

    schema_version: str
    dataset_name: str
    dataset_split: str
    row_count: int
    created_at_utc: str
    max_passages: int | None = None

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError("row_count must be >= 0.")


@dataclass(slots=True)
class IngestManifest(_DataclassModel):
    """Metadata for the silver ``passages.jsonl`` artifact (Milestone 2)."""

    schema_version: str
    dataset_name: str
    dataset_split: str
    line_count: int
    created_at_utc: str
    max_passages: int | None = None
    raw_schema_version: str | None = None
    raw_row_count: int | None = None

    def __post_init__(self) -> None:
        if self.line_count < 0:
            raise ValueError("line_count must be >= 0.")
        if self.raw_row_count is not None and self.raw_row_count < 0:
            raise ValueError("raw_row_count must be >= 0.")


@dataclass(slots=True)
class IndexBuildManifest(_DataclassModel):
    """Metadata for deterministic index build artifacts."""

    dataset_name: str
    dataset_split: str
    embedding_model_name: str
    passage_count: int
    qdrant_collection: str
    sparse_index_path: str
    silver_schema_version: str | None = None
    chunk_count: int | None = None
    chunk_schema_version: str | None = None
    chunk_artifact_path: str | None = None
    dense_indexed_from: Literal["artifact", "hf"] | None = None


@dataclass(slots=True)
class SparseIndexManifest(_DataclassModel):
    """Metadata for Qdrant sparse vector index build (Milestone 2.2)."""

    schema_version: str = SPARSE_INDEX_MANIFEST_VERSION
    silver_path_resolved: str = ""
    qdrant_url: str = ""
    qdrant_collection: str = ""
    qdrant_sparse_vector_name: str = ""
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_epsilon: float = 0.25
    sparse_upsert_batch_size: int = 256
    sparse_analyzer: str = "regex_stem_stop"
    sparse_analyzer_version: str = "1"
    document_count: int = 0
    vocabulary_size: int = 0
    points_updated: int = 0
    avg_doc_len: float = 0.0
    idf_term_count: int = 0
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        if self.idf_term_count < 0:
            raise ValueError("idf_term_count must be >= 0.")
