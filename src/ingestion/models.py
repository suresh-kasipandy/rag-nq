"""Core ingestion data models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Bump when silver JSONL schema or ingest semantics change (skip-if-fresh gate).
SILVER_SCHEMA_VERSION = "1"


class Passage(BaseModel):
    """Dataset-native passage representation."""

    passage_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str | None = None
    title: str | None = None
    question: str | None = None
    passage_type: str | None = None
    document_url: str | None = None
    long_answers: list[str] = Field(default_factory=list)


class IngestManifest(BaseModel):
    """Metadata for the silver ``passages.jsonl`` artifact (Milestone 2)."""

    schema_version: str = Field(description="Silver / ingest contract version string.")
    dataset_name: str
    dataset_split: str
    max_passages: int | None = None
    line_count: int = Field(ge=0)
    created_at_utc: str = Field(description="ISO-8601 UTC timestamp when ingest finished.")


class IndexBuildManifest(BaseModel):
    """Metadata for deterministic index build artifacts."""

    dataset_name: str
    dataset_split: str
    embedding_model_name: str
    passage_count: int
    qdrant_collection: str
    sparse_index_path: str = Field(
        description=(
            "Path to sparse index manifest JSON (M2.2+); field name kept for schema stability."
        ),
    )
    silver_schema_version: str | None = None
    dense_indexed_from: Literal["artifact", "hf"] | None = Field(
        default=None,
        description="Dense source: hf (ingest from HF) or artifact (reuse silver).",
    )


SPARSE_INDEX_MANIFEST_VERSION = "1"


class SparseIndexManifest(BaseModel):
    """Metadata for Qdrant sparse vector index build (Milestone 2.2)."""

    schema_version: str = SPARSE_INDEX_MANIFEST_VERSION
    silver_path_resolved: str
    qdrant_url: str
    qdrant_collection: str
    qdrant_sparse_vector_name: str
    bm25_k1: float
    bm25_b: float
    bm25_epsilon: float
    sparse_upsert_batch_size: int
    document_count: int
    vocabulary_size: int
    points_updated: int
    avg_doc_len: float
    idf_term_count: int = Field(ge=0, description="Number of terms with computed idf (vocabulary).")
    created_at_utc: str = Field(description="ISO-8601 UTC timestamp when sparse indexing finished.")
