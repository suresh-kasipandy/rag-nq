"""Runtime settings for Milestone 1 index building."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings for Milestone 1."""

    dataset_name: str = "sentence-transformers/NQ-retrieval"
    dataset_split: str = "train"
    dataset_passage_id_field: str = "id"
    dataset_passage_text_field: str = "text"
    dataset_source_field: str = "title"  # row field for Passage.source on legacy flat rows
    # HF: stream one split only (avoids CastError if hub Features omit optional columns vs files).
    dataset_streaming: bool = True
    ingest_show_progress: bool = True
    max_passages: int | None = Field(default=None, ge=1)

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, ge=1)

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nq_passages"
    qdrant_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    qdrant_distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"] = "Cosine"

    output_dir: Path = Path("artifacts")
    passages_jsonl: str = "passages.jsonl"
    ingest_manifest_file: str = "ingest_manifest.json"
    dense_checkpoint_file: str = "dense_checkpoint.json"
    sparse_checkpoint_file: str = "sparse_checkpoint.json"
    sparse_pass1_file: str = "sparse_pass1.json"
    sparse_index_file: str = "bm25_index.pkl"
    sparse_manifest_file: str = "sparse_index_manifest.json"
    build_manifest_file: str = "index_manifest.json"
    # Lines of silver JSONL to accumulate before each encode+upsert (RAM vs Qdrant batching).
    dense_read_batch_lines: int = Field(default=512, ge=1)
    sparse_upsert_batch_size: int = Field(default=256, ge=1)
    sparse_workers: int = Field(default=1, ge=1)
    sparse_write_concurrency: int = Field(default=1, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_epsilon: float = Field(default=0.25, ge=0)
    hybrid_rrf_k: int = Field(default=60, ge=1)
    hybrid_dense_weight: float = Field(default=1.0, ge=0)
    hybrid_sparse_weight: float = Field(default=1.0, ge=0)

    @property
    def passages_path(self) -> Path:
        """Return path where normalized passages are persisted."""
        return self.output_dir / self.passages_jsonl

    @property
    def ingest_manifest_path(self) -> Path:
        """Return path for ingest (silver) manifest."""
        return self.output_dir / self.ingest_manifest_file

    @property
    def dense_checkpoint_path(self) -> Path:
        """Return path for dense indexing checkpoint."""
        return self.output_dir / self.dense_checkpoint_file

    @property
    def sparse_index_path(self) -> Path:
        """Return path where legacy local BM25 pickle is persisted (Milestone 1–2 only)."""
        return self.output_dir / self.sparse_index_file

    @property
    def sparse_checkpoint_path(self) -> Path:
        """Return path for sparse indexing checkpoint."""
        return self.output_dir / self.sparse_checkpoint_file

    @property
    def sparse_pass1_path(self) -> Path:
        """Return path for persisted sparse pass-1 corpus statistics."""
        return self.output_dir / self.sparse_pass1_file

    @property
    def sparse_manifest_path(self) -> Path:
        """Return path for Milestone 2.2 Qdrant sparse index manifest JSON."""
        return self.output_dir / self.sparse_manifest_file

    @property
    def manifest_path(self) -> Path:
        """Return path where build manifest is persisted."""
        return self.output_dir / self.build_manifest_file

    @classmethod
    def from_env(cls) -> Settings:
        """Load selected settings from `RAG_*` environment variables."""

        overrides: dict[str, object] = {}
        if value := os.getenv("RAG_DATASET_NAME"):
            overrides["dataset_name"] = value
        if value := os.getenv("RAG_DATASET_SPLIT"):
            overrides["dataset_split"] = value
        if (value := os.getenv("RAG_DATASET_STREAMING")) is not None:
            overrides["dataset_streaming"] = value.strip().lower() in {"1", "true", "yes", "on"}
        if (value := os.getenv("RAG_INGEST_PROGRESS")) is not None:
            overrides["ingest_show_progress"] = value.strip().lower() in {"1", "true", "yes", "on"}
        if value := os.getenv("RAG_EMBEDDING_MODEL_NAME"):
            overrides["embedding_model_name"] = value
        if value := os.getenv("RAG_QDRANT_URL"):
            overrides["qdrant_url"] = value
        if value := os.getenv("RAG_QDRANT_COLLECTION"):
            overrides["qdrant_collection"] = value
        if value := os.getenv("RAG_QDRANT_SPARSE_VECTOR_NAME"):
            overrides["qdrant_sparse_vector_name"] = value
        if value := os.getenv("RAG_SPARSE_CHECKPOINT_FILE"):
            overrides["sparse_checkpoint_file"] = value
        if value := os.getenv("RAG_SPARSE_PASS1_FILE"):
            overrides["sparse_pass1_file"] = value
        if value := os.getenv("RAG_OUTPUT_DIR"):
            overrides["output_dir"] = Path(value)
        if value := os.getenv("RAG_MAX_PASSAGES"):
            overrides["max_passages"] = int(value)
        if value := os.getenv("RAG_EMBEDDING_BATCH_SIZE"):
            overrides["embedding_batch_size"] = int(value)
        if value := os.getenv("RAG_DENSE_READ_BATCH_LINES"):
            overrides["dense_read_batch_lines"] = int(value)
        if value := os.getenv("RAG_SPARSE_UPSERT_BATCH_SIZE"):
            overrides["sparse_upsert_batch_size"] = int(value)
        if value := os.getenv("RAG_SPARSE_WORKERS"):
            overrides["sparse_workers"] = int(value)
        if value := os.getenv("RAG_SPARSE_WRITE_CONCURRENCY"):
            overrides["sparse_write_concurrency"] = int(value)
        if value := os.getenv("RAG_BM25_K1"):
            overrides["bm25_k1"] = float(value)
        if value := os.getenv("RAG_BM25_B"):
            overrides["bm25_b"] = float(value)
        if value := os.getenv("RAG_BM25_EPSILON"):
            overrides["bm25_epsilon"] = float(value)
        if value := os.getenv("RAG_HYBRID_RRF_K"):
            overrides["hybrid_rrf_k"] = int(value)
        if value := os.getenv("RAG_HYBRID_DENSE_WEIGHT"):
            overrides["hybrid_dense_weight"] = float(value)
        if value := os.getenv("RAG_HYBRID_SPARSE_WEIGHT"):
            overrides["hybrid_sparse_weight"] = float(value)
        return cls(**overrides)

    @staticmethod
    def force_ingest_from_env() -> bool:
        """Return True when ``RAG_FORCE_INGEST`` requests a silver rebuild."""

        value = os.getenv("RAG_FORCE_INGEST")
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}
