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
    max_passages: int | None = Field(default=None, ge=1)

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, ge=1)

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "nq_passages_dense"
    qdrant_vector_name: str = "dense"
    qdrant_distance: Literal["Cosine", "Dot", "Euclid", "Manhattan"] = "Cosine"

    output_dir: Path = Path("artifacts")
    passages_jsonl: str = "passages.jsonl"
    sparse_index_file: str = "bm25_index.pkl"
    build_manifest_file: str = "index_manifest.json"

    @property
    def passages_path(self) -> Path:
        """Return path where normalized passages are persisted."""
        return self.output_dir / self.passages_jsonl

    @property
    def sparse_index_path(self) -> Path:
        """Return path where sparse index is persisted."""
        return self.output_dir / self.sparse_index_file

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
        if value := os.getenv("RAG_EMBEDDING_MODEL_NAME"):
            overrides["embedding_model_name"] = value
        if value := os.getenv("RAG_QDRANT_URL"):
            overrides["qdrant_url"] = value
        if value := os.getenv("RAG_QDRANT_COLLECTION"):
            overrides["qdrant_collection"] = value
        if value := os.getenv("RAG_OUTPUT_DIR"):
            overrides["output_dir"] = Path(value)
        return cls(**overrides)
