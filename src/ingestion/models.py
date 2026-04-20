"""Core ingestion data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class IndexBuildManifest(BaseModel):
    """Metadata for deterministic index build artifacts."""

    dataset_name: str
    dataset_split: str
    embedding_model_name: str
    passage_count: int
    qdrant_collection: str
    sparse_index_path: str
