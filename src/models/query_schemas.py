"""API-oriented schemas for retrieval, reranking, and generation (Milestone 0 contracts).

`Passage` in :mod:`src.ingestion.models` is the ingestion unit. :class:`PassageHit` is the
retrieval-facing view (scores/ranks may be filled per pipeline stage in later milestones).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Answer-level provenance: which supporting passage IDs were used (v1)."""

    passage_id: str = Field(min_length=1)


class PassageHit(BaseModel):
    """A retrieved passage with optional per-retriever scores and ranks."""

    passage_id: str = Field(min_length=1)
    chunk_id: str | None = Field(default=None, min_length=1)
    text: str = Field(min_length=1)
    context_text: str | None = None
    source: str | None = None
    title: str | None = None
    document_url: str | None = None
    dense_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_score: float | None = None
    sparse_rank: int | None = Field(default=None, ge=1)
    fusion_rank: int | None = Field(default=None, ge=1)


class GroundedAnswer(BaseModel):
    """LLM output constrained to evidence; citations are passage IDs only in v1.

    ``citations`` is the canonical list. ``supporting_passage_ids`` is optional
    denormalized copy for simple clients; keep them consistent when both are set.
    """

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    supporting_passage_ids: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Inbound query for retrieval or end-to-end QA (HTTP layer in a later milestone)."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=500)


class QueryResponse(BaseModel):
    """Outbound result; optional fields for retrieval-only vs full RAG responses."""

    query: str
    retrieved_passages: list[PassageHit] | None = None
    grounded: GroundedAnswer | None = None
