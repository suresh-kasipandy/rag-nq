"""API-oriented schemas for retrieval, reranking, and generation (Milestone 0 contracts).

`Passage` in :mod:`src.ingestion.models` is the ingestion unit. :class:`PassageHit` is the
retrieval-facing view (scores/ranks may be filled per pipeline stage in later milestones).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Answer-level provenance: which retrieved point IDs were used (v1)."""

    point_id: str = Field(min_length=1)


class SupportedClaim(BaseModel):
    """A generated answer claim and the retrieved point IDs that directly support it."""

    claim: str = Field(min_length=1)
    point_ids: list[str] = Field(min_length=1)


class UnsupportedClaim(BaseModel):
    """A requested or considered claim that could not be grounded in retrieved evidence."""

    claim: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DuplicateAlias(BaseModel):
    """Collapsed duplicate hit retained for retrieval debugging/provenance."""

    point_id: str = Field(min_length=1)
    dense_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_score: float | None = None
    sparse_rank: int | None = Field(default=None, ge=1)
    fusion_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None
    rerank_rank: int | None = Field(default=None, ge=1)


class DedupeMetrics(BaseModel):
    """Per-query retrieval dedupe counters."""

    raw_count: int = Field(ge=0)
    unique_count: int = Field(ge=0)
    dedupe_drop_count: int = Field(ge=0)
    dedupe_drop_rate: float = Field(ge=0.0, le=1.0)


class RetrievalStageTimings(BaseModel):
    """Per-query retrieval latency breakdown in seconds."""

    retrieve_seconds: float = Field(ge=0.0)
    fusion_seconds: float = Field(default=0.0, ge=0.0)
    rerank_seconds: float = Field(default=0.0, ge=0.0)
    dedupe_seconds: float = Field(default=0.0, ge=0.0)
    total_seconds: float = Field(ge=0.0)


class RetrievalMetrics(BaseModel):
    """Optional debug metrics for retrieval-only responses."""

    dedupe: DedupeMetrics | None = None
    timings: RetrievalStageTimings | None = None


class PassageHit(BaseModel):
    """A retrieved passage with optional per-retriever scores and ranks."""

    point_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    context_text: str | None = None
    title: str | None = None
    document_url: str | None = None
    group_id: str | None = None
    chunk_kind: str | None = None
    source_row_ordinal: int | None = Field(default=None, ge=0)
    start_candidate_idx: int | None = Field(default=None, ge=0)
    end_candidate_idx: int | None = Field(default=None, ge=0)
    passage_types: list[str] = Field(default_factory=list)
    dense_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_score: float | None = None
    sparse_rank: int | None = Field(default=None, ge=1)
    fusion_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None
    rerank_rank: int | None = Field(default=None, ge=1)
    dedupe_rank: int | None = Field(default=None, ge=1)
    duplicate_aliases: list[DuplicateAlias] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    """LLM output constrained to retrieved evidence.

    ``citations`` is the canonical list. ``supporting_point_ids`` is an optional
    denormalized copy for simple clients; keep them consistent when both are set.
    """

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    supported_claims: list[SupportedClaim] = Field(default_factory=list)
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    abstention_reason: str | None = None
    supporting_point_ids: list[str] = Field(default_factory=list)
    supporting_evidence: list[PassageHit] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Inbound query for retrieval or end-to-end QA (HTTP layer in a later milestone)."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=500)


class QueryResponse(BaseModel):
    """Outbound result; optional fields for retrieval-only vs full RAG responses."""

    query: str
    retrieved_passages: list[PassageHit] | None = None
    retrieval_metrics: RetrievalMetrics | None = None
    grounded: GroundedAnswer | None = None
