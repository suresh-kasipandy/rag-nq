"""HTTP-facing schemas for the Milestone 7 API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.retrieval.qdrant_retrievers import Mode


class RetrieveRequest(BaseModel):
    """Request body for retrieval-only diagnostics."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=500)
    mode: Mode = "hybrid"


class QueryApiRequest(RetrieveRequest):
    """Request body for retrieve plus optional grounded generation."""

    generate: bool = True


class ArtifactStatus(BaseModel):
    """Safe existence metadata for local runtime artifacts."""

    path: str
    exists: bool


class HealthResponse(BaseModel):
    """Minimal health response with no secret-bearing fields."""

    status: Literal["ok"] = "ok"
    service: str = "rag-nq-showcase"


class RootResponse(BaseModel):
    """Discoverable landing payload for browser visits to the API root."""

    service: str = "rag-nq-showcase"
    docs_url: str = "/docs"
    health_url: str = "/health"
    config_url: str = "/config"
    retrieve_url: str = "/retrieve"
    query_url: str = "/query"


class RetrievalConfigMetadata(BaseModel):
    """Safe retrieval settings exposed for diagnostics."""

    retrieve_k: int
    rerank_k: int | None
    rerank_enabled: bool
    rerank_model_name: str
    rerank_context_token_budget: int
    retrieval_dedupe_enabled: bool
    hybrid_rrf_k: int
    hybrid_dense_weight: float
    hybrid_sparse_weight: float


class GenerationConfigMetadata(BaseModel):
    """Safe generation settings exposed for diagnostics."""

    generation_provider: str
    generation_model_name: str
    generation_temperature: float
    generation_max_tokens: int
    generation_timeout_seconds: float
    generation_context_token_budget: int
    generation_min_citations: int
    generation_api_configured: bool
    generation_api_key_env_configured: bool


class RuntimeConfigResponse(BaseModel):
    """Safe runtime metadata exposed by ``/config``."""

    dataset_name: str
    dataset_split: str
    embedding_model_name: str
    qdrant_url: str
    qdrant_collection: str
    qdrant_vector_name: str
    qdrant_sparse_vector_name: str
    retrieval: RetrievalConfigMetadata
    generation: GenerationConfigMetadata
    artifacts: dict[str, ArtifactStatus]
