"""FastAPI app for Milestone 7 retrieval and grounded query endpoints."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Response

from app.api.schemas import (
    ArtifactStatus,
    GenerationConfigMetadata,
    HealthResponse,
    QueryApiRequest,
    RetrievalConfigMetadata,
    RetrieveRequest,
    RootResponse,
    RuntimeConfigResponse,
)
from src.config.settings import Settings
from src.generation.grounded import GroundedGenerator
from src.models.query_schemas import (
    GroundedAnswer,
    PassageHit,
    QueryResponse,
    RetrievalMetrics,
)
from src.observability.logging_setup import setup_logging
from src.retrieval.qdrant_retrievers import Mode, QdrantModeRetriever

LOGGER = logging.getLogger(__name__)


class ApiRetriever(Protocol):
    """Retriever interface used by API endpoints and tests."""

    last_retrieval_metrics: RetrievalMetrics | None

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        """Return retrieved passage hits."""


class ApiGenerator(Protocol):
    """Grounded generator interface used by API endpoints and tests."""

    def generate(self, query: str, hits: list[PassageHit]) -> GroundedAnswer:
        """Return a grounded answer object."""


RetrieverFactory = Callable[[Settings, Mode], ApiRetriever]
GeneratorFactory = Callable[[Settings], ApiGenerator]


def create_app(
    *,
    settings: Settings | None = None,
    retriever_factory: RetrieverFactory | None = None,
    generator_factory: GeneratorFactory | None = None,
) -> FastAPI:
    """Build the HTTP app with injectable dependencies for tests."""

    setup_logging()
    app_settings = settings or Settings.from_env()
    make_retriever = retriever_factory or _default_retriever_factory
    make_generator = generator_factory or _default_generator_factory

    app = FastAPI(
        title="RAG NQ Showcase API",
        version="0.1.0",
        description="Local API for retrieval diagnostics and grounded RAG queries.",
    )

    @app.get("/", response_model=RootResponse)
    def root() -> RootResponse:
        return RootResponse()

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/config", response_model=RuntimeConfigResponse)
    def config() -> RuntimeConfigResponse:
        return safe_runtime_config(app_settings)

    @app.post("/retrieve", response_model=QueryResponse)
    def retrieve(request: RetrieveRequest) -> QueryResponse:
        started_at = time.monotonic()
        try:
            retriever = make_retriever(app_settings, request.mode)
            hits = retriever.retrieve(request.query, top_k=request.top_k)
        except Exception as exc:
            LOGGER.exception("retrieval failed mode=%s", request.mode)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        metrics = retriever.last_retrieval_metrics or RetrievalMetrics()
        LOGGER.info(
            "retrieve complete mode=%s top_k=%s hits=%s elapsed_seconds=%.3f",
            request.mode,
            request.top_k,
            len(hits),
            time.monotonic() - started_at,
        )
        return QueryResponse(
            query=request.query,
            retrieved_passages=hits,
            retrieval_metrics=metrics,
            grounded=None,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryApiRequest) -> QueryResponse:
        retrieval_response = retrieve(
            RetrieveRequest(query=request.query, top_k=request.top_k, mode=request.mode)
        )
        if not request.generate:
            return retrieval_response
        try:
            generator = make_generator(app_settings)
            grounded = generator.generate(
                request.query,
                retrieval_response.retrieved_passages or [],
            )
        except Exception as exc:
            LOGGER.exception("grounded generation failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return retrieval_response.model_copy(update={"grounded": grounded})

    return app


def safe_runtime_config(settings: Settings) -> RuntimeConfigResponse:
    """Return runtime metadata that excludes secrets and raw environment values."""

    artifacts = {
        "raw_dataset": _artifact_status(settings.output_dir / settings.raw_dataset_jsonl),
        "index_chunks": _artifact_status(settings.index_chunks_path),
        "chunk_manifest": _artifact_status(settings.chunk_manifest_path),
        "dense_checkpoint": _artifact_status(settings.dense_checkpoint_path),
        "sparse_checkpoint": _artifact_status(settings.sparse_checkpoint_path),
        "sparse_pass1": _artifact_status(settings.sparse_pass1_path),
        "sparse_manifest": _artifact_status(settings.sparse_manifest_path),
        "retrieval_eval": _artifact_status(settings.output_dir / "retrieval_eval.json"),
    }
    generation_api_configured = bool(settings.generation_api_url)
    generation_key_env_configured = bool(settings.generation_api_key_env)
    return RuntimeConfigResponse(
        dataset_name=settings.dataset_name,
        dataset_split=settings.dataset_split,
        embedding_model_name=settings.embedding_model_name,
        qdrant_url=_safe_url(settings.qdrant_url),
        qdrant_collection=settings.qdrant_collection,
        qdrant_vector_name=settings.qdrant_vector_name,
        qdrant_sparse_vector_name=settings.qdrant_sparse_vector_name,
        retrieval=RetrievalConfigMetadata(
            retrieve_k=settings.retrieve_k,
            rerank_k=settings.rerank_k,
            rerank_enabled=settings.rerank_enabled,
            rerank_model_name=settings.rerank_model_name,
            rerank_context_token_budget=settings.rerank_context_token_budget,
            retrieval_dedupe_enabled=settings.retrieval_dedupe_enabled,
            hybrid_rrf_k=settings.hybrid_rrf_k,
            hybrid_dense_weight=settings.hybrid_dense_weight,
            hybrid_sparse_weight=settings.hybrid_sparse_weight,
        ),
        generation=GenerationConfigMetadata(
            generation_provider=settings.generation_provider,
            generation_model_name=settings.generation_model_name,
            generation_temperature=settings.generation_temperature,
            generation_max_tokens=settings.generation_max_tokens,
            generation_timeout_seconds=settings.generation_timeout_seconds,
            generation_context_token_budget=settings.generation_context_token_budget,
            generation_min_citations=settings.generation_min_citations,
            generation_api_configured=generation_api_configured,
            generation_api_key_env_configured=generation_key_env_configured,
        ),
        artifacts=artifacts,
    )


def _artifact_status(path: Path) -> ArtifactStatus:
    return ArtifactStatus(path=str(path), exists=path.is_file())


def _safe_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.hostname:
        return value
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _default_retriever_factory(settings: Settings, mode: Mode) -> ApiRetriever:
    return QdrantModeRetriever(settings=settings, mode=mode)


def _default_generator_factory(settings: Settings) -> ApiGenerator:
    return GroundedGenerator(settings=settings)


app = create_app()
