"""Index build entrypoint: chunk ingest → dense → sparse."""

from __future__ import annotations

import logging

from src.config.settings import Settings
from src.ingestion.chunk_raw import run_chunk_ingest
from src.ingestion.models import INDEX_CHUNK_SCHEMA_VERSION, IndexBuildManifest
from src.ingestion.passage_store import PassageStore
from src.observability.logging_setup import get_stage_logger, setup_logging
from src.retrieval.dense_index import DenseIndexer
from src.retrieval.sparse_qdrant import SparseQdrantIndexer

LOGGER = get_stage_logger(__name__)


def build_indexes(settings: Settings) -> IndexBuildManifest:
    """Run chunk ingest (optional skip), chunked dense upsert, and two-pass sparse build."""

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("stage start: chunk ingest", extra={"stage": "build"})
    chunk_manifest, _ = run_chunk_ingest(settings, force=Settings.force_ingest_from_env())
    chunk_count = chunk_manifest.line_count

    dense_indexed_from = "artifact"
    LOGGER.info(
        "stage start: dense index source=%s chunks=%s",
        dense_indexed_from,
        chunk_count,
        extra={"stage": "dense_index", "dense_source": dense_indexed_from},
    )

    dense_indexer = DenseIndexer(settings=settings)
    dense_result = dense_indexer.build_from_jsonl_streaming(
        settings.index_chunks_path,
        lines_per_batch=settings.dense_read_batch_lines,
        max_passages=settings.max_passages,
    )
    LOGGER.info(
        "vectors=%s vector_size=%s",
        dense_result.vector_count,
        dense_result.vector_size,
        extra={"stage": "dense_index"},
    )

    LOGGER.info("stage start: sparse index chunks=%s", chunk_count, extra={"stage": "build"})
    sparse_indexer = SparseQdrantIndexer(settings=settings)
    sparse_result = sparse_indexer.build_from_jsonl(
        settings.index_chunks_path, max_passages=settings.max_passages
    )
    LOGGER.info(
        "documents=%s sparse_vocab=%s points_updated=%s",
        sparse_result.document_count,
        sparse_result.vocabulary_size,
        sparse_result.points_updated,
        extra={"stage": "sparse_qdrant_index"},
    )

    LOGGER.info("stage start: write manifest", extra={"stage": "build"})
    manifest = IndexBuildManifest(
        dataset_name=settings.dataset_name,
        dataset_split=settings.dataset_split,
        embedding_model_name=settings.embedding_model_name,
        passage_count=chunk_count,
        chunk_count=chunk_count,
        qdrant_collection=settings.qdrant_collection,
        sparse_index_path=str(settings.sparse_manifest_path),
        chunk_schema_version=INDEX_CHUNK_SCHEMA_VERSION,
        chunk_artifact_path=str(settings.index_chunks_path),
        dense_indexed_from=dense_indexed_from,
    )
    PassageStore.write_manifest(manifest=manifest, path=settings.manifest_path)
    LOGGER.info("wrote manifest", extra={"stage": "manifest"})
    return manifest


def main() -> None:
    """CLI entrypoint."""

    setup_logging(level=logging.INFO)
    settings = Settings.from_env()
    manifest = build_indexes(settings)
    LOGGER.info("complete: %s", manifest.model_dump_json(indent=2), extra={"stage": "done"})


if __name__ == "__main__":
    main()
