"""Index build entrypoint (Milestone 1 pipeline; uses Milestone 0 logging helpers)."""

from __future__ import annotations

import logging

from src.config.settings import Settings
from src.ingestion.models import IndexBuildManifest
from src.ingestion.nq_loader import load_nq_passages
from src.ingestion.passage_store import PassageStore
from src.observability.logging_setup import get_stage_logger, setup_logging
from src.retrieval.dense_index import DenseIndexer
from src.retrieval.sparse_index import SparseIndexer

LOGGER = get_stage_logger(__name__)


def build_indexes(settings: Settings) -> IndexBuildManifest:
    """Run deterministic passage loading + dense/sparse index build."""

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    passages = load_nq_passages(settings)
    LOGGER.info("loaded %s passages", len(passages), extra={"stage": "ingest"})

    PassageStore.write_jsonl(passages=passages, path=settings.passages_path)
    LOGGER.info("wrote passages JSONL", extra={"stage": "persist"})

    dense_indexer = DenseIndexer(settings=settings)
    dense_result = dense_indexer.build(passages)
    LOGGER.info(
        "vectors=%s vector_size=%s",
        dense_result.vector_count,
        dense_result.vector_size,
        extra={"stage": "dense_index"},
    )

    sparse_indexer, sparse_result = SparseIndexer.build(passages)
    sparse_indexer.save(settings.sparse_index_path)
    LOGGER.info("documents=%s", sparse_result.document_count, extra={"stage": "sparse_index"})

    manifest = IndexBuildManifest(
        dataset_name=settings.dataset_name,
        dataset_split=settings.dataset_split,
        embedding_model_name=settings.embedding_model_name,
        passage_count=len(passages),
        qdrant_collection=settings.qdrant_collection,
        sparse_index_path=str(settings.sparse_index_path),
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
