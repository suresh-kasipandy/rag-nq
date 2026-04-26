"""CLI: retrieve evidence and run Milestone 5 grounded generation."""

from __future__ import annotations

import argparse
import json

from src.config.settings import Settings
from src.generation.grounded import GroundedGenerator
from src.models.query_schemas import QueryResponse, RetrievalMetrics
from src.retrieval.qdrant_retrievers import QdrantModeRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument(
        "--mode",
        choices=("dense", "sparse", "hybrid"),
        default="hybrid",
        help="Retrieval mode to execute before generation.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Maximum evidence hits.")
    args = parser.parse_args()

    settings = Settings.from_env()
    retriever = QdrantModeRetriever(settings=settings, mode=args.mode)
    hits = retriever.retrieve(args.query, top_k=args.top_k)
    generator = GroundedGenerator(settings=settings)
    grounded = generator.generate(args.query, hits)
    response = QueryResponse(
        query=args.query,
        retrieved_passages=hits,
        retrieval_metrics=(
            retriever.last_retrieval_metrics
            if retriever.last_retrieval_metrics is not None
            else RetrievalMetrics()
        ),
        grounded=grounded,
    )
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
