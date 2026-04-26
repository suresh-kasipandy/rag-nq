"""CLI: run Milestone 3 retrieval modes and print ranked passages as JSON."""

from __future__ import annotations

import argparse
import json

from src.config.settings import Settings
from src.retrieval.qdrant_retrievers import QdrantModeRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="User query text to retrieve against.")
    parser.add_argument(
        "--mode",
        choices=("dense", "sparse", "hybrid"),
        default="hybrid",
        help="Retrieval mode to execute.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Maximum passages to return.")
    parser.add_argument(
        "--include-metrics",
        action="store_true",
        help="Print retrieval metrics alongside hits.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    retriever = QdrantModeRetriever(settings=settings, mode=args.mode)
    hits = retriever.retrieve(args.query, top_k=args.top_k)
    if args.include_metrics:
        payload = {
            "hits": [hit.model_dump() for hit in hits],
            "metrics": (
                retriever.last_retrieval_metrics.model_dump()
                if retriever.last_retrieval_metrics is not None
                else None
            ),
        }
    else:
        payload = [hit.model_dump() for hit in hits]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
