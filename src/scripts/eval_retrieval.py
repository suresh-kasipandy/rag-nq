"""CLI: run Milestone 3.5 retrieval-only evaluation and write JSON artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import Settings
from src.evaluation.retrieval_eval import run_retrieval_evaluation


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=_positive_int, default=10, help="Metrics cutoff k.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Maximum number of eval queries built from passages.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <output_dir>/retrieval_eval.json).",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    output_path = args.output or (settings.output_dir / "retrieval_eval.json")
    report = run_retrieval_evaluation(
        settings,
        top_k=args.top_k,
        max_queries=args.max_queries,
        output_path=output_path,
    )
    print(
        f"Wrote retrieval eval report to {output_path} "
        f"for {report.run_config.query_count} queries."
    )


if __name__ == "__main__":
    main()
