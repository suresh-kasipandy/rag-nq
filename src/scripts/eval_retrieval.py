"""CLI: run Milestone 6 offline retrieval evaluation and write artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import Settings
from src.evaluation.retrieval_eval import (
    SUPPORTED_EVAL_MODES,
    run_retrieval_evaluation,
)
from src.observability.logging_setup import setup_logging
from src.retrieval.qdrant_retrievers import Mode

MODE_VALUES: set[Mode] = set(SUPPORTED_EVAL_MODES)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _positive_int_csv(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(_positive_int(stripped))
    if not values:
        raise argparse.ArgumentTypeError("must include at least one integer")
    return values


def _mode_csv(value: str) -> list[Mode]:
    modes: list[Mode] = []
    for item in value.split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in MODE_VALUES:
            raise argparse.ArgumentTypeError(f"unsupported mode {mode!r}")
        modes.append(_mode_value(mode))
    if not modes:
        raise argparse.ArgumentTypeError("must include at least one mode")
    return modes


def _mode_value(value: str) -> Mode:
    if value == "dense":
        return "dense"
    if value == "sparse":
        return "sparse"
    if value == "hybrid":
        return "hybrid"
    raise argparse.ArgumentTypeError(f"unsupported mode {value!r}")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=None,
        help="Legacy single metrics cutoff k. Prefer --k-values.",
    )
    parser.add_argument(
        "--k-values",
        type=_positive_int_csv,
        default=None,
        help="Comma-separated metrics cutoffs, e.g. 1,5,10.",
    )
    parser.add_argument(
        "--modes",
        type=_mode_csv,
        default=list(SUPPORTED_EVAL_MODES),
        help="Comma-separated retrieval modes: dense,sparse,hybrid.",
    )
    parser.add_argument(
        "--relevance-contract",
        choices=(
            "answer_overlap",
            "point_id",
            "group_id",
            "source_row_ordinal",
            "candidate_span_overlap",
        ),
        default="answer_overlap",
        help="Chunk-aware relevance contract used for metric labels.",
    )
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
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional flat CSV artifact path.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Evaluation corpus JSONL (default: index_chunks.jsonl, fallback passages.jsonl).",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    output_path = args.output or (settings.output_dir / "retrieval_eval.json")
    report = run_retrieval_evaluation(
        settings,
        top_k=args.top_k,
        k_values=args.k_values,
        modes=args.modes,
        relevance_contract=args.relevance_contract,
        max_queries=args.max_queries,
        output_path=output_path,
        corpus_path=args.corpus,
        csv_output_path=args.csv_output,
    )
    print(
        f"Wrote retrieval eval report to {output_path} "
        f"for {report.run_config.query_count} queries "
        f"using {report.run_config.relevance_contract} relevance."
    )


if __name__ == "__main__":
    main()
