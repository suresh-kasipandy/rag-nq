"""Analyze chunking strategies over the raw NQ artifact and write reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import Settings
from src.evaluation.chunking_strategy_eval import (
    run_chunking_evaluation,
    write_report_json,
    write_report_markdown,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rows",
        type=_positive_int,
        default=1000,
        help="Maximum raw rows to profile and evaluate.",
    )
    parser.add_argument(
        "--max-queries",
        type=_positive_int,
        default=100,
        help="Maximum labeled questions to use for BM25 proxy evaluation.",
    )
    parser.add_argument("--top-k", type=_positive_int, default=10, help="Retrieval cutoff.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Markdown report path (default: artifacts/chunking_evaluation_report.md).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON report path (default: artifacts/chunking_evaluation_report.json).",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    json_output = args.json_output or (settings.output_dir / "chunking_evaluation_report.json")
    markdown_output = args.markdown_output or (
        settings.output_dir / "chunking_evaluation_report.md"
    )
    report = run_chunking_evaluation(
        settings.raw_dataset_path,
        max_rows=args.max_rows,
        max_queries=args.max_queries,
        top_k=args.top_k,
    )
    write_report_json(json_output, report)
    write_report_markdown(markdown_output, report)
    print(
        f"Wrote chunking evaluation reports to {markdown_output} and {json_output} "
        f"for {report.dataset_profile.sample_rows} rows."
    )


if __name__ == "__main__":
    main()
