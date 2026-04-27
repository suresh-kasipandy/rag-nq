"""Helpers for loading and summarizing retrieval eval artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class MetricRow:
    """One mode/k row from a retrieval eval report."""

    mode: str
    k: int
    query_count: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


@dataclass(slots=True, frozen=True)
class EvalSummary:
    """Flattened eval report plus run metadata and per-metric winners."""

    rows: list[MetricRow]
    run_config: dict[str, Any]
    winners: dict[str, MetricRow]


def load_eval_report(path: Path) -> dict[str, Any]:
    """Load a machine-readable retrieval eval artifact."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("modes"), dict):
        raise ValueError(f"Invalid retrieval eval report: {path}")
    return payload


def summarize_eval_report(report: dict[str, Any]) -> EvalSummary:
    """Flatten mode metrics and identify best rows for major metrics."""

    rows: list[MetricRow] = []
    modes = report.get("modes")
    if not isinstance(modes, dict):
        raise ValueError("Eval report missing modes.")
    for mode, mode_payload in modes.items():
        if not isinstance(mode_payload, dict):
            continue
        metrics_by_k = mode_payload.get("metrics_by_k")
        if not isinstance(metrics_by_k, dict):
            continue
        for raw_k, metric_payload in metrics_by_k.items():
            if not isinstance(metric_payload, dict):
                continue
            rows.append(_metric_row(mode=str(mode), raw_k=str(raw_k), payload=metric_payload))
    rows.sort(key=lambda row: (row.k, row.mode))
    winners = {
        "recall_at_k": _winner(rows, "recall_at_k"),
        "mrr_at_k": _winner(rows, "mrr_at_k"),
        "ndcg_at_k": _winner(rows, "ndcg_at_k"),
    }
    run_config = report.get("run_config")
    return EvalSummary(
        rows=rows,
        run_config=run_config if isinstance(run_config, dict) else {},
        winners=winners,
    )


def rows_as_dicts(rows: list[MetricRow]) -> list[dict[str, int | float | str]]:
    """Return Streamlit-friendly row dictionaries."""

    return [
        {
            "mode": row.mode,
            "k": row.k,
            "query_count": row.query_count,
            "recall_at_k": row.recall_at_k,
            "mrr_at_k": row.mrr_at_k,
            "ndcg_at_k": row.ndcg_at_k,
        }
        for row in rows
    ]


def _metric_row(*, mode: str, raw_k: str, payload: dict[str, Any]) -> MetricRow:
    return MetricRow(
        mode=mode,
        k=int(raw_k),
        query_count=int(payload.get("query_count", 0)),
        recall_at_k=float(payload.get("recall_at_k", 0.0)),
        mrr_at_k=float(payload.get("mrr_at_k", 0.0)),
        ndcg_at_k=float(payload.get("ndcg_at_k", 0.0)),
    )


def _winner(rows: list[MetricRow], metric_name: str) -> MetricRow:
    if not rows:
        raise ValueError("Cannot compute eval winners without metric rows.")
    return max(rows, key=lambda row: float(getattr(row, metric_name)))
