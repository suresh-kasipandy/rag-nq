from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from app.ui.api_client import ApiClientError, RagApiClient
from app.ui.display import DEFAULT_ABSTENTION_MESSAGE, answer_display_text, hit_rank_summary
from app.ui.eval_report import load_eval_report, rows_as_dicts, summarize_eval_report
from src.models.query_schemas import GroundedAnswer, PassageHit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_api_client_shapes_query_request_and_validates_response(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHttpResponse(
            {
                "query": "What is Paris?",
                "retrieved_passages": [{"point_id": "p1", "text": "Paris evidence."}],
                "retrieval_metrics": None,
                "grounded": {
                    "answer": "Paris evidence.",
                    "citations": [{"point_id": "p1"}],
                    "abstained": False,
                    "supporting_point_ids": ["p1"],
                    "supporting_evidence": [{"point_id": "p1", "text": "Paris evidence."}],
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = RagApiClient(base_url="http://api.local/", timeout_seconds=3.0).query(
        query="What is Paris?",
        mode="hybrid",
        top_k=5,
        generate=True,
    )

    assert captured["url"] == "http://api.local/query"
    assert captured["timeout"] == pytest.approx(3.0)
    assert captured["body"] == {
        "query": "What is Paris?",
        "mode": "hybrid",
        "top_k": 5,
        "generate": True,
    }
    assert response.grounded is not None
    assert response.grounded.citations[0].point_id == "p1"


def test_streamlit_entrypoint_imports_when_working_directory_is_app() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_path('streamlit_app.py', run_name='streamlit_smoke')",
        ],
        cwd=PROJECT_ROOT / "app",
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_api_client_maps_http_errors(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.HTTPError(
            url="http://api.local/retrieve",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=FakeErrorBody(),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ApiClientError, match="API returned 503"):
        RagApiClient(base_url="http://api.local").retrieve(
            query="q",
            mode="dense",
            top_k=1,
        )


def test_api_client_maps_schema_errors(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        return FakeHttpResponse({"unexpected": "shape"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ApiClientError, match="did not match schema"):
        RagApiClient(base_url="http://api.local").query(
            query="What is Paris?",
            mode="hybrid",
            top_k=5,
            generate=True,
        )


def test_answer_display_text_uses_abstention_reason_without_populating_answer() -> None:
    grounded = GroundedAnswer(
        answer="",
        abstained=True,
        abstention_reason="Retrieved evidence did not support the requested comparison.",
    )

    text = answer_display_text(grounded)

    assert DEFAULT_ABSTENTION_MESSAGE in text
    assert "Retrieved evidence did not support" in text


def test_answer_display_text_returns_grounded_answer_when_not_abstained() -> None:
    assert answer_display_text(GroundedAnswer(answer="Paris.", abstained=False)) == "Paris."


def test_hit_rank_summary_lists_available_ranks() -> None:
    hit = PassageHit(
        point_id="p1",
        text="evidence",
        dense_rank=1,
        sparse_rank=2,
        fusion_rank=3,
        rerank_rank=4,
        dedupe_rank=5,
    )

    assert hit_rank_summary(hit) == "dense #1 | sparse #2 | fusion #3 | rerank #4 | dedupe #5"


def test_eval_report_summary_flattens_rows_and_winners(tmp_path: Path) -> None:
    path = tmp_path / "retrieval_eval.json"
    path.write_text(
        json.dumps(
            {
                "modes": {
                    "dense": {
                        "metrics_by_k": {
                            "1": {
                                "query_count": 2,
                                "recall_at_k": 0.5,
                                "mrr_at_k": 0.4,
                                "ndcg_at_k": 0.3,
                            }
                        }
                    },
                    "sparse": {
                        "metrics_by_k": {
                            "1": {
                                "query_count": 2,
                                "recall_at_k": 0.6,
                                "mrr_at_k": 0.5,
                                "ndcg_at_k": 0.7,
                            }
                        }
                    },
                },
                "run_config": {"query_count": 2},
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_eval_report(load_eval_report(path))

    assert len(summary.rows) == 2
    assert summary.winners["recall_at_k"].mode == "sparse"
    assert summary.winners["ndcg_at_k"].ndcg_at_k == pytest.approx(0.7)
    assert rows_as_dicts(summary.rows)[0]["k"] == 1


def test_eval_report_rejects_invalid_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid retrieval eval report"):
        load_eval_report(path)


class FakeErrorBody:
    def read(self) -> bytes:
        return b'{"detail":"down"}'

    def close(self) -> None:
        return None
