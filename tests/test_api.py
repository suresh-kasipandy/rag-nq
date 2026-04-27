from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.main import create_app, safe_runtime_config
from src.config.settings import Settings
from src.models.query_schemas import Citation, GroundedAnswer, PassageHit, RetrievalMetrics
from src.retrieval.qdrant_retrievers import Mode


class FakeRetriever:
    def __init__(self, *, mode: Mode) -> None:
        self.mode = mode
        self.last_retrieval_metrics = RetrievalMetrics()

    def retrieve(self, query: str, top_k: int) -> list[PassageHit]:
        return [
            PassageHit(
                point_id=f"{self.mode}-1",
                text=f"Evidence for {query}",
                title="Doc",
                dense_rank=1 if self.mode in {"dense", "hybrid"} else None,
                sparse_rank=1 if self.mode in {"sparse", "hybrid"} else None,
            )
        ][:top_k]


class FakeGenerator:
    def generate(self, query: str, hits: list[PassageHit]) -> GroundedAnswer:
        return GroundedAnswer(
            answer=f"Grounded answer for {query}",
            citations=[Citation(point_id=hits[0].point_id)],
            abstained=False,
            supporting_point_ids=[hits[0].point_id],
            supporting_evidence=hits[:1],
        )


def test_root_endpoint_points_to_api_docs_and_core_routes(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "rag-nq-showcase",
        "docs_url": "/docs",
        "health_url": "/health",
        "config_url": "/config",
        "retrieve_url": "/retrieve",
        "query_url": "/query",
    }


def test_favicon_endpoint_avoids_browser_404_noise(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/favicon.ico")

    assert response.status_code == 204
    assert response.content == b""


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "rag-nq-showcase"}


def test_config_endpoint_exposes_safe_runtime_metadata(tmp_path: Path) -> None:
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        qdrant_url="https://user:password@qdrant.example:6333",
        generation_api_url="https://provider.example",
        generation_api_key_env="OPENAI_API_KEY",
    )
    settings.index_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_chunks_path.write_text("{}\n", encoding="utf-8")
    client = _client(tmp_path, settings=settings)

    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["qdrant_collection"] == settings.qdrant_collection
    assert payload["qdrant_url"] == "https://qdrant.example:6333"
    assert payload["retrieval"]["hybrid_dense_weight"] == 0.5
    assert payload["generation"]["generation_api_configured"] is True
    assert payload["generation"]["generation_api_key_env_configured"] is True
    assert payload["artifacts"]["index_chunks"]["exists"] is True
    assert "password" not in response.text
    assert "OPENAI_API_KEY" not in response.text
    assert "provider.example" not in response.text


def test_retrieve_endpoint_returns_hits_and_metrics(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/retrieve",
        json={"query": "What is Paris?", "top_k": 1, "mode": "sparse"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "What is Paris?"
    assert payload["grounded"] is None
    assert payload["retrieved_passages"][0]["point_id"] == "sparse-1"
    assert payload["retrieval_metrics"] == {"dedupe": None, "timings": None}


def test_query_endpoint_can_generate_grounded_answer(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/query",
        json={"query": "What is Paris?", "top_k": 1, "mode": "hybrid", "generate": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieved_passages"][0]["point_id"] == "hybrid-1"
    assert payload["grounded"]["answer"] == "Grounded answer for What is Paris?"
    assert payload["grounded"]["citations"] == [{"point_id": "hybrid-1"}]


def test_query_endpoint_can_skip_generation(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/query",
        json={"query": "What is Paris?", "top_k": 1, "mode": "dense", "generate": False},
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is None


def test_retrieve_endpoint_maps_runtime_failures_to_503(tmp_path: Path) -> None:
    def failing_retriever_factory(settings: Settings, mode: Mode):
        del settings, mode
        return SimpleNamespace(
            last_retrieval_metrics=None,
            retrieve=lambda query, top_k: (_ for _ in ()).throw(RuntimeError("qdrant down")),
        )

    client = _client(tmp_path, retriever_factory=failing_retriever_factory)

    response = client.post("/retrieve", json={"query": "q", "top_k": 1, "mode": "dense"})

    assert response.status_code == 503
    assert response.json()["detail"] == "qdrant down"


def test_safe_runtime_config_does_not_expose_secret_names_or_urls(tmp_path: Path) -> None:
    settings = Settings(
        output_dir=tmp_path / "artifacts",
        qdrant_url="https://user:password@qdrant.example:6333",
        generation_api_url="https://provider.example",
        generation_api_key_env="SECRET_ENV_NAME",
    )

    payload = safe_runtime_config(settings).model_dump_json()

    assert "SECRET_ENV_NAME" not in payload
    assert "provider.example" not in payload
    assert "password" not in payload
    assert "https://qdrant.example:6333" in payload
    assert "generation_api_configured" in payload


def _client(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    retriever_factory=None,
) -> TestClient:
    app = create_app(
        settings=settings or Settings(output_dir=tmp_path / "artifacts"),
        retriever_factory=(
            retriever_factory
            or (lambda settings, mode: FakeRetriever(mode=mode))
        ),
        generator_factory=lambda settings: FakeGenerator(),
    )
    return TestClient(app)
