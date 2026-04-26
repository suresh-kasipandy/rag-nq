from __future__ import annotations

import json
from typing import Any

import pytest

from src.config.settings import Settings
from src.generation.grounded import (
    GroundedGenerator,
    GroundedPromptBuilder,
    HeuristicGroundedClient,
    OpenAILLMClient,
    build_default_llm_client,
)
from src.models.query_schemas import PassageHit


class FakeLLMClient:
    def __init__(self, payload: dict[str, object] | str) -> None:
        self.payload = payload
        self.last_prompt: str | None = None
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None
        self.last_timeout_seconds: float | None = None

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> str:
        self.last_prompt = prompt
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        self.last_timeout_seconds = timeout_seconds
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_prompt_builder_includes_ids_metadata_and_bounded_context() -> None:
    hit = PassageHit(
        point_id="p1",
        text="short text",
        context_text="one two three four five",
        title="Doc",
        chunk_kind="table_span",
        passage_types=["table"],
    )

    prompt = GroundedPromptBuilder(context_token_budget=3).build(query="q", hits=[hit])

    assert prompt.evidence == [hit]
    assert "point_id=p1" in prompt.text
    assert "Title: Doc" in prompt.text
    assert "Chunk kind: table_span" in prompt.text
    assert "Passage types: table" in prompt.text
    assert "Content: one two three" in prompt.text
    assert "four" not in prompt.text


def test_prompt_builder_only_allows_rendered_evidence_to_be_cited() -> None:
    hits = [
        PassageHit(point_id="p1", text="one two three"),
        PassageHit(point_id="p2", text="four five six"),
    ]

    prompt = GroundedPromptBuilder(context_token_budget=3).build(query="q", hits=hits)

    assert [hit.point_id for hit in prompt.evidence] == ["p1"]
    assert "point_id=p1" in prompt.text
    assert "point_id=p2" not in prompt.text


def test_grounded_generator_returns_valid_point_id_citations_and_supporting_evidence() -> None:
    client = FakeLLMClient(
        {
            "answer": "The release date was 9 May 2012.",
            "citations": ["p1", {"point_id": "missing"}],
            "abstained": False,
        }
    )
    hit = PassageHit(point_id="p1", text="Xbox 360WW: 9 May 2012", title="Minecraft")
    settings = Settings(
        generation_temperature=0.1,
        generation_max_tokens=99,
        generation_timeout_seconds=8.0,
    )
    generator = GroundedGenerator(settings=settings, client=client)

    answer = generator.generate("When?", [hit])

    assert answer.abstained is False
    assert answer.answer == "The release date was 9 May 2012."
    assert [citation.point_id for citation in answer.citations] == ["p1"]
    assert answer.supporting_point_ids == ["p1"]
    assert answer.supporting_evidence[0].point_id == "p1"
    assert client.last_temperature == pytest.approx(0.1)
    assert client.last_max_tokens == 99
    assert client.last_timeout_seconds == pytest.approx(8.0)


def test_grounded_generator_abstains_when_evidence_is_missing() -> None:
    client = FakeLLMClient({"answer": "Unsupported", "citations": ["p1"], "abstained": False})
    generator = GroundedGenerator(settings=Settings(), client=client)

    answer = generator.generate("q", [])

    assert answer.abstained is True
    assert answer.citations == []
    assert answer.supporting_evidence == []
    assert client.last_prompt is None


def test_grounded_generator_abstains_when_citations_are_invalid() -> None:
    client = FakeLLMClient({"answer": "Unsupported", "citations": ["missing"], "abstained": False})
    generator = GroundedGenerator(settings=Settings(), client=client)

    answer = generator.generate("q", [PassageHit(point_id="p1", text="evidence")])

    assert answer.abstained is True
    assert answer.citations == []


def test_grounded_generator_rejects_citation_to_unrendered_evidence() -> None:
    client = FakeLLMClient(
        {"answer": "Uses hidden evidence.", "citations": ["p2"], "abstained": False}
    )
    generator = GroundedGenerator(
        settings=Settings(generation_context_token_budget=3),
        client=client,
    )

    answer = generator.generate(
        "q",
        [
            PassageHit(point_id="p1", text="one two three"),
            PassageHit(point_id="p2", text="four five six"),
        ],
    )

    assert answer.abstained is True


def test_heuristic_client_returns_first_evidence_citation() -> None:
    generator = GroundedGenerator(settings=Settings(), client=HeuristicGroundedClient())

    answer = generator.generate("q", [PassageHit(point_id="p1", text="First sentence. Second.")])

    assert answer.abstained is False
    assert answer.answer == "First sentence."
    assert [citation.point_id for citation in answer.citations] == ["p1"]


def test_openai_client_posts_chat_completion_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeHttpResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "The release date was 9 May 2012.",
                                    "citations": ["p1"],
                                    "abstained": False,
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAILLMClient(model_name="gpt-test")

    out = client.complete("prompt", temperature=0.0, max_tokens=128, timeout_seconds=4.0)

    assert json.loads(out)["citations"] == ["p1"]
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["messages"][1] == {"role": "user", "content": "prompt"}
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["max_tokens"] == 128
    assert captured["timeout"] == pytest.approx(4.0)


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAILLMClient(model_name="gpt-test")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        client.complete("prompt", temperature=0.0, max_tokens=1, timeout_seconds=1.0)


def test_build_default_llm_client_supports_openai_provider() -> None:
    settings = Settings(
        generation_provider="openai",
        generation_model_name="gpt-test",
        generation_api_url="https://example.test/chat",
        generation_api_key_env="MY_OPENAI_KEY",
    )

    client = build_default_llm_client(settings)

    assert isinstance(client, OpenAILLMClient)
    assert client.model_name == "gpt-test"
    assert client.url == "https://example.test/chat"
    assert client.api_key_env == "MY_OPENAI_KEY"


def test_build_default_llm_client_requires_openai_model_override() -> None:
    settings = Settings(generation_provider="openai")

    with pytest.raises(RuntimeError, match="RAG_GENERATION_MODEL_NAME must be set"):
        build_default_llm_client(settings)


def test_grounded_generator_can_use_openai_provider_in_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: Any, timeout: float) -> FakeHttpResponse:
        del request, timeout
        return FakeHttpResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "The release date was 9 May 2012.",
                                    "citations": ["p1"],
                                    "abstained": False,
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-token")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    settings = Settings(generation_provider="openai", generation_model_name="gpt-test")
    generator = GroundedGenerator(settings=settings)

    answer = generator.generate(
        "When?",
        [PassageHit(point_id="p1", text="Xbox 360WW: 9 May 2012", title="Minecraft")],
    )

    assert answer.abstained is False
    assert answer.answer == "The release date was 9 May 2012."
    assert answer.supporting_point_ids == ["p1"]
