"""Milestone 5 grounded generation over retrieved evidence."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.config.settings import Settings
from src.models.query_schemas import (
    Citation,
    GroundedAnswer,
    PassageHit,
    SupportedClaim,
    UnsupportedClaim,
)

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
_POINT_ID_RE = re.compile(r"point_id=([^\]\s]+)")
_CONTENT_RE = re.compile(r"Content:\s*(.+?)(?:\n\nEvidence \[E\d+\]|\Z)", re.DOTALL)


class LLMClient(Protocol):
    """Minimal text-completion protocol used by grounded generation."""

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> str: ...


@dataclass(slots=True, frozen=True)
class GroundedPrompt:
    """Rendered generation prompt plus the evidence hits it contains."""

    text: str
    evidence: list[PassageHit]


@dataclass(slots=True)
class GroundedPromptBuilder:
    """Build strict answer prompts from retrieved evidence."""

    context_token_budget: int

    def build(self, *, query: str, hits: Sequence[PassageHit]) -> GroundedPrompt:
        budget_remaining = self.context_token_budget
        blocks: list[str] = []
        rendered_evidence: list[PassageHit] = []
        for hit in hits:
            body = _evidence_text(hit)
            tokens = body.split()
            if budget_remaining <= 0:
                break
            elif len(tokens) > budget_remaining:
                body = " ".join(tokens[:budget_remaining])
            if not body.strip():
                continue
            budget_remaining -= len(body.split())
            rendered_evidence.append(hit)
            blocks.append(_format_evidence_block(len(rendered_evidence), hit, body))
        prompt = "\n\n".join(
            [
                "Grounding contract:",
                "You are not a general-knowledge assistant. You are an evidence-bound answer composer.",
                "Before writing the answer, decompose the requested answer into the minimum facts "
                "needed to answer it. Only include facts that are directly supported by the "
                "retrieved evidence.",
                "For every sentence in the answer:",
                "- The sentence must be supported by one or more retrieved point_ids.",
                "- The cited point_ids must support the specific sentence, not just the broad topic.",
                "- Do not transfer a fact from one entity, event, time, work, or category to another.",
                "- Do not fill missing links with outside knowledge.",
                "- Do not use plausible world knowledge unless it appears in the retrieved evidence.",
                "If the evidence supports only a narrower answer than the user asked for, answer "
                "narrowly and state the limitation.",
                "If the evidence does not support a useful answer, return abstained=true.",
                "Return only JSON with keys: abstained, answer, citations, supported_claims, "
                "unsupported_claims, abstention_reason.",
                "citations must be point_id values from the provided evidence.",
                "supported_claims must contain objects with keys: claim, point_ids.",
                "unsupported_claims must contain objects with keys: claim, reason.",
                f"Question: {query}",
                "\n\n".join(blocks) if blocks else "No evidence provided.",
            ]
        )
        return GroundedPrompt(text=prompt, evidence=rendered_evidence)


@dataclass(slots=True)
class GroundedGenerator:
    """Generate answer JSON, validate citations, and preserve supporting evidence."""

    settings: Settings
    client: LLMClient | None = None
    prompt_builder: GroundedPromptBuilder | None = None

    def __post_init__(self) -> None:
        if self.prompt_builder is None:
            self.prompt_builder = GroundedPromptBuilder(
                context_token_budget=self.settings.generation_context_token_budget
            )
        if self.client is None:
            self.client = build_default_llm_client(self.settings)

    def generate(self, query: str, hits: Sequence[PassageHit]) -> GroundedAnswer:
        prompt = self.prompt_builder.build(query=query, hits=hits)
        if not prompt.evidence:
            return _abstain()
        raw = self.client.complete(
            prompt.text,
            temperature=self.settings.generation_temperature,
            max_tokens=self.settings.generation_max_tokens,
            timeout_seconds=self.settings.generation_timeout_seconds,
        )
        return _grounded_answer_from_response(
            raw,
            evidence=prompt.evidence,
            min_citations=self.settings.generation_min_citations,
        )


class HeuristicGroundedClient:
    """Deterministic local client for smoke tests and offline demos."""

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> str:
        del temperature, max_tokens, timeout_seconds
        point_match = _POINT_ID_RE.search(prompt)
        content_match = _CONTENT_RE.search(prompt)
        if point_match is None or content_match is None:
            return json.dumps({"answer": "", "citations": [], "abstained": True})
        sentence = _first_sentence(content_match.group(1).strip())
        point_id = point_match.group(1)
        return json.dumps(
            {
                "answer": sentence,
                "citations": [point_id],
                "abstained": not bool(sentence),
                "supported_claims": (
                    [{"claim": sentence, "point_ids": [point_id]}] if sentence else []
                ),
                "unsupported_claims": [],
                "abstention_reason": None if sentence else "No supported evidence sentence found.",
            }
        )


@dataclass(slots=True, frozen=True)
class HttpJsonLLMClient:
    """Small HTTP JSON adapter for provider-specific local/proxy endpoints."""

    url: str
    model_name: str
    api_key_env: str | None = None

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = os.getenv(self.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Generation request failed: {exc}") from exc
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            for key in ("text", "content", "response", "answer"):
                value = parsed.get(key)
                if isinstance(value, str):
                    return value
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    text = first.get("text")
                    if isinstance(text, str):
                        return text
                    message = first.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"]
        raise RuntimeError("Generation response did not contain text content.")


@dataclass(slots=True, frozen=True)
class OpenAILLMClient:
    """OpenAI-compatible Chat Completions adapter."""

    model_name: str
    api_key_env: str = "OPENAI_API_KEY"
    url: str = OPENAI_CHAT_COMPLETIONS_URL

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> str:
        if _looks_like_openai_api_key(self.api_key_env):
            raise RuntimeError(
                "RAG_GENERATION_API_KEY_ENV must be an environment variable name "
                "such as OPENAI_API_KEY, not the API key value."
            )
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for OpenAI generation.")
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a grounded QA system. Return only valid JSON with "
                            "keys: abstained, answer, citations, supported_claims, "
                            "unsupported_claims, abstention_reason."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"OpenAI generation request failed: {exc}") from exc
        parsed = json.loads(body)
        content = _openai_chat_content(parsed)
        if content is None:
            raise RuntimeError("OpenAI response did not contain message content.")
        return content


def build_default_llm_client(settings: Settings) -> LLMClient:
    """Return the configured Milestone 5 LLM client."""

    if settings.generation_provider == "heuristic":
        return HeuristicGroundedClient()
    if settings.generation_provider == "http_json":
        if not settings.generation_api_url:
            raise RuntimeError("RAG_GENERATION_API_URL is required for http_json generation.")
        return HttpJsonLLMClient(
            url=settings.generation_api_url,
            model_name=settings.generation_model_name,
            api_key_env=settings.generation_api_key_env,
        )
    if settings.generation_provider == "openai":
        if settings.generation_model_name == "local-grounded-heuristic":
            raise RuntimeError("RAG_GENERATION_MODEL_NAME must be set for OpenAI generation.")
        return OpenAILLMClient(
            model_name=settings.generation_model_name,
            api_key_env=settings.generation_api_key_env or "OPENAI_API_KEY",
            url=settings.generation_api_url or OPENAI_CHAT_COMPLETIONS_URL,
        )
    raise ValueError(f"Unsupported generation provider {settings.generation_provider!r}.")


def _grounded_answer_from_response(
    raw: str, *, evidence: Sequence[PassageHit], min_citations: int
) -> GroundedAnswer:
    payload = _parse_response_json(raw)
    if payload is None:
        return _abstain()
    if payload.get("abstained") is True:
        return _abstain(
            reason=_optional_string(payload.get("abstention_reason")),
            unsupported_claims=_extract_unsupported_claims(payload.get("unsupported_claims")),
        )
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return _abstain()
    valid_by_id = {hit.point_id: hit for hit in evidence}
    cited_ids = _extract_cited_point_ids(payload.get("citations"))
    valid_ids = [point_id for point_id in cited_ids if point_id in valid_by_id]
    if len(valid_ids) < min_citations:
        return _abstain()
    supported_claims = _extract_supported_claims(
        payload.get("supported_claims"), valid_point_ids=set(valid_ids)
    )
    unsupported_claims = _extract_unsupported_claims(payload.get("unsupported_claims"))
    supporting = [valid_by_id[point_id].model_copy(deep=True) for point_id in valid_ids]
    return GroundedAnswer(
        answer=answer.strip(),
        citations=[Citation(point_id=point_id) for point_id in valid_ids],
        abstained=False,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        abstention_reason=None,
        supporting_point_ids=valid_ids,
        supporting_evidence=supporting,
    )


def _parse_response_json(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _openai_chat_content(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _looks_like_openai_api_key(value: str) -> bool:
    return value.startswith(("sk-", "sk_proj_", "sk-proj-"))


def _extract_cited_point_ids(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            point_id = item
        elif isinstance(item, dict) and isinstance(item.get("point_id"), str):
            point_id = item["point_id"]
        else:
            continue
        if point_id and point_id not in out:
            out.append(point_id)
    return out


def _extract_supported_claims(raw: object, *, valid_point_ids: set[str]) -> list[SupportedClaim]:
    if not isinstance(raw, list):
        return []
    out: list[SupportedClaim] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
            continue
        point_ids = _extract_point_id_list(item.get("point_ids"))
        valid_ids = [point_id for point_id in point_ids if point_id in valid_point_ids]
        if not valid_ids:
            continue
        out.append(SupportedClaim(claim=item["claim"].strip(), point_ids=valid_ids))
    return out


def _extract_unsupported_claims(raw: object) -> list[UnsupportedClaim]:
    if not isinstance(raw, list):
        return []
    out: list[UnsupportedClaim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = item.get("claim")
        reason = item.get("reason")
        if not isinstance(claim, str) or not isinstance(reason, str):
            continue
        claim = claim.strip()
        reason = reason.strip()
        if claim and reason:
            out.append(UnsupportedClaim(claim=claim, reason=reason))
    return out


def _extract_point_id_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for value in raw:
        if isinstance(value, str) and value and value not in out:
            out.append(value)
    return out


def _optional_string(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _evidence_text(hit: PassageHit) -> str:
    if hit.context_text and hit.context_text.strip():
        return hit.context_text.strip()
    return hit.text.strip()


def _format_evidence_block(index: int, hit: PassageHit, body: str) -> str:
    fields = [
        f"Evidence [E{index}] point_id={hit.point_id}",
        f"Title: {hit.title or ''}",
        f"Chunk kind: {hit.chunk_kind or ''}",
        f"Passage types: {', '.join(hit.passage_types)}",
        f"Content: {body}",
    ]
    return "\n".join(fields)


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(.+?[.!?])(?:\s|$)", text)
    return (match.group(1) if match else text).strip()


def _abstain(
    *, reason: str | None = None, unsupported_claims: list[UnsupportedClaim] | None = None
) -> GroundedAnswer:
    return GroundedAnswer(
        answer="",
        citations=[],
        abstained=True,
        unsupported_claims=unsupported_claims or [],
        abstention_reason=reason,
    )
