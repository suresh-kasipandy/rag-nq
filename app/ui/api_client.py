"""Small typed HTTP client for the Milestone 7 API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ValidationError

from app.api.schemas import HealthResponse, RuntimeConfigResponse
from src.models.query_schemas import QueryResponse

RetrievalMode = Literal["dense", "sparse", "hybrid"]


class ApiClientError(RuntimeError):
    """Raised when the Streamlit UI cannot reach or parse the API."""


@dataclass(slots=True, frozen=True)
class RagApiClient:
    """HTTP client for FastAPI endpoints used by the Streamlit UI."""

    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 30.0

    def health(self) -> HealthResponse:
        return _validate_response(
            HealthResponse,
            self._request_json("GET", "/health"),
            endpoint="/health",
        )

    def config(self) -> RuntimeConfigResponse:
        return _validate_response(
            RuntimeConfigResponse,
            self._request_json("GET", "/config"),
            endpoint="/config",
        )

    def retrieve(self, *, query: str, mode: RetrievalMode, top_k: int) -> QueryResponse:
        payload = {"query": query, "mode": mode, "top_k": top_k}
        return _validate_response(
            QueryResponse,
            self._request_json("POST", "/retrieve", payload),
            endpoint="/retrieve",
        )

    def query(
        self, *, query: str, mode: RetrievalMode, top_k: int, generate: bool
    ) -> QueryResponse:
        payload = {"query": query, "mode": mode, "top_k": top_k, "generate": generate}
        return _validate_response(
            QueryResponse,
            self._request_json("POST", "/query", payload),
            endpoint="/query",
        )

    def _request_json(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _join_url(self.base_url, path),
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ApiClientError(f"API returned {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ApiClientError(f"API request failed: {exc}") from exc


def _join_url(base_url: str, path: str) -> str:
    normalized_base = base_url.rstrip("/") + "/"
    return urljoin(normalized_base, path.lstrip("/"))


def _validate_response[ResponseModel: BaseModel](
    model_type: type[ResponseModel],
    payload: Any,
    *,
    endpoint: str,
) -> ResponseModel:
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ApiClientError(f"API response from {endpoint} did not match schema: {exc}") from exc
