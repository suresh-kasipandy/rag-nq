"""Validate configuration and runtime dependencies for Milestone 0."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from src.config.settings import Settings


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Single doctor check outcome."""

    name: str
    ok: bool
    detail: str


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def check_qdrant_ready(qdrant_url: str, *, timeout_sec: float = 2.0) -> CheckResult:
    """GET ``/readyz`` on the configured Qdrant base URL."""

    base = qdrant_url.rstrip("/")
    target = f"{base}/readyz"
    request = Request(target, method="GET")
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            status = getattr(response, "status", 200)
            if status == 200:
                return CheckResult(name="qdrant:readyz", ok=True, detail=f"reachable ({target})")
            return CheckResult(
                name="qdrant:readyz",
                ok=False,
                detail=f"unexpected HTTP status {status} from {target}",
            )
    except HTTPError as exc:
        return CheckResult(
            name="qdrant:readyz",
            ok=False,
            detail=f"HTTP {exc.code} from {target}: {exc.reason!r}",
        )
    except URLError as exc:
        detail = f"{type(exc).__name__}: {exc.reason!r}"
        return CheckResult(name="qdrant:readyz", ok=False, detail=detail)
    except OSError as exc:
        return CheckResult(name="qdrant:readyz", ok=False, detail=f"{type(exc).__name__}: {exc}")


def _try_import(module: str) -> CheckResult:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        return CheckResult(name=f"import:{module}", ok=False, detail=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected import errors
        return CheckResult(name=f"import:{module}", ok=False, detail=str(exc))
    return CheckResult(name=f"import:{module}", ok=True, detail="ok")


def run_doctor() -> list[CheckResult]:
    """Run config validation and critical dependency import checks."""

    results: list[CheckResult] = []
    settings: Settings | None = None

    try:
        settings = Settings.from_env()
        results.append(CheckResult(name="config", ok=True, detail="Settings valid"))
    except ValidationError as exc:
        results.append(CheckResult(name="config", ok=False, detail=str(exc)))
    except Exception as exc:
        results.append(CheckResult(name="config", ok=False, detail=str(exc)))

    for module in (
        "pydantic",
        "datasets",
        "qdrant_client",
        "rank_bm25",
        "sentence_transformers",
    ):
        results.append(_try_import(module))

    if settings is not None and _env_flag_enabled("RAG_DOCTOR_CHECK_QDRANT"):
        results.append(check_qdrant_ready(settings.qdrant_url))

    return results


def main() -> int:
    """CLI entrypoint; exit 0 if all checks pass, 1 otherwise."""

    results = run_doctor()
    failed = [r for r in results if not r.ok]
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"{status}\t{r.name}\t{r.detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
