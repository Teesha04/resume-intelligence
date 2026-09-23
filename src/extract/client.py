"""LLM provider adapter.

Design goals (spec section 6):
  * provider-specific code lives behind ONE small interface, swappable
  * structured output (JSON) enforced; we validate with Pydantic downstream
  * failures raise LLMError and are caught per-resume, never batch-fatal
  * keys come from the environment (config.Settings), never hard-coded

Concurrency: the pipeline calls these synchronously from a bounded thread
pool (see pipeline.py). That keeps the concurrency model simple and easy to
reason about versus mixing async clients.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..cache import DiskCache

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call cannot produce usable structured output."""


class _RateLimiter:
    """Simple thread-safe limiter keeping calls under N per minute.

    Free LLM tiers enforce a low requests-per-minute quota; without pacing, a
    50-resume batch trips 429s and silently falls back to deterministic scoring
    for most candidates (inconsistent results). Pacing is the reliable fix.
    """

    def __init__(self, per_minute: int) -> None:
        self.min_interval = 60.0 / per_minute if per_minute and per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval


def _retry_delay_seconds(error: str) -> float | None:
    """Extract a retry delay from a provider error message, if present."""
    for pattern in (r"retry in (\d+(?:\.\d+)?)s", r"retryDelay'?:\s*'?(\d+)s"):
        m = re.search(pattern, error)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]: ...


class NullLLMClient:
    """Used when no key is configured. Always fails softly."""

    name = "null"

    def generate_json(self, **_: Any) -> dict[str, Any]:
        raise LLMError("LLM disabled: no API key configured")


class GeminiClient:
    """Google Gemini via the `google-genai` SDK."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        *,
        timeout: float = 20.0,
        max_retries: int = 2,
        requests_per_minute: int = 12,
        on_rate_limit: str = "wait",
        max_wait_seconds: float = 90.0,
        wait_budget_seconds: float = 600.0,
        cache: DiskCache | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_rate_limit = on_rate_limit
        self.max_wait_seconds = max_wait_seconds
        self.wait_budget_seconds = wait_budget_seconds
        self.cache = cache
        self._limiter = _RateLimiter(requests_per_minute)
        self._client = None

        # Observability counters (read by the pipeline for the batch summary).
        self.rate_limit_waits = 0
        self.total_wait_seconds = 0.0

    def _get_client(self):
        if self._client is None:
            from google import genai  # imported lazily so no key => no import cost

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        cache_payload = {
            "provider": self.name,
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "schema": schema.model_json_schema() if schema else None,
            "temperature": temperature,
        }
        if self.cache:
            cached = self.cache.get("llm", cache_payload)
            if cached is not None:
                return cached

        from google.genai import types

        client = self._get_client()
        last_error: Exception | None = None

        # Try with a response schema first (strongest structured output), then
        # fall back to JSON mime-type only if the schema is rejected by the SDK.
        attempts: list[dict[str, Any]] = []
        base_cfg: dict[str, Any] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "system_instruction": system,
        }
        if schema is not None:
            attempts.append({**base_cfg, "response_schema": schema})
        attempts.append(dict(base_cfg))

        for config_kwargs in attempts:
            attempt = 0
            while True:
                self._limiter.wait()
                try:
                    response = client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                    parsed = self._parse_response(response)
                    if self.cache:
                        self.cache.set("llm", cache_payload, parsed)
                    return parsed
                except Exception as exc:  # noqa: BLE001 - adapter boundary
                    last_error = exc
                    message = str(exc).lower()
                    is_quota = any(
                        token in message
                        for token in ("429", "quota", "rate limit", "resource_exhausted",
                                      "resource exhausted", "exceeded your current quota")
                    )
                    is_transient = is_quota or any(
                        token in message for token in ("timeout", "unavailable", "503", "500")
                    )

                    # Per-minute quota: pause until it resets, then resume the
                    # SAME request instead of falling back to deterministic.
                    if is_quota and self.on_rate_limit == "wait":
                        delay = _retry_delay_seconds(str(exc)) or 60.0
                        delay = min(delay, self.max_wait_seconds)
                        if self.total_wait_seconds + delay <= self.wait_budget_seconds:
                            log.warning(
                                "LLM rate limit reached; waiting %.0fs for the quota to "
                                "reset, then resuming…", delay,
                            )
                            time.sleep(delay)
                            self.rate_limit_waits += 1
                            self.total_wait_seconds += delay
                            continue  # retry the same request; no attempt consumed

                    if not is_transient or attempt >= self.max_retries:
                        break  # schema problem or exhausted retries: next strategy
                    attempt += 1
                    delay = _retry_delay_seconds(str(exc))
                    if delay:
                        # Respect the server's own backoff hint (capped).
                        time.sleep(min(delay, self.max_wait_seconds))
                    log.debug("LLM transient error, retrying (%s): %s", attempt, exc)

        raise LLMError(f"Gemini call failed: {type(last_error).__name__}: {last_error}")

    @staticmethod
    def _parse_response(response: Any) -> dict[str, Any]:
        text = getattr(response, "text", None)
        if not text:
            raise LLMError("empty response from model")
        # Models sometimes wrap JSON in markdown fences despite the mime type.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            cleaned = cleaned.removeprefix("json").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMError(f"response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("expected a JSON object at the top level")
        return data


def build_llm_client(
    *,
    provider: str,
    api_key: str | None,
    model: str,
    timeout: float,
    max_retries: int,
    requests_per_minute: int = 12,
    on_rate_limit: str = "wait",
    max_wait_seconds: float = 90.0,
    wait_budget_seconds: float = 600.0,
    cache: DiskCache | None = None,
) -> LLMClient:
    """Factory. Returns a NullLLMClient when no key is available."""
    if not api_key:
        log.info("No LLM API key configured; running in deterministic-only mode.")
        return NullLLMClient()
    if provider.lower() in {"gemini", "google"}:
        return GeminiClient(
            api_key, model,
            timeout=timeout,
            max_retries=max_retries,
            requests_per_minute=requests_per_minute,
            on_rate_limit=on_rate_limit,
            max_wait_seconds=max_wait_seconds,
            wait_budget_seconds=wait_budget_seconds,
            cache=cache,
        )
    log.warning("Unknown LLM provider %r; falling back to deterministic-only.", provider)
    return NullLLMClient()
