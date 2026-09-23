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
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from ..cache import DiskCache

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call cannot produce usable structured output."""


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
        model: str = "gemini-2.0-flash",
        *,
        timeout: float = 20.0,
        max_retries: int = 2,
        cache: DiskCache | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = cache
        self._client = None

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
            for attempt in range(self.max_retries + 1):
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
                    transient = any(
                        token in str(exc).lower()
                        for token in ("429", "rate", "timeout", "unavailable", "503", "500")
                    )
                    if not transient or attempt == self.max_retries:
                        break  # schema problem or exhausted retries: next strategy
                    log.debug("Gemini transient error, retrying (%s): %s", attempt + 1, exc)

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
    cache: DiskCache | None = None,
) -> LLMClient:
    """Factory. Returns a NullLLMClient when no key is available."""
    if not api_key:
        log.info("No LLM API key configured; running in deterministic-only mode.")
        return NullLLMClient()
    if provider.lower() in {"gemini", "google"}:
        return GeminiClient(api_key, model, timeout=timeout, max_retries=max_retries, cache=cache)
    log.warning("Unknown LLM provider %r; falling back to deterministic-only.", provider)
    return NullLLMClient()
