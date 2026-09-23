"""Tiny on-disk cache for network responses (LLM + GitHub).

Rationale (spec section 9): "Avoid repeated network calls when data can be
cached during the run." A content-addressed JSON cache also makes reruns
during development near-instant and deterministic, and lets us respect API
rate limits without extra bookkeeping.

Cache keys are a hash of the request payload, so identical resumes / usernames
never hit the network twice. Failures are never cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _path(self, namespace: str, key: str) -> Path:
        return self.dir / namespace / f"{key}.json"

    def get(self, namespace: str, payload: Any) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(namespace, self._key(payload))
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, namespace: str, payload: Any, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, self._key(payload))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, default=str), encoding="utf-8")
        except OSError as exc:  # cache is best-effort; never fatal
            log.debug("Cache write failed for %s: %s", namespace, exc)


class NullCache(DiskCache):
    """A cache that stores nothing, for tests or --no-cache."""

    def __init__(self) -> None:
        super().__init__(Path(".cache"), enabled=False)
