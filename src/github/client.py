"""GitHub REST client (lightweight enrichment).

Behaviour required by the spec:
  * public API only, token from env (raises rate limit 60 -> 5000/hr)
  * never fatal: every failure maps to a GitHubStatus and returns 0 points
  * cache responses to avoid repeat calls within/between runs
  * in-run circuit breaker: once rate-limited, stop calling and mark the rest

We fetch at most three cheap endpoints per candidate:
  /users/{u}                   -> profile (public repo count, existence)
  /users/{u}/events/public     -> recent activity
  /users/{u}/repos?sort=pushed -> maintained/relevant repositories
"""

from __future__ import annotations

import logging

import httpx

from ..cache import DiskCache
from ..config import Settings
from ..schemas import GitHubEnrichment, GitHubStatus
from .scoring import enrich_from_signals

log = logging.getLogger(__name__)

API = "https://api.github.com"
NOT_FOUND_USERNAMES = {"", "none", "na", "n/a", "username"}


class GitHubClient:
    def __init__(self, settings: Settings, cache: DiskCache | None = None) -> None:
        self.settings = settings
        self.cache = cache
        self._rate_limited = False  # in-run circuit breaker
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ai-resume-screener",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        self._client = httpx.Client(
            base_url=API,
            headers=headers,
            timeout=settings.request_timeout_seconds,
        )

    # -- low level ---------------------------------------------------------
    def _get(self, path: str) -> tuple[int, object | None]:
        """Return (status_code, json). Raises only on transport errors."""
        if self.cache:
            cached = self.cache.get("github", {"path": path})
            if cached is not None:
                return cached["status"], cached["data"]
        resp = self._client.get(path)
        data = resp.json() if resp.content else None
        if self.cache and resp.status_code == 200:
            self.cache.set("github", {"path": path}, {"status": resp.status_code, "data": data})
        return resp.status_code, data

    # -- public ------------------------------------------------------------
    def enrich(self, username: str | None) -> GitHubEnrichment:
        if not username or username.lower() in NOT_FOUND_USERNAMES:
            return GitHubEnrichment(status=GitHubStatus.NO_PROFILE, summary="No GitHub profile on resume")

        if self._rate_limited:
            return GitHubEnrichment(
                status=GitHubStatus.RATE_LIMITED,
                username=username,
                summary="GitHub rate limit already hit this run",
            )

        try:
            status, profile = self._get(f"/users/{username}")
        except Exception as exc:  # noqa: BLE001 - transport boundary
            log.warning("GitHub transport error for %s: %s", username, exc)
            return GitHubEnrichment(status=GitHubStatus.ERROR, username=username, error=str(exc),
                                    summary="GitHub request failed")

        if status == 404:
            return GitHubEnrichment(status=GitHubStatus.NOT_FOUND, username=username,
                                    summary="GitHub profile not found")
        if status in (403, 429):
            self._rate_limited = True
            log.warning("GitHub rate limited (status %s); disabling further calls", status)
            return GitHubEnrichment(status=GitHubStatus.RATE_LIMITED, username=username,
                                    summary="GitHub API rate limited")
        if status != 200 or not isinstance(profile, dict):
            return GitHubEnrichment(status=GitHubStatus.ERROR, username=username,
                                    error=f"status {status}", summary="GitHub profile unavailable")

        events: list[dict] = []
        repos: list[dict] = []
        try:
            ev_status, ev_data = self._get(f"/users/{username}/events/public?per_page=100")
            if ev_status == 200 and isinstance(ev_data, list):
                events = ev_data
            repo_status, repo_data = self._get(
                f"/users/{username}/repos?sort=pushed&direction=desc&per_page=30"
            )
            if repo_status == 200 and isinstance(repo_data, list):
                repos = repo_data
        except Exception as exc:  # noqa: BLE001
            log.warning("GitHub secondary fetch failed for %s: %s", username, exc)

        return enrich_from_signals(
            username=username,
            profile=profile,
            events=events,
            repos=repos,
            th=self.settings.thresholds,
        )

    def close(self) -> None:
        self._client.close()
