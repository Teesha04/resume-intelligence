"""Pure GitHub scoring — no network, fully unit-testable.

Capped at 10 points total (spec section 5):
  0-5  recent activity  (recency of events/commits, plus volume)
  0-5  maintained & relevant repositories (Python/AI repos updated recently)

Missing/private/rate-limited GitHub scores 0 and never fails screening.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import Thresholds
from ..matching import match_terms
from ..schemas import GitHubEnrichment, GitHubStatus
from ..vocab import AGENTIC_CORE_TERMS, AI_TERMS


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def days_since(value: str | None) -> int | None:
    dt = _parse_iso(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    return max(0, delta.days)


def score_activity(days_since_last: int | None, recent_event_count: int) -> float:
    """0-5 points for recent public activity."""
    if days_since_last is None:
        return 0.0
    if days_since_last <= 30:
        base = 4.0
    elif days_since_last <= 90:
        base = 3.0
    elif days_since_last <= 180:
        base = 2.0
    elif days_since_last <= 365:
        base = 1.0
    else:
        base = 0.5

    # Volume boost: sustained activity, not a single commit.
    if recent_event_count >= 5:
        base += 1.0
    elif recent_event_count >= 2:
        base += 0.5

    return round(min(5.0, base), 2)


def _repo_is_ai_related(repo: dict) -> bool:
    blob = " ".join(
        [
            repo.get("name") or "",
            repo.get("description") or "",
            " ".join(repo.get("topics") or []),
        ]
    )
    hits = set(match_terms(blob, AI_TERMS))
    return bool(hits & AGENTIC_CORE_TERMS) or repo.get("language") == "Python"


def score_repos(repos: list[dict], th: Thresholds) -> tuple[float, list[str]]:
    """0-5 points for maintained + relevant public repositories.

    Returns (score, names of relevant maintained repos).
    """
    active = 0
    relevant: list[str] = []

    for repo in repos:
        if repo.get("fork"):
            continue
        since = days_since(repo.get("pushed_at"))
        if since is None or since > th.github_repo_fresh_days:
            continue
        active += 1
        if _repo_is_ai_related(repo):
            relevant.append(repo.get("name") or "unnamed")

    score = 0.8 * active + 1.2 * len(relevant)
    return round(min(5.0, score), 2), relevant[:10]


def enrich_from_signals(
    *,
    username: str,
    profile: dict,
    events: list[dict],
    repos: list[dict],
    th: Thresholds,
) -> GitHubEnrichment:
    """Combine raw API payloads into a scored GitHubEnrichment."""
    pushed_dates = [e.get("created_at") for e in events if e.get("created_at")]
    # Fall back to repo push dates when the events feed is empty (common for
    # users who keep activity private).
    if not pushed_dates:
        pushed_dates = [r.get("pushed_at") for r in repos if r.get("pushed_at")]

    newest = max((d for d in pushed_dates if d), default=None)
    last_days = days_since(newest)

    recent_event_count = sum(
        1 for e in events if (days_since(e.get("created_at")) or 10**9) <= th.github_active_days
    )
    activity = score_activity(last_days, recent_event_count)
    repo_score, relevant = score_repos(repos, th)

    parts: list[str] = []
    if last_days is None:
        parts.append("no recent public activity")
    else:
        parts.append(f"last public activity ~{last_days}d ago")
    if recent_event_count:
        parts.append(f"{recent_event_count} event(s) in {th.github_active_days}d")
    if relevant:
        parts.append(f"relevant repos: {', '.join(relevant[:3])}")

    return GitHubEnrichment(
        status=GitHubStatus.OK,
        username=username,
        summary="; ".join(parts) or "no public signal",
        recent_activity_score=activity,
        repos_score=repo_score,
        public_repos=profile.get("public_repos"),
        days_since_last_activity=last_days,
        relevant_repos=relevant,
    )
