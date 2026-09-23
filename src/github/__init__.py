"""GitHub enrichment: activity + relevant repositories (capped at 10 points)."""

from .client import GitHubClient
from .scoring import days_since, enrich_from_signals, score_activity, score_repos

__all__ = [
    "GitHubClient",
    "enrich_from_signals",
    "score_activity",
    "score_repos",
    "days_since",
]
