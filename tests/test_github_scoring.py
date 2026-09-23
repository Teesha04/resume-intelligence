"""GitHub scoring is pure and testable without network."""

from datetime import datetime, timedelta, timezone

from src.config import get_settings
from src.github.scoring import enrich_from_signals, score_activity, score_repos
from src.schemas import GitHubStatus

TH = get_settings().thresholds


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_activity_recency_bands():
    assert score_activity(10, 0) > score_activity(120, 0) > score_activity(400, 0)
    assert score_activity(None, 0) == 0.0


def test_activity_volume_boost():
    assert score_activity(20, 6) > score_activity(20, 0)


def test_activity_capped_at_5():
    assert score_activity(1, 100) <= 5.0


def test_repos_relevance_and_freshness():
    repos = [
        {"name": "agent", "language": "Python", "pushed_at": _iso(10), "description": "RAG agent", "topics": [], "fork": False},
        {"name": "site", "language": "JavaScript", "pushed_at": _iso(10), "description": "web", "topics": [], "fork": False},
        {"name": "stale", "language": "Python", "pushed_at": _iso(900), "description": "old", "topics": [], "fork": False},
    ]
    score, relevant = score_repos(repos, TH)
    assert score > 0
    assert "agent" in relevant and "stale" not in relevant


def test_forks_are_ignored():
    repos = [{"name": "forked", "language": "Python", "pushed_at": _iso(5), "fork": True}]
    score, relevant = score_repos(repos, TH)
    assert score == 0.0 and relevant == []


def test_enrich_uses_repo_push_when_no_events():
    repos = [{"name": "p", "language": "Python", "pushed_at": _iso(5), "topics": [], "fork": False}]
    e = enrich_from_signals(username="u", profile={"public_repos": 1}, events=[], repos=repos, th=TH)
    assert e.status == GitHubStatus.OK
    assert e.recent_activity_score > 0
    assert e.score() <= 10


def test_total_github_capped_at_10():
    repos = [{"name": f"p{i}", "language": "Python", "pushed_at": _iso(1), "description": "llm agent",
              "topics": ["llm"], "fork": False} for i in range(20)]
    events = [{"created_at": _iso(1)} for _ in range(50)]
    e = enrich_from_signals(username="u", profile={"public_repos": 20}, events=events, repos=repos, th=TH)
    assert e.score() <= 10.0
