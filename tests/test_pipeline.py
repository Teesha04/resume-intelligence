"""Batch pipeline: counts, ordering, isolation, determinism."""

from pathlib import Path

from src.cache import NullCache
from src.config import get_settings
from src.extract import NullLLMClient
from src.pipeline import run_screening
from src.schemas import GitHubEnrichment, GitHubStatus

ELIGIBLE = """Asha Rao
asha@example.com
Skills: Python, FastAPI, LangGraph, PostgreSQL, Docker
Projects
Agent
Multi-agent LangGraph workflow with retrieval via embeddings, tool calling and state."""

REJECTED = """John Roe
john@example.com
Skills: JavaScript, React, Next.js, Java
Projects
Site
React storefront with Stripe."""


class FakeGitHub:
    """Offline GitHub stub: always 'no profile'."""

    def enrich(self, username):
        return GitHubEnrichment(status=GitHubStatus.NO_PROFILE, username=username,
                                summary="No GitHub profile on resume")

    def close(self):
        pass


def _build_set(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text(ELIGIBLE, encoding="utf-8")
    (tmp_path / "b.txt").write_text(REJECTED, encoding="utf-8")
    (tmp_path / "corrupt.pdf").write_bytes(b"%PDF-1.4 broken \x00")
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    (tmp_path / "a_copy.txt").write_text(ELIGIBLE, encoding="utf-8")  # duplicate
    return tmp_path


def _run(tmp_path: Path):
    return run_screening(
        tmp_path,
        get_settings(),
        llm_client=NullLLMClient(),
        github_client=FakeGitHub(),
        cache=NullCache(),
    )


def test_batch_counts_and_isolation(tmp_path: Path):
    report = _run(_build_set(tmp_path))
    s = report.summary
    assert s.total_resumes == 5
    assert s.parsed_ok == 2           # a, b (a_copy is a duplicate, not OK)
    assert s.duplicates_skipped == 1
    assert s.parse_failed == 2         # corrupt pdf + empty txt
    assert s.eligible == 1
    assert s.rejected == 4


def test_eligible_ranked_first(tmp_path: Path):
    report = _run(_build_set(tmp_path))
    assert report.candidates[0].eligible is True
    assert report.candidates[0].rank == 1
    assert report.candidates[0].candidate_name == "Asha Rao"


def test_unparsed_candidates_have_no_rank(tmp_path: Path):
    report = _run(_build_set(tmp_path))
    for c in report.candidates:
        if not c.eligible:
            assert c.rank is None
            assert c.rejection_reasons


def test_batch_is_deterministic(tmp_path: Path):
    r1 = _run(tmp_path)
    r2 = _run(tmp_path)
    scores1 = [(c.candidate_name, c.total_score) for c in r1.candidates]
    scores2 = [(c.candidate_name, c.total_score) for c in r2.candidates]
    assert scores1 == scores2
