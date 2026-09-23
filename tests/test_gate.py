"""Gate-first ordering and GitHub tagging behaviour."""

from pathlib import Path

from src.cache import NullCache
from src.config import get_settings
from src.pipeline import run_screening
from src.schemas import GitHubEnrichment, GitHubStatus

ELIGIBLE = """Asha Rao
asha@example.com
Skills: Python, FastAPI, LangGraph, PostgreSQL
Projects
Agent
Multi-agent LangGraph workflow with retrieval via embeddings, tool calling and state."""

NO_PYTHON = """John Roe
john@example.com
Skills: JavaScript, React, Next.js, Java
Projects
Site
React storefront with Stripe."""

NO_AI = """Amy Lee
amy@example.com
Skills: Python, Django, PostgreSQL
Projects
Blog API
REST API in Django with PostgreSQL and JWT auth."""


class SpyLLM:
    name = "spy"

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def generate_json(self, *, system, prompt, schema=None, temperature=0.0):
        self.calls += 1
        self.prompts.append(prompt)
        return {
            "projects": [{
                "name": "Agent",
                "description": "LangGraph multi-agent with retrieval and tool calling.",
                "technologies": ["LangGraph"],
                "domain": "ai_agentic",
                "signals": {"orchestration": True, "retrieval": True, "tool_use": True},
                "quality_score": 8.0,
                "quality_rationale": "Multi-agent orchestration with retrieval.",
                "cited_evidence": "multi-agent LangGraph workflow with retrieval",
                "is_thin_wrapper": False,
                "is_tutorial": False,
            }],
        }


class SpyGitHub:
    def __init__(self, result: GitHubEnrichment | None = None):
        self.calls: list = []
        self._result = result

    def enrich(self, username):
        self.calls.append(username)
        if self._result is not None:
            return self._result
        return GitHubEnrichment(status=GitHubStatus.NO_PROFILE, username=username,
                                summary="No GitHub profile on resume")

    def close(self):
        pass


def _write(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_gate_eliminates_before_llm_and_github(tmp_path: Path):
    _write(tmp_path, "eligible.txt", ELIGIBLE)
    _write(tmp_path, "nopython.txt", NO_PYTHON)
    _write(tmp_path, "noai.txt", NO_AI)

    llm = SpyLLM()
    gh = SpyGitHub()
    report = run_screening(tmp_path, get_settings(), llm_client=llm,
                           github_client=gh, cache=NullCache())

    assert report.summary.eligible == 1
    # LLM was called exactly once — only for the eligible survivor.
    assert llm.calls == 1
    # GitHub was called exactly once (for the survivor), never for rejects.
    assert len(gh.calls) == 1
    assert report.summary.llm_skipped_by_gate == 2


def test_eliminated_candidates_are_marked_not_enriched(tmp_path: Path):
    _write(tmp_path, "nopython.txt", NO_PYTHON)
    report = run_screening(tmp_path, get_settings(), llm_client=SpyLLM(),
                           github_client=SpyGitHub(), cache=NullCache())
    c = report.candidates[0]
    assert c.eligible is False
    assert c.github.status == GitHubStatus.DISABLED
    assert "hard gate" in c.github.summary
    assert c.llm_applied is False
    assert c.rejection_reasons


def test_github_rate_limited_is_tagged_with_neutral_placeholder(tmp_path: Path):
    _write(tmp_path, "eligible.txt", ELIGIBLE)
    flagged = GitHubEnrichment(status=GitHubStatus.RATE_LIMITED, username="x",
                               summary="rate limited", flagged=True)
    report = run_screening(tmp_path, get_settings(), llm_client=SpyLLM(),
                           github_client=SpyGitHub(flagged), cache=NullCache())
    c = report.candidates[0]
    assert c.github.flagged is True
    # Neutral placeholder (5/10) — NOT nil, and not a genuine 0.
    assert c.score_breakdown.github == 5.0
    assert any("tagged" in r for r in c.score_breakdown.rationale["github"])
    assert report.summary.github_tagged_rate_limited == 1


def test_llm_applied_flag_and_warning_on_failure(tmp_path: Path):
    _write(tmp_path, "eligible.txt", ELIGIBLE)
    report = run_screening(tmp_path, get_settings(), llm_client=SpyLLM(),
                           github_client=SpyGitHub(), cache=NullCache())
    assert report.candidates[0].llm_applied is True
    assert report.summary.llm_scored == 1

    class Boom:
        name = "boom"

        def generate_json(self, **kwargs):
            from src.extract.client import LLMError

            raise LLMError("nope")

    report2 = run_screening(tmp_path, get_settings(), llm_client=Boom(),
                            github_client=SpyGitHub(), cache=NullCache())
    assert report2.candidates[0].llm_applied is False
    assert any(w.startswith(("llm_extraction_failed", "llm_scoring_failed"))
               for w in report2.candidates[0].warnings)


def test_llm_prompt_is_trimmed_to_project_context(tmp_path: Path):
    # A marker in a non-project section must NOT be sent to the LLM.
    _write(tmp_path, "eligible.txt", ELIGIBLE + "\nEducation\nSECRET_UNIVERSITY_MARKER\n")
    llm = SpyLLM()
    run_screening(tmp_path, get_settings(), llm_client=llm,
                  github_client=SpyGitHub(), cache=NullCache())
    assert llm.calls == 1
    prompt = llm.prompts[0]
    assert "PROJECTS" in prompt
    assert "SECRET_UNIVERSITY_MARKER" not in prompt


def test_llm_project_quality_and_evidence_flow_into_score(tmp_path: Path):
    _write(tmp_path, "eligible.txt", ELIGIBLE)
    report = run_screening(tmp_path, get_settings(), llm_client=SpyLLM(),
                           github_client=SpyGitHub(), cache=NullCache())
    c = report.candidates[0]
    ai_rationale = " ".join(c.score_breakdown.rationale["ai_project_depth"])
    assert "Multi-agent orchestration" in ai_rationale  # LLM rationale surfaced
