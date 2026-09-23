"""Scoring: weights, differentiation, penalties, clamping."""

from src.config import get_settings
from src.eligibility import evaluate_eligibility
from src.extract.deterministic import deterministic_extract
from src.schemas import Document, GitHubEnrichment
from src.scoring import derive_strengths_concerns, score_candidate

SETTINGS = get_settings()

AGENTIC = """Asha Rao
Skills: Python, FastAPI, PostgreSQL, Redis, Docker, GCP, LangGraph, pgvector
Experience
Backend Intern - async FastAPI services with Redis caching, unit tests and observability.
Projects
Agentic Triage
Multi-agent LangGraph workflow with retrieval over pgvector embeddings, tool calling,
persistent state and an evaluation harness. Dockerized on Cloud Run."""

THIN = """Bob Kim
Skills: Python, OpenAI API, LangChain
Projects
Chatbot
A chatbot that calls the OpenAI API to answer questions."""


def _score(text: str):
    doc = Document(source_file="x", filename="x.pdf", text=text)
    extracted = deterministic_extract(doc)
    elig = evaluate_eligibility(extracted, has_text=True)
    return score_candidate(extracted, elig, GitHubEnrichment(), SETTINGS)


def test_weights_sum_to_100():
    assert SETTINGS.weights.total() == 100.0


def test_agentic_scores_higher_than_thin_wrapper():
    assert _score(AGENTIC).total() > _score(THIN).total()


def test_thin_wrapper_is_penalised():
    assert _score(THIN).penalties < 0


def test_thin_wrapper_ai_depth_is_low():
    assert _score(THIN).ai_project_depth < 0.5 * SETTINGS.weights.ai_project_depth


def test_total_is_clamped_to_0_100():
    from src.schemas import ScoreBreakdown

    assert ScoreBreakdown(penalties=-500).total() == 0.0
    assert ScoreBreakdown(**{k: 100 for k in
        ["ai_project_depth", "python_backend", "cloud_fullstack", "github", "engineering_depth"]}).total() == 100.0


def test_breakdown_has_rationale_per_category():
    b = _score(AGENTIC)
    for category in ["ai_project_depth", "python_backend", "cloud_fullstack",
                     "github", "engineering_depth", "penalties"]:
        assert b.rationale.get(category), f"missing rationale for {category}"


def test_ineligible_candidate_is_not_scored():
    doc = Document(source_file="x", filename="x", text="Java, React, Spring Boot only.")
    extracted = deterministic_extract(doc)
    elig = evaluate_eligibility(extracted, has_text=True)
    b = score_candidate(extracted, elig, GitHubEnrichment(), SETTINGS)
    assert b.total() == 0.0
    assert "eligibility" in b.rationale


def test_strengths_and_concerns_are_populated_for_strong_candidate():
    doc = Document(source_file="x", filename="x", text=AGENTIC)
    extracted = deterministic_extract(doc)
    elig = evaluate_eligibility(extracted, has_text=True)
    b = score_candidate(extracted, elig, GitHubEnrichment(), SETTINGS)
    strengths, concerns = derive_strengths_concerns(extracted, b, GitHubEnrichment(), SETTINGS)
    assert any("AI" in s for s in strengths)
