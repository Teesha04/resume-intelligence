"""Eligibility: the hard filter must match the spec's intent exactly."""

from src.eligibility import evaluate_eligibility
from src.extract.deterministic import deterministic_extract
from src.schemas import Document

AGENTIC = """Asha Rao
asha@example.com
Skills: Python, FastAPI, LangGraph, PostgreSQL
Projects
Support Agent
Built a multi-agent RAG workflow with tool calling and pgvector retrieval."""

JS_ONLY = """John Roe
john@example.com
Skills: JavaScript, React, Next.js, Java, Spring Boot
Projects
Storefront
React + Spring Boot e-commerce site with Stripe."""

PY_ONLY = """Amy Lee
amy@example.com
Skills: Python, Django, PostgreSQL
Projects
Blog API
REST API in Django with PostgreSQL and JWT auth."""

PY_AND_JS_AND_AI = """Sam Ray
sam@example.com
Skills: JavaScript, React, Next.js, Python, LangChain
Projects
Docs Q&A
React frontend over a Python LangChain RAG service with embeddings."""

CV_ONLY_SKILL = """Uma Nair
uma@example.com
Skills: Python, PyTorch, OpenCV"""

CV_PROJECT = """Uma Nair
uma@example.com
Skills: Python, PyTorch, OpenCV
Projects
Defect Detector
Trained a CNN in PyTorch to classify manufacturing defects from images."""


def _elig(text: str, has_text: bool = True):
    doc = Document(source_file="x", filename="x.pdf", text=text)
    return evaluate_eligibility(deterministic_extract(doc), has_text=has_text)


def test_python_and_agentic_is_eligible():
    assert _elig(AGENTIC).eligible is True


def test_javascript_only_is_rejected():
    result = _elig(JS_ONLY)
    assert result.eligible is False
    assert "No evidence of Python stack" in result.rejection_reasons
    assert "No AI/agentic project evidence" in result.rejection_reasons


def test_python_only_no_ai_is_rejected():
    result = _elig(PY_ONLY)
    assert result.eligible is False
    assert result.has_python is True
    assert result.has_ai is False


def test_js_present_but_python_ai_satisfied_is_eligible():
    assert _elig(PY_AND_JS_AND_AI).eligible is True


def test_general_ai_project_with_python_is_eligible():
    assert _elig(CV_PROJECT).eligible is True


def test_general_ai_skill_without_project_is_not_ai_evidence():
    result = _elig(CV_ONLY_SKILL)
    assert result.eligible is False
    assert result.has_ai is False


def test_empty_text_rejected_with_reason():
    result = _elig("", has_text=False)
    assert result.eligible is False
    assert "No extractable resume text" in result.rejection_reasons


def test_matched_skills_present_for_rejected_candidate():
    result = _elig(JS_ONLY)
    assert "Java" in result.matched_skills
    assert "React" in result.matched_skills


def test_llm_produced_projects_do_not_affect_eligibility():
    """Eligibility must depend only on raw-text evidence, never LLM projects."""
    from src.schemas import Evidence, ExtractedResume, Project, ProjectDomain

    extracted = ExtractedResume(
        candidate_name="X",
        skills=["Python"],
        python_evidence=[Evidence(term="Python")],
        ai_evidence=[],  # no raw-text AI evidence
        projects=[Project(name="An AI agent", domain=ProjectDomain.AI_AGENTIC)],
    )
    result = evaluate_eligibility(extracted, has_text=True)
    assert result.has_python is True
    assert result.has_ai is False
    assert result.eligible is False
