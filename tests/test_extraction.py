"""Extraction: deterministic fields and graceful LLM handling."""

from src.extract import NullLLMClient, extract_resume
from src.extract.client import LLMError
from src.extract.deterministic import deterministic_extract
from src.schemas import Document

RESUME = """Jane Doe
jane.doe@example.com
github.com/janedoe
Skills: Python, FastAPI, LangChain, PostgreSQL
Projects
RAG Assistant
Built a RAG pipeline with embeddings and vector search over docs, served via FastAPI."""


def _doc(text: str = RESUME) -> Document:
    return Document(source_file="jane.pdf", filename="jane.pdf", text=text)


def test_deterministic_core_fields():
    r = deterministic_extract(_doc())
    assert r.candidate_name == "Jane Doe"
    assert r.email == "jane.doe@example.com"
    assert r.github_url == "https://github.com/janedoe"
    assert r.github_username == "janedoe"
    assert "Python" in r.skills and "LangChain" in r.skills
    assert r.extraction_method == "deterministic"


def test_github_ignores_non_profile_paths():
    doc = _doc("Jane\njane@x.com\ngithub.com/features\nSkills: Python")
    assert deterministic_extract(doc).github_username is None


def test_name_falls_back_to_filename():
    doc = Document(source_file="resumes/teesha_shah_resume.pdf",
                   filename="teesha_shah_resume.pdf", text="Python developer\nSkills: Python")
    assert "Teesha" in deterministic_extract(doc).candidate_name


def test_null_client_returns_deterministic():
    r = extract_resume(_doc(), NullLLMClient())
    assert r.extraction_method == "deterministic"


def test_llm_failure_is_graceful():
    class Boom:
        name = "boom"

        def generate_json(self, **kwargs):
            raise LLMError("429 rate limited")

    r = extract_resume(_doc(), Boom())
    assert r.extraction_method == "deterministic"
    assert any("llm_extraction_failed" in w for w in r.warnings)


def test_llm_merge_prefers_model_projects():
    class Fake:
        name = "fake"

        def generate_json(self, *, system, prompt, schema=None, temperature=0.0):
            return {
                "candidate_name": "Jane Doe",
                "skills": ["Python", "LangGraph"],
                "projects": [{
                    "name": "Agentic Assistant",
                    "description": "LangGraph multi-agent with tool calling into an API.",
                    "technologies": ["Python", "LangGraph"],
                    "domain": "ai_agentic",
                    "signals": {"orchestration": True, "tool_use": True},
                    "quality_score": 8.0,
                    "quality_rationale": "Multi-agent orchestration with tools.",
                    "is_thin_wrapper": False,
                    "is_tutorial": False,
                }],
                "experience_highlights": [],
            }

    r = extract_resume(_doc(), Fake())
    assert r.extraction_method == "hybrid"
    assert r.projects[0].name == "Agentic Assistant"
    assert r.projects[0].quality_score == 8.0
    # deterministic regex fields are preserved
    assert r.email == "jane.doe@example.com"
