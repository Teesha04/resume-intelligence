"""Extraction: deterministic parsing plus optional LLM refinement."""

from .client import GeminiClient, LLMClient, LLMError, NullLLMClient, build_llm_client
from .deterministic import deterministic_extract
from .extractor import build_scoring_context, extract_resume, refine_with_llm, score_projects_with_llm
from .llm_schemas import LLMExtraction, LLMProject, LLMProjectAssessment
from .sections import line_assignments, split_sections

__all__ = [
    "deterministic_extract",
    "extract_resume",
    "refine_with_llm",
    "score_projects_with_llm",
    "build_scoring_context",
    "split_sections",
    "line_assignments",
    "LLMClient",
    "LLMError",
    "NullLLMClient",
    "GeminiClient",
    "build_llm_client",
    "LLMExtraction",
    "LLMProject",
    "LLMProjectAssessment",
]
