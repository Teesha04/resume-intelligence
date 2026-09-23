"""Extraction orchestration: deterministic pass, then optional LLM refinement.

Contract:
  * `extract_resume` NEVER raises. LLM/validation failures are recorded as
    warnings and the deterministic result is returned instead.
  * Deterministic fields (email, GitHub) win when present, because regexes are
    more reliable than a model for those.
  * The LLM wins for semantic fields (projects, skills breadth, quality
    signals) because that is where its judgement adds value.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from ..schemas import AIProjectSignals, Document, ExtractedResume, Project, ProjectDomain
from .client import LLMClient, LLMError, NullLLMClient
from .deterministic import deterministic_extract
from .llm_schemas import LLMExtraction, LLMProjectAssessment
from .prompts import (
    EXTRACTION_SYSTEM,
    PROJECT_SCORING_SYSTEM,
    build_extraction_prompt,
    build_project_scoring_prompt,
)
from .sections import split_sections

log = logging.getLogger(__name__)


def _to_project(p) -> Project:
    return Project(
        name=p.name.strip()[:120],
        description=p.description.strip()[:1500],
        technologies=sorted({t.strip() for t in p.technologies if t.strip()}),
        domain=p.domain if isinstance(p.domain, ProjectDomain) else ProjectDomain.UNKNOWN,
        quality_score=float(p.quality_score),
        quality_rationale=p.quality_rationale or None,
        cited_evidence=(getattr(p, "cited_evidence", "") or None),
        signals=p.signals if isinstance(p.signals, AIProjectSignals) else AIProjectSignals(),
        is_thin_wrapper=bool(p.is_thin_wrapper),
        is_tutorial=bool(p.is_tutorial),
    )


def build_scoring_context(text: str, max_chars: int = 6000) -> str:
    """Text passed to the project-scoring LLM call.

    Prefer the PROJECTS + EXPERIENCE sections (where AI work lives) rather than
    the whole resume — far fewer tokens. Falls back to the full text when those
    sections aren't detected, so we never lose projects entirely.
    """
    sections = split_sections(text)
    parts: list[str] = []
    if sections.get("projects", "").strip():
        parts.append("PROJECTS\n" + sections["projects"])
    if sections.get("experience", "").strip():
        parts.append("EXPERIENCE\n" + sections["experience"])
    context = "\n\n".join(parts).strip() or text
    return context[:max_chars]


def _merge(base: ExtractedResume, llm: LLMExtraction) -> ExtractedResume:
    name = (llm.candidate_name or "").strip()
    if not name or len(name.split()) > 5 or any(ch.isdigit() for ch in name):
        name = base.candidate_name

    skills = list(dict.fromkeys([*base.skills, *[s.strip() for s in llm.skills if s.strip()]]))

    if llm.projects:
        projects = [_to_project(p) for p in llm.projects]
    else:
        projects = base.projects

    experience = llm.experience_highlights or base.experience

    return base.model_copy(
        update={
            "candidate_name": name,
            "skills": skills,
            "projects": projects,
            "experience": experience,
            "extraction_method": "hybrid",
        }
    )


def extract_resume(doc: Document, client: LLMClient | None = None) -> ExtractedResume:
    """Extract candidate info from a parsed document. Never raises.

    Deterministic extraction always runs; the LLM optionally refines it. In the
    pipeline, eligibility is decided on the deterministic result BEFORE this
    refinement is applied to survivors.
    """
    base = deterministic_extract(doc)
    if not doc.text or not doc.text.strip():
        base.warnings.append("no_text_to_extract")
        return base
    return refine_with_llm(base, doc.text, client)


def refine_with_llm(
    base: ExtractedResume, text: str, client: LLMClient | None
) -> ExtractedResume:
    """Apply LLM extraction/scoring on top of a deterministic result.

    Never raises: any failure is recorded as a warning and the deterministic
    result is returned unchanged.
    """
    if client is None or isinstance(client, NullLLMClient) or not text.strip():
        return base

    try:
        data = client.generate_json(
            system=EXTRACTION_SYSTEM,
            prompt=build_extraction_prompt(text),
            schema=LLMExtraction,
            temperature=0.0,
        )
        llm = LLMExtraction.model_validate(data)
    except (LLMError, ValidationError) as exc:
        log.warning("LLM refinement failed: %s", exc)
        base.warnings.append(f"llm_extraction_failed:{type(exc).__name__}")
        return base

    return _merge(base, llm)


def score_projects_with_llm(
    base: ExtractedResume, text: str, client: LLMClient | None
) -> ExtractedResume:
    """Score projects with the LLM, keeping deterministic fields as the base.

    This is the pipeline's LLM step for gate survivors. It sends only the
    PROJECTS + EXPERIENCE context (not the whole resume), so the prompt is much
    smaller and the call is cheaper/faster. Never raises: on failure the
    deterministic result is returned with a warning.
    """
    if client is None or isinstance(client, NullLLMClient) or not text.strip():
        return base

    context = build_scoring_context(text)
    if not context.strip():
        return base

    try:
        data = client.generate_json(
            system=PROJECT_SCORING_SYSTEM,
            prompt=build_project_scoring_prompt(context),
            schema=LLMProjectAssessment,
            temperature=0.0,
        )
        assessment = LLMProjectAssessment.model_validate(data)
    except (LLMError, ValidationError) as exc:
        log.warning("LLM project scoring failed: %s", exc)
        base.warnings.append(f"llm_scoring_failed:{type(exc).__name__}")
        return base

    if not assessment.projects:
        # Nothing scored; keep deterministic projects (and their heuristic quality).
        return base.model_copy(update={"extraction_method": "hybrid"})

    projects = [_to_project(p) for p in assessment.projects]
    return base.model_copy(
        update={"projects": projects, "extraction_method": "hybrid"}
    )
