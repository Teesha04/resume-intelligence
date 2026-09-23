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
from .llm_schemas import LLMExtraction
from .prompts import EXTRACTION_SYSTEM, build_extraction_prompt

log = logging.getLogger(__name__)


def _to_project(p) -> Project:
    return Project(
        name=p.name.strip()[:120],
        description=p.description.strip()[:1500],
        technologies=sorted({t.strip() for t in p.technologies if t.strip()}),
        domain=p.domain if isinstance(p.domain, ProjectDomain) else ProjectDomain.UNKNOWN,
        quality_score=float(p.quality_score),
        quality_rationale=p.quality_rationale or None,
        signals=p.signals if isinstance(p.signals, AIProjectSignals) else AIProjectSignals(),
        is_thin_wrapper=bool(p.is_thin_wrapper),
        is_tutorial=bool(p.is_tutorial),
    )


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
    """Extract candidate info from a parsed document. Never raises."""
    base = deterministic_extract(doc)

    if not doc.text or not doc.text.strip():
        base.warnings.append("no_text_to_extract")
        return base

    if client is None or isinstance(client, NullLLMClient):
        return base

    try:
        data = client.generate_json(
            system=EXTRACTION_SYSTEM,
            prompt=build_extraction_prompt(doc.text),
            schema=LLMExtraction,
            temperature=0.0,
        )
        llm = LLMExtraction.model_validate(data)
    except (LLMError, ValidationError) as exc:
        log.warning("LLM extraction failed for %s: %s", doc.filename, exc)
        base.warnings.append(f"llm_extraction_failed:{type(exc).__name__}")
        return base

    return _merge(base, llm)
