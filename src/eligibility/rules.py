"""Hard eligibility filter — deterministic, no LLM (spec section 3).

A candidate is eligible iff BOTH hold:
  1. Python evidence exists (skill, project tech, work tech, impl language).
  2. AI/agentic evidence exists: a core LLM/agentic term, or an actual AI
     project (agentic or general ML/CV/NLP).

Deliberately NOT rejected for also knowing JS/Java/React — those are fine
as long as Python + AI are present.

Design notes:
  * Kept out of the LLM so eligibility is predictable and unit-testable.
  * We use evidence objects (term + source + context) so rejections can cite
    exactly what was/wasn't found.
"""

from __future__ import annotations

from ..schemas import Eligibility, Evidence, EvidenceSource, ExtractedResume
from ..vocab import AGENTIC_CORE_TERMS, GENERAL_AI_TERMS


def _python_matches(extracted: ExtractedResume) -> list[Evidence]:
    return list(extracted.python_evidence)


def _agentic_matches(extracted: ExtractedResume) -> list[Evidence]:
    """Core LLM/agentic signals found in the raw resume text (any section)."""
    return [e for e in extracted.ai_evidence if e.term in AGENTIC_CORE_TERMS]


def _general_ai_in_project(extracted: ExtractedResume) -> list[Evidence]:
    """General ML/CV/NLP terms, but only where attached to a project or job.

    A classical-ML *project* counts as an AI project; the same word appearing
    only in a skills list or coursework does not.
    """
    return [
        e for e in extracted.ai_evidence
        if e.term in GENERAL_AI_TERMS
        and e.source in (EvidenceSource.PROJECT, EvidenceSource.EXPERIENCE)
    ]


def evaluate_eligibility(extracted: ExtractedResume, *, has_text: bool = True) -> Eligibility:
    """Return an Eligibility verdict with explicit rejection reasons.

    IMPORTANT: this is deliberately based only on `python_evidence` and
    `ai_evidence`, which come from deterministic scanning of the *raw resume
    text*. It never consults LLM-produced projects, so eligibility cannot be
    influenced by the model (spec section 6).
    """
    rejection_reasons: list[str] = []

    python_ev = _python_matches(extracted)
    agentic_ev = _agentic_matches(extracted)
    general_project_ev = _general_ai_in_project(extracted)

    has_python = bool(python_ev)
    # AI evidence = a core agentic/LLM signal, or a genuine AI/ML project.
    has_ai = bool(agentic_ev) or bool(general_project_ev)

    if not has_text:
        rejection_reasons.append("No extractable resume text")
    if not has_python:
        rejection_reasons.append("No evidence of Python stack")
    if not has_ai:
        rejection_reasons.append("No AI/agentic project evidence")

    eligible = has_python and has_ai and has_text

    return Eligibility(
        eligible=eligible,
        rejection_reasons=rejection_reasons,
        matched_skills=list(extracted.skills),
        has_python=has_python,
        has_ai=has_ai,
    )
