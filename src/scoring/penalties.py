"""Project-quality penalties (spec section 4, bottom).

Two deductions:
  * Thin wrapper: an "AI project" that is really a single LLM/API call with no
    retrieval, orchestration, tools, evaluation, state or backend workflow.
    Scaled 5-15 points by how shallow it is. Only applies to projects that
    actually invoke an LLM/API (a classical CV model is not a "wrapper").
  * Tutorial / undetailed: named like a tutorial, or listed without
    implementation detail.

Penalties are capped so a bad set of projects can't drive a score below zero
(the final total is additionally clamped to [0, 100] in ScoreBreakdown).
"""

from __future__ import annotations

from ..config import Penalties, Thresholds
from ..schemas import ExtractedResume, Project
from .rubric import (
    _ai_projects,
    deterministic_signals,
    has_meaningful_workflow,
    llm_model_present,
    project_quality,
)

TUTORIAL_MARKERS = ("tutorial", "follow-along", "follow along", "clone of", "from youtube", "course project")
UNDETAILED_WORD_LIMIT = 15


def _is_thin(p: Project, th: Thresholds) -> bool:
    """A bare LLM/API call with no meaningful workflow."""
    if p.is_thin_wrapper:  # LLM said so explicitly
        return True
    signals = p.signals if p.quality_score is not None else deterministic_signals(p)
    return llm_model_present(p) and not has_meaningful_workflow(signals)


def _is_tutorial_or_undetailed(p: Project) -> tuple[bool, str]:
    if p.is_tutorial:
        return True, "tutorial/boilerplate"
    blob = f"{p.name} {p.description}".lower()
    if any(marker in blob for marker in TUTORIAL_MARKERS):
        return True, "tutorial-style"
    if len(p.description.split()) < UNDETAILED_WORD_LIMIT:
        return True, "listed without implementation detail"
    return False, ""


def _penalty_for_thin(q: float, pen: Penalties) -> float:
    """Deeper into the 'thin' range => larger deduction (5..15)."""
    depth_fraction = max(0.0, min(1.0, q / 10.0))
    return pen.thin_wrapper_min + (1.0 - depth_fraction) * (pen.thin_wrapper_max - pen.thin_wrapper_min)


def compute_penalties(
    extracted: ExtractedResume, th: Thresholds, pen: Penalties
) -> tuple[float, list[str]]:
    total = 0.0
    reasons: list[str] = []

    for p in _ai_projects(extracted):
        q = project_quality(p)

        if _is_thin(p, th):
            deduction = _penalty_for_thin(q, pen)
            total -= deduction
            reasons.append(
                f"Thin LLM/API wrapper, no meaningful workflow: {p.name or 'untitled'} (-{deduction:.1f})"
            )

        is_tut, label = _is_tutorial_or_undetailed(p)
        if is_tut:
            total -= pen.tutorial_listed
            reasons.append(f"{label}: {p.name or 'untitled'} (-{pen.tutorial_listed:.1f})")

    if total < -pen.max_total:
        reasons.append(f"Penalties capped at -{pen.max_total:.1f}")
        total = -pen.max_total

    if not reasons:
        reasons.append("No project-quality penalties")
    return round(total, 2), reasons
