"""Scoring orchestration.

`score_candidate` assembles the ScoreBreakdown and, crucially, attaches the
per-category rationale so the final JSON explains every point.

Ineligible candidates are not scored (spec section 2: "Score only eligible
candidates"), but we still return an empty breakdown with the rejection
reasons recorded for transparency.
"""

from __future__ import annotations

from ..config import Settings
from ..schemas import Eligibility, ExtractedResume, GitHubEnrichment, ScoreBreakdown
from .penalties import compute_penalties
from .rubric import (
    score_ai_project_depth,
    score_cloud_fullstack,
    score_engineering_depth,
    score_github,
    score_python_backend,
)


def score_candidate(
    extracted: ExtractedResume,
    eligibility: Eligibility,
    github: GitHubEnrichment,
    settings: Settings,
) -> ScoreBreakdown:
    w = settings.weights
    th = settings.thresholds

    if not eligibility.eligible:
        return ScoreBreakdown(
            rationale={"eligibility": eligibility.rejection_reasons or ["Ineligible"]}
        )

    ai_pts, ai_reason, _ = score_ai_project_depth(extracted, w, th)
    py_pts, py_reason = score_python_backend(extracted, w, th)
    cl_pts, cl_reason = score_cloud_fullstack(extracted, w, th)
    en_pts, en_reason = score_engineering_depth(extracted, w, th)
    gh_pts, gh_reason = score_github(github.score(), w)
    pen_pts, pen_reason = compute_penalties(extracted, th, settings.penalties)

    return ScoreBreakdown(
        ai_project_depth=ai_pts,
        python_backend=py_pts,
        cloud_fullstack=cl_pts,
        github=gh_pts,
        engineering_depth=en_pts,
        penalties=pen_pts,
        rationale={
            "ai_project_depth": ai_reason,
            "python_backend": py_reason,
            "cloud_fullstack": cl_reason,
            "engineering_depth": en_reason,
            "github": gh_reason,
            "penalties": pen_reason,
        },
    )


def derive_strengths_concerns(
    extracted: ExtractedResume,
    breakdown: ScoreBreakdown,
    github: GitHubEnrichment,
    settings: Settings,
) -> tuple[list[str], list[str]]:
    """Short human-readable strengths/concerns derived from the breakdown."""
    w = settings.weights
    strengths: list[str] = []
    concerns: list[str] = []

    if breakdown.ai_project_depth >= 0.6 * w.ai_project_depth:
        strengths.append("Strong AI/agentic project depth")
    elif breakdown.ai_project_depth <= 0.25 * w.ai_project_depth:
        concerns.append("Limited AI project depth")

    if breakdown.python_backend >= 0.6 * w.python_backend:
        strengths.append("Solid Python/backend engineering evidence")
    elif breakdown.python_backend <= 0.3 * w.python_backend:
        concerns.append("Weak Python/backend evidence")

    if breakdown.cloud_fullstack >= 0.6 * w.cloud_fullstack:
        strengths.append("Cloud/deployment and full-stack exposure")
    if breakdown.cloud_fullstack <= 0.2 * w.cloud_fullstack:
        concerns.append("Little cloud/deployment evidence")

    if breakdown.penalties < 0:
        concerns.append("Project-quality penalties applied (see score breakdown)")
    if github.status.value == "ok" and github.score() >= 5:
        strengths.append("Active, relevant public GitHub")
    elif github.username and github.status.value != "ok":
        concerns.append(f"GitHub not enriched ({github.status.value})")

    return strengths, concerns
