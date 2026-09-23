"""Scoring: the 100-point rubric, penalties, and explanation generation."""

from .penalties import compute_penalties
from .rubric import (
    deterministic_signals,
    estimate_project_quality,
    project_quality,
    score_ai_project_depth,
    score_cloud_fullstack,
    score_engineering_depth,
    score_github,
    score_python_backend,
)
from .scorer import derive_strengths_concerns, score_candidate

__all__ = [
    "score_candidate",
    "derive_strengths_concerns",
    "compute_penalties",
    "score_ai_project_depth",
    "score_python_backend",
    "score_cloud_fullstack",
    "score_engineering_depth",
    "score_github",
    "deterministic_signals",
    "estimate_project_quality",
    "project_quality",
]
