"""Hard, rule-based eligibility filtering (no LLM)."""

from .rules import evaluate_eligibility

__all__ = ["evaluate_eligibility"]
