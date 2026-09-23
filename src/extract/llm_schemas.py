"""Pydantic models for the LLM's structured output.

These are passed to the provider as a response schema AND used to validate the
returned JSON locally, so a malformed model response degrades gracefully
instead of corrupting downstream scoring.

Note: the schema is intentionally flat-friendly (lists of primitives, nested
object with booleans) to maximise compatibility across providers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import AIProjectSignals, ProjectDomain


class LLMProject(BaseModel):
    name: str = Field(description="Short project title as written on the resume.")
    description: str = Field(description="2-4 sentence faithful summary of what was built.")
    technologies: list[str] = Field(default_factory=list)
    domain: ProjectDomain = ProjectDomain.UNKNOWN

    # Rubric signals — each must be justified by the description.
    signals: AIProjectSignals = Field(default_factory=AIProjectSignals)

    quality_score: float = Field(
        default=0.0, ge=0, le=10,
        description="Depth as a real system (see rubric in the prompt).",
    )
    quality_rationale: str = Field(default="", description="One sentence citing resume evidence.")
    cited_evidence: str = Field(
        default="", description="Short verbatim snippet from the resume supporting the score."
    )
    is_thin_wrapper: bool = Field(
        default=False,
        description="True if this is only a single LLM/API call with no meaningful workflow.",
    )
    is_tutorial: bool = Field(
        default=False,
        description="True if it mirrors a well-known tutorial with no evidence of ownership.",
    )


class LLMExtraction(BaseModel):
    candidate_name: str = ""
    skills: list[str] = Field(default_factory=list)
    projects: list[LLMProject] = Field(default_factory=list)
    experience_highlights: list[str] = Field(default_factory=list)


class LLMProjectAssessment(BaseModel):
    """Response schema for the project-scoring-only LLM call.

    Smaller than LLMExtraction: the deterministic pass already extracted the
    candidate fields, so the model only judges project quality.
    """

    projects: list[LLMProject] = Field(default_factory=list)
