"""Pydantic schemas: the single source of truth for data flowing through
the pipeline.

These models serve three purposes:
  1. In-process contracts between pipeline stages.
  2. The JSON schema we hand to the LLM for structured output.
  3. The shape of the final results.json.

Design note: eligibility and score *components* are plain numbers computed
outside the LLM. The LLM only ever fills fields that require semantic
judgement (e.g. project quality), and every such field carries a rationale.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class ParseStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"            # parsed but no usable text
    FAILED = "failed"          # extractor raised
    DUPLICATE = "duplicate"    # same content hash as an earlier file


class Document(BaseModel):
    """A single ingested file, before any interpretation."""

    source_file: str
    filename: str
    text: str = ""
    page_count: int | None = None
    parse_status: ParseStatus = ParseStatus.OK
    error: str | None = None
    content_hash: str = ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
class EvidenceSource(str, Enum):
    """Where a piece of evidence was found. Drives confidence weighting."""

    SKILLS = "skills"          # bare skills list (weakest)
    PROJECT = "project"        # a described project (strong)
    EXPERIENCE = "experience"  # internship / work (strong)
    SUMMARY = "summary"        # profile summary / other prose
    UNKNOWN = "unknown"


class Evidence(BaseModel):
    """A matched term plus the context that justifies crediting it."""

    term: str
    source: EvidenceSource = EvidenceSource.UNKNOWN
    context: str = ""  # short verbatim-ish snippet from the resume


# Maps a canonical resume section to the confidence source of its evidence.
# Lives here (dependency-free) so both extract/ and matching.py can use it
# without an import cycle.
SECTION_TO_SOURCE: dict[str, EvidenceSource] = {
    "summary": EvidenceSource.SUMMARY,
    "skills": EvidenceSource.SKILLS,
    "projects": EvidenceSource.PROJECT,
    "experience": EvidenceSource.EXPERIENCE,
    "education": EvidenceSource.UNKNOWN,
    "achievements": EvidenceSource.UNKNOWN,
    "other": EvidenceSource.UNKNOWN,
}


class ProjectDomain(str, Enum):
    AI_AGENTIC = "ai_agentic"
    AI_GENERAL = "ai_general"
    WEB_FULLSTACK = "web_fullstack"
    DATA = "data"
    OTHER = "other"
    UNKNOWN = "unknown"


class AIProjectSignals(BaseModel):
    """Rubric dimensions for an AI project (each 0/1 as judged with evidence)."""

    retrieval: bool = False        # RAG, vector search, embeddings
    tool_use: bool = False         # tool/function calling
    state_or_memory: bool = False  # agent state, memory, sessions
    orchestration: bool = False    # multi-agent, LangGraph, ADK, DAGs
    evaluation: bool = False       # eval harness, metrics, guardrails
    product_logic: bool = False    # real business logic around the model
    backend_logic: bool = False    # APIs, queues, persistence, async
    data_processing: bool = False  # ETL, chunking, cleaning, pipelines

    def count(self) -> int:
        return sum(bool(v) for v in self.model_dump().values())


class Project(BaseModel):
    """A project parsed from the resume, optionally quality-judged by the LLM."""

    name: str = ""
    description: str = ""

    technologies: list[str] = Field(default_factory=list)
    domain: ProjectDomain = ProjectDomain.UNKNOWN

    # --- LLM-judged, evidence-backed (None if deterministic-only mode) ---
    quality_score: float | None = Field(
        default=None, ge=0, le=10,
        description="0-10 depth score for the project as a real system.",
    )
    quality_rationale: str | None = None
    signals: AIProjectSignals = Field(default_factory=AIProjectSignals)
    is_thin_wrapper: bool = False
    is_tutorial: bool = False


class ExtractedResume(BaseModel):
    """Structured candidate information extracted from one resume."""

    candidate_name: str = "Unknown Candidate"
    email: str | None = None
    github_url: str | None = None
    github_username: str | None = None

    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)

    # Evidence collected during extraction (used by eligibility + scoring).
    python_evidence: list[Evidence] = Field(default_factory=list)
    ai_evidence: list[Evidence] = Field(default_factory=list)
    other_evidence: list[Evidence] = Field(default_factory=list)

    extraction_method: str = "deterministic"  # or "llm"
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
class Eligibility(BaseModel):
    eligible: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    has_python: bool = False
    has_ai: bool = False


# ---------------------------------------------------------------------------
# GitHub enrichment
# ---------------------------------------------------------------------------
class GitHubStatus(str, Enum):
    OK = "ok"
    NO_PROFILE = "no_profile"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


class GitHubEnrichment(BaseModel):
    status: GitHubStatus = GitHubStatus.DISABLED
    username: str | None = None
    summary: str = ""
    recent_activity_score: float = 0.0   # 0-5
    repos_score: float = 0.0             # 0-5
    public_repos: int | None = None
    days_since_last_activity: int | None = None
    relevant_repos: list[str] = Field(default_factory=list)
    error: str | None = None

    def score(self) -> float:
        return round(self.recent_activity_score + self.repos_score, 2)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
class ScoreBreakdown(BaseModel):
    """Per-category points. `penalties` is <= 0; `total` is clamped to 0-100."""

    ai_project_depth: float = 0.0
    python_backend: float = 0.0
    cloud_fullstack: float = 0.0
    github: float = 0.0
    engineering_depth: float = 0.0
    penalties: float = 0.0

    # Evidence strings keyed by category, for explainability.
    rationale: dict[str, list[str]] = Field(default_factory=dict)

    def total(self) -> float:
        raw = (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
            + self.penalties
        )
        return round(max(0.0, min(100.0, raw)), 2)


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------
class CandidateResult(BaseModel):
    """One row of results.json."""

    rank: int | None = None
    candidate_name: str = "Unknown Candidate"
    eligible: bool = False
    total_score: float | None = None
    score_breakdown: ScoreBreakdown | None = None

    matched_skills: list[str] = Field(default_factory=list)
    project_summary: str = ""
    github_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)

    github: GitHubEnrichment = Field(default_factory=GitHubEnrichment)

    # Provenance / debugging
    source_file: str = ""
    parse_status: ParseStatus = ParseStatus.OK
    warnings: list[str] = Field(default_factory=list)


class BatchSummary(BaseModel):
    """Top-level batch counters required by the spec."""

    total_resumes: int = 0
    parsed_ok: int = 0
    parse_failed: int = 0
    duplicates_skipped: int = 0
    eligible: int = 0
    rejected: int = 0
    llm_used: bool = False
    github_enriched: int = 0
    github_failures: int = 0
    duration_seconds: float = 0.0


class ScreeningReport(BaseModel):
    """The complete output document."""

    summary: BatchSummary = Field(default_factory=BatchSummary)
    candidates: list[CandidateResult] = Field(default_factory=list)
