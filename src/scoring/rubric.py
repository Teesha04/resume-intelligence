"""The 100-point rubric, computed so it is explainable point-by-point.

Weights (spec section 4):
  ai_project_depth   40
  python_backend     30
  cloud_fullstack    15
  github             10
  engineering_depth   5
  penalties        -5..-15

Approach:
  * Deterministic, evidence-source-weighted scoring is ALWAYS computed.
  * When the LLM supplied a per-project `quality_score`, it refines
    `ai_project_depth`; otherwise a transparent heuristic estimates quality
    from rubric signals detected in the project text.
  * Every category returns (points, rationale_strings) so results.json can
    show *why* a candidate scored what they did.
"""

from __future__ import annotations

from ..config import Thresholds, Weights
from ..matching import match_terms
from ..schemas import (
    AIProjectSignals,
    Evidence,
    EvidenceSource,
    ExtractedResume,
    Project,
    ProjectDomain,
)
from ..vocab import (
    AGENTIC_CORE_TERMS,
    AI_TERMS,
    BACKEND_TERMS,
    CLOUD_TERMS,
    ENGINEERING_TERMS,
    PYTHON_TERMS,
)

# ---------------------------------------------------------------------------
# Term importance weights (relative, not points).
# Core language/framework terms count more than peripheral libraries.
# ---------------------------------------------------------------------------
PYTHON_BACKEND_WEIGHTS: dict[str, float] = {
    # Python core
    "Python": 2.0, "FastAPI": 1.8, "Django": 1.5, "Flask": 1.2, "asyncio": 1.2,
    "SQLAlchemy": 1.0, "Celery": 1.0, "Pydantic": 0.8, "pytest": 0.8,
    "Pandas": 0.8, "NumPy": 0.8, "scikit-learn": 0.8, "Streamlit": 0.6,
    "SciPy": 0.5, "uvicorn": 0.5, "BeautifulSoup": 0.5, "Selenium": 0.5,
    "Poetry": 0.4,
    # Backend / data stores
    "PostgreSQL": 1.5, "Redis": 1.2, "REST API": 1.0, "SQL": 0.8,
    "MySQL": 0.8, "MongoDB": 0.8, "Kafka": 0.8, "RabbitMQ": 0.8,
    "WebSockets": 0.6, "GraphQL": 0.6, "gRPC": 0.5,
}

CLOUD_WEIGHTS: dict[str, float] = {
    "Docker": 1.5, "GCP": 1.5, "AWS": 1.5, "Azure": 1.2, "Kubernetes": 1.2,
    "CI/CD": 1.0, "Terraform": 0.8, "Nginx": 0.5, "Serverless": 0.8,
    "Vercel": 0.5, "Render": 0.5, "Heroku": 0.5, "Firebase": 0.6,
    "Supabase": 0.6,
    # Full-stack is a *supporting* signal (lower weight).
    "React": 0.6, "Next.js": 0.5, "Tailwind": 0.3, "Streamlit": 0.3,
}

# Normalisation targets: reaching this weighted sum saturates the category.
PYTHON_BACKEND_TARGET = 7.0
CLOUD_TARGET = 4.0


def _source_multiplier(source: EvidenceSource, th: Thresholds) -> float:
    return {
        EvidenceSource.PROJECT: th.source_weight_project,
        EvidenceSource.EXPERIENCE: th.source_weight_experience,
        EvidenceSource.SKILLS: th.source_weight_skills,
        EvidenceSource.SUMMARY: 0.5,
        EvidenceSource.UNKNOWN: 0.6,
    }.get(source, 0.5)


def _best_source_per_term(evidence: list[Evidence], th: Thresholds) -> dict[str, float]:
    """Best (highest) source multiplier seen for each canonical term."""
    best: dict[str, float] = {}
    for ev in evidence:
        mult = _source_multiplier(ev.source, th)
        if mult > best.get(ev.term, 0.0):
            best[ev.term] = mult
    return best


def _weighted_terms_score(
    evidence: list[Evidence],
    term_weights: dict[str, float],
    th: Thresholds,
    target: float,
    cap: float,
) -> tuple[float, list[str]]:
    """Sum(term_weight * best_source_mult) normalised to `cap`, with evidence."""
    best = _best_source_per_term(evidence, th)
    total = 0.0
    rationale: list[str] = []
    for term, mult in best.items():
        w = term_weights.get(term)
        if not w:
            continue
        contrib = w * mult
        total += contrib
        if mult >= th.source_weight_project:
            where = "project"
        elif mult >= th.source_weight_experience:
            where = "experience"
        elif mult <= th.source_weight_skills:
            where = "skills (list-only)"
        else:
            where = "context"
        rationale.append(f"{term} ({where})")
    points = min(cap, (total / target) * cap) if target > 0 else 0.0
    return round(points, 2), rationale


# ---------------------------------------------------------------------------
# AI / agentic project depth (40)
# ---------------------------------------------------------------------------
AI_SIGNAL_TERMS = {
    "retrieval": {"RAG", "Embeddings", "Vector Search", "Pinecone", "FAISS", "Chroma", "Weaviate", "Qdrant", "pgvector"},
    # NB: LangChain/LlamaIndex are frameworks, not proof of orchestration.
    "orchestration": {"LangGraph", "CrewAI", "AutoGen", "Google ADK", "Multi-Agent", "AI Agents", "Semantic Kernel", "Haystack", "DSPy"},
    "tool_use": {"Tool Calling"},
    "evaluation": {"Evaluation", "LangSmith"},
    "model": {"LLM", "OpenAI API", "Anthropic", "Gemini", "Hugging Face", "Fine-tuning"},
}

# Terms indicating the project actually invokes an LLM/API (vs classical ML).
LLM_MODEL_SIGNALS: set[str] = {
    "LLM", "OpenAI API", "Anthropic", "Gemini", "Hugging Face", "Fine-tuning",
    "LangChain", "LlamaIndex", "MCP", "Prompt Engineering", "DSPy",
}


def llm_model_present(project: Project) -> bool:
    text = f"{project.name} {project.description} {' '.join(project.technologies)}"
    return bool(set(match_terms(text, AI_TERMS)) & LLM_MODEL_SIGNALS)


def has_meaningful_workflow(s: AIProjectSignals) -> bool:
    """True if the project shows real system depth (not just a bare API call)."""
    return bool(
        s.retrieval or s.orchestration or s.tool_use
        or s.evaluation or s.backend_logic or s.state_or_memory
    )


def deterministic_signals(project: Project) -> AIProjectSignals:
    """Estimate rubric signals from project text (used when no LLM)."""
    text = f"{project.name} {project.description} {' '.join(project.technologies)}"
    hits = set(match_terms(text, AI_TERMS))
    lowered = text.lower()

    def has_any(*words: str) -> bool:
        return any(word in lowered for word in words)

    # Backend evidence must be real terms, not a substring like the "api" inside
    # "OpenAI API". Use vocab matching plus a few explicit workflow phrases.
    backend_hits = (
        set(match_terms(text, BACKEND_TERMS))
        | set(match_terms(text, CLOUD_TERMS))
        | set(match_terms(text, PYTHON_TERMS))
    )
    backend_logic = bool(
        backend_hits & {"PostgreSQL", "MySQL", "MongoDB", "Redis", "REST API", "Kafka",
                        "RabbitMQ", "gRPC", "GraphQL", "WebSockets", "Docker", "GCP",
                        "AWS", "Kubernetes", "FastAPI", "Django", "Flask", "Celery"}
    ) or has_any("message queue", "task queue", "endpoint", "microservice", "rest api")

    return AIProjectSignals(
        retrieval=bool(hits & AI_SIGNAL_TERMS["retrieval"]) or has_any("retriev", "semantic search"),
        tool_use=bool(hits & AI_SIGNAL_TERMS["tool_use"]) or has_any("tool call", "function call"),
        state_or_memory=has_any("memory", "state", "session", "conversation history", "persist"),
        orchestration=bool(hits & AI_SIGNAL_TERMS["orchestration"]) or has_any("workflow", "multi-step", "planner"),
        evaluation=bool(hits & AI_SIGNAL_TERMS["evaluation"]) or has_any("evaluat", "benchmark", "guardrail"),
        product_logic=has_any("user", "customer", "dashboard", "product", "business", "deploy"),
        backend_logic=backend_logic,
        data_processing=has_any("pipeline", "etl", "chunk", "preprocess", "clean", "vectoriz"),
    )


def estimate_project_quality(project: Project) -> float:
    """Transparent 0-10 quality estimate from rubric signals (no LLM path)."""
    s = deterministic_signals(project)
    score = 2.0  # baseline: it is at least an AI project
    score += 1.5 if s.retrieval else 0
    score += 1.5 if s.orchestration else 0
    score += 1.0 if s.tool_use else 0
    score += 1.0 if s.state_or_memory else 0
    score += 1.0 if s.evaluation else 0
    score += 1.0 if s.backend_logic else 0
    score += 1.0 if s.data_processing else 0
    score += 1.0 if s.product_logic else 0
    score = min(10.0, score)
    # Without a retrieval/orchestration/tool/eval/backend/state workflow this is
    # a shallow AI project regardless of product/data prose.
    if not has_meaningful_workflow(s):
        score = min(score, 3.0)
    return round(score, 2)


def project_quality(project: Project) -> float:
    return project.quality_score if project.quality_score is not None else estimate_project_quality(project)


def _ai_projects(extracted: ExtractedResume) -> list[Project]:
    return [p for p in extracted.projects if p.domain in (ProjectDomain.AI_AGENTIC, ProjectDomain.AI_GENERAL)]


def score_ai_project_depth(
    extracted: ExtractedResume, w: Weights, th: Thresholds
) -> tuple[float, list[str], list[str]]:
    """Returns (points, rationale, concerns)."""
    ai_projects = _ai_projects(extracted)
    rationale: list[str] = []
    concerns: list[str] = []

    if not ai_projects:
        agentic = sorted({e.term for e in extracted.ai_evidence if e.term in AGENTIC_CORE_TERMS})
        if agentic:
            pts = min(th.framework_name_only_credit * w.ai_project_depth, len(agentic) * 2.0)
            rationale.append(f"AI framework(s) named but no AI project described: {', '.join(agentic)}")
            concerns.append("No concrete AI project evidence (framework names only)")
            return round(pts, 2), rationale, concerns
        return 0.0, ["No AI/agentic projects found"], ["No AI project evidence"]

    qualities = [project_quality(p) for p in ai_projects]
    best = max(qualities)
    solid = [q for q in qualities if q > th.thin_wrapper_quality]

    depth_points = (best / 10.0) * 0.70 * w.ai_project_depth
    breadth_points = min(1.0, len(solid) / 3.0) * 0.15 * w.ai_project_depth
    best_signals = max(
        (p.signals.count() if p.quality_score is not None else deterministic_signals(p).count())
        for p in ai_projects
    )
    signal_points = (best_signals / 8.0) * 0.15 * w.ai_project_depth

    for p, q in zip(ai_projects, qualities):
        rationale.append(f"{p.name or 'untitled'}: depth {q:.1f}/10")
    rationale.append(
        f"best-depth {depth_points:.1f} + breadth {breadth_points:.1f} "
        f"({len(solid)} solid project(s)) + signals {signal_points:.1f}"
    )
    if best < th.thin_wrapper_quality:
        concerns.append("AI project(s) look like thin wrappers (low depth)")

    return round(min(w.ai_project_depth, depth_points + breadth_points + signal_points), 2), rationale, concerns


# ---------------------------------------------------------------------------
# Python & backend (30)
# ---------------------------------------------------------------------------
def score_python_backend(
    extracted: ExtractedResume, w: Weights, th: Thresholds
) -> tuple[float, list[str]]:
    evidence = [*extracted.python_evidence, *extracted.other_evidence]
    points, terms = _weighted_terms_score(
        evidence, PYTHON_BACKEND_WEIGHTS, th, PYTHON_BACKEND_TARGET, w.python_backend
    )
    rationale = [f"Python/backend evidence: {', '.join(terms)}"] if terms else ["No Python/backend evidence"]
    return points, rationale


# ---------------------------------------------------------------------------
# Cloud / deployment / full stack (15)
# ---------------------------------------------------------------------------
def score_cloud_fullstack(
    extracted: ExtractedResume, w: Weights, th: Thresholds
) -> tuple[float, list[str]]:
    evidence = [*extracted.other_evidence, *extracted.python_evidence]
    points, terms = _weighted_terms_score(
        evidence, CLOUD_WEIGHTS, th, CLOUD_TARGET, w.cloud_fullstack
    )
    rationale = [f"Cloud/deploy/full-stack evidence: {', '.join(terms)}"] if terms else ["No cloud/deployment evidence"]
    return points, rationale


# ---------------------------------------------------------------------------
# Engineering depth (5)
# ---------------------------------------------------------------------------
def score_engineering_depth(
    extracted: ExtractedResume, w: Weights, th: Thresholds
) -> tuple[float, list[str]]:
    hits = sorted({ev.term for ev in extracted.other_evidence if ev.term in ENGINEERING_TERMS})
    if not hits:
        return 0.0, ["No engineering-depth signals found"]
    # Reward breadth of non-trivial engineering signals, capped at the weight.
    points = min(w.engineering_depth, len(hits) * (w.engineering_depth / 4.0))
    return round(points, 2), [f"Engineering signals: {', '.join(hits)}"]


# ---------------------------------------------------------------------------
# GitHub (10) — enrichment already returns 0-5 + 0-5
# ---------------------------------------------------------------------------
def score_github(github_score: float, w: Weights) -> tuple[float, list[str]]:
    capped = min(w.github, max(0.0, github_score))
    return round(capped, 2), [f"GitHub enrichment: {capped:.1f}/10"]
