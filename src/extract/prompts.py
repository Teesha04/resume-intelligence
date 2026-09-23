"""Prompts for the LLM.

Kept separate from logic so prompt iteration never touches code paths, and so
the provider adapter stays generic (it only sees `system` + `prompt` strings).

Prompt-engineering choices:
  * Explicit "do not invent / do not credit unmentioned tech" guardrails to
    fight hallucination, since scores must be resume-evidenced.
  * A written rubric for quality_score so 0-10 judgments are consistent across
    resumes and reproducible at temperature 0.
  * Required rationale per project => explainability (spec section 7).
  * Only the resume text is provided; eligibility is decided elsewhere.
"""

from __future__ import annotations

EXTRACTION_SYSTEM = (
    "You are a meticulous technical resume analyst. You extract only information "
    "that is explicitly supported by the resume text. You never invent skills, "
    "project details, or metrics. When evidence is absent you leave fields empty "
    "rather than guessing. You output strict JSON only."
)

EXTRACTION_PROMPT = """\
Analyse the resume text below and return a single JSON object.

## Extraction rules
- `candidate_name`: the person's name. Empty string if genuinely unclear.
- `skills`: technologies/languages/frameworks the candidate lists or uses.
  Use canonical names (e.g. "PostgreSQL" not "postgres", "LangChain", "FastAPI").
- `projects`: every distinct project. For each project:
  - `name`, `description` (2-4 sentences, faithful to the resume).
  - `technologies`: only technologies explicitly attached to THIS project.
  - `domain`: one of: ai_agentic, ai_general, web_fullstack, data, other, unknown.
    Use `ai_agentic` for LLM/agent/RAG systems; `ai_general` for ML/CV/NLP that
    is not agentic; `unknown` if unclear.
- `experience_highlights`: short bullet strings of internship/work achievements.
- Do NOT credit a technology to a project unless it is mentioned near it.

## Signal rubric (set true only with explicit evidence)
- `retrieval`: RAG, vector search, embeddings, chunking, indexing.
- `tool_use`: tool/function calling, external API/tool invocation by a model.
- `state_or_memory`: agent state, memory, conversation/session persistence.
- `orchestration`: multi-agent, LangGraph/ADK/planner, DAGs, routing, retries between steps.
- `evaluation`: eval harness, metrics, benchmarking, guardrails, regression tests for AI.
- `product_logic`: real user/product/business logic around the model.
- `backend_logic`: APIs, queues, persistence, async work driven by the model.
- `data_processing`: ETL, cleaning, chunking, transformation pipelines.

## quality_score (0-10) — project depth as a real system
- 0-2: thin wrapper: a single LLM/API call, no workflow, retrieval, state,
  evaluation, or product logic. (Set `is_thin_wrapper` = true.)
- 3-5: some structure: e.g. basic RAG or a single agent with a couple of tools,
  limited evaluation or product logic.
- 6-8: substantive: retrieval + tool use + state/orchestration, meaningful
  backend/persistence, some evaluation or product logic.
- 9-10: production-minded: multi-step orchestration, evaluation, failure
  handling, scaling or observability, and clear product/business impact.
- `is_tutorial`: true if the project is a known tutorial/boilerplate with no
  evidence of the candidate's own design or extensions.
- `quality_rationale`: one sentence citing the specific resume evidence.

Respond with JSON only, matching the provided schema exactly.

## RESUME TEXT
{resume_text}
"""


def build_extraction_prompt(resume_text: str, max_chars: int = 14000) -> str:
    text = resume_text if len(resume_text) <= max_chars else resume_text[:max_chars]
    return EXTRACTION_PROMPT.format(resume_text=text)


PROJECT_SCORING_SYSTEM = (
    "You are a senior engineer evaluating a candidate's AI/LLM projects for an "
    "SDE internship focused on Python and agentic systems. You assess only what "
    "the supplied text supports, you never invent technologies, and you explain "
    "every score with a reason and a short quoted snippet. You output strict JSON."
)

# --- Shared rubric text (kept identical to the extraction prompt) ----------
_SIGNAL_RUBRIC = """\
## Signal rubric (set true only with explicit evidence)
- `retrieval`: RAG, vector search, embeddings, chunking, indexing.
- `tool_use`: tool/function calling, external API/tool invocation by a model.
- `state_or_memory`: agent state, memory, conversation/session persistence.
- `orchestration`: multi-agent, LangGraph/ADK/planner, DAGs, routing, retries between steps.
- `evaluation`: eval harness, metrics, benchmarking, guardrails, regression tests for AI.
- `product_logic`: real user/product/business logic around the model.
- `backend_logic`: APIs, queues, persistence, async work driven by the model.
- `data_processing`: ETL, cleaning, chunking, transformation pipelines.

## quality_score (0-10) — project depth as a real system
- 0-2: thin wrapper: a single LLM/API call, no workflow, retrieval, state,
  evaluation, or product logic. (Set `is_thin_wrapper` = true.)
- 3-5: some structure: e.g. basic RAG or a single agent with a couple of tools,
  limited evaluation or product logic.
- 6-8: substantive: retrieval + tool use + state/orchestration, meaningful
  backend/persistence, some evaluation or product logic.
- 9-10: production-minded: multi-step orchestration, evaluation, failure
  handling, scaling or observability, and clear product/business impact.
- `is_tutorial`: true if the project is a known tutorial/boilerplate with no
  evidence of the candidate's own design or extensions.
- `quality_rationale`: one sentence stating WHY the project got this score.
- `cited_evidence`: a short verbatim snippet from the text that supports it."""

PROJECT_SCORING_PROMPT = """\
Below is the project / experience context for ONE candidate. Identify each
distinct project, decide whether it is AI/agentic, and score its depth.

## Rules
- Only include projects supported by the text below. NEVER invent projects or
  technologies.
- `domain`: one of ai_agentic, ai_general, web_fullstack, data, other, unknown.
- Do NOT credit a technology to a project unless it appears near it in the text.
- Score ONLY the projects; candidate fields are already extracted elsewhere.

""" + _SIGNAL_RUBRIC + """

Respond with JSON only, matching this shape exactly:
{"projects": [{"name": ..., "description": ..., "technologies": [...],
"domain": ..., "signals": {...}, "quality_score": ..., "quality_rationale": ...,
"cited_evidence": ..., "is_thin_wrapper": ..., "is_tutorial": ...}]}

## CANDIDATE PROJECT / EXPERIENCE TEXT
{context}
"""


def build_project_scoring_prompt(context: str, max_chars: int = 6000) -> str:
    text = context if len(context) <= max_chars else context[:max_chars]
    # Use replace, not format: the prompt contains literal JSON braces.
    return PROJECT_SCORING_PROMPT.replace("{context}", text)
