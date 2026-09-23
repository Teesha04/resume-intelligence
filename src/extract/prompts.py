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
