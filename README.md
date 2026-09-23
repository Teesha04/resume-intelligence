# AI Resume Screening & Ranking System

A small, production-minded backend that ingests a folder of resumes, applies a
**hard, rule-based eligibility filter**, scores eligible candidates on a
100-point rubric, enriches them with public GitHub activity, and returns a
ranked shortlist with **evidence for every decision**.

```
python main.py --input ./resumes --output ./output/results.json
```

---

## 1. Quickstart

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) enable the LLM + higher GitHub rate limit
cp .env.example .env               # then fill GEMINI_API_KEY / GITHUB_TOKEN

# 3. Run
python main.py --input ./resumes --output ./output/results.json
```

The system runs **without any API keys** (deterministic mode) — the LLM and
GitHub token are strictly optional enhancements.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--limit N` | Process only the first N resumes (fast iteration) |
| `--csv PATH` | Also write a flat CSV |
| `--no-llm` | Force deterministic-only mode |
| `--no-cache` | Disable the on-disk network cache |
| `--top N` | Rows shown in the terminal summary |
| `-v` | Debug logging |

### Optional API

```bash
uvicorn src.api:app --reload
# POST /screen  {"input_dir": "./resumes", "limit": 50}
# GET  /results
```

### Tests

```bash
pytest -q
```

---

## 2. Architecture

```
main.py                     CLI entry point (argparse)
src/
  config.py                 weights, thresholds, model + secrets (pydantic-settings)
  schemas.py                Pydantic contracts for every pipeline stage + JSON output
  vocab.py                  shared skill/AI/cloud/engineering vocabulary
  matching.py               word-boundary, separator-tolerant term matching
  cache.py                  content-addressed on-disk cache for LLM/GitHub
  ingest/
    parsers.py              PDF (pdfplumber -> pypdf fallback), DOCX, TXT; never raises
    loader.py               discovery, dedupe by content hash, dispatch
  extract/
    sections.py             flexible resume section detection (incl. inline headings)
    deterministic.py        regex/vocab extraction (always runs) + evidence collection
    client.py               LLM adapter (Gemini + Null fallback)
    prompts.py              extraction + project-quality rubric prompt
    llm_schemas.py          structured-output models for the LLM
    extractor.py            deterministic pass -> optional LLM refinement (merge)
  eligibility/rules.py      HARD filter: Python evidence AND AI/agentic evidence
  scoring/
    rubric.py               the 100-point rubric (evidence-source weighted)
    penalties.py            thin-wrapper + tutorial deductions
    scorer.py               orchestration + strengths/concerns
  github/
    client.py               GitHub REST client, cache, rate-limit circuit breaker
    scoring.py              pure activity/repo scoring (0-5 + 0-5)
  pipeline.py               batch orchestration, bounded concurrency, per-item isolation
  report.py                 JSON / CSV writers + terminal summary
  api.py                    optional FastAPI wrapper
tests/                      unit + integration tests (53 tests)
tests/fixtures/generate_resumes.py   synthetic resume generator (dev + tests)
```

### Data flow

```
resumes/ ──▶ ingest (isolated parse) ──▶ deterministic extract ──▶ LLM refine (optional)
                                              │
                                              ▼
                                   HARD eligibility filter (no LLM)
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼ (eligible)                                ▼ (ineligible)
              score rubric + GitHub enrichment              reject with reasons
                        │
                        ▼
                 rank ──▶ results.json (+ summary, rationale)
```

---

## 3. Output

`results.json` contains a `summary` block (batch counters) and `candidates`
(eligible first, ranked; then rejected/unparsed). Every eligible candidate has
a `score_breakdown` **with a `rationale` list per category**.

```json
{
  "rank": 1,
  "candidate_name": "Asha Rao",
  "eligible": true,
  "total_score": 86.4,
  "score_breakdown": {
    "ai_project_depth": 33.7,
    "python_backend": 24.3,
    "cloud_fullstack": 10.7,
    "github": 8.0,
    "engineering_depth": 5.0,
    "penalties": 0.0,
    "rationale": { "ai_project_depth": ["Agentic Triage: depth 9.0/10", "..."] }
  },
  "matched_skills": ["Python", "FastAPI", "LangGraph", "..."],
  "project_summary": "...",
  "github_summary": "last public activity ~12d ago; relevant repos: ...",
  "strengths": ["Strong AI/agentic project depth", "..."],
  "concerns": ["Limited Redis evidence"],
  "rejection_reasons": [],
  "github": { "status": "ok", "recent_activity_score": 4.0, "repos_score": 4.0 }
}
```

---

## 4. Design Decisions

### 4.1 Filtering strategy (hard eligibility, no LLM)

A candidate is **eligible only if both** hold, evaluated by deterministic code
(`src/eligibility/rules.py`) so the outcome is predictable and unit-tested:

1. **Python evidence** — `Python` (or a Python-specific framework/library) must
   match as a real token. Matching is word-boundary-aware, so `JavaScript` never
   satisfies `Java` and `PostgreSQL` never satisfies `SQL`.
2. **AI/agentic evidence** — either a *core LLM/agentic* signal (LangChain,
   LangGraph, Google ADK, RAG, embeddings, vector search, tool calling,
   multi-agent, evaluation, …) **or** a genuine AI project (agentic or general
   ML/CV/NLP).

Additional rules:
- JavaScript/Java/React are **not** grounds for rejection when Python + AI are
  also present (explicitly required by the spec).
- A generic AI *skill* with no project (e.g. `PyTorch` listed but nothing built)
  is deliberately **not** enough to satisfy the AI condition.
- The hard filter is kept **out of the LLM** entirely.

### 4.2 Scoring strategy (100 points, explainable)

| Category | Weight | How it is earned |
|---|---|---|
| AI / agentic / RAG project depth | 40 | best project depth (70%) + breadth of solid projects (15%) + rubric signals (15%) |
| Python & backend engineering | 30 | weighted skills, scaled by **where** they appear |
| Cloud / deployment / full-stack | 15 | Docker/GCP/AWS/K8s/CI-CD; React/Next as supporting signals |
| GitHub activity | 10 | 0-5 recent activity + 0-5 maintained/relevant repos |
| Engineering depth | 5 | testing, architecture, caching, queues, observability, concurrency |
| Project-quality penalties | −5…−15 | thin LLM/API wrappers; tutorials/undetailed projects |

The central idea is **evidence-source weighting**. Every skill match records
where it was found, and the same term earns different credit:

| Source | Multiplier |
|---|---|
| Project description | 1.0 |
| Internship / work | 0.9 |
| Preamble / summary | 0.5 |
| Skills list only | 0.4 |

So "Python" in a shipped project is worth far more than "Python" in a skills
list — directly implementing the spec's "reward evidence over keyword-only
skill lists". The same applies to the AI category: framework names appearing
only in a skills list earn a small capped credit (`0.25 × 40`), never full
marks.

**Penalties** are separate and explainable:
- *Thin wrapper* — a project that invokes an LLM/API but lacks retrieval,
  orchestration, tools, evaluation, state or backend workflow. Scaled 5-15 by
  shallowness. Classical ML projects (e.g. a CNN) are **not** treated as
  wrappers.
- *Tutorial / undetailed* — name/description matches tutorial markers, or the
  description is too short to show implementation detail.

Only eligible candidates are scored; ineligible ones still appear in the output
with explicit rejection reasons.

### 4.3 LLM usage

The LLM is an **optional accelerator, never a dependency**:

- **Deterministic extraction always runs first**; the LLM then refines
  projects/skills and adds per-project quality judgments. If the LLM is absent
  or fails, the deterministic result is used and a warning is recorded.
- **Structured output**: the model is given a Pydantic response schema
  (`src/extract/llm_schemas.py`) and its JSON is validated locally; invalid
  output degrades gracefully.
- **Evidence required**: every LLM project judgment includes a
  `quality_rationale`, and a written rubric in the prompt makes 0-10 scores
  consistent (temperature 0).
- **Adapter pattern**: all provider-specific code lives in
  `src/extract/client.py` behind a one-method interface; adding OpenAI/Claude is
  a new class, nothing else changes.
- **Rate limiting**: free LLM tiers enforce a low requests-per-minute quota, so
  the adapter paces calls (`LLM_REQUESTS_PER_MINUTE`) and honours the server's
  own `retryDelay` on 429s. Without this, a 50-resume batch trips the quota and
  silently falls back to deterministic scoring for most candidates, producing
  inconsistent results.
- **Response caching**: successful extractions are cached by content hash, so
  reruns and re-scoring are near-instant and don't re-spend quota.
- **Failure isolation**: `LLMError`/validation errors are caught per resume; one
  bad call never fails the batch.
- **Keys from env only** (`GEMINI_API_KEY`); nothing is hard-coded.
- Default model: `gemini-3.1-flash-lite` (fast, cheap, structured-output capable).

Deterministic mode still produces a full ranking via a transparent
signal-based quality estimate, so the system is fully usable with zero keys.

### 4.4 GitHub scoring

- Only the **public REST API** is used, with an optional token read from
  `GITHUB_TOKEN` (raises the limit from 60 to 5000 req/hr).
- Signals: recency of public events/commits (falling back to repository
  `pushed_at`), count of recent events, number of maintained repositories, and
  repositories that are Python/AI-relevant.
- **Cap 10 points**: `0-5` activity + `0-5` repositories.
- **Never fatal**: `no_profile`, `not_found`, `rate_limited`, and `error` are
  recorded as statuses and score 0. A 403/429 trips an in-run circuit breaker so
  we stop burning the rate limit.
- **Cached** per username, and only fetched for eligible candidates (GitHub only
  feeds scoring).

### 4.5 Reliability

- Parsers never raise; each file becomes `ok` / `empty` / `failed` / `duplicate`.
- **Dual PDF extraction with quality-based selection**: pdfplumber and pypdf are
  both run, and the higher-quality text wins (pypdf is often markedly better for
  two-column resumes and for PDFs where pdfplumber glues words together).
  Quality is scored by counting recognisable words and penalising long glued
  tokens; pdfplumber is kept unless pypdf is clearly better (>15%).
- Word-boundary matching also tolerates glued forms (`MachineLearning`) and
  hyphen/space variants (`multi-agent`).
- Content-hash dedupe handles the same resume submitted twice (or as both PDF
  and DOCX).
- PDF glyph artefacts (`(cid:123)`) are stripped at the cleaning boundary.
- Bounded concurrency (`MAX_CONCURRENCY`) over a thread pool; one resume's
  failure cannot abort the batch.
- `float`/`None` handling: totals are clamped to `[0, 100]`; penalties cannot go
  below the configured cap.

---

## 5. If I Had More Time

1. **LLM-assisted, cached resume embeddings + calibration set.** Build a small
   labelled set of resumes and tune weights/thresholds against it, rather than
   hand-set values; measure rank correlation with human review.
2. **Two-pass LLM extraction with self-critique.** A second pass that checks
   every credited skill/project claim against the raw text would cut
   hallucination further, especially on unusual layouts.
3. **Richer GitHub + other signals.** Commit-message topic modelling,
   per-language byte counts via the `/languages` endpoint, stars/forks weighting,
   and optional portfolio/LinkedIn enrichment.
4. **Better PDF robustness.** OCR fallback (Tesseract) for scanned/image-only
   resumes, and table-aware extraction for two-column layouts.

---

## 6. Notes on the resume set

- `resumes/` holds the provided candidate set (50 PDFs); `output/results.json`
  and `output/results.csv` are generated from it. The input folder is
  git-ignored so candidate PII is never committed.
- `resumes_sample/` contains a **synthetic** set generated by
  `tests/fixtures/generate_resumes.py` (50 candidates + edge cases: corrupt PDF,
  empty file, duplicate, unsupported type). It exercises every branch of the
  pipeline and is handy for demos and regression checks.

Run against either with:

```bash
python main.py --input ./resumes --output ./output/results.json
```

### Runtime notes

- **Without `GEMINI_API_KEY`** the pipeline runs deterministically (AI project
  depth uses the transparent signal heuristic). With a key, project quality is
  judged by the LLM with evidence.
- **Without `GITHUB_TOKEN`** GitHub's public limit is 60 req/hr, which the
  50-resume run will exhaust; enrichment is then recorded as `rate_limited` and
  scores 0 (screening still completes). Set a token for full enrichment.
- `Machine Learning` / `Deep Learning` / `CV` / `NLP` projects count as **AI
  project** evidence for eligibility, but score low on the agentic-focused
  40-point depth category — so agentic candidates rank above generic-ML ones.
