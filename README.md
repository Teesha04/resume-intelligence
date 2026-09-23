# AI Resume Screening & Ranking System

A small, production-minded backend that ingests a folder of resumes, applies a
**hard, rule-based eligibility filter first**, scores the survivors on a
100-point rubric, enriches them with public GitHub activity, uses an LLM to judge
**project quality with cited evidence**, and returns a ranked shortlist where
**every point is explained**.

```
python main.py --input ./resumes --output ./output/results.json
```

> **Runs with zero API keys.** Without a key the system is fully deterministic
> (transparent signal-based quality estimate). With a `GEMINI_API_KEY` it adds
> LLM project scoring; with a `GITHUB_TOKEN` it adds full GitHub enrichment.

---

## Table of contents

1. [Quickstart](#1-quickstart)
2. [How it works (pipeline)](#2-how-it-works-pipeline)
3. [Architecture / project layout](#3-architecture--project-layout)
4. [Eligibility: the hard gate](#4-eligibility-the-hard-gate)
5. [Scoring: the 100-point rubric](#5-scoring-the-100-point-rubric)
6. [LLM usage](#6-llm-usage)
7. [GitHub enrichment](#7-github-enrichment)
8. [Output reference](#8-output-reference)
9. [Configuration reference](#9-configuration-reference)
10. [CLI & API](#10-cli--api)
11. [Reliability & failure modes](#11-reliability--failure-modes)
12. [Tests](#12-tests)
13. [Performance & cost](#13-performance--cost)
14. [Design Decisions](#14-design-decisions)
15. [Known limitations](#15-known-limitations)
16. [If I Had More Time](#16-if-i-had-more-time)
17. [Reproducing the provided results](#17-reproducing-the-provided-results)

---

## 1. Quickstart

```bash
# 1. Environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) enable the LLM + higher GitHub rate limit
cp .env.example .env                 # then fill GEMINI_API_KEY / GITHUB_TOKEN

# 3. Run
python main.py --input ./resumes --output ./output/results.json
```

The pipeline prints a batch summary and writes `results.json` (+ optional CSV).

### Useful flags

| Flag | Purpose |
|------|---------|
| `--input, -i DIR` | Directory of resumes (required) |
| `--output, -o PATH` | JSON output path (default `output/results.json`) |
| `--csv PATH` | Also write a flat CSV |
| `--limit N` | Process only the first N resumes (fast iteration) |
| `--top N` | Rows shown in the terminal summary |
| `--no-llm` | Force deterministic-only mode |
| `--no-cache` | Disable the on-disk network cache |
| `--verbose, -v` | Debug logging |

---

## 2. How it works (pipeline)

```
resumes/
   │
   ▼
[1] INGEST            discover files, de-duplicate by content hash,
   │                  parse PDF/DOCX/TXT (each file isolated; never raises)
   ▼
[2] DETERMINISTIC     regex + vocabulary extraction: name, email, GitHub,
    EXTRACT           skills, rough projects, and *evidence* (term + source
   │                  + context). Cheap; always runs.
   ▼
[3] HARD GATE         eligibility = has Python AND has AI/agentic evidence.
   │                  Runs on raw-text evidence only. NO network.
   │
   ├── eliminated ──────────────────────────▶ rejected with explicit reasons
   │                                          (no GitHub call, no LLM call)
   ▼ survivors
[4] GITHUB            only if a username exists. NEVER eliminates; failures
   │                  are tagged. Runs before the LLM.
   ▼
[5] LLM SCORING       optional. Scores each AI project's depth with a short
   │                  rationale + cited evidence. Wait-and-resume on rate limits.
   ▼
[6] SCORE + RANK      deterministic categories + GitHub + LLM project depth,
                      penalties, clamp to [0,100], rank eligible candidates.
```

Two consequences fall out of this order:

- **Eliminated candidates cost nothing** — no GitHub requests, no LLM tokens.
- **The model can never change who is eligible** — eligibility is frozen at
  step 3 before any LLM call, which is exactly what the spec asks for.

---

## 3. Architecture / project layout

```
main.py                      CLI entry point (argparse)
requirements.txt / .env.example
README.md

src/
  config.py                  Settings (pydantic-settings): weights, thresholds,
                             model, secrets. Business logic holds no constants.
  schemas.py                 Pydantic contracts for every stage + the JSON output
  vocab.py                   Tiered term vocabularies (Python/AI/backend/cloud/eng)
  matching.py                Word-boundary, separator-tolerant term matching
  cache.py                   Content-addressed on-disk cache (llm/ + github/)
  pipeline.py                Batch orchestration, bounded concurrency, isolation
  report.py                  JSON / CSV writers + terminal summary
  api.py                     Optional FastAPI wrapper (/health, /screen, /results)

  ingest/
    parsers.py               PDF (pdfplumber + pypdf, pick best), DOCX, TXT
    loader.py                Discovery, dedupe, dispatch

  extract/
    sections.py              Resume section detection (incl. inline "Skills: ...")
    deterministic.py         Field extraction + evidence collection (always runs)
    client.py                LLM adapter: GeminiClient + NullLLMClient, rate limiter
    prompts.py               Full-extraction prompt + project-scoring prompt
    llm_schemas.py           Structured-output models for the LLM
    extractor.py             refine_with_llm() and score_projects_with_llm()

  eligibility/
    rules.py                 The hard gate (deterministic, no LLM)

  scoring/
    rubric.py                The 100-point rubric formulas + signal detection
    penalties.py             Thin-wrapper / tutorial deductions
    scorer.py                Orchestration + strengths/concerns derivation

  github/
    client.py                REST client, cache, rate-limit circuit breaker, tagging
    scoring.py               Pure activity/repo scoring (0-5 + 0-5)

tests/                       66 unit + integration tests
tests/fixtures/generate_resumes.py   Synthetic resume generator (dev + tests)
```

---

## 4. Eligibility: the hard gate

**File:** `src/eligibility/rules.py`. Fully deterministic; runs before any
network call.

A candidate is **eligible only if both** conditions hold:

1. **Python evidence** — `Python` (or a Python-only framework/library such as
   FastAPI, Django, Flask, pandas, asyncio) must appear as a real token.
2. **AI/agentic evidence** — a core LLM/agentic signal, **or** a genuine AI
   project (agentic or general ML/CV/NLP) in a project/experience section.

If either is missing, the candidate is eliminated with an explicit reason.

### 4.1 Positive-only, never negative

The gate asks **only** "is Python present?" and "is AI present?". It never asks
"is JavaScript present?". So:

| Resume | Result |
|---|---|
| JavaScript + React, no Python/AI | rejected (missing Python **and** AI) |
| Java + Spring Boot + React, no Python/AI | rejected (missing Python **and** AI) |
| JavaScript + React **+** Python **+** AI | **eligible** |
| Python + FastAPI, no AI | rejected (missing AI) |
| Python + AI written in JS/TS | rejected (missing Python) — Python is a hard requirement |

Rejection reasons name only the *unmet requirement*:

```json
["No evidence of Python stack", "No AI/agentic project evidence"]
```

`matched_skills` is populated for **every** candidate (eligible or not), so a
rejected Java/React profile still shows `["Java", "React", "Spring Boot"]`.

### 4.2 AI evidence tiers

The AI vocabulary is tiered so a coursework mention can't rescue an empty
profile:

- **Tier 1 (core LLM/agentic)** — LangChain, LangGraph, LlamaIndex, Google ADK,
  CrewAI, AutoGen, Semantic Kernel, DSPy, MCP, OpenAI/Anthropic/Gemini, RAG,
  embeddings, vector search (Pinecone/FAISS/Chroma/Qdrant/pgvector), tool/function
  calling, multi-agent, agents, prompting, evaluation, fine-tuning, LoRA, vLLM.
- **Tier 2 (AI ecosystem)** — LLM/large language model, foundation model,
  GenAI, Hugging Face, Bedrock, SageMaker, Azure AI, ASR/TTS, summarization, NER.
- **Tier 3 (general ML/DS)** — machine learning, deep learning, neural networks,
  CNN/RNN/LSTM, XGBoost, scikit-learn, TensorFlow, PyTorch, NLP, computer vision,
  data science. **Only counts inside a project/experience section.**

A Tier-3 *skill* with no project (e.g. "PyTorch" in a skills list) does **not**
satisfy the AI condition — you have to have built something.

### 4.3 Matching rules (precision)

Matching is word-boundary-aware and separator-tolerant, which prevents the usual
false positives and negatives:

| Trap | Handling |
|---|---|
| `Java` inside `JavaScript` | boundary → no match |
| `SQL` inside `PostgreSQL` | boundary → no match |
| `CV` = **curriculum vitae** | never matched; only `computer vision` counts |
| `embedding` vs `embedded systems` | only `embedding` matches |
| `vector` alone | requires `vector search/database/store/embedding` |
| `model` alone | never matches (`data model` is not AI) |
| `Machine Learning` written `MachineLearning` | glued forms tolerated |
| `multi agent` / `multi-agent` | hyphen/space/glue all tolerated |

Because the gate is load-bearing, vocabulary quality is a first-class concern;
see [matching.py](src/matching.py) and [vocab.py](src/vocab.py).

---

## 5. Scoring: the 100-point rubric

Implemented deterministically in `src/scoring/`; the LLM only refines the
**AI project depth** category (see §6). All formulas live in `rubric.py`.

### 5.1 Weights

| Category | Weight | Source |
|---|---|---|
| AI / Agentic / RAG project depth | **40** | deterministic signals + optional LLM |
| Python & backend engineering | **30** | deterministic, evidence-source weighted |
| Cloud / deployment / full-stack | **15** | deterministic, evidence-source weighted |
| GitHub activity | **10** | GitHub enrichment (0-5 + 0-5) |
| Engineering depth signals | **5** | deterministic breadth |
| Project-quality penalties | **−5 … −15** | thin wrapper / tutorial |

### 5.2 Evidence-source weighting (the core idea)

Every matched term records **where** it was found, and the same term earns
different credit:

| Source | Multiplier |
|---|---|
| Project description | **1.0** |
| Internship / work | **0.9** |
| Summary / preamble | 0.5 |
| Skills list only | **0.4** |

So "Python in a shipped project" ≫ "Python in a skills list" — directly
implementing the spec's *"reward evidence in projects/internships over
keyword-only skill lists."*

### 5.3 Per-category formulas

**AI / Agentic / RAG project depth (40)**
```
depth   = (best project quality / 10) × 0.70 × 40      # ≤ 28
breadth = min(1, solid_AI_projects / 3) × 0.15 × 40    # ≤ 6
signals = (best project signal count / 8) × 0.15 × 40  # ≤ 6
ai_project_depth = min(40, depth + breadth + signals)
```
- Signals detected per project: `retrieval`, `tool_use`, `state_or_memory`,
  `orchestration`, `evaluation`, `product_logic`, `backend_logic`,
  `data_processing`.
- **Framework name in a skills list only** (no project) → capped at
  `0.25 × 40 = 10`.
- A "solid" project is one whose quality `> 3/10`.

**Python & backend engineering (30)**
```
score = min(30, (Σ term_weight × best_source_multiplier) / 7.0 × 30)
term weights: Python 2.0 · FastAPI 1.8 · Django 1.5 · PostgreSQL 1.5 ·
              asyncio 1.2 · Redis 1.2 · Flask 1.2 · SQLAlchemy/Celery 1.0 ·
              REST API 1.0 · pytest/Pandas/NumPy 0.8 · …
```

**Cloud / deployment / full-stack (15)**
```
score = min(15, (Σ term_weight × best_source_multiplier) / 4.0 × 15)
term weights: Docker 1.5 · GCP/AWS 1.5 · Azure/Kubernetes 1.2 · CI/CD 1.0 ·
              Terraform 0.8 · React 0.6 · Next.js 0.5 · Tailwind 0.3
```
React/Next.js are **supporting** signals only.

**GitHub activity (10)** — see §7.

**Engineering depth signals (5)**
```
score = min(5, distinct_signals × 1.25)
signals: Testing · Architecture · Caching · Queues · Observability ·
         Concurrency · Failure Handling · Performance · Security
```

### 5.4 Penalties

```
thin wrapper:   −(5 + (1 − quality/10) × 10)     # scales 5…15
tutorial OR description < 15 words:  −3 each
total penalties capped at −15; final total clamped to [0, 100]
```

A **thin wrapper** = a project that invokes an LLM/API but has no retrieval,
orchestration, tools, evaluation, state, or backend workflow. Guardrails:

- A classical ML project (e.g. a CNN) is **not** a wrapper.
- `LangChain` alone is **not** treated as orchestration (that mistake was found
  and fixed during testing on real resumes).

### 5.5 Explanations

Every category attaches a `rationale` list. Example (real output):

```json
"ai_project_depth": [
  "Adaptive Agentic RAG - Multi-Agent Research System: depth 9.0/10 — Multi-agent orchestration with retrieval and tool calling.",
  "best-depth 25.2 + breadth 4.0 (2 solid project(s)) + signals 4.5"
],
"penalties": ["Thin LLM/API wrapper, no meaningful workflow: Chatbot (-12.0)"]
```

---

## 6. LLM usage

The LLM is an **optional accelerator, never a dependency**. If it is absent or
fails, the deterministic estimate is used and a warning is recorded.

### 6.1 What the LLM does (and doesn't)

**Does:** score each AI project's depth (0-10) and return, per project, the
rubric `signals`, a one-sentence `quality_rationale`, and a short
`cited_evidence` snippet.

**Does not:** decide eligibility, extract name/email/skills/experience, or score
Python/cloud/engineering. Those stay deterministic so ranking remains testable.

### 6.2 Structured output

- Response schema: `LLMProjectAssessment` (`src/extract/llm_schemas.py`) with
  nested `AIProjectSignals`.
- Validated locally with Pydantic; malformed JSON is rejected and the
  deterministic value is used (`llm_scoring_failed` warning).
- `temperature=0` for reproducibility.

### 6.3 Small prompts

For gate survivors the call receives **only the PROJECTS + EXPERIENCE context**
(`build_scoring_context`), not the whole resume, and returns **only project
scores** — smaller input and output than full extraction. The context falls back
to full text if those sections aren't detected, so projects are never lost.

### 6.4 Adapter pattern

```
LLMClient (Protocol)          one method: generate_json(system, prompt, schema, temperature)
├── GeminiClient              google-genai, structured output, rate limiter, retries
└── NullLLMClient             used when no key — raises LLMError, never crashes
```
`build_llm_client(...)` is the single factory. Adding OpenAI/Claude is one new
class; nothing else changes. **Keys come only from the environment.**

### 6.5 Rate limits: wait-and-resume

Free tiers enforce a low requests-per-minute quota. The adapter:

1. **Paces** calls (`LLM_REQUESTS_PER_MINUTE`, default 12, under the 15/min tier).
2. On a quota error, **waits for the reset and resumes the same request**
   (honouring the server's `retryDelay`), printing e.g.
   `LLM rate limit reached; waiting 55s for the quota to reset, then resuming…`.
   It does **not** silently fall back to deterministic scoring, which would make
   rankings inconsistent across candidates.
3. Only if the wait budget (`LLM_WAIT_BUDGET_SECONDS`) is exhausted does a single
   resume degrade to its deterministic score — never the whole batch.

Counters `llm_rate_limit_waits` and `llm_wait_seconds` are reported in the
summary.

### 6.6 Caching

Successful responses are cached on disk by a hash of
`(provider, model, system, prompt, schema, temperature)`, so reruns and
re-scoring are near-instant and don't re-spend quota.

### 6.7 Failure isolation

`LLMError` and validation errors are caught **per resume**; the batch always
completes. `--no-llm` forces deterministic-only mode.

---

## 7. GitHub enrichment

**Runs before the LLM**, only for gate survivors that have a GitHub username on
the resume. It **never** affects eligibility.

### 7.1 Signals (public REST API)

- `/users/{u}` — profile (public repo count, existence)
- `/users/{u}/events/public` — recent public events
- `/users/{u}/repos?sort=pushed` — maintained/relevant repositories

Recency falls back to repository `pushed_at` when the events feed is empty
(common for private activity).

### 7.2 Scoring (cap 10)

```
recent_activity (0-5):  recency bands (≤30d→4, ≤90d→3, ≤180d→2, ≤365d→1, else 0.5)
                        + volume boost (≥5 events→+1, ≥2→+0.5), capped at 5
repos (0-5):            0.8 × maintained repos (pushed ≤180d)
                        + 1.2 × Python/AI-relevant maintained repos, capped at 5
github = min(10, recent_activity + repos)
```

### 7.3 Failure handling & tagging

| Status | Meaning | Score |
|---|---|---|
| `ok` | Enriched | 0-10 |
| `no_profile` | No GitHub on the resume | 0 (a *known* fact) |
| `not_found` | Profile doesn't exist | 0 |
| `rate_limited` | API limit hit (in-run circuit breaker stops further calls) | **tagged → neutral placeholder** |
| `error` | Transport/other failure | **tagged → neutral placeholder** |
| `disabled` | Candidate eliminated by the gate | n/a |

A rate-limited/errored profile is **unknown**, not inactive, so it is tagged
(`github.flagged = true`) and receives a deterministic **neutral placeholder**
(default `5.0/10`, `GITHUB_RATE_LIMITED_PLACEHOLDER`) instead of 0. We
deliberately avoid random values — they would break reproducibility and
explainability, which the assignment requires.

### 7.4 Token & caching

`GITHUB_TOKEN` (env only) raises the limit from 60 to 5000 req/hr. Successful
responses are cached per username; failures are not cached.

---

## 8. Output reference

### 8.1 `results.json`

```json
{
  "summary": {
    "total_resumes": 50, "parsed_ok": 50, "parse_failed": 0,
    "duplicates_skipped": 0, "eligible": 41, "rejected": 9,
    "llm_used": true, "github_enriched": 6, "github_failures": 10,
    "llm_skipped_by_gate": 9, "llm_scored": 41,
    "llm_rate_limit_waits": 0, "llm_wait_seconds": 0.0,
    "github_tagged_rate_limited": 10, "github_tagged_error": 0,
    "github_no_profile": 25, "duration_seconds": 419.37
  },
  "candidates": [ { /* see below */ }, ... ]
}
```

`candidates` are ordered: **eligible (by rank), then rejected/unparsed**.

### 8.2 Candidate object

| Field | Type | Notes |
|---|---|---|
| `rank` | int \| null | 1..N for eligible; `null` otherwise |
| `candidate_name` | str | from deterministic parsing (email/file fallback) |
| `eligible` | bool | hard-gate verdict |
| `total_score` | float \| null | 0-100 for eligible; `null` otherwise |
| `score_breakdown` | object \| null | per-category points + `rationale` |
| `matched_skills` | list[str] | always populated |
| `project_summary` | str | best project + short description |
| `github_summary` | str | activity/repo summary or failure reason |
| `strengths` / `concerns` | list[str] | derived from the breakdown |
| `rejection_reasons` | list[str] | empty for eligible |
| `github` | object | `status`, scores, `flagged`, `relevant_repos`, … |
| `source_file` | str | provenance |
| `parse_status` | str | `ok` / `empty` / `failed` / `duplicate` |
| `llm_applied` | bool | true if LLM project scoring succeeded |
| `warnings` | list[str] | e.g. `llm_scoring_failed:LLMError` |

### 8.3 `score_breakdown`

```json
{
  "ai_project_depth": 40.0,
  "python_backend": 30.0,
  "cloud_fullstack": 15.0,
  "github": 9.5,
  "engineering_depth": 5.0,
  "penalties": 0.0,
  "rationale": {
    "ai_project_depth": ["...", "..."],
    "python_backend": ["Python/backend evidence: Python (project), FastAPI (experience), ..."],
    "cloud_fullstack": ["Cloud/deploy/full-stack evidence: GCP (project), Docker (experience)"],
    "github": ["GitHub enrichment: 9.5/10"],
    "engineering_depth": ["Engineering signals: Caching, Testing, Concurrency, ..."],
    "penalties": ["No project-quality penalties"]
  }
}
```

`score_breakdown.total()` clamps to `[0, 100]`.

---

## 9. Configuration reference

All tunables live in `src/config.py`; secrets and overrides come from `.env`
(see `.env.example`).

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | Enables LLM scoring. Absent → deterministic-only |
| `LLM_PROVIDER` | `gemini` | Adapter selection |
| `LLM_MODEL` | `gemini-3.1-flash-lite` | Model (older `gemini-2.0-flash` is retired) |
| `LLM_REQUESTS_PER_MINUTE` | `12` | Pacing to stay under the tier limit |
| `LLM_ON_RATE_LIMIT` | `wait` | `wait` (resume after reset) or `fail` |
| `LLM_MAX_WAIT_SECONDS` | `90` | Cap for a single wait |
| `LLM_WAIT_BUDGET_SECONDS` | `600` | Total wait budget per run |
| `GITHUB_TOKEN` | *(unset)* | 60 → 5000 req/hr |
| `GITHUB_RATE_LIMITED_PLACEHOLDER` | `5` | Neutral value (of 10) for tagged profiles |
| `MAX_CONCURRENCY` | `5` | Thread-pool size (bounded) |
| `REQUEST_TIMEOUT_SECONDS` | `20` | HTTP timeout |
| `CACHE_DIR` | `.cache` | On-disk network cache |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Scoring config (in `config.py`)

- `Weights` — the five category weights + GitHub split (must total 100).
- `Thresholds` — evidence-source multipliers, thin-wrapper quality cut-off,
  framework-name-only credit, GitHub activity/repo windows.
- `Penalties` — thin-wrapper range, tutorial deduction, cap.

---

## 10. CLI & API

### CLI

```bash
python main.py --input ./resumes --output ./output/results.json [--csv out.csv] [--limit N] [--top N] [--no-llm] [--no-cache] [-v]
```

### Optional FastAPI

```bash
uvicorn src.api:app --reload

GET  /health     -> {"status":"ok","llm_available":true,"github_token":false}
POST /screen     -> {"input_dir":"./resumes","limit":50}  → runs the pipeline
GET  /results    -> last ScreeningReport
```

The API reuses the exact same `run_screening` pipeline (no duplicated logic).

---

## 11. Reliability & failure modes

| Failure | Behaviour |
|---|---|
| Malformed / unreadable PDF | `parse_status=failed`, recorded, batch continues |
| Empty / image-only file | `parse_status=empty`, rejected with reason |
| Duplicate file (same bytes) | de-duplicated by content hash (`duplicate`) |
| Unsupported extension | skipped, logged |
| Two-column / glued-word PDF | both extractors run; the higher-quality text wins |
| Missing name/email/GitHub | safe fallbacks (email local part, filename) |
| LLM error / invalid JSON | per-resume warning; deterministic score used |
| LLM rate limit | wait-and-resume; never falls back mid-batch |
| GitHub rate limit / private | tagged, neutral placeholder, batch continues |
| Unexpected exception | caught per document; batch never aborts |

Parsers **never raise**. Duplicate detection, per-file isolation, and bounded
concurrency (`MAX_CONCURRENCY`) are the backbone of this.

---

## 12. Tests

```bash
pytest -q          # 66 tests
```

Coverage highlights:

- **Matching precision** — `Java`≠`JavaScript`, `SQL`≠`PostgreSQL`, `CV` trap,
  phrase variants, generic "Tools" ≠ tool-calling.
- **Section detection** — inline `Skills: ...`, own-line headings, preamble.
- **Eligibility** — JS-only rejected, Python-only rejected, JS+Python+AI
  eligible, general-ML project eligible, general-ML *skill* not eligible,
  empty text rejected, `matched_skills` on rejects, and a regression test that
  **LLM projects cannot influence eligibility**.
- **Scoring** — weights sum to 100, agentic > thin wrapper, thin wrapper
  penalised, total clamped, rationale present per category.
- **Gate ordering** — LLM/GitHub are **never called** for eliminated candidates
  (spy clients); GitHub `rate_limited` → tagged + placeholder; `llm_applied`
  flag; prompt is trimmed (an Education-section marker never reaches the LLM).
- **LLM adapter** — rate-limiter pacing, quota wait-then-resume, fail-fast mode,
  budget exhaustion, retry-delay parsing.
- **Ingestion** — corrupt/missing/empty/unsupported/duplicate handling.
- **GitHub scoring** — recency bands, volume boost, repo relevance, forks
  ignored, cap at 10.
- **Pipeline** — counts, ordering, determinism (same input → same output).

---

## 13. Performance & cost

- **Deterministic path is milliseconds per resume** (regex only). A key-less run
  over 50 PDFs is a few seconds.
- **The LLM is the only real cost.** With the gate removing clear rejects, only
  survivors are scored. Free-tier pacing (`12/min`) means a cold run of ~41
  survivors takes several minutes; successful responses are cached so reruns are
  near-instant.
- Measured on the provided set: **50 ingested, 41 eligible, 9 rejected**, gate
  skipped 9 candidates from every network call.
- Biggest remaining latency lever: the model's *thinking* budget / output-token
  caps (see §16).

---

## 14. Design Decisions

### 14.1 Filtering strategy

Eligibility is a **hard, deterministic, positive-only** rule that runs **first**.
The spec's requirement that "hard eligibility checks stay outside the LLM" is
taken literally: the gate consumes only raw-text evidence, and the LLM cannot
grant or revoke eligibility. This makes the filter predictable and unit-testable,
and means eliminated candidates never cost an API call.

The trade-off: because "no AI" is a hard first-pass rule, the AI vocabulary is
load-bearing. To stop silent false negatives, the vocabulary is tiered and
includes glued forms and ML terms, and rejections are auditable from
`matched_skills` + `rejection_reasons`.

### 14.2 Scoring strategy

The rubric is fully deterministic and **evidence-source weighted** — the same
skill earns more when it appears in a project than in a skills list. The AI
category is the only one the LLM refines, and any model score must come with a
rationale and a cited snippet or it is discarded. Penalties target exactly what
the spec calls out: thin LLM/API wrappers and tutorial/undetailed projects. A
framework name in a skills list is capped so it can't reach the top.

### 14.3 LLM usage

Used for **judgement**, not for plumbing. Structured output (Pydantic), a written
rubric, `temperature=0`, small prompts (projects/experience only), an adapter so
the provider is swappable, env-only keys, per-resume failure isolation, response
caching, and wait-and-resume on rate limits. Deterministic mode is a first-class
path, not a degraded one.

### 14.4 GitHub scoring

Lightweight and never fatal. A capped 10 points split 5/5 between recency+volume
and maintained/relevant repos. Failures are tagged, and unknown profiles get a
neutral placeholder rather than a misleading 0 — while a genuinely absent
profile still scores 0, because that is a known fact.

---

## 15. Known limitations

- **Rejected candidates' project names are noisy** (`SOFT SKILLS`, `Live Demo:
  https://…`). They never reach the LLM, so their projects come only from the
  rough deterministic splitter. This does not affect ranking.
- **Two-column PDFs** are largely handled by the dual-extractor, but pathological
  layouts or scanned/image-only resumes still degrade (no OCR).
- **Section detection is heuristic**; unusual headings may fall into `other`
  (still scanned for evidence, but not attributed to a project).
- **GitHub without a token** hits 60 req/hr and gets tagged; results are better
  with a `GITHUB_TOKEN`.
- **LLM score calibration** is prompt-driven (no labelled ground-truth set).

---

## 16. If I Had More Time

1. **Cut LLM latency** — disable/minimize the model's thinking budget and cap
   `max_output_tokens`. This is the largest wall-clock item in a cold run.
2. **Higher throughput** — a second API key/project or batching several resumes
   per "score these projects" call, to escape the per-minute quota.
3. **Incremental runs** — cache the final `CandidateResult` per file hash so
   adding one resume reprocesses only that one.
4. **GraphQL GitHub** — fetch profile + repos + contributions in one request
   instead of three, easing the rate limit.
5. **Calibration set** — a small labelled set of resumes to tune weights and
   thresholds and measure rank correlation against human review.
6. **OCR + table-aware parsing** — Tesseract fallback for scans, and a
   column-aware PDF reader for pathological layouts.
7. **Richer signals** — commit-message topics, `/languages` byte counts,
   stars/forks weighting, portfolio/LinkedIn enrichment.

---

## 17. Reproducing the provided results

```bash
# Deterministic-only (no keys)
python main.py --input ./resumes --output ./output/results.json --no-llm

# Full run (LLM + GitHub); requires GEMINI_API_KEY / GITHUB_TOKEN in .env
python main.py --input ./resumes --output ./output/results.json --csv ./output/results.csv
```

- `resumes/` holds the provided 50-PDF set and is **git-ignored** so candidate
  PII is never committed. `output/results.json` / `output/results.csv` are
  generated from it.
- `resumes_sample/` is a **synthetic** set (50 candidates + edge cases: corrupt
  PDF, empty file, duplicate, unsupported type) produced by
  `tests/fixtures/generate_resumes.py`, useful for demos and regression checks:

```bash
python tests/fixtures/generate_resumes.py --out resumes_sample --count 50
python main.py --input ./resumes_sample --output ./output/results_sample.json
```
