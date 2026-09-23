"""Process / design report PDF.

Generates a narrative PDF that explains WHAT the system did, WHY each decision
was made, and the full end-to-end process — with the real numbers from a run.

Usage (library):
    from src.process_report import write_process_pdf
    write_process_pdf(report, "output/process_report.pdf")

Usage (CLI helper):
    python -m src.process_report output/results.json output/process_report.pdf
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape as esc

from .schemas import ScreeningReport


def write_process_pdf(report: ScreeningReport, output_path: Path, top: int = 25) -> Path:
    """Build a multi-page process/design PDF from a ScreeningReport."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = getSampleStyleSheet()
    NAVY = colors.HexColor("#1f3b57")
    GREY = colors.HexColor("#555555")
    st_title = ParagraphStyle("t", parent=base["Title"], fontSize=24, leading=28, textColor=NAVY)
    st_sub = ParagraphStyle("s", parent=base["Normal"], fontSize=11, textColor=GREY, alignment=1)
    st_h1 = ParagraphStyle("h1", parent=base["Heading1"], fontSize=15, textColor=NAVY, spaceBefore=10, spaceAfter=6)
    st_h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, textColor=NAVY, spaceBefore=8, spaceAfter=4)
    st_p = ParagraphStyle("p", parent=base["Normal"], fontSize=9.5, leading=13, spaceAfter=5)
    st_small = ParagraphStyle("sm", parent=base["Normal"], fontSize=8, textColor=GREY, leading=10)
    st_cell = ParagraphStyle("c", parent=base["Normal"], fontSize=8, leading=10)
    st_bullet = ParagraphStyle("b", parent=st_p, leftIndent=10, bulletIndent=2, spaceAfter=2)

    def P(text: str) -> Paragraph:
        return Paragraph(text, st_p)

    def B(items: Iterable[str]) -> list:
        return [Paragraph(f"• {t}", st_bullet) for t in items]

    def table(rows, widths, header_color=NAVY, align_center_from=2):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), header_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (align_center_from, 1), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    s = report.summary
    eligible = [c for c in report.candidates if c.eligible]
    rejected = [c for c in report.candidates if not c.eligible]
    story: list = []

    # =====================================================================
    # Cover
    # =====================================================================
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph("AI Resume Screening &amp; Ranking", st_title))
    story.append(Paragraph("What it does, why it works this way, and how it was built", st_sub))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Process &amp; design report &nbsp;•&nbsp; generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle("c2", parent=st_sub, fontSize=9)))
    story.append(Spacer(1, 20))
    cover = [
        ["Resumes ingested", str(s.total_resumes)],
        ["Eligible / rejected", f"{s.eligible} / {s.rejected}"],
        ["LLM used", str(s.llm_used)],
        ["Skipped by hard gate (no LLM/GitHub)", str(s.llm_skipped_by_gate)],
        ["GitHub enriched / no profile", f"{s.github_enriched} / {s.github_no_profile}"],
        ["GitHub tagged (rate-limited/error)",
         f"{s.github_tagged_rate_limited} / {s.github_tagged_error}"],
        ["Run duration (s)", str(s.duration_seconds)],
    ]
    story.append(table([["Metric", "Value"]] + cover, [110 * mm, 60 * mm], align_center_from=1))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This document explains the objective, the end-to-end pipeline, the reasoning behind "
        "each design decision, and the actual results of the run over the provided resume set.",
        st_small))
    story.append(PageBreak())

    # =====================================================================
    # 1. Objective
    # =====================================================================
    story.append(Paragraph("1. Objective", st_h1))
    story.append(P(
        "Build a small, production-minded backend that ingests a folder of resumes, filters "
        "candidates against a minimum Python + AI/agentic bar, scores the survivors on a "
        "100-point rubric, enriches them with public GitHub activity, and returns a ranked "
        "shortlist in which <b>every score is explainable and backed by resume evidence</b>."))
    story.append(P("Explicit constraints respected throughout:"))
    story.extend(B([
        "A CLI is sufficient; no frontend, no auth, no database, no vector store, no deployment.",
        "A simple, correct, explainable backend beats an over-engineered application.",
        "The ranking must be reproducible and testable; hard eligibility checks stay outside the LLM.",
        "One bad resume, one failed model call, or one rate-limited API must not fail the batch.",
    ]))

    # =====================================================================
    # 2. Pipeline
    # =====================================================================
    story.append(Paragraph("2. The pipeline, end to end", st_h1))
    flow = [
        ["Stage", "What happens", "Why it is here"],
        ["1. Ingest", "Discover files, de-duplicate by content hash, parse PDF/DOCX/TXT",
         "Failures isolated per file so a malformed resume can't abort the batch"],
        ["2. Deterministic extract", "Regex + vocabulary: name, email, GitHub, skills, rough projects, evidence",
         "Cheap, always available, and gives the LLM a validated base instead of a blank slate"],
        ["3. HARD GATE (Python AND AI)", "Eligibility decided on raw-text evidence only",
         "The cheapest possible filter, run first; eliminated candidates cost nothing"],
        ["4. GitHub enrichment", "Only for survivors with a username; never eliminates",
         "An additional positive signal, not an eligibility requirement"],
        ["5. LLM scoring", "Optional: scores each AI project's depth + evidence",
         "Semantic judgement where it adds the most value, kept off the critical path"],
        ["6. Score + rank", "Combine categories, apply penalties, clamp 0-100, rank",
         "One deterministic final number per candidate with a full rationale trail"],
    ]
    story.append(table(flow, [32 * mm, 66 * mm, 72 * mm], align_center_from=99))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Why this order matters", st_h2))
    story.extend(B([
        "<b>Gate before the LLM:</b> a candidate with no Python or no AI evidence is eliminated "
        "before any network call, so the model never sees candidates that cannot rank.",
        "<b>Frozen eligibility:</b> because eligibility is computed before the LLM, the model can "
        "neither grant nor revoke eligibility — the hard filter is provably outside the model.",
        "<b>GitHub before the LLM:</b> independent enrichment that overlaps with LLM pacing.",
    ]))

    # =====================================================================
    # 3. Eligibility
    # =====================================================================
    story.append(Paragraph("3. Eligibility — the hard gate", st_h1))
    story.append(P(
        "A candidate is <b>eligible only if both</b> hold, decided by deterministic code "
        "(<i>src/eligibility/rules.py</i>) on raw-text evidence:"))
    story.extend(B([
        "<b>Python evidence</b> — Python (or a Python-only framework such as FastAPI, Django, "
        "pandas, asyncio) appears as a real token.",
        "<b>AI/agentic evidence</b> — a core LLM/agentic signal, or a genuine AI project "
        "(agentic or general ML/CV/NLP) inside a project/experience section.",
    ]))
    story.append(Paragraph("Positive-only, never negative", st_h2))
    story.append(P(
        "The gate asks only “is Python present?” and “is AI present?”. It never rejects *because* "
        "JavaScript, Java, React or Next.js is present — those only contribute supporting score and "
        "appear in <i>matched_skills</i>. Rejection reasons name only the unmet requirement."))
    cases = [
        ["Resume profile", "Outcome"],
        ["JavaScript / React only", "Rejected — missing Python AND AI"],
        ["Java / Spring Boot / React only", "Rejected — missing Python AND AI"],
        ["JavaScript + React + Python + AI", "Eligible (JS is irrelevant to the gate)"],
        ["Python + FastAPI, no AI", "Rejected — missing AI"],
        ["AI project written in JS/TS, no Python", "Rejected — Python is a hard requirement"],
    ]
    story.append(table(cases, [95 * mm, 75 * mm], align_center_from=99))
    story.append(Spacer(1, 5))
    story.append(Paragraph("AI evidence tiers (why a coursework mention can't rescue a profile)", st_h2))
    tiers = [
        ["Tier", "Examples", "Counts as AI?"],
        ["1 — core LLM/agentic", "LangChain, LangGraph, Google ADK, RAG, embeddings, vector search, "
         "tool calling, multi-agent, evaluation, fine-tuning, vLLM", "Yes — anywhere"],
        ["2 — AI ecosystem", "LLM, foundation model, GenAI, Hugging Face, Bedrock, SageMaker, ASR/TTS",
         "Yes — anywhere"],
        ["3 — general ML/DS", "machine learning, deep learning, CNN/RNN, XGBoost, scikit-learn, "
         "TensorFlow, PyTorch, NLP, computer vision", "Only inside a project / experience section"],
    ]
    story.append(table(tiers, [30 * mm, 100 * mm, 40 * mm], align_center_from=99))
    story.append(Spacer(1, 5))
    story.append(Paragraph("Matching precision (false positives are the enemy)", st_h2))
    story.extend(B([
        "<b>Java ≠ JavaScript</b>, <b>SQL ≠ PostgreSQL</b> — enforced by word boundaries.",
        "<b>“CV” is never matched</b> — it means curriculum vitae on nearly every resume; only "
        "“computer vision” counts.",
        "“embedded systems” ≠ “embedding”; “data model” ≠ AI model; “travel agent” is not an AI agent.",
        "Glued and hyphenated forms tolerated: “MachineLearning”, “multi-agent”, “multi agent”.",
    ]))

    # =====================================================================
    # 4. Scoring
    # =====================================================================
    story.append(Paragraph("4. Scoring — the 100-point rubric", st_h1))
    weights = [
        ["Category", "Weight", "How it is earned"],
        ["AI / Agentic / RAG project depth", "40",
         "best project depth (70%) + breadth of solid projects (15%) + rubric signals (15%)"],
        ["Python & backend engineering", "30", "weighted terms × evidence-source multiplier"],
        ["Cloud / deployment / full-stack", "15", "Docker/GCP/AWS/K8s/CI-CD; React/Next supporting"],
        ["GitHub activity", "10", "0-5 recent activity + 0-5 maintained/relevant repositories"],
        ["Engineering depth signals", "5", "testing, architecture, caching, queues, observability…"],
        ["Project-quality penalties", "−5…−15", "thin LLM/API wrappers; tutorial/undetailed projects"],
    ]
    story.append(table(weights, [52 * mm, 18 * mm, 100 * mm], align_center_from=1))
    story.append(Spacer(1, 6))
    story.append(Paragraph("The core idea: evidence-source weighting", st_h2))
    story.append(P(
        "Every matched term records <b>where</b> it was found, and the same term earns different "
        "credit. This directly implements the spec's requirement to reward evidence in "
        "projects/internships over keyword-only skill lists."))
    src = [
        ["Where the term appears", "Multiplier"],
        ["Project description", "1.0"],
        ["Internship / work experience", "0.9"],
        ["Summary / preamble", "0.5"],
        ["Skills list only", "0.4"],
    ]
    story.append(table(src, [110 * mm, 60 * mm], align_center_from=1))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Why these penalties", st_h2))
    story.extend(B([
        "<b>Thin wrapper (−5…−15):</b> an “AI project” that is really a single LLM/API call with no "
        "retrieval, orchestration, tools, evaluation, state or backend workflow.",
        "<b>Tutorial / undetailed (−3):</b> named like a tutorial, or described too briefly to show "
        "implementation detail or ownership.",
        "<b>Framework name in a skills list only</b> is capped at 0.25 × 40, so it can never reach "
        "the top on naming alone.",
        "A classical ML project (e.g. a CNN) is <b>not</b> a thin wrapper.",
    ]))

    # =====================================================================
    # 5. LLM
    # =====================================================================
    story.append(Paragraph("5. How and why the LLM is used", st_h1))
    story.extend(B([
        "<b>Scope:</b> score each AI project's depth (0-10) and return the rubric signals, a short "
        "rationale, and a short cited-evidence snippet. It does not decide eligibility and does not "
        "extract name/email/skills (those stay deterministic).",
        "<b>Structured output:</b> the model is given a Pydantic response schema; the JSON is "
        "validated locally. Invalid output is discarded and the deterministic value is used.",
        "<b>Evidence required:</b> a score without a rationale/citation is not trusted.",
        "<b>Small prompts:</b> the call receives only the PROJECTS + EXPERIENCE context, not the "
        "whole resume, and returns only project scores.",
        "<b>Adapter:</b> one interface (<i>LLMClient</i>) with a Gemini implementation and a null "
        "fallback; swapping provider is one class. Keys come only from the environment.",
        "<b>Rate limits:</b> calls are paced under the tier's per-minute quota, and on a quota error "
        "the adapter waits for the reset and resumes the same request (printing a clear message) "
        "rather than silently falling back to deterministic scoring.",
        "<b>Failure isolation:</b> one failed call degrades a single resume and never the batch.",
        "<b>Zero-key mode:</b> without an API key the system still produces a full ranking via a "
        "transparent signal-based quality estimate.",
    ]))

    # =====================================================================
    # 6. GitHub
    # =====================================================================
    story.append(Paragraph("6. GitHub enrichment", st_h1))
    story.extend(B([
        "Public REST API only; an optional token (from env) raises the limit from 60 to 5000 req/hr.",
        "Signals: recency and volume of public events (falling back to repository push dates), "
        "maintained repositories, and Python/AI-relevant repositories.",
        "Scoring: 0-5 activity + 0-5 repositories, hard cap 10.",
        "It never affects eligibility; a missing/private/rate-limited profile is a status, not a failure.",
        "<b>Tagging:</b> a rate-limited or errored profile is <i>unknown</i>, not inactive, so it is "
        "tagged and given a deterministic neutral placeholder (5/10) instead of 0. A genuinely "
        "absent profile still scores 0, because that is a known fact. No random values — they would "
        "break reproducibility.",
    ]))

    story.append(PageBreak())

    # =====================================================================
    # 7. Results
    # =====================================================================
    story.append(Paragraph("7. Results of this run", st_h1))
    story.append(P(
        f"Of <b>{s.total_resumes}</b> resumes, <b>{s.eligible}</b> were eligible and "
        f"<b>{s.rejected}</b> were rejected. The hard gate removed "
        f"<b>{s.llm_skipped_by_gate}</b> candidate(s) from every network call; "
        f"<b>{s.llm_scored}</b> survivor(s) received LLM project scoring. "
        f"GitHub was enriched for <b>{s.github_enriched}</b> candidate(s); "
        f"<b>{s.github_no_profile}</b> had no profile and "
        f"<b>{s.github_tagged_rate_limited + s.github_tagged_error}</b> were tagged (rate-limited/error)."))
    story.append(Spacer(1, 4))

    header = ["#", "Candidate", "Total", "AI", "PY", "Cloud", "GH", "Eng", "Pen"]
    rows = [header]
    for c in eligible[:top]:
        b = c.score_breakdown
        rows.append([
            str(c.rank), Paragraph(esc(c.candidate_name[:32]), st_cell),
            f"{c.total_score:.1f}", f"{b.ai_project_depth:.0f}", f"{b.python_backend:.0f}",
            f"{b.cloud_fullstack:.0f}", f"{b.github:.0f}", f"{b.engineering_depth:.0f}",
            f"{b.penalties:.0f}",
        ])
    story.append(Paragraph(f"Shortlist — top {min(top, len(eligible))} eligible candidates", st_h2))
    story.append(table(rows, [8 * mm, 60 * mm, 14 * mm, 12 * mm, 12 * mm, 15 * mm, 11 * mm, 12 * mm, 12 * mm]))
    if len(eligible) > top:
        story.append(Paragraph(f"… and {len(eligible) - top} more eligible candidate(s).", st_small))

    story.append(Paragraph("Why the top candidates ranked here", st_h2))
    for c in eligible[:5]:
        b = c.score_breakdown
        block = [
            Paragraph(f"<b>#{c.rank} {esc(c.candidate_name)}</b> — {c.total_score:.1f}/100", st_p),
            Paragraph(f"<i>Project:</i> {esc(c.project_summary[:240])}", st_small),
        ]
        if c.github_summary:
            block.append(Paragraph(f"<i>GitHub:</i> {esc(c.github_summary)}", st_small))
        for line in b.rationale.get("ai_project_depth", [])[:3]:
            block.append(Paragraph(f"• {esc(line[:220])}", st_small))
        for line in b.rationale.get("penalties", []):
            if "No project-quality" not in line:
                block.append(Paragraph(f"• penalty: {esc(line[:200])}", st_small))
        story.append(KeepTogether(block + [Spacer(1, 5)]))

    if rejected:
        story.append(Paragraph(f"Rejected / unparsed ({len(rejected)}) and why", st_h2))
        rows = [["Candidate", "Reason(s)"]]
        for c in rejected:
            rows.append([Paragraph(esc(c.candidate_name[:34]), st_cell),
                         Paragraph(esc("; ".join(c.rejection_reasons)), st_cell)])
        story.append(table(rows, [62 * mm, 108 * mm], header_color=colors.HexColor("#7a1f1f"),
                           align_center_from=99))

    # =====================================================================
    # 8. Reliability
    # =====================================================================
    story.append(Paragraph("8. Reliability — what could go wrong, and what happens", st_h1))
    rel = [
        ["Failure", "Behaviour"],
        ["Malformed / unreadable PDF", "Recorded as failed; batch continues"],
        ["Empty / image-only file", "Recorded as empty; rejected with a reason"],
        ["Duplicate file (same bytes)", "De-duplicated by content hash"],
        ["Two-column / glued-word PDF", "Both extractors run; the higher-quality text wins"],
        ["Missing name / email / GitHub", "Safe fallbacks (email local part, filename)"],
        ["LLM error or invalid JSON", "Per-resume warning; deterministic score used"],
        ["LLM rate limit", "Waits for reset and resumes; never derails the batch"],
        ["GitHub rate limit / private", "Tagged + neutral placeholder; batch continues"],
        ["Unexpected exception", "Caught per document; batch never aborts"],
    ]
    story.append(table(rel, [64 * mm, 106 * mm], align_center_from=99))

    # =====================================================================
    # 9. Limitations & next steps
    # =====================================================================
    story.append(Paragraph("9. Limitations and what I would do next", st_h1))
    story.append(Paragraph("Known limitations", st_h2))
    story.extend(B([
        "Rejected candidates' project names are noisy — they never reach the LLM, so their projects "
        "come only from the rough deterministic splitter (does not affect ranking).",
        "Two-column PDFs are largely handled, but scans/image-only resumes are not (no OCR).",
        "Section detection is heuristic; unusual headings may land in a catch-all bucket.",
        "GitHub without a token hits the 60 req/hr limit and gets tagged.",
        "LLM score calibration is prompt-driven (no labelled ground-truth set).",
    ]))
    story.append(Paragraph("If I had more time", st_h2))
    story.extend(B([
        "Cut LLM latency: disable/minimize the model's thinking budget and cap output tokens.",
        "Higher throughput: a second key/model or batching several resumes per scoring call.",
        "Incremental runs: cache the final per-candidate result by file hash.",
        "One-request GraphQL GitHub instead of three REST calls.",
        "A labelled calibration set to tune weights and measure rank correlation vs human review.",
        "OCR and column-aware parsing for pathological layouts.",
    ]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Eligibility is a hard, deterministic filter (Python AND AI/agentic evidence). Scores are "
        "evidence-source weighted; the LLM contributes AI project depth with cited evidence. "
        "GitHub rate-limited/errored profiles are tagged and given a neutral placeholder.",
        st_small))

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
        title="AI Resume Screening — Process & Design Report", author="Resume Screener",
    )
    doc.build(story)
    return output_path


if __name__ == "__main__":  # pragma: no cover - manual helper
    import sys

    from .schemas import ScreeningReport

    if len(sys.argv) < 3:
        print("usage: python -m src.process_report <results.json> <out.pdf>")
        raise SystemExit(2)
    rep = ScreeningReport.model_validate_json(Path(sys.argv[1]).read_text())
    out = write_process_pdf(rep, Path(sys.argv[2]))
    print(f"wrote {out}")
