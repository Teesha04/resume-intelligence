"""Output rendering: machine-readable JSON + a concise terminal report + PDFs.

The JSON is the primary deliverable (spec section 7). The terminal report and
PDF are conveniences so a reviewer can see the shortlist and *why* at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape as _esc

from .schemas import ScreeningReport


def _clip(text: object, n: int = 140) -> str:
    """Whitespace-normalise, then truncate at a word boundary with an ellipsis."""
    t = " ".join(str(text or "").split())
    if len(t) <= n:
        return _esc(t)
    cut = t[:n]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return _esc(cut) + "…"


def write_json(report: ScreeningReport, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def write_csv(report: ScreeningReport, output_path: Path) -> Path:
    """Optional flat CSV for spreadsheet review."""
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "candidate_name", "eligible", "det_total_score", "total_score", "llm_delta",
        "ai_project_depth", "ai_project_depth_deterministic", "python_backend", "cloud_fullstack",
        "github", "engineering_depth", "penalties", "matched_skills", "rejection_reasons",
        "source_file",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in report.candidates:
            b = r.score_breakdown
            db = r.deterministic_breakdown
            det = r.deterministic_total_score
            writer.writerow({
                "rank": r.rank if r.rank is not None else "",
                "candidate_name": r.candidate_name,
                "eligible": r.eligible,
                "det_total_score": round(det, 2) if det is not None else "",
                "total_score": round(r.total_score, 2) if r.total_score is not None else "",
                "llm_delta": round((r.total_score - det), 2) if (r.total_score is not None and det is not None) else "",
                "ai_project_depth": b.ai_project_depth if b else "",
                "ai_project_depth_deterministic": db.ai_project_depth if db else "",
                "python_backend": b.python_backend if b else "",
                "cloud_fullstack": b.cloud_fullstack if b else "",
                "github": b.github if b else "",
                "engineering_depth": b.engineering_depth if b else "",
                "penalties": b.penalties if b else "",
                "matched_skills": "; ".join(r.matched_skills),
                "rejection_reasons": "; ".join(r.rejection_reasons),
                "source_file": r.source_file,
            })
    return output_path


def render_console(report: ScreeningReport, top: int = 10) -> str:
    s = report.summary
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(" AI RESUME SCREENING — BATCH SUMMARY")
    lines.append("=" * 78)
    lines.append(
        f" total={s.total_resumes}  parsed_ok={s.parsed_ok}  failed={s.parse_failed}  "
        f"duplicates={s.duplicates_skipped}"
    )
    lines.append(
        f" eligible={s.eligible}  rejected={s.rejected}  llm_used={s.llm_used}"
    )
    lines.append(
        f" funnel: skipped_by_gate={s.llm_skipped_by_gate}  llm_scored={s.llm_scored}  "
        f"rate_limit_waits={s.llm_rate_limit_waits} ({s.llm_wait_seconds}s waited)"
    )
    lines.append(
        f" github: ok={s.github_enriched}  no_profile={s.github_no_profile}  "
        f"tagged(rate_limited={s.github_tagged_rate_limited}, error={s.github_tagged_error})"
    )
    lines.append(f" duration={s.duration_seconds}s")
    lines.append("")

    ranked = [r for r in report.candidates if r.eligible]
    if ranked:
        lines.append(
            f"{'Rank':<5}{'Det':<7}{'Final':<7}{'Candidate':<26}{'AI':<9}"
            f"{'PY':<5}{'Cloud':<6}{'GH':<5}{'Pen'}"
        )
        lines.append("-" * 82)
        for r in ranked[:top]:
            b = r.score_breakdown
            det = r.deterministic_total_score
            det_s = f"{det:.1f}" if det is not None else "-"
            det_ai = f"{r.deterministic_breakdown.ai_project_depth:.0f}" if r.deterministic_breakdown else "-"
            llm_ai = f"{b.ai_project_depth:.0f}"
            lines.append(
                f"{r.rank:<5}{det_s:<7}{r.total_score:<7.1f}{r.candidate_name[:25]:<26}"
                f"{det_ai + '->' + llm_ai:<9}{b.python_backend:<5.0f}"
                f"{b.cloud_fullstack:<6.0f}{b.github:<5.0f}{b.penalties:<.0f}"
            )
        if len(ranked) > top:
            lines.append(f" ... and {len(ranked) - top} more eligible candidate(s)")
    else:
        lines.append("No eligible candidates.")

    rejected = [r for r in report.candidates if not r.eligible]
    if rejected:
        lines.append("")
        lines.append(f"Rejected/unparsed: {len(rejected)}")
        for r in rejected[:top]:
            reason = "; ".join(r.rejection_reasons) or "ineligible"
            lines.append(f"  - {r.candidate_name[:30]:<31} {reason[:60]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def write_pdf(report: ScreeningReport, output_path: Path, top: int = 25) -> Path:
    """Render a human-readable PDF: summary, shortlist (deterministic vs LLM),
    per-candidate score combination, and rejected candidates with reasons."""
    from datetime import datetime

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

    NAVY = colors.HexColor("#1f3b57")
    RED = colors.HexColor("#7a1f1f")
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=base["Heading1"], fontSize=17, textColor=NAVY, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, textColor=NAVY,
                        spaceBefore=10, spaceAfter=4)
    small = ParagraphStyle("small", parent=base["Normal"], fontSize=8, textColor=colors.grey)
    body = ParagraphStyle("body", parent=base["Normal"], fontSize=9, leading=12, spaceAfter=3)
    cell = ParagraphStyle("cell", parent=base["Normal"], fontSize=7.5, leading=9.5)

    def T(rows, widths, header=NAVY, center_from=99, extra=None):
        style = [
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), header),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (center_from, 1), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
        ]
        if extra:
            style.extend(extra)
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle(style))
        return t

    s = report.summary
    eligible = [c for c in report.candidates if c.eligible]
    rejected = [c for c in report.candidates if not c.eligible]
    story: list = []

    # ---- Header + summary -------------------------------------------------
    story.append(Paragraph("AI Resume Screening &amp; Ranking — Results", h1))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
        f"LLM used: {s.llm_used} &nbsp;|&nbsp; duration: {s.duration_seconds}s", small))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Batch summary", h2))
    story.append(T([
        ["Total resumes", str(s.total_resumes), "Eligible", str(s.eligible)],
        ["Parsed OK", str(s.parsed_ok), "Rejected", str(s.rejected)],
        ["Parse failed", str(s.parse_failed), "Duplicates skipped", str(s.duplicates_skipped)],
        ["LLM scored (survivors)", str(s.llm_scored), "Skipped by hard gate", str(s.llm_skipped_by_gate)],
        ["GitHub enriched", str(s.github_enriched), "GitHub no profile", str(s.github_no_profile)],
        ["GitHub tagged (rate-limited)", str(s.github_tagged_rate_limited),
         "GitHub tagged (error)", str(s.github_tagged_error)],
    ], [50 * mm, 28 * mm, 50 * mm, 28 * mm], center_from=1))

    # ---- Shortlist: deterministic vs LLM vs final -------------------------
    story.append(Paragraph(f"Shortlist — eligible candidates ({len(eligible)})", h2))
    story.append(Paragraph(
        "<b>Det</b> = deterministic baseline (no LLM) &nbsp;•&nbsp; <b>Final</b> = with LLM "
        "project scoring &nbsp;•&nbsp; <b>Δ</b> = LLM contribution &nbsp;•&nbsp; "
        "<b>AI</b> shows det→llm depth. Every other category is deterministic and unchanged.",
        small))
    story.append(Spacer(1, 3))
    rows = [["#", "Candidate", "Det", "Final", "Δ", "AI (det→llm)", "PY", "Cloud", "GH", "Pen"]]
    for c in eligible[:top]:
        b = c.score_breakdown
        det = c.deterministic_total_score
        det_s = f"{det:.1f}" if det is not None else "—"
        delta = f"{(c.total_score - det):+.1f}" if det is not None else "—"
        det_ai = c.deterministic_breakdown.ai_project_depth if c.deterministic_breakdown else None
        llm_ai = c.llm_ai_project_depth if c.llm_ai_project_depth is not None else b.ai_project_depth
        ai_s = f"{det_ai:.0f}→{llm_ai:.0f}" if det_ai is not None else f"{llm_ai:.0f}"
        rows.append([str(c.rank), Paragraph(_esc(c.candidate_name[:32]), cell), det_s,
                     f"{c.total_score:.1f}", delta, ai_s, f"{b.python_backend:.0f}",
                     f"{b.cloud_fullstack:.0f}", f"{b.github:.0f}", f"{b.penalties:.0f}"])
    story.append(T(rows, [8 * mm, 44 * mm, 13 * mm, 13 * mm, 12 * mm, 24 * mm,
                          11 * mm, 14 * mm, 11 * mm, 11 * mm]))
    if len(eligible) > top:
        story.append(Paragraph(f"… and {len(eligible) - top} more eligible candidate(s).", small))

    # ---- How deterministic + LLM combine (per candidate) ------------------
    story.append(PageBreak())
    story.append(Paragraph("How the final score is built (deterministic vs LLM)", h2))
    story.append(Paragraph(
        "Every category is computed deterministically from resume evidence. The LLM refines "
        "<b>only</b> the AI / Agentic project-depth category, scoring each AI project 0-10 with a "
        "rationale and cited evidence. The <b>final</b> score is the deterministic score with the "
        "AI-depth category replaced by the LLM judgement; all other categories are identical. "
        "For the score breakdown below, the AI-depth row shows what the deterministic signal "
        "estimate gave versus what the LLM decided.", body))
    story.append(Spacer(1, 4))

    for c in eligible[: min(10, len(eligible))]:
        b, db = c.score_breakdown, c.deterministic_breakdown
        det = c.deterministic_total_score
        delta = (c.total_score - det) if det is not None else None
        header_line = f"<b>#{c.rank} {_esc(c.candidate_name)}</b> — final {c.total_score:.1f}"
        if delta is not None:
            header_line += (f" &nbsp;·&nbsp; deterministic {det:.1f} "
                            f"&nbsp;·&nbsp; LLM {delta:+.1f}")
        block = [Paragraph(header_line, body)]
        if db:
            def row(label, d, l):
                return [label, f"{d:.1f}", f"{l:.1f}", f"{l - d:+.1f}"]

            rows = [["Category", "Deterministic", "With LLM", "Change"],
                    row("AI project depth (40)", db.ai_project_depth, b.ai_project_depth),
                    row("Python & backend (30)", db.python_backend, b.python_backend),
                    row("Cloud / full-stack (15)", db.cloud_fullstack, b.cloud_fullstack),
                    row("GitHub (10)", db.github, b.github),
                    row("Engineering depth (5)", db.engineering_depth, b.engineering_depth),
                    row("Penalties", db.penalties, b.penalties),
                    ["TOTAL", f"{db.total():.1f}", f"{b.total():.1f}", f"{(b.total() - db.total()):+.1f}"]]
            block.append(T(rows, [58 * mm, 30 * mm, 30 * mm, 26 * mm], center_from=1, extra=[
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fff3cd")),
                ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#e8eef4")),
                ("FONTNAME", (0, len(rows) - 1), (-1, len(rows) - 1), "Helvetica-Bold"),
            ]))
            ai_note = (b.rationale.get("ai_project_depth") or ["—"])[0]
            block.append(Paragraph(f"<i>AI depth:</i> {_clip(ai_note, 170)}", small))
        block.append(Paragraph(f"<i>Project:</i> {_clip(c.project_summary, 210)}", small))
        story.append(KeepTogether(block + [Spacer(1, 7)]))

    # ---- Rejected with reasons -------------------------------------------
    if rejected:
        story.append(PageBreak())
        story.append(Paragraph(f"Rejected candidates ({len(rejected)}) — and why", h2))
        story.append(Paragraph(
            "Eligibility is a hard, deterministic filter: a candidate must show <b>Python</b> AND "
            "<b>AI/agentic</b> evidence. Reasons name only the unmet requirement — never the "
            "presence of an “unwanted” skill. <i>matched_skills</i> is the evidence that was found.",
            small))
        story.append(Spacer(1, 3))
        rows = [["Candidate", "Rejection reason(s)", "Matched skills (what we did find)"]]
        for c in rejected:
            rows.append([
                Paragraph(_esc(c.candidate_name[:32]), cell),
                Paragraph(_esc("; ".join(c.rejection_reasons) or "ineligible"), cell),
                Paragraph(_esc(", ".join(c.matched_skills[:10]) or "—"), cell),
            ])
        story.append(T(rows, [40 * mm, 62 * mm, 68 * mm], header=RED))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Eligibility is a hard, deterministic filter (Python AND AI/agentic evidence). Scores are "
        "evidence-source weighted; the LLM contributes AI project depth with cited evidence. GitHub "
        "rate-limited/errored profiles are tagged and given a neutral placeholder.", small))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillGray(0.45)
        canvas.drawString(15 * mm, 9 * mm, "AI Resume Screening & Ranking — Results")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=15 * mm,
        title="AI Resume Screening — Results", author="Resume Screener",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path
