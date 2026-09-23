"""Output rendering: machine-readable JSON + a concise terminal report.

The JSON is the primary deliverable (spec section 7). The terminal report is a
small convenience so a reviewer can see the shortlist at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import ScreeningReport


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
        "rank", "candidate_name", "eligible", "total_score",
        "ai_project_depth", "python_backend", "cloud_fullstack",
        "github", "engineering_depth", "penalties",
        "matched_skills", "source_file",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in report.candidates:
            b = r.score_breakdown
            writer.writerow({
                "rank": r.rank if r.rank is not None else "",
                "candidate_name": r.candidate_name,
                "eligible": r.eligible,
                "total_score": r.total_score if r.total_score is not None else "",
                "ai_project_depth": b.ai_project_depth if b else "",
                "python_backend": b.python_backend if b else "",
                "cloud_fullstack": b.cloud_fullstack if b else "",
                "github": b.github if b else "",
                "engineering_depth": b.engineering_depth if b else "",
                "penalties": b.penalties if b else "",
                "matched_skills": "; ".join(r.matched_skills),
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
        lines.append(f"{'Rank':<5}{'Score':<7}{'Candidate':<28}{'AI':<5}{'PY':<5}{'Cloud':<6}{'GH':<5}{'Pen'}")
        lines.append("-" * 78)
        for r in ranked[:top]:
            b = r.score_breakdown
            lines.append(
                f"{r.rank:<5}{r.total_score:<7.1f}{r.candidate_name[:27]:<28}"
                f"{b.ai_project_depth:<5.0f}{b.python_backend:<5.0f}"
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


def write_pdf(report: ScreeningReport, output_path: Path, top: int = 20) -> Path:
    """Render a human-readable PDF report (summary + shortlist + rejections).

    Optional: requires `reportlab` (declared in requirements.txt).
    """
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9)
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=10)

    s = report.summary
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
        title="AI Resume Screening — Results",
    )
    story: list = []

    story.append(Paragraph("AI Resume Screening &amp; Ranking — Results", h1))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
        f"LLM used: {s.llm_used} &nbsp;|&nbsp; duration: {s.duration_seconds}s", small))
    story.append(Spacer(1, 6))

    # --- Batch summary -----------------------------------------------------
    story.append(Paragraph("Batch summary", h2))
    summary_rows = [
        ["Total resumes", str(s.total_resumes), "Eligible", str(s.eligible)],
        ["Parsed OK", str(s.parsed_ok), "Rejected", str(s.rejected)],
        ["Parse failed", str(s.parse_failed), "Duplicates skipped", str(s.duplicates_skipped)],
        ["LLM scored (survivors)", str(s.llm_scored), "Skipped by hard gate", str(s.llm_skipped_by_gate)],
        ["GitHub enriched", str(s.github_enriched), "GitHub no profile", str(s.github_no_profile)],
        ["GitHub tagged (rate-limited)", str(s.github_tagged_rate_limited),
         "GitHub tagged (error)", str(s.github_tagged_error)],
    ]
    t = Table(summary_rows, colWidths=[52 * mm, 28 * mm, 52 * mm, 28 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)

    # --- Shortlist ---------------------------------------------------------
    eligible = [c for c in report.candidates if c.eligible]
    story.append(Paragraph(f"Shortlist — eligible candidates ({len(eligible)})", h2))
    header = ["#", "Candidate", "Total", "AI", "PY", "Cloud", "GH", "Eng", "Pen"]
    rows = [header]
    for c in eligible[:top]:
        b = c.score_breakdown
        rows.append([
            str(c.rank), c.candidate_name[:30],
            f"{c.total_score:.1f}", f"{b.ai_project_depth:.0f}", f"{b.python_backend:.0f}",
            f"{b.cloud_fullstack:.0f}", f"{b.github:.0f}", f"{b.engineering_depth:.0f}",
            f"{b.penalties:.0f}",
        ])
    t = Table(rows, colWidths=[9 * mm, 62 * mm, 15 * mm, 13 * mm, 13 * mm, 16 * mm, 12 * mm, 13 * mm, 13 * mm],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    if len(eligible) > top:
        story.append(Paragraph(f"… and {len(eligible) - top} more eligible candidate(s).", small))

    # --- Why the top candidates ranked here --------------------------------
    story.append(Paragraph("Score breakdown — top candidates", h2))
    for c in eligible[: min(10, len(eligible))]:
        b = c.score_breakdown
        story.append(Paragraph(f"<b>#{c.rank} {c.candidate_name}</b> — {c.total_score:.1f}/100", body))
        story.append(Paragraph(f"<i>Project:</i> {c.project_summary[:220]}", cell))
        if c.github_summary:
            story.append(Paragraph(f"<i>GitHub:</i> {c.github_summary}", cell))
        for line in b.rationale.get("ai_project_depth", [])[:4]:
            story.append(Paragraph(f"• {line[:220]}", cell))
        for line in b.rationale.get("penalties", []):
            if "No project-quality" not in line:
                story.append(Paragraph(f"• penalty: {line[:200]}", cell))
        story.append(Spacer(1, 5))

    # --- Rejected ----------------------------------------------------------
    rejected = [c for c in report.candidates if not c.eligible]
    if rejected:
        story.append(Paragraph(f"Rejected / unparsed ({len(rejected)})", h2))
        rows = [["Candidate", "Reason"]]
        for c in rejected:
            rows.append([c.candidate_name[:32], Paragraph("; ".join(c.rejection_reasons), cell)])
        t = Table(rows, colWidths=[60 * mm, 108 * mm], repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7a1f1f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Eligibility is a hard, deterministic filter (Python AND AI/agentic evidence). "
        "Scores are evidence-source weighted; the LLM contributes AI project depth with cited "
        "evidence. GitHub rate-limited/errored profiles are tagged and given a neutral placeholder.",
        small))

    doc.build(story)
    return output_path
