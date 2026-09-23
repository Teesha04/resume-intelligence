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
