"""Resume section detection.

Resumes don't share a layout or section naming (spec section 2), so we detect
headings heuristically and fall back gracefully:

  * A heading is a short line (<~60 chars) that is a known alias, optionally
    followed by a colon, and not ending in a sentence-like period.
  * Content before the first heading is tagged SUMMARY.
  * Unrecognised headings become OTHER but still get scanned for evidence.

The output lets us attach a *source* to every skill mention, which drives the
confidence weighting in scoring.
"""

from __future__ import annotations

import re

from ..schemas import SECTION_TO_SOURCE, EvidenceSource

# Canonical section -> aliases (lower-cased, matched loosely).
SECTION_ALIASES: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "profile", "about me", "about", "objective", "career objective"],
    "skills": ["skills", "technical skills", "technologies", "tech stack", "technical proficiencies",
               "core competencies", "skills & interests", "skills and interests", "tools"],
    "projects": ["projects", "personal projects", "academic projects", "key projects", "selected projects",
                 "technical projects", "side projects", "notable projects"],
    "experience": ["experience", "work experience", "professional experience", "internship", "internships",
                   "internship experience", "employment", "work history", "relevant experience"],
    "education": ["education", "academics", "academic background", "educational background"],
    "achievements": ["achievements", "awards", "honors", "honours", "certifications", "publications"],
    "other": [],  # catch-all
}

# Map canonical section -> evidence source used for confidence weighting.
# (Defined in schemas to avoid import cycles.)


def _normalise_heading(line: str) -> str:
    h = line.strip().lower()
    h = h.strip("•-–—*#: ")
    h = re.sub(r"\s+", " ", h)
    return h


def _match_section(line: str) -> str | None:
    """Return canonical section for a heading-like line, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    # Headings are not prose: reject lines with many words or trailing period.
    if stripped.endswith(".") and len(stripped.split()) > 3:
        return None
    norm = _normalise_heading(stripped)
    if not norm:
        return None
    for section, aliases in SECTION_ALIASES.items():
        if section == "other":
            continue
        for alias in aliases:
            if norm == alias or norm.startswith(alias + " ") or norm == alias + "s":
                return section
    return None


def _split_inline_heading(line: str) -> tuple[str | None, str | None]:
    """Handle 'Heading: content' on one line, e.g. 'Skills: Python, FastAPI'.

    Returns (section, remainder) when the prefix is a known heading.
    """
    m = re.match(r"^\s*([A-Za-z][A-Za-z &/+\-]{1,40}?)\s*:\s*(.+)$", line.strip())
    if not m:
        return None, None
    section = _match_section(m.group(1))
    if section is None:
        return None, None
    return section, m.group(2).strip()


def line_assignments(text: str) -> list[tuple[str, str]]:
    """Return [(section, line), ...] in document order."""
    assignments: list[tuple[str, str]] = []
    current = "summary"  # preamble before the first heading
    for line in text.splitlines():
        inline_section, remainder = _split_inline_heading(line)
        if inline_section is not None:
            current = inline_section
            if remainder:
                assignments.append((inline_section, remainder))
            continue
        section = _match_section(line)
        if section is not None:
            current = section
            continue
        assignments.append((current, line))
    return assignments


def split_sections(text: str) -> dict[str, str]:
    """Return {canonical_section: joined_text}."""
    buckets: dict[str, list[str]] = {k: [] for k in SECTION_ALIASES}
    for section, line in line_assignments(text):
        buckets.setdefault(section, []).append(line)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}


def source_for_section(section: str) -> EvidenceSource:
    return SECTION_TO_SOURCE.get(section, EvidenceSource.UNKNOWN)
