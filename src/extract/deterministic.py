"""Deterministic (no-LLM) extraction.

This pass always runs. It guarantees the pipeline produces usable output even
when no API key is configured, and it gives the LLM pass a validated skeleton
to refine rather than a blank slate (cheaper + more reliable).

Extracted deterministically:
  * email, GitHub profile URL/username, candidate name
  * skills (canonicalised against vocab.py)
  * python / AI / supporting evidence with source + context
  * a rough project list (title + description + technologies), refined later
"""

from __future__ import annotations

import re
from pathlib import Path

from ..matching import match_terms, scan_evidence
from ..schemas import (
    Document,
    Evidence,
    EvidenceSource,
    ExtractedResume,
    Project,
    ProjectDomain,
)
from ..vocab import (
    AGENTIC_CORE_TERMS,
    AI_TERMS,
    ALL_SKILL_GROUPS,
    BACKEND_TERMS,
    CLOUD_TERMS,
    ENGINEERING_TERMS,
    PYTHON_TERMS,
)
from .sections import _match_section, line_assignments, split_sections

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9_\-]+)", re.IGNORECASE)
DATE_RE = re.compile(
    r"(20\d{2}|19\d{2})|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b|"
    r"\bpresent\b",
    re.IGNORECASE,
)
BULLET_PREFIX = ("-", "•", "*", "–", "—", "▪", "◦", "‣")

# AI-adjacent terms that signal *agentic* (vs generic ML/AI) depth.
AGENTIC_TERMS = AGENTIC_CORE_TERMS
WEB_TERMS = {"React", "Next.js", "Tailwind", "Node.js", "Express", "TypeScript", "JavaScript"}
DATA_TERMS = {"Pandas", "NumPy", "SQL", "PostgreSQL", "MySQL", "MongoDB", "scikit-learn", "SciPy"}


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------
def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def extract_github(text: str) -> tuple[str | None, str | None]:
    """Return (profile_url, username) for the first GitHub profile mention."""
    m = GITHUB_RE.search(text)
    if not m:
        return None, None
    username = m.group(1).rstrip(".,;:)/")
    # Skip obvious non-profiles captured from paths like /features or /orgs.
    if username.lower() in {"features", "orgs", "about", "pricing", "marketplace", "topics", "settings"}:
        return None, None
    return f"https://github.com/{username}", username


ROLE_WORDS = {
    "developer", "engineer", "student", "intern", "internship", "analyst", "designer",
    "manager", "scientist", "architect", "resume", "curriculum", "vitae", "profile",
    "summary", "contact", "email", "phone", "portfolio", "github", "linkedin",
}


def _name_from_contact_line(text: str, email: str | None) -> str | None:
    """Recover a name from a line like 'Prathamesh +91 73490 41840 | Bangalore'.

    Uses the leading name token plus the email local part to reconstruct the
    surname when the resume splits the name across lines.
    """
    for line in text.splitlines()[:6]:
        m = re.match(r"^\s*([A-Za-z][A-Za-z.'\-]*)\s+[+|]?\s*\+?\d", line)
        if not m:
            continue
        first = m.group(1)
        if first.lower() in ROLE_WORDS:
            continue
        if email:
            local = re.sub(r"\d+", "", email.split("@")[0]).lower()
            if local.startswith(first.lower()) and len(local) > len(first) + 1:
                surname = local[len(first):]
                if surname.isalpha():
                    return f"{first.title()} {surname.title()}"
        return first.title()
    return None


def extract_name(text: str, fallback_filename: str = "", email: str | None = None) -> str:
    """Best-effort name: first plausible 'human name' line, else email, else filename."""
    for line in text.splitlines()[:15]:
        candidate = line.strip()
        if not candidate or "@" in candidate or "http" in candidate.lower():
            continue
        if any(ch.isdigit() for ch in candidate):
            continue
        words = candidate.split()
        if not (2 <= len(words) <= 4):
            continue
        # Allow letters, dots, hyphens, apostrophes only.
        if not re.fullmatch(r"[A-Za-z][A-Za-z.\-' ]*", candidate):
            continue
        lowered = candidate.lower()
        if any(bad in lowered for bad in ("resume", "curriculum", "vitae", "profile", "summary")):
            continue
        # Names are capitalised; reject lowercase sentence fragments.
        if candidate == lowered:
            continue
        # Never accept a section heading (e.g. "TECHNICAL SKILLS").
        if _match_section(candidate) is not None:
            continue
        # Reject role/tech lines such as "Python developer" or "Backend Engineer".
        if set(re.findall(r"[a-z]+", lowered)) & ROLE_WORDS:
            continue
        tech_hits: set[str] = set()
        for group in ALL_SKILL_GROUPS.values():
            tech_hits.update(match_terms(candidate, group))
        if tech_hits:
            continue
        return candidate

    # Fall back to the email local part (handles names split across lines by a
    # phone number, e.g. "Prathamesh +91 ... | Bangalore").
    contact_name = _name_from_contact_line(text, email)
    if contact_name:
        return contact_name

    if email:
        local = email.split("@")[0]
        tokens = [t for t in re.split(r"[._\-]+|\d+", local) if len(t) >= 2]
        if len(tokens) >= 2:
            return " ".join(t.title() for t in tokens)

    stem = Path(fallback_filename).stem
    cleaned = re.sub(r"[_\-]+", " ", stem)
    cleaned = re.sub(r"\b(resume|cv|final|updated|new|\d+|candidate)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else "Unknown Candidate"


# ---------------------------------------------------------------------------
# Skills / evidence
# ---------------------------------------------------------------------------
def extract_all_skills(text: str) -> list[str]:
    """Canonical skill names across all vocab groups, preserving group order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for group in ("python", "ai", "backend", "cloud", "engineering"):
        for canonical in match_terms(text, ALL_SKILL_GROUPS[group]):
            if canonical not in seen:
                seen.add(canonical)
                ordered.append(canonical)
    return ordered


def extract_evidence(text: str) -> tuple[list[Evidence], list[Evidence], list[Evidence]]:
    """Return (python_evidence, ai_evidence, other_evidence) with sources."""
    assignments = line_assignments(text)
    python_ev = scan_evidence(assignments, PYTHON_TERMS)
    ai_ev = scan_evidence(assignments, AI_TERMS)
    supporting = {**BACKEND_TERMS, **CLOUD_TERMS, **ENGINEERING_TERMS}
    other_ev = scan_evidence(assignments, supporting)
    return python_ev, ai_ev, other_ev


# ---------------------------------------------------------------------------
# Rough project extraction
# ---------------------------------------------------------------------------
def _looks_like_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(BULLET_PREFIX):
        return False
    # Project titles frequently append dates and a technology list, so they can
    # be long. Lines with an explicit separator or a date are strong signals.
    if "|" in stripped or DATE_RE.search(stripped):
        return len(stripped) <= 160 and not stripped.endswith(".")
    if len(stripped) > 90:
        return False
    return len(stripped.split()) <= 8 and not stripped.endswith(".")


def _classify_domain(text: str) -> ProjectDomain:
    terms = set()
    for group in ALL_SKILL_GROUPS.values():
        terms.update(match_terms(text, group))
    if terms & AGENTIC_TERMS:
        return ProjectDomain.AI_AGENTIC
    if any(t in AI_TERMS for t in terms):
        return ProjectDomain.AI_GENERAL
    if terms & WEB_TERMS:
        return ProjectDomain.WEB_FULLSTACK
    if terms & DATA_TERMS:
        return ProjectDomain.DATA
    return ProjectDomain.OTHER if terms else ProjectDomain.UNKNOWN


def extract_projects_rough(sections: dict[str, str]) -> list[Project]:
    """Split the projects section into blocks and classify each."""
    text = sections.get("projects", "")
    if not text.strip():
        return []

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if current and _looks_like_title(line):
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    projects: list[Project] = []
    for block in blocks:
        block = [ln for ln in block if ln.strip()]
        if not block:
            continue
        body = "\n".join(block)
        title = block[0].strip()
        # A title often carries technologies after a separator.
        title, _, tail = title.partition("|")
        description = "\n".join(block[1:]).strip() or tail.strip()
        tech = extract_all_skills(body)
        projects.append(
            Project(
                name=title.strip()[:120],
                description=description[:1200],
                technologies=sorted(set(tech)),
                domain=_classify_domain(body),
            )
        )
    return projects


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def deterministic_extract(doc: Document) -> ExtractedResume:
    text = doc.text or ""
    sections = split_sections(text)
    python_ev, ai_ev, other_ev = extract_evidence(text)
    github_url, github_username = extract_github(text)
    email = extract_email(text)

    experience = [sections["experience"]] if sections.get("experience") else []
    education = [sections["education"]] if sections.get("education") else []

    return ExtractedResume(
        candidate_name=extract_name(text, doc.filename, email),
        email=email,
        github_url=github_url,
        github_username=github_username,
        skills=extract_all_skills(text),
        projects=extract_projects_rough(sections),
        experience=experience,
        education=education,
        python_evidence=python_ev,
        ai_evidence=ai_ev,
        other_evidence=other_ev,
        extraction_method="deterministic",
    )
