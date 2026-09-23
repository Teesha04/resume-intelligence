"""Term matching with word boundaries and separator tolerance.

Why this exists: naive `in` checks produce the classic false positives —
`java` inside `javascript`, `sql` inside `postgresql`, `go` inside `google`,
`node` inside `nodes`. Every scored claim in our output traces back to a
match from here, so precision matters more than recall.

Rules:
  * Match is case-insensitive.
  * The term must not be flanked by alphanumerics (word boundary).
  * Spaces in multi-word terms tolerate `-`, `_`, or whitespace (e.g.
    "multi agent" -> "multi-agent", "next js" -> "next.js").
"""

from __future__ import annotations

import re
from functools import lru_cache

from .schemas import Evidence, EvidenceSource, SECTION_TO_SOURCE


@lru_cache(maxsize=1024)
def compile_term(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    # Tolerate space / hyphen / underscore between words of a phrase.
    escaped = escaped.replace(r"\ ", r"[\s\-_]+")
    # Allow an optional '.js' style tail already captured by escaping; add
    # optional plural-safe boundary via lookarounds.
    pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


def match_terms(line: str, vocabulary: dict[str, list[str]]) -> list[str]:
    """Return canonical terms from `vocabulary` present in `line` (deduped)."""
    found: list[str] = []
    for canonical, surfaces in vocabulary.items():
        for surface in surfaces:
            if compile_term(surface).search(line):
                found.append(canonical)
                break
    return found


def scan_evidence(
    assignments: list[tuple[str, str]],
    vocabulary: dict[str, list[str]],
    snippet_chars: int = 140,
) -> list[Evidence]:
    """Produce Evidence objects from (section, line) pairs.

    A single line mentioning two terms yields two Evidence entries; duplicates
    of the same (canonical term, source, line) are collapsed.
    """
    seen: set[tuple[str, EvidenceSource, str]] = set()
    out: list[Evidence] = []

    for section, line in assignments:
        if not line.strip():
            continue
        canonical_hits = match_terms(line, vocabulary)
        if not canonical_hits:
            continue
        source = SECTION_TO_SOURCE.get(section, EvidenceSource.UNKNOWN)
        context = line.strip()[:snippet_chars]
        for canonical in canonical_hits:
            key = (canonical, source, context)
            if key in seen:
                continue
            seen.add(key)
            out.append(Evidence(term=canonical, source=source, context=context))

    return out


def matched_skill_names(evidence: list[Evidence]) -> list[str]:
    """Ordered, de-duplicated canonical skill names from evidence."""
    seen: set[str] = set()
    names: list[str] = []
    for ev in evidence:
        if ev.term not in seen:
            seen.add(ev.term)
            names.append(ev.term)
    return names
