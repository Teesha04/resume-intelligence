"""Per-file parsers with strict error isolation.

Contract: a parser NEVER raises. It returns a Document whose `parse_status`
records success/failure and whose `error` explains what happened. This is the
single most important reliability property of the batch (spec section 9).
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from ..schemas import Document, ParseStatus

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md"}


def content_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


_CID_RE = re.compile(r"\(cid:\d+\)")  # pdfplumber glyph artefact for icon fonts

# Common English/resume words used to score extraction quality. A good
# extraction yields more recognisable words and fewer glued/garbled tokens.
_COMMON_WORDS: frozenset[str] = frozenset(
    """
    the and for with using built build developed develop experience project projects
    skills skill summary education work intern internship engineer engineering software
    data machine learning model models system systems application applications web app
    design designed implemented implementation created create managed led team tools
    technologies technology framework frameworks api backend frontend database server
    client user users feature features deployed deployment cloud docker python java
    javascript react node software analysis analytics platform service services
    research presented conference published award awards university institute degree
    bachelor master gpa cgpa location email phone linkedin github portfolio personal
    details current responsibilities achieved improved reduced increased optimized
    testing test tests integration unit performance scalable architecture real time
    """.split()
)


def _text_quality(text: str) -> float:
    """Heuristic quality score for choosing between two PDF extractions.

    Rewards recognisable words (separated, real) and penalises long glued
    tokens ("andbuildingproduction") that indicate lost spacing.
    """
    words = re.findall(r"[A-Za-z]{2,}", text.lower())
    if not words:
        return -1.0
    common = sum(1 for w in words if w in _COMMON_WORDS)
    glued = sum(1 for tok in text.split() if len(tok) > 20)
    no_separator = len(text.split())
    # Prefer more separated tokens, more common words, fewer glued tokens.
    return common * 2.0 + no_separator * 0.05 - glued * 5.0


def _clean(text: str) -> str:
    """Normalise whitespace without destroying line structure."""
    text = _CID_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.replace("\x00", " ").splitlines()]
    # Collapse runs of blank lines to a single blank line.
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _parse_pdf_plumber(path: Path) -> tuple[str, int | None]:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts), page_count


def _parse_pdf_pypdf(path: Path) -> tuple[str, int | None]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts), len(reader.pages)


def _parse_pdf(path: Path) -> tuple[str, int | None]:
    """Run both extractors and keep the higher-quality text.

    Neither extractor dominates: pdfplumber usually handles layout better, but
    pypdf often recovers correct reading order for multi-column resumes and
    preserves spaces when pdfplumber glues words together. We therefore score
    both outputs and pick the best, which is a cheap, general robustness win.
    """
    candidates: list[tuple[str, str, int | None]] = []
    errors: list[str] = []
    for name, fn in (("pdfplumber", _parse_pdf_plumber), ("pypdf", _parse_pdf_pypdf)):
        try:
            text, pages = fn(path)
            if text and text.strip():
                candidates.append((name, text, pages))
            else:
                errors.append(f"{name}: no text")
        except Exception as exc:  # noqa: BLE001 - deliberate broad catch
            errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
            log.debug("PDF parse via %s failed for %s", name, path.name)

    if not candidates:
        raise RuntimeError("; ".join(errors) or "unknown PDF error")
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][2]

    # Prefer pdfplumber (better layout/ordering) unless pypdf is clearly better,
    # to avoid switching on negligible score differences.
    best_name, best_text, best_pages = candidates[0]
    best_q = _text_quality(best_text)
    for name, text, pages in candidates[1:]:
        q = _text_quality(text)
        if q > best_q * 1.15:
            best_name, best_text, best_pages, best_q = name, text, pages, q
    log.debug("Chose %s extraction for %s", best_name, path.name)
    return best_text, best_pages


# ---------------------------------------------------------------------------
# DOCX (bonus)
# ---------------------------------------------------------------------------
def _parse_docx(path: Path) -> tuple[str, int | None]:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]
    # Tables often hold skills/contact info; capture them too.
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts), None


# ---------------------------------------------------------------------------
# TXT / MD
# ---------------------------------------------------------------------------
def _parse_text(path: Path) -> tuple[str, int | None]:
    raw = path.read_bytes()
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding), None
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), None


_PARSERS = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".txt": _parse_text,
    ".md": _parse_text,
}


def parse_file(path: Path) -> Document:
    """Parse one file into a Document. Never raises."""
    ext = path.suffix.lower()
    base = Document(source_file=str(path), filename=path.name)

    try:
        raw = path.read_bytes()
        base.content_hash = content_hash(raw)
    except Exception as exc:  # noqa: BLE001
        base.parse_status = ParseStatus.FAILED
        base.error = f"unreadable file: {exc}"
        return base

    parser = _PARSERS.get(ext)
    if parser is None:
        base.parse_status = ParseStatus.FAILED
        base.error = f"unsupported extension: {ext}"
        return base

    try:
        text, pages = parser(path)
    except Exception as exc:  # noqa: BLE001 - isolation boundary
        base.parse_status = ParseStatus.FAILED
        base.error = f"{type(exc).__name__}: {exc}"
        log.warning("Failed to parse %s: %s", path.name, base.error)
        return base

    base.text = _clean(text)
    base.page_count = pages

    if not base.text:
        # Scanned/image-only PDF or empty file: parsed, but nothing usable.
        base.parse_status = ParseStatus.EMPTY
        base.error = "no extractable text (possibly image-only/scanned)"
        log.warning("Empty text for %s", path.name)

    return base
