"""Directory discovery + dedupe + dispatch.

Responsibilities:
  * find candidate resume files under an input directory
  * skip unsupported extensions (recorded, not fatal)
  * de-duplicate identical content (same file submitted twice, or .pdf + .docx)
  * delegate to parsers.parse_file (which never raises)
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..schemas import Document, ParseStatus
from .parsers import SUPPORTED_EXTENSIONS, parse_file

log = logging.getLogger(__name__)


def discover_files(input_dir: Path, recursive: bool = True) -> tuple[list[Path], list[str]]:
    """Return (supported_files, skipped_names), sorted for deterministic runs."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    pattern = "**/*" if recursive else "*"
    supported: list[Path] = []
    skipped: list[str] = []

    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        # Ignore our own bookkeeping and hidden files (e.g. .DS_Store).
        if path.name.startswith("."):
            continue
        if "__MACOSX" in path.parts:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(path)
        else:
            skipped.append(path.name)

    return supported, skipped


def load_documents(input_dir: Path) -> tuple[list[Document], int]:
    """Parse every supported file.

    Returns (documents, duplicate_count). Documents retain duplicates'
    metadata but are marked DUPLICATE and removed from the processing list
    downstream. Ordering is deterministic.
    """
    files, skipped = discover_files(input_dir)
    if skipped:
        log.info("Skipped %d unsupported file(s): %s", len(skipped), ", ".join(skipped[:10]))

    documents: list[Document] = []
    seen_hashes: dict[str, str] = {}
    duplicates = 0

    for path in files:
        doc = parse_file(path)

        if doc.content_hash:
            first_seen = seen_hashes.get(doc.content_hash)
            if first_seen is not None:
                duplicates += 1
                doc.parse_status = ParseStatus.DUPLICATE
                doc.error = f"duplicate of {first_seen}"
                log.info("Duplicate skipped: %s (same as %s)", path.name, first_seen)
                documents.append(doc)
                continue
            seen_hashes[doc.content_hash] = path.name

        documents.append(doc)

    return documents, duplicates
