#!/usr/bin/env python3
"""CLI entry point.

    python main.py --input ./resumes --output ./output/results.json

Optional flags:
    --limit N        process only the first N resumes (quick runs)
    --csv PATH       also write a flat CSV
    --no-llm         force deterministic-only mode
    --no-cache       disable the on-disk network cache
    --top N          rows shown in the terminal summary
    --verbose        debug logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python main.py` from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cache import DiskCache, NullCache  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.extract import NullLLMClient  # noqa: E402
from src.pipeline import run_screening  # noqa: E402
from src.report import render_console, write_csv, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resume-screener",
        description="AI Resume Screening & Ranking System",
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Directory containing resume files (PDF/DOCX/TXT).")
    parser.add_argument("--output", "-o", type=Path, default=Path("output/results.json"),
                        help="Path to the JSON results file.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N resumes.")
    parser.add_argument("--top", type=int, default=10, help="Rows in the terminal summary.")
    parser.add_argument("--no-llm", action="store_true", help="Force deterministic-only mode.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the network cache.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    settings = get_settings()
    if args.no_llm:
        settings.llm_enabled = False

    cache = NullCache() if args.no_cache else DiskCache(settings.cache_dir)
    llm_client = NullLLMClient() if args.no_llm else None

    if not Path(args.input).exists():
        print(f"error: input directory not found: {args.input}", file=sys.stderr)
        return 2

    report = run_screening(
        Path(args.input),
        settings,
        llm_client=llm_client,
        cache=cache,
        limit=args.limit,
    )

    print(render_console(report, top=args.top))

    out = write_json(report, args.output)
    print(f"\nWrote {out}  ({len(report.candidates)} candidates)")
    if args.csv:
        print(f"Wrote {write_csv(report, args.csv)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
