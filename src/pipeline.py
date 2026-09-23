"""Batch orchestration.

Flow (spec section 1):
  ingest -> extract -> eligibility -> (score) -> GitHub enrich -> rank -> report

Reliability properties:
  * Per-document try/except: one bad resume yields a failed CandidateResult,
    never an aborted batch.
  * Bounded concurrency via a thread pool (settings.max_concurrency) because
    the expensive steps (LLM, GitHub) are I/O bound.
  * GitHub is only fetched for *eligible* candidates (it feeds scoring, which
    only applies to eligible candidates) — this conserves API rate limit and
    is recorded in the batch summary.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Settings, get_settings
from .eligibility import evaluate_eligibility
from .extract import build_llm_client, extract_resume
from .github import GitHubClient
from .ingest import load_documents
from .schemas import (
    BatchSummary,
    CandidateResult,
    Document,
    GitHubEnrichment,
    GitHubStatus,
    ParseStatus,
    ScreeningReport,
)
from .scoring import derive_strengths_concerns, score_candidate

log = logging.getLogger(__name__)


def _project_summary(extracted) -> str:
    if not extracted.projects:
        return "No projects described."
    best = extracted.projects[0]
    desc = " ".join(best.description.split())
    if desc:
        return f"{best.name or 'Untitled project'}: {desc[:240]}"
    return best.name or "Untitled project"


def _process_document(
    doc: Document,
    *,
    llm_client,
    github_client: GitHubClient,
    settings: Settings,
) -> CandidateResult:
    """Process a single document end-to-end. Never raises."""
    result = CandidateResult(source_file=doc.source_file, parse_status=doc.parse_status)

    # Unreadable / empty / duplicate documents are reported, not processed.
    if doc.parse_status in (ParseStatus.FAILED, ParseStatus.EMPTY, ParseStatus.DUPLICATE):
        result.eligible = False
        reason = doc.error or doc.parse_status.value
        result.rejection_reasons = [f"Resume not processed ({doc.parse_status.value}): {reason}"]
        result.candidate_name = doc.filename
        return result

    try:
        extracted = extract_resume(doc, llm_client)
        eligibility = evaluate_eligibility(extracted, has_text=bool(doc.text.strip()))

        result.candidate_name = extracted.candidate_name
        result.eligible = eligibility.eligible
        result.matched_skills = eligibility.matched_skills
        result.rejection_reasons = eligibility.rejection_reasons
        result.project_summary = _project_summary(extracted)
        result.warnings = extracted.warnings

        if not eligibility.eligible:
            # No scoring / GitHub for ineligible candidates.
            result.github = GitHubEnrichment(
                status=GitHubStatus.DISABLED,
                username=extracted.github_username,
                summary="Not enriched (candidate ineligible)",
            )
            return result

        github = github_client.enrich(extracted.github_username)
        result.github = github
        result.github_summary = github.summary

        breakdown = score_candidate(extracted, eligibility, github, settings)
        result.score_breakdown = breakdown
        result.total_score = breakdown.total()

        strengths, concerns = derive_strengths_concerns(extracted, breakdown, github, settings)
        result.strengths = strengths
        result.concerns = concerns

        return result

    except Exception as exc:  # noqa: BLE001 - last-resort isolation boundary
        log.exception("Unexpected failure processing %s", doc.filename)
        result.eligible = False
        result.parse_status = ParseStatus.FAILED
        result.candidate_name = doc.filename
        result.rejection_reasons = [f"Internal processing error: {type(exc).__name__}: {exc}"]
        return result


def _rank(results: list[CandidateResult]) -> None:
    """Assign ranks: eligible candidates by score desc, then the rest."""
    eligible = [r for r in results if r.eligible]
    eligible.sort(key=lambda r: (-(r.total_score or 0.0), r.candidate_name.lower()))
    for i, r in enumerate(eligible, start=1):
        r.rank = i
    for r in results:
        if not r.eligible:
            r.rank = None


def run_screening(
    input_dir: Path,
    settings: Settings | None = None,
    *,
    llm_client=None,
    github_client: GitHubClient | None = None,
    cache=None,
    limit: int | None = None,
) -> ScreeningReport:
    settings = settings or get_settings()
    start = time.perf_counter()

    documents, duplicates = load_documents(Path(input_dir))
    if limit:
        documents = documents[:limit]

    llm_client = llm_client if llm_client is not None else build_llm_client(
        provider=settings.llm_provider,
        api_key=settings.gemini_api_key,
        model=settings.llm_model,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        requests_per_minute=settings.llm_requests_per_minute,
        cache=cache,
    )
    owns_github = github_client is None
    github_client = github_client or GitHubClient(settings, cache=cache)

    to_process = [d for d in documents if d.parse_status == ParseStatus.OK]
    results: list[CandidateResult] = []

    log.info(
        "Ingested %d file(s): %d processable, %d duplicate(s)",
        len(documents), len(to_process), duplicates,
    )

    try:
        with ThreadPoolExecutor(max_workers=max(1, settings.max_concurrency)) as pool:
            futures = {
                pool.submit(
                    _process_document,
                    doc,
                    llm_client=llm_client,
                    github_client=github_client,
                    settings=settings,
                ): doc
                for doc in to_process
            }
            for future in as_completed(futures):
                results.append(future.result())

        # Non-processable documents still appear in the output for accounting.
        for doc in documents:
            if doc.parse_status != ParseStatus.OK:
                results.append(
                    _process_document(
                        doc, llm_client=llm_client, github_client=github_client, settings=settings
                    )
                )
    finally:
        if owns_github:
            github_client.close()

    _rank(results)

    summary = BatchSummary(
        total_resumes=len(documents),
        parsed_ok=sum(1 for d in documents if d.parse_status == ParseStatus.OK),
        parse_failed=sum(
            1 for d in documents if d.parse_status in (ParseStatus.FAILED, ParseStatus.EMPTY)
        ),
        duplicates_skipped=duplicates,
        eligible=sum(1 for r in results if r.eligible),
        rejected=sum(1 for r in results if not r.eligible),
        llm_used=bool(settings.llm_available),
        github_enriched=sum(1 for r in results if r.github.status == GitHubStatus.OK),
        github_failures=sum(
            1 for r in results
            if r.github.status in (GitHubStatus.RATE_LIMITED, GitHubStatus.ERROR)
        ),
        duration_seconds=round(time.perf_counter() - start, 2),
    )

    # Eligible candidates first (ranked), then rejected, then unparsed.
    ordered = (
        sorted([r for r in results if r.eligible], key=lambda r: r.rank or 0)
        + sorted([r for r in results if not r.eligible], key=lambda r: r.candidate_name.lower())
    )

    return ScreeningReport(summary=summary, candidates=ordered)
