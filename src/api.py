"""Optional thin FastAPI wrapper around the same pipeline.

Run:
    uvicorn src.api:app --reload
    # or: python main.py --serve

Endpoints (spec section 8):
    POST /screen   { "input_dir": "./resumes", "limit": 50 }
    GET  /results  -> last ScreeningReport
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import run_screening
from .schemas import ScreeningReport

app = FastAPI(
    title="AI Resume Screening & Ranking",
    version="0.1.0",
    description="Ingest resumes, hard-filter, score, enrich with GitHub, and rank.",
)

_LAST_REPORT: ScreeningReport | None = None


class ScreenRequest(BaseModel):
    input_dir: str = Field(..., description="Directory containing resume files.")
    limit: int | None = Field(None, ge=1, description="Optional cap on resumes processed.")


class ScreenResponse(BaseModel):
    summary: dict
    results_url: str = "/results"


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_available": settings.llm_available,
        "github_token": bool(settings.github_token),
    }


@app.post("/screen", response_model=ScreenResponse)
def screen(req: ScreenRequest) -> ScreenResponse:
    global _LAST_REPORT
    directory = Path(req.input_dir)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"input_dir not found: {req.input_dir}")

    _LAST_REPORT = run_screening(directory, get_settings(), limit=req.limit)
    return ScreenResponse(summary=_LAST_REPORT.summary.model_dump(mode="json"))


@app.get("/results", response_model=ScreeningReport)
def results() -> ScreeningReport:
    if _LAST_REPORT is None:
        raise HTTPException(status_code=404, detail="No screening run yet. POST /screen first.")
    return _LAST_REPORT
