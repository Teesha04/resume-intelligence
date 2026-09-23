"""Configuration: weights, thresholds, model + secrets.

Everything tunable lives here so business logic never hard-codes a magic
number. Secrets are read only from environment variables (see .env.example).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Weights(BaseModel):
    """Scoring weights. Must sum to 100 (excluding penalties)."""

    ai_project_depth: float = 40.0
    python_backend: float = 30.0
    cloud_fullstack: float = 15.0
    github: float = 10.0
    engineering_depth: float = 5.0

    # GitHub is split into two explainable halves.
    github_activity: float = 5.0
    github_repos: float = 5.0

    def total(self) -> float:
        return (
            self.ai_project_depth
            + self.python_backend
            + self.cloud_fullstack
            + self.github
            + self.engineering_depth
        )


class Penalties(BaseModel):
    """Project-quality penalties (applied as deductions, never below zero)."""

    thin_wrapper_min: float = 5.0
    thin_wrapper_max: float = 15.0
    tutorial_listed: float = 3.0
    max_total: float = 15.0


class Thresholds(BaseModel):
    """Cut-offs for evidence-based credit.

    These encode the spec's intent: a keyword buried in a skills list earns
    less than the same keyword attached to a real project or internship.
    """

    # Multiplier applied by evidence source (skills list vs project vs work).
    source_weight_project: float = 1.0
    source_weight_experience: float = 0.9
    source_weight_skills: float = 0.4

    # AI project rubric: a project scoring below this is "thin".
    thin_wrapper_quality: float = 3.0
    # Credit given for a framework name with no context.
    framework_name_only_credit: float = 0.25

    # GitHub
    github_active_days: int = 90          # "recent" activity window
    github_repo_fresh_days: int = 180     # "maintained" repo window


class Settings(BaseSettings):
    """Environment-backed settings (loaded from .env when present)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    llm_model: str = "gemini-3.1-flash-lite"
    llm_enabled: bool = True
    llm_max_retries: int = 2
    # Free tiers enforce a low requests-per-minute quota; pace to stay under it.
    llm_requests_per_minute: int = 12
    # On a per-minute quota error: "wait" (pause until it resets, then resume)
    # or "fail" (give up on that call immediately).
    llm_on_rate_limit: str = "wait"
    llm_max_wait_seconds: float = 90.0     # cap for a single wait
    llm_wait_budget_seconds: float = 600.0  # total wait budget per run

    # --- GitHub ---
    github_token: str | None = None
    # Neutral placeholder (out of 10) for profiles tagged rate-limited/error,
    # so an unknown is not scored as genuine inactivity.
    github_rate_limited_placeholder: float = 5.0

    # --- Runtime ---
    max_concurrency: int = 5
    request_timeout_seconds: float = 20.0
    cache_dir: Path = Path(".cache")
    log_level: str = "INFO"

    # --- Scoring (nested, overridable) ---
    weights: Weights = Field(default_factory=Weights)
    penalties: Penalties = Field(default_factory=Penalties)
    thresholds: Thresholds = Field(default_factory=Thresholds)

    @property
    def llm_available(self) -> bool:
        """True only when an LLM is both enabled and configured with a key."""
        if not self.llm_enabled:
            return False
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so env/.env is parsed once per process."""
    return Settings()
