"""Configuration management for bdd_playwright_service using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import (
    APISettings as _BaseAPISettings,
)
from w2v_common.config import (
    AuthSettings,
    DatabaseSettings,
    DownstreamSettings,
    LoggingSettings,
    RedisSettings,
    StorageSettings,
)


class APISettings(_BaseAPISettings):
    """BDD Playwright service API settings (port 6313)."""

    port: int = Field(default=6313, description="API server port")


class GenerationSettings(BaseSettings):
    """BDD generation configuration."""

    max_scenarios_per_feature: int = Field(default=10, description="Max scenarios per feature file")
    max_features_per_job: int = Field(default=20, description="Max feature files per job")
    parallel_generation_workers: int = Field(
        default=4, description="Parallel LLM generation workers"
    )

    model_config = SettingsConfigDict(
        env_prefix="GENERATION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class NarrationSettings(BaseSettings):
    """Narration humanization configuration."""

    default_tone: str = Field(default="tutorial", description="Default narration tone")
    llm_polish_enabled: bool = Field(
        default=True, description="Enable LLM polish pass for narration"
    )
    # FALLBACK ONLY. The orchestrator now sends a MEASURED word budget alongside the
    # ordered duration (``narration_words_target`` / ``narration_words_per_scenario``,
    # see ``services/duration_budget.narration_word_budgets``), and that wins wherever
    # it is present — it is derived from the DELIVERED speech rate, which this service
    # cannot observe. This constant only sizes the locally derived budget for callers
    # that send no word budget at all. It is a nominal rate, not a measurement: the
    # estate's delivered rate has been measured well below it (~94-152 wpm depending
    # on the voice), so a budget derived from it runs LONG. Do not treat it as truth
    # and do not re-derive it here — fix the number at the orchestrator, which knows.
    words_per_minute: int = Field(
        default=150, description="Estimated words per minute for duration calculation"
    )

    model_config = SettingsConfigDict(
        env_prefix="NARRATION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(default=30, description="Max requests per minute per tenant")
    concurrent_jobs_per_tenant: int = Field(default=5, description="Max concurrent jobs per tenant")

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class FetchSettings(BaseSettings):
    """In-service page fetcher (the self-contained replacement for webscraper)."""

    timeout_seconds: float = Field(
        default=30.0, description="Per-page navigation deadline in seconds"
    )
    post_load_delay_seconds: float = Field(
        default=2.0,
        description="Dwell after load so client-rendered SPAs hydrate before the DOM snapshot",
    )
    max_bytes: int = Field(default=2_000_000, description="Largest HTML page accepted, in bytes")
    max_pages: int = Field(default=20, description="Largest batch accepted, in pages")
    allow_private: bool = Field(
        default=False,
        description=(
            "Allow RFC1918/ULA targets. Off by default: an in-cluster browser pointed at "
            "arbitrary caller URLs must not reach the pod network unless the operator opts in"
        ),
    )
    user_agent: str = Field(
        default="bdd-playwright-service/0.1 (page inspection)",
        description="Default User-Agent for fetched pages",
    )

    model_config = SettingsConfigDict(
        env_prefix="BDD_FETCH_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CleanupSettings(BaseSettings):
    """Job cleanup configuration."""

    retention_days: int = Field(default=30, description="Days to retain completed/failed jobs")
    stuck_job_timeout_minutes: int = Field(
        default=60, description="Minutes after which a processing job is considered stuck"
    )
    pending_job_timeout_minutes: int = Field(
        default=360,
        description=(
            "Minutes after which a job still in 'pending' is considered never-dispatched. "
            "Deliberately far longer than stuck_job_timeout_minutes: 'pending' is a "
            "legitimate queued state, so a short cutoff would fail out live queued work."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="CLEANUP_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    app_name: str = Field(default="BDD Playwright Service", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development", description="Environment (development, staging, production)"
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    downstream: DownstreamSettings = Field(default_factory=DownstreamSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    narration: NarrationSettings = Field(default_factory=NarrationSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cleanup: CleanupSettings = Field(default_factory=CleanupSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_database_url(self) -> str:
        """Return the configured database connection URL."""
        return self.database.url

    def get_redis_url(self) -> str:
        """Return the configured Redis connection URL."""
        return self.redis.url

    def is_production(self) -> bool:
        """Return True if the environment is production."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return True if the environment is development."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful for testing)."""
    get_settings.cache_clear()
