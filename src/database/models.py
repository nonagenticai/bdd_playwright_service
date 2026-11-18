"""SQLAlchemy ORM models for bdd_playwright_service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from w2v_common.database.rls import register_rls_ddl

# ⛔ ``as_uuid=False`` is load-bearing, not a style choice. With ``as_uuid=True``
# SQLAlchemy hands back ``uuid.UUID`` objects, which silently breaks every
# ``tenant_id: str`` annotation, every ``==`` against a string tenant, every
# ``str``-keyed cache key and every JSON serialisation in this service — the shape
# that has frozen outboxes elsewhere in the fleet. ``as_uuid=False`` keeps the
# Python value a ``str`` while the COLUMN is a real Postgres ``uuid``, so the
# database gains the type check, the index gains 16-byte keys, and the RLS
# predicate can cast the GUC once (``…::uuid``) instead of casting the column on
# every row.


class DurationBudget(TypedDict):
    """How an ORDERED film duration was split across the generated scenarios.

    Present (non-``None``) only when the caller sent
    ``target_duration_seconds``. ``dwell_seconds`` bounds BOTH the picture and
    the narration length, so voiceover and footage are sized from one number.
    ``order_met`` is ``False`` when a clamp — the per-scenario floor, or
    the ``HERO_DWELL_MS`` ceiling — made the delivered length miss the order;
    that is RECORDED, never raised, because generation itself succeeded.
    """

    ordered_seconds: float
    scenario_count: int
    per_scenario_seconds: float
    narration_word_budget: int
    floor_seconds: float
    floor_applied: bool
    ceiling_seconds: float
    ceiling_applied: bool
    #: The PICTURE budget after the hero-share cap — what the package will actually
    #: dwell for. Below ``per_scenario_seconds`` when one beat would otherwise have
    #: claimed most of the order and been pinned inelastic by the orchestrator. The
    #: narration word budget is sized from THIS number, not from
    #: ``per_scenario_seconds`` — a line written for the uncapped share outlasts the
    #: footage it is spoken over.
    dwell_seconds: float
    hero_cap_share: float
    hero_cap_applied: bool
    projected_total_seconds: float
    order_met: bool


class UngroundedGherkinConcept(TypedDict):
    """One demo ``Then`` step naming UI the pre-interaction page snapshot does not show.

    ``concept`` is the judged claim (a quoted string, or ``<modifier> <ui-noun>`` such as
    ``completed count``). Recorded only when the one-shot grounding re-ask still returned
    it; the step itself is KEPT — this is a warning, never a drop.
    """

    file: str
    scenario: str
    step: str
    concept: str


class UngroundedAssertion(TypedDict):
    """One assertion the recording-stream grounding pass REMOVED, and why.

    ``claim`` is the selector token the pre-interaction DOM does not carry
    (``#todo-count``, ``.todo-count``, ``[data-testid=…]``) or the matcher that no
    snapshot can verify (``toHaveCSS``); ``selector`` is the locator string it came
    from (empty for a matcher claim).
    """

    file: str
    step: str
    selector: str
    claim: str


class StepCoverage(TypedDict):
    """The stored shape of ``bdd_jobs.step_coverage``.

    How much of a generated suite PASSES WITHOUT ACTING: an auto-stubbed step's
    whole body is ``waitForLoadState`` + a dwell, so a majority-stubbed "demo"
    films a page that never changed. A ``TypedDict`` rather than a pydantic model
    because this value is never validated or parsed — it is produced in one place
    (``StepCoverageReport.as_dict``), written verbatim to a JSONB column and
    returned verbatim by ``GET /api/bdd/{job_id}/result``. Declared here, beside
    the column, so the producer depends on the storage shape and not the reverse.

    ``degraded`` is ``stub_ratio >= degraded_threshold``; ``stubbed_steps`` is a
    bounded sample of the non-acting step texts. ``duration_budget`` reuses this
    same visibility channel for the ordered-duration split (``None`` when no
    duration was ordered) rather than opening a parallel one.
    """

    steps_total: int
    steps_stubbed: int
    stub_ratio: float
    degraded: bool
    degraded_threshold: float
    stubbed_steps: list[str]
    duration_budget: DurationBudget | None
    #: Assertions dropped because their selector is not grounded in the scraped DOM
    #: (count, plus a bounded sample). The action steps around them are kept.
    ungrounded_assertions_dropped: int
    ungrounded_assertions: list[UngroundedAssertion]
    #: Demo ``Then`` steps whose Gherkin TEXT names UI the page snapshot does not show and
    #: that survived the grounding re-ask (count, plus a bounded sample). Kept, not dropped.
    ungrounded_gherkin_concepts_flagged: int
    ungrounded_gherkin_concepts: list[UngroundedGherkinConcept]


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


class BDDJob(Base):
    """Tracks a BDD generation job."""

    __tablename__ = "bdd_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # `index=True` makes a unique index (`ix_bdd_jobs_job_id`). The bare
    # `unique=True` flag would additionally emit a UniqueConstraint named
    # `bdd_jobs_job_id_key` — which prod does not have (see migration 0002).
    # The unique index alone is sufficient.
    job_id: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", server_default="pending")
    stream: Mapped[str] = mapped_column(
        String(32)
    )  # qa_suite, accessibility, documentation, demo_video

    # Input
    input_type: Mapped[str | None] = mapped_column(
        String(32), default=None
    )  # web_url, mobile_app, gherkin_upload, inspection_passthrough
    target_url: Mapped[str | None] = mapped_column(Text, default=None)
    app_path: Mapped[str | None] = mapped_column(Text, default=None)
    inspection_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    # Configuration
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=dict, server_default="{}")
    narration_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    # Results
    features_generated: Mapped[int | None] = mapped_column(default=0, server_default="0")
    step_defs_generated: Mapped[int | None] = mapped_column(default=0, server_default="0")
    artifact_paths: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    narration_script: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    # How much of the generated suite PASSES WITHOUT ACTING (auto-stubbed steps):
    # {steps_total, steps_stubbed, stub_ratio, degraded, degraded_threshold,
    # stubbed_steps}. Written for the demo_video stream; `{}` elsewhere. This is
    # what lets a caller distinguish a stubbed scenario from a real one without
    # parsing generator logs.
    step_coverage: Mapped[StepCoverage | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    # Downstream job references
    downstream_jobs: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )

    # Timing — DateTime(timezone=True) matches 0001's schema and prod's TIMESTAMPTZ.
    # Without the explicit type, SQLAlchemy infers DateTime() (no tz) and
    # autogenerate forever reports a type-mismatch with prod.
    generation_time_seconds: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class BDDUsage(Base):
    """Daily usage tracking per tenant."""

    __tablename__ = "bdd_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False))
    date: Mapped[str] = mapped_column(String(10))
    jobs_started: Mapped[int | None] = mapped_column(default=0, server_default="0")
    jobs_completed: Mapped[int | None] = mapped_column(default=0, server_default="0")
    jobs_failed: Mapped[int | None] = mapped_column(default=0, server_default="0")
    features_generated: Mapped[int | None] = mapped_column(default=0, server_default="0")

    __table_args__ = (UniqueConstraint("tenant_id", "date", name="uq_bdd_usage_tenant_date"),)


class NarrationTemplate(Base):
    """Custom narration template per tenant."""

    __tablename__ = "narration_templates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False))
    name: Mapped[str] = mapped_column(String(256))
    tone: Mapped[str] = mapped_column(String(64))
    given_patterns: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    when_patterns: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    then_patterns: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, default=dict, server_default="{}"
    )
    is_default: Mapped[bool | None] = mapped_column(default=False, server_default="false")
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_narration_template_tenant_name"),
    )


# Phase D RLS: register the ENABLE + tenant_isolation policy DDL (after_create) so
# tests' create_all applies the exact policies the 0003_rls migration ships to prod.
# Inert until the runtime connects as the restricted non-owner role with
# TENANT_RLS_MODE=enforce.
register_rls_ddl(Base.metadata)

# Phase D RLS: also create the SECURITY DEFINER cross-tenant cleanup resolvers in the
# create_all (test) path, single-sourced with the migration.
from src.database.rls_resolvers import register_resolvers  # noqa: E402

register_resolvers(Base.metadata)
