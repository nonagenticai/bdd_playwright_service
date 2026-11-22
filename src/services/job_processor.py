"""Pipeline orchestrator that processes BDD generation jobs end-to-end."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from typing import Any, cast

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.discovery import service_url

from src.api.constants import STREAM_ACCESSIBILITY, STREAM_DEMO_VIDEO, STREAM_QA_SUITE
from src.api.schemas.gherkin import GherkinScenario, GherkinStep
from src.config import get_settings
from src.database.models import StepCoverage
from src.services.duration_budget import (
    DurationBudgetPlan,
    narration_word_budgets,
    plan_duration_budget,
    raise_to_clip_floor,
    trim_to_word_budget,
)
from src.services.generation.config_generator import ConfigGenerator
from src.services.generation.feature_generator import (
    UNGROUNDED_CONCEPTS_KEY,
    FeatureGenerator,
)
from src.services.generation.llm_client import LLMClient
from src.services.generation.qa_suite import (
    build_grounded_features,
    validate_features,
)
from src.services.generation.step_def_generator import (
    GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
    StepCoverageReport,
    StepDefGenerator,
    ensure_dwell_budget,
)
from src.services.input.gherkin_parser import GherkinParser
from src.services.input.html_structure import extract_structure
from src.services.input.inspection_adapter import InspectionAdapter
from src.services.input.page_fetcher import FetchedImage, FetchedPage, PageFetcher
from src.services.job_manager import JobManager
from src.services.narration.humanizer import (
    MIN_SHORTFALL_WORDS,
    SHORTFALL_TRIGGER_RATIO,
    NarrationHumanizer,
    PolishReport,
)
from src.services.narration.template_engine import NarrationTemplateEngine
from src.services.narration.types import (
    NarrationSection,
    WordBudgetReport,
    WordBudgetSectionRow,
)
from src.services.package_builder import PackageBuilder
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

#: A JSON-shaped payload at a storage/DB boundary (features, validation reports).
JsonDict = dict[str, Any]


def _grounded_qa_enabled(config: JsonDict) -> bool:
    """True when this QA job uses the deterministic, snapshot-grounded path.

    ``generation_mode`` defaults to ``grounded`` for the QA stream: the legacy path
    (LLM-authored features AND LLM-authored step definitions, packaged with the
    demo-video config) is what shipped unexecutable suites. A job that carries an
    ``auth_config`` stays on the legacy path — its package needs a storage-state
    setup project the grounded vocabulary does not model yet.
    """
    if config.get("auth_config"):
        return False
    return config.get("generation_mode", "grounded") != "legacy"


_CLIP_TAG_RE = re.compile(r"^@?(clip-\d+)$")


def _clip_id_for(tags: list[str], fallback_index: int) -> str:
    """Return the scenario's ``clip-<n>`` join id from its tags.

    Prefers the deterministic ``@clip-<n>`` tag stamped at generation time;
    falls back to a document-order index when (defensively) the tag is absent —
    e.g. an externally-supplied feature file via ``gherkin_upload``.
    """
    for tag in tags:
        m = _CLIP_TAG_RE.match(tag.strip())
        if m:
            return m.group(1)
    return f"clip-{fallback_index}"


def _auth_config_from_pack(auth: dict[str, Any]) -> dict[str, Any]:
    """Map a knowledge pack's ``access.auth`` onto the FTR ``auth_config`` shape.

    Carries the login url + username + optional selector overrides, plus the
    ``sso_button``/``success_url`` fields that FTR forwards into the package
    ``.env`` so the scaffold's SSO/IdP login branch fires. The password is NOT
    copied — it is injected from the ``AUTH_PASSWORD`` env at record time.
    """
    cfg: dict[str, Any] = {
        "login_url": auth.get("login_url", ""),
        "username": auth.get("username", ""),
    }
    for key in (
        "username_selector",
        "password_selector",
        "submit_selector",
        "sso_button",
        "success_url",
    ):
        if auth.get(key):
            cfg[key] = auth[key]
    return cfg


_MAX_INSPECT_SCREENS = 20


def _interior_urls(base_url: str, target_pages: list[str]) -> tuple[list[str], int]:
    """Absolute interior URLs to inspect: the landing page first, then each
    ``target_pages`` entry (a path or an absolute URL) resolved against the base,
    de-duplicated in order.

    Returns ``(urls, dropped)`` where ``dropped`` counts entries trimmed past
    ``_MAX_INSPECT_SCREENS`` (the caller logs it — never a silent cap).
    """
    from urllib.parse import urljoin

    urls: list[str] = []
    seen: set[str] = set()
    for candidate in [base_url, *target_pages]:
        if not candidate:
            continue
        absolute = (
            candidate
            if candidate.startswith(("http://", "https://"))
            else urljoin(base_url, candidate)
        )
        if absolute not in seen:
            seen.add(absolute)
            urls.append(absolute)
    dropped = max(0, len(urls) - _MAX_INSPECT_SCREENS)
    return urls[:_MAX_INSPECT_SCREENS], dropped


def _snapshot_entry_page(entry: object) -> dict[str, Any] | None:
    """One inspection page from a snapshot entry, or ``None`` when unusable."""
    if not isinstance(entry, dict):
        return None
    url = str(entry.get("url") or "").strip()
    html = entry.get("html")
    if not url or not isinstance(html, str) or not html:
        return None
    links = entry.get("links") or []
    images = entry.get("images") or []
    page = FetchedPage(
        url=url,
        title=str(entry.get("title") or ""),
        html=html,
        text=str(entry.get("text") or ""),
        links=[str(link) for link in links if isinstance(link, str)],
        images=[FetchedImage(src=str(src)) for src in images if isinstance(src, str)],
    )
    return _page_from_scrape(page)


def _page_from_scrape(response: Any) -> dict[str, Any]:
    """One inspection ``pages[]`` entry from a webscraper ScrapeResponse.

    ``headings`` / ``has_forms`` / ``interactive_elements`` / ``page_type`` are
    derived locally from the rendered ``html`` (via :func:`extract_structure`):
    webscraper returns the DOM but not the structure, and the demo generators consume
    that structure to emit concrete Playwright selectors. Empty html → empty
    structure (no regression versus the prior text-only inspection).
    """
    structure = extract_structure(getattr(response, "html", "") or "")
    if structure["elements_truncated"]:
        logger.debug(
            "page %s: interactive-element cap hit, dropped %d",
            response.url,
            structure["elements_truncated"],
        )
    return {
        "url": response.url,
        "title": response.title,
        "html": response.html,
        # Canonical key consumed by inspection_adapter._parse_pages (was "text",
        # which the adapter never read → empty text_content).
        "text_content": response.text,
        "links": response.links,
        # webscraper returns ImageRef objects; persist only the src URL strings
        # (the KAG-era shape) so the page dict stays JSON-serialisable.
        "images": [img.src for img in response.images],
        "headings": structure["headings"],
        "has_forms": structure["has_forms"],
        "interactive_elements": structure["interactive_elements"],
        "page_type": structure["page_type"],
        # Carried, not just logged: the grounding pass in `step_def_generator`
        # deletes assertions naming elements/headings ABSENT from these lists, and
        # absence in a CAPPED list is not evidence of absence. Without these keys the
        # 41st link on a landing page reads as "not on the page".
        "elements_truncated": structure["elements_truncated"],
        "headings_truncated": structure["headings_truncated"],
        # The id/class/testid inventory the assertion-grounding pass judges CSS
        # selectors against. `None` = the DOM was not fully scanned.
        "dom_tokens": structure["dom_tokens"],
    }


def _word_budget_for(
    index: int,
    word_budgets: list[int] | None,
    budget: DurationBudgetPlan | None,
) -> int | None:
    """The word budget for the beat at ``index``, or ``None`` for "no budget".

    PRECEDENCE, and why: the ORCHESTRATOR's per-beat budget wins, because it is
    derived from the delivered speech rate; the locally derived
    ``narration_word_budget`` is the fallback, because it can only be derived from a
    configured ``wpm``; and ``None`` — neither sent — is today's unbudgeted
    behaviour, which every not-yet-deployed caller must keep getting.
    """
    if word_budgets is not None and index < len(word_budgets):
        return word_budgets[index]
    return budget.narration_word_budget if budget is not None else None


def _narrated_steps(scenario: GherkinScenario, *, budgeted: bool) -> list[GherkinStep]:
    """The steps a scenario's NARRATION is written from.

    Under a word budget the feature's ``Background`` steps are left out. The parser
    prepends them to EVERY scenario, so the template opened every beat with the same
    sentence ("First, I open the React TodoMVC demo at ...", 9 words), and the
    sentence-aware trim to a 15-word budget then kept ONLY that sentence: live job
    01a156a6-8dd9 (2026-10-20) shipped five identical 9-word lines against 15-word
    budgets, all five scenarios' own actions cut. The steps still RUN (the package and
    ``original_gherkin`` are untouched) — they are just not what each beat narrates.

    Unbudgeted narration, and a scenario that is ONLY background, keep every step.
    """
    own = scenario.steps[scenario.background_step_count :]
    if budgeted and own:
        return own
    return scenario.steps


def _reapply_word_budget(
    parts: list[NarrationSection],
    word_budgets: list[int] | None,
    budget: DurationBudgetPlan | None,
    wpm: float,
) -> None:
    """Re-apply the budget CEILING and recompute duration estimates in place.

    The LLM pass rewrites the line, so the budget is re-applied to its output —
    otherwise a longer rewritten sentence would silently spend past the budget
    the dwell was already narrowed to.
    """
    for index, part in enumerate(parts):
        words_budget = _word_budget_for(index, word_budgets, budget)
        if words_budget is not None:
            part["text"] = trim_to_word_budget(str(part.get("text", "")), words_budget)
        part["word_count"] = len(str(part.get("text", "")).split())
        part["estimated_duration_seconds"] = (part["word_count"] / wpm) * 60 if wpm else 0.0


def _log_narration_sources(job_id: str, narration_script: dict[str, Any] | None) -> None:
    """Log, once per job, how many narration lines came from the fallback template.

    A ``template_fallback`` line is the canned per-tone sentence ("Next, ...")
    emitted when no narration pattern matched a step. Observability only.

    ``narration_source`` describes the PRE-polish line. When the LLM polish pass
    was applied, every line was rewritten by the model, so a fallback template is
    not what ships. Logging that as a WARNING ("template_fallback_lines=9 of 9")
    read live on 2026-10-18 as "the LLM narration path produced nothing" while
    the same job's polish pass had succeeded. The summary therefore states
    ``llm_polished`` and WARNs only when canned fallback text is DELIVERED.
    """
    if narration_script is None:
        return
    parts = narration_script.get("parts") or []
    fallback_lines = sum(1 for p in parts if p.get("narration_source") == "template_fallback")
    fallback_steps = sum(int(p.get("template_fallback_steps") or 0) for p in parts)
    report = narration_script.get("word_budget_report") or {}
    polished = bool((report.get("polish") or {}).get("applied"))
    logger.log(
        logging.WARNING if fallback_lines and not polished else logging.INFO,
        "narration_source_summary job_id=%s total_lines=%d template_fallback_lines=%d "
        "template_fallback_steps=%d llm_polished=%s",
        job_id,
        len(parts),
        fallback_lines,
        fallback_steps,
        polished,
    )


def _word_budget_report(
    parts: list[NarrationSection],
    word_budgets: list[int] | None,
    budget: DurationBudgetPlan | None,
    polish_report: PolishReport | None,
    expansion_report: PolishReport | None,
) -> WordBudgetReport | None:
    """The per-section budget outcome as STRUCTURED data, or ``None``.

    The budget was enforced downstream as a ceiling only, and a section that came
    back SHORT was reported nowhere at all — the orchestrator saw a prose advisory
    ("clip-1 has 14 words, budgeted 24") and nothing machine-readable, so its only
    remedy was to stretch the speech to fill the slot. This is that number, on the
    wire, per section and in total.

    ``None`` when no budget applied to any section and no LLM pass ran, so a
    payload that has nothing to report is byte for byte what it is today.
    """
    rows: list[WordBudgetSectionRow] = []
    budgeted_total = 0
    delivered_total = 0
    for index, part in enumerate(parts):
        words_budget = _word_budget_for(index, word_budgets, budget)
        if words_budget is None or words_budget <= 0:
            continue
        words = int(part.get("word_count", 0) or 0)
        budgeted_total += words_budget
        delivered_total += words
        if words_budget - words >= MIN_SHORTFALL_WORDS:
            rows.append(
                {
                    "part_id": part.get("part_id"),
                    "scenario_name": part.get("scenario_name"),
                    "word_count": words,
                    "word_budget": words_budget,
                    "shortfall_words": words_budget - words,
                }
            )

    if not budgeted_total and polish_report is None and expansion_report is None:
        return None

    return {
        "budgeted_words": budgeted_total,
        "delivered_words": delivered_total,
        "shortfall_words": max(0, budgeted_total - delivered_total),
        "sections_short": rows,
        "shortfall_trigger_ratio": SHORTFALL_TRIGGER_RATIO,
        "polish": polish_report.as_dict() if polish_report is not None else None,
        "expansion": expansion_report.as_dict() if expansion_report is not None else None,
    }


def _reconstruct_gherkin(name: str, steps: list[Any]) -> str:
    """Rebuild a readable Gherkin snippet for one scenario (name + steps)."""
    lines = [f"Scenario: {name}"]
    lines.extend(f"  {step.keyword} {step.text}" for step in steps)
    return "\n".join(lines)


class JobProcessor:
    """Chains all service components to process a BDD generation job."""

    def __init__(self) -> None:
        settings = get_settings()
        # Each subsystem gets its own WM worker name so cost / usage metrics
        # in worker_management are bucketed per subsystem. A single shared
        # WMClient transport is used under the hood (see llm_client module).
        self._feature_gen = FeatureGenerator(LLMClient("bdd-feature-generator"))
        self._step_def_gen = StepDefGenerator(LLMClient("bdd-step-def-generator"))
        self._humanizer = NarrationHumanizer(LLMClient("bdd-narration-humanizer"))
        self._config_gen = ConfigGenerator()
        self._adapter = InspectionAdapter()
        self._parser = GherkinParser()
        self._storage = StorageService()
        self._package_builder = PackageBuilder(self._storage)
        self._narration_engine: NarrationTemplateEngine | None = None
        self._settings = settings

    async def aclose(self) -> None:
        """Release per-processor transports (the MinIO client's aiohttp session)."""
        await self._storage.aclose()

    async def process_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        license_token: str | None = None,
    ) -> None:
        """Run :meth:`_run_job`, then ALWAYS close this processor's storage client.

        Each route builds one ``JobProcessor`` (and so one ``StorageService``) per
        job. Before this ``finally`` the MinIO client's ``aiohttp.ClientSession``
        was never closed, and every job ended in ``asyncio ERROR Unclosed client
        session`` / ``Unclosed connector`` at GC time. Closing here — not in the
        background dispatcher — covers every caller of ``process_job``, on success
        and on failure.
        """
        try:
            await self._run_job(db, job_id, tenant_id, license_token=license_token)
        finally:
            try:
                await self.aclose()
            except OSError, RuntimeError:
                # Never mask the job's own outcome with a teardown error.
                logger.warning("Failed to close storage client for job %s", job_id, exc_info=True)

    async def _run_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: str,
        license_token: str | None = None,
    ) -> None:
        """Execute the full BDD generation pipeline for one job.

        Steps: mark processing → resolve input → generate features →
        generate step defs → generate config → store artifacts →
        narrate (optional) → build package → downstream (optional) →
        mark completed.

        Args:
            db: Async DB session.
            job_id: Job identifier.
            tenant_id: Tenant identifier (resolved from middleware).
            license_token: The request's license token, threaded through to
                downstream peer-service calls so they authenticate via
                ``X-Tenant-Id: Bearer <token>`` instead of relying on the
                trusted-network fallback. ``None`` is acceptable in tests but
                in production callers SHOULD provide it.
        """
        t0 = time.monotonic()

        # 1. Fetch job and mark processing
        job = await JobManager.get_job(db, job_id)
        if not job:
            raise RuntimeError(f"Job {job_id} not found")

        await JobManager.mark_processing(db, job_id)
        await db.commit()

        stream = str(job.stream)
        input_type = str(job.input_type or "web_url")
        config: dict[str, Any] = dict(job.config) if job.config else {}
        narration_config: dict[str, Any] = (
            dict(job.narration_config) if job.narration_config else {}
        )

        # 2. Resolve input → inspection data
        inspection = await self._resolve_input(
            input_type=input_type,
            target_url=str(job.target_url or ""),
            app_path=str(job.app_path or ""),
            config=config,
            tenant_id=tenant_id,
            license_token=license_token,
        )

        # Store inspection data on job for debugging
        job.inspection_data = inspection
        await db.commit()

        # 3. Generate features (LLM)
        features = await self._generate_features(stream, inspection, config)

        # 3a. Grounded QA: drop every scenario the snapshot cannot prove. A suite
        # with nothing left raises here — shipping an empty package as "completed"
        # is the silent-zero failure this service exists to prevent.
        grounded_qa = stream == STREAM_QA_SUITE and _grounded_qa_enabled(config)
        grounding_report: JsonDict | None = None
        if grounded_qa:
            features, grounding_report = validate_features(features, inspection)

        # 3b. Split the ORDERED film duration across the scenarios that were
        # ACTUALLY generated (never the requested cap — the generator routinely
        # emits fewer, and budgeting against an unfilled cap under-runs every
        # clip). `None` when nothing was ordered, which is the whole
        # backward-compatibility contract: every step below is then a no-op.
        budget = self._plan_budget(stream, features, config)

        # 4. Generate step definitions. The knowledge-pack showcase path and the
        # grounded QA path use fixed deterministic vocabularies (no LLM invents
        # steps); every other path asks the LLM against the inspected selectors.
        # Stub visibility: an auto-stubbed step PASSES WITHOUT ACTING, so a
        # heavily-stubbed "demo" films a page that never changed. The generator is
        # the only place that knows the share; carry it out per job (never on the
        # generator instance — one instance serves concurrent jobs).
        coverage = StepCoverageReport(
            ungrounded_gherkin_concepts=[
                concept
                for feature in features
                for concept in feature.get(UNGROUNDED_CONCEPTS_KEY) or []
            ]
        )
        step_defs = await self._generate_step_definitions(
            features=features,
            inspection=inspection,
            config=config,
            stream=stream,
            coverage=coverage,
            grounded_qa=grounded_qa,
        )

        # 4b. Make the DWELL proportionate to the order. Both step-def paths land
        # here — the LLM path above and the deterministic showcase path, whose
        # `Then I dwell for <ms> ms` steps spend from the same per-scenario
        # ledger — so a package cannot run to the fixed 22s budget when its share
        # of the order is smaller. Recorded on the coverage report either way, so
        # a clamp is visible instead of silent.
        if budget is not None:
            coverage.duration_budget = budget.as_dict()
            ensure_dwell_budget(step_defs, budget.dwell_budget_ms)
            if not budget.order_met:
                logger.warning(
                    "Job %s: ordered %.1fs over %d scenario(s) = %.2fs each, clamped to "
                    "%.2fs (floor_applied=%s ceiling_applied=%s) — the order CANNOT be "
                    "met; projected %.1fs. Recorded on step_coverage.duration_budget.",
                    job_id,
                    budget.ordered_seconds,
                    budget.scenario_count,
                    budget.ordered_seconds / budget.scenario_count,
                    budget.per_scenario_seconds,
                    budget.floor_applied,
                    budget.ceiling_applied,
                    budget.projected_total_seconds,
                )

        # 5. Generate config files (deterministic)
        auth_config = config.get("auth_config")
        playwright_config, package_json, tsconfig, auth_setup = self._generate_config_files(
            inspection=inspection,
            stream=stream,
            auth_config=auth_config,
            grounded_qa=grounded_qa,
            locale=config.get("locale"),
        )

        # 6. Store all artifacts to disk. Recording packages (demo_video) ship the
        # shared app-agnostic step-helper lib so their steps import it.
        lib_files = self._config_gen.read_scaffold_lib() if stream == STREAM_DEMO_VIDEO else None
        artifact_paths = await self._store_artifacts(
            job_id,
            features,
            step_defs,
            playwright_config,
            package_json,
            tsconfig,
            auth_setup=auth_setup,
            lib_files=lib_files,
            coverage=coverage if (stream == STREAM_DEMO_VIDEO or grounded_qa) else None,
            validation=grounding_report,
        )

        # 7. Generate narration (if enabled). For the knowledge-pack path the
        # spoken line is AUTHORED per screen — pass it through verbatim (keyed by
        # the same document-order clip id the emitter stamped) so narration is not
        # re-invented from the terse tour steps; the LLM polish pass still smooths it.
        narration_script: dict[str, Any] | None = None
        if narration_config.get("enabled", False):
            screens = config.get("knowledge_screens")
            if screens:
                narration_config = {
                    **narration_config,
                    "authored_narration": {
                        f"clip-{i}": s.get("narration", "")
                        for i, s in enumerate(screens, start=1)
                        if s.get("narration")
                    },
                    # Per-screen expected on-screen text (O1b) — carried verbatim,
                    # keyed by the same document-order clip id. Not narration and not
                    # a Playwright step: it's QA metadata the demo-video record stage
                    # asserts via OCR to catch a clip that recorded the WRONG screen
                    # (e.g. a "Logging out" interstitial) at the right length.
                    "authored_expected_text": {
                        f"clip-{i}": s["expected_text"]
                        for i, s in enumerate(screens, start=1)
                        if s.get("expected_text")
                    },
                }
            narration_script = await self._generate_narration(
                features,
                narration_config,
                recording=(stream == STREAM_DEMO_VIDEO),
                budget=budget,
                word_budgets=raise_to_clip_floor(
                    self._narration_word_budgets(
                        [str(f.get("content", "")) for f in features],
                        config.get("narration_words_target"),
                        config.get("narration_words_per_scenario"),
                    ),
                    budget,
                    config.get("narration_words_target"),
                    config.get("target_duration_seconds"),
                ),
                job_id=job_id,
            )

        # 8. Build package
        package_path = await self._package_builder.build_package(job_id, stream, artifact_paths)
        artifact_paths["package_zip"] = package_path

        # 9. Mark completed. (There is no bdd-side downstream call: the demo_video
        # deliverable is the package + narration SCRIPT; the orchestrator performs the
        # authoritative narration for the compiled video from that script. The former
        # best-effort TTS preview-audio call produced a `tts_job_id` that NOTHING
        # consumed — a permanent swallowed-404 dead-signal — so it was removed.)
        downstream_jobs: dict[str, Any] = {}
        elapsed = time.monotonic() - t0
        coverage_payload: JsonDict = dict(coverage.as_dict())
        if grounding_report is not None:
            # One place a consumer looks to learn whether the delivered suite was
            # cut down by grounding, and why.
            coverage_payload["grounding_validation"] = grounding_report
        await JobManager.mark_completed(
            db,
            job_id,
            features_generated=len(features),
            step_defs_generated=len(step_defs),
            artifact_paths=artifact_paths,
            narration_script=narration_script,
            downstream_jobs=downstream_jobs or None,
            generation_time_seconds=elapsed,
            step_coverage=cast(StepCoverage, coverage_payload),
        )
        if coverage.degraded:
            logger.warning(
                "Job %s package marked DEGRADED: %d of %d step(s) (%.0f%%) are "
                "non-acting auto-stubs — the recording will largely film a static page",
                job_id,
                coverage.steps_stubbed,
                coverage.steps_total,
                coverage.stub_ratio * 100,
            )
        await db.commit()
        _log_narration_sources(job_id, narration_script)

        # Record usage
        try:
            await JobManager.record_usage(
                db,
                tenant_id=tenant_id,
                jobs_completed=1,
                features_generated=len(features),
            )
            await db.commit()
        except OSError, RuntimeError, SQLAlchemyError:
            # Best-effort usage recording: a transient commit failure raises
            # sqlalchemy.exc.OperationalError/InterfaceError (MRO ... ->
            # SQLAlchemyError -> Exception), NOT OSError/RuntimeError. Catch the
            # SQLAlchemyError base so a dropped BDDUsage row is logged (billing
            # undercount) instead of escaping and spuriously failing an
            # already-completed job.
            logger.warning("Failed to record usage for job %s", job_id)

        logger.info(
            "BDD job %s completed: %d features, %d step defs in %.1fs",
            job_id,
            len(features),
            len(step_defs),
            elapsed,
        )

    def _resolve_knowledge_input(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve a hand-authored knowledge pack into inspection data.

        Drives a DETERMINISTIC showcase package: stash ``screens[]`` for the
        deterministic feature emitter, lift the pack's auth into ``auth_config``
        (so auth.setup + FTR SSO forwarding fire), and synthesise a light
        inspection (one page per screen) for any downstream summary. No
        crawl/LLM inspection is needed.
        """
        pack = config.get("knowledge") or {}
        access = pack.get("access") or {}
        screens = pack.get("screens") or []
        base_url = access.get("base_url", "")
        app = pack.get("app") or {}
        config["knowledge_screens"] = screens
        config["knowledge_app_name"] = app.get("name") or ""
        auth = access.get("auth")
        if auth and not config.get("auth_config"):
            config["auth_config"] = _auth_config_from_pack(auth)
        pages = [
            {
                "url": s.get("route") or base_url,
                "title": s.get("label") or s.get("id") or "",
                "page_type": s.get("section") or "screen",
                "has_forms": bool((s.get("capture") or {}).get("click")),
                "interactive_elements": [],
                "headings": [],
            }
            for s in screens
        ]
        return {
            "pages": pages,
            "navigation_graph": [],
            "platform": "web",
            "base_url": base_url,
        }

    async def _resolve_web_url_input(
        self,
        *,
        target_url: str,
        config: dict[str, Any],
        tenant_id: str | None,
    ) -> dict[str, Any]:
        """Inspect a live site with the in-service fetcher.

        The fetcher renders each page with headless Chromium (or fetches static
        HTML when ``javascript_enabled`` is false), so interior routes listed in
        ``target_pages`` are inspected alongside the landing page and the
        generated steps hit REAL interior screens.

        Login-gated apps are reached by uploading a Playwright storage-state JSON
        under the caller's upload prefix and naming it as
        ``auth_config.storage_state_path``; the job row stores only that
        reference, never the credential itself.
        """
        auth_config = config.get("auth_config") or {}
        storage_state = await self._storage_state_for(auth_config, tenant_id=tenant_id)
        urls, dropped = _interior_urls(target_url, config.get("target_pages") or [])
        if dropped:
            logger.warning(
                "web_url inspection capped at %d screens; dropped %d extra target_pages",
                _MAX_INSPECT_SCREENS,
                dropped,
            )

        fetcher = PageFetcher()
        try:
            responses = await fetcher.fetch_pages(
                urls,
                javascript_enabled=bool(config.get("javascript_enabled", True)),
                auth_config=auth_config,
                storage_state=storage_state,
            )
        finally:
            await fetcher.close()

        pages = [_page_from_scrape(r) for r in responses if not r.error]
        if not pages:
            first_error = next((r.error for r in responses if r.error), "no pages returned")
            raise RuntimeError(f"Failed to fetch {target_url}: {first_error}")
        crawl_data = {"pages": pages, "url": target_url}
        return asdict(self._adapter.from_web_crawl(crawl_data))

    async def _resolve_gherkin_upload_input(
        self, *, config: dict[str, Any], tenant_id: str | None
    ) -> dict[str, Any]:
        """Parse caller-uploaded feature files into a synthetic inspection result.

        ``feature_files`` is caller-supplied verbatim from the request body, so
        each entry is confined to the CALLER's own upload prefix
        (``uploads/{tenant_id}/``) by the storage chokepoint. A reference naming
        another tenant's upload reads as absent and is skipped; ``tenant_id`` is
        threaded from the request (never read off ``request.state`` down here)
        and an absent tenant refuses everything.
        """
        feature_files = config.get("feature_files", [])
        all_features: list[dict[str, Any]] = []
        # ⚠️ COUNT the refs that contributed nothing. ``retrieve_text`` returns
        # ``None`` for an absent/unreadable/cross-tenant ref and ``""`` for an
        # empty object; the old ``if text:`` collapsed both into a silent
        # ``continue``. A REFUSED ref is logged inside ``storage_service.retrieve``,
        # but a genuine MISS is not, and neither reached the job row or the
        # caller — so a job whose every upload was dropped ran on an EMPTY
        # corpus, let the generator invent the content, and completed green.
        # Counting is the whole fix: it does not fail the job (no comparable case
        # in this method does), it makes the loss expressible and loud.
        dropped_refs = 0
        for path in feature_files:
            text = await self._storage.retrieve_text(path, tenant_id=tenant_id or "")
            if not text:
                dropped_refs += 1
                continue
            parsed = self._parser.parse_feature_text(text)
            all_features.append(parsed.model_dump())
        if dropped_refs:
            logger.warning(
                "gherkin_upload: %d of %d feature_files refs were unreadable, absent, "
                "empty, or outside this tenant's upload prefix and contributed NO "
                "content (tenant=%s)",
                dropped_refs,
                len(feature_files),
                tenant_id or "<unset>",
            )
        # Build a synthetic inspection result from parsed features. The two
        # counters ride along into ``job.inspection_data`` (the row field this
        # dict is persisted to) so the loss is auditable per job and readable
        # through /result, not only in a pod log line.
        return {
            "pages": [],
            "navigation_graph": [],
            "platform": "web",
            "base_url": "",
            "parsed_features": all_features,
            "requested_feature_refs": len(feature_files),
            "dropped_feature_refs": dropped_refs,
        }

    async def _resolve_mobile_app_input(
        self, *, app_path: str, effective_token: str
    ) -> dict[str, Any]:
        """Inspect a mobile app through ``app_inspector_service`` (legacy path)."""
        from src.services.input.app_inspector_client import AppInspectorClient

        client = AppInspectorClient(service_url("app_inspector"), license_token=effective_token)
        try:
            insp_job_id = await client.start_inspection(app_path)
            insp_data = await client.wait_for_inspection(insp_job_id)
            result = self._adapter.from_mobile_inspection(insp_data)
            return asdict(result)
        finally:
            await client.close()

    async def _storage_state_for(
        self,
        auth_config: dict[str, Any],
        *,
        tenant_id: str | None,
    ) -> dict[str, Any] | None:
        """Read a Playwright storage-state JSON from the caller's upload prefix.

        ``storage_state_path`` is a storage reference (the same tenant-confinement
        chokepoint as ``feature_files``); a reference that resolves outside the
        caller's prefix reads as absent and the job fails loudly.
        """
        ref = auth_config.get("storage_state_path")
        if not ref:
            return None
        if not tenant_id:
            raise RuntimeError(
                "auth_config.storage_state_path requires a tenant identity on the request"
            )
        text = await self._storage.retrieve_text(str(ref), tenant_id=tenant_id)
        if not text:
            raise RuntimeError(
                f"storage_state_path {ref!r} is absent, empty, or outside this tenant's uploads"
            )
        try:
            state = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"storage_state_path {ref!r} is not valid JSON: {exc}") from exc
        if not isinstance(state, dict) or not isinstance(state.get("cookies"), list):
            raise RuntimeError(
                f"storage_state_path {ref!r} must hold a Playwright storage-state object "
                "(with a 'cookies' list)"
            )
        return state

    async def _resolve_html_snapshot_input(
        self,
        *,
        config: dict[str, Any],
        tenant_id: str | None,
    ) -> dict[str, Any]:
        """Inspect caller-captured pages from an uploaded ``html_snapshot_path``.

        The snapshot is a JSON list of ``{url, html, title?, text?, links?,
        images?}`` entries — exactly what a Playwright capture writes — so an
        authenticated or VPN-only app can be inspected without the cluster
        reaching it. Structure is derived by the same ``extract_structure`` the
        live-fetch path uses.
        """
        ref = config.get("html_snapshot_path")
        if not ref:
            raise RuntimeError("html_snapshot_path is required for html_snapshot input")
        if not tenant_id:
            raise RuntimeError("html_snapshot_path requires a tenant identity on the request")
        text = await self._storage.retrieve_text(str(ref), tenant_id=tenant_id)
        if not text:
            raise RuntimeError(
                f"html_snapshot_path {ref!r} is absent, empty, or outside this tenant's uploads"
            )
        try:
            entries = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"html_snapshot_path {ref!r} is not valid JSON: {exc}") from exc
        if not isinstance(entries, list):
            raise RuntimeError(f"html_snapshot_path {ref!r} must hold a JSON list of pages")
        if len(entries) > _MAX_INSPECT_SCREENS:
            logger.warning(
                "html_snapshot capped at %d pages; dropped %d",
                _MAX_INSPECT_SCREENS,
                len(entries) - _MAX_INSPECT_SCREENS,
            )
            entries = entries[:_MAX_INSPECT_SCREENS]

        pages: list[dict[str, Any]] = []
        for entry in entries:
            page = _snapshot_entry_page(entry)
            if page is not None:
                pages.append(page)
        if not pages:
            raise RuntimeError(
                f"html_snapshot_path {ref!r} contains no usable pages "
                "(each entry needs a non-empty 'url' and 'html')"
            )
        crawl_data = {"pages": pages, "url": pages[0]["url"]}
        return asdict(self._adapter.from_web_crawl(crawl_data))

    async def _resolve_input(
        self,
        *,
        input_type: str,
        target_url: str,
        app_path: str,
        config: dict[str, Any],
        tenant_id: str | None = None,
        license_token: str | None = None,
    ) -> dict[str, Any]:
        """Resolve input source into normalized inspection data.

        ``license_token`` (the request-scoped license token) is forwarded to the
        remaining downstream peer client (app_inspector) so it authenticates via
        ``X-Tenant-Id: Bearer <token>`` rather than relying on the trusted-network
        fallback. Falls back to the static ``settings.downstream.license_token``
        for back-compat when the request-scoped token is not available (e.g.
        background reprocessing).
        """
        effective_token = license_token or getattr(self._settings.downstream, "license_token", "")
        if input_type == "knowledge":
            return self._resolve_knowledge_input(config)

        if input_type == "web_url":
            return await self._resolve_web_url_input(
                target_url=target_url,
                config=config,
                tenant_id=tenant_id,
            )

        if input_type == "html_snapshot":
            return await self._resolve_html_snapshot_input(config=config, tenant_id=tenant_id)

        if input_type == "mobile_app":
            return await self._resolve_mobile_app_input(
                app_path=app_path, effective_token=effective_token
            )

        if input_type == "gherkin_upload":
            return await self._resolve_gherkin_upload_input(config=config, tenant_id=tenant_id)

        if input_type == "inspection_passthrough":
            inspection_job_id = config.get("inspection_job_id", "")
            if inspection_job_id:
                from src.services.input.app_inspector_client import AppInspectorClient

                client = AppInspectorClient(
                    service_url("app_inspector"), license_token=effective_token
                )
                try:
                    insp_data = await client.get_inspection(inspection_job_id)
                    result = self._adapter.from_dict(insp_data)
                    return asdict(result)
                finally:
                    await client.close()
            return {"pages": [], "navigation_graph": [], "platform": "web"}

        raise ValueError(f"Unknown input type: {input_type}")

    async def _generate_features(
        self,
        stream: str,
        inspection: dict[str, Any],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Call the appropriate feature generator for the stream type.

        Raises ``RuntimeError`` when the generator yields ZERO features. There is no
        legitimate zero-feature success on any branch: each one is a generator that
        is supposed to produce features (the deterministic knowledge-pack branch is
        only taken when ``screens`` is non-empty). An empty list means the package
        ships without a single ``.feature`` file, so bddgen runs nothing, no clip is
        recorded, and ``mark_completed(features_generated=0)`` reports success for an
        empty deliverable — the silent-zero class this service exists to avoid.
        """
        if stream == "qa_suite":
            if _grounded_qa_enabled(config):
                # Deterministic floor: page contracts + captured-link navigation,
                # built only from the snapshot. The opt-in LLM pass may add more,
                # but it speaks the same closed vocabulary and is validated below.
                features = build_grounded_features(inspection, config)
                if config.get("include_llm_scenarios"):
                    features = features + await self._feature_gen.generate_grounded_qa_features(
                        inspection, config
                    )
                return features
            features = await self._feature_gen.generate_qa_features(inspection, config)
        elif stream == "accessibility":
            features = await self._feature_gen.generate_accessibility_features(inspection, config)
        elif stream == "demo_video":
            screens = config.get("knowledge_screens")
            features = (
                # Deterministic path — a knowledge pack authored the screens.
                self._feature_gen.build_showcase_features(screens, config)
                if screens
                else await self._feature_gen.generate_demo_features(inspection, config)
            )
        elif stream == "documentation":
            features = await self._feature_gen.generate_documentation_features(inspection, config)
        else:
            raise ValueError(f"Unknown stream: {stream}")

        if not features:
            raise RuntimeError(
                f"Feature generation for stream '{stream}' produced zero feature files; "
                "the package would contain no .feature file and record no clip. "
                "Usual cause: the model answered in a fence the parser could not read, "
                "or returned prose instead of Gherkin."
            )
        return features

    async def _generate_step_definitions(
        self,
        *,
        features: list[JsonDict],
        inspection: JsonDict,
        config: JsonDict,
        stream: str,
        coverage: StepCoverageReport,
        grounded_qa: bool,
    ) -> list[JsonDict]:
        """Pick the step-definition source for this job.

        Deterministic first: a knowledge pack or a grounded QA job ships a fixed,
        CI-compiled vocabulary. Only the remaining streams ask the LLM.
        """
        if config.get("knowledge_screens"):
            return self._config_gen.read_showcase_steps()
        if grounded_qa:
            # No LLM writes TypeScript on this path; that is what shipped createBd(),
            # non-async arrows and quoted "{string}" placeholders.
            return self._config_gen.read_qa_steps()
        return await self._step_def_gen.generate(
            features,
            inspection,
            inspection.get("platform", "web"),
            auth_config=config.get("auth_config"),
            recording=(stream == STREAM_DEMO_VIDEO),
            coverage=coverage,
        )

    def _generate_config_files(
        self,
        *,
        inspection: JsonDict,
        stream: str,
        auth_config: JsonDict | None,
        grounded_qa: bool,
        locale: str | None = None,
    ) -> tuple[str, str, str, str | None]:
        """Return ``(playwright_config, package_json, tsconfig, auth_setup)``."""
        base_url = inspection.get("base_url", "")
        if grounded_qa:
            # QA config: video OFF, one browser, env-overridable baseURL/locale. The
            # demo scaffold's 1080p video x two browsers is why the NetCare run hung.
            playwright_config = self._config_gen.generate_qa_config(base_url, locale=locale)
        else:
            # Video streams keep a missing step runnable (so the suite still records);
            # the strict streams want a missing step to fail at generation time.
            missing_steps = (
                "fail-on-gen"
                if stream in (STREAM_QA_SUITE, STREAM_ACCESSIBILITY)
                else "fail-on-run"
            )
            playwright_config = self._config_gen.generate_playwright_config(
                base_url,
                inspection.get("platform", "web"),
                has_auth=bool(auth_config),
                missing_steps=missing_steps,
            )
        auth_setup: str | None = None
        if auth_config:
            login_url = auth_config.get("login_url", base_url)
            auth_setup = self._config_gen.generate_auth_setup(login_url)
        return (
            playwright_config,
            self._config_gen.generate_qa_package_json()
            if grounded_qa
            else self._config_gen.generate_package_json(),
            self._config_gen.generate_tsconfig(),
            auth_setup,
        )

    async def _store_artifacts(
        self,
        job_id: str,
        features: list[dict[str, Any]],
        step_defs: list[dict[str, Any]],
        playwright_config: str,
        package_json: str,
        tsconfig: str,
        *,
        auth_setup: str | None = None,
        lib_files: dict[str, str] | None = None,
        coverage: StepCoverageReport | None = None,
        validation: JsonDict | None = None,
    ) -> JsonDict:
        """Write all generated files to storage and return path mapping."""
        paths: dict[str, Any] = {"feature_files": [], "step_def_files": []}

        for feature in features:
            filename = feature.get("filename", "feature.feature")
            content = feature.get("content", "")
            path = await self._storage.store_text(job_id, content, f"features/{filename}")
            paths["feature_files"].append(path)

        for step_def in step_defs:
            filename = step_def.get("filename", "steps.ts")
            content = step_def.get("content", "")
            path = await self._storage.store_text(job_id, content, f"steps/{filename}")
            paths["step_def_files"].append(path)

        paths["playwright_config"] = await self._storage.store_text(
            job_id, playwright_config, "playwright.config.ts"
        )
        paths["package_json"] = await self._storage.store_text(job_id, package_json, "package.json")
        paths["tsconfig"] = await self._storage.store_text(job_id, tsconfig, "tsconfig.json")

        if auth_setup:
            paths["auth_setup"] = await self._storage.store_text(
                job_id, auth_setup, "auth.setup.ts"
            )

        if lib_files:
            lib_paths: list[str] = []
            for relpath, content in lib_files.items():
                lib_paths.append(await self._storage.store_text(job_id, content, relpath))
            paths["lib_files"] = lib_paths

        if coverage is not None:
            # Rides INSIDE the downloadable package (the `*.json` include pattern
            # picks it up), so whoever runs the package — not only whoever polls
            # the API — can tell a stubbed suite from a real one.
            paths["step_coverage"] = await self._storage.store_text(
                job_id,
                json.dumps(coverage.as_dict(), indent=2, sort_keys=True),
                "step_coverage.json",
            )

        if validation is not None:
            # Same rule for grounding: the runner must be able to see which scenarios
            # were rejected (and why) without polling the API.
            paths["grounding_validation"] = await self._storage.store_text(
                job_id,
                json.dumps(validation, indent=2, sort_keys=True),
                "validation.json",
            )

        return paths

    def _plan_budget(
        self,
        stream: str,
        features: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> DurationBudgetPlan | None:
        """The per-scenario second budget for this job, or ``None``.

        ``None`` — today's behaviour — whenever no duration was ordered, or the
        stream is not the recording stream (only demo_video ships the dwell
        ledger, so a budget elsewhere would bound narration against footage that
        does not exist).

        The scenario count is COUNTED from the generated feature text with the
        same parser the narration pass uses, so the two agree by construction.
        """
        if stream != STREAM_DEMO_VIDEO:
            return None
        ordered = config.get("target_duration_seconds")
        if not isinstance(ordered, int | float) or isinstance(ordered, bool):
            return None
        scenario_count = sum(
            len(self._parser.parse_feature_text(str(f.get("content", ""))).scenarios)
            for f in features
        )
        return plan_duration_budget(
            float(ordered),
            scenario_count,
            words_per_minute=self._settings.narration.words_per_minute,
        )

    def _narration_word_budgets(
        self,
        feature_contents: list[str],
        words_target: object,
        words_per_scenario: object,
    ) -> list[int] | None:
        """The orchestrator-supplied per-scenario word budget, or ``None``.

        ``None`` — today's behaviour — whenever the caller sent neither
        ``narration_words_target`` nor a usable ``narration_words_per_scenario``.
        Unlike ``_plan_budget`` this is NOT gated on the stream or on an ordered
        duration: a caller that knows the word count it wants gets it honoured even
        when it ordered no duration, which is the whole point of the contract.

        The scenario count is COUNTED from the generated feature text with the same
        parser the narration pass uses, so the split lines up with the parts. Takes
        the already-extracted feature text (not the raw feature dicts) and the two
        raw config values so the signature carries no ``dict[str, Any]`` beyond what
        ``narration_word_budgets`` itself already accepts as untyped input.
        """
        scenario_count = sum(
            len(self._parser.parse_feature_text(content).scenarios) for content in feature_contents
        )
        return narration_word_budgets(words_target, words_per_scenario, scenario_count)

    async def _expand_short_sections(
        self,
        parts: list[NarrationSection],
        tone: str,
        word_budgets: list[int] | None,
        budget: DurationBudgetPlan | None,
        wpm: float,
        *,
        job_id: str | None = None,
    ) -> tuple[list[NarrationSection], PolishReport | None]:
        """Re-ask the LLM for any section that came back UNDER its word budget.

        Only the short sections are re-sent, with the length requirement leading
        the prompt, and an answer is accepted per section only when it is
        actually LONGER than what it replaces — a re-ask that came back shorter
        is a worse answer to the only question being asked, so it is dropped.
        The ceiling is re-applied to whatever is kept.

        Every re-asked section logs one ``narration_reask_section`` line —
        word counts only, never the text — so a re-ask that was silently
        dropped is distinguishable from one that landed. ``after_words`` is the
        re-ask ANSWER's length (what the keep/drop decision compared), not the
        post-trim length.
        """
        short: list[int] = []
        for index, part in enumerate(parts):
            words_budget = _word_budget_for(index, word_budgets, budget)
            if words_budget is None or words_budget <= 0:
                continue
            words = len(str(part.get("text", "")).split())
            if (
                words < words_budget * SHORTFALL_TRIGGER_RATIO
                and words_budget - words >= MIN_SHORTFALL_WORDS
            ):
                short.append(index)
        if not short:
            return parts, None

        subset = [parts[i] for i in short]
        subset_budgets = [int(_word_budget_for(i, word_budgets, budget) or 0) for i in short]
        logger.warning(
            "Narration UNDER budget on %d of %d section(s) after polish (%s); re-asking "
            "with the length requirement leading",
            len(short),
            len(parts),
            ", ".join(
                f"{parts[i].get('part_id')}: {len(str(parts[i].get('text', '')).split())}"
                f"/{subset_budgets[n]}"
                for n, i in enumerate(short)
            ),
        )
        expanded, report = await self._humanizer.expand_to_budget(subset, tone, subset_budgets)

        for n, index in enumerate(short):
            new_text = str(expanded[n].get("text", ""))
            before_words = len(str(parts[index].get("text", "")).split())
            after_words = len(new_text.split())
            replaced = after_words > before_words
            logger.info(
                "narration_reask_section job_id=%s section=%s budget_words=%d "
                "before_words=%d after_words=%d replaced=%s",
                job_id,
                parts[index].get("part_id") or index,
                subset_budgets[n],
                before_words,
                after_words,
                "true" if replaced else "false",
            )
            if not replaced:
                continue
            parts[index]["text"] = trim_to_word_budget(new_text, subset_budgets[n])
            parts[index]["word_count"] = len(str(parts[index]["text"]).split())
            parts[index]["estimated_duration_seconds"] = (
                (parts[index]["word_count"] / wpm) * 60 if wpm else 0.0
            )
        return parts, report

    async def _generate_narration(
        self,
        features: list[dict[str, Any]],
        narration_config: dict[str, Any],
        *,
        recording: bool = False,
        budget: DurationBudgetPlan | None = None,
        word_budgets: list[int] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate narration from features using template + optional LLM polish.

        One narration ``part`` is produced **per scenario** (not per feature),
        keyed by the deterministic ``@clip-<n>`` tag stamped at generation time.
        ``part_id`` is the join spine the demo-video pipeline uses to bind a
        scenario's narration audio to its recorded Playwright clip and its
        compile shot. Order follows document order so the playback sequence is
        stable.

        ``recording`` marks the demo_video stream, whose package carries the
        force-injected pacing hook. Only then does each part publish
        ``expected_recording_seconds`` — the guaranteed per-scenario clip floor.
        The other streams ship no ``lib/`` and no hook, so claiming a recording
        duration for them would be a fabricated number.

        ``budget`` is the ordered-duration split, when one was ordered. It caps
        each part's word count and becomes the published
        ``expected_recording_seconds``, so the spoken line and the clip it rides
        are sized from the same number. ``None`` leaves narration exactly as it
        is generated today.

        ``word_budgets`` is the ORCHESTRATOR's own per-scenario word budget, when it
        sent one. It OVERRIDES the locally derived word count, because the caller
        knows the delivered speech rate while this service only knows a configured
        ``wpm``. It is also the number handed to the polish prompt, so the script is
        WRITTEN TO FILL the ordered duration instead of merely being trimmed back to
        it. ``None`` — the payload of any orchestrator that has not deployed the
        contract yet — changes nothing.
        """
        tone = narration_config.get("tone", "tutorial")
        engine = NarrationTemplateEngine(tone=tone)
        wpm = self._settings.narration.words_per_minute
        # Authored spoken lines (knowledge-pack path) win over template synthesis,
        # keyed by the same clip id; the optional polish pass still runs on them.
        authored: dict[str, str] = narration_config.get("authored_narration") or {}
        # Per-clip expected on-screen text (O1b) — QA metadata, keyed by the same
        # clip id. Rides on each part so the demo-video record stage can assert it
        # via OCR. The polish pass copies every non-``text`` key, so it survives.
        expected_text: dict[str, str] = narration_config.get("authored_expected_text") or {}

        parts: list[NarrationSection] = []
        fallback_index = 0
        for feature in features:
            content = feature.get("content", "")
            parsed = self._parser.parse_feature_text(content)
            for scenario in parsed.scenarios:
                fallback_index += 1
                part_id = _clip_id_for(scenario.tags, fallback_index)
                # `fallback_index` is 1-based and follows the same document order the
                # orchestrator planned its split in.
                part_words = _word_budget_for(fallback_index - 1, word_budgets, budget)
                steps_as_dicts = [
                    {"keyword": s.keyword, "text": s.text}
                    for s in _narrated_steps(scenario, budgeted=part_words is not None)
                ]
                # The budget reaches the TEMPLATE, not just the trim. The template
                # emits one sentence per step and is otherwise blind to the budget,
                # so it handed the polish pass a body it had to nearly double in one
                # shot. Authored narration is passed through verbatim as before.
                fallback_before = engine.fallback_steps
                narrated = authored.get(part_id) or engine.humanize_scenario(
                    scenario.name, steps_as_dicts, word_budget=part_words
                )
                # Where the pre-polish line came from. Observability only — the
                # polish pass carries it through untouched like every non-text key.
                fallback_steps = engine.fallback_steps - fallback_before
                narration_source = (
                    "authored"
                    if authored.get(part_id)
                    else ("template_fallback" if fallback_steps else "pattern")
                )
                if part_words is not None:
                    # BUDGET THE WORDS, not just the picture. Narration and dwell
                    # are the two numbers this service authors, and a line written
                    # for a 24s clip spoken over a 15s one is truncated audio.
                    # Trimming is deterministic and sentence-aware; a prompt asking
                    # for brevity would be advisory, which is the arrangement that
                    # produced the bimodal-duration defect in the first place.
                    narrated = trim_to_word_budget(narrated, part_words)
                word_count = len(narrated.split())
                part: NarrationSection = {
                    "part_id": part_id,
                    "scenario_name": scenario.name,
                    "text": narrated,
                    "original_gherkin": _reconstruct_gherkin(scenario.name, scenario.steps),
                    "word_count": word_count,
                    "estimated_duration_seconds": (word_count / wpm) * 60 if wpm else 0.0,
                    "narration_source": narration_source,
                    "template_fallback_steps": fallback_steps,
                }
                if expected_text.get(part_id):
                    part["expected_text"] = expected_text[part_id]
                if recording:
                    # ⭐ THE EXPORTED PER-SCENARIO RECORDING DURATION, on the wire.
                    #
                    # `estimated_duration_seconds` above is how long the NARRATION
                    # takes to speak; this is how long the CLIP is guaranteed to
                    # last. They are different numbers and conflating them is how a
                    # 4-second narration line came to imply a 4-second clip.
                    #
                    # Owned by `HERO_DWELL_MS` in
                    # `templates/scaffold/lib/recording_helpers.ts`, read (not
                    # restated) via `step_def_generator`. w2v_orchestrator must size
                    # its scenario count off THIS instead of its own hardcoded
                    # `_SECONDS_PER_SCENARIO`, so the number lives in one repo.
                    #
                    # A FLOOR, not a prediction: the scenario's own step actions run
                    # before the dwell, so real clips land at or above it. Sizing a
                    # timeline off a floor under-fills at worst, never over-fills.
                    #
                    # With an ordered duration it is the BUDGETED share instead of
                    # the default — the ledger in `lib/recording_helpers.ts` was
                    # narrowed to exactly this number by `steps_dwell_budget.ts`,
                    # so the value the orchestrator reads keeps describing what the
                    # package will actually record.
                    #
                    # It is ``dwell_seconds``, i.e. the budget AFTER the hero-share
                    # cap — the number the ledger was actually narrowed to. Using
                    # the uncapped ``per_scenario_seconds`` here would publish a
                    # clip floor the package does not record.
                    part["expected_recording_seconds"] = (
                        budget.dwell_seconds
                        if budget is not None
                        else GUARANTEED_RECORDING_SECONDS_PER_SCENARIO
                    )
                parts.append(part)

        # Optional LLM polish pass (preserves every key except a smoothed ``text``).
        # BEST-EFFORT: polish is a cosmetic uplift over the already-complete deterministic
        # template narration. A WM/LLM failure here must NOT fail the whole job and discard
        # the generated features/step-defs/package — degrade to the template ``parts``.
        polish_report: PolishReport | None = None
        expansion_report: PolishReport | None = None
        if narration_config.get("llm_polish", True) and parts:
            try:
                parts, polish_report = await self._humanizer.polish_with_report(
                    parts, tone, word_budgets=word_budgets
                )
                # Recompute duration estimates from the polished text. The polish
                # pass rewrites the line, so the budget is re-applied to its
                # output — otherwise a longer polished sentence would silently
                # spend past the budget the dwell was already narrowed to.
                _reapply_word_budget(parts, word_budgets, budget, wpm)
                # THE MISSING SYMMETRIC HALF. ``trim_to_word_budget`` above is a
                # CEILING and only a ceiling — nothing in this service could ever
                # lengthen a section that came back UNDER its budget, so a short
                # script left the slot unfilled and the orchestrator stretched the
                # speech (its 0.80 rate floor) to cover the gap. Measured live: this
                # service returned ~58% of the budget on every beat of two runs.
                parts, expansion_report = await self._expand_short_sections(
                    parts, tone, word_budgets, budget, wpm, job_id=job_id
                )
            except (OSError, RuntimeError, httpx.HTTPError, ValueError) as exc:
                # ``ValueError`` is NOT defensive padding — it is the base of the two
                # failure modes this exact call stack actually produces, neither of
                # which is an OSError/RuntimeError/HTTPError, so both escaped a handler
                # whose whole purpose is that they must not:
                #
                # * ``w2v_common.tenant.UnvouchedTenantError`` (-> ``TenantError`` ->
                #   ``ValueError``) is raised by the outbound identity event hook in
                #   ``w2v_common.auth.identity_context`` when a DELEGATED hop to a
                #   METERED destination has unvouched provenance. worker_management is
                #   a metered destination, so every polish call goes through that gate.
                # * ``json.JSONDecodeError`` (-> ``ValueError``) from
                #   ``BaseServiceClient._request``'s ``return response.json()`` when WM
                #   answers 2xx with a non-JSON body (an ingress/proxy error page).
                #
                # Either one marked the job ``failed`` and discarded the already-complete
                # features, step-defs and package — for a pass this comment calls
                # cosmetic — and named WM in the error, so it read as a WM outage.
                # Same shape as the ``SQLAlchemyError`` added to the usage-recording
                # handler above: catch what the layer below actually raises.
                logger.warning(
                    "Narration LLM polish (best-effort) failed; using template narration: %s",
                    exc,
                )

        total_words = sum(p.get("word_count", 0) for p in parts)
        script: dict[str, Any] = {
            "parts": parts,
            "total_words": total_words,
            "estimated_total_duration_seconds": (total_words / wpm) * 60 if wpm else 0,
            "tone": tone,
        }
        report = _word_budget_report(parts, word_budgets, budget, polish_report, expansion_report)
        if report is not None:
            # STRUCTURED, not prose. The residual shortfall is what the orchestrator
            # needs in order to stop stretching speech blindly: it can read exactly
            # how many words short each beat is (and whether an LLM response was
            # discarded on the way) instead of inferring it from a log line.
            script["word_budget_report"] = report
        return script
