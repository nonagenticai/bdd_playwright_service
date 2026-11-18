"""Cross-tenant confinement for caller-supplied ``feature_files`` storage refs.

``feature_files`` is a list of storage references supplied verbatim in the
request body (``QASuiteRequest`` / ``DocumentationRequest`` /
``DemoVideoRequest``). Uploads land under ``uploads/{tenant_id}/`` — a guessable
namespace — so an unconfined read lets tenant A name tenant B's upload and have
its Gherkin content parsed into A's ``inspection_data``, readable through
``/result`` and ``/download``.

Every case here drives the real exploit path (``JobProcessor._resolve_input``
with ``input_type="gherkin_upload"``) against a REAL ``StorageService`` on a
real temp storage root and a REAL ``GherkinParser`` — nothing about the read is
mocked, so a passing test is evidence about the shipped code path, not about a
stub.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.services.job_processor import JobProcessor
from src.services.storage_service import StorageService

TENANT_A = "01950000-0000-7000-8000-00000000000a"
TENANT_B = "01950000-0000-7000-8000-00000000000b"

TENANT_B_FEATURE = "Feature: TENANT B SECRET\n  Scenario: leak\n    Given I exist\n"
A_OWN = "Feature: TENANT A OWN\n  Scenario: mine\n    Given I exist\n"


@pytest.fixture
def storage_root(monkeypatch) -> Iterator[Path]:
    """Point the local storage backend at a fresh temp root."""
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("STORAGE_LOCAL_PATH", tmpdir)
    from src.config import clear_settings_cache

    clear_settings_cache()
    yield Path(tmpdir)
    clear_settings_cache()


@pytest.fixture
async def seeded(storage_root: Path) -> dict[str, str]:
    """Seed one upload per tenant; return their storage refs."""
    storage = StorageService()
    ref_b = await storage.store_bytes(
        f"uploads/{TENANT_B}", TENANT_B_FEATURE.encode("utf-8"), "secret.feature"
    )
    ref_a = await storage.store_bytes(f"uploads/{TENANT_A}", A_OWN.encode("utf-8"), "own.feature")
    return {"a": ref_a, "b": ref_b, "root": str(storage_root)}


@pytest.fixture
def processor(storage_root: Path) -> JobProcessor:
    """A JobProcessor whose StorageService uses the temp root (LLM stubbed out)."""
    with patch("src.services.job_processor.LLMClient", return_value=MagicMock()):
        return JobProcessor()


async def _resolve(processor: JobProcessor, refs: list[str], tenant_id: str) -> dict[str, Any]:
    return await processor._resolve_input(
        input_type="gherkin_upload",
        target_url="",
        app_path="",
        config={"feature_files": refs},
        tenant_id=tenant_id,
    )


def _names(result: dict[str, Any]) -> list[str]:
    return [f.get("name", "") for f in result["parsed_features"]]


class TestCrossTenantFeatureFiles:
    """Tenant A must never read a ref that resolves outside ``uploads/tenant-a/``."""

    async def test_exploit_sibling_tenant_ref_is_refused(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """THE BUG: tenant A submits tenant B's upload ref and gets B's content."""
        result = await _resolve(processor, [seeded["b"]], TENANT_A)
        assert _names(result) == [], (
            f"cross-tenant read succeeded: {_names(result)} (leaked tenant B's feature)"
        )

    async def test_traversal_out_of_own_prefix_is_refused(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """``uploads/<a>/../<b>/x.feature`` must normalise to B and be refused."""
        ref = f"{seeded['root']}/uploads/{TENANT_A}/../{TENANT_B}/secret.feature"
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_relative_traversal_ref_is_refused(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """A store-relative ref with ``..`` segments is refused."""
        ref = f"uploads/{TENANT_A}/../{TENANT_B}/secret.feature"
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_absolute_path_outside_root_is_refused(
        self, processor: JobProcessor, seeded: dict[str, str], tmp_path: Path
    ):
        """An absolute path outside the storage root entirely is refused."""
        outside = tmp_path / "etc_passwd.feature"
        outside.write_text("Feature: OUTSIDE ROOT\n  Scenario: s\n    Given I exist\n")
        result = await _resolve(processor, [str(outside)], TENANT_A)
        assert _names(result) == []

    async def test_url_encoded_traversal_is_refused(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """``%2e%2e`` / ``%2f`` must be normalised before comparing, then refused."""
        ref = f"uploads/{TENANT_A}/%2e%2e%2f{TENANT_B}/secret.feature"
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_double_encoded_traversal_is_refused(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """``%252e%252e`` (double-encoded) is refused too."""
        ref = f"uploads/{TENANT_A}/%252e%252e%252f{TENANT_B}/secret.feature"
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_backslash_separator_traversal_is_refused(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """Backslashes must be treated as separators, not as filename characters."""
        ref = f"uploads/{TENANT_A}\\..\\{TENANT_B}\\secret.feature"
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_s3_url_ref_is_refused(self, processor: JobProcessor, seeded: dict[str, str]):
        """A full ``s3://`` URL is not a confined ref."""
        ref = f"s3://w2v-bdd-playwright/uploads/{TENANT_B}/secret.feature"
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_http_url_ref_is_refused(self, processor: JobProcessor, seeded: dict[str, str]):
        """A full ``http://`` URL is not a confined ref."""
        ref = f"http://minio.test:8389/w2v-bdd-playwright/uploads/{TENANT_B}/secret.feature"
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_sibling_prefix_lookalike_is_refused(
        self, processor: JobProcessor, storage_root: Path
    ):
        """``uploads/<tenant>evil/`` merely STARTS WITH the tenant id — refuse it.

        This is the case a naive ``startswith(f"uploads/{tenant}")`` check lets
        through; the comparison must be on path SEGMENTS.
        """
        storage = StorageService()
        ref = await storage.store_bytes(
            f"uploads/{TENANT_A}evil", TENANT_B_FEATURE.encode("utf-8"), "secret.feature"
        )
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_job_artifact_prefix_is_not_readable(
        self, processor: JobProcessor, storage_root: Path
    ):
        """Refs must live under ``uploads/``; a job-artifact key is not a valid input."""
        storage = StorageService()
        ref = await storage.store_bytes(
            "some-other-job", TENANT_B_FEATURE.encode("utf-8"), "x.feature"
        )
        result = await _resolve(processor, [ref], TENANT_A)
        assert _names(result) == []

    async def test_missing_tenant_refuses_everything(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """No tenant in scope → fail closed, not open."""
        result = await _resolve(processor, [seeded["a"]], "")
        assert _names(result) == []


class TestLegitimateFlowStillWorks:
    """Positive controls — without these, a check that refuses EVERYTHING passes."""

    async def test_own_ref_still_resolves_and_parses(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """The caller's own upload ref (as returned by /gherkin/upload) still works."""
        result = await _resolve(processor, [seeded["a"]], TENANT_A)
        assert _names(result) == ["TENANT A OWN"]
        assert result["parsed_features"][0]["scenarios"]

    async def test_own_store_relative_ref_still_resolves(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """A store-relative form of the caller's own ref also works."""
        result = await _resolve(processor, [f"uploads/{TENANT_A}/own.feature"], TENANT_A)
        assert _names(result) == ["TENANT A OWN"]

    async def test_own_ref_survives_mixed_batch_with_foreign_ref(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """A batch mixing own + foreign refs keeps own, drops foreign."""
        result = await _resolve(processor, [seeded["a"], seeded["b"]], TENANT_A)
        assert _names(result) == ["TENANT A OWN"]

    async def test_tenant_b_can_read_its_own_upload(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """Confinement is per-caller, not a global denylist of the seeded file."""
        result = await _resolve(processor, [seeded["b"]], TENANT_B)
        assert _names(result) == ["TENANT B SECRET"]


class TestStorageChokepoint:
    """Direct tests of the ``StorageService`` chokepoint on BOTH backends."""

    async def test_retrieve_requires_tenant_and_confines_local(self, seeded: dict[str, str]):
        storage = StorageService()
        assert await storage.retrieve_text(seeded["b"], tenant_id=TENANT_A) is None
        assert await storage.retrieve_text(seeded["a"], tenant_id=TENANT_A) == A_OWN

    async def test_minio_backend_is_confined_too(self, monkeypatch, tmp_path: Path):
        """Production is MinIO — the guard must fire BEFORE the object read.

        The prior code passed the ref straight to ``get_bytes(ref)`` on this
        backend with no check at all.
        """
        monkeypatch.setenv("STORAGE_BACKEND", "minio")
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
        monkeypatch.setenv("STORAGE_MINIO_BUCKET", "w2v-bdd-playwright")
        from src.config import clear_settings_cache

        clear_settings_cache()

        storage = StorageService()
        assert storage.backend == "minio"

        from unittest.mock import AsyncMock

        with (
            patch.object(
                type(storage._storage), "initialize", new_callable=AsyncMock, return_value=None
            ),
            patch.object(
                type(storage._storage),
                "get_bytes",
                new_callable=AsyncMock,
                return_value=TENANT_B_FEATURE.encode("utf-8"),
            ) as mock_get,
        ):
            assert (
                await storage.retrieve(f"uploads/{TENANT_B}/secret.feature", tenant_id=TENANT_A)
                is None
            )
            mock_get.assert_not_called()

            assert (
                await storage.retrieve(f"uploads/{TENANT_A}/own.feature", tenant_id=TENANT_A)
                is not None
            )
            mock_get.assert_called_once_with(f"uploads/{TENANT_A}/own.feature")

        clear_settings_cache()


class TestConfinementGuardDirectly:
    """Direct tests of ``_confine_to_tenant`` — backend-independent.

    The ``_resolve_input`` cases above can fail-safe on the local backend simply
    because a malformed ref names no existing file; these bind the GUARD itself,
    so a weakened check (naive ``startswith``, no decoding, no normalization)
    fails here even when the filesystem would have covered for it.
    """

    REFUSED: tuple[str, ...] = (
        # another tenant, verbatim
        f"uploads/{TENANT_B}/secret.feature",
        # traversal out of own prefix
        f"uploads/{TENANT_A}/../{TENANT_B}/secret.feature",
        "../../etc/passwd",
        f"uploads/{TENANT_A}/../../uploads/{TENANT_B}/secret.feature",
        # url-encoded + double-encoded separators
        f"uploads/{TENANT_A}/%2e%2e%2f{TENANT_B}/secret.feature",
        f"uploads/{TENANT_A}/%252e%252e%252f{TENANT_B}/secret.feature",
        f"uploads%2f{TENANT_B}%2fsecret.feature",
        # backslash separators
        f"uploads\\{TENANT_B}\\secret.feature",
        f"uploads/{TENANT_A}\\..\\{TENANT_B}\\secret.feature",
        # absolute paths outside the storage root
        "/etc/passwd",
        f"/srv/other/uploads/{TENANT_A}/x.feature",
        # full URLs
        f"s3://w2v-bdd-playwright/uploads/{TENANT_B}/secret.feature",
        f"http://minio.test:8389/uploads/{TENANT_B}/secret.feature",
        f"https://minio.test:8389/uploads/{TENANT_A}/own.feature",
        # sibling whose NAME starts with the tenant id
        f"uploads/{TENANT_A}evil/secret.feature",
        f"uploads/{TENANT_A}-2/secret.feature",
        # not under uploads/ at all
        f"{TENANT_A}/secret.feature",
        "some-job-id/features/x.feature",
        # degenerate
        "",
        "uploads",
        f"uploads/{TENANT_A}",
        f"uploads/{TENANT_A}/",
        "uploads//secret.feature",
        f"uploads/{TENANT_A}/x\x00.feature",
    )

    ALLOWED: tuple[str, ...] = (
        f"uploads/{TENANT_A}/own.feature",
        f"uploads/{TENANT_A}/nested/own.feature",
        f"uploads/{TENANT_A}/./own.feature",
        f"uploads/{TENANT_A}/sub/../own.feature",
    )

    @pytest.mark.parametrize("ref", REFUSED)
    def test_refused(self, storage_root: Path, ref: str):
        assert StorageService()._confine_to_tenant(ref, TENANT_A) is None

    @pytest.mark.parametrize("ref", ALLOWED)
    def test_allowed(self, storage_root: Path, ref: str):
        key = StorageService()._confine_to_tenant(ref, TENANT_A)
        assert key is not None
        assert key.split("/")[:2] == ["uploads", TENANT_A]

    def test_absolute_own_ref_under_root_is_allowed(self, storage_root: Path):
        storage = StorageService()
        ref = str(storage_root / "uploads" / TENANT_A / "own.feature")
        assert storage._confine_to_tenant(ref, TENANT_A) == f"uploads/{TENANT_A}/own.feature"

    @pytest.mark.parametrize("tenant", ["", "..", ".", "a/b", f"../{TENANT_B}"])
    def test_degenerate_tenant_refuses(self, storage_root: Path, tenant: str):
        """A tenant id that is empty or itself path-shaped refuses everything."""
        assert (
            StorageService()._confine_to_tenant(f"uploads/{TENANT_A}/own.feature", tenant) is None
        )


class TestEverySchemaCarryingFeatureFiles:
    """Every request schema with ``feature_files`` must funnel into the guarded path.

    Coverage claim: the field exists on exactly these three schemas (accessibility
    has no ``feature_files``), and each route maps it to ``input_type ==
    "gherkin_upload"``, which is the single code path the chokepoint guards.
    """

    def test_exactly_these_schemas_carry_the_field(self):
        from src.api.schemas.accessibility import AccessibilityRequest
        from src.api.schemas.demo_video import DemoVideoRequest
        from src.api.schemas.documentation import DocumentationRequest
        from src.api.schemas.qa_suite import QASuiteRequest

        assert "feature_files" in QASuiteRequest.model_fields
        assert "feature_files" in DocumentationRequest.model_fields
        assert "feature_files" in DemoVideoRequest.model_fields
        assert "feature_files" not in AccessibilityRequest.model_fields

    def test_every_route_maps_feature_files_to_gherkin_upload(self):
        from src.api.routes.demo_video import _detect_input_type as demo_detect
        from src.api.routes.documentation import _detect_input_type as doc_detect
        from src.api.routes.qa_suite import _detect_input_type as qa_detect
        from src.api.schemas.demo_video import DemoVideoRequest
        from src.api.schemas.documentation import DocumentationRequest
        from src.api.schemas.qa_suite import QASuiteRequest

        refs = [f"uploads/{TENANT_B}/secret.feature"]
        assert qa_detect(QASuiteRequest(feature_files=refs))[0] == "gherkin_upload"
        assert doc_detect(DocumentationRequest(feature_files=refs))[0] == "gherkin_upload"
        assert demo_detect(DemoVideoRequest(feature_files=refs))[0] == "gherkin_upload"


class TestDroppedRefsAreCounted:
    """A ref that contributes NOTHING must be counted, logged, and put on the row.

    The confinement fix above made a hostile ref read as absent — correct, and
    deliberately silent to the attacker. But the same ``if text:`` also swallows
    an HONEST miss: a ref the caller mistyped, an object reaped by the bucket's
    expiry policy, an empty upload. ``None`` (absent) and ``""`` (empty) were
    indistinguishable from "skip", the count never left the loop, and a job whose
    every ref was dropped produced ``parsed_features == []``, let the generator
    invent the content, and completed green.

    Every case here drives the real path against a REAL ``StorageService`` on a
    real temp root — the same harness the confinement tests use.
    """

    async def test_missing_ref_is_counted(self, processor: JobProcessor, seeded: dict[str, str]):
        """A ref naming nothing at all is a dropped ref, not a silent skip."""
        result = await _resolve(processor, [f"uploads/{TENANT_A}/does-not-exist.feature"], TENANT_A)
        assert result["parsed_features"] == []
        assert result["requested_feature_refs"] == 1
        assert result["dropped_feature_refs"] == 1

    async def test_refused_cross_tenant_ref_is_counted(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """The confinement refusal stays silent to the caller but is still COUNTED."""
        result = await _resolve(processor, [seeded["b"]], TENANT_A)
        assert result["parsed_features"] == []
        assert result["dropped_feature_refs"] == 1

    async def test_partial_drop_is_counted_alongside_the_survivors(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """One good ref + two dead ones: 1 parsed, 2 dropped, 3 requested."""
        refs = [
            seeded["a"],
            seeded["b"],
            f"uploads/{TENANT_A}/typo.feature",
        ]
        result = await _resolve(processor, refs, TENANT_A)
        assert _names(result) != []
        assert result["requested_feature_refs"] == 3
        assert result["dropped_feature_refs"] == 2

    async def test_no_drops_reports_zero(self, processor: JobProcessor, seeded: dict[str, str]):
        """Control: an all-good submission reports zero drops, not a missing key."""
        result = await _resolve(processor, [seeded["a"]], TENANT_A)
        assert _names(result) != []
        assert result["requested_feature_refs"] == 1
        assert result["dropped_feature_refs"] == 0

    async def test_drops_are_logged_with_the_count(
        self, processor: JobProcessor, seeded: dict[str, str], caplog
    ):
        """The count must reach the operator's log, not only the row."""
        import logging

        with caplog.at_level(logging.WARNING, logger="src.services.job_processor"):
            await _resolve(
                processor,
                [f"uploads/{TENANT_A}/a.feature", f"uploads/{TENANT_A}/b.feature"],
                TENANT_A,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "an all-dropped gherkin_upload logged nothing"
        assert "2 of 2" in warnings[0].getMessage()

    async def test_all_dropped_is_visible_on_the_persisted_inspection_data(
        self, processor: JobProcessor, seeded: dict[str, str]
    ):
        """``_resolve_input``'s return IS ``job.inspection_data`` (job_processor
        assigns it verbatim), so asserting on this dict is asserting on the row."""
        result = await _resolve(processor, [seeded["b"], seeded["b"]], TENANT_A)
        assert result["dropped_feature_refs"] == result["requested_feature_refs"] == 2
