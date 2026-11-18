"""Tests for PackageBuilder service."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import pytest

from src.services.package_builder import PackageBuilder
from src.services.storage_service import StorageService


@pytest.fixture
def storage_service(monkeypatch):
    """Create a StorageService with a temp directory."""
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("STORAGE_LOCAL_PATH", tmpdir)
    from src.config import clear_settings_cache

    clear_settings_cache()
    return StorageService()


@pytest.fixture
def package_builder(storage_service):
    """Create a PackageBuilder with a real StorageService."""
    return PackageBuilder(storage_service)


class TestPackageBuilder:
    async def test_build_package_creates_zip(self, package_builder, storage_service):
        """build_package should create a zip file from job artifacts."""
        job_id = "pkg-test-001"

        # Create sample artifacts
        features_dir = storage_service.get_features_dir(job_id)
        steps_dir = storage_service.get_steps_dir(job_id)

        (features_dir / "login.feature").write_text("Feature: Login\n  Scenario: Login\n")
        (features_dir / "dashboard.feature").write_text("Feature: Dashboard\n  Scenario: View\n")
        (steps_dir / "login.steps.ts").write_text("import { Given } from '@cucumber/cucumber';")
        (steps_dir / "dashboard.steps.ts").write_text("import { When } from '@cucumber/cucumber';")

        # Also create a config file in the job root
        job_dir = storage_service.get_job_dir(job_id)
        (job_dir / "playwright.config.json").write_text('{"timeout": 30000}')

        artifact_paths = {
            "features": str(features_dir),
            "steps": str(steps_dir),
        }

        result = await package_builder.build_package(job_id, "qa_suite", artifact_paths)

        assert Path(result).exists()
        assert result.endswith(".zip")

        # Verify zip contents
        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert "features/login.feature" in names
            assert "features/dashboard.feature" in names
            assert "steps/login.steps.ts" in names
            assert "steps/dashboard.steps.ts" in names
            assert "playwright.config.json" in names

    async def test_build_package_empty_job(self, package_builder):
        """build_package should create an empty zip for jobs with no artifacts."""
        job_id = "pkg-empty-001"
        result = await package_builder.build_package(job_id, "accessibility", {})

        assert Path(result).exists()
        with zipfile.ZipFile(result, "r") as zf:
            assert zf.namelist() == []

    async def test_build_package_excludes_packages_dir(self, package_builder, storage_service):
        """build_package should not include the packages directory itself."""
        job_id = "pkg-exclude-001"

        features_dir = storage_service.get_features_dir(job_id)
        (features_dir / "test.feature").write_text("Feature: Test\n")

        # Create a decoy file in packages dir
        packages_dir = storage_service.get_packages_dir(job_id)
        (packages_dir / "old.zip").write_bytes(b"fake zip")

        result = await package_builder.build_package(job_id, "documentation", {})

        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert "features/test.feature" in names
            assert not any("packages" in name for name in names)

    async def test_build_zip_with_yaml_files(self, package_builder, storage_service):
        """build_package should include YAML config files."""
        job_id = "pkg-yaml-001"

        job_dir = storage_service.get_job_dir(job_id)
        (job_dir / "config.yml").write_text("browser: chromium\n")
        (job_dir / "env.yaml").write_text("base_url: https://example.com\n")

        result = await package_builder.build_package(job_id, "qa_suite", {})

        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert "config.yml" in names
            assert "env.yaml" in names

    async def test_build_package_returns_correct_path(self, package_builder, storage_service):
        """build_package should return a path in the packages subdirectory."""
        job_id = "pkg-path-001"
        result = await package_builder.build_package(job_id, "demo_video", {})

        assert "packages" in result
        assert f"bdd-demo_video-{job_id[:8]}" in result

    async def test_build_package_includes_recording_lib(self, package_builder, storage_service):
        """A demo_video package ships the shared recording-helper lib — both the TS
        helpers (lib/*.ts) and the ESM fixture-reset utilities (lib/*.mjs)."""
        job_id = "pkg-lib-001"
        job_dir = storage_service.get_job_dir(job_id)
        lib_dir = job_dir / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        (lib_dir / "recording_helpers.ts").write_text("export const HERO_DWELL_MS = 22000;\n")
        (lib_dir / "fixture_reset.mjs").write_text(
            "export async function openWithSniffedAuth() {}\n"
        )

        result = await package_builder.build_package(job_id, "demo_video", {})

        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            assert "lib/recording_helpers.ts" in names
            assert "lib/fixture_reset.mjs" in names
