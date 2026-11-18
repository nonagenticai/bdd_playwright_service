"""Tests for storage service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.storage_service import StorageService


@pytest.fixture
def storage_service(monkeypatch):
    """Create a StorageService with a temp directory."""
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("STORAGE_LOCAL_PATH", tmpdir)
    from src.config import clear_settings_cache

    clear_settings_cache()
    return StorageService()


class TestStorageService:
    async def test_initialize(self, storage_service):
        """initialize should create the root storage directory."""
        await storage_service.initialize()
        assert storage_service._local_path.exists()

    def test_get_job_dir(self, storage_service):
        """get_job_dir should create and return job directory."""
        job_dir = storage_service.get_job_dir("test-job-123")
        assert job_dir.exists()
        assert job_dir.name == "test-job-123"

    def test_get_features_dir(self, storage_service):
        """get_features_dir should create and return features subdirectory."""
        features_dir = storage_service.get_features_dir("test-job-123")
        assert features_dir.exists()
        assert features_dir.name == "features"

    def test_get_steps_dir(self, storage_service):
        """get_steps_dir should create and return steps subdirectory."""
        steps_dir = storage_service.get_steps_dir("test-job-123")
        assert steps_dir.exists()
        assert steps_dir.name == "steps"

    async def test_store_text(self, storage_service):
        """store_text should write text content to a file."""
        path = await storage_service.store_text("job-1", "hello world", "test.txt")
        assert Path(path).exists()
        assert Path(path).read_text() == "hello world"

    async def test_store_bytes(self, storage_service):
        """store_bytes should write bytes to a file."""
        data = b"\x00\x01\x02\x03"
        path = await storage_service.store_bytes("job-1", data, "test.bin")
        assert Path(path).exists()
        assert Path(path).read_bytes() == data

    async def test_store_text_nested(self, storage_service):
        """store_text should create parent directories for nested paths."""
        path = await storage_service.store_text("job-1", "content", "sub/dir/file.txt")
        assert Path(path).exists()

    async def test_retrieve(self, storage_service):
        """retrieve should read file bytes for the caller's own upload ref.

        ``retrieve`` is the door for CALLER-SUPPLIED references only, so it takes
        a required ``tenant_id`` and reads only under ``uploads/{tenant_id}/``.
        """
        path = await storage_service.store_bytes(
            "uploads/01950000-0000-7000-8000-000000000001", b"data", "file.bin"
        )
        result = await storage_service.retrieve(
            path, tenant_id="01950000-0000-7000-8000-000000000001"
        )
        assert result == b"data"

    async def test_retrieve_nonexistent(self, storage_service):
        """retrieve should return None for missing files."""
        result = await storage_service.retrieve(
            "/nonexistent/file.bin", tenant_id="01950000-0000-7000-8000-000000000001"
        )
        assert result is None

    async def test_retrieve_text(self, storage_service):
        """retrieve_text should read file content as string."""
        path = await storage_service.store_text(
            "uploads/01950000-0000-7000-8000-000000000001", "hello", "file.txt"
        )
        result = await storage_service.retrieve_text(
            path, tenant_id="01950000-0000-7000-8000-000000000001"
        )
        assert result == "hello"

    async def test_store_text_rejects_traversal(self, storage_service):
        """store_text must refuse a filename that escapes the storage root."""
        with pytest.raises(ValueError, match="storage root"):
            await storage_service.store_text("job-1", "x", "../../etc/evil.feature")
        with pytest.raises(ValueError, match="storage root"):
            await storage_service.store_text("job-1", "x", "steps/../../../../../tmp/evil.ts")

    async def test_store_bytes_rejects_traversal(self, storage_service):
        """store_bytes must refuse a filename that escapes the storage root."""
        with pytest.raises(ValueError, match="storage root"):
            await storage_service.store_bytes("uploads/tenant", b"x", "../../../tmp/evil.feature")

    async def test_retrieve_denies_out_of_root(self, storage_service, tmp_path):
        """retrieve/retrieve_text must deny (return None) for out-of-root paths."""
        outside = tmp_path / "secret.txt"
        outside.write_text("top secret")
        assert (
            await storage_service.retrieve(
                str(outside), tenant_id="01950000-0000-7000-8000-000000000001"
            )
            is None
        )
        assert (
            await storage_service.retrieve_text(
                str(outside), tenant_id="01950000-0000-7000-8000-000000000001"
            )
            is None
        )


async def test_minio_backend_is_honoured(monkeypatch, tmp_path: Path) -> None:
    """Prod sets STORAGE_BACKEND=minio (bucket w2v-bdd-playwright). The wrapper
    now HONOURS it so generated artifacts live in the shared object store and a
    download served by any replica resolves — the prior code force-downgraded to
    the pod-local backend, 404'ing cross-pod. MinIO stores by object KEY, not a
    pod-local path."""
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
    monkeypatch.setenv("STORAGE_BACKEND", "minio")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.test:9000")
    monkeypatch.setenv("STORAGE_MINIO_BUCKET", "w2v-bdd-playwright")
    from src.config import clear_settings_cache

    clear_settings_cache()

    fake_client = MagicMock()
    fake_client.bucket_exists = AsyncMock(return_value=True)
    fake_client.put_object = AsyncMock()
    fake_minio_cls = MagicMock(return_value=fake_client)

    try:
        with patch("miniopy_async.Minio", fake_minio_cls):
            svc = StorageService()
            assert svc.backend == "minio"
            await svc.initialize()  # constructs the client, verifies the bucket
            ref = await svc.store_text("job-x", "hi", "features/a.feature")
        # The returned reference is the object KEY (survives across replicas),
        # not an absolute pod-local path.
        assert ref == "job-x/features/a.feature"
        fake_client.put_object.assert_awaited_once()
    finally:
        clear_settings_cache()


async def test_minio_store_text_rejects_traversal(monkeypatch, tmp_path: Path) -> None:
    """The traversal guard is backend-agnostic: it operates on the object key, so
    a ``..`` segment is rejected under the MinIO backend too (no filesystem to
    resolve against)."""
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
    monkeypatch.setenv("STORAGE_BACKEND", "minio")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.test:9000")
    from src.config import clear_settings_cache

    clear_settings_cache()
    try:
        with patch("miniopy_async.Minio", MagicMock()):
            svc = StorageService()
            with pytest.raises(ValueError, match="storage root"):
                await svc.store_text("job-x", "x", "../../etc/evil.feature")
    finally:
        clear_settings_cache()
