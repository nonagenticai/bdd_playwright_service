"""Package builder for assembling BDD artifacts into downloadable zip files."""

from __future__ import annotations

import asyncio
import logging
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

# Artifact patterns included in a downloadable package, matched against each
# stored key's path RELATIVE to the job prefix (e.g. ``features/login.feature``).
_INCLUDE_PATTERNS: tuple[str, ...] = (
    "features/*.feature",
    "steps/*.ts",
    "lib/*.ts",  # shared recording-helper library (demo_video packages)
    "lib/*.mjs",  # fixture-reset utilities (ESM, not caught by *.ts)
    "*.ts",  # root-level .ts files: playwright.config.ts, auth.setup.ts
    "*.json",
    "*.yml",
    "*.yaml",
)


class PackageBuilder:
    """Builds downloadable zip packages from generated BDD artifacts.

    Assembles the zip from the SHARED object store (not pod-local disk): it lists
    the job's stored artifacts, streams each into a zip on local scratch, then
    streams the finished zip back into the store. This keeps downloads working
    across replicas — the producing pod and the pod serving ``/download`` need
    not be the same.
    """

    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    async def build_package(
        self,
        job_id: str,
        stream: str,
        artifact_paths: dict[str, Any],
    ) -> str:
        """Build a zip package of generated artifacts and store it.

        Args:
            job_id: The job identifier.
            stream: The BDD stream type (qa_suite, accessibility, etc.).
            artifact_paths: Dict mapping artifact names to store references
                (unused for assembly — the zip is built from the store listing —
                kept for signature stability).

        Returns:
            The storage reference for the zip (MinIO object key or absolute local
            path), which the download route streams back.
        """
        # 1. Collect the job's stored artifacts (keys relative to the job prefix).
        entries = await self._collect_entries(job_id)

        # 2. Assemble the zip on local scratch (streaming each artifact in).
        packages_dir = self._storage.get_packages_dir(job_id)
        output_path = packages_dir / f"bdd-{stream}-{job_id[:8]}.zip"
        await asyncio.to_thread(self._write_zip, output_path, entries)

        # 3. Stream the finished zip into the shared store (never buffered whole
        #    in memory) and return the durable reference.
        ref = await self._storage.store_file(job_id, f"packages/{output_path.name}", output_path)
        logger.info("Built package %s for job %s (%d entries)", ref, job_id, len(entries))
        return ref

    async def _collect_entries(self, job_id: str) -> list[tuple[str, bytes]]:
        """Return ``(arcname, bytes)`` for every artifact that belongs in the zip.

        Lists the job's stored references, filters to the include patterns while
        excluding the packages dir itself (so a prior zip is never nested), and
        reads each artifact's bytes from the store.
        """
        refs = await self._storage.list_job_artifacts(job_id)
        entries: list[tuple[str, bytes]] = []
        seen: set[str] = set()
        for ref in refs:
            arcname = self._arcname(job_id, ref)
            if arcname is None or arcname in seen:
                continue
            rel = PurePosixPath(arcname)
            if "packages" in rel.parts:
                continue
            if not any(rel.match(pattern) for pattern in _INCLUDE_PATTERNS):
                continue
            data = await self._storage.get_artifact_bytes(ref)
            if data is None:
                logger.warning("Artifact %s vanished while building package for %s", ref, job_id)
                continue
            seen.add(arcname)
            entries.append((arcname, data))
        return entries

    @staticmethod
    def _arcname(job_id: str, ref: str) -> str | None:
        """Strip the ``{job_id}/`` prefix from a stored reference to get the
        in-zip name; ``None`` if the reference is not under the job prefix."""
        rel = PurePosixPath(ref)
        parts = rel.parts
        if job_id not in parts:
            return None
        idx = parts.index(job_id)
        tail = parts[idx + 1 :]
        return str(PurePosixPath(*tail)) if tail else None

    @staticmethod
    def _write_zip(output_path: Path, entries: list[tuple[str, bytes]]) -> Path:
        """Create a deflate zip from ``(arcname, bytes)`` entries.

        Written incrementally to ``output_path`` on disk — only one artifact's
        bytes are held in memory at a time (BDD artifacts are small text files;
        the large one is the zip itself, which callers stream via ``put_file``).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for arcname, data in entries:
                zf.writestr(arcname, data)
        logger.debug("Created zip %s with %d entries", output_path, len(entries))
        return output_path
