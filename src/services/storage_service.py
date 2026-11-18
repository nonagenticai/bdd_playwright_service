"""File storage service for BDD generation artifacts.

Thin domain wrapper over the fleet-shared
:class:`w2v_common.storage.StorageService` facade. The raw object I/O
(``put_bytes`` / ``get_bytes`` / ``put_file`` / ``get_object_stream``) is
delegated to the shared facade so every service shares one storage
implementation; this wrapper keeps only the bdd-specific concerns the facade
has no notion of:

* the job/features/steps/packages **key layout** (``{job_id}/…``), and
* the **path-traversal boundary** (``_safe_key`` / ``_within_root``) that
  confines every write target to the job/upload prefix — the facade does not
  sandbox arbitrary ``../`` segments in caller-supplied filenames, so the guard
  must live here, ahead of every delegated call, and
* the **cross-tenant read boundary** (``_confine_to_tenant``) that pins every
  caller-supplied read reference to ``uploads/{tenant_id}/`` on BOTH backends.

**Multi-replica safety.** This service runs (or will run) with >1 replica, and
the download route (``GET /api/bdd/{job_id}/download``) can land on any pod. The
prior implementation forced the ``local`` backend and wrote artifacts to a
pod-local ``emptyDir`` — a download that hit a non-producing replica 404'd even
though the DB row said ``completed``. We now honour the *configured* backend
(``STORAGE_BACKEND=minio`` in prod, bucket ``w2v-bdd-playwright``), so every
artifact lives in the shared object store and ``artifact_paths`` holds object
keys, not pod-local paths. ``local_path`` is retained only as the on-pod scratch
dir the package builder assembles + zips into before streaming the zip up.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

from w2v_common.storage import StorageService as SharedStorageService

from src.config import get_settings

if TYPE_CHECKING:
    from w2v_common.storage.service import StreamedObject

logger = logging.getLogger(__name__)

# The prefix every caller-supplied read reference must live under, and the number
# of percent-decoding passes applied before normalization. Three passes covers the
# single- and double-encoded forms; a reference still changing after that is
# adversarial by construction and is refused outright rather than decoded further.
UPLOADS_PREFIX = "uploads"
_MAX_DECODE_PASSES = 3


class StorageService:
    """Manages object storage for generated BDD artifacts.

    Honours the configured backend (``minio`` in prod, ``local`` in dev/CI).
    ``store_*`` returns the reference ``retrieve``/``open_stream`` read back —
    an object key for the MinIO backend, an absolute path for the local backend.
    """

    def __init__(self) -> None:
        settings = get_settings()
        # Honour the configured backend. Prod sets STORAGE_BACKEND=minio (bucket
        # w2v-bdd-playwright, keys already wired in gitops) so artifacts are
        # shared across replicas; dev/CI default to the local backend.
        self._storage = SharedStorageService(settings.storage)
        self._backend = settings.storage.backend
        self._local_path = Path(settings.storage.local_path)
        # Absolute storage boundary for the local backend. Every local write
        # target and read source is confined to this root so a path-traversal
        # segment in a caller-/LLM-supplied filename cannot escape it.
        self._root = self._local_path.resolve()
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def backend(self) -> str:
        """The active storage backend (``local`` or ``minio``)."""
        return self._backend

    # --- guards --------------------------------------------------------------

    def _safe_key(self, job_id: str, filename: str) -> str:
        """Compose ``{job_id}/{filename}`` into a normalized, traversal-safe key.

        Rejects any ``..`` segment that would escape the ``job_id`` prefix (an
        LLM-/caller-supplied filename could carry one). Works backend-agnostically
        on the object key string — MinIO has no filesystem to resolve against.
        """
        raw = f"{job_id}/{filename}"
        normalized = posixpath.normpath(raw)
        prefix = posixpath.normpath(job_id)
        if normalized != prefix and not normalized.startswith(prefix + "/"):
            raise ValueError(f"refusing to write outside storage root: {job_id}/{filename}")
        return normalized

    def _within_root(self, candidate: Path) -> bool:
        """Return True if ``candidate`` (resolved) is the local root or inside it."""
        return candidate == self._root or self._root in candidate.parents

    @staticmethod
    def _decode_ref(ref: str) -> str | None:
        """Percent-decode + separator-normalize a caller-supplied reference.

        Returns the decoded reference with ``\\`` folded to ``/``, or ``None`` if
        the reference is structurally not a store reference at all: a URL with a
        scheme (``s3://``, ``http://`` — those name an object we must not resolve
        by hand), a NUL byte, or a reference still mutating after
        ``_MAX_DECODE_PASSES`` decoding rounds.
        """
        if "://" in ref:
            return None
        current = ref
        for _ in range(_MAX_DECODE_PASSES):
            decoded = unquote(current)
            if decoded == current:
                break
            current = decoded
        else:
            return None
        if "\x00" in current:
            return None
        return current.replace("\\", "/")

    def _confine_to_tenant(self, ref: str, tenant_id: str) -> str | None:
        """Normalize ``ref`` and confine it to ``uploads/{tenant_id}/``.

        Returns the store-relative key (``uploads/{tenant_id}/…``) when the
        reference provably resolves inside the caller's own upload prefix, else
        ``None``. Fails CLOSED: an empty/absent tenant refuses everything.

        The comparison is on **path segments** of the *normalized* reference, not
        on a string prefix — a sibling directory whose name merely starts with the
        tenant id (``uploads/{tenant}evil/…``) is a different segment and is
        refused. Normalization collapses ``..`` first, so
        ``uploads/{a}/../{b}/x`` is compared as ``uploads/{b}/x``. Both backends
        share this guard: MinIO has no filesystem to resolve against, so the check
        is done on the key string and must precede the object read.
        """
        if not ref or not tenant_id or "/" in tenant_id or tenant_id in (".", ".."):
            return None

        decoded = self._decode_ref(ref)
        if decoded is None:
            return None

        normalized = posixpath.normpath(decoded)
        if posixpath.isabs(normalized):
            # An absolute reference is the LOCAL backend's shape (what ``store_*``
            # returns there); it is never a MinIO object key. Resolve it fully
            # (so a symlink planted inside the tree cannot point out of it) and
            # re-express it relative to the storage root.
            if self._backend != "local":
                return None
            candidate = Path(normalized).resolve()
            if not self._within_root(candidate):
                return None
            relative = candidate.relative_to(self._root).as_posix()
        else:
            relative = normalized

        segments = relative.split("/")
        if len(segments) < 3 or any(s in ("", ".", "..") for s in segments):
            return None
        if segments[0] != UPLOADS_PREFIX or segments[1] != tenant_id:
            return None
        return relative

    async def _ensure_ready(self) -> None:
        """Lazily initialize the backend once per instance (idempotent).

        Every entry point that touches the store funnels through here so a
        freshly-constructed ``StorageService`` (job processor, gherkin upload
        route) can reach MinIO without the caller having to remember to call
        :meth:`initialize`. For the local backend this just ensures the root dir
        exists; for MinIO it constructs the client + verifies the bucket.
        """
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._storage.initialize()
            self._initialized = True

    # --- lifecycle -----------------------------------------------------------

    async def initialize(self) -> None:
        """Ensure the backend is ready (create local dir / verify MinIO bucket)."""
        await self._ensure_ready()
        logger.info("Storage initialized (backend=%s, root=%s)", self._backend, self._local_path)

    async def aclose(self) -> None:
        """Close the MinIO client's HTTP session (idempotent; no-op on ``local``).

        ``miniopy_async.Minio`` lazily opens an ``aiohttp.ClientSession`` on its
        first request and never closes it on its own. Every ``JobProcessor`` (one
        per job) builds its own ``StorageService``, so without this each MinIO job
        leaked one session + connector, which aiohttp reported at GC time as
        ``asyncio ERROR Unclosed client session`` / ``Unclosed connector`` right
        after the package zip was streamed (live, 2026-10-18).

        The shared ``w2v_common.storage.StorageService`` facade exposes no close
        hook, so this reaches its ``_minio_client`` directly. A later use
        re-initializes (``_ensure_ready``) instead of touching a closed client.
        """
        client = getattr(self._storage, "_minio_client", None)
        close_session = getattr(client, "close_session", None)
        if close_session is not None:
            await close_session()
        self._initialized = False

    async def check_health(self) -> bool:
        """Is the CONFIGURED backend reachable right now?

        Delegates to the shared facade, which stats the bucket on ``minio`` and
        stats the root dir on ``local``. Deliberately does NOT call
        :meth:`_ensure_ready` — a probe must observe the store, never create a
        bucket as a side effect — so an instance that never initialized reports
        ``False`` rather than lazily fixing itself mid-probe.

        Exists because ``/health`` used to probe ``local_path`` unconditionally.
        In prod (``STORAGE_BACKEND=minio``) that path is only the pod-local
        scratch dir the package builder zips into: it exists and is writable on
        a pod whose object store is unreachable, so every job failed at the
        storage step while ``/health`` reported ``"storage": "healthy"`` — and
        the health cache then served that answer for its whole TTL.
        """
        return await self._storage.check_health()

    # --- local scratch dirs (package build workspace) ------------------------
    #
    # The package builder assembles + zips a job's artifacts on the producing
    # pod's local disk (``local_path``) before streaming the zip to the shared
    # store; these helpers create that scratch layout. They are NOT the durable
    # store — the durable copy is whatever ``store_*`` returns.

    def get_job_dir(self, job_id: str) -> Path:
        """Return the local scratch directory for a job, creating it if needed."""
        job_dir = self._local_path / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def get_features_dir(self, job_id: str) -> Path:
        """Return the local features sub-directory for a job."""
        features_dir = self.get_job_dir(job_id) / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        return features_dir

    def get_steps_dir(self, job_id: str) -> Path:
        """Return the local step definitions sub-directory for a job."""
        steps_dir = self.get_job_dir(job_id) / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        return steps_dir

    def get_packages_dir(self, job_id: str) -> Path:
        """Return the local packages sub-directory for a job."""
        packages_dir = self.get_job_dir(job_id) / "packages"
        packages_dir.mkdir(parents=True, exist_ok=True)
        return packages_dir

    # --- writes --------------------------------------------------------------

    async def store_text(self, job_id: str, content: str, filename: str) -> str:
        """Write text content under the job's prefix.

        Returns:
            The storage reference — a MinIO object key (minio backend) or an
            absolute path (local backend).
        """
        key = self._safe_key(job_id, filename)
        await self._ensure_ready()
        ref = await self._storage.put_bytes(key, content.encode("utf-8"))
        logger.debug("Stored %d chars to %s", len(content), ref)
        return ref

    async def store_bytes(self, job_id: str, data: bytes, filename: str) -> str:
        """Write raw bytes under the job's prefix.

        Returns:
            The storage reference (MinIO object key or absolute local path).
        """
        key = self._safe_key(job_id, filename)
        await self._ensure_ready()
        ref = await self._storage.put_bytes(key, data)
        logger.debug("Stored %d bytes to %s", len(data), ref)
        return ref

    async def store_file(self, job_id: str, filename: str, local_path: str | Path) -> str:
        """Stream a file already on local disk into the store WITHOUT buffering it
        in memory — the streaming path for large artifacts (the package zip).

        Uses the facade's ``put_file`` (``fput_object`` multipart for MinIO,
        buffered copy for local), so a big zip never lands in RAM.

        Returns:
            The storage reference (MinIO object key or absolute local path).
        """
        key = self._safe_key(job_id, filename)
        await self._ensure_ready()
        ref = await self._storage.put_file(key, local_path, content_type="application/zip")
        logger.debug("Streamed file %s to %s", local_path, ref)
        return ref

    # --- reads ---------------------------------------------------------------

    async def list_job_artifacts(self, job_id: str) -> list[str]:
        """List every stored reference under a job's prefix.

        Returns MinIO object keys (minio) or ``local_path``-relative paths
        (local), each starting with ``{job_id}/``. The package builder uses this
        to assemble the downloadable zip from the shared store.
        """
        await self._ensure_ready()
        return await self._storage.list_prefix(job_id)

    async def get_artifact_bytes(self, ref: str) -> bytes | None:
        """Read raw bytes for a reference returned by :meth:`list_job_artifacts`.

        Unlike :meth:`retrieve`, this takes a store reference/key directly (the
        facade joins a relative local key onto ``local_path``); it applies no
        local-root guard because the key came from our own prefix listing.
        """
        await self._ensure_ready()
        return await self._storage.get_bytes(ref)

    async def retrieve(self, ref: str, *, tenant_id: str) -> bytes | None:
        """Read a CALLER-SUPPLIED reference and return its bytes, or None.

        This is the single door for untrusted references (the ``feature_files``
        entries in the qa_suite / documentation / demo_video request bodies), so
        ``tenant_id`` is required and the read is confined to
        ``uploads/{tenant_id}/`` on BOTH backends — see
        :meth:`_confine_to_tenant`. Anything that normalizes outside that prefix
        (another tenant's tree, ``..`` traversal, an absolute path outside the
        storage root, a percent-encoded separator, an ``s3://``/``http://`` URL)
        is denied by returning ``None``, indistinguishable from a genuine miss.

        This previously claimed to deny "another tenant's tree" while only
        checking the storage ROOT — a sibling tenant directory is inside the root,
        so the claim was false and every tenant's uploads were world-readable to
        every other tenant. Reads of our OWN generated artifacts do not come
        through here: they use :meth:`get_artifact_bytes` (keys from our own
        prefix listing) and :meth:`open_stream` (a package ref from the job row,
        already tenant-scoped by ``JobManager.get_job``).
        """
        key = self._confine_to_tenant(ref, tenant_id)
        if key is None:
            logger.warning(
                "Refusing storage read outside caller's upload prefix (tenant=%s, ref=%r)",
                tenant_id,
                ref,
            )
            return None
        await self._ensure_ready()
        if self._backend == "local":
            file_path = (self._root / key).resolve()
            if not self._within_root(file_path) or not file_path.exists():
                return None
            return await self._storage.get_bytes(str(file_path))
        return await self._storage.get_bytes(key)

    async def retrieve_text(self, ref: str, *, tenant_id: str) -> str | None:
        """Read a caller-supplied reference as text, or None if absent/denied.

        Same confinement contract as :meth:`retrieve` — the reference must
        resolve inside ``uploads/{tenant_id}/`` on either backend.
        """
        data = await self.retrieve(ref, tenant_id=tenant_id)
        return data.decode("utf-8") if data is not None else None

    async def open_stream(
        self, ref: str, *, request_headers: dict[str, str] | None = None
    ) -> StreamedObject:
        """Open a streaming, range-aware read of a stored reference.

        Never buffers the whole object in memory — the download route streams the
        package zip straight from the shared store to the client. Forwards
        ``Range`` / ``If-*`` headers so partial/conditional GETs work. The caller
        MUST ``aclose()`` the result.
        """
        await self._ensure_ready()
        return await self._storage.get_object_stream(ref, request_headers=request_headers)
