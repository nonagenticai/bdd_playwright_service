"""BDD Playwright API routes.

⛔ ``LICENSE_GATE`` is mounted on the ONE top-level router below, not on the
individual sub-routers. Every route this package exposes lives under
``/api/bdd`` and every one of them already resolves a real tenant via
``src.api.routes._helpers.get_tenant_id`` (which 401s rather than defaulting) —
so the whole surface is tenant-scoped and BILLED. ``bdd_playwright_service`` is
in ``METERED_DESTINATION_SERVICES``; until now it authenticated *identity* and
never asked *entitlement*, so a tenant whose licence was expired or revoked
still presented a syntactically valid token and was served.

Mounting on the parent (rather than repeating ``dependencies=`` on seven
sub-routers) is deliberate: a new sub-router added here inherits the gate by
construction instead of by somebody remembering.

Dormant on merge: ``require_license`` is a no-op unless
``W2V_LICENSE_GATE_ENABLED=true``, which gitops arms separately.

Exemptions are w2v_common's, never ours: it exempts ``PUBLIC_ENDPOINTS`` plus
``/health``, ``/healthz`` and ``/ready``. The fleet probes ``/health/live`` and
``/health/ready``, which are served by ``src/api/health.py`` on a DIFFERENT
router that is not gated at all. Do not hand-roll an exemption list here — a
wrong one 402s the kubelet probe and CrashLoops the pod.
"""

from fastapi import APIRouter
from w2v_common.auth import LICENSE_GATE

from src.api.routes.accessibility import router as accessibility_router
from src.api.routes.demo_video import router as demo_video_router
from src.api.routes.documentation import router as documentation_router
from src.api.routes.gherkin import router as gherkin_router
from src.api.routes.jobs import router as jobs_router
from src.api.routes.packages import router as packages_router
from src.api.routes.qa_suite import router as qa_suite_router
from src.api.routes.snapshot import router as snapshot_router

router = APIRouter(prefix="/api/bdd", tags=["bdd"], dependencies=list(LICENSE_GATE))

router.include_router(jobs_router)
router.include_router(qa_suite_router)
router.include_router(accessibility_router)
router.include_router(documentation_router)
router.include_router(demo_video_router)
router.include_router(packages_router)
router.include_router(gherkin_router)
router.include_router(snapshot_router)
