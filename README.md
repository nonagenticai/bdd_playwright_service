# BDD Playwright Service

Generates Gherkin feature files and playwright-bdd TypeScript step definitions from website and app inspection data, plus humanized narration. Port 6313.

## What BDD Playwright Service Does

When the orchestrator (or a developer) needs an executable test suite, an accessibility audit, interactive documentation, or a product demo video for a web or mobile app, it POSTs a job to this service. The request carries an input source — a live URL to inspect, an `app_path` for a mobile screen (resolved through `app_inspector_service`), uploaded `.feature` files, or a passthrough `inspection_job_id`. The service detects which source is present, records the job in PostgreSQL, and returns a `job_id` immediately; all real work happens asynchronously in a background processor so callers poll rather than block.

Behind the async boundary the service runs a multi-pass generation pipeline. A deterministic template pass turns inspection data into candidate scenarios, then LLM passes author the Gherkin (`feature_generator`), synthesize the TypeScript step definitions (`step_def_generator`), and humanize the narration (`humanizer`). Every LLM call is routed through `worker_management`'s `POST /workers/{name}/message` gateway using bound worker names (`bdd-feature-generator`, `bdd-step-def-generator`, `bdd-narration-humanizer`) — the service never talks to inference-router or model providers directly, so cost and usage are attributed per worker. A deterministic dedup-and-strip pass then drops duplicate step definitions and normalizes blocks so the emitted `npx playwright test` project actually compiles; this is a compile guarantee, not cosmetic cleanup, because the LLM re-emits duplicates and a demo run hard-fails on the first compile error.

The output is a downloadable, self-contained playwright-bdd project: a config scaffold (`playwright.config.ts`, `package.json`, `tsconfig.json`), templated auth setup, the shared app-agnostic step-helper library (`lib/recording_helpers.ts`), the generated `.feature` files, and per-scenario narration. `package_builder` zips it and `storage_service` persists it to MinIO; the caller downloads the artifact or reads the structured result. Gherkin here serves a dual purpose — it is both an executable Playwright suite and the structured source for TTS narration.

For the demo-video stream the service is the first stage of a three-service pipeline and does not execute tests or record video itself. `feature_generator._stamp_clip_tags` stamps a deterministic `@clip-<n>` tag above every scenario in global document order (idempotent on re-stamp). That `clip-<n>` id is the join spine: per-scenario narration parts, the per-scenario Playwright video clips recorded downstream, and the orchestrator's compile shots all key off the same id. The generated package is handed via the orchestrator to `frontend_test_runner_service` (port 6316), which runs the suite and harvests one video clip per tagged scenario; the orchestrator then narrates and compiles them into the demo MP4. Because three services depend on those ids, scenarios must never be reordered or renumbered after stamping.

The service is stateful and tenant-isolated. It persists jobs, usage, and narration templates to Postgres 18 over psycopg3, caches through Redis when available (Redis failure is non-fatal), and runs a background cleanup loop that reaps stuck and expired jobs. Row-level security enforces tenant isolation once the app connects as the restricted `w2v_bdd_playwright_app` role under `TENANT_RLS_MODE=enforce`.

## Features

- **Four generation streams** — one async API produces QA suites, WCAG accessibility scenarios, screenshot documentation, and demo/tutorial videos so a single service covers the whole BDD-authoring surface for web and mobile apps
- **Multi-source input detection** — accepts a live URL, a mobile `app_path` (via `app_inspector_service`), uploaded `.feature` files, or an `inspection_job_id` passthrough, mutually detected per request so callers pick the source that fits their pipeline
- **Gateway-routed LLM generation** — every generation and narration LLM call goes through `worker_management`'s worker gateway under named workers, so the service never binds to a provider and every token is cost-attributed per worker
- **Deterministic compile guarantee** — dedup-and-strip passes remove duplicate step definitions and normalize blocks so the emitted playwright-bdd project compiles on the first `npx playwright test`, preventing hard failures in downstream demo runs
- **`@clip-<n>` join spine** — stamps a deterministic, idempotent clip tag on every scenario so narration parts, recorded video clips, and compile shots stay aligned across three services in the demo_video pipeline
- **Dual-purpose Gherkin** — the same feature files drive Playwright execution and supply the structured source for TTS narration, avoiding a second authoring step
- **Humanized narration** — a deterministic template engine plus an optional LLM polish pass and tone presets turn scenario steps into natural spoken narration with words-per-minute duration estimates
- **App-agnostic recording scaffold** — ships a shared `lib/recording_helpers.ts` step-helper library (draft-chaining, visible typing, coach-mark dismissal, wizard advancement) that carries no per-app knowledge, so generated and hand-authored packages share one canonical helper set
- **Async job model** — POST returns a `job_id` and work runs in a background processor, so long generation jobs never block the caller and status is polled
- **Background cleanup** — a lifespan loop reaps stuck jobs past a timeout and deletes jobs past the retention window so the jobs table and MinIO storage do not grow unbounded
- **Downloadable artifacts** — `package_builder` and `storage_service` assemble and persist a self-contained playwright-bdd project to MinIO for one-request download
- **Health and request tracing** — standard liveness/readiness probes plus a request-id middleware bind a correlation id before auth and handlers run

## Grounded QA suites (`POST /api/bdd/qa-suite`)

The QA stream defaults to **grounded** mode (`generation_mode: "grounded"`):

- Features are generated deterministically from the captured page structure — one
  page-contract scenario per captured page (open it, assert its heading, title,
  landmarks and URL path) plus a navigation scenario for links whose target was
  captured too. No absolute URLs: paths are resolved against `BASE_URL`.
- Step definitions are a **static, closed vocabulary** (`src/templates/scaffold/qa/
  steps.ts`), emitted verbatim. No LLM writes TypeScript on this path.
- Every scenario — deterministic or opt-in LLM — is validated against the snapshot
  before packaging. Scenarios asserting a heading/link/button/title/text the capture
  does not contain, an absolute URL, an uncaptured path, or a step outside the
  vocabulary are **dropped and reported** (`validation.json` in the package,
  `step_coverage.grounding_validation` on the job). If nothing survives, the job
  fails rather than shipping an empty suite.
- The package config is QA-specific: `video: 'off'`, one chromium project,
  `BASE_URL` overridable at run time.

Run a generated package with:

```bash
npm install
BASE_URL=https://your-app npx bddgen && npx playwright test
```

`include_llm_scenarios: true` lets the LLM add scenarios inside the same vocabulary;
they are grounded and dropped under the same rules. Requests carrying `auth_config`
(login-gated storage state) stay on `generation_mode: "legacy"` — the grounded
vocabulary has no authenticated setup project yet. Business-state flows (expiry,
retention, emailed confirmations) cannot be grounded in a snapshot and are
hand-authored, not generated.

The generated package's TypeScript is compiled in CI (`package-compile` in
`.github/workflows/ci-public.yml`: `bddgen` + `tsc --noEmit` over a rendered
package), because the service image has no Node runtime.

## Authentication

Auth enabled by default (`AUTH_ENABLED=true`; k8s sets it explicitly). The `w2v_common` `AuthMiddleware` reads the license token from `X-Tenant-Id: Bearer <license_token>` — **not** the `Authorization` header, which is reserved for Keycloak JWTs from the SPA. Tokens are introspected against the licenses service and cached for 5 minutes (`cache_ttl=300`). A per-tenant `RateLimitMiddleware` runs ahead of auth.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service, results cached 5 min |
| 2 | Trusted network | Synthetic in-cluster identity for source IPs in `TRUSTED_CIDRS`, off by default (`AUTH_TRUST_INCLUSTER_IDENTITY=false`, fail-closed) |

For service-to-service calls the license token is extracted per request and `X-Tenant-Id`, `X-License-Id`, and `X-Subject-Id` are forwarded as plain headers (no `Bearer` prefix) to `worker_management` and downstream services.

Row-level security enforces tenant isolation at the database: the `20260617_0000_rls` migration ENABLEs `tenant_isolation` and installs an `after_begin` GUC re-bind for multi-commit background jobs, active once the app connects as `w2v_bdd_playwright_app` with `TENANT_RLS_MODE=enforce`.

Exempt paths (no token required): `/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/ready`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/metrics`.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6313
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### BDD `/api/bdd`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/qa-suite` | Generate a downloadable playwright-bdd QA project |
| POST | `/accessibility` | Generate WCAG-focused BDD scenarios |
| POST | `/documentation` | Generate step-by-step docs with screenshots |
| POST | `/demo-video` | Generate a demo/tutorial package with `@clip-<n>` tags + narration |
| POST | `/gherkin/upload` | Accept uploaded `.feature` files as input |
| GET | `/{job_id}` | Poll job status |
| GET | `/{job_id}/result` | Retrieve the full generation result |
| GET | `/{job_id}/download` | Download the generated artifact zip |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Health check |
| GET | `/health/live` | Liveness probe |
| GET | `/healthz` | Liveness probe (alias) |
| GET | `/health/ready` | Readiness probe (checks DB) |
| GET | `/readyz` | Readiness probe (alias) |

## Configuration

All configuration via environment variables. Defaults inherited from `w2v_common` base settings unless overridden.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6313` | Listen port (k8s sets `6313`; base default is `8080`, `8000` is rejected) |
| `API_DEBUG` | `false` | Enable debug mode |
| `AUTH_ENABLED` | `true` | Toggle license-token auth middleware |
| `AUTH_TRUST_INCLUSTER_IDENTITY` | `false` | Grant synthetic identity to trusted CIDRs (fail-closed by default) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_JSON_LOGS` | `false` | Structured JSON logging |
| `DATABASE_URL` | -- | PostgreSQL (psycopg3) connection URL |
| `DATABASE_POOL_SIZE` | `10` | Connection pool size |
| `DATABASE_POOL_MIN_SIZE` | `2` | Minimum connections in pool |
| `DATABASE_MAX_OVERFLOW` | `20` | Max connections beyond pool size |
| `TENANT_RLS_MODE` | -- | Set `enforce` to activate row-level security |
| `REDIS_URL` | -- | Redis connection URL (caching; non-fatal if absent) |
| `GENERATION_MAX_SCENARIOS_PER_FEATURE` | `10` | Max scenarios per feature file |
| `GENERATION_MAX_FEATURES_PER_JOB` | `20` | Max feature files per job |
| `GENERATION_PARALLEL_GENERATION_WORKERS` | `4` | Parallel LLM generation workers |
| `NARRATION_DEFAULT_TONE` | `tutorial` | Default narration tone |
| `NARRATION_LLM_POLISH_ENABLED` | `true` | Enable the LLM polish pass for narration |
| `NARRATION_WORDS_PER_MINUTE` | `150` | WPM used for narration duration estimates |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `30` | Max requests per minute per tenant |
| `RATE_LIMIT_CONCURRENT_JOBS_PER_TENANT` | `5` | Max concurrent jobs per tenant |
| `CLEANUP_RETENTION_DAYS` | `30` | Days to retain completed/failed jobs |
| `CLEANUP_STUCK_JOB_TIMEOUT_MINUTES` | `60` | Minutes before a processing job is considered stuck |

## Architecture

```
src/
├── asgi.py                            # FastAPI app, lifespan (DB/Redis/storage/cleanup), middleware, OpenAPI, health
├── config.py                          # Pydantic settings composed from w2v_common base settings
├── api/
│   ├── constants.py                   # Shared API constants
│   ├── health.py                      # Health/liveness/readiness endpoints
│   ├── routes/
│   │   ├── __init__.py                # Aggregates sub-routers under prefix /api/bdd
│   │   ├── qa_suite.py                # POST /qa-suite
│   │   ├── accessibility.py           # POST /accessibility
│   │   ├── documentation.py           # POST /documentation
│   │   ├── demo_video.py              # POST /demo-video
│   │   ├── gherkin.py                 # POST /gherkin/upload
│   │   ├── jobs.py                    # GET /{job_id}, /{job_id}/result
│   │   └── packages.py                # GET /{job_id}/download
│   └── schemas/                       # Per-stream request/response DTOs (qa_suite, accessibility, ...)
├── auth/
│   └── middleware.py                  # Thin re-export of w2v_common AuthMiddleware
├── middleware/
│   ├── rate_limiter.py                # Per-tenant rate limiting
│   └── request_id.py                  # Request-id correlation (outermost layer)
├── cache/
│   └── redis_client.py                # Redis client init/close
├── database/
│   ├── connection.py                  # Async engine lifecycle
│   ├── session.py                     # Session factory
│   ├── models.py                      # bdd_jobs, bdd_usage, narration_templates
│   ├── rls_events.py                  # after_begin GUC re-bind for RLS
│   ├── rls_resolvers.py               # Tenant resolution for RLS
│   └── migrations/versions/           # Alembic migrations (initial, chain-align, RLS)
├── services/
│   ├── job_manager.py                 # Job lifecycle (create/update/query)
│   ├── job_processor.py               # Async dispatch + @clip-<n> id mapping
│   ├── background.py                  # Background task plumbing
│   ├── cleanup_service.py             # Stuck/expired job reaping
│   ├── package_builder.py             # Assembles the downloadable project
│   ├── storage_service.py             # MinIO persistence
│   ├── generation/
│   │   ├── feature_generator.py       # Gherkin authoring + @clip tag stamping
│   │   ├── step_def_generator.py      # TypeScript step defs + dedup/strip
│   │   ├── config_generator.py        # Deterministic config scaffold (no LLM)
│   │   └── llm_client.py              # Adapter routing every LLM call via worker_management
│   ├── narration/
│   │   ├── template_engine.py         # Deterministic narration templating
│   │   ├── humanizer.py               # Optional LLM polish pass
│   │   └── tone_presets.py            # Narration tone presets
│   ├── input/
│   │   ├── app_inspector_client.py    # Mobile screen discovery client
│   │   ├── inspection_adapter.py      # Normalizes inspection data
│   │   └── gherkin_parser.py          # Parses uploaded .feature files
│   └── clients/
│       ├── tts_client.py              # TTS narration audio client
│       ├── content_client.py          # Optional narration enhancement client
│       └── video_production_client.py # Optional recording client
└── templates/                         # Recording-package scaffold: playwright config, auth.setup.ts,
                                       # lib/recording_helpers.ts (canonical), lib/fixture_reset.mjs
```

The `specra/` directory (git-excluded, local-only) is a reference recording package that consumes the `templates/` scaffold — it is not part of the service and is not published.

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run pytest tests/unit/ -v               # Unit tests
uv run pytest tests/integration/ -v        # Integration tests
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

Top-level tests include `test_auth_middleware.py`, `test_openapi.py`, and `test_rls_isolation.py`. Coverage threshold: 80% (enforced in `pyproject.toml`).

This service *generates* Gherkin `.feature` files and playwright-bdd suites as its product — those suites are executed downstream by `frontend_test_runner_service`, not here. There is no BDD suite checked into this repo's `tests/`.

Run `bash scripts/ci-local.sh` before pushing — it mirrors the archetype `Lint & Unit Tests` job exactly (uv sync, ruff check, ruff format, ty check src/, pytest, detect-secrets, pip-audit, bandit), so a green local run guarantees a green CI lint/unit stage.

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container, and `scripts/pip-audit.sh`. No image is published from this repository.


Per PR: ruff + ty + pytest + bandit + pip-audit + detect-secrets, then build an amd64 image, deploy to a per-commit `test-<sha>` namespace, and `kubectl rollout status` to confirm the pod reaches Ready. On merge to main the two-tag promote (`:sha-<commit>` → in-cluster smoke → `:latest`) runs and Flux rolls out the new digest.

Image pushed to `registry.test:5000/bdd_playwright_service:sha-<commit>`, promoted to `:latest` on success.

## Docker

```bash
docker build -t bdd-playwright-service .
docker run -p 6313:6313 bdd-playwright-service
docker run -p 6313:6313 -e LOG_LEVEL=DEBUG -e AUTH_ENABLED=false -e DATABASE_URL=postgresql+psycopg://host/db bdd-playwright-service
```

## Dependencies

- **worker_management** (port 8887) — gateway for every LLM generation and narration call via `POST /workers/{name}/message`
- **app_inspector_service** (port 6311) — mobile screen discovery for `app_path` inputs
- **frontend_test_runner_service** (port 6316) — runs the generated suite and harvests per-`@clip-<n>` video clips (reached via the orchestrator)
- **video_production_service** (port 6308) — optional recording
- **tts_service** (tts.test:9202) — narration audio synthesis
- **content_service** (port 6305) — optional narration enhancement
- **w2v_orchestrator** — upstream caller for demo_video, documentation, and QA stages
- **PostgreSQL 18** — persistent storage via psycopg3, through PgBouncer (`pgbouncer.pgbouncer:6432`) in-cluster
- **Redis** — caching (non-fatal if unavailable)
- **MinIO** — artifact storage for downloadable packages
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic v2 / pydantic-settings** — request/response validation and settings
- **SQLAlchemy (async)** — ORM over psycopg3
- **Alembic** — database migrations (run in an initContainer, never `create_all`)
- **httpx** — async HTTP client for downstream calls
- **Jinja2** — scaffold and narration templating
- **w2v-common** (>=0.30.0) — shared middleware, auth, settings, error envelope, RLS helpers

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. Tenant and licence identity is forwarded from the caller. It runs with
no licence switch set; unset means off.
