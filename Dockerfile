# Multi-stage Dockerfile for bdd_playwright_service
#
# Single-context build: w2v-common is resolved from pypiserver.test
# via the uv.lock file (which pins w2v-common to the pypiserver-internal index).
# Build with: docker buildx build -f Dockerfile .

ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Resolve and install deps; w2v-common comes from pypiserver.test
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself
COPY src/ ./src/
COPY alembic.ini ./alembic.ini

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 bddservice && \
    mkdir -p /app /app/logs /app/data && \
    chown -R bddservice:bddservice /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=bddservice:bddservice /app/.venv /app/.venv

# Copy application code
COPY --chown=bddservice:bddservice src/ ./src/
COPY --chown=bddservice:bddservice alembic.ini ./alembic.ini

ENV PATH="/app/.venv/bin:$PATH"

# Headless Chromium for the in-service page fetcher. Installed system-wide
# (PLAYWRIGHT_BROWSERS_PATH) so the unprivileged runtime user can read it and a
# read-only root filesystem stays possible. --with-deps pulls the shared
# libraries and fonts Chromium needs on debian-slim.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright

USER 1000:1000

# Port 6313 (BDD Playwright Service)
EXPOSE 6313

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.

CMD ["python", "-m", "src.asgi"]
