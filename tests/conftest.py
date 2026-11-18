"""Shared test fixtures for bdd_playwright_service."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

# Capture the test DB URL BEFORE the autouse mock_env_vars fixture monkeypatches
# DATABASE_URL to its in-test default. CI sets DATABASE_URL to its
# bdd_playwright_service_ci postgres; locally falls back to the default below
# (developer must run a local postgres to execute db_engine tests).
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://test:test@localhost:5432/test",  # pragma: allowlist secret
)
# w2v_common >= 0.15.0 dropped the localhost defaults on RedisSettings.url and
# StorageSettings.minio_endpoint; seed module-level so Settings() instantiation
# during pytest collection doesn't fail. The autouse mock_env_vars fixture
# re-applies them after its env-clearing sweep.
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_MINIO_ENDPOINT", "localhost:9000")
_TEST_DATABASE_URL = os.environ["DATABASE_URL"]


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Remove CI env vars that interfere with config-default tests, then set test defaults."""
    for key in list(os.environ):
        if any(
            key.startswith(p)
            for p in (
                "DATABASE_",
                "AUTH_",
                "REDIS_",
                "API_",
                "STORAGE_",
                "LOG_",
                "CACHE_",
                "TRACING_",
                "LLM_",
                "FFMPEG_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://test:test@localhost:5432/test",  # pragma: allowlist secret
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("STORAGE_LOCAL_PATH", "/tmp/test_bdd_playwright")
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("ENVIRONMENT", "testing")
    # LLM access flows through worker_management (WMClient); no per-service LLM env needed.


@pytest.fixture
def mock_db_session():
    """Mock async database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    client.exists = AsyncMock(return_value=False)
    client.ping = AsyncMock(return_value=True)
    return client


@pytest.fixture
def sample_inspection_result():
    """Sample normalized inspection result for BDD generation tests."""
    return {
        "pages": [
            {
                "page_id": "page_0",
                "url": "https://example.com/login",
                "title": "Login",
                "page_type": "login",
                "text_content": "Welcome Sign In Username Password",
                "interactive_elements": [
                    {
                        "tag": "input",
                        "type": "text",
                        "name": "username",
                        "label": "Username",
                        "selector": "#username",
                    },
                    {
                        "tag": "input",
                        "type": "password",
                        "name": "password",
                        "label": "Password",
                        "selector": "#password",
                    },
                    {
                        "tag": "button",
                        "type": "submit",
                        "text": "Sign In",
                        "selector": "button[type='submit']",
                    },
                ],
                "headings": ["Welcome"],
                "has_forms": True,
            },
            {
                "page_id": "page_1",
                "url": "https://example.com/dashboard",
                "title": "Dashboard",
                "page_type": "content",
                "text_content": "Home Dashboard Recent Items Settings",
                "interactive_elements": [
                    {
                        "tag": "a",
                        "text": "Settings",
                        "href": "/settings",
                        "selector": "a[href='/settings']",
                    },
                ],
                "headings": ["Home", "Dashboard"],
                "has_forms": False,
            },
        ],
        "navigation_graph": [
            {
                "from_page": "page_0",
                "to_page": "page_1",
                "action": "submit",
                "element_label": "Sign In",
            }
        ],
        "platform": "web",
        "base_url": "https://example.com",
    }


@pytest.fixture
def sample_gherkin_feature():
    """Sample Gherkin feature file content."""
    return """Feature: User Login
  As a user
  I want to authenticate with my credentials
  So that I can access my dashboard

  Scenario: Successful login
    Given I am on the login page
    When I enter "testuser" in the username field
    And I enter "password123" in the password field
    And I click the "Sign In" button
    Then I should be redirected to the dashboard
    And I should see "Welcome" on the page

  Scenario: Failed login with invalid credentials
    Given I am on the login page
    When I enter "invalid" in the username field
    And I enter "wrong" in the password field
    And I click the "Sign In" button
    Then I should see an error message
"""


# ---------------------------------------------------------------------------
# Real database fixtures (postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine():
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to _TEST_DATABASE_URL captured at module load (CI postgres in
    CI, localhost:5432/test locally). Runs Base.metadata.create_all
    (idempotent) then TRUNCATEs every model table so each test starts from
    a clean slate.
    """
    from src.database.models import Base

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine):
    """Real async_sessionmaker for bdd_playwright_service tests."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(db_session_factory):
    """Real AsyncSession for a single test."""
    async with db_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Real Redis fixtures (fakeredis)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_redis_client():
    """RedisClient backed by fakeredis (real commands, in-memory)."""
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client


# ---------------------------------------------------------------------------
# Tenant identity constants
# ---------------------------------------------------------------------------
# w2v_common >= 0.67.0 parses every tenant id as a ``uuid.UUID`` and RAISES
# (InvalidTenantIdError) on anything else, so a slug fixture like ``"tenant-1"``
# now dies at auth with a 401 before the assertion under test ever runs. These
# are real, distinct uuids.
TENANT_A = "11111111-1111-4111-8111-111111111111"
TENANT_B = "22222222-2222-4222-8222-222222222222"
