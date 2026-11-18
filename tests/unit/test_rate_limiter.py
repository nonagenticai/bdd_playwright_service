"""Tests for the atomic rate-limit middleware and the concurrency-slot lifecycle.

Covers the two defects fixed together (both ported from the equivalent fix in
``frontend_test_runner_service``):

- **D1 — the concurrent-job cap was structurally DEAD.** ``dispatch`` read the
  integer key ``bdd_playwright:concurrent:{tenant_id}`` and nothing in the
  service ever wrote it, so ``get()`` always returned ``None`` and the cap could
  never fire. The old ``test_rate_limit_under_limit_allows`` PINNED that: it
  mocked ``get -> None`` with the comment "no concurrent count", i.e. it asserted
  the broken state, and no test exercised the cap at all. It is replaced here by
  tests that put REAL members in the running-jobs set.
- **D2 — the sliding window was check-then-act** across an await
  (``ZREMRANGEBYSCORE``/``ZCARD`` … ``ZADD``), so N bursting requests at N-1/N
  all observed the same under-limit count and all admitted.

⚠️ These tests use ``fakeredis`` + ``lupa`` and let the REAL Lua script run
against REAL stored state. A fake that returned ``None``/``0`` for everything
would let the cap tests pass with no cap in the code at all — the exact
structural vacuity that let D1 ship. Every assertion below depends on data the
implementation actually wrote or read.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import httpx
import pytest
from fastapi import HTTPException
from starlette.responses import Response

from src.middleware.rate_limiter import (
    RateLimitMiddleware,
    concurrent_jobs_key,
    slot_ttl_seconds,
    window_key,
    window_member,
)

TENANT = "01950000-0000-7000-8000-00000000ffff"
OTHER_TENANT = "01950000-0000-7000-8000-00000000aaaa"

# A deadline comfortably in the future — a live running job's slot.
_FUTURE = time.time() + 3600.0


class _YieldingRedis:
    """fakeredis proxy where every command SUSPENDS the event loop first.

    ⚠️ Load-bearing for the race test, not decoration. ``fakeredis`` resolves a
    command without ever yielding, so ``asyncio.gather`` over N coroutines that
    only await fakeredis runs them one-at-a-time to completion and NO
    interleaving is possible — a check-then-act limiter would look atomic and
    ``test_burst_admits_exactly_n`` would pass against the very bug it exists to
    catch (verified: it did). A real Redis command is a network round-trip and
    always suspends; this models that, so the interleaving the bug needs can
    actually happen.
    """

    def __init__(self, fake: fakeredis.aioredis.FakeRedis) -> None:
        self._fake = fake

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._fake, name)

        async def _call(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return await attr(*args, **kwargs)

        return _call


class _StubRedisClient:
    """Stub for ``w2v_common.RedisClient`` delegating to a fakeredis instance."""

    def __init__(self, fake: fakeredis.aioredis.FakeRedis) -> None:
        self._fake = fake

    def _ensure_connected(self) -> Any:
        return _YieldingRedis(self._fake)

    async def eval_script(
        self,
        script: str,
        keys: list[str] | None = None,
        args: list[Any] | None = None,
    ) -> Any:
        # One suspension for the round-trip, then the WHOLE script executes
        # without interleaving — exactly Redis's EVAL guarantee.
        await asyncio.sleep(0)
        return await cast(
            Any,
            self._fake.eval(script, len(keys or []), *(keys or []), *(args or [])),
        )


class _SpyingStubRedisClient(_StubRedisClient):
    """``_StubRedisClient`` that RECORDS every admission-script evaluation.

    Load-bearing for the through-the-app tests, not decoration. Those assert on a
    status code the CLIENT sees, and a 401/500 produced by a layer OUTSIDE the
    limiter has a status code too. The recorded list is positive evidence that
    ``RateLimitMiddleware.dispatch`` actually reached Redis -- i.e. that the code
    under test executed at all. An empty list means something answered first, and
    the test says so instead of quietly agreeing.
    """

    def __init__(self, fake: fakeredis.aioredis.FakeRedis, calls: list[str]) -> None:
        super().__init__(fake)
        self._calls = calls

    async def eval_script(
        self,
        script: str,
        keys: list[str] | None = None,
        args: list[Any] | None = None,
    ) -> Any:
        self._calls.append(script)
        return await super().eval_script(script, keys=keys, args=args)


def _make_request(
    method: str = "POST",
    path: str = "/api/bdd/qa-suite",
    tenant_id: str = TENANT,
):
    """Create a mock Starlette Request."""
    request = MagicMock()
    request.method = method
    request.url = MagicMock()
    request.url.path = path
    request.state = MagicMock()
    request.state.tenant_id = tenant_id
    return request


def _make_settings(rpm: int = 30, concurrent: int = 5, stuck_minutes: int = 60):
    """Create a mock Settings object with rate_limit + cleanup config."""
    settings = MagicMock()
    settings.rate_limit.requests_per_minute = rpm
    settings.rate_limit.concurrent_jobs_per_tenant = concurrent
    settings.cleanup.stuck_job_timeout_minutes = stuck_minutes
    return settings


@pytest.fixture
def middleware():
    """Create a RateLimitMiddleware instance with a dummy app."""
    app = MagicMock()
    return RateLimitMiddleware(app)


@pytest.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    fake = fakeredis.aioredis.FakeRedis()
    yield fake
    await fake.aclose()


async def _dispatch(middleware, request) -> int:
    """Run the middleware and reduce the outcome to a status code."""
    call_next = AsyncMock(return_value=Response(status_code=200))
    try:
        result = await middleware.dispatch(request, call_next)
    except HTTPException as exc:
        return exc.status_code
    return result.status_code


class TestPassThrough:
    """Requests outside the limited surface are never touched."""

    @patch("src.middleware.rate_limiter.get_settings")
    async def test_non_post_requests_pass_through(self, mock_get_settings, middleware):
        """GET requests not rate limited."""
        request = _make_request(method="GET")
        expected_response = Response(status_code=200)
        call_next = AsyncMock(return_value=expected_response)

        result = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert result == expected_response

    @patch("src.middleware.rate_limiter.get_settings")
    async def test_non_bdd_endpoints_pass_through(self, mock_get_settings, middleware):
        """POST to other paths not limited."""
        request = _make_request(path="/api/health")
        expected_response = Response(status_code=200)
        call_next = AsyncMock(return_value=expected_response)

        result = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert result == expected_response


class TestConcurrentCap:
    """D1: the cap must actually COUNT running jobs, not read a dead key."""

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_sixth_job_rejected_while_five_in_flight(
        self, mock_get_settings, mock_get_redis, middleware, fake_redis
    ):
        """THE headline case: cap=5, five live slots held, the 6th POST 429s.

        Under the dead-counter implementation this returned 200 — nothing ever
        wrote the key it read.
        """
        mock_get_settings.return_value = _make_settings(rpm=1000, concurrent=5)
        mock_get_redis.return_value = _StubRedisClient(fake_redis)

        await fake_redis.zadd(concurrent_jobs_key(TENANT), {f"job-{i}": _FUTURE for i in range(5)})

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(_make_request(), AsyncMock())

        assert exc_info.value.status_code == 429
        assert "Concurrent job limit exceeded: 5 jobs" in exc_info.value.detail

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_fifth_job_admitted_while_four_in_flight(
        self, mock_get_settings, mock_get_redis, middleware, fake_redis
    ):
        """The cap is 5, not 4: with four live slots the next job is admitted."""
        mock_get_settings.return_value = _make_settings(rpm=1000, concurrent=5)
        mock_get_redis.return_value = _StubRedisClient(fake_redis)

        await fake_redis.zadd(concurrent_jobs_key(TENANT), {f"job-{i}": _FUTURE for i in range(4)})

        assert await _dispatch(middleware, _make_request()) == 200

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_expired_slots_are_pruned_and_do_not_block(
        self, mock_get_settings, mock_get_redis, middleware, fake_redis
    ):
        """THE TTL BACKSTOP: five slots whose owners died without releasing sit in
        the set with PAST deadlines. Admission prunes them and admits — a lost
        release must self-heal, never lock a tenant out permanently."""
        mock_get_settings.return_value = _make_settings(rpm=1000, concurrent=5)
        mock_get_redis.return_value = _StubRedisClient(fake_redis)

        past = time.time() - 1.0
        await fake_redis.zadd(concurrent_jobs_key(TENANT), {f"dead-{i}": past for i in range(5)})

        assert await _dispatch(middleware, _make_request()) == 200
        assert await fake_redis.zcard(concurrent_jobs_key(TENANT)) == 0

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_live_counted_expired_pruned(
        self, mock_get_settings, mock_get_redis, middleware, fake_redis
    ):
        """5 live + 9 expired → effective count 5 → 429. Expired members must not
        pad the count, live members must not be pruned."""
        mock_get_settings.return_value = _make_settings(rpm=1000, concurrent=5)
        mock_get_redis.return_value = _StubRedisClient(fake_redis)

        past = time.time() - 1.0
        await fake_redis.zadd(
            concurrent_jobs_key(TENANT),
            {
                **{f"live-{i}": _FUTURE for i in range(5)},
                **{f"dead-{i}": past for i in range(9)},
            },
        )

        with pytest.raises(HTTPException) as exc_info:
            await middleware.dispatch(_make_request(), AsyncMock())
        assert exc_info.value.status_code == 429
        assert await fake_redis.zcard(concurrent_jobs_key(TENANT)) == 5

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_another_tenants_jobs_do_not_count(
        self, mock_get_settings, mock_get_redis, middleware, fake_redis
    ):
        """The cap is PER TENANT."""
        mock_get_settings.return_value = _make_settings(rpm=1000, concurrent=5)
        mock_get_redis.return_value = _StubRedisClient(fake_redis)

        await fake_redis.zadd(
            concurrent_jobs_key(OTHER_TENANT), {f"job-{i}": _FUTURE for i in range(9)}
        )

        assert await _dispatch(middleware, _make_request()) == 200


class TestAtomicSlidingWindow:
    """D2: assert exactly N requests are admitted under a concurrent burst."""

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_burst_admits_exactly_n(
        self, mock_get_settings, mock_get_redis, middleware, fake_redis
    ):
        """100 concurrent POSTs against rpm=10 admit exactly 10.

        The check-then-act version admitted far more: every coroutine read the
        same ZCARD before any of them reached its ZADD.
        """
        rpm = 10
        mock_get_settings.return_value = _make_settings(rpm=rpm, concurrent=1000)
        mock_get_redis.return_value = _StubRedisClient(fake_redis)

        results = await asyncio.gather(
            *[_dispatch(middleware, _make_request()) for _ in range(100)]
        )

        admitted = sum(1 for s in results if s == 200)
        rejected = sum(1 for s in results if s == 429)
        assert admitted + rejected == 100
        assert admitted == rpm, (
            f"Atomic sliding-window must admit exactly {rpm} of 100 bursting "
            f"requests; got admitted={admitted}, rejected={rejected}"
        )
        assert await fake_redis.zcard(window_key(TENANT)) == rpm

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_separate_tenants_have_independent_windows(
        self, mock_get_settings, mock_get_redis, middleware, fake_redis
    ):
        rpm = 5
        mock_get_settings.return_value = _make_settings(rpm=rpm, concurrent=1000)
        mock_get_redis.return_value = _StubRedisClient(fake_redis)

        a = await asyncio.gather(*[_dispatch(middleware, _make_request()) for _ in range(10)])
        b = await asyncio.gather(
            *[_dispatch(middleware, _make_request(tenant_id=OTHER_TENANT)) for _ in range(10)]
        )

        assert sum(1 for s in a if s == 200) == rpm
        assert sum(1 for s in b if s == 200) == rpm

    def test_same_tick_members_are_distinct(self):
        """The retired implementation used ``str(now)`` as the ZSET member, so two
        requests on the same clock tick collapsed to ONE member and the window
        undercounted. Members must be unique even for an identical timestamp."""
        now = 1234.5
        assert window_member(now) != window_member(now)


class TestFailOpen:
    """A rate-limit backend blip must never 500 a legitimate request."""

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_redis_unavailable_allows_request(
        self, mock_get_settings, mock_get_redis, middleware
    ):
        mock_get_settings.return_value = _make_settings(rpm=10, concurrent=5)
        mock_get_redis.side_effect = ConnectionError("Redis is down")

        request = _make_request()
        expected_response = Response(status_code=200)
        call_next = AsyncMock(return_value=expected_response)

        result = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert result == expected_response

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_redis_error_on_eval_fails_open(
        self, mock_get_settings, mock_get_redis, middleware
    ):
        """``redis.exceptions.RedisError`` does NOT subclass OSError/RuntimeError."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        mock_get_settings.return_value = _make_settings(rpm=10, concurrent=5)
        mock_redis = MagicMock()
        mock_redis.eval_script = AsyncMock(side_effect=RedisConnectionError("boom"))
        mock_get_redis.return_value = mock_redis

        request = _make_request()
        expected_response = Response(status_code=200)
        call_next = AsyncMock(return_value=expected_response)

        result = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert result == expected_response


class TestSlotTtl:
    """The backstop deadline is grounded and cannot be zeroed away."""

    @patch("src.middleware.rate_limiter.get_settings")
    def test_ttl_tracks_stuck_job_timeout(self, mock_get_settings):
        mock_get_settings.return_value = _make_settings(stuck_minutes=60)
        assert slot_ttl_seconds() == 3600

    @patch("src.middleware.rate_limiter.get_settings")
    def test_ttl_floored(self, mock_get_settings):
        """A misconfigured 0 must not expire every slot instantly and silently
        re-open the unbounded-spend hole."""
        mock_get_settings.return_value = _make_settings(stuck_minutes=0)
        assert slot_ttl_seconds() == 60


class TestSlotLifecycle:
    """``_process_job_bg`` must ZADD its slot before the ``try`` and ZREM it in the
    ``finally`` — on EVERY exit path."""

    @staticmethod
    def _patches(fake_redis):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _noop_db_session():
            yield object()

        from src.services import background

        return (
            patch.object(background, "get_redis_client", lambda: _StubRedisClient(fake_redis)),
            patch.object(background, "get_db_session", _noop_db_session),
            patch.object(background, "slot_ttl_seconds", lambda: 3600),
            patch.object(background.JobManager, "mark_failed", new=AsyncMock()),
        )

    async def test_completed_job_releases_its_slot(self, fake_redis):
        """A finished job frees its slot, so the next one is admitted."""
        from src.services import background

        processor = MagicMock()
        processor.process_job = AsyncMock(return_value=None)

        p = self._patches(fake_redis)
        with p[0], p[1], p[2]:
            # Prove the slot EXISTS while the job runs, not merely that it is
            # absent afterwards (which an implementation that never wrote it
            # would also satisfy).
            held: list[int] = []

            async def _observe(*_a, **_k):
                held.append(await fake_redis.zcard(concurrent_jobs_key(TENANT)))

            processor.process_job = AsyncMock(side_effect=_observe)
            await background._process_job_bg(processor, "job-1", TENANT)

        assert held == [1], f"slot must be held DURING the job, saw zcard={held}"
        assert await fake_redis.zcard(concurrent_jobs_key(TENANT)) == 0

    async def test_failing_job_releases_its_slot(self, fake_redis):
        """A job that raises still releases — else the cap is a one-way ratchet."""
        from src.services import background

        processor = MagicMock()
        processor.process_job = AsyncMock(side_effect=RuntimeError("boom"))

        p = self._patches(fake_redis)
        with p[0], p[1], p[2], p[3]:
            await background._process_job_bg(processor, "job-2", TENANT)

        assert await fake_redis.zcard(concurrent_jobs_key(TENANT)) == 0

    async def test_cancelled_job_releases_its_slot(self, fake_redis):
        """Cancellation re-raises out of ``_process_job_bg``; the slot still frees."""
        from src.services import background

        processor = MagicMock()
        processor.process_job = AsyncMock(side_effect=asyncio.CancelledError())

        p = self._patches(fake_redis)
        with p[0], p[1], p[2], p[3], pytest.raises(asyncio.CancelledError):
            await background._process_job_bg(processor, "job-3", TENANT)

        assert await fake_redis.zcard(concurrent_jobs_key(TENANT)) == 0

    async def test_slot_deadline_is_in_the_future_and_bounded(self, fake_redis):
        """The member's SCORE is a future expiry deadline, so a lost release
        self-heals at that deadline rather than leaking forever."""
        from src.services import background

        processor = MagicMock()
        scores: list[float] = []

        async def _observe(*_a, **_k):
            scores.extend(
                s
                for _, s in await fake_redis.zrange(
                    concurrent_jobs_key(TENANT), 0, -1, withscores=True
                )
            )

        processor.process_job = AsyncMock(side_effect=_observe)

        p = self._patches(fake_redis)
        with p[0], p[1], p[2]:
            await background._process_job_bg(processor, "job-4", TENANT)

        assert len(scores) == 1
        assert time.time() < scores[0] <= time.time() + 3600

    async def test_a_leaked_slot_does_not_block_forever(self, fake_redis):
        """END TO END: a job whose task is hard-killed (no finally) leaks a member,
        but the member's deadline expires it, so a later admission still passes."""
        await fake_redis.zadd(concurrent_jobs_key(TENANT), {"killed": time.time() - 1})
        app = MagicMock()
        mw = RateLimitMiddleware(app)
        with (
            patch(
                "src.middleware.rate_limiter.get_settings",
                return_value=_make_settings(rpm=1000, concurrent=1),
            ),
            patch(
                "src.middleware.rate_limiter.get_redis_client",
                return_value=_StubRedisClient(fake_redis),
            ),
        ):
            assert await _dispatch(mw, _make_request()) == 200


class TestRefusalReachesTheClientAs429:
    """The cap's 429 must survive the trip OUT through the assembled ASGI stack.

    ⚠️ **Every other test in this file is blind to what the CLIENT sees.** They
    call ``middleware.dispatch(...)`` directly and inspect the ``HTTPException``
    object, a scope in which "status_code == 429" is true of an exception that
    might still reach the wire as a 500. The distinction is not academic:
    Starlette assembles the app as

        ServerErrorMiddleware -> <every app.add_middleware layer> ->
        ExceptionMiddleware -> router

    (``starlette.applications.build_middleware_stack``), so ``ExceptionMiddleware``
    -- the layer that renders an ``HTTPException`` -- sits INSIDE this middleware.
    An ``HTTPException`` raised in ``dispatch`` travels OUTWARD past it, and on a
    bare FastAPI app it becomes a **500** (measured).

    What saves it is one specific, non-obvious dependency:
    ``register_error_envelope_handlers`` (``src/asgi.py``) registers a handler for
    bare ``Exception``, and ``build_middleware_stack`` routes any handler keyed on
    ``Exception``/``500`` to ``ServerErrorMiddleware`` -- the OUTERMOST layer. That
    handler (``_unhandled_exception_handler``) calls ``_find_http_exception`` and
    renders an ``HTTPException`` as the HTTP error it is. So the 429 is correct
    **because of a w2v_common catch-all**, not because raising from middleware is
    safe in general. This class pins that end-to-end property against the REAL
    ``src.asgi.app``, so deleting that call reds here.

    ⛔ **Why this class builds its own auth precondition (regression, PR #151).**
    The first version of these tests asserted that ``request.state.tenant_id``
    would be ``"anonymous"`` "because the suite runs with ``AUTH_ENABLED=false``".
    That is not a property of the code -- it is an accident of two things this
    class does not control:

    * ``conftest.mock_env_vars`` sets ``AUTH_ENABLED=false`` in an **autouse
      fixture**, i.e. per test, at CALL time; but
    * ``src/asgi.py`` decides ``if settings.auth.enabled: app.add_middleware(
      AuthMiddleware)`` **once, at import time**.

    So the stack shape is fixed by whatever the ambient environment said when
    ``src.asgi`` was FIRST imported. Import it from inside a test (as this file
    does) and the fixture has already forced auth off. Let ``tests/test_openapi.py``
    import it at module scope during COLLECTION -- which is what happens in a full
    run -- and CI's ambient ``AUTH_ENABLED=true`` wins instead, ``AuthMiddleware``
    is registered OUTSIDE the limiter, and the request is rejected 401 before
    ``RateLimitMiddleware.dispatch`` ever runs. Green locally, red in CI, and the
    code under test never executed in either case.

    The fix is to stop inheriting the precondition. These tests now:

    1. authenticate the request the way ``tests/test_auth_middleware.py`` does
       (stub ``w2v_auth._introspect_license``, clear the license cache, send
       ``X-Tenant-Id: Bearer <token>``), so an auth layer -- if the ambient
       environment put one there -- ADMITS rather than rejects; and
    2. seed the running-jobs slot under BOTH tenant keys the limiter could
       legitimately read: the authenticated ``TENANT``, and the ``"anonymous"``
       fallback used when no auth layer bound one. Whichever stack shape this
       process ended up with, the cap has a live slot to trip on.

    Neither step lets the test pass for the wrong reason -- see
    ``_assert_refused_by_the_limiter``, which fails loudly on a 401 and requires
    positive evidence that the admission script actually ran.

    Verified under BOTH stack shapes:
      pytest tests/unit/test_rate_limiter.py                          (no auth layer)
      AUTH_ENABLED=true pytest tests/test_openapi.py tests/unit/test_rate_limiter.py
                                                                      (auth layer)
    """

    #: The token the stubbed introspection accepts. Never reaches a real service.
    LICENSE_TOKEN = "rate-limiter-test-token"  # pragma: allowlist secret
    #: The limiter's fallback when nothing bound ``request.state.tenant_id``.
    ANON = "anonymous"

    @pytest.fixture(autouse=True)
    def _auth_admits(self, monkeypatch):
        """Make any ``AuthMiddleware`` in the stack ADMIT this request.

        Mirrors ``tests/test_auth_middleware.py``. The license cache must be
        cleared too, or a cached verdict from an earlier test decides this one.
        If ``AuthMiddleware`` is not in the stack, none of this is consulted and
        the request is anonymous -- both shapes are handled by the seeding below.
        """
        import w2v_common.auth.middleware as w2v_auth

        w2v_auth._get_license_cache().clear()
        self._introspections: list[str] = []

        async def _fake_introspect(token: str) -> dict[str, Any]:
            self._introspections.append(token)
            return {
                "active": True,
                "tenant_id": TENANT,
                "subject_id": "subj-rate-limit",
                "license_id": "lic-rate-limit",
                "roles": ["user"],
            }

        monkeypatch.setattr(w2v_auth, "_introspect_license", _fake_introspect)
        yield
        w2v_auth._get_license_cache().clear()

    @pytest.fixture
    def real_app(self):
        """The service's own ASGI app -- middleware stack and handlers included.

        Imported without running the lifespan (``ASGITransport`` does not start
        one), so no database or Redis is touched: the limiter refuses before
        ``call_next`` and the route body never executes.
        """
        from src.asgi import app

        return app

    @staticmethod
    def _auth_layer_present(app) -> bool:
        from w2v_common.auth.middleware import AuthMiddleware

        return any(m.cls is AuthMiddleware for m in app.user_middleware)

    async def _post(self, app) -> httpx.Response:
        # ``raise_app_exceptions=False`` makes the transport behave like a real
        # client: an exception that escapes the app becomes the status the wire
        # would carry, instead of being re-raised into the test. Without it a
        # regression reds as "HTTPException propagated" -- true, but it hides the
        # status code, and the status code is the entire subject here.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/api/bdd/qa-suite",
                json={},
                headers={"X-Tenant-Id": f"Bearer {self.LICENSE_TOKEN}"},
            )

    async def _seed_running_slot(self, fake_redis) -> None:
        """Hold one concurrency slot under EVERY tenant key the limiter may read."""
        for tenant in (TENANT, self.ANON):
            await fake_redis.zadd(concurrent_jobs_key(tenant), {"job-0": _FUTURE})

    async def _seed_full_window(self, fake_redis) -> None:
        """Fill the 1-request sliding window under EVERY tenant key."""
        now = time.time()
        for tenant in (TENANT, self.ANON):
            await fake_redis.zadd(window_key(tenant), {window_member(now): now})

    def _diagnose(self, response: httpx.Response, admissions: list[str]) -> str:
        """Explain a non-429 in the terms of the failure mode that produced it.

        A message BUILDER, deliberately not an assertion: the status code is
        asserted in the test body, but a bare ``assert 401 == 429`` tells the
        next reader nothing, and this branch has already spent one CI cycle on
        exactly that. Each branch names a distinct way the request can fail to
        reach the code under test.
        """
        body = response.text[:400]
        if response.status_code == 401:
            return (
                "AuthMiddleware ran BEFORE RateLimitMiddleware and rejected the request, "
                "so the code under test never executed. The auth stub in `_auth_admits` "
                f"did not take effect. Body: {body}"
            )
        if not admissions:
            return (
                "the rate limiter never reached Redis -- something answered before "
                f"RateLimitMiddleware.dispatch ran (status {response.status_code}): {body}"
            )
        return (
            f"the cap refused with {response.status_code}; a 5xx here inverts every "
            f"caller's retry-on-429 / alarm-on-5xx split. Body: {body}"
        )

    def _assert_refused_by_the_limiter(
        self,
        response: httpx.Response,
        app,
        expected_message: str,
        admissions: list[str],
    ) -> None:
        """A 429 is only the right answer if THE LIMITER is the one that gave it.

        The status code itself is asserted in each TEST BODY -- it is what these
        tests are named for, and a reader (and the fleet's vacuous-test gate,
        which scans function bodies) should find it where the name is. This
        helper carries the ATTRIBUTION checks, each closing a way a 429 could be
        the right number for the wrong reason:

        1. **The admission script ran.** If any earlier layer short-circuits, the
           limiter never reaches Redis and this list stays empty -- so a 429
           invented by some other component cannot satisfy the test.
        2. **Auth actually ADMITTED**, rather than being absent. Only asserted
           when an auth layer is really in the stack, since that is the half of
           the precondition this test cannot control.
        3. **The message is the limiter's own.** No other layer in this app emits
           these strings, so the refusal is attributed, not merely counted.

        The 401 case -- an auth layer answering before the limiter, the exact
        regression that reddened CI -- is named by :meth:`_diagnose`, which
        supplies the message for the body's status assertion.
        """
        assert admissions, (
            "the rate limiter never reached Redis -- something answered before "
            f"RateLimitMiddleware.dispatch ran (status {response.status_code}): "
            f"{response.text[:400]}"
        )
        if self._auth_layer_present(app):
            assert self._introspections == [self.LICENSE_TOKEN], (
                "an auth layer IS registered but never introspected our token; the "
                "request did not travel the authenticated path we think it did"
            )
        body = response.json()
        assert body["error"]["status"] == 429
        assert expected_message in body["error"]["message"]

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_concurrent_cap_reaches_the_client_as_429(
        self, mock_get_settings, mock_get_redis, real_app, fake_redis
    ):
        """A caller at the concurrency cap gets 429 -- the status callers retry on."""
        mock_get_settings.return_value = _make_settings(rpm=1000, concurrent=1)
        admissions: list[str] = []
        mock_get_redis.return_value = _SpyingStubRedisClient(fake_redis, admissions)
        await self._seed_running_slot(fake_redis)

        response = await self._post(real_app)

        # The status code IS the subject of this test, so it is asserted HERE, in
        # the body the test's name is attached to -- not only inside a helper.
        assert response.status_code == 429, self._diagnose(response, admissions)
        self._assert_refused_by_the_limiter(
            response, real_app, "Concurrent job limit exceeded: 1 jobs", admissions
        )

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_sliding_window_reaches_the_client_as_429(
        self, mock_get_settings, mock_get_redis, real_app, fake_redis
    ):
        """The window branch refuses the same way -- both raise sites matter."""
        mock_get_settings.return_value = _make_settings(rpm=1, concurrent=1000)
        admissions: list[str] = []
        mock_get_redis.return_value = _SpyingStubRedisClient(fake_redis, admissions)
        await self._seed_full_window(fake_redis)

        response = await self._post(real_app)

        assert response.status_code == 429, self._diagnose(response, admissions)
        self._assert_refused_by_the_limiter(
            response, real_app, "Rate limit exceeded: 1 requests per minute", admissions
        )

    @patch("src.middleware.rate_limiter.get_redis_client")
    @patch("src.middleware.rate_limiter.get_settings")
    async def test_admitted_request_is_not_refused_by_the_limiter(
        self, mock_get_settings, mock_get_redis, real_app, fake_redis
    ):
        """Control: under both limits the limiter ADMITS.

        Without this, both tests above would still pass if the limiter refused
        unconditionally -- the 429 has to be the cap firing, not a constant.
        """
        mock_get_settings.return_value = _make_settings(rpm=1000, concurrent=5)
        admissions: list[str] = []
        mock_get_redis.return_value = _SpyingStubRedisClient(fake_redis, admissions)

        response = await self._post(real_app)

        assert admissions, "the limiter never ran, so 'admitted' proves nothing"
        assert response.status_code != 429, (
            f"the limiter refused an under-limit request: {response.text[:400]}"
        )
        # Deliberately NOT an assertion about the final status code. Once the
        # limiter admits, the request continues to the route, and what happens
        # there is shape-dependent and none of this class's business: with no
        # auth layer the route itself rejects an anonymous caller (401 -- see
        # tests/unit/test_tenant_required.py), and with one it proceeds toward a
        # database this test deliberately never starts. Pinning a downstream
        # status here would re-import exactly the environment dependence this
        # class exists to eliminate. What the control must show is only that the
        # limiter ran and did NOT refuse -- so the 429s above are the cap firing
        # rather than a constant.
        assert "limit exceeded" not in response.text, (
            f"an under-limit request still carries a limiter refusal: {response.text[:400]}"
        )
