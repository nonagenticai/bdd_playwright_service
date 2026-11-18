"""``_acquire_concurrency_slot`` must answer "is there a member to release".

The bug: the ``ZADD`` that creates the slot and the housekeeping ``EXPIRE`` that
bounds the key sat under ONE ``except`` that returned ``False``. Those are two
round trips, so a connection reset BETWEEN them — a Redis restart/failover,
exactly the blip the handler exists for — reported "not registered" for a slot
that HAD been added. ``_process_job_bg`` gates its ``finally``
``_release_concurrency_slot`` on that return value, so the release was skipped
and the member sat in the per-tenant running-jobs zset until its own deadline
score expired it.

The admission path counts that member, so the tenant is short one concurrency
slot for the whole slot TTL per blip; a few of them zero out the default cap and
every ``POST /execute`` 429s with "Concurrent job limit exceeded" while nothing
is running. A leaked slot is the failure signature this fleet has already paid
for more than once.
"""

from __future__ import annotations

from typing import cast

import pytest

# ⚠️ redis-py's OWN ConnectionError, NOT the builtin of the same name. They
# render identically, which is how a double raising the builtin has previously
# "proved" a handler that never actually caught the library's exception.
from redis.exceptions import ConnectionError as RedisConnectionError

import src.services.background as background

pytestmark = pytest.mark.unit


class _FakeRedis:
    """Records zadd/zrem; ``expire`` raises the way a mid-sequence blip does."""

    def __init__(self, *, expire_raises: bool = False, zadd_raises: bool = False) -> None:
        self.expire_raises = expire_raises
        self.zadd_raises = zadd_raises
        self.members: dict[str, float] = {}
        self.zrem_calls: list[str] = []

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if self.zadd_raises:
            raise RedisConnectionError("connection reset by peer")
        self.members.update(mapping)
        return len(mapping)

    async def expire(self, key: str, ttl: int) -> bool:
        if self.expire_raises:
            raise RedisConnectionError("connection reset by peer")
        return True

    async def zrem(self, key: str, member: str) -> int:
        self.zrem_calls.append(member)
        self.members.pop(member, None)
        return 1


class _FakeRedisClient:
    def __init__(self, fake: _FakeRedis) -> None:
        self._fake = fake

    def _ensure_connected(self) -> _FakeRedis:
        return self._fake


@pytest.fixture
def wire_redis(monkeypatch: pytest.MonkeyPatch):
    def _wire(fake: _FakeRedis) -> _FakeRedis:
        monkeypatch.setattr(background, "get_redis_client", lambda: _FakeRedisClient(fake))
        return fake

    return _wire


class TestAcquireConcurrencySlot:
    async def test_redis_connection_error_is_the_library_type_not_the_builtin(self) -> None:
        """Guard the guard: a same-named builtin would make every case below vacuous."""
        assert not issubclass(RedisConnectionError, ConnectionError), (
            "redis.exceptions.ConnectionError must be the library type; if this "
            "import ever resolves to the builtin, the fakes stop exercising the "
            "handler's RedisError arm at all"
        )

    async def test_expire_failure_after_a_successful_zadd_still_reports_acquired(
        self, wire_redis
    ) -> None:
        fake = wire_redis(_FakeRedis(expire_raises=True))
        acquired = await background._acquire_concurrency_slot("tenant-a", "job-1")
        assert fake.members, "precondition: the ZADD landed, so a member exists"
        assert acquired is True, (
            "returning False here skips the finally-release and strands the member "
            "until its deadline score prunes it, costing the tenant a slot"
        )

    async def test_zadd_failure_reports_not_acquired(self, wire_redis) -> None:
        fake = wire_redis(_FakeRedis(zadd_raises=True))
        assert await background._acquire_concurrency_slot("tenant-a", "job-1") is False
        assert not fake.members

    async def test_healthy_path_acquires(self, wire_redis) -> None:
        fake = wire_redis(_FakeRedis())
        assert await background._acquire_concurrency_slot("tenant-a", "job-1") is True
        assert "job-1" in fake.members


class TestSlotIsReleasedAfterAnExpireBlip:
    """The end-to-end property: the slot the job took is the slot it gives back."""

    async def test_process_job_bg_releases_the_slot_despite_the_expire_blip(
        self, monkeypatch: pytest.MonkeyPatch, wire_redis
    ) -> None:
        import contextlib

        fake = wire_redis(_FakeRedis(expire_raises=True))

        @contextlib.asynccontextmanager
        async def _fake_session():  # type: ignore[no-untyped-def]
            yield object()

        monkeypatch.setattr(background, "get_db_session", _fake_session)
        monkeypatch.setattr(background, "bind_task_identity", lambda *a, **k: None)
        monkeypatch.setattr(background, "set_correlation_context", lambda *a, **k: None)

        class _Processor:
            async def process_job(self, *args: object, **kwargs: object) -> None:
                return None

        from src.services.job_processor import JobProcessor

        await background._process_job_bg(
            cast(JobProcessor, _Processor()),
            "job-1",
            "01950000-0000-7000-8000-000000000001",
        )

        assert fake.zrem_calls == ["job-1"], (
            "the finished job must give its concurrency slot back; leaving it "
            "costs the tenant that slot until the deadline score prunes it"
        )
        assert fake.members == {}
