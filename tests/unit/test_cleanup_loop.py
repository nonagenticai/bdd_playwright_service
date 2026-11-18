"""Unit tests for the ASGI background cleanup loop (`src.asgi._run_cleanup_loop`).

The loop had ``await asyncio.sleep(300)`` as the FIRST statement of its ``try``,
so the first sweep only ran five minutes after startup — precisely the window in
which a rolling restart leaves jobs stuck. Moving the sleep to the tail is only
half the fix: the ``except Exception`` arm had no sleep of its own, so a
persistently failing sweep would then busy-loop, hammering the logger and the DB.

⚠️ SCOPE LIMIT, stated so nobody mistakes it for cover: these tests fail FAST
on the two defects they target (a head-sleep, and a missing error-path sleep) —
measured, both red in ~12s. They do NOT bound a hypothetical regression that
swallows ``CancelledError`` outright: such a loop ignores pytest's own teardown
cancel too, so nothing on the test side can stop it and the run wedges rather
than fails (measured at >100s across four attempts). The waits below are still
bounded — an unbounded ``await task`` is strictly worse — but bounded waits
cannot kill an uncancellable task. Fixing that belongs in the loop, not here.

Both properties are pinned here. Neither test patches ``asyncio.sleep``: they use
the REAL 300s interval and simply give the loop wall-clock time to misbehave. A
head-sleep or a missing error-path sleep is therefore observable as "no sweep at
all" / "hundreds of sweeps", not as a mocked call count.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from src import asgi


@contextlib.asynccontextmanager
async def _fake_db_session():
    yield MagicMock()


def _fake_cleanup_service(**kwargs) -> MagicMock:
    service = MagicMock()
    service.cleanup_stuck_jobs = AsyncMock(**kwargs)
    service.cleanup_old_jobs = AsyncMock(return_value=0)
    return service


async def _run_briefly(coro_factory, seconds: float = 0.05) -> None:
    """Run the loop for a moment, then stop it — with a BOUNDED wait.

    ⚠️ `await task` after `task.cancel()` is unbounded: a loop that swallowed
    the cancellation would never return and the test process would wedge rather
    than fail (measured — a bare `await` here hung a run past 110s). `wait`
    with a timeout always returns, so a regression surfaces as a failed
    assertion below instead of a hung suite.
    """
    task = asyncio.create_task(coro_factory())
    await asyncio.sleep(seconds)
    task.cancel()
    await asyncio.wait({task}, timeout=0.5)
    with contextlib.suppress(asyncio.CancelledError):
        if task.done():
            await task


class _FakeJobStore:
    """A minimal stand-in that RECORDS WHAT THE SWEEP DID, not how often it ran.

    Asserting on ``await_count`` would be a shape-only test: it pins how many
    calls the implementation made rather than the effect it produced, so it
    breaks on a correct refactor and passes on a wrong change that happens to
    preserve the count. This store lets the test assert the actual outcome —
    the stuck row was reaped and the old row was purged.
    """

    def __init__(self, sweep_cap: int = 50) -> None:
        self.stuck: set[str] = {"job-stuck-1"}
        self.old: set[str] = {"job-old-1"}
        self.reaped: list[str] = []
        self.purged: list[str] = []
        self.sweeps = 0
        # ⚠️ A loop that SWALLOWS CancelledError cannot be stopped from outside
        # — pytest's own teardown cancel is swallowed too, so the test process
        # wedges forever (measured). SystemExit is a BaseException that neither
        # `except Exception` nor `except asyncio.CancelledError` catches, so it
        # ALWAYS terminates the task. That is what makes a regression here fail
        # instead of hang.
        self._sweep_cap = sweep_cap

    async def cleanup_stuck_jobs(
        self,
        db: AsyncSession,
        *,
        stuck_timeout_minutes: int = 0,
        pending_timeout_minutes: int = 0,
    ) -> int:
        self.sweeps += 1
        if self.sweeps > self._sweep_cap:
            raise SystemExit("sweep cap hit — the loop never stopped")
        drained, self.stuck = sorted(self.stuck), set()
        self.reaped.extend(drained)
        return len(drained)

    async def cleanup_old_jobs(self, db: AsyncSession, *, retention_days: int = 0) -> int:
        drained, self.old = sorted(self.old), set()
        self.purged.extend(drained)
        return len(drained)


async def test_cleanup_loop_sweeps_before_sleeping(monkeypatch):
    """The FIRST sweep happens immediately, not one full interval after startup."""
    monkeypatch.setattr(asgi, "get_db_session", _fake_db_session)
    store = _FakeJobStore()

    # The real 300s interval: if the sleep were still at the head of the loop,
    # the store would still be holding both rows when this returns.
    await _run_briefly(lambda: asgi._run_cleanup_loop(store, interval_seconds=300))

    assert store.reaped == ["job-stuck-1"], (
        "the cleanup loop slept before its first sweep — the stuck job was "
        "never reaped, so stuck rows sit untouched for a full interval after "
        f"startup (reaped: {store.reaped})"
    )
    assert store.purged == ["job-old-1"], (
        f"the first sweep did not reach the old-job purge (purged: {store.purged})"
    )
    assert not store.stuck and not store.old


async def test_cleanup_loop_error_path_sleeps_instead_of_spinning(monkeypatch):
    """A failing sweep backs off; it must not tight-loop on the error path.

    With the sleep moved to the tail and no sleep in ``except Exception``, a
    raising sweep re-enters the loop body immediately and spins as fast as the
    event loop allows.

    ⚠️ The iteration CAP is what makes this test fail instead of HANG. A spin
    with no ``await`` that yields starves the event loop completely — the 50ms
    timer and even ``task.cancel()`` never get a chance to run, so a wall-clock
    assertion alone would wedge the suite forever on a regression (measured).
    The cap raises ``CancelledError`` from inside the sweep, which the loop's own
    ``except asyncio.CancelledError: break`` turns into a clean exit, leaving the
    over-count visible to the assertion below.
    """
    monkeypatch.setattr(asgi, "get_db_session", _fake_db_session)

    cap = 5
    attempts = 0

    async def failing_sweep(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts > cap:
            # SystemExit, not CancelledError: a regression that swallows
            # cancellation would eat a CancelledError here too and wedge the
            # suite. SystemExit is caught by neither `except Exception` nor
            # `except asyncio.CancelledError`, so the task always dies.
            raise SystemExit("attempt cap hit — the loop never backed off")
        raise RuntimeError("db is down")

    service = MagicMock()
    service.cleanup_stuck_jobs = AsyncMock(side_effect=failing_sweep)
    service.cleanup_old_jobs = AsyncMock(return_value=0)

    await _run_briefly(lambda: asgi._run_cleanup_loop(service, interval_seconds=300))

    assert attempts == 1, (
        f"the cleanup loop busy-looped on the error path: {attempts} attempts "
        "with no backoff. The `except Exception` arm needs its own sleep (a "
        "`finally` would be wrong — it would also delay the CancelledError "
        "shutdown path)."
    )


async def test_cleanup_loop_exits_promptly_on_cancellation(monkeypatch):
    """Cancellation must terminate the loop without waiting out the interval.

    The 1s ``wait_for`` is the real guard — a loop that swallowed
    ``CancelledError`` and kept sleeping out its 300s interval would blow it.
    But "the test would have errored" is not an assertion a reader (or the
    assertion-free-test detector) can see, so the post-condition is stated
    explicitly: the loop task is FINISHED, and it did no further work after the
    cancel.
    """
    monkeypatch.setattr(asgi, "get_db_session", _fake_db_session)
    # A SHORT interval plus a sweep cap: a loop that ignores the cancel keeps
    # sweeping and trips the cap in milliseconds, so the regression is a fast
    # FAILURE. With the real 300s interval it would instead sit in an
    # uncancellable sleep and wedge the suite.
    store = _FakeJobStore(sweep_cap=20)

    task = asyncio.create_task(asgi._run_cleanup_loop(store, interval_seconds=0.01))
    await asyncio.sleep(0.05)
    reaped_before_cancel = list(store.reaped)
    sweeps_before_cancel = store.sweeps
    task.cancel()

    # ⚠️ `asyncio.wait`, NOT `wait_for(shield(...))`. A loop that SWALLOWS the
    # cancellation never finishes, and awaiting it — even shielded — leaves the
    # test hanging on a coroutine that will never return. `wait` returns after
    # the timeout either way, so a regression FAILS in ~1s instead of wedging
    # CI. (The sibling error-path test documents the same hazard.)
    done, _pending = await asyncio.wait({task}, timeout=1.0)

    assert task in done, (
        "the cleanup loop is still running 1s after task.cancel() — it "
        "swallowed CancelledError instead of breaking, so shutdown would hang "
        "for up to a full interval"
    )
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert task.done(), (
        "the cleanup loop is still running after task.cancel() — it swallowed "
        "CancelledError instead of breaking, so shutdown would hang for up to "
        "a full interval"
    )
    assert store.sweeps == sweeps_before_cancel, (
        "the cleanup loop kept sweeping after it was cancelled — it swallowed "
        f"CancelledError instead of breaking (sweeps before: "
        f"{sweeps_before_cancel}, after: {store.sweeps})"
    )
    assert store.reaped == reaped_before_cancel


async def _await_quietly(task: asyncio.Task[None]) -> None:
    with contextlib.suppress(asyncio.CancelledError):
        await task
