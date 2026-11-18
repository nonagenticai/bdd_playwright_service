"""Shutdown must drain in-flight job tasks BEFORE it disposes the DB engine.

``_process_job_bg`` already catches ``asyncio.CancelledError`` and calls
``JobManager.mark_failed`` — which matches ``status IN ('pending','processing')`` —
so the rescue that turns a stranded row terminal exists. It just could never run on
the real SIGTERM path: the lifespan cancelled ONLY ``cleanup_task``, never touched
``_background_tasks``, and went straight to ``_shutdown_database()``. The job tasks
were then cancelled by the event loop at interpreter teardown, and their rescue
opened a NEW ``get_db_session()`` against an engine that was already disposed, so it
raised and the row stayed ``pending`` — the one state nothing reclaimed.

These tests pin the ORDERING as an observable effect, not as a call sequence: the
stand-in task records whether the engine was still live at the moment its
cancellation handler ran. A drain that happens after disposal (or not at all) is
visible as ``engine_alive is False`` / a task that never got cancelled.
"""

from __future__ import annotations

import asyncio

import pytest

from src import asgi
from src.services import background


@pytest.fixture(autouse=True)
def _clean_task_registry():
    """Never leak a stand-in task into another test's ``_background_tasks``."""
    background._background_tasks.clear()
    yield
    for task in list(background._background_tasks):
        task.cancel()
    background._background_tasks.clear()


class _EngineState:
    """Stands in for the DB engine's liveness, flipped by the shutdown step."""

    def __init__(self) -> None:
        self.alive = True
        self.rescue_saw_engine_alive: bool | None = None


def _register_job_task(state: _EngineState) -> asyncio.Task[None]:
    """A stand-in for ``_process_job_bg``: long-running, with a DB-touching rescue."""

    async def _job() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # This mirrors `_process_job_bg`'s rescue: it needs a LIVE engine to open
            # a session and run mark_failed. Recording liveness here is what makes the
            # ordering observable instead of merely asserted.
            state.rescue_saw_engine_alive = state.alive
            raise

    task = asyncio.create_task(_job())
    background._background_tasks.add(task)
    task.add_done_callback(background._background_tasks.discard)
    return task


def _stub_shutdown_steps(monkeypatch, state: _EngineState) -> None:
    async def _noop() -> None:
        return None

    async def _dispose() -> None:
        state.alive = False

    monkeypatch.setattr(asgi, "_shutdown_wm_client", _noop)
    monkeypatch.setattr(asgi, "_shutdown_redis", _noop)
    monkeypatch.setattr(asgi, "_shutdown_database", _dispose)


async def _idle_cleanup_task() -> asyncio.Task[None]:
    async def _idle() -> None:
        await asyncio.sleep(3600)

    return asyncio.create_task(_idle())


async def test_shutdown_drains_job_tasks_while_the_engine_is_still_live(monkeypatch):
    """The cancellation rescue must observe a LIVE engine, i.e. drain precedes dispose."""
    state = _EngineState()
    _stub_shutdown_steps(monkeypatch, state)
    monkeypatch.setattr(background, "DRAIN_SETTLE_SECONDS", 0.05)
    monkeypatch.setattr(background, "DRAIN_CANCEL_SECONDS", 1.0)

    job_task = _register_job_task(state)
    cleanup_task = await _idle_cleanup_task()

    await asyncio.wait_for(asgi._shutdown(cleanup_task), timeout=5.0)

    assert job_task.done(), (
        "shutdown returned with an in-flight job task still pending — "
        "`_background_tasks` is never awaited or cancelled, so the task is reaped at "
        "interpreter teardown, long after the engine is gone, and its row stays "
        "`pending` forever"
    )
    assert state.rescue_saw_engine_alive is True, (
        "the job task's cancellation rescue ran AFTER the DB engine was disposed "
        f"(engine alive at rescue: {state.rescue_saw_engine_alive}) — its "
        "`mark_failed` would raise on a closed engine and the row would stay `pending`"
    )
    assert state.alive is False, "the engine was never disposed at all"


async def test_drain_lets_a_nearly_finished_task_commit_its_own_result(monkeypatch):
    """A task about to finish is SETTLED, not cancelled into a spurious failure."""
    finished: list[str] = []

    async def _job() -> None:
        try:
            await asyncio.sleep(0.02)
            finished.append("committed")
        except asyncio.CancelledError:
            finished.append("cancelled")
            raise

    task = asyncio.create_task(_job())
    background._background_tasks.add(task)
    task.add_done_callback(background._background_tasks.discard)

    await background.drain_background_tasks(settle_seconds=1.0, cancel_seconds=1.0)

    assert finished == ["committed"], (
        f"the drain cancelled a task that was about to finish (observed: {finished})"
    )


async def test_drain_is_a_noop_with_no_in_flight_tasks():
    """Shutdown on an idle pod must not wait out the settle window."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    await background.drain_background_tasks(settle_seconds=5.0, cancel_seconds=5.0)
    assert loop.time() - started < 1.0


async def test_drain_does_not_wedge_on_a_task_that_swallows_cancellation():
    """A shutdown path must be BOUNDED even against a misbehaving task."""

    swallowed = 0

    async def _stubborn() -> None:
        # Swallows cancellation, but only until a WALL-CLOCK deadline. A task that
        # swallows forever is unstoppable by anything in the process, so it would wedge
        # the SUITE at loop teardown instead of testing the drain.
        nonlocal swallowed
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 1.0
        while loop.time() < deadline:
            try:
                await asyncio.sleep(0.02)
            except asyncio.CancelledError:
                swallowed += 1

    task = asyncio.create_task(_stubborn())
    background._background_tasks.add(task)
    task.add_done_callback(background._background_tasks.discard)

    # The assertion IS the timeout: an unbounded drain never returns here.
    await asyncio.wait_for(
        background.drain_background_tasks(settle_seconds=0.05, cancel_seconds=0.1),
        timeout=2.0,
    )

    assert swallowed > 0, "the drain never cancelled the in-flight task"
    assert not task.done(), (
        "the stand-in was supposed to survive the drain — this test proves the drain "
        "RETURNS anyway, so a misbehaving job cannot hold shutdown past the grace period"
    )
    await asyncio.wait({task}, timeout=3.0)
