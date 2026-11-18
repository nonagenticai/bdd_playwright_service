"""The RLS re-bind listener must pass the tenant to postgres as TEXT.

Regression guard. The ContextVar holds a ``uuid.UUID``, and psycopg adapts that
to the postgres ``uuid`` type. ``set_config`` is only defined as
``set_config(text, text, bool)``, so binding a UUID raised::

    psycopg.errors.UndefinedFunction:
        function set_config(unknown, uuid, boolean) does not exist

Under ``TENANT_RLS_MODE=enforce`` that fired on the FIRST transaction of every
request, so the whole service 500'd -- observed live as
``POST /api/bdd/demo-video returned 500`` failing the showcase E2E scenario.
Every other service in the fleet already casts with ``str(...)``; this one did not.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import src.database.rls_events as rls_events

_TENANT = uuid.UUID("e148e8af-3751-49bf-a0fa-56ab46d31ddc")


def _capture_listener() -> Any:
    """Install the module's listener against a stub event API and return it."""
    captured: dict[str, Any] = {}

    class _FakeEvent:
        @staticmethod
        def listens_for(_target: object, _name: str) -> Any:
            def _deco(fn: Any) -> Any:
                captured["fn"] = fn
                return fn

            return _deco

    with patch.object(rls_events, "event", _FakeEvent):
        rls_events._installed = False
        rls_events.install_rls_tenant_rebind()
        rls_events._installed = False
    return captured["fn"]


def _run(tenant: object, mode: str = "enforce") -> list[tuple[str, tuple[Any, ...]]]:
    """Fire the listener once, returning the SQL it emitted and its parameters."""
    emitted: list[tuple[str, tuple[Any, ...]]] = []
    connection = MagicMock()
    connection.exec_driver_sql = lambda sql, params: emitted.append((sql, params))

    listener = _capture_listener()
    with (
        patch.object(rls_events, "tenant_rls_mode", lambda: mode),
        patch.object(rls_events, "get_tenant_id_from_context", lambda: tenant),
    ):
        listener(MagicMock(), MagicMock(), connection)
    return emitted


def test_tenant_is_bound_as_text_not_uuid() -> None:
    emitted = _run(_TENANT)

    assert len(emitted) == 1
    sql, params = emitted[0]
    assert "set_config('app.current_tenant'" in sql
    bound = params[0]
    assert isinstance(bound, str), (
        f"bound {type(bound).__name__}; postgres has no set_config(...uuid...)"
    )
    assert bound == str(_TENANT)


def test_listener_is_inert_when_not_enforcing() -> None:
    assert _run(_TENANT, mode="permissive") == []


def test_listener_is_inert_without_a_tenant() -> None:
    assert _run(None) == []
