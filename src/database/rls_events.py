"""Phase D RLS — re-bind the tenant GUC at the start of every transaction.

``bind_session_tenant`` (and w2v_common ``get_db``/``get_db_session``) set
``app.current_tenant`` with ``set_config(..., is_local=true)`` — **transaction
scoped**, so the GUC is dropped on every ``commit()``. Request handlers that commit
once are fine, but the background BDD pipeline shares one session across several
``commit()`` calls (``mark_processing`` → … → result persistence): after the first
commit the GUC is gone and, under enforce, every later write fail-closes to 0 rows.

This installs a SQLAlchemy ``after_begin`` listener that re-applies the GUC from the
per-task identity ContextVar at the start of EVERY transaction (including the fresh
transaction opened after a commit). Inert unless ``TENANT_RLS_MODE=enforce`` and a
tenant is in context. Scope is the bdd_playwright process only (blast radius
per-service); the cleanup sweep stays cross-tenant via SECURITY DEFINER resolvers.
"""

from __future__ import annotations

from sqlalchemy import Connection, event
from sqlalchemy.orm import Session, SessionTransaction
from w2v_common.auth.identity_context import get_tenant_id_from_context
from w2v_common.database.rls import tenant_rls_mode

_installed = False


def install_rls_tenant_rebind() -> None:
    """Register the after_begin GUC re-bind listener once per process."""
    global _installed
    if _installed:
        return

    @event.listens_for(Session, "after_begin")
    def _rebind_tenant(
        session: Session, transaction: SessionTransaction, connection: Connection
    ) -> None:
        if tenant_rls_mode() != "enforce":
            return
        tenant = get_tenant_id_from_context()
        if not tenant:
            return
        # Sync event running in SQLAlchemy's greenlet — emit via the DBAPI cursor.
        # ``str(tenant)``: the ContextVar holds a ``uuid.UUID``, and psycopg adapts
        # that to the postgres ``uuid`` type -- which matches no overload of
        # ``set_config(text, text, bool)``, so every transaction under enforce died
        # with ``UndefinedFunction``. Every other service in the fleet already casts.
        connection.exec_driver_sql(
            "SELECT set_config('app.current_tenant', %s, true)", (str(tenant),)
        )

    _installed = True
