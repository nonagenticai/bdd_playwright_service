"""SECURITY DEFINER cross-tenant maintenance resolvers (single source).

The cleanup loop is cross-tenant and runs with no tenant context; under RLS enforce
the runtime (a non-bypass role) would fail-closed to 0 rows. These resolvers are
owned by the table owner, so they bypass ENABLE and operate across all tenants.

The DDL is single-sourced here so BOTH the alembic migration (prod) and the
``create_all`` ``after_create`` event (tests) create identical functions. The
migration additionally ``REVOKE``s EXECUTE from PUBLIC and re-GRANTs it to the app
role in-band (see :data:`GRANT_RESOLVERS_SQL`); tests run as the owner so EXECUTE
works without the grant.

⚠️ **Two prune-stuck constants, on purpose.** :data:`PRUNE_STUCK_SQL_0003` is the
ERA-CORRECT one-argument body that migration ``0003_rls`` created, frozen so a
fresh ``alembic upgrade head`` from zero still replays what actually happened (and
so 0003's ``REVOKE ... prune_stuck_bdd_jobs(timestamptz)`` still names a function
that exists at that point in the chain). :data:`PRUNE_STUCK_SQL` is TODAY's
two-argument body, created by ``0005_prune_pending``. A migration is a historical
record; it must not move when the code moves.

⛔ **BOTH OVERLOADS STAY RESIDENT — this is a ROLLOUT-SAFETY requirement, not
tidiness.** The live rollout spec for ``bdd-playwright-service`` is
``replicas: 1``, ``maxSurge: 25%``, ``maxUnavailable: 25%``. Kubernetes rounds
maxSurge UP to 1 and maxUnavailable DOWN to 0, so the OLD pod keeps serving until
the NEW pod is Ready: **the two versions OVERLAP**. The migration runs with the new
pod. If it DROPPED ``prune_stuck_bdd_jobs(timestamptz)``, the old pod's cleanup
loop — which calls the one-argument form on its periodic tick — would fail with
``function prune_stuck_bdd_jobs(timestamp with time zone) does not exist`` for the
whole length of the rollout. That is a self-inflicted outage of the very reclaim
path this change exists to fix.

So :func:`upgrade_statements` creates BOTH arities and drops NEITHER, and both get
a ``pg_roles``-guarded ``GRANT EXECUTE`` (a new signature is a new function with an
EMPTY ACL — the v1 grant does not carry over, and the v1 grant is never revoked).
Arities differ (1 vs 2), so calls stay unambiguous — no ``function
prune_stuck_bdd_jobs(...) is not unique``. That is PROVEN against a real Postgres by
``TestResolverOverloadsSurviveARollingUpdate``, not assumed.

**Dropping the one-argument form is a SAFE FOLLOW-UP PR** — once this revision is
live on every replica, no pod calls the one-argument form any more, and a later
migration may execute :data:`DROP_PRUNE_STUCK_V1_SQL`. It must not happen in the
same release that introduces the two-argument form.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

#: The restricted, non-bypass runtime role. Absent in CI (tests run as the owner),
#: which is exactly why the GRANT below is guarded rather than unconditional.
APP_ROLE = "w2v_bdd_playwright_app"

#: ⛔ FROZEN — the body migration ``0003_rls`` created. Do NOT "fix" it to match
#: :data:`PRUNE_STUCK_SQL`; 0003 must keep describing the schema of its own era.
#:
#: It is ALSO the body the one-argument overload keeps AFTER ``0005_prune_pending``:
#: during a rolling update the outgoing pod still calls this signature, and it must
#: behave exactly as it always did (``status = 'processing'`` only — the pending
#: widening belongs to the two-argument form the NEW pod calls). Re-running it as
#: ``CREATE OR REPLACE`` in 0005 is a no-op on an existing database and preserves
#: the function's ACL; it is what materialises the overload on a fresh one.
PRUNE_STUCK_SQL_0003 = """
CREATE OR REPLACE FUNCTION prune_stuck_bdd_jobs(p_cutoff timestamptz)
  RETURNS integer LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    WITH u AS (
      UPDATE bdd_jobs
      SET status = 'failed',
          error_message = 'Job timed out (stuck-job cleanup)',
          completed_at = now(),
          updated_at = now()
      WHERE status = 'processing' AND updated_at < p_cutoff
      RETURNING 1
    ) SELECT count(*)::int FROM u;
  $fn$
"""

#: Drop of the one-argument signature.
#:
#: ⛔ DELIBERATELY **NOT** IN :func:`upgrade_statements`. ``CREATE OR REPLACE
#: FUNCTION`` cannot change a function's ARITY, so the two-argument form is a
#: SECOND, overloaded ``prune_stuck_bdd_jobs`` and the one-argument one survives —
#: and during a rolling update that is exactly what we need, because the outgoing
#: pod is still calling it (see the module docstring). Executing this in the same
#: release that adds the two-argument form breaks the old pod for the length of the
#: rollout. It belongs in a LATER migration, once no running pod calls one argument.
#:
#: Still used by ``0005_prune_pending``'s ``downgrade()``, which walks the database
#: back to the 0003-era state where only the one-argument form existed.
DROP_PRUNE_STUCK_V1_SQL = "DROP FUNCTION IF EXISTS prune_stuck_bdd_jobs(timestamptz)"

#: ⚠️ TWO SEPARATE CUTOFFS, and they must stay separate.
#:
#: ``pending`` rows were reclaimed by NOTHING: ``prune_stuck_bdd_jobs`` matched only
#: ``processing`` and ``prune_old_bdd_jobs`` DELETEs only terminal rows, so a job
#: whose row committed in the route but whose dispatched task died before
#: ``mark_processing`` (SIGKILL, eviction, or a SIGTERM whose rescue hit an
#: already-disposed engine) sat in ``pending`` forever and its caller polled forever.
#:
#: ⛔ Reusing ``p_cutoff`` for both would be a REGRESSION, not a fix: ``pending`` is a
#: LEGITIMATE queued state, and a 60-minute clock would fail out live queued work.
#: The pending clock is therefore its own parameter, defaults far longer (6h), and
#: runs off ``created_at`` — ``updated_at`` is set at insert and never touched again
#: while a job waits, so for a pending row the two are the same instant, and
#: ``created_at`` says what we actually mean: how long since the job was accepted.
PRUNE_STUCK_SQL = """
CREATE OR REPLACE FUNCTION prune_stuck_bdd_jobs(
    p_cutoff timestamptz, p_pending_cutoff timestamptz
)
  RETURNS integer LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    WITH u AS (
      UPDATE bdd_jobs
      SET status = 'failed',
          error_message = CASE
            WHEN status = 'pending'
              THEN 'Job never started (pending-job cleanup)'
            ELSE 'Job timed out (stuck-job cleanup)'
          END,
          completed_at = now(),
          updated_at = now()
      WHERE (status = 'processing' AND updated_at < p_cutoff)
         OR (status = 'pending' AND created_at < p_pending_cutoff)
      RETURNING 1
    ) SELECT count(*)::int FROM u;
  $fn$
"""

PRUNE_OLD_SQL = """
CREATE OR REPLACE FUNCTION prune_old_bdd_jobs(p_cutoff timestamptz)
  RETURNS integer LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    WITH d AS (
      DELETE FROM bdd_jobs
      WHERE created_at < p_cutoff AND status IN ('completed', 'failed')
      RETURNING 1
    ) SELECT count(*)::int FROM d;
  $fn$
"""

#: ⚠️ These revoke from **PUBLIC** only — never from :data:`APP_ROLE`. The
#: one-argument entry is a re-assertion of what ``0003_rls`` already did (needed on
#: the ``create_all`` test path, which never replays 0003) and is a no-op in prod;
#: it does NOT touch the app role's grant, which the outgoing pod is still using.
REVOKE_RESOLVERS_SQL: tuple[str, ...] = (
    "REVOKE EXECUTE ON FUNCTION prune_stuck_bdd_jobs(timestamptz) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION prune_stuck_bdd_jobs(timestamptz, timestamptz) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION prune_old_bdd_jobs(timestamptz) FROM PUBLIC",
)

#: ⚠️ A NEW SIGNATURE IS BORN WITH AN EMPTY ACL. The v1 grant was applied out-of-band
#: to ``prune_stuck_bdd_jobs(timestamptz)`` and does NOT carry over to the two-argument
#: function, so without this the reaper would hit ``permission denied for function
#: prune_stuck_bdd_jobs`` in prod while CI — which runs as the owner — stayed green and
#: blind. Guarded on ``pg_roles`` because the app role does not exist in CI.
#: ⚠️ BOTH arities are granted, inside the same guard. The one-argument GRANT is what
#: keeps the OUTGOING pod's cleanup loop working for the length of the rolling update
#: (and re-establishes the grant on the ``create_all`` test path, which never replays
#: ``0003_rls``). It is idempotent in prod and must not be turned into a REVOKE.
#: ⚠️ LITERAL, not an f-string over ``APP_ROLE``: a role name cannot be parameterised
#: in DDL, so interpolation here is un-auditable by ruff (S608) for no benefit. The
#: constant above documents the name; ``test_grant_names_the_app_role`` keeps the two
#: from drifting.
GRANT_RESOLVERS_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_bdd_playwright_app') THEN
    GRANT EXECUTE ON FUNCTION prune_stuck_bdd_jobs(timestamptz)
      TO w2v_bdd_playwright_app;
    GRANT EXECUTE ON FUNCTION prune_stuck_bdd_jobs(timestamptz, timestamptz)
      TO w2v_bdd_playwright_app;
    GRANT EXECUTE ON FUNCTION prune_old_bdd_jobs(timestamptz) TO w2v_bdd_playwright_app;
  END IF;
END $$
"""


def upgrade_statements() -> tuple[str, ...]:
    """The full current resolver DDL: create BOTH arities, revoke PUBLIC, guarded grant.

    ⛔ There is deliberately NO ``DROP FUNCTION ... prune_stuck_bdd_jobs(timestamptz)``
    here. ``bdd-playwright-service`` rolls with ``replicas: 1`` / ``maxSurge: 25%``
    (→1) / ``maxUnavailable: 25%`` (→0), so the outgoing pod serves until the new pod
    is Ready and its cleanup loop keeps calling the ONE-argument form while this
    migration runs. Dropping it here would break that loop for the whole rollout.
    Dropping it is a safe FOLLOW-UP migration; see :data:`DROP_PRUNE_STUCK_V1_SQL`.
    """
    return (
        PRUNE_STUCK_SQL_0003,
        PRUNE_STUCK_SQL,
        PRUNE_OLD_SQL,
        *REVOKE_RESOLVERS_SQL,
        GRANT_RESOLVERS_SQL,
    )


def register_resolvers(metadata: MetaData) -> None:
    """Create the prune resolvers after the tables exist (test create_all path)."""

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        # Mirrors ``alembic upgrade head``: BOTH arities end up resident, exactly as
        # they are in prod after a rolling update. The test database is reused across
        # runs and every statement here is idempotent, so re-applying is safe.
        for stmt in upgrade_statements():
            connection.execute(text(stmt))

    event.listen(metadata, "after_create", _apply)
