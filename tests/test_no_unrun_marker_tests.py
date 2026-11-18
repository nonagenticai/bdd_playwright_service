"""A marker that CI never re-selects makes its tests run on NO machine.

``pyproject.toml``'s ``addopts`` carries ``-m 'not integration'`` and
``.github/workflows/ci-cd.yaml`` re-selects that marker NOWHERE — no
``integration_test_select`` input, no second pytest invocation. So every test
wearing ``integration`` is deselected in PR CI, on main, and nightly: it
executes on no machine at all.

That is not hypothetical in this fleet. In ``w2v_orchestrator`` the seven
route-boundary tests guarding cross-tenant artifact adoption sat marker-hidden
for weeks — unwiring the guard they covered was a GREEN diff. A fix whose only
proof is a marker-hidden test is an unverified fix.

This gate makes the hole un-reopenable: a marker listed in ``addopts``'s
deselection expression may only be worn by a file that appears in
``ALLOWED_UNRUN`` below, with a written reason that is PRINTED on every run.

Three deliberate design choices, each learned the hard way in this fleet:

* **AST, not regex.** A previous gate here used a bare regex for its escape
  hatch and matched its own marker string inside a DOCSTRING, silencing itself.
  A regex that matches nothing also passes vacuously. ``ast`` sees only real
  attribute references, so a module that merely *talks about*
  ``pytest.mark.integration`` in prose (this one does, repeatedly) is invisible
  to it.
* **The allowlist is a Python dict whose reason is ECHOED.** A fleet gate once
  parsed its waiver reason and then discarded it, so every waiver rendered
  identically and a deliberate exemption was indistinguishable from an
  accident. Here the reason is printed on every run — the exclusion is loud, not
  a silent default.
* **It has VACUITY FLOORS.** "Nothing is marked wrong" is trivially satisfied by
  deleting the tests. ``MIN_CI_SELECTED_TESTS`` is a count floor over the whole
  suite, and ``REQUIRED_CI_SELECTED`` names the specific tenant-isolation tests
  that must exist AND be CI-selected — so deleting or re-hiding them fails just
  as loudly as marking them.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TESTS_ROOT = _REPO_ROOT / "tests"

#: Files permitted to wear a never-selected marker, mapped to WHY they cannot
#: run in CI. Adding an entry is a deliberate, reviewable statement that the
#: test needs something CI genuinely does not have — a live peer service or real
#: outbound network. Needing Postgres or Redis is NOT a qualifying reason: the
#: archetype provisions an ephemeral Postgres (``has_postgres: true``,
#: ``postgres_isolation: schema``) and a Redis (``has_redis: true``) per job.
ALLOWED_UNRUN: dict[str, str] = {}

#: THE HOLE THE 2026-09-07 VERSION OF THIS GATE LEFT OPEN.
#:
#: ``ALLOWED_UNRUN`` polices SILENCE — it demands a written, printed reason — but
#: it never demanded that anything ever RUN the waived file. "Run it by hand" was
#: an accepted reason, and for months nobody did. When the four contract classes
#: were finally executed against the live cluster on 2026-09-08 they were not
#: merely unrun, they were WRONG: one asserted ``POST /api/inspect/`` (trailing
#: slash), which the live service answers with a flat 404, and two asserted a
#: ``tts_service`` that is deployed on no machine in the fleet.
#:
#: So a waiver now costs two things, not one: a reason, AND a lane that really
#: executes the checks — a workflow file that exists, is dispatchable, and
#: references the artifact it runs. This is the assertion that stops "it needs
#: live peers" from being a permanent excuse for zero coverage.
REQUIRED_LANE: dict[str, tuple[str, str]] = {}

#: VACUITY FLOOR 1 — tenant-isolation tests that must exist AND be CI-selected.
#: These are the tests whose silent disappearance would be indistinguishable
#: from a passing suite: cross-tenant ``feature_files`` confinement (a guessable
#: ``uploads/{tenant_id}/`` namespace), route-level tenant scoping, the
#: tenant-required 401 boundary, and the RLS role isolation proof. Do not
#: shorten this list to make the file pass.
REQUIRED_CI_SELECTED: dict[str, tuple[str, ...]] = {
    "tests/unit/test_feature_files_tenant_confinement.py": (
        "test_exploit_sibling_tenant_ref_is_refused",
        "test_traversal_out_of_own_prefix_is_refused",
        "test_url_encoded_traversal_is_refused",
        "test_double_encoded_traversal_is_refused",
        "test_sibling_prefix_lookalike_is_refused",
        "test_missing_tenant_refuses_everything",
        "test_minio_backend_is_confined_too",
    ),
    "tests/unit/test_route_tenant_scoping.py": (
        "test_cross_tenant_404",
        "test_cross_tenant_and_genuine_miss_share_the_same_not_found_shape",
        "test_get_job_with_mismatched_tenant_returns_none",
    ),
    "tests/unit/test_tenant_required.py": (
        "test_create_route_without_tenant_is_401",
        "test_read_route_without_tenant_is_401",
        "test_helper_rejects_the_nil_uuid_by_value",
    ),
    "tests/test_rls_isolation.py": ("test_rls_isolates_tenants_under_restricted_role",),
}

#: VACUITY FLOOR 2 — the minimum number of test functions that must live in
#: ``tests/`` WITHOUT wearing a deselected marker. Measured 2026-09-07: 891
#: CI-selected test functions (897 total, minus the 6 waived contract tests).
#: The floor sits below that so ordinary churn does not trip it, but wholesale
#: deletion — or mass re-marking, which moves tests out of this count — does.
MIN_CI_SELECTED_TESTS = 830


def _deselected_markers() -> set[str]:
    """The markers ``addopts`` excludes, read from pyproject — not hardcoded.

    Reading it back means adding a SECOND marker to the deselection expression
    without allowlisting or wiring it also trips this gate.
    """
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    addopts = cfg["tool"]["pytest"]["ini_options"]["addopts"]
    if isinstance(addopts, list):
        addopts = " ".join(addopts)
    # `-m 'not a and not b'` -> {"a", "b"}
    tokens = addopts.replace("'", " ").replace('"', " ").split()
    out: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok == "not" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt not in {"not", "and", "or"}:
                out.add(nxt)
    return out


def _marker_names_used(path: Path) -> set[str]:
    """Every ``<something>.mark.<name>`` referenced anywhere in the module.

    Covers decorators (function AND class), module-level ``pytestmark`` (scalar
    or list), and marks applied inside ``pytest.param``. Comments and docstrings
    are not AST attribute nodes, so prose about a marker cannot trip this.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = node.value
        if isinstance(parent, ast.Attribute) and parent.attr == "mark":
            names.add(node.attr)  # pytest.mark.<name>
        elif isinstance(parent, ast.Name) and parent.id == "mark":
            names.add(node.attr)  # from pytest import mark; mark.<name>
    return names


def _collected_test_names(path: Path) -> set[str]:
    """``test_*`` functions in a module, via AST (no import side effects).

    Walks the whole tree, not just module top level: this suite puts most of its
    tenant-isolation tests inside ``Test*`` classes.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test_")
    }


def test_addopts_deselection_is_covered_by_this_gate() -> None:
    """The expression must be parseable and non-degenerate, or the gate is inert."""
    markers = _deselected_markers()
    assert markers, (
        "Could not parse any deselected marker out of pyproject.toml's "
        "[tool.pytest.ini_options] addopts. This gate reads that expression to "
        "know what to police; if the parse yields nothing the gate silently "
        "polices nothing. Fix the parser in _deselected_markers(), do not "
        "delete this assertion."
    )
    declared = set(
        tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"][
            "markers"
        ]
    )
    declared_names = {m.split(":", 1)[0].strip() for m in declared}
    unknown = markers - declared_names
    assert not unknown, (
        f"addopts deselects marker(s) {sorted(unknown)} that are not declared in "
        "[tool.pytest.ini_options].markers. A typo'd marker name in the "
        "deselection expression deselects nothing and hides the mistake."
    )


def test_no_test_wears_a_marker_ci_never_selects(capsys: pytest.CaptureFixture[str]) -> None:
    markers = _deselected_markers()
    offenders: dict[str, set[str]] = {}
    waived: list[str] = []

    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        used = _marker_names_used(path) & markers
        if not used:
            continue
        if rel in ALLOWED_UNRUN:
            reason = ALLOWED_UNRUN[rel].strip()
            assert reason, (
                f"{rel} is in ALLOWED_UNRUN with an EMPTY reason. A waiver whose "
                "reason is blank is indistinguishable from an accident — state "
                "what CI genuinely lacks (a live peer service, real network)."
            )
            # PRINTED, not merely parsed: a fleet gate once collected its waiver
            # reasons and discarded them, so every waiver rendered identically.
            waived.append(f"  {rel} [{','.join(sorted(used))}] — {reason}")
            continue
        offenders[rel] = used

    if waived:
        with capsys.disabled():
            print(
                "\nDELIBERATELY UNRUN IN CI (each waiver and its stated reason):\n"
                + "\n".join(sorted(waived))
            )

    if offenders:
        detail = "\n".join(f"  {f}: {sorted(m)}" for f, m in sorted(offenders.items()))
        pytest.fail(
            "These test files wear a pytest marker that pyproject's `addopts` "
            "DESELECTS and that CI re-selects NOWHERE — so they execute on no "
            f"machine at all:\n{detail}\n\n"
            "In w2v_orchestrator the equivalent hole hid seven cross-tenant "
            "route guards for weeks: unwiring them was a green diff.\n\n"
            "Fix it by REMOVING the marker (the archetype gives every job an "
            "ephemeral Postgres and a Redis, so 'it needs a DB' is not a reason "
            "to hide a test), or — if the test truly needs a live peer service "
            "or real outbound network — add it to ALLOWED_UNRUN in "
            f"{Path(__file__).name} with a written reason, which is a reviewable "
            "act rather than a silent one."
        )


def test_the_tenant_isolation_tests_are_present_and_ci_selected() -> None:
    """VACUITY FLOOR 1: deleting the tests must fail as loudly as hiding them.

    Without this, ``test_no_test_wears_a_marker_ci_never_selects`` is satisfied
    by an empty ``tests/`` tree.
    """
    markers = _deselected_markers()
    problems: list[str] = []

    for rel, required in REQUIRED_CI_SELECTED.items():
        path = _REPO_ROOT / rel
        if not path.exists():
            problems.append(f"{rel} is GONE — it held {len(required)} required tests.")
            continue
        present = _collected_test_names(path)
        missing = [n for n in required if n not in present]
        if missing:
            problems.append(f"{rel}: missing required test(s) {missing}")
        hidden = _marker_names_used(path) & markers
        if hidden:
            problems.append(
                f"{rel}: marker(s) {sorted(hidden)} would deselect its required tests in CI."
            )

    assert not problems, (
        "The tenant-isolation tests this gate exists to keep RUNNING are not all "
        "present and CI-selected:\n  " + "\n  ".join(problems) + "\n\n"
        "These cover cross-tenant `feature_files` confinement over the guessable "
        "`uploads/{tenant_id}/` namespace, route-level tenant scoping, the "
        "tenant-required 401 boundary, and RLS role isolation. Without them, "
        "deleting a confinement call from a route handler is a green diff. "
        "Restore them — do not shorten REQUIRED_CI_SELECTED to make this pass."
    )


def test_the_suite_has_not_been_emptied_to_satisfy_this_gate() -> None:
    """VACUITY FLOOR 2: a count floor over everything CI actually selects."""
    markers = _deselected_markers()
    total = 0
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if _marker_names_used(path) & markers:
            continue  # not CI-selected; deliberately excluded from the count
        total += len(_collected_test_names(path))

    assert total >= MIN_CI_SELECTED_TESTS, (
        f"Only {total} CI-selected test functions remain under tests/, below the "
        f"floor of {MIN_CI_SELECTED_TESTS} (891 at the 2026-09-07 measurement). "
        "A gate that only asserts 'nothing is marked wrong' is satisfiable by "
        "deleting the tests, or by mass-marking them `integration`; this is the "
        "assertion that catches that."
    )


def test_every_waived_file_has_a_lane_that_actually_executes_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A waiver without a lane is the dead pile with paperwork attached.

    This is the assertion the previous version of this gate was missing, which is
    why four contract tests could sit correctly waived, loudly documented, and
    still never execute a single assertion anywhere.
    """
    problems: list[str] = []
    lanes: list[str] = []

    for rel in sorted(ALLOWED_UNRUN):
        entry = REQUIRED_LANE.get(rel)
        if entry is None:
            problems.append(
                f"{rel} is waived in ALLOWED_UNRUN but names no lane in REQUIRED_LANE. "
                "A written reason explains why per-PR CI cannot run it; it does not "
                "make anything run it. Wire a workflow that does, and name it here."
            )
            continue
        workflow_rel, must_reference = entry
        workflow = _REPO_ROOT / workflow_rel
        if not workflow.exists():
            problems.append(f"{rel}: its lane {workflow_rel} does not exist.")
            continue
        body = workflow.read_text()
        if must_reference not in body:
            problems.append(
                f"{rel}: lane {workflow_rel} no longer references '{must_reference}', "
                "so it cannot be executing these checks."
            )
            continue
        if "workflow_dispatch" not in body:
            problems.append(
                f"{rel}: lane {workflow_rel} is not dispatchable (no workflow_dispatch). "
                "A lane nobody can run on demand is a lane whose result nobody reads."
            )
            continue
        lanes.append(f"  {rel} -> {workflow_rel} (references '{must_reference}')")

    if lanes:
        with capsys.disabled():
            print("\nOUT-OF-CI LANES THAT DO EXECUTE THE WAIVED FILES:\n" + "\n".join(lanes))

    assert not problems, "\n  ".join(["Waived files without a real executing lane:", *problems])


def test_no_stale_lane_entries() -> None:
    """A lane declared for a file that is no longer waived is misleading."""
    orphans = sorted(set(REQUIRED_LANE) - set(ALLOWED_UNRUN))
    assert not orphans, (
        f"REQUIRED_LANE names {orphans}, which are not in ALLOWED_UNRUN. Either the "
        "waiver was removed (delete the lane entry) or the key is a typo — in which "
        "case the real waived file is silently laneless again."
    )


def test_stale_allowlist_entries_are_removed() -> None:
    """A waiver for a file that no longer wears the marker is dead weight."""
    markers = _deselected_markers()
    stale = []
    for rel, reason in ALLOWED_UNRUN.items():
        path = _REPO_ROOT / rel
        if not path.exists() or not (_marker_names_used(path) & markers):
            stale.append(f"{rel} (waived because: {reason})")
    assert not stale, (
        "ALLOWED_UNRUN entries no longer needed (file gone, or marker removed):\n  "
        + "\n  ".join(stale)
        + "\nDelete them so the allowlist keeps meaning what it says."
    )
