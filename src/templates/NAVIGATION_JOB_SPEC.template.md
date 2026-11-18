# <App> recording — navigation job spec (TEMPLATE)

> Copy this into a hand-authored recording package as `NAVIGATION_JOB_SPEC.md` and
> fill it in. It is the authoritative runbook for a footage-driven (`showcase` /
> `demo_video`) recording: the in-app navigation each clip must produce, the issues
> found across render rounds, and how each was resolved. Delete this quote block
> and the `<…>` placeholders when you fill it in.

**Purpose.** The video is recorded by **FTR re-running this package in-cluster**
(headless Chromium on the FTR pod) — NOT by a local dev run — so every requirement
below must hold on the FTR pod, which is **slower and heavier than a dev Mac**.
That gap is the root of most issues: things that render instantly locally lag or
don't paint in-cluster, so local validation can MASK a bug. Always confirm on the
in-cluster render, not just locally.

Pipeline: `brief.package_url` → FTR `POST /api/test-runner/execute` (runs `npm ci`
→ `npx bddgen` → `npx playwright test --timeout <brief.timeout_per_test_ms>`, one
video per scenario) → `demo_video_compile` joins each `@clip-N` to its narration
line and freezes/trims the clip to the narrated beat. Set
`brief.timeout_per_test_ms` high enough for the slowest clip (live waits, heavy
canvases) so it isn't killed by the default per-test cap.

---

## Clip ↔ beat map (what each recorded clip MUST show)

Each scenario is tagged `@clip-<n>` — the **join spine** binding footage ↔
narration ↔ compile shot. Never reorder or renumber after the narration brief is
written to it.

| Clip | Beat (brief line) | Required on-screen content |
|---|---|---|
| clip-1 | `<b?-id>` | `<what must be visible>` |
| clip-2 | `<b?-id>` | `<…>` |
| … | | |

Notes on framing (per `brief.lines[]`):
- `clip_anchor:"end"` shows the **settled TAIL** (skips app load-in / white) — use
  it for any clip that ends on a hero; the step must then hold a long stable dwell
  (see `heroDwell` in `lib/recording_helpers`).
- `clip_lead_seconds:N` skips a fixed head (keeps typing/detection motion) — more
  fragile than `end` (local vs slower FTR timing), so prefer a tail-anchored hero.

---

## Issues found & resolutions

Log every render-round defect here with symptom → root cause → resolution →
validation status. Most defects are the local-vs-FTR gap.

### ISSUE 1 — `<short title>`  *(clip-N…)*
**Symptom (in-cluster):** `<what went wrong on the FTR pod>`
**Root cause:** `<why>`
**Resolution:** `<what changed — step logic, a lib helper, a brief knob>`
**Validation:** `<confirmed on which render / still STANDBY>`

---

## Environment invariants (must hold for the recording to succeed)
- Orchestrator honors `brief.timeout_per_test_ms` (set it to the slowest clip's
  worst case + margin; keep the full-suite total under FTR's wall budget).
- The recording account is **CLEAN of the demo entities** before each run (a
  `_cleanup.mjs` built on `lib/fixture_reset.mjs` must print `CLEANUP_OK`) — fixed
  demo names collide across runs and stall the wizard clips.
- `brief.auth_config` carries `login_url` / `username` / `password` (FTR writes
  them to a 0600 `.env`; the package's `auth.setup.ts` reads `process.env.AUTH_*`).
- Any hardcoded entity ids in steps are **env-overridable with a fallback** (e.g.
  `process.env.<APP>_PROJECT_ID || '<uuid>'`) so a re-seed needs no code change.
- Note which page types render reliably in-cluster (DOM/SVG usually fine; heavy
  WebGL/canvas needs the load-in waited out + an explicit fit — see
  `recoverEditorIfCrashed` / `fitCanvasByKeys`).
