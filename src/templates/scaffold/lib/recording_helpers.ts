// Shared step helpers for app-recording playwright-bdd packages.
//
// These are the app-AGNOSTIC recording idioms extracted from the first
// hand-authored recording package (Specra onboarding, w2v video #1). Every
// helper takes the app-specific bits (selectors, labels, storage keys, ids) as
// parameters so it carries no per-app knowledge. A recording package — generated
// by bdd_playwright_service or hand-authored from the scaffold — imports these
// instead of re-deriving them.
//
// Design rules that make these work as VIDEO footage (not just as tests):
//   • Each @clip-<n> scenario records in its own fresh context = its own video
//     clip, so wizard state does NOT carry across clips — chain it via disk
//     (saveDraft / resumeFromDraft) or hand ids off through a sidecar file.
//   • The compositor shows each clip via clip_anchor="end" (the settled TAIL),
//     so a clip must hold a LONG stable hero at the end (heroDwell), longer than
//     the narration line riding it, and NEVER end mid-transition.
//   • Prefer MOTION over a frozen frame where narration is long (dwellWithScroll).
//
// This file is the CANONICAL copy. `bdd_playwright_service`'s ConfigGenerator
// emits it into every generated package; hand-authored packages (e.g.
// specra/recording-package/lib/recording_helpers.ts) carry a byte-identical copy.
// Keep them in sync — edit here first.

import { type Page, type Locator, expect, test } from '@playwright/test';
import * as fs from 'node:fs';
import * as path from 'node:path';

/** Longest a settled hero should be held so clip_anchor="end" only ever shows
 *  the stable tail — longer than the longest narration line riding the clip.
 *
 *  ⚠️ DO NOT SHORTEN. The dwell holds the screen WHILE THE NARRATION PLAYS, so it
 *  must stay >= the longest answer line for render's `-t` to trim cleanly;
 *  shortening it truncates the voiceover. Total demo duration is reconciled
 *  upstream by SCENARIO COUNT, never by cutting this.
 *
 *  This value is the DEFAULT per-scenario dwell budget and its CEILING (see the
 *  ledger below — an ordered film duration may narrow the budget, never raise
 *  it past this), and is
 *  the single owner of the number: `step_def_generator.HERO_DWELL_MS` PARSES it
 *  out of this file rather than restating it, and publishes it to the
 *  orchestrator as `narration_script.parts[].expected_recording_seconds`. */
export const HERO_DWELL_MS = 22_000;

/** Timeout for the best-effort `waitForLoadState('networkidle').catch(...)`
 *  settle calls below. `networkidle` is a NAVIGATION-class wait, so with no
 *  explicit timeout it falls back to `navigationTimeout` (15_000ms in
 *  playwright.config.ts) rather than the shorter `actionTimeout`, and the
 *  `.catch()` swallows a timeout silently — so an unresponsive page pays the
 *  full 15s before the recording continues. That stall lands directly in the
 *  recorded clip, and the clip's measured length feeds the orchestrator's
 *  duration solver as a hard minimum, so keep this short. */
export const NETWORK_IDLE_TIMEOUT_MS = 4_000;

// ── per-scenario dwell LEDGER (this is what makes clip length DETERMINISTIC) ───
//
// Demo-video duration used to be BIMODAL: a scenario whose steps happened to call
// `heroDwell` recorded 16.7-36.9s, and one whose steps did not recorded 1.24-3.94s
// — an 18x swing, with BOTH outcomes reporting `status=completed` / pass_rate 100%.
// Downstream that surfaces as `timeline GROSSLY UNDERFULL — 4.06s of the 30.00s
// target (13.5%)`. The cause was that the dwell was only ASKED FOR in the
// generator prompt while `softExpectVisible`/`resilientClick` from this same
// module were FORCE-INJECTED; a prompt rule is advisory, an injection is not.
//
// The fix has two halves. The generator now force-injects an `After` hook
// (`steps_pacing.ts`) that calls `heroDwell(page)` on EVERY scenario, and this
// ledger makes that hook idempotent WITH RESPECT TO WHAT THE MODEL ALREADY
// EMITTED: every discretionary hold spends from one per-scenario budget of
// HERO_DWELL_MS, so a scenario that already dwelled in its own steps gets a
// top-up of zero instead of a doubled 44s clip.
//
// Keyed by `page` because each @clip-<n> scenario records in its OWN fresh
// context, so its `page` is a distinct object — per-scenario accounting with no
// hook-ordering assumption and nothing to reset. Same shape as the generator's
// `stubDwell` accounting, deliberately.
const dwellSpent = new WeakMap<Page, number>();

// ── ORDERED-DURATION budget (the ledger's cap, narrowable per package) ────────
//
// HERO_DWELL_MS is the DEFAULT per-scenario budget, and stays the ceiling. But a
// caller can order a total film duration whose share per scenario is SMALLER
// than it (a 30s order over 2 scenarios is 15s each), and until this existed
// every clip ran to the fixed 22s budget regardless — which is why a 30s order
// delivered 47.7s. `bdd_playwright_service` emits a `steps_dwell_budget.ts`
// module calling `setScenarioDwellBudgetMs` when, and only when, a duration was
// ordered; with no order the budget is HERO_DWELL_MS and nothing changes.
//
// It NARROWS only. Raising the ledger above HERO_DWELL_MS would ask for a dwell
// the per-test Playwright timeout kills mid-hook, so the setter clamps.
let scenarioDwellBudgetMs: number = HERO_DWELL_MS;

// Whether a caller actually ORDERED a duration. This is the discriminator that
// keeps the no-order path byte-identical: `steps_dwell_budget.ts` is emitted ONLY
// when a duration was ordered, so this flips only then. The generated
// `steps_autostub.ts` shares THIS ledger when an order exists (so hero + stub
// cannot each spend a full budget on one clip) and keeps its own independent
// timeout-derived budget when none does — which is exactly today's behaviour.
let dwellBudgetOrdered = false;

/** Narrow this package's per-scenario dwell budget (ms). Clamped to
 *  `[0, HERO_DWELL_MS]`; call once at module load, before any scenario runs. */
export function setScenarioDwellBudgetMs(ms: number): void {
  scenarioDwellBudgetMs = Math.max(0, Math.min(HERO_DWELL_MS, Math.round(ms)));
  dwellBudgetOrdered = true;
}

/** True once an ordered duration has narrowed this package's dwell budget. */
export function isDwellBudgetOrdered(): boolean {
  return dwellBudgetOrdered;
}

/** How much of this scenario's dwell budget is still unspent (ms). */
export function dwellRemainingMs(page: Page): number {
  return Math.max(0, scenarioDwellBudgetMs - (dwellSpent.get(page) ?? 0));
}

/** Spend `ms` of the scenario's dwell budget, clamped to what is left.
 *
 *  EXPORTED because the generated `steps_autostub.ts` spends from THIS ledger
 *  too. It used to keep its own private WeakMap and its own ~20.25s ceiling
 *  derived from `PW_TEST_TIMEOUT_MS`, which meant a scenario could hold its
 *  ordered share here AND up to ~20s more there — two budgets, one clip, and
 *  only this one narrowed by `setScenarioDwellBudgetMs`. That is how a 30s
 *  order came back with 43.56s of footage. One ledger, one ceiling. */
export async function spendDwell(page: Page, ms: number): Promise<void> {
  const grant = Math.min(Math.max(0, ms), dwellRemainingMs(page));
  if (grant <= 0) return;
  dwellSpent.set(page, (dwellSpent.get(page) ?? 0) + grant);
  await page.waitForTimeout(grant);
}

/**
 * Hold the current (settled) frame as the clip's tail.
 *
 * Spends from the per-scenario budget, so calling it twice in one scenario holds
 * HERO_DWELL_MS in TOTAL rather than twice over. That is what lets the generated
 * `After` hook top every scenario up to the same length without double-dwelling
 * the ones whose steps already called it.
 */
export async function heroDwell(page: Page, ms: number = HERO_DWELL_MS): Promise<void> {
  await spendDwell(page, ms);
}

// ── draft-chaining across isolated per-clip contexts ──────────────────────────
// A stateful wizard recorded as separate clips loses its in-progress
// localStorage between clips. Persist it after each stage (saveDraft) and
// re-inject it before the SPA boots on the next stage (resumeFromDraft) so the
// wizard RESUMES rather than restarts.

function dottedGet(obj: unknown, dotted: string): unknown {
  return dotted.split('.').reduce<unknown>((acc, key) => {
    if (acc && typeof acc === 'object' && key in (acc as Record<string, unknown>)) {
      return (acc as Record<string, unknown>)[key];
    }
    return undefined;
  }, obj);
}

/**
 * Persist a wizard's localStorage entry to disk so the next clip can resume it.
 *
 * The SPA writes its draft on a DEBOUNCE after a step succeeds, so this polls
 * until the draft materializes. Pass `requirePath` (e.g. `"data.project.id"`) to
 * poll until the persisted draft actually carries a freshly-created id — without
 * it a plain read can race AHEAD of the server-persist and capture a pre-create
 * draft, sending the next clip to the wrong step.
 */
export async function saveDraft(
  page: Page,
  opts: { storageKey: string; file: string; requirePath?: string; timeoutMs?: number },
): Promise<void> {
  const { storageKey, file, requirePath, timeoutMs = 20_000 } = opts;
  const deadline = timeoutMs;
  const stepMs = 500;
  let draft: string | null = null;
  for (let waited = 0; waited < deadline; waited += stepMs) {
    draft = await page.evaluate((key) => window.localStorage.getItem(key), storageKey);
    if (draft && draft.length > 2) {
      if (!requirePath) break;
      try {
        if (dottedGet(JSON.parse(draft), requirePath) != null) break;
      } catch {
        /* draft mid-write — keep polling */
      }
    }
    await page.waitForTimeout(stepMs);
  }
  const ok =
    draft != null &&
    draft.length > 2 &&
    (!requirePath || (() => {
      try {
        return dottedGet(JSON.parse(draft as string), requirePath) != null;
      } catch {
        return false;
      }
    })());
  if (!ok) {
    throw new Error(
      `localStorage["${storageKey}"]${requirePath ? ` (path ${requirePath})` : ''} not ready ` +
        `after ${timeoutMs}ms — cannot chain to the next clip`,
    );
  }
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, draft as string, 'utf8');
}

/**
 * Re-inject a saved draft (before the SPA boots) and navigate to `url` so the
 * wizard resumes from it. `url` should be the plain resume route (NOT a
 * fresh/reset route).
 */
export async function resumeFromDraft(
  page: Page,
  opts: { storageKey: string; file: string; url: string; settleMs?: number },
): Promise<void> {
  const { storageKey, file, url, settleMs = 1_500 } = opts;
  const draft = fs.readFileSync(file, 'utf8');
  await page.context().addInitScript(
    ([key, value]) => {
      try {
        window.localStorage.setItem(key, value);
      } catch {
        /* origin not established yet — the goto below re-runs this */
      }
    },
    [storageKey, draft] as [string, string],
  );
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('networkidle', { timeout: NETWORK_IDLE_TIMEOUT_MS }).catch(() => {});
  await page.waitForTimeout(settleMs);
}

// ── sidecar id hand-off between clips ─────────────────────────────────────────

/** Write a small JSON sidecar (e.g. `{ projectId }`) for a later clip to read. */
export function writeJsonFile(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(data), 'utf8');
}

/** Read a JSON sidecar written by an earlier clip. */
export function readJsonFile<T = Record<string, unknown>>(file: string): T {
  return JSON.parse(fs.readFileSync(file, 'utf8')) as T;
}

/** Pull the first capture group of `re` out of the current page URL, or null. */
export function idFromUrl(page: Page, re: RegExp): string | null {
  const m = page.url().match(re);
  return (m && m[1]) || null;
}

// ── visible typing ────────────────────────────────────────────────────────────

/** Type character-by-character so the recording shows human-paced typing. */
export async function typeVisibly(
  target: Locator,
  value: string,
  opts: { delayMs?: number; clearFirst?: boolean; dismissTypeahead?: boolean; dwellMs?: number } = {},
): Promise<void> {
  const { delayMs = 45, clearFirst = false, dismissTypeahead = false, dwellMs = 700 } = opts;
  await target.click();
  if (clearFirst) await target.fill('');
  await target.pressSequentially(value, { delay: delayMs });
  if (dismissTypeahead) await target.page().keyboard.press('Escape');
  await target.page().waitForTimeout(dwellMs);
}

// ── resilient interaction (a missing selector must NOT blank the whole clip) ──
//
// A generated step whose authored selector no longer resolves would THROW and
// abort the whole scenario → the clip records BLANK footage (the app is never
// driven) → the demo_video blank-clip gate hard-fails the paid render. These
// helpers degrade a missing target GRACEFULLY: try the authored selector, then
// accessible locators derived from the element's human label, and on a total
// miss record a machine-readable `soft-fail` annotation + `##SOFT_FAIL##` log
// and RETURN WITHOUT THROWING — the page stays on-screen (non-blank) and the
// frontend_test_runner summary still demotes a soft-failed scenario from
// passed, so pass_rate stays honest. A self-heal attempt never throws itself.

/** Per-attempt timeout — short so N broken steps stay within the clip budget
 *  instead of pinning the global 10s actionTimeout each. */
export const RESILIENT_ATTEMPT_MS = 2_500;

/** Record a non-fatal miss on both channels the runner can read (JSON-report
 *  annotation + stdout marker), then let the caller continue. */
function softFail(action: string, primary: string, label: string, err: unknown): void {
  const detail = JSON.stringify({ action, selector: primary, label, error: String(err) });
  try {
    test.info().annotations.push({ type: 'soft-fail', description: detail });
  } catch {
    // Not inside a running test (e.g. imported by a unit test) — the log marker
    // below still carries the signal; never let annotation bookkeeping throw.
  }
  console.warn(`##SOFT_FAIL## ${detail}`);
}

/** Accessible locators derived from a human label, tried in order after the
 *  authored selector. Empty when no usable label is supplied. */
function labelLocators(page: Page, label: string): Locator[] {
  const name = (label || '').trim();
  if (!name) return [];
  return [
    page.getByRole('button', { name }),
    page.getByRole('link', { name }),
    page.getByLabel(name),
    page.getByText(name),
  ];
}

/**
 * Click `primary` (an authored CSS/XPath/text selector); if it never resolves,
 * fall back to accessible locators built from `label`. On total failure record
 * a soft-fail and RETURN (do not throw) so the clip stays non-blank. Prefer this
 * over a bare `page.click` in every generated/authored recording step.
 */
/**
 * Assert that NAVIGATION SUCCEEDED — the only thing an "I am on the X page"
 * precondition can honestly assert.
 *
 * A landing `Given` that instead asserts some NAMED element ("the TodoMVC link")
 * is the single most expensive brittleness in a generated recording package: the
 * element is chosen from the PRE-interaction inspection snapshot (often invented
 * outright), the hard `expect(...).toBeVisible()` throws, step 1 of the scenario
 * aborts, and every later When/Then NEVER RUNS — so the clip records a static
 * landing page with zero interaction. Measured live: 112/112 demo masters static,
 * median duplicate_frame_ratio 0.905.
 *
 * This THROWS on a real navigation failure (still at `about:blank`, blank body) —
 * that is a product/config bug the recording must not paper over. It deliberately
 * says nothing about page CONTENT; decorative content checks belong in
 * `softExpectVisible`.
 */
export async function assertNavigated(
  page: Page,
  opts: { timeout?: number } = {},
): Promise<void> {
  const { timeout = 15_000 } = opts;
  await page.waitForLoadState('domcontentloaded', { timeout }).catch(() => {});
  const url = page.url();
  if (!/^https?:\/\//i.test(url)) {
    throw new Error(`navigation precondition failed: page is at '${url || 'about:blank'}'`);
  }
  await expect(page.locator('body')).toBeVisible({ timeout });
  // A 404/500 error page is at an http(s) URL and has a visible `body`, so every
  // check above PASSES on one and the "demo" films the error page. The document's
  // delivered status is the cheap discriminator: a PerformanceNavigationTiming
  // entry folds a redirect CHAIN into one entry and reports the status of the
  // document that was finally delivered, so a legitimate 30x -> 200 redirect reads
  // as 200 and is untouched.
  //
  // SOFT, not fatal, and deliberately so: throwing here would abort the scenario
  // and blank the rest of the clip — the exact failure mode `softExpectVisible`
  // exists to avoid — and would newly redden runs that ship today. `softFail`
  // records the machine-readable annotation the runner already reads (it demotes
  // a soft-failed scenario from passed), so the signal is honest without being
  // destructive. `0` means "no measurement" (client-side SPA route, or a browser
  // without `responseStatus`) and changes nothing.
  const status = await page
    .evaluate(() => {
      const nav = performance.getEntriesByType('navigation')[0] as
        | (PerformanceEntry & { responseStatus?: number })
        | undefined;
      return nav && typeof nav.responseStatus === 'number' ? nav.responseStatus : 0;
    })
    .catch(() => 0);
  if (status >= 400) {
    softFail(
      'navigation-status',
      url,
      `HTTP ${status}`,
      new Error(`navigated document responded HTTP ${status}`),
    );
  }
}

/**
 * A NON-FATAL "is it visible?" check. Waits for `target`, and on a miss records a
 * soft-fail (same two channels as `resilientClick`/`resilientFill`) and RETURNS
 * `false` instead of throwing — so one wrong selector costs one assertion, not
 * the remaining 90% of the scenario. Use for DECORATIVE presence checks inside a
 * navigation precondition; keep a hard `expect` for the Then step that carries
 * the scenario's real claim.
 */
export async function softExpectVisible(
  page: Page,
  target: Locator | string,
  label = '',
  opts: { timeout?: number } = {},
): Promise<boolean> {
  const { timeout = RESILIENT_ATTEMPT_MS } = opts;
  const described = typeof target === 'string' ? target : String(target);
  const loc = typeof target === 'string' ? page.locator(target) : target;
  try {
    await loc.first().waitFor({ state: 'visible', timeout });
    return true;
  } catch (err) {
    softFail('expect-visible', described, label, err);
    return false;
  }
}

/** Resolve a resilient helper's PRIMARY target.
 *
 * `primary` accepts a Locator as well as a selector string. A generated step that
 * needs a SCOPED target — "the checkbox inside the 'Walk the dog' row" — cannot
 * express that as a page-wide selector string, and when this parameter was
 * `string` only the model fell back to an unscoped `input[type="checkbox"]`
 * (which on TodoMVC resolves to the toggle-all box, not the row's toggle) and
 * left the scoped Locator it had already built bound and unused. Since
 * `resilientClick` soft-fails without throwing, the click step still "passed" and
 * the defect surfaced one step later as `expect(...).toBeChecked()` receiving an
 * unchecked box. Accepting a Locator here is what makes the scoped target
 * expressible at all; `_ensure_scoped_resilient_targets` in the generator rewires
 * the call deterministically. */
function primaryLocator(page: Page, primary: string | Locator): Locator | null {
  if (typeof primary !== 'string') return primary.first();
  return primary.trim() ? page.locator(primary).first() : null;
}

/** A human-readable description of a resilient helper's primary target, for the
 *  soft-fail record (a Locator stringifies to its own selector chain). */
function describePrimary(primary: string | Locator): string {
  return typeof primary === 'string' ? primary : String(primary);
}

export async function resilientClick(
  page: Page,
  primary: string | Locator,
  label = '',
  opts: { timeout?: number; dwellMs?: number } = {},
): Promise<void> {
  const { timeout = RESILIENT_ATTEMPT_MS, dwellMs = 800 } = opts;
  const candidates: Locator[] = [];
  const target = primaryLocator(page, primary);
  if (target) candidates.push(target);
  for (const loc of labelLocators(page, label)) candidates.push(loc.first());
  let lastErr: unknown = 'no candidate selector';
  for (const loc of candidates) {
    try {
      await loc.click({ timeout });
      await page.waitForTimeout(dwellMs);
      return;
    } catch (err) {
      lastErr = err;
    }
  }
  softFail('click', describePrimary(primary), label, lastErr);
}

/**
 * Fill `primary` with `value`; fall back to accessible locators from `label`.
 * Soft-fails (no throw) on total miss so the clip stays non-blank. Prefer this
 * over a bare `page.fill` in every generated/authored recording step.
 */
export async function resilientFill(
  page: Page,
  primary: string | Locator,
  value: string,
  label = '',
  opts: { timeout?: number; dwellMs?: number } = {},
): Promise<void> {
  const { timeout = RESILIENT_ATTEMPT_MS, dwellMs = 500 } = opts;
  const candidates: Locator[] = [];
  const target = primaryLocator(page, primary);
  if (target) candidates.push(target);
  for (const loc of labelLocators(page, label)) candidates.push(loc.first());
  let lastErr: unknown = 'no candidate selector';
  for (const loc of candidates) {
    try {
      await loc.fill(value, { timeout });
      await page.waitForTimeout(dwellMs);
      return;
    } catch (err) {
      lastErr = err;
    }
  }
  softFail('fill', describePrimary(primary), label, lastErr);
}

// ── motion dwell (scroll easing so a long clip has motion, not a frozen frame) ─

/** Smoothly ease the page through a set of scroll offsets, then optionally hold.
 *
 *  Its holds spend from the SAME per-scenario budget as `heroDwell` — it is the
 *  motion-flavoured alternative to a frozen hero, not an addition to it. Without
 *  the shared ledger a scenario that used both would record ~32s while its
 *  siblings recorded 22s, re-introducing exactly the variance the ledger exists
 *  to remove. */
export async function dwellWithScroll(
  page: Page,
  opts: { offsets?: number[]; stepMs?: number; holdMs?: number } = {},
): Promise<void> {
  const { offsets = [350, 700, 350, 0], stepMs = 2_400, holdMs = 0 } = opts;
  for (const y of offsets) {
    await page.evaluate((yy) => window.scrollTo({ top: yy, behavior: 'smooth' }), y);
    await spendDwell(page, stepMs);
  }
  if (holdMs) await spendDwell(page, holdMs);
}

// ── coach-mark / tooltip dismissal ────────────────────────────────────────────

/**
 * Kill an onboarding coach-mark / tooltip so it never overlays the hero. Tries a
 * dismiss affordance (button OR plain text — the role varies), then hard-hides
 * the container via the DOM as a fallback. Idempotent.
 */
export async function dismissCoachMark(
  page: Page,
  opts: { contains: RegExp; dismissLabel?: RegExp; ancestorLevels?: number; maxTextLen?: number },
): Promise<void> {
  const { contains, dismissLabel = /Got it/i, ancestorLevels = 4, maxTextLen = 400 } = opts;
  for (const loc of [
    page.getByRole('button', { name: dismissLabel }).first(),
    page.getByText(dismissLabel).first(),
  ]) {
    if (await loc.isVisible({ timeout: 1_000 }).catch(() => false)) {
      await loc.click().catch(() => {});
      await page.waitForTimeout(400);
    }
  }
  await page
    .evaluate(
      ([src, levels, maxLen]) => {
        const re = new RegExp(src, 'i');
        for (const el of Array.from(document.querySelectorAll('body *'))) {
          const t = (el as HTMLElement).innerText || '';
          if (re.test(t) && t.length < (maxLen as number)) {
            let n: HTMLElement | null = el as HTMLElement;
            for (let i = 0; i < (levels as number) && n; i++) n = n.parentElement;
            (n || (el as HTMLElement)).style.display = 'none';
          }
        }
      },
      [contains.source, ancestorLevels, maxTextLen] as [string, number, number],
    )
    .catch(() => {});
}

// ── name-collision clearing (recording against a possibly-polluted account) ───

/**
 * Repeatedly accept a "Use suggested name" prompt until none remains. Each
 * accept can itself collide, so loop. Returns after the suggestion is gone.
 */
export async function clearNameCollisions(
  page: Page,
  opts: { suggestion?: RegExp; maxRounds?: number; settleMs?: number } = {},
): Promise<void> {
  const { suggestion = /Use suggested name:/i, maxRounds = 8, settleMs = 600 } = opts;
  for (let i = 0; i < maxRounds; i++) {
    const s = page.getByText(suggestion).first();
    if (await s.isVisible({ timeout: 1_500 }).catch(() => false)) {
      await s.click();
      await page.waitForTimeout(settleMs);
    } else {
      return;
    }
  }
}

/**
 * Click a submit CTA and confirm the step ADVANCED (a marker locator becomes
 * hidden) — do NOT assert on a permanent stepper label, which false-positives
 * while a slow action is still in flight. Retries via `beforeRetry` (e.g.
 * clearNameCollisions) if a late collision keeps the step open.
 */
export async function submitUntilAdvanced(
  page: Page,
  opts: {
    submit: Locator;
    advancedWhenHidden: Locator;
    beforeRetry?: () => Promise<void>;
    maxRetries?: number;
    timeoutMs?: number;
  },
): Promise<void> {
  const { submit, advancedWhenHidden, beforeRetry, maxRetries = 3, timeoutMs = 8_000 } = opts;
  await submit.click();
  for (let i = 0; i < maxRetries; i++) {
    if (await advancedWhenHidden.isHidden({ timeout: timeoutMs }).catch(() => false)) return;
    if (beforeRetry) await beforeRetry();
    await submit.click().catch(() => {});
  }
}

/**
 * Walk a multi-step wizard by clicking the primary advance CTA at each step
 * until `doneWhen(page)` is true (e.g. the URL reaches the editor). Labels vary
 * per step, so pass a broad `advanceLabels` regex; confirm any interstitial
 * modal via `confirmLabels`. Clicks the LAST matching (footer/primary) button.
 */
export async function advanceWizardUntil(
  page: Page,
  opts: {
    advanceLabels: RegExp;
    doneWhen: (page: Page) => boolean | Promise<boolean>;
    confirmLabels?: RegExp;
    maxSteps?: number;
    settleMs?: number;
    logPrefix?: string;
  },
): Promise<void> {
  const { advanceLabels, doneWhen, confirmLabels, maxSteps = 10, settleMs = 3_000, logPrefix } = opts;
  for (let step = 0; step < maxSteps; step++) {
    if (await doneWhen(page)) return;
    if (confirmLabels) {
      const yes = page.getByRole('button', { name: confirmLabels }).first();
      if (await yes.isVisible({ timeout: 800 }).catch(() => false)) {
        await yes.click().catch(() => {});
        await page.waitForTimeout(1_000);
      }
    }
    const cands = page.getByRole('button', { name: advanceLabels });
    const n = await cands.count().catch(() => 0);
    let clicked = false;
    for (let k = n - 1; k >= 0; k--) {
      const b = cands.nth(k);
      if ((await b.isVisible().catch(() => false)) && (await b.isEnabled().catch(() => false))) {
        const label = (await b.innerText().catch(() => '')).trim();
        await b.scrollIntoViewIfNeeded().catch(() => {});
        await b.click().catch(() => {});
        if (logPrefix) console.log(`${logPrefix} step ${step}: clicked "${label}" @ ${page.url()}`);
        clicked = true;
        break;
      }
    }
    await page.waitForTimeout(settleMs);
    if (!clicked) {
      if (logPrefix) {
        const btns = await page.locator('button:visible').allInnerTexts().catch(() => []);
        console.log(`${logPrefix} step ${step}: no advance CTA @ ${page.url()} — ${JSON.stringify(btns.filter(Boolean))}`);
      }
      return;
    }
  }
}

// ── tail via Back-nav ─────────────────────────────────────────────────────────

/**
 * When a Create/submit button advances AWAY from the "money shot" form, step
 * BACK to it and hold — clip_anchor="end" then lands on the settled filled form
 * (timing-robust) rather than the transient next step.
 */
export async function tailViaBack(
  page: Page,
  opts: { backLabel?: RegExp; settleMs?: number } = {},
): Promise<void> {
  const { backLabel = /^Back$/i, settleMs = 1_500 } = opts;
  const back = page.getByRole('button', { name: backLabel }).first();
  if (await back.isVisible({ timeout: 8_000 }).catch(() => false)) {
    await back.click().catch(() => {});
    await page.waitForTimeout(settleMs);
  }
}

// ── heavy WebGL/canvas editor helpers ─────────────────────────────────────────

/**
 * Detect a "the editor crashed" screen and recover via its reload affordance,
 * waiting the load-in out. Returns true if it recovered (so the caller can
 * re-settle / re-fit). A reload can itself re-crash, so it retries `attempts`×.
 */
export async function recoverEditorIfCrashed(
  page: Page,
  opts: { crashText?: RegExp; reloadLabel?: RegExp; loadingText?: RegExp; attempts?: number } = {},
): Promise<boolean> {
  const {
    crashText = /Something went wrong in the editor/i,
    reloadLabel = /Reload editor/i,
    loadingText = /Loading floor plan/i,
    attempts = 1,
  } = opts;
  let recovered = false;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const crash = page.getByText(crashText).first();
    if (!(await crash.isVisible({ timeout: 1_500 }).catch(() => false))) break;
    recovered = true;
    await page.getByRole('button', { name: reloadLabel }).first().click().catch(() => {});
    await page.waitForLoadState('networkidle', { timeout: NETWORK_IDLE_TIMEOUT_MS }).catch(() => {});
    await page.getByText(loadingText).first().waitFor({ state: 'hidden', timeout: 90_000 }).catch(() => {});
    await page.waitForTimeout(4_000);
  }
  return recovered;
}

/**
 * Frame a canvas/plan into view via a multi-key zoom-to-fit SEQUENCE (canvas
 * editors often lack one reliable "fit" key). Focus the canvas, press each key,
 * then recover+re-fit if the sequence tripped an editor crash.
 */
export async function fitCanvasByKeys(
  page: Page,
  opts: {
    keys: string[];
    focus?: { x: number; y: number };
    perKeyMs?: number;
    settleMs?: number;
    recover?: boolean;
    recoverKeys?: string[];
  },
): Promise<void> {
  const {
    keys,
    focus = { x: 950, y: 520 },
    perKeyMs = 700,
    settleMs = 1_500,
    recover = true,
    recoverKeys,
  } = opts;
  await page.mouse.click(focus.x, focus.y);
  await page.waitForTimeout(500);
  for (const k of keys) {
    await page.keyboard.press(k).catch(() => {});
    await page.waitForTimeout(perKeyMs);
  }
  await page.waitForTimeout(settleMs);
  if (recover && (await recoverEditorIfCrashed(page))) {
    await page.mouse.click(focus.x, focus.y);
    await page.waitForTimeout(500);
    for (const k of recoverKeys ?? keys) {
      await page.keyboard.press(k).catch(() => {});
      await page.waitForTimeout(perKeyMs);
    }
    await page.waitForTimeout(settleMs);
  }
}
