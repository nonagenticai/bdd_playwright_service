import { type Page } from '@playwright/test';
import { createBdd } from 'playwright-bdd';
import { heroDwell, resilientClick, NETWORK_IDLE_TIMEOUT_MS } from '../lib/recording_helpers';

const { When, Then } = createBdd();

// Deterministic showcase-tour steps — the fixed step library that a knowledge
// pack's screens[] drives (see feature_generator.build_showcase_features). Each
// scenario is its OWN @clip-N clip (one video per Playwright test): open a
// route, wait past the SPA boot for the screen's content, optionally interact,
// then hold a long settled hero tail (compile shows the clip via
// clip_anchor="end", so the stable TAIL is what plays). No LLM invents these
// steps; a screen edited in the YAML re-emits the feature that calls them.

// BUDGET COUPLING (do not raise blindly): the orchestrator sends
// `timeout_per_test_ms` (default 20000) and frontend_test_runner applies it as the
// Playwright `--timeout`, which OVERRIDES this package's own config and is exported
// to the steps as PW_TEST_TIMEOUT_MS. A wait LONGER than that timeout is unreachable
// dead code: Playwright kills the test mid-step, the hero tail never records and the
// clip comes out black (OpenCV sharpness 0.0 -> demo_video_record blank-gate
// hard-fails the paid render). So derive BOTH waits from that budget.
//
// The invariant actually guaranteed here (an earlier version's comment claimed a
// ceiling that a 1000ms floor silently outranked): SETTLE <= 0.1*B and
// CONTENT <= 0.5*B - SETTLE, so CONTENT + SETTLE <= 0.5*B whenever the 250ms floor
// does not bind. The floor binds only for B < ~625ms, where CONTENT + SETTLE <= 312
// -- still under B. So the two waits together can never reach the per-test budget at
// any B, leaving the goto and networkidle their own half. FTR accepts a
// timeout_per_test_ms as low as 1000, so this must hold at the small end too.
const PW_TEST_TIMEOUT_MS = Number(process.env.PW_TEST_TIMEOUT_MS) || 20000;
const SETTLE_MS = Math.min(1_500, Math.floor(PW_TEST_TIMEOUT_MS * 0.1));
const CONTENT_WAIT_MS = Math.max(250, Math.floor(PW_TEST_TIMEOUT_MS * 0.5) - SETTLE_MS);

// Resolve a screen's `route` against the baseURL fixture WITHOUT dropping
// baseURL's own sub-path. A bare `page.goto('/build')` resolves against the
// ORIGIN, so a baseURL like https://host/app silently loses `/app` and the clip
// records the wrong (usually blank) page. `new URL(route, baseURL)` has the same
// defect for a leading-slash route — concatenate instead. An already-absolute
// route (a full URL) is passed through untouched.
function absoluteNavUrl(baseURL: string | undefined, route: string): string {
  if (/^(?:[A-Za-z][A-Za-z0-9+.-]*:|\/\/)/.test(route)) return route;
  const base = (baseURL ?? '').replace(/\/+$/, '');
  if (!base) return route;
  const rel = route.replace(/^\/+/, '');
  return rel ? `${base}/${rel}` : base;
}

// Navigate to a route, wait past the SPA boot/data load, and confirm the real
// content container has rendered (not just the app shell) before the hero tail.
async function open(
  page: Page,
  baseURL: string | undefined,
  route: string,
  waitFor: string,
): Promise<void> {
  await page.goto(absoluteNavUrl(baseURL, route), { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('networkidle', { timeout: NETWORK_IDLE_TIMEOUT_MS }).catch(() => {});
  const sel = waitFor && waitFor.trim() ? waitFor : 'body';
  await page.locator(sel).first().waitFor({ state: 'visible', timeout: CONTENT_WAIT_MS });
  await page.waitForTimeout(SETTLE_MS); // let the content settle before the tail
}

When(
  'I open {string} and wait for {string}',
  async ({ page, baseURL }, route: string, waitFor: string) => {
    await open(page, baseURL, route, waitFor);
  },
);

When('I click {string}', async ({ page }, selector: string) => {
  // Resilient: a stale selector soft-fails (keeps the clip non-blank) instead
  // of throwing and aborting the whole scenario's recording.
  await resilientClick(page, selector);
});

When('I scroll to {string}', async ({ page }, selector: string) => {
  await page.locator(selector).first().scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
});

Then('I dwell for {int} ms', async ({ page }, ms: number) => {
  await heroDwell(page, ms);
});
