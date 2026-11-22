// Grounded QA step vocabulary — the ONLY step definitions a grounded QA package
// ships. Emitted verbatim by bdd_playwright_service (never LLM-authored), compiled
// once in CI (bddgen + tsc), so what the generator emits is what was tested.
//
// The registry in src/services/generation/qa_vocabulary.py is the source of truth;
// a unit test asserts this file and that registry describe the same steps. When you
// add a step here, add it there (and to the deterministic builder only if it can be
// grounded in the page snapshot).
//
// ⚠️ The patterns carry {string} UNQUOTED. {string} already matches the surrounding
// quotes in a cucumber expression; `"{string}"` asks for a literal quote as well and
// can never match — the exact defect that shipped 19 dead step patterns.
import { expect } from '@playwright/test';
import { createBdd } from 'playwright-bdd';

const { Given, When, Then } = createBdd();

/** baseURL + a site-relative path, keeping any base path the app is mounted under. */
function appUrl(baseURL: string | undefined, path: string): string {
  if (!baseURL) {
    throw new Error('baseURL is not configured; run with BASE_URL=<app origin>');
  }
  const root = baseURL.endsWith('/') ? baseURL : `${baseURL}/`;
  return new URL(path.replace(/^\/+/, ''), root).toString();
}

/** The pathname a browser is on, independent of query/fragment. */
function currentPath(pageUrl: string): string {
  try {
    return new URL(pageUrl).pathname;
  } catch {
    return pageUrl;
  }
}

const normalise = (value: string): string => value.replace(/\s+/g, ' ').trim();

Given('the user opens the page {string}', async ({ page, baseURL }, path: string) => {
  await page.goto(appUrl(baseURL, path), { waitUntil: 'domcontentloaded' });
});

When('the user clicks the link {string}', async ({ page }, name: string) => {
  await page.getByRole('link', { name, exact: true }).first().click();
  await page.waitForLoadState('domcontentloaded');
});

Then('the page URL path is {string}', async ({ page }, path: string) => {
  const expected = path.startsWith('/') ? path : `/${path}`;
  await expect.poll(() => currentPath(page.url())).toBe(expected);
});

Then('the page shows the heading {string}', async ({ page }, name: string) => {
  await expect(page.getByRole('heading', { name, exact: true }).first()).toBeVisible();
});

Then('the page title contains {string}', async ({ page }, value: string) => {
  await expect
    .poll(async () => normalise(await page.title()))
    .toContain(normalise(value));
});

Then('the page contains the link {string}', async ({ page }, name: string) => {
  await expect(page.getByRole('link', { name, exact: true }).first()).toBeAttached();
});

Then('the page contains the button {string}', async ({ page }, name: string) => {
  await expect(page.getByRole('button', { name, exact: true }).first()).toBeAttached();
});

Then('the page contains the text {string}', async ({ page }, value: string) => {
  await expect(page.getByText(normalise(value), { exact: false }).first()).toBeAttached();
});
