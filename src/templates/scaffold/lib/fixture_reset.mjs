// Reusable fixture-reset utilities for app-recording packages.
//
// Before recording against a live account, demo entities with fixed names must
// be cleared so a re-run doesn't collide (duplicate-name dialogs / "(2)" name
// suffixes stall the wizard clips). The clean way to do that WITHOUT a separate
// API credential is to reuse the browser session's own bearer token: load the
// saved storageState, let the SPA make its first authenticated /api/ call, sniff
// the Authorization header off it, then drive the REST API directly.
//
// This module is app-AGNOSTIC. A per-app `_cleanup.mjs` imports these and
// supplies only the app-specific bit (which entities to delete). Canonical copy
// in bdd_playwright_service/src/templates/scaffold/lib/ — keep package copies in
// sync.

/**
 * Open an authenticated page and capture the bearer token off its first /api/
 * request. Returns { browser, context, page, authHeader } — the caller closes
 * the browser. Throws if no token was seen (auth expired / storageState stale).
 *
 * @param {import('@playwright/test').BrowserType} chromium
 * @param {object} opts
 * @param {string} opts.baseURL
 * @param {string} [opts.storageState='.auth/user.json']
 * @param {string} [opts.warmUrl='/dashboard']  page that triggers an /api/ call
 * @param {string} [opts.apiPathMatch='/api/']
 * @param {number} [opts.settleMs=3000]
 */
export async function openWithSniffedAuth(chromium, opts) {
  const {
    baseURL,
    storageState = '.auth/user.json',
    warmUrl = '/dashboard',
    apiPathMatch = '/api/',
    settleMs = 3000,
  } = opts;
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ storageState, baseURL });
  const page = await context.newPage();

  let authHeader = null;
  page.on('request', (req) => {
    if (!authHeader && req.url().includes(apiPathMatch)) {
      const h = req.headers();
      if (h['authorization']) authHeader = h['authorization'];
    }
  });
  await page.goto(warmUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.waitForTimeout(settleMs);
  if (!authHeader) {
    await browser.close();
    throw new Error('no Authorization header captured — storageState may be expired');
  }
  return { browser, context, page, authHeader };
}

/** GET `base + pathTemplate(offset)` in a paginated loop, concatenating arrays. */
export async function fetchAllPaged(page, base, headers, pathTemplate, opts = {}) {
  const { pageSize = 100, maxOffset = 1000 } = opts;
  let all = [];
  for (let off = 0; off < maxOffset; off += pageSize) {
    const r = await page.request.get(base + pathTemplate(off, pageSize), { headers });
    if (r.status() !== 200) break;
    const j = await r.json().catch(() => null);
    const pageItems = Array.isArray(j) ? j : (j && j.items) || [];
    all = all.concat(pageItems);
    if (pageItems.length < pageSize) break;
  }
  return all;
}

/** DELETE each `base + path`, logging status. Honours DRY=1 (log, don't delete). */
export async function deleteEach(page, base, headers, paths, { dry = false, label = 'DELETE' } = {}) {
  for (const p of paths) {
    if (dry) {
      console.log(`  DRY ${label}`, p);
      continue;
    }
    const r = await page.request.delete(base + p, { headers });
    console.log(`  ${label}`, p, '->', r.status());
  }
}
