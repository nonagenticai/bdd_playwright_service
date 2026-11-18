import { test as setup, expect } from '@playwright/test';

// Authenticates once and saves the storageState the `chromium`/`mobile-*`
// projects depend on, so every scenario starts already signed in.
//
// frontend_test_runner writes a .env (mode 0600) from the run's auth_config, so
// these are populated at load time — credentials never appear in the package:
//   BASE_URL         the app root (used for the already-authenticated bypass)
//   AUTH_LOGIN_URL   the app login route (falls back to the generated default)
//   AUTH_USERNAME    the account
//   AUTH_PASSWORD    injected at submit time
// Optional overrides (for SSO / non-default success routes):
//   AUTH_SSO_BUTTON  regex for an IdP redirect button (e.g. "Sign in with …")
//   AUTH_SUCCESS_URL regex the URL must reach after login (default: not /login)

const AUTH_FILE = '.auth/user.json';

// Selectors that identify a login screen — reused by the bypass probe and the
// form fill so "are we on login?" and "fill the username" agree.
const USERNAME_SELECTOR =
  'input[name="email"], input[name="username"], input[type="email"], ' +
  'input[id*="email"], input[id*="user"], #username';

// URL shapes that mean "this is a login / IdP screen" (Keycloak, generic SSO).
const LOGIN_URL_RE = /login|sign-?in|\bauth\b|realms\/|openid|\/kc(?:[/-]|$)/i;

setup('authenticate', async ({ page }) => {
  const loginUrl = process.env.AUTH_LOGIN_URL || '{{LOGIN_URL}}';
  const baseUrl = process.env.BASE_URL || loginUrl;
  const username = process.env.AUTH_USERNAME || '';
  const password = process.env.AUTH_PASSWORD || '';
  const ssoButton = process.env.AUTH_SSO_BUTTON;
  const successUrl = process.env.AUTH_SUCCESS_URL;

  const usernameInput = page.locator(USERNAME_SELECTOR).first();

  // ── Already-authenticated bypass ─────────────────────────────────────────
  // Go to the app first. If it does NOT bounce us to a login/IdP screen — a
  // public target, a still-valid session cookie, or an injected storageState —
  // there is nothing to log into: save the current state and skip the form.
  // This is the most-repeated hand-patch when a recording target isn't gated.
  await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForLoadState('networkidle').catch(() => {});
  const bounced =
    LOGIN_URL_RE.test(page.url()) ||
    (await usernameInput.isVisible({ timeout: 3_000 }).catch(() => false));
  if (!bounced) {
    await page.context().storageState({ path: AUTH_FILE });
    return;
  }

  // ── Login required ───────────────────────────────────────────────────────
  // Land on the login route unless the app already redirected us to a form.
  if (!(await usernameInput.isVisible().catch(() => false))) {
    await page.goto(loginUrl, { waitUntil: 'domcontentloaded' });
  }

  // Hydration-tolerant SSO handoff: many SPAs render a login CARD that redirects
  // to an IdP (Keycloak) rather than an inline form. If the username field isn't
  // present yet AND an SSO button is configured, click it with a generous
  // auto-wait (the SvelteKit splash can delay hydration 10-20s). If the app
  // auto-redirects straight to the IdP, the field is already there and we skip.
  if (ssoButton && !(await usernameInput.isVisible().catch(() => false))) {
    await page
      .getByRole('button', { name: new RegExp(ssoButton, 'i') })
      .click({ timeout: 60_000 })
      .catch(() => {});
  }

  await usernameInput.waitFor({ state: 'visible', timeout: 60_000 });
  await usernameInput.fill(username);

  const passwordInput = page
    .locator('input[name="password"], input[type="password"], input[id*="pass"], #password')
    .first();
  await passwordInput.fill(password);

  const submitBtn = page
    .locator(
      'button[type="submit"], #kc-login, button:has-text("Sign in"), ' +
        'button:has-text("Log in"), button:has-text("Login"), input[type="submit"]',
    )
    .first();

  const successRe = successUrl ? new RegExp(successUrl, 'i') : null;
  await Promise.all([
    successRe
      ? page.waitForURL(successRe, { timeout: 60_000 })
      : page.waitForURL((url) => !url.pathname.includes('login'), { timeout: 60_000 }),
    submitBtn.click(),
  ]);

  if (successRe) await expect(page).toHaveURL(successRe, { timeout: 30_000 });
  await page.context().storageState({ path: AUTH_FILE });
});
