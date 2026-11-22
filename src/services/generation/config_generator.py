"""Deterministic config file generation for BDD projects (no LLM)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates" / "scaffold"

# Recording resolution for desktop packages. A bare ``video: 'on'`` has no explicit
# size, so Playwright scales the viewport down to fit inside 800x800 — a 1280x720
# Desktop Chrome viewport records at 800x450, which the 1920x1080 compiler then
# upscales 2.4x into visibly soft footage. Both the video size AND the viewport must
# be pinned, and the viewport override must be emitted AFTER the ``...devices[...]``
# spread (the presets carry their own 1280x720 viewport and would clobber it).
_HD_WIDTH = 1920
_HD_HEIGHT = 1080
_HD_VIEWPORT = f"viewport: {{ width: {_HD_WIDTH}, height: {_HD_HEIGHT} }}"
_HD_VIDEO = f"video: {{ mode: 'on', size: {{ width: {_HD_WIDTH}, height: {_HD_HEIGHT} }} }}"

# 2x CAPTURE (chromium only). Pinning ``video.size`` above buys a 1080p CANVAS, not
# 1080p of CONTENT: Playwright never UPSCALES into the recorded frame — it draws the
# page picture at 1:1 in the top-left and pads the rest grey (MEASURED: a 1280x720
# viewport recorded at 2560x1440 yields a white region of exactly 1278x718). And
# ``deviceScaleFactor`` on the CONTEXT is ignored by the video recorder: a dsf-2
# capture is byte-for-byte a dsf-1 capture (sobel edge energy 902435 vs 903078,
# 0.07% apart). So the only lever that puts REAL pixels in a clip is the
# BROWSER-level flag, which rasterizes the CSS viewport into a 2x surface.
#
# MEASURED on TodoMVC (the demo_video e2e target): the app's card is 548px wide in
# today's 1920x1080 clip and 1098px wide with this flag. The compiler masters at
# 2560x1440, so today it upscales ~550px of real source; with the flag its punch-in
# crops real pixels instead.
#
# The flag is Chromium-only, so it is scoped to the chromium PROJECT rather than the
# top-level ``use`` — firefox keeps the unchanged full-HD path and never sees a
# Chromium argument. The CSS viewport drops to 1280x720 (the desktop standard) so the
# 2x surface lands exactly on the 2560x1440 master instead of overshooting it.
_CAPTURE_CSS_WIDTH = 1280
_CAPTURE_CSS_HEIGHT = 720
_CAPTURE_SCALE = 2
_CAPTURE_VIEWPORT = f"viewport: {{ width: {_CAPTURE_CSS_WIDTH}, height: {_CAPTURE_CSS_HEIGHT} }}"
_CAPTURE_LAUNCH = f"launchOptions: {{ args: ['--force-device-scale-factor={_CAPTURE_SCALE}'] }}"
_CAPTURE_VIDEO = (
    f"video: {{ mode: 'on', size: {{ width: {_CAPTURE_CSS_WIDTH * _CAPTURE_SCALE}, "
    f"height: {_CAPTURE_CSS_HEIGHT * _CAPTURE_SCALE} }} }}"
)
#: Everything the chromium project overrides, emitted AFTER the device spread.
_CHROMIUM_CAPTURE = f"{_CAPTURE_VIEWPORT}, {_CAPTURE_LAUNCH}, {_CAPTURE_VIDEO}"

# FIREFOX STAYS AT 1x — 1920x1080 viewport recorded at 1920x1080, matching the
# delivery target. MEASURED 2026-09-12 on Playwright 1.50.0 with the installed
# firefox-1490 build (one revision off the 1490/1471 pin; the failure is a
# Playwright recorder property, not a Gecko build detail). Frame 45 of a real
# recording of a 1px black/white stripe pattern, Laplacian variance at the
# compiler's 2560x1440 master size:
#
#   1280x720 @1x, lanczos-upscaled to 2560x1440 ......... 27287   (baseline)
#   1920x1080 @1x, lanczos-upscaled to 2560x1440 ........ 34427   <- SHIPPED
#   launchOptions.firefoxUserPrefs devPixelsPerPx=2 ..... 11798   REGRESSION
#   context deviceScaleFactor: 2 ........................ (see below) NO-OP
#
# BOTH 2x approaches fail, for DIFFERENT reasons — do not retry either:
#
# * ``firefoxUserPrefs: {'layout.css.devPixelsPerPx': '2'}`` is a NO-OP on the
#   PAGE: the recorded page still reports ``innerWidth`` 1280 and
#   ``devicePixelRatio`` **1** — Playwright's context ``deviceScaleFactor``
#   (default 1) overrides the Gecko pref, so layout never goes 2x. The 2560x1440
#   frame is filled edge to edge, but it is a BLURRY UPSCALE carrying LESS detail
#   than a plain lanczos upscale of the 1x capture (11798 vs 27287) at 4x the
#   bitrate. It is a regression dressed as an improvement.
# * ``deviceScaleFactor: 2`` on the CONTEXT *does* move the page —
#   ``devicePixelRatio`` reads **2** while ``innerWidth`` stays 1280 — but the
#   VIDEO RECORDER ignores it: the 2560x1440 frame contains the 1x picture drawn
#   1:1 in the top-left 1280x720 and 75% grey padding. Proven by cropping that
#   corner and diffing it against the 1x control: mean absolute difference 0.31/255,
#   Laplacian variance 217893 vs 217595. Its whole-frame variance (54386) looks
#   like a win ONLY because flat grey is 75% of the frame — an area artifact, not
#   detail. Combining both levers is worst of all (11094).
#
# So Firefox cannot do 2x capture in this Playwright version. A clean, honest 1x at
# the delivery resolution beats a fake 2x: 1920x1080 measured 34427 at master size,
# ahead of every other option including the 1280x720 baseline.
#
# (``avg_frame_rate 0/0`` in ffprobe is NOT a symptom of any of this — the 1x
# control and the 1920x1080 clip report it too. It is how Playwright writes its
# webm container.)
_FIREFOX_CAPTURE = f"{_HD_VIEWPORT}, {_HD_VIDEO}"

# WebKit (``mobile-safari``) has NO equivalent and deliberately stays at 1:1: there is
# no pref channel like ``firefoxUserPrefs`` for it in Playwright, and #157 measured
# that ``deviceScaleFactor`` on the CONTEXT is ignored by the video recorder (a dsf-2
# capture was byte-for-byte a dsf-1 capture). Asking WebKit for 2x would therefore buy
# a larger frame holding the same pixels plus grey padding — strictly worse.

# EVERY project records at an EXPLICIT size — not just chromium. A project that
# carries no `video.size` inherits one, and where the inherited value is a bare
# `video: 'on'` (the mobile path, until now) Playwright supplies no size at all and
# scales the viewport down to fit inside 800x800. A Pixel 5's 393x851 viewport then
# records at 369x800 and the compiler upscales that into a 2560x1440 master.
#
# ⚠️ The chromium 2x path is NOT copied onto the MOBILE projects, and that is
# deliberate (and firefox has NO working 2x lever at all — see _FIREFOX_CAPTURE):
#
# * ``--force-device-scale-factor`` is a CHROMIUM launch argument. `mobile-safari`
#   is WebKit and `firefox` is Gecko; neither accepts it.
# * The mobile presets carry their OWN ``deviceScaleFactor`` (Pixel 5 = 2.75,
#   iPhone 13 = 3) as part of the emulation. Forcing a browser-level factor of 2
#   over that does not sharpen the clip — it fights the emulation the project
#   exists to perform.
# * #157 MEASURED that a size ABOVE what the browser actually rasterizes is not
#   upscaled into the frame: Playwright draws the picture 1:1 in the top-left and
#   pads the rest grey. So asking for `css x deviceScaleFactor` on a project with
#   no proven browser-level 2x surface would produce a LARGER frame containing the
#   SAME pixels plus grey padding — strictly worse than today.
#
# The sound, measurement-consistent size for a mobile project is therefore
# its own viewport, 1:1: it removes the 800x800 downscale without claiming
# resolution the browser is not rasterizing. Emitted as a helper so the size is
# read from the device descriptor rather than restated as a literal that goes
# stale when Playwright revises a preset.
_VIDEO_AT_HELPER = (
    "// Record each project at its OWN viewport, 1:1. A bare `video: 'on'` has NO\n"
    "// size, so Playwright scales the viewport down to fit inside 800x800 (a Pixel 5\n"
    "// records 369x800) and the compiler then upscales that into its master.\n"
    "const videoAt = (d: { viewport: { width: number; height: number } }) => ({\n"
    "  mode: 'on' as const,\n"
    "  size: { width: d.viewport.width, height: d.viewport.height },\n"
    "});"
)

# Per-test budget for the `setup` project ONLY. `auth.setup.ts` waits up to 60s
# for the SSO handoff and for the username field to appear after hydration, then
# up to 30s asserting the post-login URL — so it needs a ceiling above the sum of
# its own longest legs, not the fail-fast 20s scenario budget. This is one login
# per run, not a per-scenario cost.
_AUTH_SETUP_TIMEOUT_MS = 120_000


class ConfigGenerator:
    """Generates project configuration files for playwright-bdd projects."""

    def generate_playwright_config(
        self,
        base_url: str,
        platform: str = "web",
        *,
        has_auth: bool = False,
        missing_steps: str = "fail-on-run",
    ) -> str:
        """Generate a playwright.config.ts file.

        Args:
            base_url: The application base URL for test runs.
            platform: Target platform (web or mobile).
            has_auth: Whether authentication setup project should be included.
            missing_steps: playwright-bdd ``missingSteps`` mode. The video streams
                (demo_video / documentation) use ``'fail-on-run'`` so a missing
                step still yields a runnable, recordable suite; the strict QA
                streams pass ``'fail-on-gen'`` to surface gaps at generation time.

        Returns:
            TypeScript config file content as a string.
        """
        # The scaffold template is the DESKTOP config, byte for byte — it carries no
        # ``{{PLATFORM}}`` token and hardcodes the chromium + firefox projects. So it
        # can only serve a desktop package: a mobile non-auth package taking this
        # branch silently got desktop projects and never emulated a phone at all.
        # Mobile falls through to the builder below, which honours ``platform``.
        if not has_auth and platform != "mobile":
            template_path = _TEMPLATES_DIR / "playwright.config.ts"
            if template_path.exists():
                template = template_path.read_text(encoding="utf-8")
                content = template.replace("{{BASE_URL}}", base_url)
                content = content.replace("{{PLATFORM}}", platform)
                return content.replace("{{MISSING_STEPS}}", missing_steps)

        # Build config with optional auth support
        projects = self._get_projects(platform, has_auth=has_auth)
        storage_state_line = "    storageState: '.auth/user.json',\n" if has_auth else ""
        # Mobile projects emulate a phone (Pixel 5 / iPhone 13) — forcing a desktop
        # viewport there would defeat the emulation, so full HD is desktop-only.
        # The mobile DEFAULT is still explicitly sized (off the Pixel 5 preset) so
        # that no project — including `setup`, which declares no `use` on mobile —
        # can fall through to the unsized 800x800 path. Each mobile project
        # overrides it with its own device's size below.
        video_block = (
            f"    {_HD_VIDEO},\n    {_HD_VIEWPORT},"
            if platform != "mobile"
            else "    video: videoAt(devices['Pixel 5']),"
        )
        return f"""import {{ defineConfig, devices }} from '@playwright/test';
import {{ defineBddConfig }} from 'playwright-bdd';

const testDir = defineBddConfig({{
  features: './features/**/*.feature',
  steps: './steps/**/*.ts',
  // Generate tests even when a step definition is missing (scenario still runs +
  // records video, failing at the missing step) — keep the demo's footage rather
  // than losing the whole run to a bddgen error. Keep in sync with the template.
  missingSteps: '{missing_steps}',
}});

{_VIDEO_AT_HELPER}

export default defineConfig({{
  testDir,
  outputDir: 'test-results',
  // Fail-fast budgets — keep in sync with templates/scaffold/playwright.config.ts.
  // A step whose selector never matches must fail in seconds, not pin the 30s
  // default, so the suite finishes within the frontend_test_runner wall budget
  // and its video clips are harvested. Each knob is env-overridable per job.
  timeout: Number(process.env.PW_TEST_TIMEOUT_MS) || 20_000,
  expect: {{ timeout: Number(process.env.PW_EXPECT_TIMEOUT_MS) || 5_000 }},
  use: {{
    baseURL: process.env.BASE_URL || '{base_url}',
{storage_state_line}    actionTimeout: Number(process.env.PW_ACTION_TIMEOUT_MS) || 10_000,
    navigationTimeout: Number(process.env.PW_NAVIGATION_TIMEOUT_MS) || 15_000,
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
    // Record a video per test (scenario) — the frontend_test_runner harvests
    // these per-scenario clips for the demo_video pipeline. Record at full HD:
    // a bare `video: 'on'` has no explicit size, so Playwright scales the viewport
    // down to fit inside 800x800 (1280x720 -> 800x450) and the 1920x1080 compiler
    // upscales it 2.4x into soft footage. Keep in sync with the template.
{video_block}
  }},
  projects: {projects},
  reporter: [['html'], ['json', {{ outputFile: 'test-results.json' }}]],
}});
"""

    def generate_auth_setup(self, login_url: str) -> str:
        """Generate an auth.setup.ts file for Playwright storageState auth.

        The generated setup reads AUTH_LOGIN_URL, AUTH_USERNAME, AUTH_PASSWORD
        from environment variables so credentials are never hardcoded.

        Args:
            login_url: Default login URL (overridable via AUTH_LOGIN_URL env var).

        Returns:
            TypeScript setup file content as a string.
        """
        template_path = _TEMPLATES_DIR / "auth.setup.ts"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8").replace("{{LOGIN_URL}}", login_url)

        # Inline fallback (kept in sync with templates/scaffold/auth.setup.ts).
        return f"""import {{ test as setup }} from '@playwright/test';

const authFile = '.auth/user.json';

setup('authenticate', async ({{ page }}) => {{
  const loginUrl = process.env.AUTH_LOGIN_URL || '{login_url}';
  const baseUrl = process.env.BASE_URL || loginUrl;
  const username = process.env.AUTH_USERNAME || '';
  const password = process.env.AUTH_PASSWORD || '';

  const usernameInput = page.locator(
    'input[name="email"], input[name="username"], input[type="email"], ' +
    'input[id*="email"], input[id*="user"]'
  ).first();

  // Already-authenticated bypass: if the app root doesn't bounce us to a login
  // screen (public target / valid session / injected state), skip the form.
  await page.goto(baseUrl);
  await page.waitForLoadState('networkidle').catch(() => {{}});
  const onLogin =
    /login|sign-?in|realms\\//i.test(page.url()) ||
    (await usernameInput.isVisible({{ timeout: 3000 }}).catch(() => false));
  if (!onLogin) {{
    await page.context().storageState({{ path: authFile }});
    return;
  }}

  if (!(await usernameInput.isVisible().catch(() => false))) await page.goto(loginUrl);

  // Fill login form — tries common selectors in priority order
  const passwordInput = page.locator(
    'input[name="password"], input[type="password"], input[id*="pass"]'
  ).first();

  await usernameInput.fill(username);
  await passwordInput.fill(password);

  // Click the submit button
  const submitBtn = page.locator(
    'button[type="submit"], button:has-text("Sign in"), ' +
    'button:has-text("Log in"), button:has-text("Login"), ' +
    'input[type="submit"]'
  ).first();
  await submitBtn.click();

  // Wait for navigation away from login page
  await page.waitForURL((url) => !url.pathname.includes('login'), {{
    timeout: 15000,
  }});

  // Persist authenticated state
  await page.context().storageState({{ path: authFile }});
}});
"""

    def generate_package_json(self) -> str:
        """Generate a package.json file with playwright-bdd dependencies.

        Returns:
            JSON content as a string.
        """
        template_path = _TEMPLATES_DIR / "package.json"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")

        # Inline fallback
        pkg = {
            "name": "bdd-test-suite",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "test": "npx bddgen && npx playwright test",
                "test:headed": "npx bddgen && npx playwright test --headed",
                "report": "npx playwright show-report",
            },
            "devDependencies": {
                "@playwright/test": "1.50.0",
                "playwright-bdd": "^8.0.0",
                "typescript": "^5.7.0",
            },
            # Pin the transitive @cucumber/cucumber-expressions so `npm install`
            # (no lockfile on generated packages) cannot float it to a version
            # incompatible with playwright-bdd's bddgen. Mirrors the pin the
            # hand-authored recording-package carries via its committed lockfile.
            "overrides": {
                "@cucumber/cucumber-expressions": "18.0.1",
            },
        }
        return json.dumps(pkg, indent=2) + "\n"

    def read_scaffold_lib(self) -> dict[str, str]:
        """Return the shared recording-helper library files to ship in a package.

        Recording packages (the demo_video stream + hand-authored packages) import
        these app-agnostic step helpers (draft-chaining, coach-mark dismissal,
        editor-crash recovery, zoom-to-fit, tail-via-Back-nav, fixture reset) from
        ``lib/`` rather than re-deriving them. Returns ``{relpath: content}`` mapping
        each scaffold ``lib/*`` file to its package-relative path (``lib/<name>``).
        Empty if the scaffold has no ``lib/`` dir (inline-only build).
        """
        lib_dir = _TEMPLATES_DIR / "lib"
        if not lib_dir.is_dir():
            return {}
        out: dict[str, str] = {}
        for f in sorted(lib_dir.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                out[f"lib/{f.name}"] = f.read_text(encoding="utf-8")
        return out

    def read_showcase_steps(self) -> list[dict[str, Any]]:
        """Return the fixed deterministic showcase step-definition file.

        A knowledge-pack-driven showcase package does NOT ask the LLM to invent
        step definitions — the feature only uses the fixed step vocabulary
        implemented once in ``templates/scaffold/steps/showcase.steps.ts``.
        Returns the standard step-def dict shape (``filename``, ``content``,
        ``step_count``); empty if the scaffold file is missing.
        """
        steps_path = _TEMPLATES_DIR / "steps" / "showcase.steps.ts"
        if not steps_path.is_file():
            return []
        content = steps_path.read_text(encoding="utf-8")
        return [{"filename": "showcase.steps.ts", "content": content, "step_count": 4}]

    def read_qa_steps(self) -> list[dict[str, Any]]:
        """Return the STATIC grounded-QA step-definition file.

        A grounded QA package never ships LLM-authored TypeScript: the whole file
        is the closed vocabulary in ``templates/scaffold/qa/steps.ts``, compiled once
        in CI (bddgen + tsc). A missing template is a build error — returning ``[]``
        would ship a suite whose every step is undefined.
        """
        steps_path = _TEMPLATES_DIR / "qa" / "steps.ts"
        if not steps_path.is_file():
            raise FileNotFoundError(f"grounded QA step template is missing: {steps_path}")
        content = steps_path.read_text(encoding="utf-8")
        step_calls = len(re.findall(r"^(?:Given|When|Then)\(", content, re.MULTILINE))
        return [{"filename": "qa.steps.ts", "content": content, "step_count": step_calls}]

    def generate_qa_config(self, base_url: str) -> str:
        """Generate the QA-suite ``playwright.config.ts`` (video off, one browser).

        Deliberately NOT the demo-video scaffold: that one records 1080p video on
        chromium and firefox because footage is its deliverable, and reusing it for a
        QA suite is what made a 14-scenario run hang for an hour.
        """
        template_path = _TEMPLATES_DIR / "qa" / "playwright.config.ts"
        if not template_path.is_file():
            raise FileNotFoundError(f"grounded QA config template is missing: {template_path}")
        template = template_path.read_text(encoding="utf-8")
        return template.replace("{{BASE_URL}}", base_url or "http://127.0.0.1:8000")

    def generate_tsconfig(self) -> str:
        """Generate a tsconfig.json file.

        Returns:
            JSON content as a string.
        """
        template_path = _TEMPLATES_DIR / "tsconfig.json"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")

        # Inline fallback
        tsconfig = {
            "compilerOptions": {
                "target": "ES2022",
                "module": "commonjs",
                "moduleResolution": "node",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "outDir": "./dist",
                "rootDir": ".",
            },
            "include": ["steps/**/*.ts", "playwright.config.ts"],
            "exclude": ["node_modules"],
        }
        return json.dumps(tsconfig, indent=2) + "\n"

    @staticmethod
    def _get_projects(platform: str, *, has_auth: bool = False) -> str:
        """Return Playwright project configuration based on platform and auth.

        Desktop projects re-assert the full-HD viewport *after* the
        ``...devices['Desktop Chrome']`` spread: the device preset ships its own
        1280x720 viewport and a later key wins in a JS object literal, so putting
        the override first would silently restore the 720p (=800x450 video) trap.

        The ``setup`` project needs BOTH of its own overrides, and each closes a
        way the whole auth'd run fails:

        * ``testDir: '.'`` — a project's ``testDir`` DEFAULTS to the top-level
          one, which here is the ``defineBddConfig`` output dir (``.features-gen``).
          ``auth.setup.ts`` is written to the PACKAGE ROOT (``job_processor``
          ``_store_artifacts``; zipped by ``package_builder``'s root ``*.ts``
          pattern), so ``testMatch`` was filtering a directory the file is not in.
          The setup project matched ZERO files, passed vacuously, never wrote
          ``.auth/user.json`` — and every dependent project, which declares
          ``storageState: '.auth/user.json'``, then died at context creation with
          ENOENT. The error names storage state, not login, so an auth'd package
          failed 100% of its scenarios and recorded no footage.
        * ``timeout`` — the top-level per-test budget is a deliberately fail-fast
          20s (``PW_TEST_TIMEOUT_MS``), sized for a scenario step whose selector
          must not pin the 30s default. ``auth.setup.ts`` is built around 60s
          waits (the SSO handoff, the username field after a SvelteKit splash
          that can take 10-20s to hydrate) plus a 30s URL assertion, so under a
          20s cap that hardening is dead code: the setup is killed mid-hydration
          on exactly the slow IdP-gated target it was written for, and the run
          fails the same ENOENT way. It is one login, not a per-scenario cost.
        """
        setup_project = ""
        deps = ""
        storage_state = ""
        if has_auth:
            # EXPLICIT on the setup project too, on BOTH platforms — it is a real
            # recorded test, and it was the one project left inheriting.
            #
            # * DESKTOP stated a viewport and no ``video`` at all, so it fell through
            #   to the top-level 1920x1080 — right by accident, and silently wrong the
            #   moment that default moves. It now states both.
            # * MOBILE stated NOTHING: no ``use`` key, so it inherited the top-level
            #   ``videoAt(devices['Pixel 5'])`` (393x851) while keeping Playwright's
            #   DEFAULT 1280x720 viewport — a 1280x720 page recorded into a 393x851
            #   frame. Spreading the Pixel 5 preset makes the viewport match the size
            #   it records at, and emulating the phone through login is what the rest
            #   of a mobile run does anyway.
            setup_use = (
                ", use: { ...devices['Pixel 5'], video: videoAt(devices['Pixel 5']) }"
                if platform == "mobile"
                else f", use: {{ {_HD_VIEWPORT}, {_HD_VIDEO} }}"
            )
            # `testDir` FIRST: auth.setup.ts lives at the package root, not under
            # the bddgen output dir the top-level `testDir` points at.
            setup_match = "testDir: '.', testMatch: /auth\\.setup\\.ts/"
            setup_timeout = f", timeout: {_AUTH_SETUP_TIMEOUT_MS}"
            setup_project = f"    {{ name: 'setup', {setup_match}{setup_timeout}{setup_use} }},\n"
            deps = "dependencies: ['setup'], "
            storage_state = "storageState: '.auth/user.json', "

        if platform == "mobile":
            # `videoAt` AFTER the device spread — the preset carries no `video`, but
            # an override placed first would still be at the mercy of any future
            # preset key, and the desktop path already learned that lesson.
            pixel = "...devices['Pixel 5'], video: videoAt(devices['Pixel 5'])"
            # `mobile-safari` stays 1:1 on purpose: WebKit has no browser-level 2x
            # lever in Playwright, and the video recorder ignores context
            # `deviceScaleFactor` (MEASURED for firefox — see _FIREFOX_CAPTURE), so
            # there is no knob that would put real 2x pixels in the frame, only grey
            # padding.
            iphone = "...devices['iPhone 13'], video: videoAt(devices['iPhone 13'])"
            return f"""[
{setup_project}    {{ name: 'mobile-chrome', {deps}use: {{ {storage_state}{pixel} }} }},
    {{ name: 'mobile-safari', {deps}use: {{ {storage_state}{iphone} }} }},
  ]"""
        return f"""[
{setup_project}    {{
      name: 'chromium',
      {deps}use: {{ {storage_state}...devices['Desktop Chrome'], {_CHROMIUM_CAPTURE} }},
    }},
    {{
      name: 'firefox',
      {deps}use: {{ {storage_state}...devices['Desktop Firefox'], {_FIREFOX_CAPTURE} }},
    }},
  ]"""
