import { defineConfig, devices } from '@playwright/test';
import { defineBddConfig } from 'playwright-bdd';

const testDir = defineBddConfig({
  features: './features/**/*.feature',
  steps: './steps/**/*.ts',
  // Don't hard-fail generation when the LLM misses a step definition — generate
  // the test anyway. The scenario still runs and records its video (failing at
  // the missing step), so the demo_video pipeline keeps its footage instead of
  // losing the whole run to a bddgen error.
  missingSteps: '{{MISSING_STEPS}}',
});

export default defineConfig({
  testDir,
  outputDir: 'test-results',
  // Fail-fast budgets. A step whose selector never matches the live DOM must
  // fail in seconds, not pin the default 30s per-test timeout — a suite of such
  // steps otherwise overruns the frontend_test_runner wall budget and the run is
  // killed before any video clip is harvested. Each knob is env-overridable so
  // the runner can tune per job. (A failing scenario still records its video, so
  // failing fast still yields footage to mux.)
  timeout: Number(process.env.PW_TEST_TIMEOUT_MS) || 20_000,
  expect: { timeout: Number(process.env.PW_EXPECT_TIMEOUT_MS) || 5_000 },
  use: {
    baseURL: '{{BASE_URL}}',
    actionTimeout: Number(process.env.PW_ACTION_TIMEOUT_MS) || 10_000,
    navigationTimeout: Number(process.env.PW_NAVIGATION_TIMEOUT_MS) || 15_000,
    screenshot: 'only-on-failure',
    trace: 'on-first-retry',
    // Record a video per test (scenario) — the frontend_test_runner harvests
    // these per-scenario clips for the demo_video pipeline. 'on' (not
    // 'retain-on-failure') so footage survives even when a scenario fails fast.
    // Record at full HD: a bare `video: 'on'` has NO explicit size, so Playwright
    // scales the viewport down to fit inside 800x800 — a 1280x720 Desktop Chrome
    // viewport records at 800x450 and the 1920x1080 compiler then upscales it
    // 2.4x, producing visibly soft footage. Pin both size and viewport.
    video: { mode: 'on', size: { width: 1920, height: 1080 } },
    viewport: { width: 1920, height: 1080 },
  },
  projects: [
    // The device presets carry their own 1280x720 viewport, so the 1920x1080
    // override MUST come AFTER the spread or it is clobbered back to 720p.
    {
      name: 'chromium',
      // 2x CAPTURE. Pinning `video.size` alone does NOT buy resolution: Playwright
      // never UPSCALES into the recorded frame — it draws the page picture at 1:1
      // in the top-left and pads the rest grey (MEASURED: a 1280x720 viewport
      // recorded at 2560x1440 puts a white region of exactly 1278x718 in the
      // frame). And `deviceScaleFactor` on the CONTEXT is ignored by the video
      // recorder — a dsf-2 capture is byte-for-byte an dsf-1 capture (edge energy
      // 902435 vs 903078, 0.07% apart). So the ONLY lever that puts real pixels
      // in the clip is the BROWSER-level flag below, which makes Chromium
      // rasterize the 1280x720 layout viewport into a 2560x1440 surface.
      // MEASURED on TodoMVC: the app's card is 548px wide in today's 1920x1080
      // clip and 1098px wide with this flag — the compiler's punch-in then crops
      // real pixels instead of upscaling ~550px of source to a 2560x1440 master.
      // Chromium-only, which is why it is scoped to THIS project and not to the
      // top-level `use`: firefox gets the unchanged full-HD path above.
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        launchOptions: { args: ['--force-device-scale-factor=2'] },
        video: { mode: 'on', size: { width: 2560, height: 1440 } },
      },
    },
    {
      // EXPLICIT video size on the project, not inherited from the top-level `use`:
      // every project must state the size it records at, so none can fall through
      // to the unsized 800x800 path.
      //
      // FIREFOX STAYS AT 1x, at the delivery resolution. MEASURED 2026-09-12
      // (Playwright 1.50.0, firefox-1490), Laplacian variance of a real recorded
      // frame at the compiler's 2560x1440 master size:
      //   1280x720 @1x upscaled ................................ 27287
      //   1920x1080 @1x upscaled ............................... 34427  <- this
      //   firefoxUserPrefs devPixelsPerPx=2 .................... 11798  REGRESSION
      //   context deviceScaleFactor: 2 ......................... NO-OP
      // The Gecko pref never reaches layout (the page still reads
      // devicePixelRatio 1) and yields a blurry upscale with LESS detail than a
      // plain upscale of the 1x capture, at 4x the bitrate. `deviceScaleFactor: 2`
      // DOES take (devicePixelRatio reads 2, innerWidth stays 1280) but the video
      // recorder ignores it: the 2560x1440 frame holds the 1x picture 1:1 in the
      // top-left and 75% grey padding (that crop diffs against the 1x control at
      // 0.31/255). Firefox cannot do 2x capture in this version — do not retry
      // either lever. See `_FIREFOX_CAPTURE` in config_generator.py.
      name: 'firefox',
      use: {
        ...devices['Desktop Firefox'],
        viewport: { width: 1920, height: 1080 },
        video: { mode: 'on', size: { width: 1920, height: 1080 } },
      },
    },
  ],
  reporter: [['html'], ['json', { outputFile: 'test-results.json' }]],
});
