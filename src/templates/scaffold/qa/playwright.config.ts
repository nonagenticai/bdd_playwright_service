// QA-suite config — NOT the demo-video config. The demo scaffold records 1080p
// video on chromium AND firefox because the clips are the deliverable; applying it
// to a QA suite is what made a 14-scenario run hang for ~60 minutes with zero
// progress. Here the suite is meant to answer "does the app still honour its page
// contracts", quickly: video off, one browser, fail-on-gen for missing steps.
import { defineConfig } from '@playwright/test';
import { defineBddConfig } from 'playwright-bdd';

const testDir = defineBddConfig({
  features: './features/**/*.feature',
  steps: './steps/*.ts',
  // Every emitted step is backed by the static vocabulary, so a missing step is a
  // generator bug: refuse to run rather than hand the caller a half-executable suite.
  missingSteps: 'fail-on-gen',
});

export default defineConfig({
  testDir,
  outputDir: 'test-results',
  fullyParallel: false,
  // Bounded per scenario: a locator that never resolves must fail in seconds, not
  // pin the run. Override per environment with PW_*.
  timeout: Number(process.env.PW_TEST_TIMEOUT_MS) || 30_000,
  expect: { timeout: Number(process.env.PW_EXPECT_TIMEOUT_MS) || 5_000 },
  workers: Number(process.env.PW_WORKERS) || 2,
  use: {
    // The capture-time origin is only a default; a suite is portable to another
    // environment with BASE_URL, and features never carry an absolute URL.
    baseURL: process.env.BASE_URL || '{{BASE_URL}}',
    actionTimeout: Number(process.env.PW_ACTION_TIMEOUT_MS) || 10_000,
    navigationTimeout: Number(process.env.PW_NAVIGATION_TIMEOUT_MS) || 15_000,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    video: 'off',
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  reporter: [['html'], ['json', { outputFile: 'test-results.json' }]],
});
