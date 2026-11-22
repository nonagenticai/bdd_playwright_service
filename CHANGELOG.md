# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Grounded QA mode** for `POST /api/bdd/qa-suite` (default). Deterministic
  page-contract + captured-link navigation features, a static closed step vocabulary
  (`templates/scaffold/qa/steps.ts`), snapshot validation that drops ungrounded
  scenarios with a report (`validation.json`, `step_coverage.grounding_validation`),
  and a QA-specific config (video off, one browser, `BASE_URL` env). `include_llm_scenarios`
  opts into LLM-authored scenarios inside the same vocabulary; `generation_mode: "legacy"`
  keeps the old path (also used when `auth_config` is set).
- `package-compile` CI job: renders a grounded package and runs `bddgen` + `tsc --noEmit`
  on it — the compile gate the Node-less service runtime cannot provide.

### Fixed

- `html_structure` now emits each interactive element's `role`, and checkbox/radio
  accessible names come from their `<label>` rather than the `value` attribute
  (`getByRole('checkbox', { name: '0' })` could never match).

