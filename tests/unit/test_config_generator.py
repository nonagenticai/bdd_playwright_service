"""Tests for deterministic config generation."""

from __future__ import annotations

import json
import pathlib
from typing import ClassVar

import pytest

from src.services.generation.config_generator import ConfigGenerator

# Literals (not imported from src) so a typo in the generator's own constants is
# caught rather than mirrored.
_HD_VIEWPORT = "viewport: { width: 1920, height: 1080 }"
_HD_VIDEO_SIZE = "size: { width: 1920, height: 1080 }"
# The chromium project captures at 2x: a 1280x720 CSS viewport rasterized into a
# 2560x1440 surface by the browser-level flag (the ONLY lever that adds real
# pixels — `video.size` pads rather than upscales, and context `deviceScaleFactor`
# is ignored by the video recorder).
_CAPTURE_VIEWPORT = "viewport: { width: 1280, height: 720 }"
_CAPTURE_FLAG = "launchOptions: { args: ['--force-device-scale-factor=2'] }"
_CAPTURE_VIDEO_SIZE = "size: { width: 2560, height: 1440 }"
# Firefox records 1:1 at the delivery resolution. MEASURED 2026-09-12: neither 2x
# lever works — the Gecko pref never reaches layout and yields a BLURRIER frame than
# a plain upscale of the 1x capture, and context `deviceScaleFactor` moves the page
# but not the video recorder (1x picture in the top-left, 75% grey padding). See
# `_FIREFOX_CAPTURE` in the generator for the numbers.
_FIREFOX_PREF = "firefoxUserPrefs"
_FIREFOX_DSF = "deviceScaleFactor"


def _code(cfg: str) -> str:
    """``cfg`` with its ``//`` comment lines dropped.

    Both the template and the builder EXPLAIN the traps they close at length, and
    the prose necessarily names the very literals these tests look for (e.g. the
    chromium launch flag, in the comment saying Firefox cannot take it). A substring
    check against the raw file therefore matches documentation, not configuration.
    """
    return "\n".join(line for line in cfg.splitlines() if not line.strip().startswith("//"))


class TestConfigGenerator:
    """Tests for ConfigGenerator deterministic output."""

    def setup_method(self):
        self.gen = ConfigGenerator()

    def test_generate_playwright_config_contains_base_url(self):
        """Playwright config should contain the provided base URL."""
        result = self.gen.generate_playwright_config("https://app.example.com")
        assert "https://app.example.com" in result

    def test_generate_playwright_config_contains_bdd_imports(self):
        """Playwright config should import playwright-bdd."""
        result = self.gen.generate_playwright_config("https://example.com")
        assert "defineBddConfig" in result
        assert "defineConfig" in result

    def test_generate_playwright_config_features_and_steps(self):
        """Playwright config should reference features and steps directories."""
        result = self.gen.generate_playwright_config("https://example.com")
        assert "features" in result
        assert "steps" in result

    def test_generate_playwright_config_web_platform(self):
        """Web platform config should include desktop browser projects."""
        result = self.gen.generate_playwright_config("https://example.com", platform="web")
        assert "chromium" in result or "Desktop Chrome" in result

    def test_generate_playwright_config_template_has_failfast_timeouts(self):
        """The static (no-auth) template must carry env-overridable fail-fast
        timeouts so a hung selector fails in seconds, not the 30s default."""
        result = self.gen.generate_playwright_config("https://example.com")
        for knob in (
            "PW_TEST_TIMEOUT_MS",
            "PW_EXPECT_TIMEOUT_MS",
            "PW_ACTION_TIMEOUT_MS",
            "PW_NAVIGATION_TIMEOUT_MS",
        ):
            assert knob in result, f"template missing {knob}"
        assert "actionTimeout" in result
        assert "navigationTimeout" in result

    def test_generate_playwright_config_missing_steps_failonrun(self):
        """Both config paths default to missingSteps:'fail-on-run' so an uncovered
        step doesn't fail bddgen and lose the whole run's footage."""
        template_path = self.gen.generate_playwright_config("https://example.com")
        auth_path = self.gen.generate_playwright_config("https://example.com", has_auth=True)
        for result in (template_path, auth_path):
            assert "missingSteps" in result
            assert "fail-on-run" in result

    def test_generate_playwright_config_missing_steps_override(self):
        """Strict QA streams can request 'fail-on-gen'; both paths honor it."""
        template_path = self.gen.generate_playwright_config(
            "https://example.com", missing_steps="fail-on-gen"
        )
        auth_path = self.gen.generate_playwright_config(
            "https://example.com", has_auth=True, missing_steps="fail-on-gen"
        )
        for result in (template_path, auth_path):
            assert "missingSteps: 'fail-on-gen'" in result
            assert "fail-on-run" not in result
            assert "{{MISSING_STEPS}}" not in result  # placeholder fully substituted

    def test_generate_playwright_config_auth_path_imports_devices(self):
        """The has_auth/inline path emits `...devices[...]` in its projects, so it
        MUST import `devices` — else the config throws `ReferenceError` at load."""
        result = self.gen.generate_playwright_config("https://example.com", has_auth=True)
        assert "devices[" in result  # projects reference devices
        assert "import { defineConfig, devices }" in result

    def test_generate_playwright_config_auth_path_has_failfast_timeouts(self):
        """The inline has_auth path must carry the same fail-fast timeouts —
        the two config paths must stay in sync."""
        result = self.gen.generate_playwright_config("https://example.com", has_auth=True)
        for knob in (
            "PW_TEST_TIMEOUT_MS",
            "PW_EXPECT_TIMEOUT_MS",
            "PW_ACTION_TIMEOUT_MS",
            "PW_NAVIGATION_TIMEOUT_MS",
        ):
            assert knob in result, f"auth-path config missing {knob}"
        assert "storageState" in result  # auth path still wires storage state

    def test_generate_playwright_config_records_full_hd(self):
        """Both config paths must pin an explicit 1920x1080 video size + viewport.

        A bare ``video: 'on'`` has no size, so Playwright scales the viewport into
        800x800 — a 1280x720 device viewport records at 800x450 and the 1920x1080
        compiler upscales it 2.4x into visibly soft footage.
        """
        template_path = self.gen.generate_playwright_config("https://example.com")
        auth_path = self.gen.generate_playwright_config("https://example.com", has_auth=True)
        for result in (template_path, auth_path):
            assert _HD_VIDEO_SIZE in result
            assert "mode: 'on'" in result
            assert _HD_VIEWPORT in result
            # The sizeless default is exactly the bug — it must be gone. Match on
            # directive lines only; the explanatory comments quote `video: 'on'`.
            directives = [
                line.strip() for line in result.splitlines() if not line.lstrip().startswith("//")
            ]
            assert not any(line.startswith("video: 'on'") for line in directives)

    def test_generate_playwright_config_viewport_not_clobbered_by_device_spread(self):
        """``devices['Desktop Chrome']`` carries its own 1280x720 viewport, and a
        later key wins in a JS object literal — so every desktop project's viewport
        override MUST be emitted AFTER the ``...devices[...]`` spread."""
        for has_auth in (False, True):
            result = self.gen.generate_playwright_config("https://example.com", has_auth=has_auth)
            # The two desktop projects override to DIFFERENT viewports — chromium
            # to the 1280x720 it rasterizes 2x, firefox to a 1:1 1920x1080 (MEASURED:
            # firefox has no working 2x lever). Either way the override must land
            # AFTER its own device spread or the preset's own viewport wins.
            for device, viewport in (
                ("Desktop Chrome", _CAPTURE_VIEWPORT),
                ("Desktop Firefox", _HD_VIEWPORT),
            ):
                spread = f"...devices['{device}']"
                assert spread in result, f"missing project for {device}"
                spread_idx = result.index(spread)
                viewport_idx = result.find(viewport, spread_idx)
                assert viewport_idx > spread_idx, (
                    f"{device} viewport override must come after the device spread, "
                    "or the preset's own viewport wins and video drops to 800x450"
                )

    def test_chromium_project_captures_at_2x(self):
        """The chromium project must carry ALL THREE halves of the 2x capture.

        Pinning ``video.size`` alone buys a bigger CANVAS, not more content:
        Playwright never upscales into the recorded frame, it draws the page at 1:1
        top-left and pads the rest grey (measured: a 1280x720 viewport recorded at
        2560x1440 gives a 1278x718 white region). Context ``deviceScaleFactor`` is
        ignored by the video recorder outright. So the browser-level flag is what
        actually puts real pixels in the clip, and dropping any one of the three
        silently returns the capture to ~550px of real source upscaled into a
        2560x1440 master.
        """
        for has_auth in (False, True):
            result = self.gen.generate_playwright_config("https://example.com", has_auth=has_auth)
            spread_idx = result.index("...devices['Desktop Chrome']")
            for fragment in (_CAPTURE_VIEWPORT, _CAPTURE_FLAG, _CAPTURE_VIDEO_SIZE):
                idx = result.find(fragment, spread_idx)
                assert idx > spread_idx, (
                    f"chromium project missing {fragment!r} after the device spread "
                    f"(has_auth={has_auth})"
                )

    def test_force_device_scale_factor_is_chromium_only(self):
        """``--force-device-scale-factor`` is a Chromium argument.

        Hoisting it into the top-level ``use`` would hand it to the firefox project
        too, so the flag must appear exactly once — inside the chromium project,
        after its spread and before the firefox one.
        """
        for has_auth in (False, True):
            # Comment-stripped: the firefox project's own comment NAMES this flag to
            # explain why Gecko cannot take it, so a raw count matches the prose.
            result = _code(
                self.gen.generate_playwright_config("https://example.com", has_auth=has_auth)
            )
            assert result.count("--force-device-scale-factor") == 1
            flag_idx = result.index(_CAPTURE_FLAG)
            assert flag_idx > result.index("...devices['Desktop Chrome']")
            assert flag_idx < result.index("...devices['Desktop Firefox']")

    @pytest.mark.parametrize("platform", ["web", "mobile"])
    def test_setup_project_can_actually_find_auth_setup_ts(self, platform):
        """The `setup` project MUST override `testDir`, or it matches zero files.

        A Playwright project's `testDir` defaults to the top-level one, which here
        is the `defineBddConfig(...)` output dir (`.features-gen`). `auth.setup.ts`
        is written to the PACKAGE ROOT (`job_processor._store_artifacts`, zipped by
        `package_builder`'s root `*.ts` pattern), so `testMatch` alone filtered a
        directory the file is not in: setup ran nothing, passed vacuously, never
        wrote `.auth/user.json`, and every dependent project — each carrying
        `storageState: '.auth/user.json'` — then failed at context creation with
        ENOENT. 100% of scenarios fail and zero clips are recorded, while the error
        names storage state rather than login.
        """
        result = self.gen.generate_playwright_config(
            "https://example.com", platform=platform, has_auth=True
        )
        setup_line = next(line for line in result.splitlines() if "name: 'setup'" in line)
        assert "testDir: '.'" in setup_line, (
            "setup project inherits the bddgen output dir as testDir and cannot "
            f"see the package-root auth.setup.ts: {setup_line}"
        )
        # The match itself must survive the change — testDir alone would sweep in
        # every root-level .ts (playwright.config.ts).
        assert "testMatch: /auth\\.setup\\.ts/" in setup_line
        # And the projects that depend on it still expect the state file it writes.
        assert "dependencies: ['setup']" in result
        assert "storageState: '.auth/user.json'" in result

    @pytest.mark.parametrize("platform", ["web", "mobile"])
    def test_setup_project_budget_exceeds_the_waits_in_auth_setup_ts(self, platform):
        """The setup project needs its OWN timeout, above auth.setup.ts's waits.

        The top-level per-test budget is a deliberately fail-fast 20s, sized for a
        scenario step whose selector must not pin the 30s default. `auth.setup.ts`
        is built on 60s waits (SSO handoff; the username field after a SvelteKit
        splash that can take 10-20s to hydrate) plus a 30s post-login URL
        assertion, so under the inherited 20s cap that hardening is dead code —
        setup is killed mid-hydration on exactly the slow IdP-gated target it was
        written for, and the run fails the same ENOENT way.

        Asserted against the waits PARSED OUT of the shipped auth.setup.ts, not a
        copied literal, so raising a wait there without raising this budget fails.
        """
        import re

        setup_src = self.gen.generate_auth_setup("https://app.example.com/login")
        waits_ms = [int(m.replace("_", "")) for m in re.findall(r"timeout:\s*([\d_]+)", setup_src)]
        assert waits_ms, "no timeouts found in auth.setup.ts — parse is stale"
        longest = max(waits_ms)

        result = self.gen.generate_playwright_config(
            "https://example.com", platform=platform, has_auth=True
        )
        setup_line = next(line for line in result.splitlines() if "name: 'setup'" in line)
        match = re.search(r"timeout:\s*([\d_]+)", setup_line)
        assert match, f"setup project has no timeout of its own: {setup_line}"
        budget = int(match.group(1).replace("_", ""))
        assert budget > longest, (
            f"setup budget {budget}ms must exceed the longest wait in auth.setup.ts "
            f"({longest}ms), or the login is killed mid-wait"
        )
        # Scenario projects keep the fail-fast budget — this must not become global.
        assert "PW_TEST_TIMEOUT_MS) || 20_000" in result

    def test_generate_playwright_config_mobile_keeps_device_viewport(self):
        """Full HD is desktop-only: forcing a 1920x1080 viewport onto Pixel 5 /
        iPhone 13 projects would defeat the mobile emulation.

        Note the mobile *projects* only materialize on the inline path — a no-auth
        request always returns the static (desktop) scaffold template regardless of
        platform, which is a pre-existing quirk of ``generate_playwright_config``.
        """
        result = self.gen.generate_playwright_config(
            "https://example.com", platform="mobile", has_auth=True
        )
        assert "...devices['Pixel 5']" in result
        assert _HD_VIEWPORT not in result
        assert "video: 'on'" in result  # sizeless default is correct for phones

    def test_generate_package_json_valid_json(self):
        """Package JSON should be valid JSON."""
        result = self.gen.generate_package_json()
        pkg = json.loads(result)
        assert isinstance(pkg, dict)

    def test_generate_package_json_has_playwright_bdd(self):
        """Package JSON should include playwright-bdd dependency."""
        result = self.gen.generate_package_json()
        pkg = json.loads(result)
        dev_deps = pkg.get("devDependencies", {})
        assert "playwright-bdd" in dev_deps

    def test_generate_package_json_has_test_script(self):
        """Package JSON should include a test script."""
        result = self.gen.generate_package_json()
        pkg = json.loads(result)
        scripts = pkg.get("scripts", {})
        assert "test" in scripts

    def test_generate_package_json_has_playwright(self):
        """Package JSON should include @playwright/test dependency."""
        result = self.gen.generate_package_json()
        pkg = json.loads(result)
        dev_deps = pkg.get("devDependencies", {})
        assert "@playwright/test" in dev_deps

    def test_generate_tsconfig_valid_json(self):
        """tsconfig should be valid JSON."""
        result = self.gen.generate_tsconfig()
        tsconfig = json.loads(result)
        assert isinstance(tsconfig, dict)

    def test_generate_tsconfig_has_compiler_options(self):
        """tsconfig should have compilerOptions."""
        result = self.gen.generate_tsconfig()
        tsconfig = json.loads(result)
        assert "compilerOptions" in tsconfig
        assert tsconfig["compilerOptions"]["strict"] is True

    def test_generate_tsconfig_includes_steps(self):
        """tsconfig should include steps directory."""
        result = self.gen.generate_tsconfig()
        tsconfig = json.loads(result)
        include = tsconfig.get("include", [])
        assert any("steps" in entry for entry in include)

    def test_generate_tsconfig_target_es2022(self):
        """tsconfig should target ES2022."""
        result = self.gen.generate_tsconfig()
        tsconfig = json.loads(result)
        assert tsconfig["compilerOptions"]["target"] == "ES2022"

    def test_generate_package_json_is_private(self):
        """Package JSON should be marked private."""
        result = self.gen.generate_package_json()
        pkg = json.loads(result)
        assert pkg.get("private") is True

    def test_generate_package_json_pins_cucumber_expressions(self):
        """Generated package.json must pin the transitive
        @cucumber/cucumber-expressions. Generated packages ship no lockfile, so
        `npm install` would otherwise float this playwright-bdd transitive to a
        version whose bddgen is incompatible and the record path dies."""
        result = self.gen.generate_package_json()
        pkg = json.loads(result)
        assert pkg["overrides"]["@cucumber/cucumber-expressions"] == "18.0.1"

    def test_generate_package_json_inline_fallback_pins_cucumber_expressions(
        self, tmp_path, monkeypatch
    ):
        """When the scaffold template is absent, the inline fallback emitter must
        carry the identical pin (both emitters are kept in sync)."""
        from src.services.generation import config_generator as cg

        # Point the template dir at an empty tmp dir so generate_package_json()
        # cannot find package.json and takes the inline-fallback branch.
        monkeypatch.setattr(cg, "_TEMPLATES_DIR", tmp_path)
        result = ConfigGenerator().generate_package_json()
        pkg = json.loads(result)
        assert pkg["overrides"]["@cucumber/cucumber-expressions"] == "18.0.1"

    def test_generate_tsconfig_includes_lib(self):
        """tsconfig should include the shared recording-helper lib directory so
        tsc typechecks helpers imported by the steps."""
        result = self.gen.generate_tsconfig()
        include = json.loads(result).get("include", [])
        assert any("lib" in entry for entry in include)

    def test_generate_auth_setup_reads_env_and_login_url(self):
        """auth.setup.ts should substitute the login URL and read AUTH_* from env
        (never hardcode credentials)."""
        result = self.gen.generate_auth_setup("https://app.example.com/login")
        assert "https://app.example.com/login" in result
        assert "process.env.AUTH_USERNAME" in result
        assert "process.env.AUTH_PASSWORD" in result
        assert ".auth/user.json" in result

    @pytest.mark.parametrize("emitter", ["template", "inline"])
    def test_generate_auth_setup_has_already_authenticated_bypass(
        self, emitter, tmp_path, monkeypatch
    ):
        """The setup must probe the app root first and skip the login form when
        the target isn't gated (public / valid session / injected state) — the
        most-repeated manual patch.

        This assertion is ORDERED on purpose. Its predecessor only checked that
        the strings `process.env.BASE_URL` and `storageState` appeared SOMEWHERE
        in the file — and both survive the bypass block's deletion (line 30
        computes `baseUrl` regardless, line 94 saves storageState at the end of
        the login path). Deleting the entire bypass left it green, which is the
        definition of a test that cannot fail for its stated reason. It also
        claimed to cover the inline fallback while only ever exercising the
        template, so it is parametrized over both emitters now.
        """
        from src.services.generation import config_generator as cg

        if emitter == "inline":
            # Empty template dir => generate_auth_setup takes the inline fallback.
            monkeypatch.setattr(cg, "_TEMPLATES_DIR", tmp_path)
            result = ConfigGenerator().generate_auth_setup("https://app.example.com/login")
            assert "test as setup" in result  # sanity: we really got the fallback
        else:
            result = self.gen.generate_auth_setup("https://app.example.com/login")

        assert "process.env.BASE_URL" in result

        # The bypass is a SHAPE, not a set of tokens: go to the app root, and if it
        # did not bounce us to a login screen, save the state and RETURN — all of
        # that strictly before the first credential fill.
        first_fill = result.index(".fill(")
        goto_base = result.index("goto(baseUrl")
        save = result.index("storageState({ path")
        early_return = result.index("return;")

        assert goto_base < first_fill, "the app root must be probed before filling a login form"
        assert save < first_fill, "the bypass must save storageState before any form fill"
        assert goto_base < save < early_return < first_fill, (
            "bypass order must be goto(baseUrl) -> storageState save -> return, "
            "all before the first fill()"
        )
        # And the save on the bypass path must be the state file the projects depend on.
        assert ".auth/user.json" in result[:first_fill]

    def test_read_scaffold_lib_returns_recording_helpers(self):
        """The shared recording-helper library ships under lib/ for demo_video
        packages — both the TS helpers and the .mjs fixture-reset utilities."""
        lib = self.gen.read_scaffold_lib()
        assert "lib/recording_helpers.ts" in lib
        assert "lib/fixture_reset.mjs" in lib
        # Non-empty, and every key is package-relative under lib/.
        assert all(rel.startswith("lib/") and content for rel, content in lib.items())
        assert "export" in lib["lib/recording_helpers.ts"]


class TestEveryProjectRecordsAtAnExplicitSize:
    """5a — a bare ``video: 'on'`` records at Playwright's 800x800 fit-box.

    Only the chromium desktop project got an explicit size in #157. The mobile
    package emitted a bare ``video: 'on'`` for BOTH of its projects, so a Pixel 5
    recorded 369x800 and the compiler upscaled that into a 2560x1440 master.
    """

    gen = ConfigGenerator()

    #: The non-`setup` projects each platform must emit.
    _PLATFORM_PROJECTS: ClassVar[dict[str, set[str]]] = {
        "web": {"chromium", "firefox"},
        "mobile": {"mobile-chrome", "mobile-safari"},
    }

    @staticmethod
    def _projects(cfg: str) -> dict[str, str]:
        """Every emitted project's body, keyed by name — ENUMERATED, not grepped.

        ⛔ THIS ENUMERATION IS THE POINT. The obvious form of this test — filter the
        config's lines for those containing ``video:`` and assert each states a size
        — is STRUCTURALLY BLIND to the defect it claims to catch: a project that
        emits NO ``video`` key at all contributes no line to filter, so it passes
        silently. That is precisely how two projects survived the first pass at
        this: the DESKTOP ``setup`` project stated a viewport and no ``video``, and
        the MOBILE ``setup`` project stated no ``use`` at all and recorded a
        1280x720 page into an inherited 393x851 frame.

        So the projects are enumerated first, and every one of them must then be
        shown to state a size. A project that goes missing fails the set check; a
        project that states nothing fails the per-project check.
        """
        code = _code(cfg)
        chunks = code.split("name: '")[1:]
        return {chunk.split("'", 1)[0]: chunk for chunk in chunks}

    def test_every_project_states_a_video_size_on_every_platform(self):
        for platform in ("web", "mobile"):
            for has_auth in (False, True):
                cfg = self.gen.generate_playwright_config(
                    "https://example.com", platform=platform, has_auth=has_auth
                )
                where = f"platform={platform} has_auth={has_auth}"
                projects = self._projects(cfg)
                expected = set(self._PLATFORM_PROJECTS[platform])
                if has_auth:
                    expected.add("setup")
                assert set(projects) == expected, f"{where}: emitted {sorted(projects)}"
                # The unsized form is the defect itself; it must appear nowhere.
                assert "video: 'on'" not in _code(cfg), (
                    f"{where} still emits an UNSIZED `video: 'on'`, which Playwright "
                    "scales into an 800x800 box"
                )
                for name, body in projects.items():
                    # PRESENCE first — this is the assertion a `video:`-line filter
                    # cannot make, because an absent key emits no line.
                    assert "video:" in body, (
                        f"{where}: project `{name}` emits NO `video` key. It inherits "
                        "whatever the top-level `use` happens to say, which for the "
                        "mobile path is a size that does not match its viewport."
                    )
                    value = body.split("video:", 1)[1][:120]
                    assert "size:" in value or "videoAt(" in value, (
                        f"{where}: project `{name}` records at no stated size: {value!r}"
                    )

    def test_a_mobile_package_gets_mobile_projects_even_without_auth(self):
        """``platform`` was IGNORED on the no-auth path.

        ``generate_playwright_config`` returned the scaffold template whenever
        ``has_auth`` was false, and that template is the DESKTOP config: it carries
        no ``{{PLATFORM}}`` token and hardcodes chromium + firefox. So a mobile
        package with no login silently shipped desktop projects and never emulated a
        phone at all — the platform argument had no effect whatsoever.
        """
        cfg = self.gen.generate_playwright_config(
            "https://example.com", platform="mobile", has_auth=False
        )
        assert set(self._projects(cfg)) == {"mobile-chrome", "mobile-safari"}
        assert "devices['Pixel 5']" in cfg
        assert "devices['Desktop Chrome']" not in cfg

    def test_the_desktop_no_auth_path_is_still_the_template_verbatim(self):
        """The platform guard must not disturb the desktop no-auth output.

        Desktop with no auth still comes from the scaffold template, byte for byte
        after substitution — the guard only diverts MOBILE into the builder.
        """
        template = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src"
            / "templates"
            / "scaffold"
            / "playwright.config.ts"
        )
        expected = (
            template.read_text(encoding="utf-8")
            .replace("{{BASE_URL}}", "https://example.com")
            .replace("{{PLATFORM}}", "web")
            .replace("{{MISSING_STEPS}}", "fail-on-run")
        )
        assert (
            self.gen.generate_playwright_config("https://example.com", platform="web") == expected
        )

    def test_each_mobile_project_records_at_its_own_device_size(self):
        cfg = self.gen.generate_playwright_config(
            "https://example.com", platform="mobile", has_auth=True
        )
        # The size is READ FROM the device descriptor, never restated as a literal
        # that goes stale when Playwright revises a preset.
        assert "const videoAt = " in cfg
        assert "size: { width: d.viewport.width, height: d.viewport.height }" in cfg
        assert "video: videoAt(devices['Pixel 5'])" in cfg
        assert "video: videoAt(devices['iPhone 13'])" in cfg
        # ...and AFTER the device spread, the ordering trap #157 documented.
        pixel = cfg.split("name: 'mobile-chrome'")[1].split("},")[0]
        assert pixel.index("...devices['Pixel 5']") < pixel.index("video: videoAt")

    def test_a_mobile_project_never_gets_the_chromium_2x_launch_argument(self):
        """The presets carry their OWN deviceScaleFactor (Pixel 5 = 2.75), and
        `mobile-safari` is WebKit, which takes no Chromium launch argument."""
        cfg = self.gen.generate_playwright_config(
            "https://example.com", platform="mobile", has_auth=True
        )
        assert "--force-device-scale-factor" not in cfg
        assert "launchOptions" not in cfg

    def test_the_desktop_firefox_project_records_1to1_at_the_delivery_size(self):
        """Firefox must record 1920x1080 into a 1920x1080 frame — no fake 2x.

        MEASURED 2026-09-12 (Playwright 1.50.0, firefox-1490), Laplacian variance
        of a real recorded frame resampled to the compiler's 2560x1440 master:
        1920x1080 @1x = 34427, beating both the 1280x720 @1x baseline (27287) and
        `firefoxUserPrefs: devPixelsPerPx=2` (11798, a REGRESSION at 4x the
        bitrate). Context `deviceScaleFactor: 2` does raise the page's
        devicePixelRatio to 2 but the VIDEO RECORDER ignores it: the 2560x1440
        frame holds the 1x picture 1:1 in the top-left plus 75% grey padding.

        So the firefox project must carry NEITHER 2x lever, and must not claim the
        2560x1440 capture size that only chromium actually rasterizes.
        """
        for has_auth in (False, True):
            cfg = _code(
                self.gen.generate_playwright_config("https://example.com", has_auth=has_auth)
            )
            firefox = cfg.split("name: 'firefox'")[1]
            assert _HD_VIEWPORT in firefox
            assert _HD_VIDEO_SIZE in firefox
            # ...and NONE of the three levers that were measured not to work.
            assert _CAPTURE_VIDEO_SIZE not in firefox
            assert _FIREFOX_PREF not in firefox
            assert _FIREFOX_DSF not in firefox
            assert "--force-device-scale-factor" not in firefox


class TestQaSuiteConfig:
    """The QA config is deliberately NOT the demo-video scaffold."""

    def test_qa_config_is_fast_single_browser_and_portable(self):
        cfg = _code(ConfigGenerator().generate_qa_config("http://127.0.0.1:8322"))
        assert "video: 'off'" in cfg
        assert cfg.count("name: 'chromium'") == 1
        assert "firefox" not in cfg
        assert "process.env.BASE_URL" in cfg
        assert "'http://127.0.0.1:8322'" in cfg
        assert "missingSteps: 'fail-on-gen'" in cfg
        assert "PW_WORKERS" in cfg

    def test_read_qa_steps_returns_the_static_vocabulary(self):
        steps = ConfigGenerator().read_qa_steps()
        assert len(steps) == 1
        assert steps[0]["filename"] == "qa.steps.ts"
        assert steps[0]["step_count"] == 8
        assert "createBdd()" in steps[0]["content"]

    def test_qa_package_pins_a_runnable_playwright(self):
        pkg = json.loads(ConfigGenerator().generate_qa_package_json())
        # 1.50's installer hangs on Node 26; the QA suite tracks the target apps.
        assert pkg["devDependencies"]["@playwright/test"] == "1.64.1"
        assert "@types/node" in pkg["devDependencies"]
        assert pkg["devDependencies"]["playwright-bdd"] == "^9.2.1"

    def test_qa_config_embeds_the_capture_locale(self):
        cfg = _code(ConfigGenerator().generate_qa_config("http://x", locale="nl-NL"))
        assert 'locale: process.env.BDD_LOCALE || "nl-NL"' in cfg
        plain = _code(ConfigGenerator().generate_qa_config("http://x"))
        assert "locale:" not in plain
