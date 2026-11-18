"""Tests for configuration management."""

from __future__ import annotations

from src.config import Settings, clear_settings_cache, get_settings


class TestSettings:
    """Tests for the Settings configuration class."""

    def test_default_settings(self):
        """Default settings should have expected values."""
        clear_settings_cache()
        settings = get_settings()
        assert settings.app_name == "BDD Playwright Service"
        assert settings.app_version == "0.1.0"
        # api.port default comes from w2v_common (8000); overridden via API_PORT env at deploy
        assert settings.narration.default_tone == "tutorial"
        assert settings.generation.max_scenarios_per_feature == 10
        assert settings.generation.max_features_per_job == 20

    def test_llm_config_removed(self):
        """LLM config is removed — all LLM calls route through worker_management."""
        clear_settings_cache()
        settings = get_settings()
        assert not hasattr(settings, "llm"), (
            "settings.llm should be gone — LLM access is via WMClient now"
        )

    def test_settings_from_env(self, monkeypatch):
        """Settings should load from environment variables."""
        clear_settings_cache()
        monkeypatch.setenv("API_PORT", "9999")
        monkeypatch.setenv("NARRATION_DEFAULT_TONE", "demo")

        settings = Settings()
        assert settings.api.port == 9999
        assert settings.narration.default_tone == "demo"

    def test_database_settings(self):
        """Database settings should have expected defaults."""
        clear_settings_cache()
        settings = get_settings()
        assert settings.database.pool_size == 10
        assert settings.database.max_overflow == 20
        assert settings.database.pool_timeout == 30

    def test_downstream_settings_only_holds_license_token(self):
        """DownstreamSettings should expose only the shared license_token field.

        Per-peer URLs are no longer carried by config -- they are resolved at
        the call-site via ``w2v_common.discovery.service_url()``.
        """
        clear_settings_cache()
        settings = get_settings()
        # license_token is the only field on the base DownstreamSettings.
        assert settings.downstream.license_token == ""
        # The removed per-peer URL fields must NOT be present on the model.
        for removed in (
            "app_inspector_url",
            "video_production_url",
            "tts_url",
            "content_url",
            "webscraper_url",
        ):
            assert not hasattr(settings.downstream, removed), (
                f"settings.downstream.{removed} should have been removed"
            )

    def test_downstream_license_token_from_env(self, monkeypatch):
        """DOWNSTREAM_LICENSE_TOKEN env var should populate license_token."""
        clear_settings_cache()
        monkeypatch.setenv("DOWNSTREAM_LICENSE_TOKEN", "tok-bdd-123")
        settings = Settings()
        assert settings.downstream.license_token == "tok-bdd-123"

    def test_service_url_resolves_known_peers(self):
        """service_url() must produce the discovery-default URL for peers used by this service."""
        from w2v_common.discovery import service_url

        # In-cluster default form (no W2V_LOCAL set -> svc.cluster.local DNS),
        # except TTS which discovery routes to the mini2 launchd host service
        # at tts:9202 (host service; no namespace).
        assert service_url("app_inspector").startswith("http://app-inspector-service.w2v")
        assert service_url("video_production").startswith("http://video-production-service.w2v")
        assert service_url("tts").startswith("http://tts:9202")
        assert service_url("content").startswith("http://content-service.w2v")
        # scraping goes straight to webscraper (its own namespace; KAG retired)
        assert service_url("webscraper").startswith("http://webscraper.webscraper")

    def test_service_url_respects_per_peer_override(self, monkeypatch):
        """W2V_<NAME>_URL env var must override the discovered URL."""
        from w2v_common.discovery import service_url

        monkeypatch.setenv("W2V_TTS_URL", "http://tts.local:6194")
        assert service_url("tts") == "http://tts.local:6194"

    def test_settings_caching(self):
        """get_settings should return cached instance."""
        clear_settings_cache()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_clear_settings_cache(self):
        """clear_settings_cache should reset the cached instance."""
        clear_settings_cache()
        s1 = get_settings()
        clear_settings_cache()
        s2 = get_settings()
        assert s1 is not s2

    def test_is_production(self, monkeypatch):
        """is_production should check environment string."""
        clear_settings_cache()
        monkeypatch.setenv("ENVIRONMENT", "production")
        settings = Settings()
        assert settings.is_production()
        assert not settings.is_development()

    def test_is_development(self):
        """is_development should check environment string."""
        clear_settings_cache()
        settings = get_settings()
        assert not settings.is_production()

    def test_logging_level_validation(self, monkeypatch):
        """Logging level should be uppercased and validated."""
        clear_settings_cache()
        monkeypatch.setenv("LOG_LEVEL", "debug")
        settings = Settings()
        assert settings.logging.level == "DEBUG"

    def test_cors_origins_parsing(self, monkeypatch):
        """CORS origins should be parsed from JSON env var (pydantic-settings 2.x)."""
        clear_settings_cache()
        monkeypatch.setenv("API_CORS_ORIGINS", '["http://a.com","http://b.com"]')
        settings = Settings()
        assert settings.api.cors_origins == ["http://a.com", "http://b.com"]
