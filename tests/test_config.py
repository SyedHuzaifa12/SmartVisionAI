"""Tests for config.get_settings."""

from config import get_settings


def test_get_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-flash")

    settings = get_settings()

    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "gemini-1.5-flash"
    assert settings.has_gemini_key is True


def test_settings_without_key_reports_not_configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    settings = get_settings()

    assert settings.has_gemini_key is False
