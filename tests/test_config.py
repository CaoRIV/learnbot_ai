import config


def test_api_key_validation_rejects_placeholders():
    assert config.is_configured_api_key(None) is False
    assert config.is_configured_api_key("") is False
    assert config.is_configured_api_key("Your_SILICONFLOW_API_KEY") is False
    assert config.is_configured_api_key("sk-real-value") is True


def test_provider_selection_uses_supported_value(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert config.choose_default_provider() == "gemini"
    assert config.choose_default_provider("openai") == "openai"


def test_invalid_provider_falls_back_to_siliconflow(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    assert config.choose_default_provider() == "siliconflow"
