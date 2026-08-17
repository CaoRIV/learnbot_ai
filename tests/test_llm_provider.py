import importlib
import json

import config
import llm_provider


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _capture_post(monkeypatch, response_payload):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return DummyResponse(response_payload)

    monkeypatch.setattr(llm_provider.requests, "post", fake_post)
    return captured


def test_siliconflow_provider_uses_chat_completions(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-siliconflow")
    monkeypatch.setenv("SILICONFLOW_MODEL_NAME", "test-silicon-model")
    captured = _capture_post(
        monkeypatch,
        {"choices": [{"message": {"content": "Câu trả lời SiliconFlow"}}]},
    )

    result = llm_provider.call_llm("Xin chào", provider="siliconflow")

    payload = json.loads(captured["data"].decode("utf-8"))
    assert result == "Câu trả lời SiliconFlow"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer sk-siliconflow"
    assert payload["model"] == "test-silicon-model"
    assert payload["messages"][0]["content"] == "Xin chào"


def test_openai_provider_uses_bearer_auth(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-openai-model")
    captured = _capture_post(
        monkeypatch,
        {"choices": [{"message": {"content": "Câu trả lời OpenAI"}}]},
    )

    result = llm_provider.call_llm("Kiểm tra", provider="openai")

    payload = json.loads(captured["data"].decode("utf-8"))
    assert result == "Câu trả lời OpenAI"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-openai"
    assert payload["model"] == "test-openai-model"


def test_gemini_provider_uses_generate_content(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GEMINI_MODEL_NAME", "test-gemini-model")
    captured = _capture_post(
        monkeypatch,
        {"candidates": [{"content": {"parts": [{"text": "Câu trả lời Gemini"}]}}]},
    )

    result = llm_provider.call_llm("Kiểm tra", provider="gemini")

    payload = json.loads(captured["data"].decode("utf-8"))
    assert result == "Câu trả lời Gemini"
    assert captured["url"].endswith(
        "/v1beta/models/test-gemini-model:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "gemini-key"
    assert payload["contents"][0]["parts"][0]["text"] == "Kiểm tra"


def test_missing_api_key_does_not_make_network_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def unexpected_post(*args, **kwargs):
        raise AssertionError("Không được gọi mạng khi thiếu API key")

    monkeypatch.setattr(llm_provider.requests, "post", unexpected_post)
    result = llm_provider.call_llm("Xin chào", provider="openai")

    assert "chưa cấu hình OPENAI_API_KEY" in result


def test_missing_ollama_host_does_not_affect_import_or_provider(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "siliconflow")

    reloaded_config = importlib.reload(config)
    reloaded_provider = importlib.reload(llm_provider)

    assert reloaded_config.DEFAULT_MODEL_CHOICE == "siliconflow"
    assert reloaded_provider.get_provider_name() == "siliconflow"
