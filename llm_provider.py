"""Lớp trừu tượng gọi LLM qua API bên ngoài.

Provider được chọn bằng ``LLM_PROVIDER`` hoặc đối số ``provider``. Module này
không đọc, không yêu cầu và không gọi bất kỳ dịch vụ LLM local nào.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from config import SUPPORTED_LLM_PROVIDERS, is_configured_api_key


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: Optional[str]
    api_url: str
    model: str


def get_provider_name(provider=None):
    """Trả về provider hợp lệ, ưu tiên đối số rồi đến ``LLM_PROVIDER``."""
    selected = (provider or os.getenv("LLM_PROVIDER") or "siliconflow").strip().lower()
    if selected not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Provider LLM không hợp lệ: {selected}. "
            f"Các giá trị hỗ trợ: {', '.join(SUPPORTED_LLM_PROVIDERS)}."
        )
    return selected


def get_provider_config(provider=None):
    """Đọc cấu hình provider tại thời điểm gọi để hỗ trợ test và đổi env."""
    selected = get_provider_name(provider)
    if selected == "siliconflow":
        return ProviderConfig(
            name=selected,
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            api_url=os.getenv(
                "SILICONFLOW_API_URL",
                "https://api.siliconflow.cn/v1/chat/completions",
            ),
            model=os.getenv(
                "SILICONFLOW_MODEL_NAME",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            ),
        )
    if selected == "openai":
        return ProviderConfig(
            name=selected,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_url=os.getenv(
                "OPENAI_API_URL",
                "https://api.openai.com/v1/chat/completions",
            ),
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        )
    return ProviderConfig(
        name=selected,
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        api_url=os.getenv(
            "GEMINI_API_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        model=os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash"),
    )


def _normalize_chat_completions_url(api_url):
    url = api_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _extract_chat_content(result):
    choices = result.get("choices") or []
    if not choices:
        return "Lỗi: API không trả về nội dung hợp lệ."
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return f"{content}<think>{reasoning}</think>" if reasoning else str(content)


def _extract_gemini_content(result):
    candidates = result.get("candidates") or []
    if not candidates:
        return "Lỗi: Gemini không trả về nội dung hợp lệ."
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    content = "".join(text_parts).strip()
    return content or "Lỗi: Gemini không trả về nội dung văn bản."


def _post_json(url, headers, payload):
    response = requests.post(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _call_openai_compatible(config, prompt, temperature, max_tokens):
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if config.name == "siliconflow":
        payload.update({"top_p": 0.7, "top_k": 50, "frequency_penalty": 0.5, "n": 1})
    result = _post_json(
        _normalize_chat_completions_url(config.api_url),
        {
            "Authorization": f"Bearer {config.api_key.strip()}",
            "Content-Type": "application/json; charset=utf-8",
        },
        payload,
    )
    return _extract_chat_content(result)


def _call_gemini(config, prompt, temperature, max_tokens):
    base_url = config.api_url.strip().rstrip("/")
    if base_url.endswith(":generateContent"):
        url = base_url
    else:
        url = f"{base_url}/models/{config.model}:generateContent"
    result = _post_json(
        url,
        {
            "x-goog-api-key": config.api_key.strip(),
            "Content-Type": "application/json; charset=utf-8",
        },
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        },
    )
    return _extract_gemini_content(result)


def call_llm(prompt, provider=None, temperature=0.7, max_tokens=1024):
    """Gọi provider đã chọn và luôn trả về chuỗi tiếng Việt khi có lỗi."""
    try:
        config = get_provider_config(provider)
    except ValueError as exc:
        return f"Lỗi: {exc}"

    if not is_configured_api_key(config.api_key):
        env_name = {
            "siliconflow": "SILICONFLOW_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[config.name]
        return f"Lỗi: chưa cấu hình {env_name} cho provider {config.name}."

    try:
        if config.name in ("siliconflow", "openai"):
            return _call_openai_compatible(config, prompt, temperature, max_tokens)
        return _call_gemini(config, prompt, temperature, max_tokens)
    except requests.exceptions.HTTPError as exc:
        logging.error("Provider %s trả về lỗi HTTP: %s", config.name, exc)
        return f"Lỗi khi gọi {config.name}: {exc}"
    except requests.exceptions.RequestException as exc:
        logging.error("Không thể kết nối provider %s: %s", config.name, exc)
        return f"Không thể kết nối {config.name}: {exc}"
    except (TypeError, ValueError, KeyError) as exc:
        logging.error("Không thể phân tích phản hồi từ %s: %s", config.name, exc)
        return f"Phản hồi từ {config.name} không hợp lệ."
