"""Cấu hình tập trung cho learnbot_ai."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning(
        "Không tìm thấy .env, đang dùng example.env. "
        "Hãy sao chép example.env thành .env và điền API key thật."
    )
load_dotenv(dotenv_path)


def is_configured_api_key(api_key):
    """Kiểm tra API key có phải là giá trị người dùng đã cấu hình hay không."""
    return bool(
        api_key
        and api_key.strip()
        and not api_key.strip().lower().startswith(("your", "replace_", "changeme"))
    )


# Tìm kiếm web
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

# Provider LLM chạy qua API bên ngoài
SUPPORTED_LLM_PROVIDERS = ("siliconflow", "openai", "gemini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "siliconflow").strip().lower()
if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
    logging.warning(
        "LLM_PROVIDER=%s không hợp lệ; chuyển về siliconflow. Các giá trị hỗ trợ: %s",
        LLM_PROVIDER,
        ", ".join(SUPPORTED_LLM_PROVIDERS),
    )
    LLM_PROVIDER = "siliconflow"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv(
    "SILICONFLOW_API_URL",
    "https://api.siliconflow.cn/v1/chat/completions",
)
SILICONFLOW_MODEL_NAME = os.getenv(
    "SILICONFLOW_MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL",
    "https://api.openai.com/v1/chat/completions",
)
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")

MODEL_CHOICES = list(SUPPORTED_LLM_PROVIDERS)
MODEL_DISPLAY_NAMES = {
    "siliconflow": "SiliconFlow API",
    "openai": "OpenAI API",
    "gemini": "Gemini API",
}
DEFAULT_MODEL_CHOICE = LLM_PROVIDER


def choose_default_provider(provider=None):
    """Chuẩn hóa provider được truyền vào hoặc lấy từ biến môi trường."""
    selected = (provider or os.getenv("LLM_PROVIDER") or "siliconflow").strip().lower()
    if selected not in SUPPORTED_LLM_PROVIDERS:
        return "siliconflow"
    return selected


# Tham số RAG
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
HYBRID_ALPHA = 0.7
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5
MAX_RETRIEVAL_ITERATIONS = 3
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")

# Cấu hình runtime nhẹ, phù hợp Windows
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
