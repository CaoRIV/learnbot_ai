"""Cấu hình tập trung cho learnbot_ai."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

dotenv_path = PROJECT_ROOT / ".env"
if not dotenv_path.exists():
    dotenv_path = PROJECT_ROOT / "example.env"
    logging.warning(
        "Không tìm thấy .env, đang dùng example.env. "
        "Hãy sao chép example.env thành .env và điền API key thật."
    )
load_dotenv(dotenv_path)


def resolve_database_path(value=None):
    """Chuẩn hóa đường dẫn SQLite tương đối theo thư mục gốc dự án."""
    configured_path = Path(value or os.getenv("DATABASE_PATH", "data/learnbot.db"))
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    return configured_path.resolve()


DATABASE_PATH = resolve_database_path()


def resolve_index_directory(value=None):
    """Chuẩn hóa thư mục snapshot chỉ mục theo thư mục gốc dự án."""
    configured_path = Path(value or os.getenv("INDEX_DIRECTORY", "data/indexes"))
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    return configured_path.resolve()


INDEX_DIRECTORY = resolve_index_directory()


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
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
HYBRID_ALPHA = 0.7
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 5
MAX_RETRIEVAL_ITERATIONS = 3
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")

DEFAULT_MIN_RELEVANCE_SCORE = 0.35


def resolve_min_relevance_score(value=None):
    """Đọc ngưỡng liên quan 0–1 và dùng mặc định khi cấu hình sai."""
    configured_value = (
        os.getenv("MIN_RELEVANCE_SCORE", str(DEFAULT_MIN_RELEVANCE_SCORE))
        if value is None
        else value
    )
    try:
        score = float(configured_value)
    except (TypeError, ValueError):
        score = None

    if score is None or not 0.0 <= score <= 1.0:
        logging.warning(
            "MIN_RELEVANCE_SCORE=%s không hợp lệ; sử dụng giá trị mặc định %.2f",
            configured_value,
            DEFAULT_MIN_RELEVANCE_SCORE,
        )
        return DEFAULT_MIN_RELEVANCE_SCORE
    return score


MIN_RELEVANCE_SCORE = resolve_min_relevance_score()

# Giới hạn tài liệu tải lên để tránh chiếm quá nhiều RAM trên máy cấu hình thấp.
try:
    MAX_UPLOAD_SIZE_MB = max(1, int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")))
except ValueError:
    logging.warning("MAX_UPLOAD_SIZE_MB không hợp lệ; sử dụng giá trị mặc định 25 MB")
    MAX_UPLOAD_SIZE_MB = 25

# Cấu hình runtime nhẹ, phù hợp Windows
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://huggingface.co")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
