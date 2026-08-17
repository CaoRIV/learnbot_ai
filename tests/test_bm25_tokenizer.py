import re

from core.bm25_index import tokenize_vietnamese


def _legacy_chinese_style_tokenize(text):
    """Mô phỏng cách tokenizer cũ làm vỡ các ký tự tiếng Việt có dấu."""
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+|[^\x00-\x7F]", text)
        if not token.isspace()
    ]


def test_underthesea_preserves_vietnamese_compound_words():
    text = "Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò"

    legacy_tokens = _legacy_chinese_style_tokenize(text)
    vietnamese_tokens = tokenize_vietnamese(text)

    assert vietnamese_tokens != legacy_tokens
    assert "chàng" in vietnamese_tokens
    assert "trai" in vietnamese_tokens
    assert "quảng trị" in vietnamese_tokens
    assert "khởi nghiệp" in vietnamese_tokens
    assert len(vietnamese_tokens) < len(legacy_tokens)
