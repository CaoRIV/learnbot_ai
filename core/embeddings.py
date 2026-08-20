"""Mô hình embedding ánh xạ văn bản vào không gian vector.

Mô hình mặc định hỗ trợ đa ngôn ngữ, bao gồm tiếng Việt, và tạo vector
384 chiều phù hợp với kho FAISS hiện tại. Mô hình được tải về ở lần chạy đầu.
"""

import logging
import numpy as np
from functools import lru_cache

EMBED_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


@lru_cache(maxsize=1)
def get_embed_model():
    """Tải mô hình embedding một lần rồi tái sử dụng từ bộ nhớ đệm."""
    from sentence_transformers import SentenceTransformer

    logging.info("Đang tải mô hình embedding: %s", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    logging.info(
        "Đã tải mô hình embedding, số chiều đầu ra: %s",
        model.get_sentence_embedding_dimension(),
    )
    return model


def encode_texts(texts, show_progress=False):
    """Mã hóa danh sách văn bản thành vector.

    Tham số:
        texts: Danh sách văn bản.
        show_progress: Có hiển thị thanh tiến trình hay không.

    Trả về:
        Mảng ``float32`` có kích thước ``(số_văn_bản, số_chiều)``.
    """
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return np.array(embeddings).astype("float32")


def encode_query(query):
    """Mã hóa một truy vấn thành vector.

    Trả về:
        Mảng ``float32`` có kích thước ``(1, số_chiều)``.
    """
    model = get_embed_model()
    embedding = model.encode([query])
    return np.array(embedding).astype("float32")
