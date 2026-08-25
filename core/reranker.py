"""
重排序器 —— 对检索结果进行二次精排

学习要点：
- 两阶段检索（Recall + Rerank）是工业界常用的范式
- Recall 阶段用高效检索（FAISS/BM25）从大量文档中召回候选
- Rerank 阶段用更精确的模型（交叉编码器/LLM）对候选精排
- 交叉编码器比双塔模型更精确，但速度更慢（适合对少量候选精排）
"""

import logging
import re
import threading
from functools import lru_cache
from config import RERANK_METHOD
from llm_provider import call_llm

# 交叉编码器（懒加载 + 线程安全）
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder():
    """懒加载交叉编码器模型（双重检查锁定，线程安全）"""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _cross_encoder = CrossEncoder(
                        'sentence-transformers/distiluse-base-multilingual-cased-v2'
                    )
                    logging.info("Đã tải mô hình CrossEncoder")
                except Exception as e:
                    logging.error("Không thể tải CrossEncoder: %s", e)
                    _cross_encoder = None
    return _cross_encoder


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """使用交叉编码器对检索结果进行重排序"""
    if not docs:
        return []

    encoder = get_cross_encoder()
    if encoder is None:
        logging.warning("CrossEncoder không khả dụng, bỏ qua bước xếp hạng lại")
        return _fallback_results(doc_ids, docs, metadata_list)

    cross_inputs = [[query, doc] for doc in docs]
    try:
        scores = encoder.predict(cross_inputs)
        results = [
            (doc_id, {'content': doc, 'metadata': meta, 'score': float(score)})
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        return results[:top_k]
    except Exception as e:
        logging.error("CrossEncoder xếp hạng lại thất bại: %s", e)
        return _fallback_results(doc_ids, docs, metadata_list)


@lru_cache(maxsize=32)
def get_llm_relevance_score(query, doc):
    """Dùng provider LLM qua API để chấm độ liên quan, có cache."""
    try:
        prompt = f"""Hãy đánh giá độ liên quan giữa truy vấn và đoạn tài liệu dưới đây.
        Điểm 0 nghĩa là hoàn toàn không liên quan, điểm 10 nghĩa là rất liên quan.
        Chỉ trả về một số nguyên từ 0 đến 10, không giải thích.

        Truy vấn: {query}
        Đoạn tài liệu: {doc}
        Điểm liên quan (0-10):"""

        result = call_llm(prompt, temperature=0.0, max_tokens=16).strip()
        try:
            return max(0, min(10, float(result)))
        except ValueError:
            match = re.search(r'\b([0-9]|10)\b', result)
            return float(match.group(1)) if match else 5.0
    except Exception as e:
        logging.error("LLM chấm điểm liên quan thất bại: %s", e)
        return 5.0


def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5):
    """使用 LLM 逐一评分进行重排序"""
    if not docs:
        return []
    results = []
    for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
        score = get_llm_relevance_score(query, doc)
        results.append((doc_id, {'content': doc, 'metadata': meta, 'score': score / 10.0}))
    results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    return results[:top_k]


def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5):
    """对检索结果进行重排序（统一入口）"""
    if method is None:
        method = RERANK_METHOD

    if method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    elif method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    else:
        return _fallback_results(doc_ids, docs, metadata_list)


def _fallback_results(doc_ids, docs, metadata_list):
    """回退方案：按原始顺序返回"""
    return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
            for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
