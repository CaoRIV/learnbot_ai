"""Chỉ mục BM25 với tokenizer tiếng Việt Underthesea."""

import logging

import numpy as np
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize


def tokenize_vietnamese(text):
    """Tách từ tiếng Việt và chuẩn hóa chữ thường cho BM25."""
    if not text:
        return []
    return [token.strip().lower() for token in word_tokenize(text) if token.strip()]


class BM25IndexManager:
    """Xây dựng, tìm kiếm và quản lý chỉ mục BM25 tiếng Việt."""

    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    def build_index(self, documents, doc_ids):
        """Xây chỉ mục BM25 từ danh sách tài liệu."""
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        self.tokenized_corpus = [tokenize_vietnamese(doc) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logging.info("Đã xây chỉ mục BM25 cho %s tài liệu", len(documents))
        return True

    def search(self, query, top_k=5):
        """Tìm các tài liệu liên quan bằng BM25."""
        if not self.bm25_index:
            return []

        tokenized_query = tokenize_vietnamese(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                results.append({
                    'id': self.doc_mapping[idx],
                    'score': float(bm25_scores[idx]),
                    'content': self.raw_corpus[idx]
                })
        return results

    def clear(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    def replace_with(self, other):
        """Thay toàn bộ dữ liệu bằng một chỉ mục đã xây dựng thành công."""
        if not isinstance(other, BM25IndexManager):
            raise TypeError("Chỉ mục thay thế phải là một BM25IndexManager")
        self.bm25_index = other.bm25_index
        self.doc_mapping = other.doc_mapping
        self.tokenized_corpus = other.tokenized_corpus
        self.raw_corpus = other.raw_corpus
        logging.info("Đã thay chỉ mục BM25 với %s tài liệu", len(self.raw_corpus))


# Singleton dùng chung trong tiến trình
bm25_manager = BM25IndexManager()
