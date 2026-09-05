"""
向量存储 —— FAISS 向量索引管理

学习要点：
- FAISS (Facebook AI Similarity Search) 是高效的向量相似度检索库
- IndexFlatL2: 暴力搜索，精确但慢。适合小数据集（<1万）
- IndexIVFFlat: 倒排索引，先聚类再搜索。适合中等数据集
- IndexIVFPQ: 乘积量化，牺牲精度换效率。适合大数据集（>10万）
- 本项目根据向量数量自动选择最优索引类型
"""

import logging
import threading
import numpy as np
from faiss import IndexFlatL2, IndexIVFFlat, IndexIVFPQ, clone_index


index_lock = threading.RLock()


class AutoFaissIndex:
    """
    自动选择 FAISS 索引类型的封装类

    根据数据量自动选择最优索引类型：
    - 小数据集（<1万）: FlatL2（精确搜索）
    - 中等数据集（1万-10万）: IVFFlat（近似搜索）
    - 大数据集（>10万）: IVFPQ（高效近似搜索）
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.nlist = None
        self.m = None
        self.nprobe = None
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000

    @property
    def ntotal(self):
        return self.index.ntotal if self.index else 0

    def select_index_type(self, num_vectors):
        """根据向量数量自动选择最优索引类型"""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = "FlatL2"
            self.index = IndexFlatL2(self.dimension)
            self.nprobe = 1
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = "IVFFlat"
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(10, max(1, int(self.nlist * 0.1)))
        else:
            self.index_type = "IVFPQ"
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8)
            self.nprobe = min(32, max(1, int(self.nlist * 0.05)))

        logging.info("Đã chọn chỉ mục %s cho %s vector", self.index_type, num_vectors)
        return self.index_type

    def train(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.train(vectors)

    def add(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"] and not self.index.is_trained:
            self.train(vectors)
        self.index.add(vectors)

    def search(self, query_vectors, k=5):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.nprobe = self.nprobe
        return self.index.search(query_vectors, k)

    def get_index_info(self):
        return {
            "index_type": self.index_type, "dimension": self.dimension,
            "nlist": self.nlist, "nprobe": self.nprobe, "size": self.ntotal
        }


class VectorStore:
    """
    向量存储管理器

    封装 FAISS 索引及其关联的文档内容和元数据映射。
    解决原代码中 4 个全局变量的管理问题。
    """

    def __init__(self):
        self.index = None           # AutoFaissIndex 实例
        self.contents_map = {}      # chunk_id -> 文本内容
        self.metadatas_map = {}     # chunk_id -> 元数据
        self.id_order = []          # 按顺序记录的 chunk_id 列表
        self.snapshot_id = None     # Snapshot bền vững tương ứng trong SQLite

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        构建 FAISS 索引

        Args:
            chunks: 文本片段列表
            chunk_ids: 片段 ID 列表
            metadatas: 元数据列表
            embeddings: 向量数组 (numpy, float32)
        """
        dimension = embeddings.shape[1]
        num_vectors = len(chunks)

        auto_index = AutoFaissIndex(dimension=dimension)
        auto_index.select_index_type(num_vectors)

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        auto_index.add(embeddings)
        self.index = auto_index
        self.snapshot_id = None
        logging.info(
            "Đã xây chỉ mục FAISS gồm %s phân đoạn, loại %s",
            self.index.ntotal,
            auto_index.index_type,
        )

    def add_chunks(self, chunks, chunk_ids, metadatas, embeddings):
        """Thêm các phân đoạn mới mà không tạo lại embedding cũ."""
        if not chunks:
            return
        if not (len(chunks) == len(chunk_ids) == len(metadatas) == len(embeddings)):
            raise ValueError("Số phân đoạn, ID, metadata và embedding không khớp")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Danh sách phân đoạn mới chứa ID trùng")
        if any(chunk_id in self.contents_map for chunk_id in chunk_ids):
            raise ValueError("ID phân đoạn mới đã tồn tại trong kho vector")
        if self.index is None:
            self.build_index(chunks, chunk_ids, metadatas, embeddings)
            return
        if embeddings.shape[1] != self.index.dimension:
            raise ValueError("Số chiều embedding mới không khớp chỉ mục FAISS")

        self.index.add(embeddings)
        for chunk_id, chunk, metadata in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = metadata
            self.id_order.append(chunk_id)
        self.snapshot_id = None
        logging.info("Đã thêm %s phân đoạn mới vào kho vector", len(chunks))

    def clone(self):
        """Tạo bản sao độc lập để xây candidate index an toàn."""
        cloned = VectorStore()
        if self.index is not None:
            auto_index = AutoFaissIndex(dimension=self.index.dimension)
            auto_index.index = clone_index(self.index.index)
            auto_index.index_type = self.index.index_type
            auto_index.nlist = self.index.nlist
            auto_index.m = self.index.m
            auto_index.nprobe = self.index.nprobe
            cloned.index = auto_index
        cloned.contents_map = dict(self.contents_map)
        cloned.metadatas_map = {
            chunk_id: dict(metadata)
            for chunk_id, metadata in self.metadatas_map.items()
        }
        cloned.id_order = list(self.id_order)
        cloned.snapshot_id = self.snapshot_id
        return cloned

    def search(self, query_embedding, k=10):
        """
        搜索最相似的向量

        Returns:
            (docs, doc_ids, metadatas)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas = [], [], []
            for faiss_idx in I[0]:
                if faiss_idx != -1 and faiss_idx < len(self.id_order):
                    original_id = self.id_order[faiss_idx]
                    if original_id in self.contents_map:
                        docs.append(self.contents_map[original_id])
                        doc_ids.append(original_id)
                        metadatas.append(self.metadatas_map.get(original_id, {}))
            return docs, doc_ids, metadatas
        except Exception as e:
            logging.error("Truy xuất FAISS gặp lỗi: %s", e)
            return [], [], []

    @property
    def is_ready(self):
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_chunks(self):
        return self.index.ntotal if self.index is not None else 0

    def clear(self):
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        self.snapshot_id = None
        logging.info("Đã xóa kho vector trong bộ nhớ")

    def replace_with(self, other):
        """Thay toàn bộ dữ liệu bằng một kho đã xây dựng thành công."""
        if not isinstance(other, VectorStore):
            raise TypeError("Kho thay thế phải là một VectorStore")
        self.index = other.index
        self.contents_map = other.contents_map
        self.metadatas_map = other.metadatas_map
        self.id_order = other.id_order
        self.snapshot_id = other.snapshot_id
        logging.info("Đã thay kho vector với %s phân đoạn", self.total_chunks)


# 模块级单例
vector_store = VectorStore()
