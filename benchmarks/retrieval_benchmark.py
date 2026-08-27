"""Benchmark FAISS, BM25 và hybrid retrieval mà không gọi LLM API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Một số thư viện Hugging Face đọc cờ offline ngay khi import. Đặt cờ sớm khi
# CLI có ``--offline`` để không phát sinh yêu cầu mạng dù model đã nằm trong cache.
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from config import EMBED_MODEL_NAME, HYBRID_ALPHA, RERANK_MODEL_NAME
from core.bm25_index import BM25IndexManager
from core.retriever import hybrid_merge
from core.vector_store import VectorStore


SUPPORTED_METHODS = ("bm25", "faiss", "hybrid", "hybrid_rerank")
DEFAULT_METHODS = ("bm25", "faiss", "hybrid")
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "data" / "vietnamese_retrieval.json"
)


@dataclass(frozen=True)
class BenchmarkChunk:
    id: str
    content: str
    source: str
    page: int

    @property
    def metadata(self) -> dict:
        return {"source": self.source, "page": self.page}


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    expected_chunk_ids: tuple[str, ...]
    expected_source: str
    expected_page: int


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    version: str
    description: str
    chunks: tuple[BenchmarkChunk, ...]
    questions: tuple[BenchmarkQuestion, ...]


def _required_text(record: dict, field: str, record_label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_label} thiếu trường văn bản hợp lệ: {field}")
    return value.strip()


def _positive_page(record: dict, record_label: str) -> int:
    page = record.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise ValueError(f"{record_label} có số trang không hợp lệ")
    return page


def load_benchmark_dataset(path: str | Path) -> BenchmarkDataset:
    """Đọc và kiểm tra dataset benchmark có nhãn nguồn/trang."""
    dataset_path = Path(path)
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Không tìm thấy dataset benchmark: {dataset_path}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Dataset benchmark không phải JSON UTF-8 hợp lệ: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Dataset benchmark phải là một đối tượng JSON")
    raw_chunks = payload.get("chunks")
    raw_questions = payload.get("questions")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("Dataset benchmark chưa có chunk")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("Dataset benchmark chưa có câu hỏi")

    chunks = []
    chunk_ids = set()
    for index, raw_chunk in enumerate(raw_chunks, start=1):
        if not isinstance(raw_chunk, dict):
            raise ValueError(f"Chunk thứ {index} không phải đối tượng JSON")
        label = f"Chunk thứ {index}"
        chunk = BenchmarkChunk(
            id=_required_text(raw_chunk, "id", label),
            content=_required_text(raw_chunk, "content", label),
            source=_required_text(raw_chunk, "source", label),
            page=_positive_page(raw_chunk, label),
        )
        if chunk.id in chunk_ids:
            raise ValueError(f"ID chunk bị trùng: {chunk.id}")
        chunk_ids.add(chunk.id)
        chunks.append(chunk)

    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    questions = []
    question_ids = set()
    for index, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            raise ValueError(f"Câu hỏi thứ {index} không phải đối tượng JSON")
        label = f"Câu hỏi thứ {index}"
        question_id = _required_text(raw_question, "id", label)
        if question_id in question_ids:
            raise ValueError(f"ID câu hỏi bị trùng: {question_id}")
        question_ids.add(question_id)

        expected_ids = raw_question.get("expected_chunk_ids")
        if (
            not isinstance(expected_ids, list)
            or not expected_ids
            or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in expected_ids)
        ):
            raise ValueError(f"{label} thiếu expected_chunk_ids hợp lệ")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError(f"{label} chứa expected_chunk_ids bị trùng")
        unknown_ids = [chunk_id for chunk_id in expected_ids if chunk_id not in chunk_ids]
        if unknown_ids:
            raise ValueError(
                f"{label} tham chiếu chunk không tồn tại: {', '.join(unknown_ids)}"
            )

        expected_source = _required_text(raw_question, "expected_source", label)
        expected_page = raw_question.get("expected_page")
        if not isinstance(expected_page, int) or isinstance(expected_page, bool) or expected_page < 1:
            raise ValueError(f"{label} có expected_page không hợp lệ")
        if not any(
            chunks_by_id[chunk_id].source == expected_source
            and chunks_by_id[chunk_id].page == expected_page
            for chunk_id in expected_ids
        ):
            raise ValueError(
                f"{label} có nguồn/trang không khớp expected_chunk_ids"
            )

        questions.append(
            BenchmarkQuestion(
                id=question_id,
                question=_required_text(raw_question, "question", label),
                expected_chunk_ids=tuple(expected_ids),
                expected_source=expected_source,
                expected_page=expected_page,
            )
        )

    return BenchmarkDataset(
        name=_required_text(payload, "name", "Dataset"),
        version=_required_text(payload, "version", "Dataset"),
        description=str(payload.get("description") or "").strip(),
        chunks=tuple(chunks),
        questions=tuple(questions),
    )


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def _dataset_fingerprint(dataset: BenchmarkDataset) -> str:
    payload = {
        "name": dataset.name,
        "version": dataset.version,
        "chunks": [
            {
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "page": chunk.page,
            }
            for chunk in dataset.chunks
        ],
        "questions": [
            {
                "id": question.id,
                "question": question.question,
                "expected_chunk_ids": list(question.expected_chunk_ids),
                "expected_source": question.expected_source,
                "expected_page": question.expected_page,
            }
            for question in dataset.questions
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def calculate_metrics(
    questions: Sequence[BenchmarkQuestion],
    rankings: dict[str, list[str]],
    latencies_ms: dict[str, float],
    *,
    top_k: int,
) -> dict:
    """Tính Recall@K, MRR và độ trễ từ danh sách xếp hạng."""
    if top_k < 1:
        raise ValueError("top_k phải lớn hơn hoặc bằng 1")
    if not questions:
        raise ValueError("Không có câu hỏi để tính metrics")

    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    hit_count = 0
    query_results = []
    latency_values = []

    for question in questions:
        retrieved_ids = list(rankings.get(question.id, []))[:top_k]
        relevant_ids = set(question.expected_chunk_ids)
        matched_ids = relevant_ids.intersection(retrieved_ids)
        recall = len(matched_ids) / len(relevant_ids)
        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        latency = float(latencies_ms.get(question.id, 0.0))

        recall_sum += recall
        reciprocal_rank_sum += reciprocal_rank
        latency_values.append(latency)
        if first_relevant_rank is not None:
            hit_count += 1
        query_results.append(
            {
                "question_id": question.id,
                "question": question.question,
                "expected_chunk_ids": list(question.expected_chunk_ids),
                "expected_source": question.expected_source,
                "expected_page": question.expected_page,
                "retrieved_ids": retrieved_ids,
                "recall": recall,
                "first_relevant_rank": first_relevant_rank,
                "latency_ms": latency,
            }
        )

    question_count = len(questions)
    return {
        "recall_at_k": recall_sum / question_count,
        "mrr": reciprocal_rank_sum / question_count,
        "mean_latency_ms": sum(latency_values) / question_count,
        "p95_latency_ms": _percentile_95(latency_values),
        "hit_count": hit_count,
        "question_count": question_count,
        "queries": query_results,
    }


class RetrievalBenchmark:
    """Xây index một lần và đánh giá nhiều phương pháp retrieval."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        *,
        encode_texts_fn: Callable[[list[str]], object] | None = None,
        encode_query_fn: Callable[[str], object] | None = None,
        rerank_fn: Callable | None = None,
    ):
        self.dataset = dataset
        self.encode_texts_fn = encode_texts_fn
        self.encode_query_fn = encode_query_fn
        self.rerank_fn = rerank_fn
        self.vector_store = VectorStore()
        self.bm25_manager = BM25IndexManager()
        self.chunks_by_id = {chunk.id: chunk for chunk in dataset.chunks}

    def _encode_texts(self, texts: list[str]):
        if self.encode_texts_fn is not None:
            return self.encode_texts_fn(texts)
        from core.embeddings import encode_texts

        return encode_texts(texts, show_progress=True)

    def _encode_query(self, query: str):
        if self.encode_query_fn is not None:
            return self.encode_query_fn(query)
        from core.embeddings import encode_query

        return encode_query(query)

    def _rerank(self, query, documents, ids, metadatas, top_k):
        if self.rerank_fn is not None:
            return self.rerank_fn(query, documents, ids, metadatas, top_k)
        from core.reranker import get_cross_encoder, rerank_with_cross_encoder

        if get_cross_encoder() is None:
            raise RuntimeError(
                "Không thể tải CrossEncoder cục bộ; benchmark reranker đã dừng "
                "để tránh ghi nhận nhầm kết quả fallback."
            )
        return rerank_with_cross_encoder(
            query,
            documents,
            ids,
            metadatas,
            top_k=top_k,
        )

    def _build_indexes(self, methods: Sequence[str]) -> dict[str, float]:
        chunks = list(self.dataset.chunks)
        documents = [chunk.content for chunk in chunks]
        ids = [chunk.id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        build_times = {"bm25": 0.0, "faiss": 0.0}

        if any(method in {"bm25", "hybrid", "hybrid_rerank"} for method in methods):
            started = time.perf_counter()
            self.bm25_manager.build_index(documents, ids)
            build_times["bm25"] = (time.perf_counter() - started) * 1000

        if any(method in {"faiss", "hybrid", "hybrid_rerank"} for method in methods):
            started = time.perf_counter()
            embeddings = self._encode_texts(documents)
            self.vector_store.build_index(documents, ids, metadatas, embeddings)
            build_times["faiss"] = (time.perf_counter() - started) * 1000

        return build_times

    def _semantic_results(self, query: str, top_k: int) -> dict:
        query_embedding = self._encode_query(query)
        documents, ids, metadatas = self.vector_store.search(
            query_embedding,
            k=top_k,
        )
        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
        }

    def _result_item(self, chunk_id: str, score=None) -> dict:
        chunk = self.chunks_by_id[chunk_id]
        item = {
            "id": chunk.id,
            "source": chunk.source,
            "page": chunk.page,
        }
        if score is not None:
            item["score"] = float(score)
        return item

    def _retrieve(self, method: str, query: str, top_k: int) -> list[dict]:
        if method == "bm25":
            return [
                self._result_item(result["id"], result["score"])
                for result in self.bm25_manager.search(query, top_k=top_k)
            ]

        semantic_top_k = max(top_k, 10) if method == "hybrid_rerank" else top_k
        semantic = self._semantic_results(query, semantic_top_k)
        if method == "faiss":
            return [self._result_item(chunk_id) for chunk_id in semantic["ids"][0]]

        sparse = self.bm25_manager.search(query, top_k=semantic_top_k)
        merged = hybrid_merge(
            semantic,
            sparse,
            alpha=HYBRID_ALPHA,
            metadata_by_id=self.vector_store.metadatas_map,
        )
        if method == "hybrid":
            return [
                self._result_item(chunk_id, data["score"])
                for chunk_id, data in merged[:top_k]
            ]

        candidate = merged[:semantic_top_k]
        reranked = self._rerank(
            query,
            [data["content"] for _, data in candidate],
            [chunk_id for chunk_id, _ in candidate],
            [data["metadata"] for _, data in candidate],
            top_k,
        )
        return [
            self._result_item(chunk_id, data.get("score"))
            for chunk_id, data in reranked
        ]

    def run(
        self,
        *,
        methods: Iterable[str] = DEFAULT_METHODS,
        top_k: int = 5,
    ) -> dict:
        """Chạy benchmark và trả về báo cáo có thể ghi ra JSON/Markdown."""
        selected_methods = tuple(dict.fromkeys(methods))
        if not selected_methods:
            raise ValueError("Cần chọn ít nhất một phương pháp retrieval")
        unknown_methods = [
            method for method in selected_methods if method not in SUPPORTED_METHODS
        ]
        if unknown_methods:
            raise ValueError(
                f"Phương pháp retrieval không hợp lệ: {', '.join(unknown_methods)}"
            )
        if top_k < 1:
            raise ValueError("top_k phải lớn hơn hoặc bằng 1")

        build_times = self._build_indexes(selected_methods)
        method_reports = {}
        for method in selected_methods:
            rankings = {}
            latencies = {}
            detailed_results = {}
            for question in self.dataset.questions:
                started = time.perf_counter()
                retrieved = self._retrieve(method, question.question, top_k)
                latencies[question.id] = (time.perf_counter() - started) * 1000
                rankings[question.id] = [item["id"] for item in retrieved]
                detailed_results[question.id] = retrieved

            metrics = calculate_metrics(
                self.dataset.questions,
                rankings,
                latencies,
                top_k=top_k,
            )
            for query_result in metrics["queries"]:
                query_result["retrieved"] = detailed_results[query_result["question_id"]]
            method_reports[method] = metrics

        return {
            "dataset": {
                "name": self.dataset.name,
                "version": self.dataset.version,
                "sha256": _dataset_fingerprint(self.dataset),
                "chunk_count": len(self.dataset.chunks),
                "question_count": len(self.dataset.questions),
            },
            "config": {
                "top_k": top_k,
                "embedding_model": EMBED_MODEL_NAME,
                "reranker_model": RERANK_MODEL_NAME,
                "hybrid_alpha": HYBRID_ALPHA,
                "index_build_ms": build_times,
            },
            "methods": method_reports,
        }


def _format_markdown(report: dict) -> str:
    top_k = report["config"]["top_k"]
    dataset = report["dataset"]
    lines = [
        f"# Kết quả benchmark retrieval — {dataset['name']}",
        "",
        f"- Phiên bản dataset: `{dataset['version']}`",
        f"- SHA-256 dataset: `{dataset.get('sha256', 'không có')}`",
        f"- Số chunk: {dataset['chunk_count']}",
        f"- Số câu hỏi: {dataset['question_count']}",
        f"- Mô hình embedding: `{report['config']['embedding_model']}`",
        f"- Mô hình reranker: `{report['config'].get('reranker_model', 'không dùng')}`",
        "",
        f"| Phương pháp | Recall@{top_k} | MRR | Trung bình (ms) | P95 (ms) | Số câu trúng |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in report["methods"].items():
        lines.append(
            f"| {method} | {metrics['recall_at_k']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['mean_latency_ms']:.2f} | {metrics['p95_latency_ms']:.2f} | "
            f"{metrics['hit_count']}/{metrics['question_count']} |"
        )

    misses = []
    for method, metrics in report["methods"].items():
        for query in metrics["queries"]:
            if query["first_relevant_rank"] is None:
                misses.append((method, query))
    lines.extend(["", "## Câu hỏi chưa truy xuất đúng", ""])
    if not misses:
        lines.append("Không có câu hỏi bị trượt trong Top-K.")
    else:
        lines.extend(
            [
                "| Phương pháp | Mã câu hỏi | Câu hỏi | Chunk mong đợi |",
                "| --- | --- | --- | --- |",
            ]
        )
        for method, query in misses:
            expected = ", ".join(query["expected_chunk_ids"])
            lines.append(
                f"| {method} | {query['question_id']} | {query['question']} | {expected} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict,
    output_directory: str | Path,
    base_name: str = "retrieval_benchmark",
) -> tuple[Path, Path]:
    """Ghi cùng một báo cáo ra JSON chi tiết và Markdown tóm tắt."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{base_name}.json"
    markdown_path = output_path / f"{base_name}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_format_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _parse_methods(value: str) -> tuple[str, ...]:
    return tuple(method.strip().lower() for method in value.split(",") if method.strip())


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Đo Recall@K, MRR và độ trễ retrieval tiếng Việt.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--name", default="retrieval_benchmark")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=DEFAULT_METHODS,
        help="Danh sách cách nhau bằng dấu phẩy: bm25,faiss,hybrid,hybrid_rerank",
    )
    parser.add_argument(
        "--fail-under-recall",
        type=float,
        default=None,
        help="Trả mã lỗi nếu có phương pháp có Recall@K thấp hơn ngưỡng.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Chỉ dùng model đã có trong cache, không kiểm tra hoặc tải qua mạng.",
    )
    args = parser.parse_args(argv)

    if args.fail_under_recall is not None and not 0 <= args.fail_under_recall <= 1:
        parser.error("--fail-under-recall phải nằm trong khoảng từ 0 đến 1")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        dataset = load_benchmark_dataset(args.dataset)
        report = RetrievalBenchmark(dataset).run(methods=args.methods, top_k=args.top_k)
        json_path, markdown_path = write_reports(report, args.output_dir, args.name)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(f"Đã ghi báo cáo JSON: {json_path}")
    print(f"Đã ghi báo cáo Markdown: {markdown_path}")
    for method, metrics in report["methods"].items():
        print(
            f"{method}: Recall@{args.top_k}={metrics['recall_at_k']:.4f}, "
            f"MRR={metrics['mrr']:.4f}, trung bình={metrics['mean_latency_ms']:.2f} ms"
        )

    if args.fail_under_recall is not None:
        failed = [
            method
            for method, metrics in report["methods"].items()
            if metrics["recall_at_k"] < args.fail_under_recall
        ]
        if failed:
            print(
                "Không đạt ngưỡng Recall@K cho: " + ", ".join(failed)
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
