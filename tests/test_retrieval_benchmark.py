import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.retrieval_benchmark import (
    RetrievalBenchmark,
    calculate_metrics,
    load_benchmark_dataset,
    write_reports,
)


CANONICAL_DATASET = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "data"
    / "vietnamese_retrieval.json"
)


def _write_small_dataset(path: Path) -> Path:
    payload = {
        "name": "bo-kiem-thu-nho",
        "version": "1.0",
        "description": "Dữ liệu kiểm thử benchmark.",
        "chunks": [
            {
                "id": "chunk-password",
                "content": "Đặt lại mật khẩu tại cổng tài khoản nội bộ.",
                "source": "cong-nghe.pdf",
                "page": 2,
            },
            {
                "id": "chunk-library",
                "content": "Thư viện mở cửa từ 8 giờ đến 20 giờ.",
                "source": "thu-vien.pdf",
                "page": 4,
            },
            {
                "id": "chunk-tuition",
                "content": "Học phí được thanh toán qua chuyển khoản ngân hàng.",
                "source": "hoc-phi.pdf",
                "page": 7,
            },
        ],
        "questions": [
            {
                "id": "q-password",
                "question": "Đặt lại mật khẩu ở đâu?",
                "expected_chunk_ids": ["chunk-password"],
                "expected_source": "cong-nghe.pdf",
                "expected_page": 2,
            },
            {
                "id": "q-library",
                "question": "Thư viện mở cửa lúc nào?",
                "expected_chunk_ids": ["chunk-library"],
                "expected_source": "thu-vien.pdf",
                "expected_page": 4,
            },
            {
                "id": "q-tuition",
                "question": "Thanh toán học phí bằng cách nào?",
                "expected_chunk_ids": ["chunk-tuition"],
                "expected_source": "hoc-phi.pdf",
                "expected_page": 7,
            },
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _deterministic_embeddings(texts):
    keywords = ("mật khẩu", "thư viện", "học phí")
    rows = []
    for text in texts:
        lowered = text.lower()
        rows.append([1.0 if keyword in lowered else 0.0 for keyword in keywords])
    return np.asarray(rows, dtype=np.float32)


def test_canonical_dataset_has_vietnamese_labels_and_valid_pages():
    dataset = load_benchmark_dataset(CANONICAL_DATASET)

    assert 20 <= len(dataset.questions) <= 50
    assert len(dataset.chunks) >= len(dataset.questions)
    chunks_by_id = {chunk.id: chunk for chunk in dataset.chunks}
    for question in dataset.questions:
        assert question.question.strip()
        assert question.expected_chunk_ids
        expected = chunks_by_id[question.expected_chunk_ids[0]]
        assert question.expected_source == expected.source
        assert question.expected_page == expected.page
        assert expected.page >= 1


def test_dataset_rejects_unknown_expected_chunk(tmp_path):
    path = _write_small_dataset(tmp_path / "dataset.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["questions"][0]["expected_chunk_ids"] = ["chunk-khong-ton-tai"]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="không tồn tại"):
        load_benchmark_dataset(path)


def test_calculate_metrics_returns_recall_mrr_and_latency(tmp_path):
    dataset = load_benchmark_dataset(_write_small_dataset(tmp_path / "dataset.json"))
    rankings = {
        "q-password": ["chunk-password", "chunk-library"],
        "q-library": ["chunk-password", "chunk-library"],
        "q-tuition": ["chunk-password", "chunk-library"],
    }
    latencies = {"q-password": 1.0, "q-library": 2.0, "q-tuition": 3.0}

    metrics = calculate_metrics(dataset.questions, rankings, latencies, top_k=2)

    assert metrics["recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert metrics["mean_latency_ms"] == pytest.approx(2.0)
    assert metrics["p95_latency_ms"] == pytest.approx(3.0)
    assert metrics["hit_count"] == 2
    assert metrics["queries"][1]["first_relevant_rank"] == 2


def test_benchmark_runs_all_retrieval_methods_without_llm(tmp_path, monkeypatch):
    dataset = load_benchmark_dataset(_write_small_dataset(tmp_path / "dataset.json"))

    def fail_if_llm_is_called(*args, **kwargs):
        pytest.fail("Benchmark retrieval không được gọi LLM")

    monkeypatch.setattr("llm_provider.call_llm", fail_if_llm_is_called)
    benchmark = RetrievalBenchmark(
        dataset,
        encode_texts_fn=lambda texts: _deterministic_embeddings(texts),
        encode_query_fn=lambda query: _deterministic_embeddings([query]),
    )

    report = benchmark.run(methods=("bm25", "faiss", "hybrid"), top_k=3)

    assert set(report["methods"]) == {"bm25", "faiss", "hybrid"}
    assert report["dataset"]["question_count"] == 3
    for method in report["methods"].values():
        assert method["recall_at_k"] == 1.0
        assert method["mrr"] == 1.0
        for query_result in method["queries"]:
            assert query_result["retrieved"]
            first = query_result["retrieved"][0]
            assert first["source"].endswith(".pdf")
            assert first["page"] >= 1


def test_hybrid_rerank_uses_injected_cross_encoder_path(tmp_path, monkeypatch):
    dataset = load_benchmark_dataset(_write_small_dataset(tmp_path / "dataset.json"))
    rerank_calls = []

    def fake_rerank(query, documents, ids, metadatas, top_k):
        rerank_calls.append(query)
        return [
            (
                chunk_id,
                {"content": document, "metadata": metadata, "score": 1.0},
            )
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ][:top_k]

    monkeypatch.setattr(
        "llm_provider.call_llm",
        lambda *args, **kwargs: pytest.fail("Không được gọi LLM khi rerank"),
    )
    benchmark = RetrievalBenchmark(
        dataset,
        encode_texts_fn=lambda texts: _deterministic_embeddings(texts),
        encode_query_fn=lambda query: _deterministic_embeddings([query]),
        rerank_fn=fake_rerank,
    )

    report = benchmark.run(methods=("hybrid_rerank",), top_k=3)

    assert len(rerank_calls) == len(dataset.questions)
    assert report["methods"]["hybrid_rerank"]["recall_at_k"] == 1.0


def test_write_reports_creates_reproducible_json_and_markdown(tmp_path):
    report = {
        "dataset": {
            "name": "bo-du-lieu",
            "version": "1.0",
            "chunk_count": 3,
            "question_count": 2,
        },
        "config": {"top_k": 5, "embedding_model": "model-test"},
        "methods": {
            "hybrid": {
                "recall_at_k": 1.0,
                "mrr": 0.75,
                "mean_latency_ms": 2.5,
                "p95_latency_ms": 3.0,
                "hit_count": 2,
                "question_count": 2,
                "queries": [],
            }
        },
    }

    json_path, markdown_path = write_reports(report, tmp_path, "ket-qua")

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Recall@5" in markdown
    assert "0.7500" in markdown
    assert "hybrid" in markdown
