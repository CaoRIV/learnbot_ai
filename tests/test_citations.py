from types import SimpleNamespace

import core.generator as generator
import core.retriever as retriever
from core.evidence import Citation, RetrievedEvidence, filter_relevant_evidence


def test_citation_serializes_structured_and_legacy_local_source():
    citation = Citation(
        document="giao-trinh.pdf",
        page="7",
        chunk_id="chunk-7",
        score="0.875",
    )

    assert citation.as_dict() == {
        "document": "giao-trinh.pdf",
        "page": 7,
        "chunk_id": "chunk-7",
        "score": 0.875,
        "type": "document",
        "url": None,
    }
    assert citation.as_legacy_source() == {
        "type": "Tài liệu cục bộ",
        "source": "giao-trinh.pdf",
        "page": 7,
    }


def test_citation_serializes_web_source_without_page():
    citation = Citation(
        document="Thông báo mới",
        chunk_id="web:https://example.test/thong-bao",
        source_type="web",
        url="https://example.test/thong-bao",
    )

    assert citation.as_legacy_source() == {
        "type": "Nguồn web",
        "source": "Thông báo mới",
        "url": "https://example.test/thong-bao",
    }


def test_retrieve_evidence_preserves_retrieval_metadata_and_score(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "_recursive_retrieval_with_scores",
        lambda **kwargs: (
            ["Nội dung phân đoạn."],
            ["doc:chunk-2"],
            [{"source": "tai-lieu.pdf", "page": 2}],
            [0.83],
        ),
    )

    evidence = retriever.retrieve_evidence("Nội dung là gì?")

    assert evidence == [
        RetrievedEvidence(
            content="Nội dung phân đoạn.",
            citation=Citation(
                document="tai-lieu.pdf",
                page=2,
                chunk_id="doc:chunk-2",
                score=0.83,
            ),
            metadata={"source": "tai-lieu.pdf", "page": 2},
        )
    ]


def test_query_answer_result_returns_citations_and_keeps_string_wrapper(monkeypatch):
    evidence = RetrievedEvidence(
        content="Nội dung ở trang 5.",
        citation=Citation(
            document="quy-dinh.pdf",
            page=5,
            chunk_id="quy-dinh:5",
            score=0.94,
        ),
        metadata={"source": "quy-dinh.pdf", "page": 5},
    )
    monkeypatch.setattr(generator, "vector_store", SimpleNamespace(is_ready=True))
    monkeypatch.setattr(generator, "retrieve_evidence", lambda **kwargs: [evidence])
    monkeypatch.setattr(generator, "detect_conflicts", lambda sources: False)
    monkeypatch.setattr(
        generator,
        "call_llm",
        lambda *args, **kwargs: "Câu trả lời [quy-dinh.pdf, trang 5].",
    )

    result = generator.query_answer_result("Quy định là gì?")

    assert result.answer == "Câu trả lời [quy-dinh.pdf, trang 5]."
    assert result.citations == (evidence.citation,)
    assert generator.query_answer("Quy định là gì?") == result.answer


def test_filter_relevant_evidence_applies_threshold_and_keeps_unscored_web():
    low = RetrievedEvidence(
        content="Nội dung điểm thấp.",
        citation=Citation("thap.pdf", "chunk-low", score=0.34),
    )
    boundary = RetrievedEvidence(
        content="Nội dung đúng ngưỡng.",
        citation=Citation("dat.pdf", "chunk-boundary", score=0.35),
    )
    unscored_local = RetrievedEvidence(
        content="Nội dung cục bộ không có điểm.",
        citation=Citation("khong-diem.pdf", "chunk-unscored"),
    )
    web = RetrievedEvidence(
        content="Kết quả web.",
        citation=Citation(
            "Nguồn web",
            "web:https://example.test",
            source_type="web",
            url="https://example.test",
        ),
    )

    filtered = filter_relevant_evidence(
        [low, boundary, unscored_local, web],
        min_score=0.35,
    )

    assert filtered == [boundary, web]


def test_query_answer_result_excludes_low_score_evidence_from_prompt(monkeypatch):
    high = RetrievedEvidence(
        content="Bằng chứng đủ liên quan.",
        citation=Citation("dung.pdf", "chunk-high", page=2, score=0.8),
        metadata={"source": "dung.pdf", "page": 2},
    )
    low = RetrievedEvidence(
        content="Phân đoạn không liên quan phải bị loại.",
        citation=Citation("sai.pdf", "chunk-low", page=9, score=0.2),
        metadata={"source": "sai.pdf", "page": 9},
    )
    captured = {}
    monkeypatch.setattr(generator, "vector_store", SimpleNamespace(is_ready=True))
    monkeypatch.setattr(generator, "MIN_RELEVANCE_SCORE", 0.35)
    monkeypatch.setattr(generator, "retrieve_evidence", lambda **kwargs: [low, high])
    monkeypatch.setattr(generator, "detect_conflicts", lambda sources: False)

    def fake_call_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Câu trả lời [dung.pdf, trang 2]."

    monkeypatch.setattr(generator, "call_llm", fake_call_llm)

    result = generator.query_answer_result("Thông tin đúng là gì?")

    assert result.citations == (high.citation,)
    assert "Bằng chứng đủ liên quan." in captured["prompt"]
    assert "Phân đoạn không liên quan phải bị loại." not in captured["prompt"]
