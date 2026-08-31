from types import SimpleNamespace

import core.generator as generator
import core.retriever as retriever
from core.evidence import Citation, RetrievedEvidence


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
