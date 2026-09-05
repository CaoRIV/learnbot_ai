import asyncio

import pytest

import api_router
from api_router import QuestionRequest, ask_question, check_status
from core.evidence import Citation
from core.generator import AnswerResult
from core.index_snapshot import RestoreResult
from version import __version__


def test_status_reports_current_version_without_credentials(monkeypatch):
    monkeypatch.setattr(api_router, "SILICONFLOW_API_KEY", None)
    monkeypatch.setattr(api_router, "OPENAI_API_KEY", None)
    monkeypatch.setattr(api_router, "GEMINI_API_KEY", None)
    monkeypatch.setattr(api_router, "LLM_PROVIDER", "siliconflow")

    status = asyncio.run(check_status())

    assert status["status"] == "healthy"
    assert status["version"] == __version__
    assert status["siliconflow_configured"] is False
    assert status["openai_configured"] is False
    assert status["gemini_configured"] is False
    assert status["llm_provider"] == "siliconflow"
    assert status["min_relevance_score"] == api_router.MIN_RELEVANCE_SCORE


def test_ask_endpoint_uses_structured_citations_instead_of_answer_regex(monkeypatch):
    monkeypatch.setattr(
        api_router,
        "query_answer_result",
        lambda question, enable_web_search, model_choice: AnswerResult(
            answer=(
                "Quy trình gồm ba bước [nguon-do-llm-tu-tao.pdf, trang 99]."
            ),
            citations=(
                Citation(
                    document="huong-dan.pdf",
                    page=4,
                    chunk_id="huong-dan.pdf:chunk-4",
                    score=0.91,
                ),
            ),
        ),
    )

    response = asyncio.run(ask_question(QuestionRequest(question="Quy trình là gì?")))

    assert response["citations"] == [
        {
            "document": "huong-dan.pdf",
            "page": 4,
            "chunk_id": "huong-dan.pdf:chunk-4",
            "score": 0.91,
            "type": "document",
            "url": None,
        }
    ]
    assert response["answer_status"] == "answered"
    assert response["sources"] == [
        {"type": "Tài liệu cục bộ", "source": "huong-dan.pdf", "page": 4}
    ]
    assert response["metadata"]["citation_count"] == 1
    assert response["metadata"]["min_relevance_score"] == api_router.MIN_RELEVANCE_SCORE


def test_openapi_exposes_structured_citation_contract():
    schema = api_router.app.openapi()
    citation_schema = schema["components"]["schemas"]["CitationResponse"]
    answer_schema = schema["components"]["schemas"]["AnswerResponse"]

    assert set(citation_schema["required"]) == {
        "document",
        "chunk_id",
        "type",
    }
    assert citation_schema["properties"]["type"]["enum"] == ["document", "web"]
    assert answer_schema["properties"]["answer_status"]["enum"] == [
        "answered",
        "insufficient_evidence",
        "empty_knowledge_base",
        "error",
    ]


def test_openapi_exposes_document_list_contract():
    schema = api_router.app.openapi()
    operation = schema["paths"]["/api/documents"]["get"]
    response_schema = operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    document_schema = schema["components"]["schemas"]["DocumentResponse"]

    assert response_schema["type"] == "array"
    assert response_schema["items"]["$ref"].endswith("/DocumentResponse")
    assert set(document_schema["required"]) == {
        "id",
        "source_name",
        "file_size",
        "status",
        "chunk_count",
        "created_at",
        "updated_at",
    }
    assert document_schema["properties"]["status"]["enum"] == [
        "processing",
        "ready",
        "failed",
    ]


def test_openapi_exposes_document_delete_contract():
    schema = api_router.app.openapi()
    operation = schema["paths"]["/api/documents/{document_id}"]["delete"]
    response_schema = operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert response_schema["$ref"].endswith("/DocumentDeleteResponse")


def test_delete_endpoint_returns_remaining_chunk_count(monkeypatch):
    monkeypatch.setattr(
        api_router,
        "delete_indexed_document",
        lambda document_id, repository: 3 if document_id == "doc-1" else None,
    )

    response = asyncio.run(api_router.delete_document("doc-1"))

    assert response == {
        "status": "success",
        "message": "Đã xóa tài liệu khỏi kho tri thức.",
        "document_id": "doc-1",
        "remaining_chunks": 3,
    }


def test_delete_endpoint_returns_404_for_unknown_document(monkeypatch):
    monkeypatch.setattr(
        api_router,
        "delete_indexed_document",
        lambda document_id, repository: None,
    )

    with pytest.raises(api_router.HTTPException) as exc_info:
        asyncio.run(api_router.delete_document("missing"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Không tìm thấy tài liệu cần xóa"


def test_ask_endpoint_exposes_insufficient_evidence_status(monkeypatch):
    monkeypatch.setattr(
        api_router,
        "query_answer_result",
        lambda *args, **kwargs: AnswerResult(
            answer="Không đủ bằng chứng.",
            answer_status="insufficient_evidence",
        ),
    )

    response = asyncio.run(ask_question(QuestionRequest(question="Ngoài phạm vi?")))

    assert response["answer_status"] == "insufficient_evidence"
    assert response["citations"] == []
    assert response["sources"] == []


def test_api_lifespan_restores_active_snapshot(monkeypatch):
    expected = RestoreResult(
        True,
        "Đã khôi phục 3 phân đoạn từ snapshot.",
        snapshot_id="snapshot-test",
        chunk_count=3,
    )
    monkeypatch.setattr(api_router, "restore_indexes", lambda: expected)

    async def run_lifespan():
        async with api_router.lifespan(api_router.app):
            assert api_router.app.state.index_restore_result == expected

    asyncio.run(run_lifespan())
