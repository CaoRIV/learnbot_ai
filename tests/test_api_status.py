import asyncio

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
    assert response["sources"] == [
        {"type": "Tài liệu cục bộ", "source": "huong-dan.pdf", "page": 4}
    ]
    assert response["metadata"]["citation_count"] == 1


def test_openapi_exposes_structured_citation_contract():
    schema = api_router.app.openapi()
    citation_schema = schema["components"]["schemas"]["CitationResponse"]

    assert set(citation_schema["required"]) == {
        "document",
        "chunk_id",
        "type",
    }
    assert citation_schema["properties"]["type"]["enum"] == ["document", "web"]


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
