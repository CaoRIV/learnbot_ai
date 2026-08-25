import asyncio

import api_router
from api_router import QuestionRequest, ask_question, check_status
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


def test_ask_endpoint_extracts_pdf_page_citation(monkeypatch):
    monkeypatch.setattr(
        api_router,
        "query_answer",
        lambda question, enable_web_search, model_choice: (
            "Quy trình gồm ba bước [huong-dan.pdf, trang 4]."
        ),
    )

    response = asyncio.run(ask_question(QuestionRequest(question="Quy trình là gì?")))

    assert response["sources"] == [
        {"type": "Tài liệu cục bộ", "source": "huong-dan.pdf", "page": 4}
    ]


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
