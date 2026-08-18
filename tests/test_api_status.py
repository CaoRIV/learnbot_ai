import asyncio

import api_router
from api_router import QuestionRequest, ask_question, check_status
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
