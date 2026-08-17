import asyncio

from api_router import check_status
from version import __version__


def test_status_reports_current_version_without_credentials():
    status = asyncio.run(check_status())

    assert status["status"] == "healthy"
    assert status["version"] == __version__
    assert status["siliconflow_configured"] is False
    assert status["openai_configured"] is False
    assert status["gemini_configured"] is False
    assert status["llm_provider"] == "siliconflow"
