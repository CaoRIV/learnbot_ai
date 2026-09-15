import asyncio
from types import SimpleNamespace

import pytest

import api_router
from api_router import QuestionRequest, ask_question, check_status
from core.backup import BackupError, BackupInfo, BackupNotFoundError, BackupRestoreResult
from core.evidence import Citation
from core.generator import AnswerResult
from core.index_snapshot import RestoreResult
from version import __version__


def test_status_reports_current_version_without_credentials(monkeypatch):
    monkeypatch.setattr(api_router, "SILICONFLOW_API_KEY", None)
    monkeypatch.setattr(api_router, "OPENAI_API_KEY", None)
    monkeypatch.setattr(api_router, "GEMINI_API_KEY", None)
    monkeypatch.setattr(api_router, "LLM_PROVIDER", "siliconflow")
    monkeypatch.setattr(
        api_router,
        "get_index_status",
        lambda repository: {
            "document_count": 2,
            "stored_chunk_count": 7,
            "active_snapshot_id": "snapshot-current",
            "index_consistent": True,
        },
    )

    status = asyncio.run(check_status())

    assert status["status"] == "healthy"
    assert status["version"] == __version__
    assert status["siliconflow_configured"] is False
    assert status["openai_configured"] is False
    assert status["gemini_configured"] is False
    assert status["llm_provider"] == "siliconflow"
    assert status["min_relevance_score"] == api_router.MIN_RELEVANCE_SCORE
    assert status["document_count"] == 2
    assert status["stored_chunk_count"] == 7
    assert status["active_snapshot_id"] == "snapshot-current"
    assert status["index_consistent"] is True


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


def test_openapi_exposes_index_rebuild_contract():
    schema = api_router.app.openapi()
    operation = schema["paths"]["/api/index/rebuild"]["post"]
    response_schema = operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert response_schema["$ref"].endswith("/IndexRebuildResponse")


def test_rebuild_endpoint_returns_new_snapshot_and_chunk_count(monkeypatch):
    monkeypatch.setattr(
        api_router,
        "rebuild_indexes_from_storage",
        lambda repository: SimpleNamespace(
            snapshot_id="snapshot-new",
            chunk_count=5,
        ),
    )

    response = asyncio.run(api_router.rebuild_index())

    assert response == {
        "status": "success",
        "message": "Đã xây lại chỉ mục từ dữ liệu SQLite.",
        "snapshot_id": "snapshot-new",
        "total_chunks": 5,
    }


def test_openapi_exposes_backup_contracts():
    schema = api_router.app.openapi()

    list_schema = schema["paths"]["/api/backups"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    create_schema = schema["paths"]["/api/backups"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    restore_schema = schema["paths"]["/api/backups/{backup_id}/restore"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]

    assert list_schema["type"] == "array"
    assert list_schema["items"]["$ref"].endswith("/BackupResponse")
    assert create_schema["$ref"].endswith("/BackupResponse")
    assert restore_schema["$ref"].endswith("/BackupRestoreResponse")


def test_backup_endpoints_return_typed_results(monkeypatch):
    backup = BackupInfo(
        backup_id="backup_20260915T140000000000Z_1234abcd",
        created_at="2026-09-15T14:00:00Z",
        kind="manual",
        document_count=2,
        chunk_count=7,
        snapshot_id="snapshot-source",
        size_bytes=4096,
    )
    restored = BackupRestoreResult(
        backup_id=backup.backup_id,
        safety_backup_id="backup_20260915T140100000000Z_5678abcd",
        document_count=2,
        chunk_count=7,
        snapshot_id="snapshot-restored",
    )
    monkeypatch.setattr(api_router.backup_manager, "list_backups", lambda: [backup])
    monkeypatch.setattr(api_router.backup_manager, "create_backup", lambda: backup)
    monkeypatch.setattr(
        api_router.backup_manager,
        "restore_backup",
        lambda backup_id: restored,
    )

    assert asyncio.run(api_router.list_backups()) == [
        {
            "backup_id": backup.backup_id,
            "created_at": "2026-09-15T14:00:00Z",
            "kind": "manual",
            "document_count": 2,
            "chunk_count": 7,
            "snapshot_id": "snapshot-source",
            "size_bytes": 4096,
        }
    ]
    assert asyncio.run(api_router.create_backup()) == {
        "backup_id": backup.backup_id,
        "created_at": "2026-09-15T14:00:00Z",
        "kind": "manual",
        "document_count": 2,
        "chunk_count": 7,
        "snapshot_id": "snapshot-source",
        "size_bytes": 4096,
    }
    assert asyncio.run(api_router.restore_backup(backup.backup_id)) == {
        "status": "success",
        "message": "Đã khôi phục bản sao lưu.",
        "backup_id": backup.backup_id,
        "safety_backup_id": restored.safety_backup_id,
        "document_count": 2,
        "chunk_count": 7,
        "snapshot_id": "snapshot-restored",
    }


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (BackupNotFoundError("Không tìm thấy bản sao lưu."), 404),
        (BackupError("Checksum backup không hợp lệ."), 409),
    ],
)
def test_restore_backup_maps_expected_errors(monkeypatch, error, expected_status):
    monkeypatch.setattr(
        api_router.backup_manager,
        "restore_backup",
        lambda backup_id: (_ for _ in ()).throw(error),
    )

    with pytest.raises(api_router.HTTPException) as exc_info:
        asyncio.run(
            api_router.restore_backup(
                "backup_20260915T140000000000Z_1234abcd"
            )
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == str(error)


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
