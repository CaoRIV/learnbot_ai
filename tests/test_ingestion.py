import asyncio
import hashlib
import sqlite3
from pathlib import Path
import pytest
from fastapi import HTTPException

import api_router
import core.ingestion as ingestion_module
from core.bm25_index import BM25IndexManager, bm25_manager
from core.ingestion import (
    DocumentSource,
    IngestionResult,
    calculate_file_hash,
    ingest_documents,
)
from core.index_snapshot import IndexSnapshotStore
from core.storage import SQLiteRepository
from core.vector_store import VectorStore, vector_store


import numpy as np

@pytest.fixture
def repository(tmp_path):
    return SQLiteRepository(tmp_path / "learnbot-test.db")


def _store_ready_document(repository, document_id, source_name, chunk_id, content):
    repository.upsert_document(
        document_id=document_id,
        source_name=source_name,
        content_hash=f"hash-{document_id}",
        file_size=len(content.encode("utf-8")),
        status="ready",
    )
    repository.replace_chunks(
        document_id,
        [
            {
                "id": chunk_id,
                "chunk_index": 0,
                "content": content,
                "page": 1,
                "metadata": {
                    "source": source_name,
                    "doc_id": document_id,
                    "page": 1,
                },
            }
        ],
    )


def _build_runtime_indexes(chunks, chunk_ids, metadatas):
    runtime_vector = VectorStore()
    runtime_vector.build_index(
        chunks,
        chunk_ids,
        metadatas,
        np.ones((len(chunks), 384), dtype=np.float32),
    )
    runtime_bm25 = BM25IndexManager()
    runtime_bm25.build_index(chunks, chunk_ids)
    return runtime_vector, runtime_bm25


def test_calculate_file_hash_uses_sha256(tmp_path):
    test_file = tmp_path / "hash.txt"
    content = b"learnbot-ai"
    test_file.write_bytes(content)

    assert calculate_file_hash(test_file) == hashlib.sha256(content).hexdigest()


def test_ingest_documents_success(tmp_path, monkeypatch, repository):
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("Đây là tài liệu thử nghiệm cho pipeline RAG.", encoding="utf-8")

    # Mock encode_texts to return dummy embeddings
    monkeypatch.setattr(
        "core.ingestion.encode_texts",
        lambda chunks, show_progress=False: np.ones((len(chunks), 384), dtype=np.float32),
    )

    result = ingest_documents(
        [DocumentSource(path=str(test_file), display_name="test_doc.txt")],
        repository=repository,
    )

    assert isinstance(result, IngestionResult)
    assert result.success is True
    assert result.total_files == 1
    assert result.processed_files == 1
    assert result.failed_files == 0
    assert result.chunk_count > 0
    assert result.filenames == ["test_doc.txt"]
    assert vector_store.total_chunks > 0
    stored_documents = repository.list_documents()
    assert len(stored_documents) == 1
    assert stored_documents[0]["status"] == "ready"
    assert repository.get_chunks(stored_documents[0]["id"])


def test_ingest_documents_unsupported_extension(tmp_path, repository):
    test_file = tmp_path / "script.exe"
    test_file.write_bytes(b"binary data")

    result = ingest_documents(
        [DocumentSource(path=str(test_file), display_name="script.exe")],
        repository=repository,
    )

    assert result.success is False
    assert result.processed_files == 0
    assert result.failed_files == 1
    assert "chưa được hỗ trợ" in result.message


def test_ingest_documents_atomic_failure_preserves_existing_store(
    tmp_path, monkeypatch, repository
):
    # Establish existing state in vector store and BM25
    vector_store.build_index(["Old chunk"], ["old_1"], [{"source": "old"}], np.ones((1, 384), dtype=np.float32))
    bm25_manager.build_index(["Old chunk"], ["old_1"])
    initial_chunks = vector_store.total_chunks

    test_file = tmp_path / "bad_doc.pdf"
    test_file.write_bytes(b"invalid pdf content")

    # Mock extract_text_by_page to raise exception
    monkeypatch.setattr(
        "core.ingestion.extract_text_by_page",
        lambda path: [],
    )

    result = ingest_documents(
        [DocumentSource(path=str(test_file), display_name="bad_doc.pdf")],
        repository=repository,
    )

    assert result.success is False
    # Ensure initial vector store data is untouched
    assert vector_store.total_chunks == initial_chunks


def test_ingest_documents_skips_duplicate_extraction(
    tmp_path, monkeypatch, repository
):
    test_file = tmp_path / "duplicate.txt"
    test_file.write_text("Nội dung không cần trích xuất lại.", encoding="utf-8")
    monkeypatch.setattr(
        "core.ingestion.encode_texts",
        lambda chunks, show_progress=False: np.ones(
            (len(chunks), 384), dtype=np.float32
        ),
    )

    first = ingest_documents(
        [DocumentSource(path=str(test_file), display_name="duplicate.txt")],
        repository=repository,
    )
    monkeypatch.setattr(
        "core.ingestion.extract_text_by_page",
        lambda path: pytest.fail("Không được trích xuất lại tài liệu trùng"),
    )
    second = ingest_documents(
        [DocumentSource(path=str(test_file), display_name="duplicate.txt")],
        repository=repository,
    )

    assert first.success is True
    assert second.success is True
    assert second.processed_files == 0
    assert second.skipped_files == 1
    assert "đã tồn tại" in second.message
    assert len(repository.list_documents()) == 1


def test_ingest_documents_adds_new_document_to_existing_collection(
    tmp_path, monkeypatch, repository
):
    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"
    first_file.write_text("Tài liệu thứ nhất về động cơ.", encoding="utf-8")
    second_file.write_text("Tài liệu thứ hai về thủy lực.", encoding="utf-8")
    monkeypatch.setattr(
        "core.ingestion.encode_texts",
        lambda chunks, show_progress=False: np.ones(
            (len(chunks), 384), dtype=np.float32
        ),
    )

    first_result = ingest_documents(
        [DocumentSource(path=str(first_file), display_name="first.txt")],
        repository=repository,
    )
    first_chunk_ids = set(vector_store.id_order)
    second_result = ingest_documents(
        [DocumentSource(path=str(second_file), display_name="second.txt")],
        repository=repository,
    )

    assert first_result.success is True
    assert second_result.success is True
    assert len(repository.list_documents()) == 2
    assert first_chunk_ids < set(vector_store.id_order)
    assert vector_store.total_chunks == second_result.chunk_count


def test_delete_indexed_document_rebuilds_indexes_from_remaining_chunks(
    tmp_path, monkeypatch, repository
):
    repository.initialize()
    _store_ready_document(
        repository,
        "doc-delete",
        "xoa.txt",
        "chunk-delete",
        "Nội dung cần xóa.",
    )
    _store_ready_document(
        repository,
        "doc-keep",
        "giu.txt",
        "chunk-keep",
        "Nội dung cần giữ.",
    )
    runtime_vector, runtime_bm25 = _build_runtime_indexes(
        ["Nội dung cần xóa.", "Nội dung cần giữ."],
        ["chunk-delete", "chunk-keep"],
        [
            {"source": "xoa.txt", "doc_id": "doc-delete", "page": 1},
            {"source": "giu.txt", "doc_id": "doc-keep", "page": 1},
        ],
    )
    snapshot_store = IndexSnapshotStore(tmp_path / "indexes")
    monkeypatch.setattr(
        ingestion_module,
        "encode_texts",
        lambda chunks, show_progress=False: np.ones(
            (len(chunks), 384), dtype=np.float32
        ),
    )

    remaining_chunks = ingestion_module.delete_indexed_document(
        "doc-delete",
        repository=repository,
        snapshot_store=snapshot_store,
        target_vector_store=runtime_vector,
        target_bm25_manager=runtime_bm25,
    )

    assert remaining_chunks == 1
    assert repository.get_document("doc-delete") is None
    assert repository.get_document("doc-keep") is not None
    assert runtime_vector.id_order == ["chunk-keep"]
    assert runtime_bm25.doc_mapping == {0: "chunk-keep"}
    assert repository.get_active_snapshot()["chunk_count"] == 1
    assert runtime_vector.snapshot_id == repository.get_active_snapshot()["id"]


def test_delete_last_document_clears_runtime_indexes(tmp_path, repository):
    repository.initialize()
    _store_ready_document(
        repository,
        "doc-only",
        "duy-nhat.txt",
        "chunk-only",
        "Nội dung duy nhất.",
    )
    runtime_vector, runtime_bm25 = _build_runtime_indexes(
        ["Nội dung duy nhất."],
        ["chunk-only"],
        [{"source": "duy-nhat.txt", "doc_id": "doc-only", "page": 1}],
    )
    repository.activate_snapshot(
        snapshot_id="snapshot-old",
        embedding_model="model-a",
        snapshot_path=str(tmp_path / "indexes" / "snapshot-old"),
        chunk_count=1,
    )

    remaining_chunks = ingestion_module.delete_indexed_document(
        "doc-only",
        repository=repository,
        snapshot_store=IndexSnapshotStore(tmp_path / "indexes"),
        target_vector_store=runtime_vector,
        target_bm25_manager=runtime_bm25,
    )

    assert remaining_chunks == 0
    assert repository.get_document("doc-only") is None
    assert repository.get_active_snapshot() is None
    assert runtime_vector.is_ready is False
    assert runtime_bm25.bm25_index is None


def test_delete_failure_preserves_document_and_runtime_indexes(
    tmp_path, monkeypatch, repository
):
    repository.initialize()
    _store_ready_document(
        repository,
        "doc-delete",
        "xoa.txt",
        "chunk-delete",
        "Nội dung cần xóa.",
    )
    _store_ready_document(
        repository,
        "doc-keep",
        "giu.txt",
        "chunk-keep",
        "Nội dung cần giữ.",
    )
    runtime_vector, runtime_bm25 = _build_runtime_indexes(
        ["Nội dung cần xóa.", "Nội dung cần giữ."],
        ["chunk-delete", "chunk-keep"],
        [
            {"source": "xoa.txt", "doc_id": "doc-delete", "page": 1},
            {"source": "giu.txt", "doc_id": "doc-keep", "page": 1},
        ],
    )
    original_ids = list(runtime_vector.id_order)
    original_bm25_mapping = dict(runtime_bm25.doc_mapping)
    snapshot_store = IndexSnapshotStore(tmp_path / "indexes")
    monkeypatch.setattr(
        ingestion_module,
        "encode_texts",
        lambda chunks, show_progress=False: np.ones(
            (len(chunks), 384), dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        repository,
        "delete_document_and_activate_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("database locked")),
    )

    with pytest.raises(OSError, match="database locked"):
        ingestion_module.delete_indexed_document(
            "doc-delete",
            repository=repository,
            snapshot_store=snapshot_store,
            target_vector_store=runtime_vector,
            target_bm25_manager=runtime_bm25,
        )

    assert repository.get_document("doc-delete") is not None
    assert runtime_vector.id_order == original_ids
    assert runtime_bm25.doc_mapping == original_bm25_mapping
    assert list((tmp_path / "indexes").iterdir()) == []


def test_storage_failure_preserves_existing_index(tmp_path, monkeypatch, repository):
    vector_store.build_index(
        ["Old chunk"],
        ["old_1"],
        [{"source": "old"}],
        np.ones((1, 384), dtype=np.float32),
    )
    initial_ids = list(vector_store.id_order)
    test_file = tmp_path / "new.txt"
    test_file.write_text("Nội dung mới hợp lệ.", encoding="utf-8")
    monkeypatch.setattr(
        "core.ingestion.encode_texts",
        lambda chunks, show_progress=False: np.ones(
            (len(chunks), 384), dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        repository,
        "save_ingestion_batch",
        lambda records: (_ for _ in ()).throw(sqlite3.OperationalError("disk full")),
    )

    result = ingest_documents(
        [DocumentSource(path=str(test_file), display_name="new.txt")],
        repository=repository,
    )

    assert result.success is False
    assert "Không thể lưu" in result.message
    assert vector_store.id_order == initial_ids


class DummyUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.content = content
        self.closed = False

    async def read(self, size: int = -1):
        if not self.content:
            return b""
        chunk = self.content[:size]
        self.content = self.content[size:]
        return chunk

    async def close(self):
        self.closed = True


def test_upload_file_path_traversal_sanitization(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_router,
        "ingest_documents",
        lambda docs, progress=None: IngestionResult(
            message="OK",
            filenames=[docs[0].name],
            total_files=1,
            processed_files=1,
            failed_files=0,
            chunk_count=1,
        ),
    )

    file_obj = DummyUploadFile("../../etc/passwd.txt", b"some text content")
    response = asyncio.run(api_router.upload_file(file_obj))

    assert response["status"] == "success"
    assert response["file_info"]["filename"] == "passwd.txt"
    assert file_obj.closed is True


def test_upload_file_unsupported_extension():
    file_obj = DummyUploadFile("malicious.sh", b"echo hi")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_router.upload_file(file_obj))

    assert exc_info.value.status_code == 415
    assert file_obj.closed is False  # Rejected before opening temp file


def test_upload_file_exceeds_max_size(monkeypatch):
    monkeypatch.setattr(api_router, "MAX_UPLOAD_SIZE_MB", 1)
    monkeypatch.setattr(api_router, "MAX_UPLOAD_SIZE_BYTES", 1 * 1024 * 1024)

    large_content = b"X" * (1 * 1024 * 1024 + 10)
    file_obj = DummyUploadFile("big_file.txt", large_content)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_router.upload_file(file_obj))

    assert exc_info.value.status_code == 413
    assert file_obj.closed is True
