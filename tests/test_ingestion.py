import asyncio
from pathlib import Path
import pytest
from fastapi import HTTPException

import api_router
from core.ingestion import (
    DocumentSource,
    IngestionResult,
    ingest_documents,
)
from core.vector_store import vector_store
from core.bm25_index import bm25_manager


import numpy as np

def test_ingest_documents_success(tmp_path, monkeypatch):
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("Đây là tài liệu thử nghiệm cho pipeline RAG.", encoding="utf-8")

    # Mock encode_texts to return dummy embeddings
    monkeypatch.setattr(
        "core.ingestion.encode_texts",
        lambda chunks, show_progress=False: np.ones((len(chunks), 384), dtype=np.float32),
    )

    result = ingest_documents([DocumentSource(path=str(test_file), display_name="test_doc.txt")])

    assert isinstance(result, IngestionResult)
    assert result.success is True
    assert result.total_files == 1
    assert result.processed_files == 1
    assert result.failed_files == 0
    assert result.chunk_count > 0
    assert result.filenames == ["test_doc.txt"]
    assert vector_store.total_chunks > 0


def test_ingest_documents_unsupported_extension(tmp_path):
    test_file = tmp_path / "script.exe"
    test_file.write_bytes(b"binary data")

    result = ingest_documents([DocumentSource(path=str(test_file), display_name="script.exe")])

    assert result.success is False
    assert result.processed_files == 0
    assert result.failed_files == 1
    assert "chưa được hỗ trợ" in result.message


def test_ingest_documents_atomic_failure_preserves_existing_store(tmp_path, monkeypatch):
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

    result = ingest_documents([DocumentSource(path=str(test_file), display_name="bad_doc.pdf")])

    assert result.success is False
    # Ensure initial vector store data is untouched
    assert vector_store.total_chunks == initial_chunks


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
