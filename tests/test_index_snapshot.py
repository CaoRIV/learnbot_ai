import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from core.bm25_index import BM25IndexManager, bm25_manager
from core.index_snapshot import IndexSnapshotStore, restore_indexes
from core.ingestion import DocumentSource, ingest_documents
from core.storage import SQLiteRepository
from core.vector_store import VectorStore, vector_store


@pytest.fixture(autouse=True)
def clear_runtime_indexes():
    vector_store.clear()
    bm25_manager.clear()
    yield
    vector_store.clear()
    bm25_manager.clear()


def _document_record(document_id="doc-1", content="Bơm thủy lực cần dầu sạch."):
    return {
        "id": document_id,
        "source_name": "huong-dan.txt",
        "content_hash": f"hash-{document_id}",
        "file_size": len(content.encode("utf-8")),
        "chunks": [
            {
                "id": f"{document_id}-chunk-0",
                "chunk_index": 0,
                "content": content,
                "page": 1,
                "metadata": {
                    "source": "huong-dan.txt",
                    "doc_id": document_id,
                    "page": 1,
                },
            }
        ],
    }


def _build_indexes(record):
    chunk = record["chunks"][0]
    candidate_vector = VectorStore()
    candidate_vector.build_index(
        [chunk["content"]],
        [chunk["id"]],
        [chunk["metadata"]],
        np.ones((1, 384), dtype=np.float32),
    )
    candidate_bm25 = BM25IndexManager()
    candidate_bm25.build_index([chunk["content"]], [chunk["id"]])
    return candidate_vector, candidate_bm25


def test_snapshot_round_trip_restores_indexes_without_embedding(tmp_path, monkeypatch):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()
    snapshot_store = IndexSnapshotStore(tmp_path / "indexes")
    record = _document_record()
    candidate_vector, candidate_bm25 = _build_indexes(record)

    snapshot = snapshot_store.write_snapshot(candidate_vector, candidate_bm25)
    repository.save_ingestion_batch(
        [record],
        snapshot=snapshot.as_storage_record(),
    )

    restored_vector = VectorStore()
    restored_bm25 = BM25IndexManager()
    monkeypatch.setattr(
        "core.embeddings.encode_texts",
        lambda *args, **kwargs: pytest.fail("Khôi phục không được tạo embedding"),
    )

    result = restore_indexes(
        repository=repository,
        snapshot_store=snapshot_store,
        target_vector_store=restored_vector,
        target_bm25_manager=restored_bm25,
    )

    assert result.success is True
    assert result.chunk_count == 1
    assert restored_vector.snapshot_id == snapshot.snapshot_id
    docs, ids, metadatas = restored_vector.search(
        np.ones((1, 384), dtype=np.float32),
        k=1,
    )
    assert docs == [record["chunks"][0]["content"]]
    assert ids == [record["chunks"][0]["id"]]
    assert metadatas[0]["page"] == 1
    assert restored_bm25.doc_mapping == {0: ids[0]}
    assert restored_bm25.tokenized_corpus == candidate_bm25.tokenized_corpus


def test_restore_rejects_corrupt_snapshot_and_preserves_runtime_index(tmp_path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()
    snapshot_store = IndexSnapshotStore(tmp_path / "indexes")
    record = _document_record()
    candidate_vector, candidate_bm25 = _build_indexes(record)
    snapshot = snapshot_store.write_snapshot(candidate_vector, candidate_bm25)
    repository.save_ingestion_batch([record], snapshot=snapshot.as_storage_record())

    manifest_path = Path(snapshot.snapshot_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    active_vector = VectorStore()
    active_vector.build_index(
        ["Dữ liệu đang hoạt động"],
        ["active-1"],
        [{"source": "active.txt"}],
        np.ones((1, 384), dtype=np.float32),
    )
    active_bm25 = BM25IndexManager()
    active_bm25.build_index(["Dữ liệu đang hoạt động"], ["active-1"])

    result = restore_indexes(
        repository=repository,
        snapshot_store=snapshot_store,
        target_vector_store=active_vector,
        target_bm25_manager=active_bm25,
    )

    assert result.success is False
    assert "snapshot" in result.message.lower()
    assert active_vector.id_order == ["active-1"]
    assert repository.get_active_snapshot() is None


def test_restore_rejects_snapshot_from_another_embedding_model(tmp_path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()
    writer = IndexSnapshotStore(tmp_path / "indexes", embedding_model="model-a")
    record = _document_record()
    candidate_vector, candidate_bm25 = _build_indexes(record)
    snapshot = writer.write_snapshot(candidate_vector, candidate_bm25)
    repository.save_ingestion_batch([record], snapshot=snapshot.as_storage_record())

    result = restore_indexes(
        repository=repository,
        snapshot_store=IndexSnapshotStore(
            tmp_path / "indexes",
            embedding_model="model-b",
        ),
        target_vector_store=VectorStore(),
        target_bm25_manager=BM25IndexManager(),
    )

    assert result.success is False
    assert "mô hình embedding" in result.message.lower()
    assert repository.get_active_snapshot() is None


def test_second_ingestion_only_embeds_new_chunks(tmp_path, monkeypatch):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    snapshot_store = IndexSnapshotStore(tmp_path / "indexes")
    first_file = tmp_path / "mot.txt"
    second_file = tmp_path / "hai.txt"
    first_file.write_text("Tài liệu thứ nhất nói về động cơ.", encoding="utf-8")
    second_file.write_text("Tài liệu thứ hai nói về thủy lực.", encoding="utf-8")
    encoded_batches = []

    def fake_encode(texts, show_progress=False):
        encoded_batches.append(list(texts))
        return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr("core.ingestion.encode_texts", fake_encode)

    first = ingest_documents(
        [DocumentSource(str(first_file), "mot.txt")],
        repository=repository,
        snapshot_store=snapshot_store,
    )
    second = ingest_documents(
        [DocumentSource(str(second_file), "hai.txt")],
        repository=repository,
        snapshot_store=snapshot_store,
    )

    assert first.success is True
    assert second.success is True
    assert len(encoded_batches) == 2
    assert any("động cơ" in text for text in encoded_batches[0])
    assert all("động cơ" not in text for text in encoded_batches[1])
    assert any("thủy lực" in text for text in encoded_batches[1])
    assert vector_store.total_chunks == 2
    assert repository.get_active_snapshot()["chunk_count"] == 2


def test_document_and_snapshot_activation_share_one_transaction(tmp_path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_ingestion_batch(
            [_document_record()],
            snapshot={
                "id": "snapshot-invalid",
                "schema_version": 1,
                "embedding_model": "model-a",
                "snapshot_path": str(tmp_path / "indexes" / "snapshot-invalid"),
                "chunk_count": -1,
            },
        )

    assert repository.list_documents() == []
    assert repository.get_active_snapshot() is None


def test_failed_activation_keeps_previous_snapshot_active(tmp_path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()
    repository.activate_snapshot(
        snapshot_id="snapshot-old",
        embedding_model="model-a",
        snapshot_path=str(tmp_path / "indexes" / "snapshot-old"),
        chunk_count=1,
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_ingestion_batch(
            [_document_record()],
            snapshot={
                "id": "snapshot-invalid",
                "schema_version": 1,
                "embedding_model": "model-a",
                "snapshot_path": str(tmp_path / "indexes" / "snapshot-invalid"),
                "chunk_count": -1,
            },
        )

    assert repository.list_documents() == []
    assert repository.get_active_snapshot()["id"] == "snapshot-old"
