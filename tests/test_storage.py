import sqlite3
from pathlib import Path

import pytest

from core.storage import SQLiteRepository


def test_initialize_creates_versioned_schema_and_sqlite_pragmas(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")

    repository.initialize()
    repository.initialize()

    with repository.connection() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        versions = [
            row["version"]
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]

    assert {"schema_migrations", "documents", "chunks", "index_snapshots"} <= tables
    assert versions == [1]
    assert foreign_keys == 1
    assert journal_mode == "wal"
    assert busy_timeout == 5_000


def test_document_upsert_deduplicates_by_hash_and_manages_chunks(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()

    first = repository.upsert_document(
        document_id="doc-1",
        source_name="tai-lieu.txt",
        content_hash="same-hash",
        file_size=120,
        status="ready",
    )
    duplicate = repository.upsert_document(
        document_id="doc-2",
        source_name="tai-lieu-doi-ten.txt",
        content_hash="same-hash",
        file_size=120,
        status="ready",
    )

    assert first["id"] == "doc-1"
    assert duplicate["id"] == "doc-1"
    assert duplicate["source_name"] == "tai-lieu-doi-ten.txt"
    assert len(repository.list_documents()) == 1

    repository.replace_chunks(
        "doc-1",
        [
            {
                "id": "chunk-1",
                "chunk_index": 0,
                "content": "Nội dung tiếng Việt.",
                "page": 1,
                "metadata": {"section": "Giới thiệu"},
            },
            {
                "id": "chunk-2",
                "chunk_index": 1,
                "content": "Phân đoạn thứ hai.",
                "page": 2,
                "metadata": {},
            },
        ],
    )

    chunks = repository.get_chunks("doc-1")
    assert [chunk["id"] for chunk in chunks] == ["chunk-1", "chunk-2"]
    assert chunks[0]["metadata"] == {"section": "Giới thiệu"}

    repository.delete_document("doc-1")
    assert repository.get_chunks("doc-1") == []


def test_chunks_require_an_existing_document(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()

    with pytest.raises(sqlite3.IntegrityError):
        repository.replace_chunks(
            "missing-document",
            [
                {
                    "id": "chunk-1",
                    "chunk_index": 0,
                    "content": "Không có tài liệu cha.",
                    "page": None,
                    "metadata": {},
                }
            ],
        )


def test_replace_chunks_rolls_back_when_batch_insert_fails(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()
    repository.upsert_document(
        document_id="doc-1",
        source_name="tai-lieu.txt",
        content_hash="hash-1",
        file_size=120,
        status="ready",
    )
    repository.replace_chunks(
        "doc-1",
        [
            {
                "id": "old-chunk",
                "chunk_index": 0,
                "content": "Nội dung đang hoạt động.",
                "page": 1,
                "metadata": {},
            }
        ],
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.replace_chunks(
            "doc-1",
            [
                {
                    "id": "duplicate-chunk",
                    "chunk_index": 0,
                    "content": "Nội dung mới thứ nhất.",
                    "page": 1,
                    "metadata": {},
                },
                {
                    "id": "duplicate-chunk",
                    "chunk_index": 1,
                    "content": "Nội dung mới thứ hai.",
                    "page": 2,
                    "metadata": {},
                },
            ],
        )

    chunks = repository.get_chunks("doc-1")
    assert [chunk["id"] for chunk in chunks] == ["old-chunk"]
    assert chunks[0]["content"] == "Nội dung đang hoạt động."


def test_activate_snapshot_keeps_only_one_active_snapshot(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()

    repository.activate_snapshot(
        snapshot_id="snapshot-1",
        embedding_model="model-a",
        snapshot_path="data/indexes/snapshot-1",
        chunk_count=3,
    )
    repository.activate_snapshot(
        snapshot_id="snapshot-2",
        embedding_model="model-a",
        snapshot_path="data/indexes/snapshot-2",
        chunk_count=5,
    )

    active = repository.get_active_snapshot()
    assert active is not None
    assert active["id"] == "snapshot-2"
    assert active["chunk_count"] == 5

    with repository.connection() as connection:
        statuses = {
            row["id"]: row["status"]
            for row in connection.execute(
                "SELECT id, status FROM index_snapshots ORDER BY id"
            )
        }

    assert statuses == {"snapshot-1": "inactive", "snapshot-2": "active"}


def test_save_ingestion_batch_persists_documents_and_chunks(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()

    saved = repository.save_ingestion_batch(
        [
            {
                "id": "doc-1",
                "source_name": "tai-lieu.txt",
                "content_hash": "hash-1",
                "file_size": 120,
                "chunks": [
                    {
                        "id": "chunk-1",
                        "chunk_index": 0,
                        "content": "Nội dung đã lưu.",
                        "page": 1,
                        "metadata": {"source": "tai-lieu.txt"},
                    }
                ],
            }
        ]
    )

    assert [document["id"] for document in saved] == ["doc-1"]
    assert repository.get_document("doc-1")["status"] == "ready"
    assert repository.get_chunks("doc-1")[0]["content"] == "Nội dung đã lưu."


def test_save_ingestion_batch_rolls_back_all_documents_on_failure(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "learnbot.db")
    repository.initialize()

    with pytest.raises(sqlite3.IntegrityError):
        repository.save_ingestion_batch(
            [
                {
                    "id": "doc-1",
                    "source_name": "mot.txt",
                    "content_hash": "hash-1",
                    "file_size": 10,
                    "chunks": [
                        {
                            "id": "duplicate-chunk",
                            "chunk_index": 0,
                            "content": "Nội dung một.",
                            "page": None,
                            "metadata": {},
                        }
                    ],
                },
                {
                    "id": "doc-2",
                    "source_name": "hai.txt",
                    "content_hash": "hash-2",
                    "file_size": 20,
                    "chunks": [
                        {
                            "id": "duplicate-chunk",
                            "chunk_index": 0,
                            "content": "Nội dung hai.",
                            "page": None,
                            "metadata": {},
                        }
                    ],
                },
            ]
        )

    assert repository.list_documents() == []
