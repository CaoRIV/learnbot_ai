import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.backup import BackupError, BackupManager
from core.bm25_index import BM25IndexManager
from core.index_snapshot import (
    BM25_FILENAME,
    FAISS_FILENAME,
    MANIFEST_FILENAME,
    IndexSnapshotStore,
)
from core.storage import SQLiteRepository
from core.vector_store import VectorStore


def _build_runtime_indexes(content="Nội dung cần sao lưu.", chunk_id="chunk-1"):
    vector = VectorStore()
    vector.build_index(
        [content],
        [chunk_id],
        [{"source": "tai-lieu.txt", "doc_id": "doc-1", "page": 1}],
        np.ones((1, 384), dtype=np.float32),
    )
    bm25 = BM25IndexManager()
    bm25.build_index([content], [chunk_id])
    return vector, bm25


def _store_document(repository, content="Nội dung cần sao lưu."):
    repository.upsert_document(
        document_id="doc-1",
        source_name="tai-lieu.txt",
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        file_size=len(content.encode("utf-8")),
        status="ready",
    )
    repository.replace_chunks(
        "doc-1",
        [
            {
                "id": "chunk-1",
                "chunk_index": 0,
                "content": content,
                "page": 1,
                "metadata": {
                    "source": "tai-lieu.txt",
                    "doc_id": "doc-1",
                    "page": 1,
                },
            }
        ],
    )


def _build_manager(tmp_path, *, with_document=True, activate_snapshot=True):
    data_directory = tmp_path / "live"
    repository = SQLiteRepository(data_directory / "learnbot.db")
    repository.initialize()
    vector, bm25 = _build_runtime_indexes()
    snapshot_store = IndexSnapshotStore(data_directory / "indexes")
    if with_document:
        _store_document(repository)
    if with_document and activate_snapshot:
        snapshot = snapshot_store.write_snapshot(vector, bm25)
        repository.activate_snapshot(
            snapshot_id=snapshot.snapshot_id,
            embedding_model=snapshot.embedding_model,
            snapshot_path=snapshot.snapshot_path,
            chunk_count=snapshot.chunk_count,
            schema_version=snapshot.schema_version,
        )
        vector.snapshot_id = snapshot.snapshot_id
        bm25.snapshot_id = snapshot.snapshot_id
    manager = BackupManager(
        repository=repository,
        snapshot_store=snapshot_store,
        backup_directory=tmp_path / "backups",
        target_vector_store=vector,
        target_bm25_manager=bm25,
    )
    return manager, repository, vector, bm25


def _activate_content(manager, repository, vector, bm25, content):
    repository.replace_chunks(
        "doc-1",
        [
            {
                "id": "chunk-1",
                "chunk_index": 0,
                "content": content,
                "page": 1,
                "metadata": {
                    "source": "tai-lieu.txt",
                    "doc_id": "doc-1",
                    "page": 1,
                },
            }
        ],
    )
    candidate_vector, candidate_bm25 = _build_runtime_indexes(content)
    snapshot = manager.snapshot_store.write_snapshot(
        candidate_vector,
        candidate_bm25,
    )
    repository.activate_snapshot(
        snapshot_id=snapshot.snapshot_id,
        embedding_model=snapshot.embedding_model,
        snapshot_path=snapshot.snapshot_path,
        chunk_count=snapshot.chunk_count,
        schema_version=snapshot.schema_version,
    )
    candidate_vector.snapshot_id = snapshot.snapshot_id
    candidate_bm25.snapshot_id = snapshot.snapshot_id
    vector.replace_with(candidate_vector)
    bm25.replace_with(candidate_bm25)


def test_create_backup_copies_database_snapshot_and_manifest(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)

    info = manager.create_backup()

    backup_path = tmp_path / "backups" / info.backup_id
    manifest = json.loads(
        (backup_path / "manifest.json").read_text(encoding="utf-8")
    )
    backup_repository = SQLiteRepository(backup_path / "learnbot.db")
    backup_snapshot = backup_repository.get_active_snapshot()

    assert info.kind == "manual"
    assert info.document_count == 1
    assert info.chunk_count == 1
    assert info.snapshot_id == backup_snapshot["id"]
    assert [item["source_name"] for item in backup_repository.list_documents()] == [
        "tai-lieu.txt"
    ]
    assert len(manifest["database"]["sha256"]) == 64
    snapshot_path = backup_path / "indexes" / info.snapshot_id
    assert Path(backup_snapshot["snapshot_path"]) == snapshot_path.resolve()
    assert {path.name for path in snapshot_path.iterdir()} == {
        MANIFEST_FILENAME,
        FAISS_FILENAME,
        BM25_FILENAME,
    }
    assert manager.list_backups() == [info]


def test_create_backup_rejects_chunks_without_active_snapshot(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path, activate_snapshot=False)

    with pytest.raises(BackupError, match="xây lại chỉ mục"):
        manager.create_backup()

    backup_root = tmp_path / "backups"
    assert not backup_root.exists() or list(backup_root.iterdir()) == []


def test_create_backup_rejects_path_like_live_snapshot_id_before_copy(tmp_path):
    manager, repository, _, _ = _build_manager(tmp_path)
    with repository.connection() as connection:
        connection.execute(
            "UPDATE index_snapshots SET id = ? WHERE status = 'active'",
            ("../../escaped",),
        )

    with pytest.raises(BackupError, match="Mã snapshot"):
        manager.create_backup()

    backup_root = tmp_path / "backups"
    assert not backup_root.exists() or list(backup_root.iterdir()) == []


def test_list_backups_ignores_backup_with_changed_checksum(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)
    info = manager.create_backup()
    bm25_path = (
        tmp_path
        / "backups"
        / info.backup_id
        / "indexes"
        / info.snapshot_id
        / BM25_FILENAME
    )
    bm25_path.write_text("dữ liệu đã bị sửa", encoding="utf-8")

    assert manager.list_backups() == []


def test_list_backups_ignores_manifest_with_non_object_database_record(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)
    info = manager.create_backup()
    manifest_path = tmp_path / "backups" / info.backup_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["database"] = ["không hợp lệ"]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert manager.list_backups() == []


def test_list_backups_ignores_manifest_with_non_object_checksum_record(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)
    info = manager.create_backup()
    manifest_path = tmp_path / "backups" / info.backup_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot"]["files"][BM25_FILENAME] = []
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert manager.list_backups() == []


def test_list_backups_ignores_path_like_snapshot_id(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)
    info = manager.create_backup()
    backup_path = tmp_path / "backups" / info.backup_id
    (backup_path / "indexes" / "nested").mkdir()
    manifest_path = backup_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path_like_id = f"nested/../{info.snapshot_id}"
    manifest["snapshot_id"] = path_like_id
    manifest["snapshot"]["snapshot_id"] = path_like_id
    manifest["snapshot"]["directory"] = f"indexes/{path_like_id}"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert manager.list_backups() == []


def test_create_backup_keeps_only_ten_newest_valid_backups(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)
    created = [manager.create_backup() for _ in range(11)]

    backups = manager.list_backups()

    assert len(backups) == 10
    assert backups[0].backup_id == created[-1].backup_id
    assert created[0].backup_id not in {backup.backup_id for backup in backups}
    assert not (tmp_path / "backups" / created[0].backup_id).exists()


def test_restore_round_trip_creates_safety_backup(tmp_path):
    manager, repository, vector, bm25 = _build_manager(tmp_path)
    source = manager.create_backup()
    _activate_content(
        manager,
        repository,
        vector,
        bm25,
        "Nội dung sau thời điểm backup.",
    )

    result = manager.restore_backup(source.backup_id)

    assert result.backup_id == source.backup_id
    assert result.safety_backup_id != source.backup_id
    assert result.document_count == 1
    assert result.chunk_count == 1
    assert repository.get_chunks("doc-1")[0]["content"] == "Nội dung cần sao lưu."
    assert vector.contents_map == {"chunk-1": "Nội dung cần sao lưu."}
    assert vector.snapshot_id == repository.get_active_snapshot()["id"]
    assert bm25.raw_corpus == ["Nội dung cần sao lưu."]
    assert bm25.snapshot_id == vector.snapshot_id

    backups = {backup.backup_id: backup for backup in manager.list_backups()}
    assert backups[result.safety_backup_id].kind == "pre_restore"
    safety_repository = SQLiteRepository(
        tmp_path / "backups" / result.safety_backup_id / "learnbot.db"
    )
    assert safety_repository.get_chunks("doc-1")[0]["content"] == (
        "Nội dung sau thời điểm backup."
    )


def test_restore_empty_backup_clears_database_and_runtime(tmp_path):
    manager, repository, vector, bm25 = _build_manager(
        tmp_path,
        with_document=False,
        activate_snapshot=False,
    )
    source = manager.create_backup()
    _store_document(repository, "Nội dung được thêm sau backup rỗng.")
    _activate_content(
        manager,
        repository,
        vector,
        bm25,
        "Nội dung được thêm sau backup rỗng.",
    )

    result = manager.restore_backup(source.backup_id)

    assert result.snapshot_id is None
    assert result.document_count == 0
    assert result.chunk_count == 0
    assert repository.list_documents() == []
    assert repository.get_active_snapshot() is None
    assert vector.total_chunks == 0
    assert vector.snapshot_id is None
    assert bm25.bm25_index is None
    assert bm25.snapshot_id is None


def test_restore_rejects_path_traversal_before_creating_safety_backup(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)

    with pytest.raises(BackupError, match="Mã backup không hợp lệ"):
        manager.restore_backup("../outside")

    assert manager.list_backups() == []


def test_restore_rejects_changed_checksum_before_creating_safety_backup(tmp_path):
    manager, _, _, _ = _build_manager(tmp_path)
    source = manager.create_backup()
    backup_root = tmp_path / "backups"
    initial_directories = {path.name for path in backup_root.iterdir()}
    bm25_path = (
        backup_root
        / source.backup_id
        / "indexes"
        / source.snapshot_id
        / BM25_FILENAME
    )
    bm25_path.write_text("dữ liệu đã bị sửa", encoding="utf-8")

    with pytest.raises(BackupError, match="Checksum|Kích thước"):
        manager.restore_backup(source.backup_id)

    assert {path.name for path in backup_root.iterdir()} == initial_directories


def test_restore_failure_rolls_back_from_safety_backup(tmp_path, monkeypatch):
    manager, repository, vector, bm25 = _build_manager(tmp_path)
    source = manager.create_backup()
    current_content = "Nội dung cần giữ khi restore thất bại."
    _activate_content(
        manager,
        repository,
        vector,
        bm25,
        current_content,
    )
    original_apply = manager._apply_validated_backup
    apply_count = 0

    def fail_after_first_apply(validated):
        nonlocal apply_count
        apply_count += 1
        snapshot_id = original_apply(validated)
        if apply_count == 1:
            raise OSError("restore failed after database replacement")
        return snapshot_id

    monkeypatch.setattr(manager, "_apply_validated_backup", fail_after_first_apply)

    with pytest.raises(BackupError, match="trạng thái trước đó đã được khôi phục"):
        manager.restore_backup(source.backup_id)

    assert repository.get_chunks("doc-1")[0]["content"] == current_content
    assert vector.contents_map == {"chunk-1": current_content}
    assert bm25.raw_corpus == [current_content]
    assert any(backup.kind == "pre_restore" for backup in manager.list_backups())


def test_restore_reports_safety_backup_when_rollback_fails(tmp_path, monkeypatch):
    manager, repository, vector, bm25 = _build_manager(tmp_path)
    source = manager.create_backup()
    _activate_content(
        manager,
        repository,
        vector,
        bm25,
        "Nội dung trước restore lỗi.",
    )
    original_apply = manager._apply_validated_backup
    apply_count = 0

    def fail_restore_and_rollback(validated):
        nonlocal apply_count
        apply_count += 1
        if apply_count == 1:
            original_apply(validated)
            raise OSError("restore failed")
        raise OSError("rollback failed")

    monkeypatch.setattr(
        manager,
        "_apply_validated_backup",
        fail_restore_and_rollback,
    )

    with pytest.raises(BackupError, match="không thể tự rollback") as exc_info:
        manager.restore_backup(source.backup_id)

    safety_backups = [
        backup for backup in manager.list_backups() if backup.kind == "pre_restore"
    ]
    assert len(safety_backups) == 1
    assert safety_backups[0].backup_id in str(exc_info.value)
    assert (tmp_path / "backups" / safety_backups[0].backup_id).is_dir()
