"""Backup và phục hồi cục bộ cho SQLite cùng snapshot FAISS/BM25."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from config import BACKUP_DIRECTORY
from core.bm25_index import BM25IndexManager, bm25_manager
from core.index_snapshot import (
    BM25_FILENAME,
    FAISS_FILENAME,
    MANIFEST_FILENAME,
    IndexSnapshotStore,
    SnapshotError,
)
from core.storage import SCHEMA_VERSION, SQLiteRepository
from core.vector_store import VectorStore, index_lock, vector_store


BACKUP_FORMAT_VERSION = 1
BACKUP_DATABASE_FILENAME = "learnbot.db"
BACKUP_MANIFEST_FILENAME = "manifest.json"
DEFAULT_MAX_BACKUPS = 10
BACKUP_ID_PATTERN = re.compile(
    r"backup_\d{8}T\d{12}Z_[0-9a-f]{8}"
)
SNAPSHOT_ID_PATTERN = re.compile(r"snapshot_[0-9a-f]{32}")
SNAPSHOT_FILENAMES = (MANIFEST_FILENAME, FAISS_FILENAME, BM25_FILENAME)


class BackupError(RuntimeError):
    """Backup không hợp lệ hoặc không thể được xử lý an toàn."""


class BackupNotFoundError(BackupError):
    """Không tìm thấy backup cục bộ theo ID."""


@dataclass(frozen=True)
class BackupInfo:
    backup_id: str
    created_at: str
    kind: Literal["manual", "pre_restore"]
    document_count: int
    chunk_count: int
    snapshot_id: str | None
    size_bytes: int


@dataclass(frozen=True)
class BackupRestoreResult:
    backup_id: str
    safety_backup_id: str
    document_count: int
    chunk_count: int
    snapshot_id: str | None


@dataclass(frozen=True)
class _ValidatedBackup:
    path: Path
    info: BackupInfo
    repository: SQLiteRepository
    vector_store: VectorStore
    bm25_manager: BM25IndexManager


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    return {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _copy_sqlite_database(source_path: Path, destination_path: Path) -> None:
    source = sqlite3.connect(str(source_path), timeout=5.0)
    destination = sqlite3.connect(str(destination_path), timeout=5.0)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


class BackupManager:
    """Quản lý tối đa một nhóm nhỏ backup cục bộ đã kiểm tra checksum."""

    def __init__(
        self,
        *,
        repository: SQLiteRepository | None = None,
        snapshot_store: IndexSnapshotStore | None = None,
        backup_directory: str | Path = BACKUP_DIRECTORY,
        target_vector_store: VectorStore = vector_store,
        target_bm25_manager: BM25IndexManager = bm25_manager,
        max_backups: int = DEFAULT_MAX_BACKUPS,
    ):
        self.repository = repository or SQLiteRepository()
        self.snapshot_store = snapshot_store or IndexSnapshotStore()
        self.backup_directory = Path(backup_directory).resolve()
        self.target_vector_store = target_vector_store
        self.target_bm25_manager = target_bm25_manager
        self.max_backups = max_backups

    @staticmethod
    def _new_backup_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"backup_{timestamp}_{uuid4().hex[:8]}"

    def _backup_path(self, backup_id: str, *, must_exist: bool = False) -> Path:
        if BACKUP_ID_PATTERN.fullmatch(backup_id) is None:
            raise BackupError("Mã backup không hợp lệ.")
        path = (self.backup_directory / backup_id).resolve()
        if path.parent != self.backup_directory:
            raise BackupError("Đường dẫn backup nằm ngoài thư mục được cấu hình.")
        if path.is_symlink():
            raise BackupError("Không chấp nhận backup qua liên kết tượng trưng.")
        if must_exist and not path.is_dir():
            raise BackupNotFoundError("Không tìm thấy bản sao lưu.")
        return path

    @staticmethod
    def _require_regular_file(path: Path, root: Path) -> None:
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise BackupError("Tệp backup nằm ngoài thư mục cho phép.") from exc
        if path.is_symlink() or not path.is_file():
            raise BackupError("Backup thiếu tệp bắt buộc hoặc chứa symlink.")

    def _live_snapshot_path(self, snapshot_path: str) -> Path:
        path = Path(snapshot_path).resolve()
        try:
            path.relative_to(self.snapshot_store.index_directory)
        except ValueError as exc:
            raise BackupError(
                "Active snapshot nằm ngoài thư mục chỉ mục được cấu hình."
            ) from exc
        if path.is_symlink() or not path.is_dir():
            raise BackupError("Không tìm thấy thư mục active snapshot hợp lệ.")
        return path

    @staticmethod
    def _rewrite_active_snapshot_path(
        database_path: Path,
        snapshot_id: str,
        snapshot_path: Path,
    ) -> None:
        connection = sqlite3.connect(str(database_path), timeout=5.0)
        try:
            connection.execute(
                "UPDATE index_snapshots SET snapshot_path = ? WHERE id = ?",
                (str(snapshot_path.resolve()), snapshot_id),
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()

    def _build_manifest(
        self,
        backup_path: Path,
        backup_id: str,
        kind: Literal["manual", "pre_restore"],
        document_count: int,
        chunk_count: int,
        snapshot_id: str | None,
    ) -> dict:
        database_path = backup_path / BACKUP_DATABASE_FILENAME
        snapshot_record = None
        if snapshot_id is not None:
            snapshot_directory = backup_path / "indexes" / snapshot_id
            snapshot_record = {
                "snapshot_id": snapshot_id,
                "directory": f"indexes/{snapshot_id}",
                "files": {
                    filename: _file_record(snapshot_directory / filename)
                    for filename in SNAPSHOT_FILENAMES
                },
            }
        return {
            "format_version": BACKUP_FORMAT_VERSION,
            "backup_id": backup_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "kind": kind,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "snapshot_id": snapshot_id,
            "database": {
                "filename": BACKUP_DATABASE_FILENAME,
                **_file_record(database_path),
            },
            "snapshot": snapshot_record,
        }

    def _load_manifest(self, backup_path: Path) -> tuple[dict, BackupInfo]:
        manifest_path = backup_path / BACKUP_MANIFEST_FILENAME
        self._require_regular_file(manifest_path, backup_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BackupError("Manifest backup bị hỏng hoặc không đọc được.") from exc

        backup_id = backup_path.name
        if manifest.get("format_version") != BACKUP_FORMAT_VERSION:
            raise BackupError("Phiên bản định dạng backup không tương thích.")
        if manifest.get("backup_id") != backup_id:
            raise BackupError("Mã backup trong manifest không khớp thư mục.")
        kind = manifest.get("kind")
        if kind not in {"manual", "pre_restore"}:
            raise BackupError("Loại backup không hợp lệ.")
        created_at = manifest.get("created_at")
        try:
            datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise BackupError("Thời gian tạo backup không hợp lệ.") from exc

        document_count = manifest.get("document_count")
        chunk_count = manifest.get("chunk_count")
        if (
            not isinstance(document_count, int)
            or isinstance(document_count, bool)
            or document_count < 0
            or not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count < 0
        ):
            raise BackupError("Số lượng tài liệu hoặc phân đoạn không hợp lệ.")

        database = manifest.get("database")
        if not isinstance(database, dict):
            raise BackupError("Thông tin tệp SQLite trong manifest không hợp lệ.")
        if database.get("filename") != BACKUP_DATABASE_FILENAME:
            raise BackupError("Tên tệp SQLite trong manifest không hợp lệ.")
        database_path = backup_path / BACKUP_DATABASE_FILENAME
        self._require_regular_file(database_path, backup_path)
        self._validate_file_record(database_path, database)
        size_bytes = database_path.stat().st_size

        snapshot_id = manifest.get("snapshot_id")
        snapshot = manifest.get("snapshot")
        if snapshot_id is None:
            if chunk_count != 0 or snapshot is not None:
                raise BackupError("Backup rỗng có thông tin snapshot không hợp lệ.")
        else:
            if (
                not isinstance(snapshot_id, str)
                or SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None
            ):
                raise BackupError("Mã snapshot backup không hợp lệ.")
            if not isinstance(snapshot, dict):
                raise BackupError("Backup thiếu thông tin active snapshot.")
            if snapshot.get("snapshot_id") != snapshot_id:
                raise BackupError("Mã snapshot trong manifest không đồng nhất.")
            expected_directory = f"indexes/{snapshot_id}"
            if snapshot.get("directory") != expected_directory:
                raise BackupError("Đường dẫn snapshot trong manifest không hợp lệ.")
            snapshot_directory = backup_path / "indexes" / snapshot_id
            if snapshot_directory.is_symlink() or not snapshot_directory.is_dir():
                raise BackupError("Không tìm thấy thư mục snapshot trong backup.")
            files = snapshot.get("files")
            if not isinstance(files, dict) or set(files) != set(SNAPSHOT_FILENAMES):
                raise BackupError("Danh sách tệp snapshot trong manifest không hợp lệ.")
            if {path.name for path in snapshot_directory.iterdir()} != set(
                SNAPSHOT_FILENAMES
            ):
                raise BackupError("Thư mục snapshot chứa tệp không hợp lệ.")
            for filename in SNAPSHOT_FILENAMES:
                path = snapshot_directory / filename
                self._require_regular_file(path, backup_path)
                self._validate_file_record(path, files[filename])
                size_bytes += path.stat().st_size

        info = BackupInfo(
            backup_id=backup_id,
            created_at=created_at,
            kind=kind,
            document_count=document_count,
            chunk_count=chunk_count,
            snapshot_id=snapshot_id,
            size_bytes=size_bytes,
        )
        return manifest, info

    @staticmethod
    def _validate_file_record(path: Path, record: dict) -> None:
        if not isinstance(record, dict):
            raise BackupError("Thông tin checksum tệp backup không hợp lệ.")
        if record.get("size_bytes") != path.stat().st_size:
            raise BackupError("Kích thước tệp backup không khớp manifest.")
        if record.get("sha256") != _sha256_file(path):
            raise BackupError("Checksum tệp backup không hợp lệ.")

    def create_backup(
        self,
        kind: Literal["manual", "pre_restore"] = "manual",
        *,
        prune: bool = True,
    ) -> BackupInfo:
        if kind not in {"manual", "pre_restore"}:
            raise BackupError("Loại backup không hợp lệ.")
        with index_lock:
            self.repository.initialize()
            backup_id = self._new_backup_id()
            self.backup_directory.mkdir(parents=True, exist_ok=True)
            staging_path = self.backup_directory / f".{backup_id}.tmp"
            final_path = self._backup_path(backup_id)
            try:
                staging_path.mkdir()
                database_path = staging_path / BACKUP_DATABASE_FILENAME
                _copy_sqlite_database(self.repository.database_path, database_path)
                backup_repository = SQLiteRepository(
                    database_path,
                    migrations_dir=self.repository.migrations_dir,
                )
                documents = backup_repository.list_document_summaries()
                chunks = backup_repository.list_ready_chunks()
                active_snapshot = backup_repository.get_active_snapshot()

                if chunks and active_snapshot is None:
                    raise BackupError(
                        "Kho dữ liệu chưa có active snapshot; hãy xây lại chỉ mục trước."
                    )
                if active_snapshot is not None and (
                    active_snapshot["chunk_count"] != len(chunks) or not chunks
                ):
                    raise BackupError(
                        "SQLite và active snapshot không đồng nhất; hãy xây lại chỉ mục trước."
                    )

                snapshot_id = None
                if active_snapshot is not None:
                    snapshot_id = active_snapshot["id"]
                    if (
                        not isinstance(snapshot_id, str)
                        or SNAPSHOT_ID_PATTERN.fullmatch(snapshot_id) is None
                    ):
                        raise BackupError("Mã snapshot đang hoạt động không hợp lệ.")
                    source_snapshot = self._live_snapshot_path(
                        active_snapshot["snapshot_path"]
                    )
                    if {path.name for path in source_snapshot.iterdir()} != set(
                        SNAPSHOT_FILENAMES
                    ):
                        raise BackupError("Active snapshot chứa tệp không hợp lệ.")
                    destination_snapshot = staging_path / "indexes" / snapshot_id
                    destination_snapshot.parent.mkdir()
                    shutil.copytree(
                        source_snapshot,
                        destination_snapshot,
                        symlinks=True,
                    )
                    for filename in SNAPSHOT_FILENAMES:
                        self._require_regular_file(
                            destination_snapshot / filename,
                            staging_path,
                        )
                    self._rewrite_active_snapshot_path(
                        database_path,
                        snapshot_id,
                        final_path / "indexes" / snapshot_id,
                    )

                manifest = self._build_manifest(
                    staging_path,
                    backup_id,
                    kind,
                    len(documents),
                    len(chunks),
                    snapshot_id,
                )
                (staging_path / BACKUP_MANIFEST_FILENAME).write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(staging_path, final_path)
                _, info = self._load_manifest(final_path)
            except Exception:
                shutil.rmtree(staging_path, ignore_errors=True)
                raise

            if prune:
                self._prune_backups()
            return info

    def list_backups(self) -> list[BackupInfo]:
        if not self.backup_directory.is_dir():
            return []
        backups = []
        for path in self.backup_directory.iterdir():
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                validated_path = self._backup_path(path.name, must_exist=True)
                _, info = self._load_manifest(validated_path)
            except BackupError:
                continue
            backups.append(info)
        return sorted(
            backups,
            key=lambda item: (item.created_at, item.backup_id),
            reverse=True,
        )

    def _prune_backups(self, protected_ids: set[str] | None = None) -> None:
        protected = protected_ids or set()
        removable = [
            backup
            for backup in reversed(self.list_backups())
            if backup.backup_id not in protected
        ]
        excess = max(0, len(self.list_backups()) - self.max_backups)
        for backup in removable[:excess]:
            path = self._backup_path(backup.backup_id, must_exist=True)
            shutil.rmtree(path)

    def _validate_backup(self, backup_id: str) -> _ValidatedBackup:
        backup_path = self._backup_path(backup_id, must_exist=True)
        _, info = self._load_manifest(backup_path)
        backup_repository = SQLiteRepository(
            backup_path / BACKUP_DATABASE_FILENAME,
            migrations_dir=self.repository.migrations_dir,
        )
        with backup_repository.connection() as connection:
            versions = [
                row["version"]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
            ]
        if not versions or versions[-1] != SCHEMA_VERSION:
            raise BackupError("Phiên bản schema SQLite của backup không tương thích.")

        documents = backup_repository.list_document_summaries()
        chunks = backup_repository.list_ready_chunks()
        active_snapshot = backup_repository.get_active_snapshot()
        if len(documents) != info.document_count or len(chunks) != info.chunk_count:
            raise BackupError("Số lượng dữ liệu trong SQLite không khớp manifest.")

        if info.snapshot_id is None:
            if active_snapshot is not None or chunks:
                raise BackupError("Trạng thái snapshot của backup rỗng không hợp lệ.")
            candidate_vector = VectorStore()
            candidate_bm25 = BM25IndexManager()
        else:
            if active_snapshot is None or active_snapshot["id"] != info.snapshot_id:
                raise BackupError("Active snapshot trong SQLite không khớp manifest.")
            expected_snapshot_path = (
                backup_path / "indexes" / info.snapshot_id
            ).resolve()
            if Path(active_snapshot["snapshot_path"]).resolve() != expected_snapshot_path:
                raise BackupError("Đường dẫn active snapshot trong SQLite không hợp lệ.")
            backup_snapshot_store = IndexSnapshotStore(
                backup_path / "indexes",
                embedding_model=self.snapshot_store.embedding_model,
                schema_version=self.snapshot_store.schema_version,
            )
            try:
                candidate_vector, candidate_bm25 = (
                    backup_snapshot_store.load_snapshot(
                        active_snapshot,
                        backup_repository,
                    )
                )
            except SnapshotError as exc:
                raise BackupError(str(exc)) from exc

        return _ValidatedBackup(
            path=backup_path,
            info=info,
            repository=backup_repository,
            vector_store=candidate_vector,
            bm25_manager=candidate_bm25,
        )

    def _capture_runtime_state(self) -> tuple[tuple, tuple]:
        return (
            (
                self.target_vector_store.index,
                self.target_vector_store.contents_map,
                self.target_vector_store.metadatas_map,
                self.target_vector_store.id_order,
                self.target_vector_store.snapshot_id,
            ),
            (
                self.target_bm25_manager.bm25_index,
                self.target_bm25_manager.doc_mapping,
                self.target_bm25_manager.tokenized_corpus,
                self.target_bm25_manager.raw_corpus,
                self.target_bm25_manager.snapshot_id,
            ),
        )

    def _restore_runtime_state(self, state: tuple[tuple, tuple]) -> None:
        vector_state, bm25_state = state
        (
            self.target_vector_store.index,
            self.target_vector_store.contents_map,
            self.target_vector_store.metadatas_map,
            self.target_vector_store.id_order,
            self.target_vector_store.snapshot_id,
        ) = vector_state
        (
            self.target_bm25_manager.bm25_index,
            self.target_bm25_manager.doc_mapping,
            self.target_bm25_manager.tokenized_corpus,
            self.target_bm25_manager.raw_corpus,
            self.target_bm25_manager.snapshot_id,
        ) = bm25_state

    def _publish_runtime(
        self,
        candidate_vector: VectorStore,
        candidate_bm25: BM25IndexManager,
    ) -> None:
        previous_state = self._capture_runtime_state()
        try:
            self.target_vector_store.replace_with(candidate_vector)
            self.target_bm25_manager.replace_with(candidate_bm25)
        except Exception:
            self._restore_runtime_state(previous_state)
            raise

    def _install_snapshot(self, source: _ValidatedBackup) -> tuple[str, Path]:
        snapshot_id = f"snapshot_{uuid4().hex}"
        index_directory = self.snapshot_store.index_directory
        index_directory.mkdir(parents=True, exist_ok=True)
        staging_path = index_directory / f".{snapshot_id}.tmp"
        final_path = index_directory / snapshot_id
        source_path = source.path / "indexes" / source.info.snapshot_id
        try:
            shutil.copytree(source_path, staging_path, symlinks=True)
            for filename in SNAPSHOT_FILENAMES:
                self._require_regular_file(staging_path / filename, staging_path)
            manifest_path = staging_path / MANIFEST_FILENAME
            snapshot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_manifest["snapshot_id"] = snapshot_id
            manifest_path.write_text(
                json.dumps(snapshot_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(staging_path, final_path)
        except Exception:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise
        return snapshot_id, final_path

    def _apply_validated_backup(self, source: _ValidatedBackup) -> str | None:
        working_database = (
            self.repository.database_path.parent
            / f".restore_{uuid4().hex}.db"
        )
        installed_snapshot_id = None
        installed_snapshot_path = None
        try:
            _copy_sqlite_database(
                source.path / BACKUP_DATABASE_FILENAME,
                working_database,
            )
            if source.info.snapshot_id is not None:
                installed_snapshot_id, installed_snapshot_path = (
                    self._install_snapshot(source)
                )
                connection = sqlite3.connect(str(working_database), timeout=5.0)
                try:
                    connection.row_factory = sqlite3.Row
                    original = connection.execute(
                        "SELECT * FROM index_snapshots WHERE status = 'active'"
                    ).fetchone()
                    if original is None:
                        raise BackupError("Backup không có active snapshot trong SQLite.")
                    connection.execute(
                        "UPDATE index_snapshots SET status = 'inactive' "
                        "WHERE status = 'active'"
                    )
                    connection.execute(
                        """
                        INSERT INTO index_snapshots (
                            id, schema_version, embedding_model, snapshot_path,
                            chunk_count, status
                        ) VALUES (?, ?, ?, ?, ?, 'active')
                        """,
                        (
                            installed_snapshot_id,
                            original["schema_version"],
                            original["embedding_model"],
                            str(installed_snapshot_path),
                            original["chunk_count"],
                        ),
                    )
                    connection.commit()
                finally:
                    connection.close()
                source.vector_store.snapshot_id = installed_snapshot_id
                source.bm25_manager.snapshot_id = installed_snapshot_id

            self.repository.database_path.parent.mkdir(parents=True, exist_ok=True)
            _copy_sqlite_database(working_database, self.repository.database_path)
            self._publish_runtime(source.vector_store, source.bm25_manager)
            return installed_snapshot_id
        finally:
            for suffix in ("", "-wal", "-shm"):
                Path(f"{working_database}{suffix}").unlink(missing_ok=True)

    def restore_backup(self, backup_id: str) -> BackupRestoreResult:
        with index_lock:
            source = self._validate_backup(backup_id)
            safety = self.create_backup(kind="pre_restore", prune=False)
            try:
                snapshot_id = self._apply_validated_backup(source)
            except Exception as restore_error:
                try:
                    safety_source = self._validate_backup(safety.backup_id)
                    self._apply_validated_backup(safety_source)
                except Exception as rollback_error:
                    raise BackupError(
                        "Phục hồi thất bại và không thể tự rollback; "
                        f"bản an toàn {safety.backup_id} vẫn được giữ."
                    ) from rollback_error
                raise BackupError(
                    "Không thể phục hồi backup; trạng thái trước đó đã được khôi phục."
                ) from restore_error

            self._prune_backups(
                protected_ids={source.info.backup_id, safety.backup_id}
            )
            return BackupRestoreResult(
                backup_id=source.info.backup_id,
                safety_backup_id=safety.backup_id,
                document_count=source.info.document_count,
                chunk_count=source.info.chunk_count,
                snapshot_id=snapshot_id,
            )
