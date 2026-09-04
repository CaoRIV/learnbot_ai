"""Lớp lưu trữ SQLite cho tài liệu, phân đoạn và snapshot chỉ mục."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import DATABASE_PATH, PROJECT_ROOT


SCHEMA_VERSION = 1
DEFAULT_MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


class SQLiteRepository:
    """Quản lý dữ liệu SQLite bằng kết nối ngắn theo từng thao tác."""

    def __init__(
        self,
        database_path: str | Path = DATABASE_PATH,
        migrations_dir: str | Path = DEFAULT_MIGRATIONS_DIR,
    ):
        self.database_path = Path(database_path)
        self.migrations_dir = Path(migrations_dir)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Mở kết nối đã cấu hình và tự commit hoặc rollback transaction."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Tạo schema và áp dụng các migration chưa chạy."""
        if not self.migrations_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục migration SQLite: {self.migrations_dir}"
            )

        with self.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied_versions = {
                row["version"]
                for row in connection.execute(
                    "SELECT version FROM schema_migrations"
                )
            }

            for migration_path in sorted(self.migrations_dir.glob("*.sql")):
                version = self._migration_version(migration_path)
                if version in applied_versions:
                    continue
                connection.executescript(migration_path.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                    (version, migration_path.name),
                )

    @staticmethod
    def _migration_version(migration_path: Path) -> int:
        prefix = migration_path.stem.split("_", 1)[0]
        if not prefix.isdigit():
            raise ValueError(
                f"Tên migration không hợp lệ, cần bắt đầu bằng số: {migration_path.name}"
            )
        return int(prefix)

    @staticmethod
    def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def upsert_document(
        self,
        *,
        document_id: str,
        source_name: str,
        content_hash: str,
        file_size: int,
        status: str = "processing",
    ) -> dict[str, Any]:
        """Thêm tài liệu hoặc cập nhật bản ghi đã có cùng hash."""
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, source_name, content_hash, file_size, status
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    source_name = excluded.source_name,
                    file_size = excluded.file_size,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (document_id, source_name, content_hash, file_size, status),
            )
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return dict(row)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return self._as_dict(row)

    def get_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return self._as_dict(row)

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at, id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_document_summaries(self) -> list[dict[str, Any]]:
        """Liệt kê tài liệu và số chunk để hiển thị, không trả dữ liệu nội bộ."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    d.id,
                    d.source_name,
                    d.file_size,
                    d.status,
                    COUNT(c.id) AS chunk_count,
                    d.created_at,
                    d.updated_at
                FROM documents AS d
                LEFT JOIN chunks AS c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        """Xóa tài liệu; khóa ngoại sẽ xóa các chunk liên quan."""
        with self.connection() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,),
            )
        return cursor.rowcount > 0

    def replace_chunks(
        self,
        document_id: str,
        chunks: Iterable[dict[str, Any]],
    ) -> int:
        """Thay toàn bộ chunk của một tài liệu trong một transaction."""
        rows = [
            (
                chunk["id"],
                document_id,
                chunk["chunk_index"],
                chunk["content"],
                chunk.get("page"),
                json.dumps(
                    chunk.get("metadata", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            for chunk in chunks
        ]
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    id, document_id, chunk_index, content, page, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_chunks(self, document_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()

        chunks = []
        for row in rows:
            chunk = dict(row)
            chunk["metadata"] = json.loads(chunk.pop("metadata_json"))
            chunks.append(chunk)
        return chunks

    def list_ready_chunks(self) -> list[dict[str, Any]]:
        """Tải toàn bộ chunk của tài liệu sẵn sàng theo thứ tự ổn định."""
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.*, d.source_name
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.status = 'ready'
                ORDER BY d.created_at, d.id, c.chunk_index
                """
            ).fetchall()

        chunks = []
        for row in rows:
            chunk = dict(row)
            chunk["metadata"] = json.loads(chunk.pop("metadata_json"))
            chunks.append(chunk)
        return chunks

    def save_ingestion_batch(
        self,
        documents: Iterable[dict[str, Any]],
        snapshot: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Lưu một batch tài liệu và chunk trong cùng một transaction.

        Nếu một tài liệu có cùng hash đã tồn tại, bản ghi cũ được giữ nguyên ID
        và nội dung chunk được thay bằng kết quả xử lý mới. Bất kỳ lỗi nào cũng
        rollback toàn bộ batch.
        """
        records = list(documents)
        saved_documents: list[dict[str, Any]] = []

        with self.connection() as connection:
            for document in records:
                existing = connection.execute(
                    "SELECT id FROM documents WHERE content_hash = ?",
                    (document["content_hash"],),
                ).fetchone()
                document_id = existing["id"] if existing else document["id"]

                connection.execute(
                    """
                    INSERT INTO documents (
                        id, source_name, content_hash, file_size, status
                    ) VALUES (?, ?, ?, ?, 'ready')
                    ON CONFLICT(content_hash) DO UPDATE SET
                        source_name = excluded.source_name,
                        file_size = excluded.file_size,
                        status = 'ready',
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        document_id,
                        document["source_name"],
                        document["content_hash"],
                        document["file_size"],
                    ),
                )
                connection.execute(
                    "DELETE FROM chunks WHERE document_id = ?",
                    (document_id,),
                )
                chunk_rows = [
                    (
                        chunk["id"],
                        document_id,
                        chunk["chunk_index"],
                        chunk["content"],
                        chunk.get("page"),
                        json.dumps(
                            chunk.get("metadata", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                    for chunk in document["chunks"]
                ]
                connection.executemany(
                    """
                    INSERT INTO chunks (
                        id, document_id, chunk_index, content, page, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    chunk_rows,
                )
                row = connection.execute(
                    "SELECT * FROM documents WHERE id = ?",
                    (document_id,),
                ).fetchone()
                saved_documents.append(dict(row))

            if snapshot is not None:
                self._activate_snapshot_on_connection(connection, snapshot)

        return saved_documents

    @staticmethod
    def _activate_snapshot_on_connection(
        connection: sqlite3.Connection,
        snapshot: dict[str, Any],
    ) -> sqlite3.Row:
        connection.execute(
            "UPDATE index_snapshots SET status = 'inactive' WHERE status = 'active'"
        )
        connection.execute(
            """
            INSERT INTO index_snapshots (
                id, schema_version, embedding_model, snapshot_path,
                chunk_count, status
            ) VALUES (?, ?, ?, ?, ?, 'active')
            ON CONFLICT(id) DO UPDATE SET
                schema_version = excluded.schema_version,
                embedding_model = excluded.embedding_model,
                snapshot_path = excluded.snapshot_path,
                chunk_count = excluded.chunk_count,
                status = 'active'
            """,
            (
                snapshot["id"],
                snapshot.get("schema_version", SCHEMA_VERSION),
                snapshot["embedding_model"],
                snapshot["snapshot_path"],
                snapshot["chunk_count"],
            ),
        )
        return connection.execute(
            "SELECT * FROM index_snapshots WHERE id = ?",
            (snapshot["id"],),
        ).fetchone()

    def activate_snapshot(
        self,
        *,
        snapshot_id: str,
        embedding_model: str,
        snapshot_path: str,
        chunk_count: int,
        schema_version: int = SCHEMA_VERSION,
    ) -> dict[str, Any]:
        """Ghi snapshot mới và đảm bảo chỉ có một snapshot đang hoạt động."""
        with self.connection() as connection:
            row = self._activate_snapshot_on_connection(
                connection,
                {
                    "id": snapshot_id,
                    "schema_version": schema_version,
                    "embedding_model": embedding_model,
                    "snapshot_path": snapshot_path,
                    "chunk_count": chunk_count,
                },
            )
        return dict(row)

    def get_active_snapshot(self) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM index_snapshots WHERE status = 'active'"
            ).fetchone()
        return self._as_dict(row)

    def mark_snapshot_invalid(self, snapshot_id: str) -> None:
        """Đánh dấu snapshot lỗi để lần khởi động sau không nạp lại."""
        with self.connection() as connection:
            connection.execute(
                "UPDATE index_snapshots SET status = 'invalid' WHERE id = ?",
                (snapshot_id,),
            )
