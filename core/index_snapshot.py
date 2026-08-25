"""Lưu và khôi phục snapshot FAISS/BM25 có kiểm tra toàn vẹn."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from faiss import read_index, write_index

from config import INDEX_DIRECTORY
from core.bm25_index import BM25IndexManager, bm25_manager
from core.embeddings import EMBED_MODEL_NAME
from core.storage import SCHEMA_VERSION, SQLiteRepository
from core.vector_store import AutoFaissIndex, VectorStore, vector_store


MANIFEST_FILENAME = "manifest.json"
FAISS_FILENAME = "faiss.index"
BM25_FILENAME = "bm25.json"


class SnapshotError(RuntimeError):
    """Snapshot không thể được tin cậy hoặc không tương thích."""


@dataclass(frozen=True)
class SnapshotDescriptor:
    snapshot_id: str
    snapshot_path: str
    embedding_model: str
    chunk_count: int
    schema_version: int

    def as_storage_record(self) -> dict:
        return {
            "id": self.snapshot_id,
            "snapshot_path": self.snapshot_path,
            "embedding_model": self.embedding_model,
            "chunk_count": self.chunk_count,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class RestoreResult:
    success: bool
    message: str
    snapshot_id: str | None = None
    chunk_count: int = 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while block := file_handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class IndexSnapshotStore:
    """Quản lý snapshot phiên bản trong một thư mục cục bộ."""

    def __init__(
        self,
        index_directory: str | Path = INDEX_DIRECTORY,
        *,
        embedding_model: str = EMBED_MODEL_NAME,
        schema_version: int = SCHEMA_VERSION,
    ):
        self.index_directory = Path(index_directory).resolve()
        self.embedding_model = embedding_model
        self.schema_version = schema_version

    def _validated_snapshot_path(self, value: str | Path) -> Path:
        path = Path(value).resolve()
        try:
            path.relative_to(self.index_directory)
        except ValueError as exc:
            raise SnapshotError(
                "Đường dẫn snapshot nằm ngoài thư mục chỉ mục được cấu hình."
            ) from exc
        return path

    def write_snapshot(
        self,
        candidate_vector_store: VectorStore,
        candidate_bm25_manager: BM25IndexManager,
    ) -> SnapshotDescriptor:
        """Ghi candidate snapshot vào thư mục tạm rồi đổi tên nguyên tử."""
        if not candidate_vector_store.is_ready:
            raise SnapshotError("Không thể lưu snapshot từ kho vector trống.")

        chunk_ids = list(candidate_vector_store.id_order)
        bm25_ids = [
            candidate_bm25_manager.doc_mapping[index]
            for index in range(len(candidate_bm25_manager.doc_mapping))
        ]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise SnapshotError("Chỉ mục FAISS chứa ID phân đoạn trùng.")
        if bm25_ids != chunk_ids:
            raise SnapshotError("Thứ tự phân đoạn FAISS và BM25 không đồng nhất.")
        if candidate_vector_store.total_chunks != len(chunk_ids):
            raise SnapshotError("Số vector FAISS không khớp metadata phân đoạn.")
        if len(candidate_bm25_manager.tokenized_corpus) != len(chunk_ids):
            raise SnapshotError("Số phân đoạn BM25 không khớp FAISS.")

        snapshot_id = f"snapshot_{uuid4().hex}"
        self.index_directory.mkdir(parents=True, exist_ok=True)
        staging_path = self.index_directory / f".{snapshot_id}.tmp"
        final_path = self.index_directory / snapshot_id
        staging_path.mkdir()

        try:
            faiss_path = staging_path / FAISS_FILENAME
            bm25_path = staging_path / BM25_FILENAME
            manifest_path = staging_path / MANIFEST_FILENAME

            write_index(candidate_vector_store.index.index, str(faiss_path))
            bm25_payload = {
                "doc_ids": bm25_ids,
                "tokenized_corpus": candidate_bm25_manager.tokenized_corpus,
            }
            bm25_path.write_text(
                json.dumps(bm25_payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

            index_info = candidate_vector_store.index.get_index_info()
            manifest = {
                "snapshot_id": snapshot_id,
                "schema_version": self.schema_version,
                "embedding_model": self.embedding_model,
                "chunk_count": len(chunk_ids),
                "id_order": chunk_ids,
                "faiss": {
                    "filename": FAISS_FILENAME,
                    "sha256": _sha256_file(faiss_path),
                    "index_type": index_info["index_type"],
                    "dimension": index_info["dimension"],
                    "nlist": index_info["nlist"],
                    "nprobe": index_info["nprobe"],
                    "m": candidate_vector_store.index.m,
                },
                "bm25": {
                    "filename": BM25_FILENAME,
                    "sha256": _sha256_file(bm25_path),
                },
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(staging_path, final_path)
        except Exception:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise

        return SnapshotDescriptor(
            snapshot_id=snapshot_id,
            snapshot_path=str(final_path),
            embedding_model=self.embedding_model,
            chunk_count=len(chunk_ids),
            schema_version=self.schema_version,
        )

    def remove_snapshot(self, descriptor: SnapshotDescriptor) -> None:
        """Xóa candidate chưa được kích hoạt sau khi transaction thất bại."""
        snapshot_path = self._validated_snapshot_path(descriptor.snapshot_path)
        shutil.rmtree(snapshot_path, ignore_errors=True)

    def load_snapshot(
        self,
        snapshot_record: dict,
        repository: SQLiteRepository,
    ) -> tuple[VectorStore, BM25IndexManager]:
        """Đọc snapshot và tạo candidate index mà chưa thay runtime hiện tại."""
        snapshot_path = self._validated_snapshot_path(snapshot_record["snapshot_path"])
        manifest_path = snapshot_path / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise SnapshotError("Không tìm thấy manifest của snapshot chỉ mục.")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SnapshotError("Manifest snapshot bị hỏng hoặc không đọc được.") from exc

        if manifest.get("snapshot_id") != snapshot_record["id"]:
            raise SnapshotError("Mã snapshot trong manifest không khớp SQLite.")
        if manifest.get("schema_version") != self.schema_version:
            raise SnapshotError("Phiên bản schema của snapshot không tương thích.")
        if snapshot_record.get("schema_version") != self.schema_version:
            raise SnapshotError("Phiên bản schema snapshot trong SQLite không tương thích.")
        if manifest.get("embedding_model") != self.embedding_model:
            raise SnapshotError("Mô hình embedding của snapshot không tương thích.")
        if snapshot_record.get("embedding_model") != self.embedding_model:
            raise SnapshotError("Mô hình embedding snapshot trong SQLite không tương thích.")

        id_order = manifest.get("id_order")
        chunk_count = manifest.get("chunk_count")
        if not isinstance(id_order, list) or len(set(id_order)) != len(id_order):
            raise SnapshotError("Danh sách ID phân đoạn trong snapshot không hợp lệ.")
        if chunk_count != len(id_order) or chunk_count != snapshot_record["chunk_count"]:
            raise SnapshotError("Số phân đoạn trong snapshot không khớp SQLite.")

        faiss_info = manifest.get("faiss") or {}
        bm25_info = manifest.get("bm25") or {}
        if faiss_info.get("filename") != FAISS_FILENAME:
            raise SnapshotError("Tên tệp FAISS trong manifest không hợp lệ.")
        if bm25_info.get("filename") != BM25_FILENAME:
            raise SnapshotError("Tên tệp BM25 trong manifest không hợp lệ.")
        faiss_path = snapshot_path / faiss_info.get("filename", "")
        bm25_path = snapshot_path / bm25_info.get("filename", "")
        for path, info, label in (
            (faiss_path, faiss_info, "FAISS"),
            (bm25_path, bm25_info, "BM25"),
        ):
            if not path.is_file():
                raise SnapshotError(f"Không tìm thấy tệp {label} trong snapshot.")
            if _sha256_file(path) != info.get("sha256"):
                raise SnapshotError(f"Checksum {label} của snapshot không hợp lệ.")

        stored_chunks = repository.list_ready_chunks()
        chunks_by_id = {chunk["id"]: chunk for chunk in stored_chunks}
        if len(chunks_by_id) != len(stored_chunks) or set(chunks_by_id) != set(id_order):
            raise SnapshotError("Chunk trong SQLite không khớp snapshot chỉ mục.")

        ordered_chunks = [chunks_by_id[chunk_id] for chunk_id in id_order]
        documents = [chunk["content"] for chunk in ordered_chunks]
        metadatas = []
        for chunk in ordered_chunks:
            metadata = dict(chunk["metadata"])
            metadata.setdefault("source", chunk["source_name"])
            metadata.setdefault("doc_id", chunk["document_id"])
            metadata.setdefault("page", chunk["page"])
            metadatas.append(metadata)

        try:
            faiss_index = read_index(str(faiss_path))
        except Exception as exc:
            raise SnapshotError("Không thể đọc chỉ mục FAISS từ snapshot.") from exc
        if faiss_index.ntotal != chunk_count:
            raise SnapshotError("Số vector FAISS không khớp manifest snapshot.")
        if faiss_index.d != faiss_info.get("dimension"):
            raise SnapshotError("Số chiều FAISS không khớp manifest snapshot.")

        auto_index = AutoFaissIndex(dimension=faiss_index.d)
        auto_index.index = faiss_index
        auto_index.index_type = faiss_info.get("index_type")
        auto_index.nlist = faiss_info.get("nlist")
        auto_index.nprobe = faiss_info.get("nprobe") or 1
        auto_index.m = faiss_info.get("m")
        candidate_vector_store = VectorStore()
        candidate_vector_store.index = auto_index
        candidate_vector_store.id_order = list(id_order)
        candidate_vector_store.contents_map = dict(zip(id_order, documents))
        candidate_vector_store.metadatas_map = dict(zip(id_order, metadatas))
        candidate_vector_store.snapshot_id = snapshot_record["id"]

        try:
            bm25_payload = json.loads(bm25_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SnapshotError("Dữ liệu BM25 trong snapshot bị hỏng.") from exc
        if bm25_payload.get("doc_ids") != id_order:
            raise SnapshotError("Thứ tự phân đoạn BM25 không khớp FAISS.")
        tokenized_corpus = bm25_payload.get("tokenized_corpus")
        if not isinstance(tokenized_corpus, list) or any(
            not isinstance(tokens, list)
            or any(not isinstance(token, str) for token in tokens)
            for tokens in tokenized_corpus
        ):
            raise SnapshotError("Dữ liệu token BM25 trong snapshot không hợp lệ.")

        candidate_bm25_manager = BM25IndexManager()
        candidate_bm25_manager.load_state(tokenized_corpus, documents, id_order)
        candidate_bm25_manager.snapshot_id = snapshot_record["id"]
        return candidate_vector_store, candidate_bm25_manager


def restore_indexes(
    *,
    repository: SQLiteRepository | None = None,
    snapshot_store: IndexSnapshotStore | None = None,
    target_vector_store: VectorStore = vector_store,
    target_bm25_manager: BM25IndexManager = bm25_manager,
) -> RestoreResult:
    """Khôi phục snapshot đang hoạt động mà không tạo embedding mới."""
    storage = repository or SQLiteRepository()
    snapshots = snapshot_store or IndexSnapshotStore()
    try:
        storage.initialize()
        active_snapshot = storage.get_active_snapshot()
        if active_snapshot is None:
            return RestoreResult(False, "Chưa có snapshot chỉ mục đang hoạt động.")
        candidate_vector, candidate_bm25 = snapshots.load_snapshot(
            active_snapshot,
            storage,
        )
    except Exception as exc:
        message = str(exc) if isinstance(exc, SnapshotError) else f"{type(exc).__name__}: {exc}"
        try:
            if (
                isinstance(exc, SnapshotError)
                and "active_snapshot" in locals()
                and active_snapshot is not None
            ):
                storage.mark_snapshot_invalid(active_snapshot["id"])
        except Exception as mark_exc:
            logging.error("Không thể đánh dấu snapshot lỗi: %s", mark_exc)
        logging.error("Không thể khôi phục snapshot chỉ mục: %s", message)
        return RestoreResult(False, f"Không thể khôi phục snapshot chỉ mục: {message}")

    target_vector_store.replace_with(candidate_vector)
    target_bm25_manager.replace_with(candidate_bm25)
    logging.info(
        "Đã khôi phục snapshot %s với %s phân đoạn.",
        active_snapshot["id"],
        candidate_vector.total_chunks,
    )
    return RestoreResult(
        True,
        f"Đã khôi phục {candidate_vector.total_chunks} phân đoạn từ snapshot.",
        snapshot_id=active_snapshot["id"],
        chunk_count=candidate_vector.total_chunks,
    )
