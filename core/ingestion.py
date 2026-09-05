"""Pipeline nhập tài liệu dùng chung cho giao diện và REST API."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

from config import INDEX_DIRECTORY
from core.bm25_index import BM25IndexManager, bm25_manager
from core.document_loader import extract_text_by_page
from core.embeddings import encode_texts
from core.index_snapshot import IndexSnapshotStore, restore_indexes
from core.storage import SQLiteRepository
from core.text_splitter import split_text
from core.vector_store import VectorStore, index_lock, vector_store


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx"}
)

ProgressCallback = Callable[..., object]
_ingestion_lock = index_lock


@dataclass(frozen=True)
class DocumentSource:
    """Tệp nguồn cần nhập cùng tên hiển thị an toàn cho người dùng."""

    path: str
    display_name: str | None = None

    @property
    def name(self) -> str:
        return self.display_name or os.path.basename(self.path)


@dataclass(frozen=True)
class IngestionResult:
    """Kết quả có cấu trúc của một lần nhập tài liệu."""

    message: str
    filenames: list[str]
    total_files: int
    processed_files: int
    failed_files: int
    chunk_count: int
    skipped_files: int = 0

    @property
    def success(self) -> bool:
        return (
            self.processed_files + self.skipped_files > 0
            and self.failed_files == 0
        )


@dataclass(frozen=True)
class _PreparedDocument:
    """Tài liệu đã trích xuất, sẵn sàng lập chỉ mục và lưu SQLite."""

    document_id: str
    source_name: str
    content_hash: str
    file_size: int
    chunks: list[str]
    chunk_ids: list[str]
    metadatas: list[dict]

    def as_storage_record(self) -> dict:
        return {
            "id": self.document_id,
            "source_name": self.source_name,
            "content_hash": self.content_hash,
            "file_size": self.file_size,
            "chunks": [
                {
                    "id": chunk_id,
                    "chunk_index": index,
                    "content": content,
                    "page": metadata.get("page"),
                    "metadata": metadata,
                }
                for index, (chunk_id, content, metadata) in enumerate(
                    zip(self.chunk_ids, self.chunks, self.metadatas)
                )
            ],
        }


def _report_progress(callback: ProgressCallback | None, value: float, description: str):
    if callback is not None:
        callback(value, desc=description)


def _normalize_source(source: DocumentSource | str | os.PathLike[str]) -> DocumentSource:
    if isinstance(source, DocumentSource):
        return source
    return DocumentSource(path=os.fspath(source))


def calculate_file_hash(
    file_path: str | os.PathLike[str],
    block_size: int = 1024 * 1024,
) -> str:
    """Tính SHA-256 theo từng khối để không nạp toàn bộ tài liệu vào RAM."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _append_stored_chunks(
    repository: SQLiteRepository,
    document: dict,
    chunks: list[str],
    chunk_ids: list[str],
    metadatas: list[dict],
) -> int:
    """Thêm chunk đã lưu của một tài liệu vào dữ liệu dựng index."""
    stored_chunks = repository.get_chunks(document["id"])
    for chunk in stored_chunks:
        metadata = dict(chunk["metadata"])
        metadata.setdefault("source", document["source_name"])
        metadata.setdefault("doc_id", document["id"])
        metadata.setdefault("page", chunk["page"])
        chunks.append(chunk["content"])
        chunk_ids.append(chunk["id"])
        metadatas.append(metadata)
    return len(stored_chunks)


def delete_indexed_document(
    document_id: str,
    *,
    repository: SQLiteRepository | None = None,
    snapshot_store: IndexSnapshotStore | None = None,
    target_vector_store: VectorStore = vector_store,
    target_bm25_manager: BM25IndexManager = bm25_manager,
) -> int | None:
    """Xóa tài liệu và chỉ thay index sau khi snapshot mới đã lưu an toàn."""
    with _ingestion_lock:
        storage = repository or SQLiteRepository()
        snapshots = snapshot_store or IndexSnapshotStore(
            storage.database_path.parent / "indexes"
            if repository is not None
            else INDEX_DIRECTORY
        )
        storage.initialize()
        if storage.get_document(document_id) is None:
            return None

        stored_ready_chunks = storage.list_ready_chunks()
        remaining = [
            chunk
            for chunk in stored_ready_chunks
            if chunk["document_id"] != document_id
        ]
        base_chunk_ids = [chunk["id"] for chunk in stored_ready_chunks]
        if not remaining:
            if not storage.delete_document_and_activate_snapshot(
                document_id,
                None,
                expected_chunk_ids=base_chunk_ids,
            ):
                return None
            target_vector_store.clear()
            target_bm25_manager.clear()
            return 0

        chunks = [chunk["content"] for chunk in remaining]
        chunk_ids = [chunk["id"] for chunk in remaining]
        metadatas = []
        for chunk in remaining:
            metadata = dict(chunk["metadata"])
            metadata.setdefault("source", chunk["source_name"])
            metadata.setdefault("doc_id", chunk["document_id"])
            metadata.setdefault("page", chunk["page"])
            metadatas.append(metadata)

        embeddings = encode_texts(chunks, show_progress=False)
        candidate_vector_store = VectorStore()
        candidate_vector_store.build_index(
            chunks,
            chunk_ids,
            metadatas,
            embeddings,
        )
        candidate_bm25_manager = BM25IndexManager()
        candidate_bm25_manager.build_index(chunks, chunk_ids)

        candidate_snapshot = None
        try:
            candidate_snapshot = snapshots.write_snapshot(
                candidate_vector_store,
                candidate_bm25_manager,
            )
            deleted = storage.delete_document_and_activate_snapshot(
                document_id,
                candidate_snapshot.as_storage_record(),
                expected_chunk_ids=base_chunk_ids,
            )
            if not deleted:
                snapshots.remove_snapshot(candidate_snapshot)
                return None
        except Exception:
            if candidate_snapshot is not None:
                snapshots.remove_snapshot(candidate_snapshot)
            raise

        candidate_vector_store.snapshot_id = candidate_snapshot.snapshot_id
        candidate_bm25_manager.snapshot_id = candidate_snapshot.snapshot_id
        target_vector_store.replace_with(candidate_vector_store)
        target_bm25_manager.replace_with(candidate_bm25_manager)
        return len(chunks)


def ingest_documents(
    documents: Iterable[DocumentSource | str | os.PathLike[str]],
    progress: ProgressCallback | None = None,
    repository: SQLiteRepository | None = None,
    snapshot_store: IndexSnapshotStore | None = None,
) -> IngestionResult:
    """Trích xuất, phân đoạn và lập chỉ mục cho một nhóm tài liệu.

    Chỉ mục mới được xây trong bộ nhớ riêng. Tài liệu mới được ghi vào SQLite
    theo một transaction và chỉ sau đó mới thay FAISS/BM25 đang hoạt động.
    Vì vậy lỗi xử lý hoặc lỗi lưu trữ không làm mất kho tri thức hiện tại.
    """
    sources = [_normalize_source(document) for document in documents]
    if not sources:
        return IngestionResult(
            message="Vui lòng chọn ít nhất một tài liệu để xử lý.",
            filenames=[],
            total_files=0,
            processed_files=0,
            failed_files=0,
            chunk_count=0,
        )

    with _ingestion_lock:
        storage = repository or SQLiteRepository()
        snapshots = snapshot_store or IndexSnapshotStore(
            storage.database_path.parent / "indexes"
            if repository is not None
            else INDEX_DIRECTORY
        )
        restore_warning = None
        try:
            storage.initialize()
            stored_documents = storage.list_documents()
            active_snapshot = storage.get_active_snapshot()
            if active_snapshot is not None and (
                vector_store.snapshot_id != active_snapshot["id"]
                or bm25_manager.snapshot_id != active_snapshot["id"]
            ):
                restore_result = restore_indexes(
                    repository=storage,
                    snapshot_store=snapshots,
                )
                if not restore_result.success:
                    restore_warning = (
                        f"{restore_result.message} Đang xây lại chỉ mục từ SQLite."
                    )
                    active_snapshot = None
        except Exception as exc:
            logging.error("Không thể khởi tạo kho SQLite: %s", exc)
            return IngestionResult(
                message=f"Không thể khởi tạo kho dữ liệu SQLite: {exc}",
                filenames=[],
                total_files=len(sources),
                processed_files=0,
                failed_files=len(sources),
                chunk_count=0,
            )

        stored_by_hash = {
            document["content_hash"]: document for document in stored_documents
        }
        prepared_documents: list[_PreparedDocument] = []
        processed_filenames: list[str] = []
        accepted_filenames: list[str] = []
        messages: list[str] = []
        if restore_warning:
            messages.append(restore_warning)
        failed_files = 0
        skipped_files = 0
        accepted_hashes: set[str] = set()

        for position, source in enumerate(sources, start=1):
            display_name = source.name
            try:
                extension = Path(source.path).suffix.lower()
                if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
                    raise ValueError(f"Định dạng {extension or '(không có)'} chưa được hỗ trợ")

                content_hash = calculate_file_hash(source.path)
                if content_hash in accepted_hashes:
                    skipped_files += 1
                    accepted_filenames.append(display_name)
                    messages.append(
                        f"{display_name}: trùng với tài liệu trong cùng lần nhập, đã bỏ qua."
                    )
                    continue

                existing_document = stored_by_hash.get(content_hash)
                if existing_document and existing_document["status"] == "ready":
                    existing_chunks = storage.get_chunks(existing_document["id"])
                    if existing_chunks:
                        accepted_hashes.add(content_hash)
                        skipped_files += 1
                        accepted_filenames.append(display_name)
                        messages.append(
                            f"{display_name}: đã tồn tại trong kho dữ liệu, "
                            "không trích xuất lại."
                        )
                        continue

                _report_progress(
                    progress,
                    (position - 1) / len(sources),
                    f"Đang xử lý tài liệu {position}/{len(sources)}: {display_name}",
                )
                pages = extract_text_by_page(source.path)
                if not pages:
                    raise ValueError("Tài liệu trống hoặc không thể trích xuất văn bản")

                document_id = (
                    existing_document["id"]
                    if existing_document
                    else f"doc_{uuid4().hex}"
                )
                document_chunks: list[str] = []
                document_metadatas: list[dict] = []
                for page_data in pages:
                    page_chunks = split_text(page_data["text"])
                    document_chunks.extend(page_chunks)
                    document_metadatas.extend(
                        {
                            "source": display_name,
                            "doc_id": document_id,
                            "page": page_data["page"],
                        }
                        for _ in page_chunks
                    )

                if not document_chunks:
                    raise ValueError("Tài liệu không tạo được phân đoạn văn bản nào")

                chunk_ids = [
                    f"{document_id}_chunk_{index}"
                    for index in range(len(document_chunks))
                ]
                prepared_documents.append(
                    _PreparedDocument(
                        document_id=document_id,
                        source_name=display_name,
                        content_hash=content_hash,
                        file_size=Path(source.path).stat().st_size,
                        chunks=document_chunks,
                        chunk_ids=chunk_ids,
                        metadatas=document_metadatas,
                    )
                )
                accepted_hashes.add(content_hash)
                processed_filenames.append(display_name)
                accepted_filenames.append(display_name)

                page_summary = (
                    f" từ {len(pages)} trang" if pages[0]["page"] is not None else ""
                )
                messages.append(
                    f"{display_name}: đã tạo {len(document_chunks)} phân đoạn{page_summary}."
                )
            except Exception as exc:
                failed_files += 1
                logging.error("Không thể xử lý %s: %s", display_name, exc)
                messages.append(f"{display_name}: xử lý thất bại – {exc}")

        if not prepared_documents and skipped_files == 0:
            messages.append("\nKhông có tài liệu hợp lệ; kho tri thức hiện tại được giữ nguyên.")
            return IngestionResult(
                message="\n".join(messages),
                filenames=[],
                total_files=len(sources),
                processed_files=0,
                failed_files=failed_files,
                chunk_count=0,
            )

        all_chunks: list[str] = []
        all_metadatas: list[dict] = []
        all_ids: list[str] = []
        for document in stored_documents:
            if document["status"] == "ready":
                _append_stored_chunks(
                    storage,
                    document,
                    all_chunks,
                    all_ids,
                    all_metadatas,
                )

        base_chunk_ids = list(all_ids)

        for prepared in prepared_documents:
            all_chunks.extend(prepared.chunks)
            all_ids.extend(prepared.chunk_ids)
            all_metadatas.extend(prepared.metadatas)

        if not all_chunks:
            messages.append("\nKho dữ liệu không có phân đoạn hợp lệ để lập chỉ mục.")
            return IngestionResult(
                message="\n".join(messages),
                filenames=accepted_filenames,
                total_files=len(sources),
                processed_files=0,
                failed_files=failed_files,
                chunk_count=0,
                skipped_files=skipped_files,
            )

        runtime_matches_snapshot = bool(
            active_snapshot is not None
            and vector_store.snapshot_id == active_snapshot["id"]
            and bm25_manager.snapshot_id == active_snapshot["id"]
        )
        new_chunks = [
            chunk
            for prepared in prepared_documents
            for chunk in prepared.chunks
        ]
        new_ids = [
            chunk_id
            for prepared in prepared_documents
            for chunk_id in prepared.chunk_ids
        ]
        new_metadatas = [
            metadata
            for prepared in prepared_documents
            for metadata in prepared.metadatas
        ]

        if runtime_matches_snapshot:
            base_chunk_ids = list(vector_store.id_order)
            all_ids = list(vector_store.id_order) + new_ids
            all_chunks = [
                vector_store.contents_map[chunk_id]
                for chunk_id in vector_store.id_order
            ] + new_chunks
            all_metadatas = [
                vector_store.metadatas_map[chunk_id]
                for chunk_id in vector_store.id_order
            ] + new_metadatas

        if runtime_matches_snapshot and not prepared_documents:
            _report_progress(progress, 1.0, "Tài liệu đã có trong chỉ mục.")
            messages.append(
                f"\nHoàn tất: 0 tài liệu mới, {skipped_files} tài liệu đã có, "
                f"{failed_files} thất bại; kho hiện có {vector_store.total_chunks} phân đoạn."
            )
            return IngestionResult(
                message="\n".join(messages),
                filenames=accepted_filenames,
                total_files=len(sources),
                processed_files=0,
                failed_files=failed_files,
                chunk_count=vector_store.total_chunks,
                skipped_files=skipped_files,
            )

        _report_progress(progress, 0.8, "Đang tạo embedding...")
        if runtime_matches_snapshot:
            embeddings = encode_texts(new_chunks, show_progress=False)
            candidate_vector_store = vector_store.clone()
            candidate_vector_store.add_chunks(
                new_chunks,
                new_ids,
                new_metadatas,
                embeddings,
            )
        else:
            embeddings = encode_texts(all_chunks, show_progress=False)
            candidate_vector_store = VectorStore()
            candidate_vector_store.build_index(
                all_chunks, all_ids, all_metadatas, embeddings
            )

        _report_progress(progress, 0.9, "Đang xây chỉ mục FAISS và BM25...")
        candidate_bm25_manager = BM25IndexManager()
        candidate_bm25_manager.build_index(all_chunks, all_ids)

        candidate_snapshot = None
        try:
            candidate_snapshot = snapshots.write_snapshot(
                candidate_vector_store,
                candidate_bm25_manager,
            )
            storage.save_ingestion_batch(
                (
                    prepared.as_storage_record()
                    for prepared in prepared_documents
                ),
                snapshot=candidate_snapshot.as_storage_record(),
                expected_chunk_ids=base_chunk_ids,
            )
        except Exception as exc:
            if candidate_snapshot is not None:
                snapshots.remove_snapshot(candidate_snapshot)
            logging.error("Không thể lưu tài liệu và snapshot chỉ mục: %s", exc)
            messages.append(
                f"\nKhông thể lưu tài liệu và snapshot chỉ mục; "
                f"kho tri thức hiện tại được giữ nguyên – {exc}"
            )
            return IngestionResult(
                message="\n".join(messages),
                filenames=[],
                total_files=len(sources),
                processed_files=0,
                failed_files=failed_files + len(prepared_documents),
                chunk_count=0,
                skipped_files=skipped_files,
            )

        candidate_vector_store.snapshot_id = candidate_snapshot.snapshot_id
        candidate_bm25_manager.snapshot_id = candidate_snapshot.snapshot_id
        vector_store.replace_with(candidate_vector_store)
        bm25_manager.replace_with(candidate_bm25_manager)
        _report_progress(progress, 1.0, "Đã hoàn tất lập chỉ mục.")

        processed_files = len(processed_filenames)
        messages.append(
            f"\nHoàn tất: {processed_files} tài liệu mới, "
            f"{skipped_files} tài liệu đã có, {failed_files} thất bại; "
            f"kho hiện có {len(all_chunks)} phân đoạn."
        )
        return IngestionResult(
            message="\n".join(messages),
            filenames=accepted_filenames,
            total_files=len(sources),
            processed_files=processed_files,
            failed_files=failed_files,
            chunk_count=len(all_chunks),
            skipped_files=skipped_files,
        )
