"""Pipeline nhập tài liệu dùng chung cho giao diện và REST API."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

from core.bm25_index import BM25IndexManager, bm25_manager
from core.document_loader import extract_text_by_page
from core.embeddings import encode_texts
from core.text_splitter import split_text
from core.vector_store import VectorStore, vector_store


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx"}
)

ProgressCallback = Callable[..., object]
_ingestion_lock = threading.Lock()


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

    @property
    def success(self) -> bool:
        return self.processed_files > 0 and self.failed_files == 0


def _report_progress(callback: ProgressCallback | None, value: float, description: str):
    if callback is not None:
        callback(value, desc=description)


def _normalize_source(source: DocumentSource | str | os.PathLike[str]) -> DocumentSource:
    if isinstance(source, DocumentSource):
        return source
    return DocumentSource(path=os.fspath(source))


def ingest_documents(
    documents: Iterable[DocumentSource | str | os.PathLike[str]],
    progress: ProgressCallback | None = None,
) -> IngestionResult:
    """Trích xuất, phân đoạn và lập chỉ mục cho một nhóm tài liệu.

    Chỉ mục mới được xây trong bộ nhớ riêng và chỉ thay thế chỉ mục đang dùng
    sau khi cả FAISS lẫn BM25 đã hoàn tất. Vì vậy một lần nhập thất bại không
    làm mất kho tri thức đang hoạt động.
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
        all_chunks: list[str] = []
        all_metadatas: list[dict] = []
        all_ids: list[str] = []
        processed_filenames: list[str] = []
        messages: list[str] = []
        failed_files = 0

        for position, source in enumerate(sources, start=1):
            display_name = source.name
            try:
                extension = Path(source.path).suffix.lower()
                if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
                    raise ValueError(f"Định dạng {extension or '(không có)'} chưa được hỗ trợ")

                _report_progress(
                    progress,
                    (position - 1) / len(sources),
                    f"Đang xử lý tài liệu {position}/{len(sources)}: {display_name}",
                )
                pages = extract_text_by_page(source.path)
                if not pages:
                    raise ValueError("Tài liệu trống hoặc không thể trích xuất văn bản")

                document_id = f"doc_{uuid4().hex}"
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
                all_chunks.extend(document_chunks)
                all_metadatas.extend(document_metadatas)
                all_ids.extend(chunk_ids)
                processed_filenames.append(display_name)

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

        if not all_chunks:
            messages.append("\nKhông có tài liệu hợp lệ; kho tri thức hiện tại được giữ nguyên.")
            return IngestionResult(
                message="\n".join(messages),
                filenames=[],
                total_files=len(sources),
                processed_files=0,
                failed_files=failed_files,
                chunk_count=0,
            )

        _report_progress(progress, 0.8, "Đang tạo embedding...")
        embeddings = encode_texts(all_chunks, show_progress=False)

        _report_progress(progress, 0.9, "Đang xây chỉ mục FAISS và BM25...")
        candidate_vector_store = VectorStore()
        candidate_vector_store.build_index(
            all_chunks, all_ids, all_metadatas, embeddings
        )
        candidate_bm25_manager = BM25IndexManager()
        candidate_bm25_manager.build_index(all_chunks, all_ids)

        vector_store.replace_with(candidate_vector_store)
        bm25_manager.replace_with(candidate_bm25_manager)
        _report_progress(progress, 1.0, "Đã hoàn tất lập chỉ mục.")

        processed_files = len(processed_filenames)
        messages.append(
            f"\nHoàn tất: {processed_files}/{len(sources)} tài liệu, "
            f"{len(all_chunks)} phân đoạn."
        )
        return IngestionResult(
            message="\n".join(messages),
            filenames=processed_filenames,
            total_files=len(sources),
            processed_files=processed_files,
            failed_files=failed_files,
            chunk_count=len(all_chunks),
        )
