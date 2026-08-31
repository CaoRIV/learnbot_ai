"""Kiểu dữ liệu chuẩn cho bằng chứng retrieval và trích dẫn nguồn."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional


def _normalize_page(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _normalize_score(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


@dataclass(frozen=True)
class Citation:
    """Nguồn có cấu trúc đi kèm một phân đoạn được truy xuất."""

    document: str
    chunk_id: str
    page: Optional[int] = None
    score: Optional[float] = None
    source_type: str = "document"
    url: Optional[str] = None

    def __post_init__(self):
        source_type = "web" if self.source_type == "web" else "document"
        document = str(self.document or "Không rõ nguồn").strip() or "Không rõ nguồn"
        chunk_id = str(self.chunk_id or "").strip()
        url = str(self.url).strip() if self.url else None

        object.__setattr__(self, "document", document)
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "page", _normalize_page(self.page))
        object.__setattr__(self, "score", _normalize_score(self.score))
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "url", url)

    def as_dict(self) -> Dict[str, Any]:
        """Chuyển sang payload citation ổn định cho REST API."""
        return {
            "document": self.document,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "type": self.source_type,
            "url": self.url,
        }

    def as_legacy_source(self) -> Dict[str, Any]:
        """Tạo payload ``sources`` cũ để tương thích với frontend hiện tại."""
        if self.source_type == "web":
            source = {
                "type": "Nguồn web",
                "source": self.document,
            }
            if self.url:
                source["url"] = self.url
            return source

        source = {
            "type": "Tài liệu cục bộ",
            "source": self.document,
        }
        if self.page is not None:
            source["page"] = self.page
        return source


@dataclass(frozen=True)
class RetrievedEvidence:
    """Nội dung retrieval cùng citation được tạo trực tiếp từ metadata chỉ mục."""

    content: str
    citation: Citation
    metadata: Dict[str, Any] = field(default_factory=dict)
