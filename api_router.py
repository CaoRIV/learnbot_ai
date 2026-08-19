"""REST API FastAPI cho hệ thống hỏi đáp tài liệu tiếng Việt."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from version import __version__

from config import (
    GEMINI_API_KEY,
    LLM_PROVIDER,
    MAX_UPLOAD_SIZE_MB,
    OPENAI_API_KEY,
    SILICONFLOW_API_KEY,
    is_configured_api_key,
)
from core.generator import query_answer
from core.ingestion import (
    DocumentSource,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    ingest_documents,
)
from core.vector_store import vector_store
from features.web_search import check_serpapi_key
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
UPLOAD_COPY_CHUNK_SIZE = 1024 * 1024


class ProgressCallback:
    def __init__(self):
        self.progress = 0
        self.description = ""

    def __call__(self, progress, desc=None):
        self.progress = progress
        self.description = desc or ""
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API đã khởi động")
    yield
    logger.info("API đã dừng")


app = FastAPI(
    title="learnbot_ai API",
    description="API hỏi đáp tài liệu tiếng Việt bằng RAG và LLM qua API bên ngoài",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    enable_web_search: bool = False
    model_choice: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class FileProcessResult(BaseModel):
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None


@app.post("/api/upload", response_model=FileProcessResult)
async def upload_file(file: UploadFile = File(...)):
    """Xử lý tài liệu và đưa các phân đoạn vào kho vector."""
    raw_filename = file.filename or ""
    safe_filename = os.path.basename(raw_filename.replace("\\", "/"))
    extension = os.path.splitext(safe_filename)[1].lower()
    if not safe_filename:
        raise HTTPException(400, "Tên tài liệu không hợp lệ")
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise HTTPException(
            415,
            f"Định dạng tài liệu chưa được hỗ trợ. Các định dạng hợp lệ: {supported}",
        )

    temp_path = None
    try:
        uploaded_size = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            tmp_path = tmp.name
            temp_path = tmp_path
            while chunk := await file.read(UPLOAD_COPY_CHUNK_SIZE):
                uploaded_size += len(chunk)
                if uploaded_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        413,
                        f"Tài liệu vượt quá giới hạn {MAX_UPLOAD_SIZE_MB} MB",
                    )
                tmp.write(chunk)

        if uploaded_size == 0:
            raise HTTPException(400, "Tài liệu tải lên đang trống")

        progress = ProgressCallback()
        result = await asyncio.to_thread(
            ingest_documents,
            [DocumentSource(path=temp_path, display_name=safe_filename)],
            progress,
        )

        return {
            "status": "success" if result.success else "error",
            "message": result.message,
            "file_info": {"filename": safe_filename, "chunks": result.chunk_count},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Không thể xử lý tài liệu: %s", exc)
        raise HTTPException(500, f"Không thể xử lý tài liệu: {exc}") from exc
    finally:
        try:
            await file.close()
        except Exception as exc:
            logger.warning("Không thể đóng tệp tải lên %s: %s", safe_filename, exc)
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError as exc:
                logger.warning("Không thể xóa tệp tạm %s: %s", temp_path, exc)


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """Trả lời câu hỏi dựa trên kho tài liệu."""
    if not req.question:
        raise HTTPException(400, "Câu hỏi không được để trống")
    try:
        answer = await asyncio.to_thread(query_answer, req.question, req.enable_web_search, req.model_choice)
        sources = []
        url_matches = re.findall(
            r'\[(Nguồn web|Tài liệu cục bộ):[^\]]+\]\s*(?:\(URL:\s*([^)]+)\))?',
            answer,
        )
        for source_type, url in url_matches:
            sources.append({"type": source_type, "url": url} if url else {"type": source_type})
        for source, page in re.findall(
            r'\[([^\],\n]+),\s*trang\s*(\d+)\]',
            answer,
            flags=re.IGNORECASE,
        ):
            citation = {
                "type": "Tài liệu cục bộ",
                "source": source.strip(),
                "page": int(page),
            }
            if citation not in sources:
                sources.append(citation)

        return {
            "answer": answer, "sources": sources,
            "metadata": {
                "enable_web_search": req.enable_web_search,
                "model": req.model_choice or LLM_PROVIDER,
            }
        }
    except Exception as e:
        logger.error("Quá trình hỏi đáp gặp lỗi: %s", e)
        raise HTTPException(500, f"Không thể xử lý câu hỏi: {e}") from e


@app.get("/api/status")
async def check_status():
    return {
        "status": "healthy",
        "siliconflow_configured": is_configured_api_key(SILICONFLOW_API_KEY),
        "openai_configured": is_configured_api_key(OPENAI_API_KEY),
        "gemini_configured": is_configured_api_key(GEMINI_API_KEY),
        "llm_provider": LLM_PROVIDER,
        "serpapi_configured": check_serpapi_key(),
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks,
        "version": __version__
    }


if __name__ == "__main__":
    import uvicorn
    port = next((p for p in [17995, 17996, 17997, 17998, 17999] if is_port_available(p)), 17995)
    logger.info("Khởi động API trên cổng %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
