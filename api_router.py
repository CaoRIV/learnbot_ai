"""
REST API 模块（使用FastAPI实现）
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import re
from typing import Dict, Any, List, Optional
import logging
import asyncio
from contextlib import asynccontextmanager
from version import __version__

# 从重构后的模块导入
from config import (
    GEMINI_API_KEY,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    SILICONFLOW_API_KEY,
    is_configured_api_key,
)
from core.generator import query_answer
from core.vector_store import vector_store
from features.web_search import check_serpapi_key
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")


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
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from rag_demo import process_multiple_files
        progress = ProgressCallback()

        result_text = await asyncio.to_thread(
            process_multiple_files,
            [type('obj', (object,), {"name": tmp_path})],
            progress
        )

        os.unlink(tmp_path)
        result = result_text[0] if isinstance(result_text, tuple) else result_text
        chunk_match = re.search(r'(\d+) phân đoạn', result)
        chunks = int(chunk_match.group(1)) if chunk_match else 0
        has_error = "thất bại" in result or "Không thể" in result

        return {
            "status": "error" if has_error else "success",
            "message": result,
            "file_info": {"filename": file.filename, "chunks": chunks}
        }
    except Exception as e:
        logger.error("Không thể xử lý tài liệu: %s", e)
        raise HTTPException(500, f"Không thể xử lý tài liệu: {e}") from e


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
