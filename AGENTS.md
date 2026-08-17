# AGENTS.md

## Project
learnbot_ai — trợ lý hỏi-đáp tài liệu tiếng Việt dựa trên RAG.
Fork từ weiwill88/Local_Pdf_Chat_RAG, chuyển sang dùng LLM API thay vì Ollama.

## Rules
- Không dùng model LLM chạy local (không Ollama, không llama.cpp).
- LLM sinh câu trả lời phải gọi qua API (SiliconFlow, OpenAI, hoặc Gemini).
- Embedding và reranker vẫn có thể chạy local vì nhẹ.
- Tokenizer cho BM25 phải dùng underthesea hoặc pyvi thay vì jieba.
- Toàn bộ UI, log, prompt hệ thống dùng tiếng Việt.
- Giữ code Python 3.10+, tương thích Windows và máy cấu hình thấp (8GB RAM).
- Viết test cho các module retrieval và API call trước khi merge.

## Tech stack
- Backend: Python, FastAPI hoặc Gradio (giữ nguyên nếu phù hợp)
- Retrieval: FAISS + BM25 hybrid
- LLM: gọi API ngoài (không local)
- Language: Tiếng Việt cho toàn bộ giao diện người dùng
