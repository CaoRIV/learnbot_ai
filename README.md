<div align="center">

# learnbot_ai

Trợ lý hỏi đáp tài liệu tiếng Việt dựa trên RAG

Tiếng Việt | [English](README_EN.md)

[![CI](https://github.com/CaoRIV/learnbot_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/CaoRIV/learnbot_ai/actions/workflows/ci.yml)
[![Giấy phép: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

</div>

`learnbot_ai` giúp người dùng đặt câu hỏi bằng tiếng Việt và nhận câu trả lời dựa trên nội dung tài liệu đã tải lên. Dự án sử dụng FAISS kết hợp BM25 để truy xuất, có bước xếp hạng lại kết quả và gọi mô hình ngôn ngữ qua API để tạo câu trả lời kèm nguồn và số trang.

Dự án được phát triển từ [weiwill88/Local_Pdf_Chat_RAG](https://github.com/weiwill88/Local_Pdf_Chat_RAG), sau đó được điều chỉnh cho tài liệu tiếng Việt và loại bỏ việc chạy LLM cục bộ.

> Đây là dự án phục vụ học tập và thử nghiệm, chưa phải một dịch vụ kho tri thức sẵn sàng cho môi trường vận hành thực tế. Trước khi dùng với dữ liệu thực tế, nên bổ sung xác thực, phân quyền, lưu trữ bền vững, đánh giá chất lượng, kiểm toán bảo mật và cơ chế bảo vệ dữ liệu.

## Tính năng chính

- **Hỏi đáp dựa trên tài liệu**: chỉ sử dụng nội dung truy xuất được để trả lời; nói rõ khi tài liệu không có thông tin.
- **Trích dẫn nguồn và số trang**: câu trả lời từ PDF sử dụng định dạng `[Tên tài liệu, trang X]`.
- **Truy xuất kết hợp**: kết hợp tìm kiếm vector bằng FAISS và tìm kiếm từ khóa bằng BM25.
- **Tối ưu cho tiếng Việt**: BM25 tách từ bằng `underthesea` thay vì tokenizer tiếng Trung.
- **Xếp hạng lại kết quả**: hỗ trợ CrossEncoder hoặc chấm điểm liên quan qua LLM API.
- **Nhiều dịch vụ LLM**: chọn SiliconFlow, OpenAI hoặc Gemini qua biến `LLM_PROVIDER`.
- **Không chạy LLM cục bộ**: không phụ thuộc Ollama hoặc llama.cpp; embedding và reranker vẫn có thể chạy trên máy.
- **Nhiều định dạng tài liệu**: hỗ trợ PDF, TXT, Markdown, DOCX, XLS/XLSX và PPTX.
- **Hai cách sử dụng**: giao diện Gradio và REST API bằng FastAPI.

## Luồng xử lý RAG

```mermaid
flowchart LR
    A[Tải tài liệu] --> B[Trích xuất văn bản và số trang]
    B --> C[Chia thành các phân đoạn]
    C --> D[Tạo embedding]
    D --> E[Chỉ mục FAISS]
    C --> F[Tách từ tiếng Việt]
    F --> G[Chỉ mục BM25]
    E --> H[Truy xuất kết hợp]
    G --> H
    H --> I[Xếp hạng lại]
    I --> J[Tạo ngữ cảnh kèm nguồn và số trang]
    J --> K[Gọi LLM qua API]
    K --> L[Câu trả lời tiếng Việt có trích dẫn]
```

## Bắt đầu nhanh

### 1. Tải mã nguồn

```bash
git clone https://github.com/CaoRIV/learnbot_ai.git
cd learnbot_ai
```

### 2. Tạo môi trường Python

Dự án yêu cầu Python 3.10 trở lên.

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên Linux hoặc macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Cấu hình dịch vụ LLM

Tạo `.env` từ file mẫu.

Trên Windows PowerShell:

```powershell
Copy-Item example.env .env
```

Trên Linux hoặc macOS:

```bash
cp example.env .env
```

Mở `.env`, đặt `LLM_PROVIDER` thành một trong ba giá trị sau và điền API key tương ứng:

| Giá trị `LLM_PROVIDER` | Biến API key |
| --- | --- |
| `siliconflow` | `SILICONFLOW_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |

Không đưa API key thật lên Git. Các giá trị mẫu bắt đầu bằng `Your_` không được xem là thông tin xác thực hợp lệ.

### 4. Khởi động giao diện Gradio

```bash
python rag_demo.py
```

Ứng dụng ưu tiên địa chỉ `http://127.0.0.1:17995`. Nếu cổng này đang được sử dụng, ứng dụng sẽ lần lượt thử các cổng từ 17996 đến 17999.

Quy trình sử dụng:

1. Chọn một hoặc nhiều tài liệu.
2. Nhấn **Xử lý tài liệu** và chờ quá trình lập chỉ mục hoàn tất.
3. Nhập câu hỏi bằng tiếng Việt.
4. Chọn dịch vụ LLM rồi nhấn **Gửi câu hỏi**.

### 5. Khởi động REST API

```bash
python api_router.py
```

Các endpoint chính:

| Phương thức | Endpoint | Chức năng |
| --- | --- | --- |
| `GET` | `/api/status` | Kiểm tra trạng thái ứng dụng và cấu hình dịch vụ LLM |
| `POST` | `/api/upload` | Tải lên và xử lý tài liệu |
| `POST` | `/api/ask` | Đặt câu hỏi dựa trên tài liệu đã xử lý |

## Cấu trúc dự án

```text
├── config.py                  # Biến môi trường, mô hình và tham số RAG
├── rag_demo.py                # Giao diện Gradio
├── api_router.py              # REST API bằng FastAPI
├── llm_provider.py            # Lớp kết nối SiliconFlow/OpenAI/Gemini
├── migrations/                # Migration schema SQLite có phiên bản
├── core/
│   ├── document_loader.py     # Trích xuất nội dung và số trang
│   ├── text_splitter.py       # Chia văn bản thành các phân đoạn
│   ├── embeddings.py          # Tạo vector embedding
│   ├── storage.py             # Repository SQLite cho tài liệu và metadata
│   ├── vector_store.py        # Quản lý chỉ mục FAISS
│   ├── bm25_index.py          # Tách từ tiếng Việt và chỉ mục BM25
│   ├── retriever.py           # Truy xuất kết hợp và truy xuất đệ quy
│   ├── reranker.py            # Xếp hạng lại kết quả
│   └── generator.py           # Tạo ngữ cảnh, prompt và câu trả lời
├── features/                  # Tìm kiếm web, phát hiện xung đột và tiện ích mở rộng
├── tests/                     # Kiểm thử không cần API key thật
└── .github/                   # Cấu hình CI và các biểu mẫu GitHub
```

## Kiểm thử

Cài các thư viện dành cho phát triển:

```bash
pip install -r requirements-dev.txt
```

Chạy toàn bộ test trên Windows PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest
```

Trên Linux hoặc macOS:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

Bộ test bao gồm:

- cấu hình và lựa chọn dịch vụ LLM mặc định;
- lời gọi API khi có hoặc thiếu thông tin xác thực;
- trích xuất tài liệu và metadata số trang PDF;
- bộ tách từ tiếng Việt cho BM25;
- truy xuất BM25, FAISS và hợp nhất kết quả;
- migration, transaction và chống nhập trùng trong kho SQLite;
- prompt giới hạn câu trả lời trong nội dung tài liệu;
- trích dẫn số trang trong câu trả lời API.

GitHub Actions sẽ biên dịch mã Python và chạy test cho mỗi Pull Request.

## Biến môi trường

Xem đầy đủ tại [`example.env`](example.env). Các biến thường dùng:

| Biến | Mục đích |
| --- | --- |
| `LLM_PROVIDER` | Chọn `siliconflow`, `openai` hoặc `gemini` |
| `SILICONFLOW_API_KEY` | API key của SiliconFlow |
| `SILICONFLOW_MODEL_NAME` | ID mô hình SiliconFlow |
| `OPENAI_API_KEY` | API key của OpenAI |
| `OPENAI_API_URL` | URL tương thích OpenAI Chat Completions |
| `OPENAI_MODEL_NAME` | ID mô hình OpenAI |
| `GEMINI_API_KEY` | API key của Gemini |
| `GEMINI_API_URL` | URL cơ sở của Gemini API |
| `GEMINI_MODEL_NAME` | ID mô hình Gemini |
| `SERPAPI_KEY` | API key tùy chọn cho tìm kiếm web |
| `RERANK_METHOD` | Chọn `cross_encoder` hoặc `llm` |
| `DATABASE_PATH` | Đường dẫn file SQLite, mặc định `data/learnbot.db` |

Ứng dụng không yêu cầu `OLLAMA_HOST` và không gọi Ollama.

## Giới hạn hiện tại

- PDF chỉ được đọc từ lớp văn bản, chưa tích hợp OCR tổng quát; tài liệu scan cần được OCR trước.
- Việc đọc Excel và PowerPoint tập trung vào nội dung chữ, không giữ nguyên bố cục trực quan.
- Tài liệu và phân đoạn đã được lưu trong SQLite, nhưng snapshot FAISS/BM25 chưa
  được tải tự động khi khởi động; index được dựng lại trong lần nhập tiếp theo.
- Mô hình embedding hoặc reranker cục bộ có thể cần tải dữ liệu trong lần chạy đầu tiên.
- Câu hỏi gửi tới LLM và dịch vụ tìm kiếm web có thể được chuyển cho bên thứ ba; cần xem xét phạm vi dữ liệu trước khi sử dụng tài liệu nhạy cảm.
- Máy có 8 GB RAM nên xử lý từng nhóm tài liệu nhỏ để tránh sử dụng quá nhiều bộ nhớ.

## Đóng góp

Bạn có thể gửi báo cáo lỗi có thể tái hiện, cải thiện tài liệu hoặc Pull Request tập trung vào một thay đổi cụ thể. Trước khi đóng góp, hãy đọc [`CONTRIBUTING.md`](CONTRIBUTING.md) và [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

Không tạo Issue công khai cho lỗ hổng bảo mật. Hãy làm theo quy trình trong [`SECURITY.md`](SECURITY.md).

## Phiên bản và bảo trì

- Lịch sử thay đổi: [`CHANGELOG.md`](CHANGELOG.md)
- Các phiên bản đã phát hành: [GitHub Releases](https://github.com/CaoRIV/learnbot_ai/releases)
- Dự án gốc: [weiwill88/Local_Pdf_Chat_RAG](https://github.com/weiwill88/Local_Pdf_Chat_RAG)

## Giấy phép

Dự án được phát hành theo [giấy phép MIT](LICENSE).
