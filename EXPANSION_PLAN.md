# Kế hoạch mở rộng gọn cho learnbot_ai

## 1. Mục tiêu

Ổn định `learnbot_ai` thành một trợ lý hỏi–đáp tài liệu tiếng Việt dùng được cho
cá nhân hoặc một nhóm nhỏ, với ba kết quả rõ ràng:

- Dữ liệu tài liệu được lưu bền vững trong SQLite.
- Không phải lập chỉ mục lại sau mỗi lần khởi động.
- Câu trả lời có nguồn và không đoán khi thiếu bằng chứng.

Phạm vi hiện tại vẫn giữ nguyên: Python 3.10+, máy khoảng 8 GB RAM, FAISS +
BM25, embedding/reranker cục bộ nhẹ và LLM sinh câu trả lời qua API.

## 2. Kiến trúc lưu trữ tối giản

SQLite là **nguồn dữ liệu chính** cho tài liệu và metadata. FAISS vẫn được giữ
làm **index tìm kiếm vector cục bộ** để không phải thay đổi pipeline retrieval
hiện tại. BM25 và manifest index được lưu theo snapshot.

Cấu trúc dữ liệu dự kiến:

```text
data/
├── learnbot.db
└── indexes/
    ├── faiss.index
    ├── bm25.pkl
    └── manifest.json
```

Các bảng SQLite tối thiểu:

- `documents`: tên file, hash, kích thước, trạng thái xử lý, thời gian tạo/cập nhật.
- `chunks`: `document_id`, số thứ tự chunk, nội dung, số trang và metadata cần
  hiển thị citation.
- `index_snapshots`: phiên bản schema, model embedding, đường dẫn snapshot,
  số chunk và trạng thái hợp lệ.

Không lưu vector trong SQLite ở giai đoạn đầu. FAISS tiếp tục chịu trách nhiệm
lưu và tìm kiếm vector; SQLite lưu dữ liệu cần để quản lý và tạo lại index.

## 3. Không làm trong giai đoạn này

Chưa xây đăng nhập, multi-user, database server, object storage, Docker
production, ứng dụng mobile, chatbot bên thứ ba hoặc pipeline vision đầy đủ.
Những phần này chỉ được xem xét sau khi có người dùng và nhu cầu thực tế.

OCR cũng chưa phải mục tiêu bắt buộc; trước mắt tài liệu scan có thể được OCR
bên ngoài rồi đưa vào hệ thống như văn bản thông thường.

## 4. Lộ trình chính

### Giai đoạn 1 — SQLite và chỉ mục bền vững

**Mục tiêu:** Khởi động ứng dụng với kho tài liệu đã lập chỉ mục mà không cần
embedding lại toàn bộ.

**Việc cần làm:**

- Thêm cấu hình `DATABASE_PATH`, mặc định là `data/learnbot.db`.
- Dùng module `sqlite3` có sẵn trong Python; chưa thêm ORM hoặc database
  framework.
- Tạo migration SQL nhỏ cho ba bảng `documents`, `chunks` và
  `index_snapshots`.
- Dùng khóa chính, khóa ngoại và unique constraint để tránh tài liệu/chunk trùng.
- Tạo index cho `documents.content_hash`, `documents.status` và
  `chunks.document_id`.
- Bật `foreign_keys`, chế độ `WAL` và `busy_timeout` khi mở kết nối.
- Lưu nội dung chunk và metadata bằng `executemany` trong transaction ngắn.
- Giữ một lớp repository nhỏ để logic ingestion không phụ thuộc trực tiếp vào
  câu lệnh SQLite.
- Lưu FAISS, BM25 và manifest vào thư mục snapshot cục bộ; ghi trạng thái
  snapshot vào SQLite.
- Chỉ cập nhật database và thay snapshot sau khi toàn bộ candidate index đã xây
  thành công.
- Khi lỗi, rollback transaction và giữ snapshot trước đó.

**Test nghiệm thu:**

- Lưu rồi tải lại cho kết quả retrieval tương đương.
- Khởi động lại đọc được tài liệu và metadata từ SQLite.
- Tài liệu cùng hash không bị nhập trùng.
- Snapshot hỏng hoặc sai model được phát hiện và báo lỗi tiếng Việt.
- Nhập tài liệu thất bại không làm mất dữ liệu đang hoạt động.
- Có thể thêm tài liệu mới mà không xử lý lại các tài liệu không đổi.
- Test dùng database tạm, không phụ thuộc dịch vụ bên ngoài.

### Giai đoạn 2 — Đo chất lượng retrieval tiếng Việt

**Mục tiêu:** Có số liệu để biết thay đổi embedding, BM25 hoặc reranker có thực
sự tốt hơn không.

**Việc cần làm:**

- Tạo một tập nhỏ khoảng 20–50 câu hỏi tiếng Việt có đáp án/chunk đúng.
- Ghi rõ tài liệu và trang đúng cho từng câu hỏi.
- Viết lệnh benchmark cho `Recall@5`, `MRR` và thời gian truy vấn.
- Bổ sung test cho truy vấn tiếng Việt, hybrid merge và metadata trang.
- Đặt tên model embedding/reranker trong cấu hình thay vì hard-code.

**Test nghiệm thu:**

- Benchmark chạy được không cần API key và không tải LLM.
- Kết quả benchmark có thể tái lập trên cùng tập dữ liệu.
- Thay đổi retrieval phải cập nhật test hoặc ghi nhận số liệu trước/sau.

### Giai đoạn 3 — Citation và trả lời có điều kiện

**Mục tiêu:** Người dùng có thể kiểm tra câu trả lời và hệ thống không suy đoán
khi không có tài liệu phù hợp.

**Việc cần làm:**

- Chuẩn hóa citation theo tên tài liệu, trang và chunk ID.
- Hiển thị citation trong UI và API response.
- Thêm ngưỡng liên quan tối thiểu cho context.
- Nếu không đủ bằng chứng, trả lời rõ: chưa tìm thấy thông tin trong tài liệu.
- Giữ nội dung retrieved là dữ liệu không tin cậy, không cho phép ghi đè prompt
  hệ thống.

**Test nghiệm thu:**

- Citation trỏ đúng tài liệu và trang.
- Câu hỏi ngoài phạm vi không tạo câu trả lời khẳng định vô căn cứ.
- API và Gradio trả về cùng quy tắc citation.

## 5. Cách triển khai SQLite

- Đường dẫn database chỉ đọc từ cấu hình; file database không được commit vào Git.
- Migration có phiên bản và chạy được nhiều lần mà không phá dữ liệu.
- Mỗi thao tác ghi dùng transaction ngắn; ingestion lock hiện tại tiếp tục ngăn
  hai lần xây index chạy đồng thời.
- Dùng một connection theo thao tác hoặc theo thread, không chia sẻ connection
  toàn cục giữa FastAPI và Gradio.
- Dùng truy vấn có tham số, không nối trực tiếp dữ liệu người dùng vào SQL.
- Khi backup, sao lưu cả `learnbot.db` và thư mục `data/indexes/` cùng nhau.

## 6. Cách triển khai chung

Mỗi giai đoạn nên là một hoặc vài commit nhỏ, theo thứ tự:

1. Viết test cho hành vi mới.
2. Implement thay đổi tối thiểu.
3. Chạy toàn bộ test hiện có và test tích hợp SQLite.
4. Cập nhật README, migration và changelog nếu cách sử dụng thay đổi.
5. Chỉ chuyển sang giai đoạn sau khi tiêu chí nghiệm thu đạt.

Không thêm thư viện hoặc dịch vụ mới nếu chưa có yêu cầu cụ thể.

## 7. Backlog tùy chọn — chỉ làm khi có nhu cầu

### OCR PDF scan

Chỉ bắt đầu khi có nhiều tài liệu scan thực tế. Ưu tiên một công cụ OCR tùy chọn,
giữ được số trang và không làm chậm luồng PDF văn bản.

### PostgreSQL và multi-user

Chỉ chuyển từ SQLite sang PostgreSQL khi có nhiều người dùng ghi dữ liệu đồng
thời, cần chạy nhiều instance ứng dụng hoặc cần database trên server. Lớp
repository của Phase 1 giúp giới hạn phạm vi thay đổi khi migrate.

### Tích hợp bên ngoài

Chỉ bắt đầu sau khi API `/api/ask` ổn định và có nhu cầu sử dụng từ một kênh cụ
thể. Mỗi tích hợp phải dùng lại API lõi, không tạo pipeline RAG riêng.

## 8. Định nghĩa hoàn thành

Roadmap MVP được xem là hoàn thành khi:

- SQLite lưu được tài liệu, chunk, metadata và trạng thái index.
- FAISS/BM25 được lưu và tải lại an toàn.
- Có benchmark retrieval tiếng Việt có thể chạy lại.
- Câu trả lời có citation hoặc thông báo không đủ dữ liệu.
- Toàn bộ test pass trên môi trường hỗ trợ.
- Không dùng LLM chạy local.
- Không làm mất dữ liệu/index hiện có khi thao tác thất bại.
