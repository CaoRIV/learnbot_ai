# Thiết kế Phase 4D — Backup và restore cục bộ

## Mục tiêu

Phase 4D bổ sung khả năng tạo và phục hồi bản sao lưu cục bộ cho kho tri thức
`learnbot_ai`. Một backup phải chứa đủ SQLite và active snapshot FAISS/BM25 để
phục hồi mà không gọi API LLM hoặc tạo lại embedding.

Thiết kế ưu tiên máy Windows cấu hình thấp, không thêm thư viện, giữ SQLite là
nguồn dữ liệu chính và không mở rộng thành hệ thống backup production.

## Phạm vi

Bao gồm:

- Tạo backup thủ công trong thư mục được cấu hình, mặc định `data/backups/`.
- Liệt kê tối đa 10 backup hoàn chỉnh gần nhất.
- Kiểm tra manifest, checksum, schema và snapshot trước khi restore.
- Tự tạo safety backup của trạng thái hiện tại trước mỗi lần restore.
- Phục hồi SQLite, active snapshot và hai index trong bộ nhớ.
- API FastAPI và điều khiển tối giản trong giao diện Next.js.

Không bao gồm:

- Upload hoặc download file backup.
- Nhập ZIP hay đường dẫn backup tùy ý.
- Scheduler, cloud/object storage hoặc mã hóa backup.
- Xóa backup thủ công, incremental backup hoặc retention theo dung lượng.
- Khôi phục chéo phiên bản schema hoặc model embedding.

## Cấu trúc backup

Mỗi backup là một thư mục con trực tiếp của `BACKUP_DIRECTORY`:

```text
data/backups/
└── backup_20260915T143012123456Z_a1b2c3d4/
    ├── learnbot.db
    ├── manifest.json
    └── indexes/
        └── snapshot_<id>/
            ├── manifest.json
            ├── faiss.index
            └── bm25.json
```

Backup được dựng trong thư mục staging ẩn cùng filesystem rồi đổi tên sau khi
hoàn tất. `manifest.json` cấp backup có các trường:

- `format_version`: phiên bản định dạng backup, bắt đầu từ `1`.
- `backup_id`: trùng tên thư mục và phải khớp mẫu ID do hệ thống sinh.
- `created_at`: UTC ISO-8601.
- `kind`: `manual` hoặc `pre_restore`.
- `document_count`, `chunk_count` và `snapshot_id`.
- `database`: tên file, kích thước và SHA-256.
- `snapshot`: `null` cho kho rỗng; nếu có thì chứa đường dẫn tương đối và
  SHA-256 của ba file snapshot cố định.

Không tin cậy đường dẫn ghi trong manifest. Mọi đường dẫn phải được resolve và
kiểm tra vẫn nằm trong đúng thư mục backup hoặc thư mục index được cấu hình.
Không đi theo symlink khi kiểm tra/copy dữ liệu backup.

## Thành phần

### Cấu hình

`config.py` thêm `resolve_backup_directory()` và `BACKUP_DIRECTORY`, mặc định
`data/backups`. `example.env` ghi nhận biến này. Giới hạn retention là hằng số
10 trong module backup vì Phase 4D chưa cần cấu hình động.

### Core backup

Tạo `core/backup.py` với:

- `BackupInfo`: dữ liệu an toàn để trả qua API/UI.
- `BackupRestoreResult`: kết quả restore và ID safety backup.
- `BackupError`: lỗi validation/restore có thông báo tiếng Việt.
- `BackupManager`: tạo, liệt kê, validate, restore và retention.

Manager nhận `SQLiteRepository`, `IndexSnapshotStore`, thư mục backup và các
runtime index qua dependency injection để test bằng thư mục tạm. Tất cả thao
tác create/restore giữ `index_lock`, là khóa đang dùng chung cho ingestion,
delete, rebuild và retrieval.

SQLite được sao chép bằng `sqlite3.Connection.backup()`, không copy trực tiếp
file `.db`, nhờ đó bản sao nhất quán khi database dùng WAL. Snapshot là dữ liệu
bất biến sau khi publish nên chỉ copy thư mục active snapshot được SQLite tham
chiếu.

### API

Thêm ba endpoint:

| Phương thức | Endpoint | Kết quả |
| --- | --- | --- |
| `GET` | `/api/backups` | Danh sách backup hợp lệ, mới nhất trước |
| `POST` | `/api/backups` | Tạo backup `manual` |
| `POST` | `/api/backups/{backup_id}/restore` | Restore và trả ID safety backup |

Backup ID từ URL chỉ được dùng sau khi kiểm tra đúng mẫu và resolve thành thư
mục con trực tiếp của `BACKUP_DIRECTORY`. Backup không tồn tại trả 404; backup
hỏng hoặc không tương thích trả 409; lỗi I/O nội bộ trả 500. Log và thông báo
người dùng dùng tiếng Việt.

### Giao diện Next.js

Panel bảo trì hiển thị:

- Nút **Tạo bản sao lưu**.
- Danh sách chọn tối đa 10 backup, gồm thời điểm và nhãn “Thủ công” hoặc
  “Trước phục hồi”.
- Nút **Khôi phục**.

Trước restore, `window.confirm()` nói rõ dữ liệu hiện tại sẽ bị thay thế và hệ
thống sẽ tạo safety backup. Trong lúc create/restore, upload, delete, rebuild và
hỏi đáp bị vô hiệu hóa để UI không gửi các thao tác xung đột. Sau thành công,
UI tải lại status, tài liệu và danh sách backup. Điều khiển xuất hiện cả trong
context panel desktop và sidebar responsive.

## Luồng tạo backup

1. Giữ `index_lock` và khởi tạo repository.
2. Tạo thư mục staging dưới `BACKUP_DIRECTORY`.
3. Dùng SQLite Online Backup API tạo `learnbot.db` trong staging.
4. Đọc `ready` chunks và active snapshot từ database vừa sao chép.
5. Nếu có chunks nhưng không có active snapshot, hoặc số chunk/snapshot không
   khớp, hủy thao tác và yêu cầu rebuild trước.
6. Nếu có active snapshot, xác minh đường dẫn thuộc `INDEX_DIRECTORY`, copy ba
   file snapshot cố định và cập nhật `snapshot_path` trong database backup để
   trỏ vào bản copy.
7. Tính checksum/kích thước, ghi manifest cuối cùng rồi đổi tên staging thành
   thư mục backup.
8. Chỉ sau thành công mới xóa các backup hợp lệ cũ vượt quá 10 bản. Backup mới
   và safety backup vừa tạo luôn là các bản mới nhất nên không bị xóa.

Mọi lỗi trước bước đổi tên đều xóa staging và không ảnh hưởng kho hiện tại.

## Luồng restore

1. Giữ `index_lock`.
2. Resolve backup ID trong thư mục cấu hình; đọc manifest và kiểm tra toàn bộ
   trường, file, kích thước, checksum và schema SQLite.
3. Dùng repository trỏ vào database backup và `IndexSnapshotStore` trỏ vào
   `indexes/` của backup để load candidate FAISS/BM25. Bước này xác minh model,
   chunk ID, metadata và nội dung trước khi thay đổi live data.
4. Tạo một backup `pre_restore` của trạng thái live. Nếu không tạo được thì
   dừng restore.
5. Chuẩn bị active snapshot trong live `INDEX_DIRECTORY`. Copy qua staging;
   nếu cùng ID đã tồn tại nhưng không khớp, giữ thư mục cũ cho đến khi restore
   thành công để rollback vẫn dùng được.
6. Tạo database restore tạm từ database backup, cập nhật `snapshot_path` của
   active snapshot sang đường dẫn live, rồi dùng SQLite Online Backup API ghi
   nhất quán vào live database.
7. Publish candidate FAISS và BM25 vào runtime dưới cùng `index_lock`; với kho
   rỗng thì clear cả hai runtime index.
8. Nếu bất kỳ bước thay đổi live nào thất bại, dùng đường restore nội bộ để
   phục hồi safety backup mà không tạo thêm safety backup. Nếu rollback cũng
   lỗi, giữ nguyên safety backup và trả lỗi gồm ID đó để người vận hành biết
   bản cứu dữ liệu nằm ở đâu.
9. Sau thành công mới dọn staging/phiên bản snapshot tạm và áp dụng retention.

SQLite là nguồn dữ liệu chính. Trong tiến trình ứng dụng, `index_lock` ngăn
query hoặc mutation nhìn thấy trạng thái giữa các bước. Phase 4D không cam kết
điều phối nhiều process cùng ghi một database; nhu cầu đó thuộc backlog
PostgreSQL/multi-user.

## Quy tắc validation và lỗi

- Backup ID sai định dạng, đường dẫn thoát root, symlink hoặc file ngoài danh
  sách cố định đều bị từ chối.
- Manifest sai version, checksum, số lượng, snapshot ID, schema hoặc embedding
  model bị từ chối trước khi tạo safety backup.
- Kho rỗng hợp lệ phải có `chunk_count = 0`, `snapshot_id = null` và không có
  thư mục snapshot.
- Kho có chunks bắt buộc có đúng một active snapshot tương ứng.
- Không sửa hoặc xóa một backup nguồn trong quá trình restore.
- Thông báo API không lộ đường dẫn tuyệt đối hoặc traceback.

## Kiểm thử

Test core dùng SQLite và snapshot thật trong `tmp_path`, đồng thời mock duy nhất
bước embedding bằng vector NumPy cố định:

- Tạo backup chứa database, snapshot và manifest/checksum đúng.
- Round-trip restore đưa documents, chunks, active snapshot, FAISS và BM25 về
  đúng trạng thái đã backup.
- Backup/restore kho rỗng.
- Từ chối backup khi SQLite và active snapshot không nhất quán.
- Từ chối manifest thiếu file, sai checksum, sai schema/model hoặc path
  traversal trước khi live data thay đổi.
- Tự tạo safety backup trước restore.
- Inject lỗi sau khi live database thay đổi và xác nhận safety backup khôi phục
  documents/runtime cũ.
- Inject lỗi rollback và xác nhận safety backup vẫn tồn tại, lỗi trả đúng ID.
- Retention chỉ giữ 10 backup hợp lệ gần nhất.
- API OpenAPI và response/error mapping cho ba endpoint.
- Frontend TypeScript typecheck và production build.

Cuối cùng chạy toàn bộ `pytest`, `pnpm run typecheck`, `pnpm run build` và
`git diff --check` trước khi kết luận hoàn thành.

## Phát hành

Phase 4D nâng phiên bản minor lên `2.11.0`, cập nhật README tiếng Việt/Anh,
CHANGELOG và `EXPANSION_PLAN.md`. Không tự động tạo backup theo lịch; người dùng
chủ động tạo hoặc hệ thống tạo safety backup ngay trước restore.
