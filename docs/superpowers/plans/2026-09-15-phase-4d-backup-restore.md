# Phase 4D Backup and Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm backup/restore cục bộ có kiểm tra toàn vẹn cho SQLite và active FAISS/BM25 snapshot, kèm API và UI tiếng Việt.

**Architecture:** `BackupManager` giữ `index_lock`, sao chép SQLite bằng Online Backup API và đóng gói đúng active snapshot với manifest SHA-256. Restore validate toàn bộ backup trước, tạo `pre_restore` safety backup, áp dụng database/snapshot/runtime và dùng safety backup để rollback nếu có lỗi.

**Tech Stack:** Python 3.10+, sqlite3, pathlib, hashlib, shutil, FastAPI/Pydantic, Next.js 16, React, TypeScript.

**Spec:** `docs/superpowers/specs/2026-09-15-phase-4d-backup-restore-design.md`

## Global Constraints

- Không tạo commit; giữ toàn bộ thay đổi trong working tree theo yêu cầu người dùng.
- Không thêm dependency, scheduler, ZIP, cloud storage, upload/download backup hoặc xóa backup thủ công.
- Backup chỉ được đọc/ghi dưới `BACKUP_DIRECTORY`, mặc định `data/backups`.
- Chỉ giữ 10 backup hợp lệ mới nhất.
- UI, API errors và logs dùng tiếng Việt; tương thích Windows và máy 8GB RAM.
- Mọi hành vi core/API mới phải có test đỏ trước implementation.

---

### Task 1: Cấu hình thư mục backup

**Files:**
- Modify: `config.py`
- Modify: `example.env`
- Modify: `.gitignore`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `resolve_backup_directory(value=None) -> Path`, `BACKUP_DIRECTORY: Path`.

- [x] **Step 1: Viết test đỏ**

Thêm test gọi `config.resolve_backup_directory("custom-data/backups")` và assert bằng `config.PROJECT_ROOT / "custom-data" / "backups"`.

- [x] **Step 2: Xác nhận test đỏ**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py::test_backup_directory_is_resolved_from_project_root -q`

Expected: FAIL vì `resolve_backup_directory` chưa tồn tại.

- [x] **Step 3: Implement tối thiểu**

Thêm resolver giống `resolve_index_directory`, khai báo `BACKUP_DIRECTORY`, thêm `BACKUP_DIRECTORY=data/backups` vào `example.env` và ignore `data/backups/`.

- [x] **Step 4: Xác nhận xanh**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -q`

Expected: PASS.

### Task 2: Tạo, validate, liệt kê và retention backup

**Files:**
- Create: `core/backup.py`
- Create: `tests/test_backup.py`

**Interfaces:**
- Consumes: `SQLiteRepository`, `IndexSnapshotStore`, `BACKUP_DIRECTORY`, `index_lock`.
- Produces:
  - `class BackupError(RuntimeError)`
  - `class BackupNotFoundError(BackupError)`
  - `@dataclass(frozen=True) BackupInfo(backup_id, created_at, kind, document_count, chunk_count, snapshot_id, size_bytes)`
  - `class BackupManager(..., max_backups: int = 10)`
  - `BackupManager.create_backup(kind: Literal["manual", "pre_restore"] = "manual", *, prune: bool = True) -> BackupInfo`
  - `BackupManager.list_backups() -> list[BackupInfo]`

- [x] **Step 1: Viết test đỏ cho create/list**

Tạo SQLite/snapshot thật trong `tmp_path`, gọi `create_backup()`, assert database backup đọc được document/chunk, snapshot chứa đúng ba file, manifest có checksum literal 64 ký tự, và `list_backups()` trả đúng `BackupInfo` mới nhất trước.

- [x] **Step 2: Xác nhận test đỏ**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backup.py -k "create or list" -q`

Expected: ERROR import vì `core.backup` chưa tồn tại.

- [x] **Step 3: Implement create/list tối thiểu**

Implement ID `backup_<UTC timestamp>_<8 hex>`, strict regex, staging directory, SQLite `Connection.backup`, active snapshot copy, backup DB `snapshot_path` rewrite, manifest cố định và `os.replace(staging, final)`. `list_backups()` chỉ đọc direct child hợp lệ, bỏ qua staging/manifest hỏng và sort giảm dần theo `created_at`:

```python
with index_lock:
    staging_path.mkdir(parents=True)
    _copy_sqlite(repository.database_path, staging_path / DATABASE_FILENAME)
    snapshot = backup_repository.get_active_snapshot()
    if snapshot is not None:
        shutil.copytree(validated_live_snapshot, backup_snapshot_path)
        _rewrite_active_snapshot_path(backup_database, backup_snapshot_path)
    _write_manifest(staging_path, backup_id, kind)
    os.replace(staging_path, final_path)
```

- [x] **Step 4: Xác nhận xanh create/list**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backup.py -k "create or list" -q`

Expected: PASS.

- [x] **Step 5: Viết test đỏ validation/retention**

Thêm test: kho có chunk nhưng không active snapshot bị từ chối; checksum bị sửa không xuất hiện trong list; tạo 11 backup với clock/ID khác nhau chỉ còn 10 và bản mới nhất vẫn tồn tại.

- [x] **Step 6: Implement validation/retention**

Validate snapshot ID/count/path/files trước finalize. Sau finalize, prune chỉ các direct child có manifest hợp lệ, cũ nhất trước; không đụng staging hoặc đường dẫn ngoài root.

- [x] **Step 7: Xác nhận xanh Task 2**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backup.py -k "not restore" -q`

Expected: PASS.

### Task 3: Restore và safety rollback

**Files:**
- Modify: `core/backup.py`
- Modify: `tests/test_backup.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) BackupRestoreResult(backup_id, safety_backup_id, document_count, chunk_count, snapshot_id)`
  - `BackupManager.restore_backup(backup_id: str) -> BackupRestoreResult`
  - private `_apply_validated_backup(backup_id: str)` không tạo safety backup đệ quy.

- [x] **Step 1: Viết test đỏ round-trip**

Backup trạng thái A, thay SQLite/runtime sang trạng thái B, restore A và assert document/chunk, active snapshot, vector contents/id order và BM25 mapping đều trở lại A; assert có backup `pre_restore` chứa B.

- [x] **Step 2: Xác nhận đỏ**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backup.py::test_restore_round_trip_creates_safety_backup -q`

Expected: FAIL vì `restore_backup` chưa tồn tại.

- [x] **Step 3: Implement validate/apply/restore**

Validate strict ID/direct-child/symlink/manifest/checksum; mở repository backup không chạy migration; load candidate bằng `IndexSnapshotStore` backup. Tạo safety backup, copy snapshot qua staging vào live index root, tạo DB làm việc và rewrite active `snapshot_path`, dùng SQLite backup API ghi live DB, rồi publish candidate runtime dưới `index_lock`:

```python
with index_lock:
    source = _validate_backup(backup_id)
    candidate_vector, candidate_bm25 = _load_candidate(source)
    safety = create_backup(kind="pre_restore", prune=False)
    try:
        _apply_validated_backup(source, candidate_vector, candidate_bm25)
    except Exception as restore_error:
        try:
            _apply_validated_backup(_validate_backup(safety.backup_id))
        except Exception as rollback_error:
            raise BackupError(
                f"Phục hồi thất bại; bản an toàn {safety.backup_id} vẫn được giữ."
            ) from rollback_error
        raise BackupError("Không thể phục hồi backup; trạng thái cũ đã được khôi phục.") from restore_error
```

- [x] **Step 4: Xác nhận xanh round-trip**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backup.py::test_restore_round_trip_creates_safety_backup -q`

Expected: PASS.

- [x] **Step 5: Viết test đỏ cho empty/corruption/path traversal/rollback**

Thêm test riêng cho: restore backup rỗng clear DB/runtime; `../outside` bị từ chối; file snapshot sửa checksum bị từ chối trước safety backup; inject lỗi sau khi live DB đổi và assert safety backup khôi phục B; inject lỗi rollback và assert error chứa safety backup ID, thư mục safety còn tồn tại.

- [x] **Step 6: Implement các nhánh lỗi tối thiểu**

Mọi validation chạy trước safety backup. Khi apply lỗi, gọi `_apply_validated_backup(safety_id)` đúng một lần; nếu rollback lỗi, raise `BackupError` tiếng Việt chứa safety ID và giữ tất cả backup/staging cần chẩn đoán. Empty candidate dùng `VectorStore()` và `BM25IndexManager()`:

```python
if active_snapshot is None and not stored_chunks:
    candidate_vector = VectorStore()
    candidate_bm25 = BM25IndexManager()
elif active_snapshot is None:
    raise BackupError("Backup có dữ liệu nhưng không có snapshot đang hoạt động.")
```

- [x] **Step 7: Xác nhận xanh core**

Run: `.venv\Scripts\python.exe -m pytest tests/test_backup.py tests/test_storage.py tests/test_index_snapshot.py tests/test_ingestion.py -q`

Expected: PASS.

### Task 4: REST API backup

**Files:**
- Modify: `api_router.py`
- Modify: `tests/test_api_status.py`

**Interfaces:**
- Consumes: module-level `backup_manager = BackupManager(repository=document_repository)`.
- Produces Pydantic responses cho `GET /api/backups`, `POST /api/backups`, `POST /api/backups/{backup_id}/restore`.

- [x] **Step 1: Viết test đỏ OpenAPI và success responses**

Assert ba path có response schema typed; monkeypatch manager trả `BackupInfo`/`BackupRestoreResult` đầy đủ và assert JSON dict có `backup_id`, `kind`, counts, `snapshot_id`, `size_bytes`, `safety_backup_id`.

- [x] **Step 2: Xác nhận đỏ**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api_status.py -k "backup" -q`

Expected: FAIL vì paths chưa tồn tại.

- [x] **Step 3: Implement endpoints**

Chạy manager bằng `asyncio.to_thread`. Map backup không tồn tại sang 404, `BackupError` validation/incompatible sang 409, lỗi khác sang 500; response không trả đường dẫn filesystem.

- [x] **Step 4: Viết và chạy test error mapping**

Test literal status/detail tiếng Việt cho 404/409; run `.venv\Scripts\python.exe -m pytest tests/test_api_status.py -q` và xác nhận PASS.

### Task 5: Next.js backup controls

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/components/learnbot-workspace.tsx`
- Modify: `frontend/src/app/globals.css`

**Interfaces:**
- Produces types `BackupInfo`, `BackupRestoreResult`; functions `getBackups`, `createBackup`, `restoreBackup`.

- [x] **Step 1: Thêm client API typed**

Thêm types khớp Pydantic và ba fetch calls. Restore URL dùng `encodeURIComponent(backupId)`.

- [x] **Step 2: Thêm state/handlers**

Thêm `backups`, `selectedBackupId`, `isCreatingBackup`, `restoringBackupId`, `backupMessage`; refresh backups khi mount; create refresh list; restore dùng confirm literal tiếng Việt rồi refresh backups/documents/status.

- [x] **Step 3: Khóa thao tác xung đột**

Tạo `isMaintaining = isRebuilding || isCreatingBackup || restoringBackupId !== null` và dùng nó cho upload, delete, rebuild, ask, textarea/send và backup buttons.

- [x] **Step 4: Render responsive controls**

Trong desktop context panel và responsive sidebar, render select backup với nhãn thời gian/kind, nút tạo, nút restore và `role="status"`/`role="alert"` phù hợp. CSS dùng lại typography/border hiện tại, không thêm modal hoặc card lồng nhau.

- [x] **Step 5: Xác nhận frontend**

Run: `pnpm run typecheck` trong `frontend`, sau đó `pnpm run build`.

Expected: cả hai exit 0.

### Task 6: Phát hành và xác minh toàn bộ

**Files:**
- Modify: `version.py`
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `EXPANSION_PLAN.md`

- [x] **Step 1: Cập nhật tài liệu**

Nâng `__version__` từ `2.10.0` lên `2.11.0`; ghi ba endpoint, giới hạn 10, safety backup, local-only và không có scheduler/upload/download. Đánh dấu Phase 4D hoàn tất trong expansion plan.

- [x] **Step 2: Review code độc lập**

Review tập trung data loss, SQLite WAL, path traversal/symlink, rollback, retention và memory usage; sửa mọi Critical/Important bằng TDD.

- [x] **Step 3: Xác minh cuối**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q
Set-Location frontend
pnpm run typecheck
pnpm run build
Set-Location ..
git diff --check
git status --short
```

Expected: toàn bộ test pass, typecheck/build exit 0, không có whitespace error; working tree chỉ chứa các file Phase 4D và không có commit mới.
