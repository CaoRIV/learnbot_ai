# LearnBot workspace — page override

Tệp này ghi đè `../MASTER.md` cho giao diện ứng dụng. Kết quả tìm kiếm tự động đề xuất tím–cyan theo mẫu “AI-native”, nhưng hướng đó không phù hợp với mục tiêu sản phẩm và brand guideline đã chọn.

## Định hướng

- Sản phẩm: công cụ hỏi đáp tài liệu tiếng Việt dựa trên RAG.
- Phong cách: Swiss-functional, ấm, tối giản, thiên về công cụ làm việc.
- Bố cục: thanh điều hướng trái, vùng hội thoại chính, panel ngữ cảnh bên phải trên màn hình rộng.
- Không dùng hero marketing, gradient AI, hiệu ứng glow, card lồng card hoặc pill trang trí.

## Màu sắc

| Vai trò | Sáng | Tối |
| --- | --- | --- |
| Nền | `#f3f1eb` | `#141413` |
| Bề mặt | `#faf9f5` | `#1d1d1b` |
| Bề mặt phụ | `#e8e6dc` | `#272724` |
| Chữ chính | `#141413` | `#faf9f5` |
| Chữ phụ | `#68665f` | `#b0aea5` |
| Accent duy nhất | `#d97757` | `#e18769` |
| Thành công chức năng | `#788c5d` | `#91a876` |

Màu xanh lá chỉ dùng cho trạng thái thành công; cam đất là màu tương tác duy nhất.

## Typography

- Stack: `Be Vietnam Pro`, `Segoe UI Variable`, `Segoe UI`, sans-serif.
- Không tải font chặn render. Dùng fallback hệ thống khi font chính không có.
- Heading 600, body 400–500, metadata 12–13px.
- Nội dung câu trả lời tối đa 72ch và line-height 1.7.

## Thành phần và trạng thái

- Border 1px, radius 10–16px; shadow chỉ dùng cho composer nổi ở đáy.
- Nút có hover màu, pressed feedback và focus ring rõ.
- Upload hỗ trợ drag/drop, danh sách tệp, tiến trình, lỗi ngay cạnh thao tác.
- Chat có empty state, pending state, retryable error và nguồn trích dẫn.
- Composer luôn nhìn thấy trên desktop; trên mobile panel tài liệu mở theo drawer.
- Tôn trọng `prefers-reduced-motion`; animation trong khoảng 150–240ms.

## Responsive

- `>= 1180px`: ba vùng — điều hướng/tài liệu, hội thoại, nguồn.
- `768–1179px`: sidebar thu gọn, nguồn nằm dưới câu trả lời.
- `< 768px`: một cột, topbar gọn, tài liệu mở bằng nút, composer cố định hợp lý.
- Kiểm tra tại 375, 768, 1024 và 1440px; không có scroll ngang.
