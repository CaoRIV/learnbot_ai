# LearnBot design system

Nguồn tổng hợp: kết quả tra cứu `ui-ux-pro-max`, brand guideline được cung cấp và yêu cầu tránh phong cách “AI-native” đại trà.

## Nguyên tắc

1. LearnBot là công cụ làm việc, không phải landing page AI.
2. Giao diện ưu tiên tài liệu, câu hỏi, câu trả lời và khả năng kiểm chứng nguồn.
3. Chỉ một accent cam đất; không dùng tím–cyan, glow hoặc gradient trang trí.
4. Dùng spacing và divider để phân nhóm; card chỉ xuất hiện khi cần elevation.
5. Mọi trạng thái bất đồng bộ phải có loading, success, error và đường phục hồi rõ ràng.

## Tokens

| Token | Light | Dark |
| --- | --- | --- |
| `background` | `#f3f1eb` | `#141413` |
| `surface` | `#faf9f5` | `#1d1d1b` |
| `surface-muted` | `#e8e6dc` | `#292824` |
| `text` | `#141413` | `#faf9f5` |
| `text-muted` | `#68665f` | `#b0aea5` |
| `accent` | `#d97757` | `#e18769` |
| `success` | `#788c5d` | `#91a876` |
| `danger` | `#b95446` | `#df806f` |

- Base spacing: 4px; nhịp chính: 8, 12, 16, 24, 32, 48px.
- Border: 1px, tương phản tối thiểu rõ trên cả hai theme.
- Radius: 8px cho control, 12–16px cho surface chức năng.
- Shadow: chỉ dùng cho composer, drawer hoặc overlay thật sự nổi.

## Typography

- UI và heading: `Be Vietnam Pro`, `Segoe UI Variable`, `Segoe UI`, sans-serif.
- Metadata và số liệu: `Cascadia Code`, Consolas, monospace.
- Heading dùng weight 600; body 400–500; không dùng serif cho software UI.
- Nội dung dài có line-height 1.65–1.75 và chiều rộng tối đa 72ch.

## Interaction

- Hover/focus transition 150–240ms, không gây layout shift.
- Focus ring luôn nhìn thấy; toàn bộ luồng dùng được bằng bàn phím.
- Nút async bị disable để ngăn gửi lặp.
- Error đặt gần thao tác gây lỗi và cho phép thử lại.
- Tôn trọng `prefers-reduced-motion`.

## Responsive

- Kiểm tra bắt buộc tại 375, 768, 1024 và 1440px.
- Không scroll ngang.
- Panel phụ chuyển thành drawer hoặc ẩn có chủ đích trên màn hình nhỏ.
- Composer không bị che bởi viewport hoặc safe area.

## Không sử dụng

- Emoji làm icon; dùng SVG cùng stroke 1.8.
- Hero marketing, trust badges hoặc copy sáo rỗng trong workspace.
- AI purple, cyan CTA, outer glow, glassmorphism, noise texture.
- Pill cho mọi thành phần, card lồng card, shadow trên mọi surface.
- Animation dài hơn 300ms cho thao tác thường ngày.
