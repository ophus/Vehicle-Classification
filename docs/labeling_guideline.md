# Quy tắc quay và gán nhãn (v0.1, bản nháp: chốt sau khi xem clip thật)

## Quay
- Giá cố định, quay ngang, 1080p, 30 fps. Không đổi vị trí giữa các phiên.
- Khóa lấy nét/phơi sáng nếu máy cho phép; tắt chống rung, tự zoom, tự đóng khung.
- Tránh lá cây che ống kính (ảnh mẫu có lá lớn ở góc dưới phải).
- Mỗi phiên là một clip độc lập, ghi rõ giờ và thời tiết. Dùng file video gốc, không dùng ảnh chụp màn hình.
- Chỉ dùng video tự quay làm dữ liệu chính. Clip lấy từ nguồn khác chỉ để thử góc nhìn.

## Lớp
| label | gồm | fine_label |
|---|---|---|
| motorbike | xe máy, kể cả người lái và hành khách trong box | motorbike |
| car | sedan, SUV, hatchback, MPV, taxi, bán tải (mặc định) | car, taxi, suv, pickup |
| truck | xe có thùng hàng hoặc cabin tải | truck |
| unknown | xe khách, minibus, van, xe đạp, ba gác, phương tiện lạ | bus, minibus, van, bicycle, other |

Mặc định "bán tải → car" và "minibus/van → unknown" là đề xuất. Đổi trong `configs/default.json` (`fine_to_class`),
không cần gán nhãn lại vì mọi box đều có `fine_label`.

## Box
- Ôm sát phần nhìn thấy của xe (kể cả người lái với xe máy). Xe bị che: vẽ phần nhìn thấy, đặt `occlusion` 1 (che một phần) hoặc 2 (che nhiều).
- Xe bị cắt ở mép ảnh: `truncated: 1`.
- Xe ở xa quá nhỏ để nhận ra loại: **không gán nhãn**, thay vào đó loại vùng đó khỏi đánh giá
  (ignore region, cắt theo một đường ngang; vị trí đường cắt chốt sau khi đo kích thước box thật).
- Làn đối diện (xe nhìn từ phía sau): chưa gán nhãn ở giai đoạn đầu, đưa vào ignore region.

## Tên file
- `session_id`: `S01`, `S02`, ... Frame: `S01_000120.jpg`. Nhãn: `data/annotations/S01.json` (định dạng ở đầu `scripts/validate_dataset.py`).

## Chia tập
70% / 15% / 15% theo phiên (seed trong config). Kiểm tra mỗi tập có đủ mẫu từng lớp trước khi huấn luyện.
