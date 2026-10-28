# Nhận dạng phương tiện giao thông

Dự án đang ở giai đoạn thu thập, gán nhãn và kiểm tra dataset. Chưa có mô hình đã huấn luyện, kết quả detection/classification hay kiểm thử phần cứng được công bố.

## Môi trường

Yêu cầu Python và các thư viện trong `requirements.txt`. Tạo môi trường ảo rồi cài dependency:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Các script có `--help`. `scripts/check_env.py` báo phiên bản thư viện và kiểm tra TensorFlow có thực sự chạy phép tính trên GPU hay không.

## Quy trình dataset

1. Ghi video của nhóm vào `data/raw/` và ghi lại session, địa điểm, thời gian cùng điều kiện quan sát.
2. Trích khung hình, ví dụ: `python scripts/extract_frames.py --video data/raw/S02.mp4 --session-id S02 --fps 1`.
3. Gán nhãn: `python scripts/label_tool.py --session S02`.
4. Kiểm tra annotation và ảnh: `python scripts/validate_dataset.py --deep`.
5. Khi có ít nhất ba session độc lập và dữ liệu hợp lệ, tạo manifest chia tập theo session: `python scripts/create_splits.py`.

Validator trả mã 1 nếu phát hiện lỗi. Công cụ chia tập không di chuyển ảnh, từ chối annotation lỗi, yêu cầu ba session trở lên và mặc định không ghi đè manifest đã có. Tùy chỉnh tỷ lệ và seed trong `configs/default.json`; có thể dùng `--seed`, `--output`, `--deep` và `--overwrite`.

## Trạng thái dữ liệu hiện có

Kiểm tra repository ngày 2026-10-28 cho thấy annotation S01 khai báo 418 frame nhưng thư mục ảnh có 83 ảnh. 83 frame có ảnh được đánh dấu reviewed; 335 tham chiếu ảnh còn thiếu. Validator `--deep` báo 335 lỗi thiếu tệp. Annotation đang chứa 948 box (motorbike 463, car 463, unknown 15, truck 7), nhưng đây là số đếm thô trong JSON, không phải số liệu chất lượng dataset đã kiểm định đầy đủ. Chưa có video gốc trong `data/raw/` (chỉ có `.gitkeep`).

Vì hiện chỉ có một session và annotation còn tham chiếu ảnh thiếu, chưa thể tạo train/validation/test đáng tin cậy. Khôi phục ảnh/video gốc hoặc trích xuất lại vào một session mới; không xóa các annotation hiện có. Sau đó chạy validator và tạo split manifest.

## Kiểm thử

Test dùng `unittest`, không cần pytest:

```powershell
python -m unittest discover -s tests -v
```

Trên môi trường Windows sandbox hạn chế thư mục temp ngoài workspace, có thể đặt `TEMP` và `TMP` vào thư mục ghi được trước khi chạy test.

## Cấu trúc

- `src/annotation/`: lưu trữ nhãn và giao diện gán box OpenCV.
- `src/common/`: cấu hình đường dẫn và I/O ảnh Unicode.
- `scripts/`: kiểm tra môi trường, trích frame, gán nhãn, kiểm tra dataset và chia tập theo session.
- `tests/`: kiểm thử công cụ dataset bằng fixture tổng hợp; fixture không phải dữ liệu huấn luyện.
- `docs/report.md`: khung báo cáo với trạng thái thực nghiệm chưa có dữ liệu.
