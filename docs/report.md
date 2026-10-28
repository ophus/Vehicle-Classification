# Báo cáo kỹ thuật: Ứng dụng thị giác máy trong nhận dạng phương tiện giao thông

> Trạng thái: bản khung trong giai đoạn xây dựng dataset. Không có kết quả huấn luyện hoặc metric thực nghiệm ở thời điểm cập nhật 2026-10-28.

## 1. Tổng quan đề tài

Mục tiêu là xây dựng pipeline phát hiện vị trí và phân loại phương tiện từ ảnh/video do nhóm tự thu thập. Các lớp hiện cấu hình gồm motorbike, car, truck và unknown. Detection, classification, tracking và counting là các tác vụ riêng; ở giai đoạn hiện tại mới có công cụ annotation bbox.

## 2. Cơ sở lý thuyết

### HOG

Gradient ảnh được tính theo hai trục: `g_x = ∂I/∂x`, `g_y = ∂I/∂y`; độ lớn là `m = sqrt(g_x² + g_y²)` và hướng là `atan2(g_y, g_x)`. HOG tổng hợp hướng gradient trong cell và chuẩn hóa các block để mô tả hình dạng cục bộ. HOG + SVM là phương án baseline dự kiến, chưa được triển khai hoặc đo trong repository hiện tại.

### SVM

Linear SVM tìm siêu phẳng có margin lớn, với regularization `C` cân bằng margin và lỗi phân loại. Phương pháp đa lớp cần một chiến lược như one-vs-rest. Pipeline HOG + SVM chưa được triển khai.

### CNN

CNN học bộ lọc tích chập từ trọng số khởi tạo mới; các lớp convolution, activation và pooling tạo biểu diễn phân cấp, còn backpropagation tối ưu loss qua dữ liệu huấn luyện. Chưa có mã CNN, checkpoint hay lần huấn luyện nào.

### IoU

`IoU = diện_tích(giao(B_pred, B_gt)) / diện_tích(hợp(B_pred, B_gt))`. IoU sẽ dùng để ghép detection với ground truth khi pipeline detection và đánh giá được xây dựng.

## 3. Yêu cầu và lựa chọn thuật toán

Dataset chính phải đến từ video nhóm tự ghi. Tách dữ liệu theo session để tránh frame liền kề bị phân vào các tập khác nhau. Hiện có validator và công cụ tạo manifest theo session; script sẽ từ chối dữ liệu annotation có lỗi và yêu cầu ít nhất ba session.

## 4. Thiết kế hệ thống phần mềm và phần cứng

Đã có: trích frame, annotation JSON theo session, giao diện gán bbox OpenCV, cấu hình lớp và I/O ảnh hỗ trợ đường dẫn Unicode. Chưa có inference camera/video, giao thức Serial, firmware ESP32 hoặc mô hình.

## 5. Dataset và huấn luyện

Ngày 2026-10-28, file annotation S01 khai báo 418 frame; thư mục có 83 ảnh. Có 335 tham chiếu ảnh thiếu và 83 frame đã đánh dấu reviewed. JSON chứa tổng cộng 948 box (motorbike 463, car 463, unknown 15, truck 7); các con số này chưa được coi là thống kê dataset hợp lệ vì validator vẫn báo thiếu tệp. Không có video gốc được lưu trong `data/raw/` ngoài `.gitkeep`. Chưa thể tạo split hoặc huấn luyện.

Khi dataset hợp lệ: chạy `python scripts/validate_dataset.py --deep`, rồi `python scripts/create_splits.py`. Split phải được tạo theo session trước khi huấn luyện. Không dùng tập test để chọn siêu tham số.

## 6. Triển khai và kiểm thử

Các bài test hiện kiểm tra validator, trích frame, I/O đường dẫn Unicode, logic gán nhãn và chia session. Chúng dùng ảnh/video tổng hợp nhỏ để kiểm thử phần mềm, không dùng đánh giá chất lượng nhận dạng. Lệnh: `python -m unittest discover -s tests -v`.

## 7. Đánh giá kết quả và thảo luận

Chưa có kết quả thực nghiệm. Chỉ điền accuracy, precision, recall, F1, confusion matrix, IoU/AP/mAP, FPS hoặc độ trễ sau khi có split độc lập và chạy pipeline tương ứng. Khi báo cáo cần ghi thiết bị, điều kiện ghi hình, số mẫu hợp lệ và cách tính metric.

## 8. Kết luận và hướng phát triển

Nền tảng thu thập/gán nhãn đã có thể kiểm thử bằng fixture phần mềm, nhưng dataset thật chưa qua validator do thiếu ảnh tham chiếu. Việc tiếp theo là khôi phục/trích xuất đủ ảnh và bổ sung ít nhất hai session độc lập trước khi tạo split. Sau đó mới triển khai baseline và pipeline học máy.

## 9. Tài liệu tham khảo

- Dalal, N. and Triggs, B. “Histograms of Oriented Gradients for Human Detection,” CVPR 2005.
- Cortes, C. and Vapnik, V. “Support-vector networks,” Machine Learning, 1995.

## 10. Phụ lục

- Quy tắc gán nhãn: [labeling_guideline.md](labeling_guideline.md).
- Cấu hình lớp, split và đường dẫn: `configs/default.json`.
- Validator: `python scripts/validate_dataset.py --deep`.
