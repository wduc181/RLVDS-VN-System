# Project Overview: RLVDS-VN
Vietnam Red Light Violation Detection System


## 1. Project Title & Mô tả ngắn
**Tên dự án:** Vietnam Red Light Violation Detection System. Hệ thống Nhận diện Biển số xe và Phát hiện Vi phạm Giao thông (vượt đèn đỏ).

**Mô tả:** Hệ thống thị giác máy tính tự động phát hiện hành vi **vượt đèn đỏ** và nhận diện biển số xe máy, ô tô tại Việt Nam dựa trên kết hợp giữa không gian (polygon) và thời gian (timing).

**Mục đích:** Đây là một **Learning Project** dành cho môn Thực tập cơ sở, tập trung vào việc kết hợp mô hình học sâu (Deep Learning) với các thuật toán logic để giải quyết bài toán thực tế trong giám sát giao thông.

---

## 2. Problem Statement (Bài toán)
Hệ thống giải quyết bài toán định danh phương tiện và tự động phát hiện hành vi vi phạm tại các nút giao thông.

* **Input:** Video từ camera giám sát tại các ngã tư với góc quay cố định.
* **Logic vi phạm:** Hệ thống thiết lập một vùng đa giác (Polygon) ảo trên mặt đường và một chu kỳ đèn tín hiệu giả lập (Timing).
* **Output:** 
    * Thông tin biển số xe (Text).
    * Trạng thái vi phạm (Violation Status): Được kích hoạt nếu phát hiện biển số nằm trong vùng đa giác tại thời điểm đèn đỏ.

---

## 3. Dataset
* **Dữ liệu huấn luyện:** Vietnamese License Plate dataset từ Roboflow (8.4k images)
  * **Nguồn:** https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0
  * **Số lượng:** 8,400+ ảnh biển số xe Việt Nam
  * **Định dạng:** YOLO format (có thể export trực tiếp cho YOLOv5)
* **Dữ liệu thử nghiệm:** Các đoạn video quay tại ngã tư có vạch dừng rõ ràng để thiết lập vùng đa giác kiểm tra.

---

## 4. Method / Approach (Quy trình)
Hệ thống vận hành theo Pipeline 4 bước:
1.  **Thiết lập (Setup):** Định nghĩa vùng đa giác (Polygon) và chu kỳ đèn đỏ (ví dụ: đỏ 30s - xanh 30s) trong mã nguồn.
2.  **Phát hiện (Detection):** Sử dụng YOLOv5 để tìm vị trí biển số trong khung hình.
3.  **Kiểm tra điều kiện (Violation Check):**
    * Nếu (Trạng thái == Đèn đỏ) VÀ (Tọa độ biển số nằm trong Đa giác): Xác nhận Vi phạm.
4.  **Nhận diện & Ghi lại (Recognition & Logging):** Sử dụng OCR để đọc biển số xe vi phạm và lưu vào cơ sở dữ liệu.

---

## 5. Model / Architecture
* **Detection Model:** **YOLOv5**. Đảm bảo khả năng phát hiện vật thể nhỏ (biển số) ở tốc độ cao.
* **OCR Model:** **ppOCRv4-EN**. Tối ưu cho việc đọc ký tự Latin và chữ số trên biển số xe.
* **Logic Module:** Thuật toán kiểm tra điểm trong đa giác (Point-in-Polygon) kết hợp với bộ đếm thời gian hệ thống (System Timer).

---

## 6. Training Strategy
* **Pre-trained weights:** Sử dụng mô hình đã huấn luyện sẵn cho bài toán phát hiện biển số để đảm bảo độ chính xác ngay lập tức.
* **Fine-tuning:** Tối ưu hóa các thông số ngưỡng (threshold) để giảm thiểu trường hợp nhận diện nhầm do bóng đổ hoặc nhiễu hình ảnh.

---

## 7. Evaluation Metrics (Tiêu chí đánh giá)
* **Accuracy:** Tỷ lệ nhận diện đúng biển số.
* **Precision/Recall:** Độ chính xác trong việc bắt lỗi vi phạm (tránh bắt nhầm xe dừng trước vạch).
* **Real-time Performance:** Đảm bảo logic kiểm tra vi phạm không gây trễ (lag) luồng video.

---

## 8. Results (Kết quả kỳ vọng)
* Tự động xuất danh sách các phương tiện vượt đèn đỏ kèm biển số và thời gian cụ thể.
* Giao diện trực quan hiển thị luồng video, vùng đa giác giám sát và trạng thái đèn tín hiệu hiện tại.
* Dữ liệu vi phạm được lưu trữ tập trung tại SQLite để quản lý.

---

## 9. Risk (Rủi ro)
* **Sai số thời gian:** Logic đèn đỏ giả lập có thể không khớp hoàn toàn với đèn thực tế nếu không có sự đồng bộ từ sensor.
* **Tọa độ vật thể:** Nếu xe chỉ lấn một phần nhỏ vào vùng đa giác, việc xác định vi phạm cần cấu hình ngưỡng chính xác để tránh tranh cãi.
* **Chất lượng video:** Video có độ phân giải thấp hoặc rung lắc có thể làm biển số bị biến dạng khi kiểm tra trong vùng đa giác.

---

## 10. Project Timeline (Quá trình thực hiện)

### Phase 1: Khởi tạo & Thiết kế (Tuần 1-2)
| Task | Mô tả | Status |
|------|-------|--------|
| Phân tích yêu cầu | Xác định scope, input/output |
| Thiết kế kiến trúc | Thiết kế modular architecture |
| Tạo bộ khung dự án | Tạo directory structure, skeleton files |

### Phase 2: Phát triển Core Modules (Tuần 3-5)
| Task | Mô tả | Status |
|------|-------|--------|
| Config System | Implement Pydantic settings |
| Ingestion Module | Video capture, frame buffer |
| Detection Module | Tích hợp YOLOv5, license plate detection |
| OCR Module | Tích hợp PaddleOCR |
| Spatial Module | Point-in-polygon, zone definitions |
| Temporal Module | Traffic light FSM, violation logic |

### Phase 3: Tích hợp & UI (Tuần 6-7)
| Task | Mô tả | Status |
|------|-------|--------|
| Pipeline Integration | Kết nối các modules thành pipeline |
| Persistence Layer | SQLite database, CRUD operations |
| Streamlit UI | Giao diện web hiển thị kết quả |
| Tracking (Optional) | Multi-object tracking |

### Phase 4: Testing & Hoàn thiện (Tuần 8)
| Task | Mô tả | Status |
|------|-------|--------|
| Unit Testing | Viết tests cho từng module |
| Integration Testing | Test toàn bộ pipeline |
| Performance Tuning | Tối ưu tốc độ xử lý |
| Documentation | Hoàn thiện docs |
| Demo | Chuẩn bị demo |