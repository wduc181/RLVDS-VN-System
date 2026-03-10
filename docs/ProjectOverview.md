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

## 3. Tech Stack (Công nghệ sử dụng)

| Thành phần | Công nghệ | Vai trò |
|------------|-----------|---------|
| Ngôn ngữ | Python 3.10 | Ngôn ngữ chính |
| Detection | YOLOv5 (via torch.hub / ultralytics) | Phát hiện biển số xe |
| OCR (chính) | PaddleOCR (ppOCRv4-EN) | Đọc ký tự biển số |
| OCR (dự phòng) | YOLOv5 character detection | Detect từng ký tự trên biển số |
| Image Processing | OpenCV, NumPy | Tiền xử lý ảnh (upscale, denoise, contrast) |
| Config | Pydantic + YAML | Type-safe configuration |
| Giao diện | Streamlit | Web dashboard |
| Database | SQLite | Lưu trữ vi phạm |
| Tracking | SORT / ByteTrack (optional) | Theo dõi phương tiện qua frames |

---

## 4. Dataset
* **Dữ liệu huấn luyện:** Vietnamese License Plate dataset từ Roboflow (8.4k images)
  * **Nguồn:** https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0
  * **Số lượng:** 8,400+ ảnh biển số xe Việt Nam
  * **Định dạng:** YOLO format (export trực tiếp cho YOLOv5)
  * **Training configs:** `training/LP_detect.yml` (1 class: license_plate), `training/LP_ocr.yml`
* **Dữ liệu thử nghiệm:** Các đoạn video quay tại ngã tư có vạch dừng rõ ràng để thiết lập vùng đa giác kiểm tra.

---

## 5. Method / Approach (Quy trình)
Hệ thống vận hành theo Pipeline 6 bước:

```
Video → [Polygon Mask] → [YOLO Detect] → [Crop & Preprocess] → [OCR] → [Save DB]
              ↑                                                           ↓
        Violation Zone                                              SQLite + Ảnh
        (khi đèn đỏ)
```

1.  **Thiết lập (Setup):** Định nghĩa vùng đa giác (Polygon) và chu kỳ đèn đỏ (đỏ 30s → xanh 30s) trong config.
2.  **Polygon Masking:** Tạo mask đen, tô trắng vùng polygon, apply lên frame → chỉ giữ vùng giám sát.
3.  **Phát hiện (Detection):** Sử dụng YOLOv5 để tìm vị trí biển số trong vùng đã mask.
4.  **Crop & Tiền xử lý:** Cắt vùng biển số (mở rộng 15%), sau đó:
    * Upscale 2x (cv2.INTER_CUBIC)
    * Khử noise (cv2.fastNlMeansDenoising)
    * Tăng tương phản (CLAHE)
5.  **Kiểm tra & Nhận diện (Violation Check + OCR):**
    * Nếu (Trạng thái == Đèn đỏ) VÀ (Biển số trong vùng Polygon): → Vi phạm
    * Sử dụng PaddleOCR (hoặc YOLOv5 char detect) để đọc biển số
    * Post-process: sửa lỗi OCR phổ biến (O→0, 8→B, 6→G)
6.  **Ghi lại (Logging):** Lưu biển số, thời gian, ảnh vi phạm vào SQLite database.

---

## 6. Model / Architecture

### Detection Model
* **YOLOv5** (via `torch.hub.load('ultralytics/yolov5', 'custom', path=...)`)
* Weights: `lp_vn_det_yolov5n.pt` (nano) hoặc `lp_vn_det_yolov5s.pt` (small)
* Phát hiện vị trí biển số trong frame, output: bounding box (x1, y1, x2, y2) + confidence

### OCR — Phương pháp 1: PaddleOCR (chính)
* **ppOCRv4-EN** — nhận diện text trực tiếp từ ảnh biển số đã crop
* Score threshold: 80% — dưới 80% → "unknown"
* Xử lý multi-line (xe máy 2 dòng): nối bằng dấu "-"

### OCR — Phương pháp 2: YOLOv5 Character Detection (dự phòng)
* **YOLOv5 OCR model** — detect từng ký tự trên biển số
* Weights: `lp_vn_ocr_yolov5s_final.pt`
* Phân loại biển số 1 dòng (ô tô) vs 2 dòng (xe máy) dựa trên line alignment
* Ghép ký tự theo thứ tự tọa độ x

### Image Preprocessing Pipeline
* **Upscale:** 2x bằng cv2.INTER_CUBIC — tăng độ phân giải biển số nhỏ
* **Denoise:** cv2.fastNlMeansDenoising(h=30) — khử noise camera
* **CLAHE:** clipLimit=2.0, tileGridSize=(8,8) — tăng tương phản chữ/nền

### Logic Module
* **Polygon Masking:** cv2.fillPoly + cv2.bitwise_and → giới hạn vùng detect
* **Point-in-Polygon:** cv2.pointPolygonTest → kiểm tra biển số trong vùng
* **Traffic Light FSM:** Cycle timer (elapsed % cycle_duration) → RED/GREEN/YELLOW

---

## 7. Training Strategy
* **Pre-trained weights:** Sử dụng mô hình đã huấn luyện sẵn cho bài toán phát hiện biển số.
* **Fine-tuning:** Tối ưu hóa các thông số ngưỡng (threshold) để giảm thiểu nhận diện nhầm.
* **Training configs:**
  * `LP_detect.yml`: 1 class (license_plate), train/val split theo chuẩn YOLO
  * `LP_ocr.yml`: Config cho character detection model
* **Augmentation:** Tùy chỉnh theo đặc thù biển số VN (góc nghiêng, bóng đổ, chất lượng camera).

---

## 8. Evaluation Metrics (Tiêu chí đánh giá)
* **Detection mAP@0.5:** Độ chính xác phát hiện biển số (target ≥ 85%).
* **OCR Accuracy:** Tỷ lệ đọc đúng biển số hoàn chỉnh.
* **Precision/Recall:** Độ chính xác trong việc bắt lỗi vi phạm (tránh bắt nhầm xe dừng trước vạch).
* **F1-Score:** Chỉ số cân bằng precision/recall cho violation detection.
* **Real-time FPS:** Target ≥ 15 FPS trên GPU, ≥ 5 FPS trên CPU.

---

## 9. Results (Kết quả kỳ vọng)
* Tự động xuất danh sách các phương tiện vượt đèn đỏ kèm biển số và thời gian cụ thể.
* Giao diện trực quan (Streamlit) hiển thị luồng video, vùng đa giác giám sát và trạng thái đèn tín hiệu.
* Dữ liệu vi phạm được lưu trữ tại SQLite, hỗ trợ CRUD + CSV export.
* Hỗ trợ cả biển số 1 dòng (ô tô) và 2 dòng (xe máy).
* Data cleaning: lọc biển số invalid, dedup, frequency filter (giữ biển số xuất hiện > 5%).

---

## 10. Risk (Rủi ro)
* **Sai số thời gian:** Logic đèn đỏ giả lập có thể không khớp hoàn toàn với đèn thực tế nếu không có sự đồng bộ từ sensor.
* **Tọa độ vật thể:** Nếu xe chỉ lấn một phần nhỏ vào vùng đa giác, việc xác định vi phạm cần cấu hình ngưỡng chính xác để tránh tranh cãi.
* **Chất lượng video:** Video có độ phân giải thấp hoặc rung lắc có thể làm biển số bị biến dạng.
* **Lỗi OCR:** Ký tự dễ nhầm lẫn (O/0, B/8, G/6) — cần post-processing logic.
* **Biển số 2 dòng:** Xe máy có biển số 2 hàng, cần phân loại line alignment chính xác.

---

## 11. Project Timeline (Quá trình thực hiện)

### Phase 1: Khởi tạo & Thiết kế (Tuần 1-2)
| Task | Mô tả | Status |
|------|-------|--------|
| Phân tích yêu cầu | Xác định scope, input/output | ✅ Done |
| Thiết kế kiến trúc | Thiết kế modular architecture | ✅ Done |
| Tạo bộ khung dự án | Tạo directory structure, skeleton files | ✅ Done |
| Config System | Implement Pydantic settings + YAML | ✅ Done |
| Logger | Setup logging chuẩn cho toàn bộ app | ✅ Done |

### Phase 2: Phát triển Core Modules (Tuần 3-5)
| Task | Mô tả | Status |
|------|-------|--------|
| Ingestion Module | Video capture, frame iterator | ✅ Done |
| Detection Module | Tích hợp YOLOv5, license plate detection | ✅ Done |
| OCR Module | PaddleOCR + YOLOv5 char detect | 🔄 In Progress |
| Spatial Module | Polygon masking, point-in-polygon | ✅ Done |
| Temporal Module | Traffic light FSM, violation logic | ✅ Done |
| Image Preprocessing | Upscale, denoise, contrast (CLAHE) | 🔄 In Progress |

### Phase 3: Tích hợp & UI (Tuần 6-7)
| Task | Mô tả | Status |
|------|-------|--------|
| Pipeline Integration | Kết nối các modules thành pipeline | ⬜ Not Started |
| Persistence Layer | SQLite database, CRUD + data cleaning | ⬜ Not Started |
| Streamlit UI | Giao diện web hiển thị kết quả | ⬜ Not Started |
| Tracking (Optional) | Multi-object tracking | ⬜ Not Started |

### Phase 4: Testing & Hoàn thiện (Tuần 8)
| Task | Mô tả | Status |
|------|-------|--------|
| Unit Testing | Viết tests cho từng module | ⬜ Not Started |
| Integration Testing | Test toàn bộ pipeline | ⬜ Not Started |
| Performance Tuning | Tối ưu tốc độ xử lý | ⬜ Not Started |
| Documentation | Hoàn thiện docs | ⬜ Not Started |
| Demo | Chuẩn bị demo | ⬜ Not Started |