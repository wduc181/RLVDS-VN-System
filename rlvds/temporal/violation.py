"""
Violation Detection Logic
=========================

Mục đích:
    Kết hợp Spatial (zone) + Temporal (light) để xác định vi phạm.
    Đây là logic trung tâm — quyết định xe nào vượt đèn đỏ.

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 53-88) — toàn bộ violation flow

    camera.py flow:
      1. Kiểm tra elapsed_time % cycle < detection_duration (đèn đỏ?)
      2. Nếu đèn đỏ:
         - Tạo mask polygon → apply lên frame
         - Detect plates trong masked frame
         - Crop plate → preprocess → OCR
         - Nếu đọc được biển số → GHI NHẬN VI PHẠM (lưu CSV + ảnh)
      3. Nếu không phải đèn đỏ:
         - Gọi clean_and_update_db.py để lọc + upload data

Thư viện sử dụng:
    - datetime: Timestamp
    - Các module nội bộ: spatial, temporal, ocr, persistence

===========================================================================
Violation Condition (điều kiện vi phạm):
===========================================================================

    (Light State == RED) AND (Plate detected in Violation Zone) = VIOLATION

    Bất kỳ biển số nào được detect TRONG vùng polygon KHI đèn đỏ
    → Đều được ghi nhận là vi phạm

    Lưu ý: camera.py KHÔNG kiểm tra xe có đang di chuyển hay không
    → Đơn giản: nếu biển số xuất hiện trong zone khi đèn đỏ = vi phạm

===========================================================================
Classes cần implement:
===========================================================================

1. ViolationDetector
   - __init__(zone: ViolationZone, traffic_light: TrafficLightFSM)
     + Lưu references đến zone và traffic_light
     + Khởi tạo set() lưu biển số đã ghi nhận (tránh duplicate)

   - check_frame(frame: np.ndarray, detections: list[Detection]) -> list[dict]
     + Kiểm tra trạng thái đèn: traffic_light.is_red()
     + Nếu KHÔNG đèn đỏ: return []
     + Nếu ĐÈN ĐỎ:
       - Với mỗi detection, kiểm tra center/anchor có trong zone
       - Nếu trong zone → tạo violation_info dict
     + Return list violation_info

   - process_violation(detection: Detection, frame: np.ndarray,
                       plate_text: str) -> Violation
     + Tạo Violation dataclass (từ core/base.py)
     + Lưu: plate_text, timestamp, image_path, confidence, bbox, zone_id

   - is_duplicate(plate_text: str) -> bool
     + Kiểm tra biển số đã ghi nhận chưa (trong session hiện tại)
     + Dùng set() hoặc dict() để track

   - get_light_state() -> LightState
     + Trả về trạng thái đèn hiện tại

   Violation info dict (tương tự camera.py dòng 81-88):
     {
         "plate_text": str,        # Biển số xe
         "timestamp": str,         # datetime.now().strftime("%d/%m/%Y %H:%M:%S")
         "image_name": str,        # plate_text + str(current_time)
         "frame": np.ndarray,      # Frame chứa vi phạm (để lưu ảnh)
         "bbox": tuple,            # Bounding box biển số
         "confidence": float       # OCR confidence
     }

Flow hoàn chỉnh trong pipeline (tương ứng camera.py):
    1. traffic_light.is_red() → True
    2. zone.apply_mask(frame) → masked_frame
    3. detector.detect(masked_frame) → detections
    4. zone.draw(frame) → frame with polygon overlay
    5. Với mỗi detection:
       a. detector.crop_plate(detection, frame) → crop_img
       b. preprocess_image(crop_img) → processed
       c. ocr.recognize(processed) → plate_text
       d. Nếu plate_text != "unknown":
          - violation_detector.process_violation(...)
          - repository.save(violation)
          - Lưu ảnh frame: cv2.imwrite(image_path, frame)

TODO:
    [ ] Import ViolationZone, TrafficLightFSM, Detection, Violation, datetime
    [ ] Implement class ViolationDetector
    [ ] Implement check_frame() — kiểm tra đèn đỏ + detection in zone
    [ ] Implement process_violation() — tạo Violation dataclass
    [ ] Implement is_duplicate() — tránh ghi nhận trùng
    [ ] Test: mock đèn đỏ + mock detection in zone → verify violation detected
"""
