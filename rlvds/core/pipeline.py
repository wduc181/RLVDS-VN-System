"""
RLVDS Processing Pipeline
=========================

Mục đích:
    Orchestrate toàn bộ luồng xử lý từ video input đến lưu violation.

Luồng xử lý (Pipeline Flow):
    Frame → Detection → Tracking → Spatial Check → Temporal Check → OCR → Persist
    
    1. Ingestion: Lấy frame từ video/camera
    2. Detection: Phát hiện biển số trong frame
    3. Tracking: Theo dõi biển số qua các frame
    4. Spatial: Kiểm tra biển số có trong vùng vi phạm không
    5. Temporal: Kiểm tra đèn đỏ + trong vùng = vi phạm
    6. OCR: Đọc text biển số xe vi phạm
    7. Persist: Lưu thông tin vi phạm vào database

Input:
    - video_source: str (đường dẫn video hoặc camera ID)
    - config: Settings (cấu hình hệ thống)

Output:
    - Violations được lưu vào database
    - Có thể stream results cho visualization

Classes cần implement:
    1. Pipeline
       - __init__(config: Settings)
       - run(video_source: str) -> None
       - process_frame(frame: np.ndarray) -> list[Violation]
       - stop() -> None

TODO:
    [ ] Inject dependencies (detector, tracker, ocr, etc.)
    [ ] Implement main processing loop
    [ ] Handle exceptions gracefully
    [ ] Add logging tại mỗi stage
    [ ] Support cả real-time và batch processing
"""
