"""
RLVDS-VN Settings Module
========================

Mục đích:
    Định nghĩa cấu hình type-safe cho toàn bộ hệ thống sử dụng Pydantic.

Thư viện sử dụng:
    - pydantic: Type validation
    - pyyaml: Đọc file YAML

Các class cần implement:
    1. DetectionConfig: Cấu hình cho YOLOv5
       - model_path: str (đường dẫn file .pt)
       - confidence_threshold: float (0.0-1.0)
       - device: str ("cuda:0" hoặc "cpu")
    
    2. TrackingConfig: Cấu hình cho object tracking
       - enabled: bool
       - max_age: int (số frame giữ track)
       - iou_threshold: float
    
    3. SpatialConfig: Cấu hình vùng vi phạm
       - violation_zone: list[tuple[int, int]] (polygon vertices)
       - zone_color: tuple[int, int, int] (BGR)
    
    4. TemporalConfig: Cấu hình đèn giao thông
       - red_duration_sec: int
       - green_duration_sec: int
       - yellow_duration_sec: int
    
    5. OCRConfig: Cấu hình PaddleOCR
       - lang: str
       - use_gpu: bool
    
    6. DatabaseConfig: Cấu hình SQLite
       - url: str
    
    7. Settings: Class tổng hợp tất cả config

Hàm cần implement:
    - load_yaml_config(path) -> dict: Đọc file YAML
    - get_settings() -> Settings: Lấy cấu hình (có cache)

TODO:
    [ ] Cài đặt pydantic vào requirements.txt (pydantic>=2.0)
    [ ] Định nghĩa các BaseModel classes
    [ ] Implement load_yaml_config()
    [ ] Implement get_settings() với @lru_cache
    [ ] Hỗ trợ environment variable override (prefix RLVDS_)
"""
