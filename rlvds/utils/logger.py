"""
Logging Configuration
=====================

Mục đích:
    Setup logging cho toàn bộ application.

Thư viện sử dụng:
    - logging: Python built-in logging

Output:
    - Logger instance với format chuẩn

Hàm cần implement:
    1. setup_logger(name: str, level: str = "INFO") -> logging.Logger
       - Create logger với format chuẩn
       - Console handler
       - File handler (optional)
    
    2. get_logger(name: str) -> logging.Logger
       - Get existing logger

Log Format:
    [2024-01-01 12:00:00] [INFO] [module_name] Message here

Log Levels:
    - DEBUG: Chi tiết cho development
    - INFO: Thông tin chung
    - WARNING: Cảnh báo
    - ERROR: Lỗi nhưng app vẫn chạy
    - CRITICAL: Lỗi nghiêm trọng

TODO:
    [ ] Setup logging với colorful output (optional)
    [ ] Add rotating file handler
    [ ] Configure từ Settings
"""
