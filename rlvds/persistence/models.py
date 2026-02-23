"""
Data Models
============

Mục đích:
    Định nghĩa data model cho bảng violations trong SQLite.

Tham chiếu:
    - core/base.py::Violation dataclass — dùng cho business logic
    - Module này — dùng cho database mapping

Thư viện sử dụng:
    - dataclasses hoặc sqlite3 Row

===========================================================================
Model cần implement:
===========================================================================

1. ViolationRecord (dataclass hoặc namedtuple)
   Fields mapping với SQLite table:
     - id: int                  (PRIMARY KEY AUTOINCREMENT)
     - plate_text: str          (biển số xe — ví dụ: "29B1-12345")
     - timestamp: str           (thời điểm vi phạm — "23/02/2026 19:00:00")
     - image_path: str          (đường dẫn ảnh: "data/violations/29B1-12345_xxx.jpg")
     - confidence: float        (độ tin cậy OCR: 0.0 - 1.0)
     - zone_id: str             (vùng vi phạm: "default")
     - created_at: str          (thời điểm tạo record)

   Mapping từ sample CSV (camera.py dòng 86-87):
     CSV row: [lp, dt_string, name]
     →  plate_text = lp
     →  timestamp  = dt_string  (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
     →  image_path = "violation_data/img/" + name + ".jpg"

   Helper methods:
     - from_row(row: sqlite3.Row) -> ViolationRecord
       + Convert DB row sang dataclass
     - to_dict() -> dict
       + Convert sang dict để JSON serialize
     - from_violation(v: Violation) -> ViolationRecord
       + Convert Violation (core/base.py) sang DB record

SQL Schema (phải match với database.py):
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_text VARCHAR NOT NULL,
        timestamp DATETIME NOT NULL,
        image_path VARCHAR,
        confidence FLOAT DEFAULT 0.0,
        zone_id VARCHAR DEFAULT 'default',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

TODO:
    [ ] Import dataclasses, sqlite3
    [ ] Implement ViolationRecord dataclass
    [ ] Implement from_row(), to_dict(), from_violation()
    [ ] Đảm bảo fields match với CREATE TABLE schema
"""
