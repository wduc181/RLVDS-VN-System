"""
Database Connection & Operations
=================================

Mục đích:
    Quản lý kết nối SQLite database để lưu trữ dữ liệu vi phạm.

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 81-88) — ghi CSV
    - .github/sample/clean_and_update_db.py — MongoDB (ta thay bằng SQLite)

    camera.py sử dụng CSV đơn giản:
      with open(license_plate_file, mode='a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([lp, dt_string, name])
          cv2.imwrite("violation_data/img/" + name + ".jpg", frame)

    Ta nâng cấp lên SQLite cho structured queries + CRUD.

Thư viện sử dụng:
    - sqlite3: Built-in Python SQLite

Config (từ config/settings.py):
    database:
      url: "sqlite:///data/rlvds.db"

===========================================================================
Class cần implement:
===========================================================================

1. Database
   - __init__(db_path: str = "data/rlvds.db")
     + Lưu db_path
     + Tạo thư mục parent nếu chưa có

   - connect() -> None
     + self.conn = sqlite3.connect(db_path)
     + self.conn.row_factory = sqlite3.Row  # trả về dict-like rows

   - disconnect() -> None
     + self.conn.close()

   - create_tables() -> None
     + Tạo bảng violations nếu chưa tồn tại:
       CREATE TABLE IF NOT EXISTS violations (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           plate_text VARCHAR NOT NULL,
           timestamp DATETIME NOT NULL,
           image_path VARCHAR,
           confidence FLOAT DEFAULT 0.0,
           zone_id VARCHAR DEFAULT 'default',
           created_at DATETIME DEFAULT CURRENT_TIMESTAMP
       );
     + Tạo indexes:
       CREATE INDEX IF NOT EXISTS idx_plate_text ON violations(plate_text);
       CREATE INDEX IF NOT EXISTS idx_timestamp ON violations(timestamp);

   - execute(query: str, params: tuple = ()) -> sqlite3.Cursor
     + cursor = self.conn.cursor()
     + cursor.execute(query, params)
     + self.conn.commit()
     + return cursor

   - __enter__ / __exit__
     + Hỗ trợ context manager

Ví dụ sử dụng:
    db = Database("data/rlvds.db")
    db.connect()
    db.create_tables()
    db.execute("INSERT INTO violations (plate_text, timestamp) VALUES (?, ?)",
               ("29B1-12345", "2026-02-23 19:00:00"))
    db.disconnect()

TODO:
    [ ] Import sqlite3, pathlib
    [ ] Implement class Database
    [ ] Implement connect(), disconnect(), create_tables()
    [ ] Implement execute()
    [ ] Implement __enter__/__exit__ context manager
    [ ] Test: create in-memory DB → create tables → insert → select → verify
"""
