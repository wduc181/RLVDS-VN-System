"""
Violation Repository
=====================

Mục đích:
    Data access layer cho Violation records.
    CRUD operations + data cleaning logic.

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 81-88) — ghi CSV raw violations
    - .github/sample/clean_data.py — lọc biển số hợp lệ + frequency filter
    - .github/sample/clean_and_update_db.py — lọc + upload DB

    clean_data.py logic:
      1. Đọc CSV: df = pd.read_csv('violation_data/license_plate.csv', ...)
      2. Lọc valid: df = df[df['license_plate'].apply(check_valid_plate)]
      3. Tính frequency: license_plate_counts = df['license_plate'].value_counts()
      4. Lọc > 5%: frequent_plates = percentage[percentage > 5].index
      5. Dedup: df.sort_values('time_violation').drop_duplicates('license_plate', keep='last')
      6. Ghi valid_plate.csv

    → Ta implement logic tương tự nhưng dùng SQLite queries

Thư viện sử dụng:
    - sqlite3: Database operations (qua Database class)
    - csv: CSV export (tương thích ngược)

===========================================================================
Class cần implement:
===========================================================================

1. ViolationRepository(BaseRepository)
   - __init__(database: Database)
     + Lưu reference đến Database instance

   - save(violation: ViolationRecord) -> int
     + INSERT INTO violations (plate_text, timestamp, image_path, confidence, zone_id)
       VALUES (?, ?, ?, ?, ?)
     + Return lastrowid

   - get_by_id(violation_id: int) -> ViolationRecord | None
     + SELECT * FROM violations WHERE id = ?

   - get_all(limit: int = 100, offset: int = 0) -> list[ViolationRecord]
     + SELECT * FROM violations ORDER BY timestamp DESC LIMIT ? OFFSET ?

   - get_by_plate(plate_text: str) -> list[ViolationRecord]
     + SELECT * FROM violations WHERE plate_text = ?

   - get_by_date_range(start: str, end: str) -> list[ViolationRecord]
     + SELECT * FROM violations WHERE timestamp BETWEEN ? AND ?

   - delete(violation_id: int) -> bool
     + DELETE FROM violations WHERE id = ?

   - count() -> int
     + SELECT COUNT(*) FROM violations

   - clean_data() -> int
     + Logic từ clean_data.py:
       1. Xóa records với biển số invalid (check_valid_plate == False)
       2. Tính frequency: GROUP BY plate_text, COUNT(*)
       3. Xóa records có biển số xuất hiện < 5% tổng
       4. Dedup: giữ record mới nhất cho mỗi biển số
       5. Return số records đã xóa

   - export_csv(filepath: str) -> None
     + Xuất toàn bộ violations ra CSV file
     + Format: plate_text, timestamp, image_path
     + Tương thích với sample format (camera.py dòng 86)

   - save_violation_image(frame: np.ndarray, filename: str) -> str
     + Lưu ảnh frame vi phạm:
       path = "data/violations/" + filename + ".jpg"
       cv2.imwrite(path, frame)
       return path

Flow tương ứng camera.py dòng 81-88:
    # Trong pipeline, THAY THẾ CSV write bằng:
    repo = ViolationRepository(db)
    image_path = repo.save_violation_image(frame, f"{plate_text}_{timestamp}")
    record = ViolationRecord(plate_text=plate_text, timestamp=dt_string,
                             image_path=image_path, confidence=ocr_score)
    repo.save(record)

Flow tương ứng clean_and_update_db.py dòng 110-113:
    # Sau mỗi cycle xanh, THAY THẾ subprocess call bằng:
    cleaned = repo.clean_data()
    print(f"Cleaned {cleaned} invalid records")

TODO:
    [ ] Import Database, ViolationRecord, BaseRepository từ core.base
    [ ] Implement class ViolationRepository(BaseRepository)
    [ ] Implement save(), get_by_id(), get_all(), get_by_plate()
    [ ] Implement get_by_date_range(), delete(), count()
    [ ] Implement clean_data() — logic từ clean_data.py
    [ ] Implement export_csv() — tương thích sample format
    [ ] Implement save_violation_image()
    [ ] Test: SQLite in-memory → save 5 records → get_all → verify count
"""
