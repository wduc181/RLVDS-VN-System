"""
Violation Repository
====================

Mục đích:
    Data access layer cho Violation records.
    Tách biệt business logic khỏi database operations.

Input:
    - Database session
    - Violation data

Output:
    - CRUD results

Classes cần implement:
    1. ViolationRepository(BaseRepository)
       - __init__(database: Database)
       - save(violation: Violation) -> int (return id)
       - get_by_id(id: int) -> Violation | None
       - get_all() -> list[Violation]
       - get_by_plate(plate_text: str) -> list[Violation]
       - get_by_date_range(start: datetime, end: datetime) -> list[Violation]
       - delete(id: int) -> bool
       - count() -> int

Query Examples:
    - Tìm tất cả vi phạm trong ngày
    - Tìm vi phạm theo biển số
    - Thống kê số vi phạm theo giờ

TODO:
    [ ] Implement CRUD operations
    [ ] Add pagination cho get_all()
    [ ] Add filtering và sorting
    [ ] Handle duplicate violations
"""
