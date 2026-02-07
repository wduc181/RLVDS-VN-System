"""
Violation Zone Definitions
==========================

Mục đích:
    Định nghĩa và quản lý các vùng vi phạm trên mặt đường.

Input:
    - vertices: list[tuple[int, int]] (polygon points)
    - zone_id: str (tên zone)
    - zone_type: str (violation, warning, safe)

Output:
    - ViolationZone object với các methods check

Classes cần implement:
    1. ViolationZone(BaseSpatialReasoner)
       - __init__(zone_id: str, vertices: list[tuple])
       - is_in_zone(point: tuple) -> bool
       - is_in_zone_bbox(bbox: BoundingBox) -> bool
       - set_zone(vertices: list[tuple]) -> None
       - get_vertices() -> list[tuple]
    
    2. ZoneManager
       - __init__()
       - add_zone(zone: ViolationZone) -> None
       - remove_zone(zone_id: str) -> None
       - check_all_zones(point: tuple) -> list[str]

Cách check violation với bbox:
    - Option 1: Check center point của bbox
    - Option 2: Check bottom-center (chân xe)
    - Option 3: Check nếu bbox overlap với zone

TODO:
    [ ] Implement ViolationZone class
    [ ] Quyết định cách check violation (center vs bottom)
    [ ] Add method để vẽ zone lên frame
    [ ] Support load zones từ config/JSON
"""
