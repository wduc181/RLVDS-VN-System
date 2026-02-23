"""
Violation Zones
================

Mục đích:
    Quản lý các vùng đa giác (polygon zones) dùng để xác định
    khu vực giám sát vi phạm tại ngã tư.

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 40-41) — hardcoded polygon points
    - config/default.yaml → spatial.violation_zone — configurable polygon

Thư viện sử dụng:
    - numpy: Array operations
    - rlvds.spatial.polygon: Các hàm polygon utility

===========================================================================
ViolationZone class
===========================================================================

Class cần implement:

1. ViolationZone
   - __init__(vertices: list[list[int]], zone_id: str = "default",
              color: tuple = (0, 0, 255), thickness: int = 2)
     + Lưu vertices
     + Tạo polygon np.ndarray bằng create_polygon() từ spatial/polygon.py
     + Lưu zone_id, color, thickness

   - contains(point: tuple[float, float]) -> bool
     + Kiểm tra point có nằm trong zone
     + Gọi point_in_polygon() từ spatial/polygon.py

   - apply_mask(frame: np.ndarray) -> np.ndarray
     + Tạo masked frame chỉ giữ vùng zone
     + Gọi create_mask() từ spatial/polygon.py

   - draw(frame: np.ndarray) -> np.ndarray
     + Vẽ viền zone lên frame
     + Gọi draw_polygon() từ spatial/polygon.py

   Vertices mặc định (từ camera.py dòng 40):
     [[1000, 700], [1700, 700], [1900, 1078], [800, 1078]]
   
   Lưu ý: vertices phụ thuộc vào video/camera cụ thể
   → Cần configurable qua config/default.yaml hoặc UI

2. ZoneManager (optional)
   - Quản lý nhiều zones nếu có nhiều vùng giám sát
   - load_zones_from_config(config: SpatialConfig) -> list[ViolationZone]

TODO:
    [ ] Import numpy, các hàm từ spatial.polygon
    [ ] Implement class ViolationZone
    [ ] Implement contains(), apply_mask(), draw()
    [ ] Load vertices từ config hoặc hardcode mặc định
    [ ] Test: tạo zone → kiểm tra contains() với điểm inside/outside
"""
