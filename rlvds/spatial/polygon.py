"""
Point-in-Polygon Algorithm
==========================

Mục đích:
    Kiểm tra xem một điểm có nằm trong polygon hay không.
    Dùng để xác định xe có trong vùng vi phạm.

Thư viện sử dụng:
    - shapely: Geometry operations (recommended)
    - Hoặc implement Ray Casting algorithm thủ công

Input:
    - point: tuple[int, int] (x, y)
    - polygon: list[tuple[int, int]] (list of vertices)

Output:
    - bool: True nếu point nằm trong polygon

Hàm cần implement:
    1. point_in_polygon(point: tuple, polygon: list) -> bool
       - Sử dụng Ray Casting hoặc Shapely
    
    2. polygon_iou(poly1: list, poly2: list) -> float
       - Tính IOU giữa 2 polygons (optional)

Ví dụ sử dụng Shapely:
    from shapely.geometry import Point, Polygon
    
    polygon = Polygon([(0,0), (10,0), (10,10), (0,10)])
    point = Point(5, 5)
    result = polygon.contains(point)

TODO:
    [ ] Install shapely: pip install shapely
    [ ] Implement point_in_polygon()
    [ ] Handle edge cases (point on edge)
    [ ] Add visualization helper
"""
