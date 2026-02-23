"""
Point-in-Polygon & Polygon Masking
====================================

Mục đích:
    1. Tạo mask polygon để giới hạn vùng detect (chỉ detect trong vùng vi phạm)
    2. Kiểm tra xem một điểm có nằm trong polygon hay không

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 40-41) — định nghĩa polygon points
    - .github/sample/camera.py (dòng 56-59) — tạo mask + apply

Thư viện sử dụng:
    - opencv-python (cv2): cv2.fillPoly, cv2.bitwise_and, cv2.polylines, cv2.pointPolygonTest
    - numpy: Array operations

===========================================================================
POLYGON MASKING (từ camera.py dòng 40-41, 56-59)
===========================================================================

Hàm cần implement:

1. create_polygon(vertices: list[list[int]]) -> np.ndarray
   Chuyển đổi danh sách tọa độ thành numpy array cho OpenCV:
     points = np.array(vertices, np.int32)
     points = points.reshape((-1, 1, 2))
     return points

   Tham khảo camera.py dòng 40-41:
     points = np.array([[1000, 700], [1700, 700], [1900,1078], [800, 1078]], np.int32)
     points = points.reshape((-1, 1, 2))

2. create_mask(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray
   Tạo mask đen, tô trắng vùng polygon, apply lên frame:
     mask = np.zeros_like(frame)
     cv2.fillPoly(mask, [polygon], (255, 255, 255))
     masked_image = cv2.bitwise_and(frame, mask)
     return masked_image

   → Frame output chỉ giữ lại vùng polygon, phần ngoài đen hoàn toàn
   → Đưa masked_image vào YOLO detect để chỉ phát hiện trong vùng

3. draw_polygon(frame: np.ndarray, polygon: np.ndarray,
                color: tuple = (0, 215, 255), thickness: int = 2) -> np.ndarray
   Vẽ viền polygon lên frame để hiển thị:
     cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)
     return frame

===========================================================================
POINT-IN-POLYGON CHECK
===========================================================================

4. point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool
   Kiểm tra điểm (x, y) có nằm trong polygon:
     result = cv2.pointPolygonTest(polygon, point, measureDist=False)
     return result >= 0  # 1 = inside, 0 = on edge, -1 = outside

   Dùng khi cần kiểm tra: tọa độ center/anchor của detection có trong vùng vi phạm?

5. point_distance_to_polygon(point: tuple, polygon: np.ndarray) -> float
   Tính khoảng cách từ điểm đến cạnh polygon:
     distance = cv2.pointPolygonTest(polygon, point, measureDist=True)
     return distance
   
   Giá trị dương = bên trong, âm = bên ngoài

TODO:
    [ ] Import cv2, numpy
    [ ] Implement create_polygon()
    [ ] Implement create_mask() — tham khảo camera.py dòng 56-58
    [ ] Implement draw_polygon() — tham khảo camera.py dòng 59
    [ ] Implement point_in_polygon() — dùng cv2.pointPolygonTest
    [ ] Implement point_distance_to_polygon()
    [ ] Test: tạo polygon hình chữ nhật → kiểm tra point inside/outside
"""
