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
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


def create_polygon(vertices: List[List[int]]) -> np.ndarray:
    """Chuyển danh sách tọa độ thành numpy array cho OpenCV.

    Tham khảo camera.py dòng 40-41::

        points = np.array([[1000, 700], ...], np.int32)
        points = points.reshape((-1, 1, 2))

    Args:
        vertices: Danh sách đỉnh ``[[x1, y1], [x2, y2], ...]``.

    Returns:
        Numpy array shape ``(N, 1, 2)`` kiểu ``int32`` cho OpenCV.

    Raises:
        ValueError: Nếu polygon có ít hơn 3 đỉnh.
    """
    if len(vertices) < 3:
        if len(vertices) == 0:
            logger.warning(
                "Polygon rỗng (0 đỉnh) — trả về dummy polygon. "
                "Cập nhật config spatial.violation_zone trước khi sử dụng."
            )
            # Dummy polygon tại gốc — is_in_zone luôn False
            dummy = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.int32)
            return dummy.reshape((-1, 1, 2))
        raise ValueError(
            f"Polygon cần ít nhất 3 đỉnh, nhận {len(vertices)}"
        )
    points = np.array(vertices, dtype=np.int32)
    return points.reshape((-1, 1, 2))


def create_mask(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Tạo masked frame — chỉ giữ vùng bên trong polygon.

    Tham khảo camera.py dòng 56-58::

        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        masked_image = cv2.bitwise_and(frame, mask)

    Args:
        frame: Frame ảnh gốc ``(H, W, C)``.
        polygon: Polygon array từ ``create_polygon()``.

    Returns:
        Frame mới với vùng ngoài polygon bị tô đen.
    """
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def draw_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    color: Tuple[int, int, int] = (0, 215, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Vẽ viền polygon lên frame để hiển thị.

    Tham khảo camera.py dòng 59::

        cv2.polylines(frame, [points], isClosed=True, color=..., thickness=2)

    Args:
        frame: Frame ảnh gốc (sẽ bị thay đổi in-place).
        polygon: Polygon array từ ``create_polygon()``.
        color: Màu viền BGR.
        thickness: Độ dày viền (pixel).

    Returns:
        Frame đã vẽ viền polygon.
    """
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)
    return frame


def point_in_polygon(
    point: Tuple[float, float],
    polygon: np.ndarray,
) -> bool:
    """Kiểm tra điểm ``(x, y)`` có nằm trong polygon hay không.

    Sử dụng ``cv2.pointPolygonTest`` với ``measureDist=False``:
        - ``+1`` = bên trong
        - ``0``  = trên cạnh
        - ``-1`` = bên ngoài

    Args:
        point: Tọa độ ``(x, y)`` cần kiểm tra.
        polygon: Polygon array từ ``create_polygon()``.

    Returns:
        ``True`` nếu điểm nằm trong hoặc trên cạnh polygon.
    """
    result = cv2.pointPolygonTest(polygon, point, measureDist=False)
    return result >= 0


def point_distance_to_polygon(
    point: Tuple[float, float],
    polygon: np.ndarray,
) -> float:
    """Tính khoảng cách có dấu từ điểm đến cạnh polygon.

    Args:
        point: Tọa độ ``(x, y)`` cần kiểm tra.
        polygon: Polygon array từ ``create_polygon()``.

    Returns:
        Giá trị dương nếu bên trong, âm nếu bên ngoài, ``0`` nếu trên cạnh.
    """
    return cv2.pointPolygonTest(polygon, point, measureDist=True)
