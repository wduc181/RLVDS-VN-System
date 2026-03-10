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
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rlvds.core.base import BaseSpatialReasoner
from rlvds.spatial.polygon import (
    create_mask,
    create_polygon,
    draw_polygon,
    point_in_polygon,
)
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class ViolationZone(BaseSpatialReasoner):
    """Vùng đa giác giám sát vi phạm tại ngã tư.

    Implement ``BaseSpatialReasoner`` ABC để kết nối với
    ``ViolationDetector`` trong temporal module.

    Args:
        vertices: Danh sách đỉnh ``[[x1, y1], [x2, y2], ...]``.
        zone_id: ID định danh zone.
        color: Màu viền BGR khi vẽ lên frame.
        thickness: Độ dày viền (pixel).
    """

    def __init__(
        self,
        vertices: List[List[int]],
        zone_id: str = "default",
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> None:
        self._vertices = vertices
        self._polygon = create_polygon(vertices)
        self._zone_id = zone_id
        self._color = color
        self._thickness = thickness
        logger.info(
            "ViolationZone '%s' created with %d vertices",
            zone_id, len(vertices),
        )

    # ------------------------------------------------------------------
    # BaseSpatialReasoner interface
    # ------------------------------------------------------------------

    def set_zone(self, polygon: List[Tuple[int, int]]) -> None:
        """Cập nhật vùng đa giác giám sát.

        Args:
            polygon: Danh sách đỉnh ``[(x, y), ...]`` theo thứ tự.
        """
        self._vertices = [list(p) for p in polygon]
        self._polygon = create_polygon(self._vertices)
        logger.info(
            "ViolationZone '%s' updated with %d vertices",
            self._zone_id, len(polygon),
        )

    def is_in_zone(self, point: Tuple[float, float]) -> bool:
        """Kiểm tra điểm có nằm trong zone hay không.

        Args:
            point: Tọa độ ``(x, y)`` cần kiểm tra.

        Returns:
            ``True`` nếu điểm nằm trong hoặc trên cạnh polygon.
        """
        return point_in_polygon(point, self._polygon)

    # ------------------------------------------------------------------
    # Zone-specific API
    # ------------------------------------------------------------------

    def contains(self, point: Tuple[float, float]) -> bool:
        """Alias cho ``is_in_zone`` — kiểm tra point trong zone.

        Args:
            point: Tọa độ ``(x, y)`` cần kiểm tra.

        Returns:
            ``True`` nếu điểm nằm trong zone.
        """
        return self.is_in_zone(point)

    def apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Tạo masked frame — chỉ giữ vùng bên trong zone.

        Tham khảo camera.py dòng 56-58: tạo mask đen, tô trắng zone,
        apply ``bitwise_and`` → phần ngoài zone bị tô đen.

        Args:
            frame: Frame ảnh gốc ``(H, W, C)``.

        Returns:
            Frame mới với vùng ngoài zone bị tô đen.
        """
        return create_mask(frame, self._polygon)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ viền zone lên frame để hiển thị.

        Args:
            frame: Frame ảnh gốc (sẽ bị thay đổi in-place).

        Returns:
            Frame đã vẽ viền zone.
        """
        return draw_polygon(frame, self._polygon, self._color, self._thickness)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def zone_id(self) -> str:
        """ID định danh zone."""
        return self._zone_id

    @property
    def vertices(self) -> List[List[int]]:
        """Danh sách đỉnh polygon."""
        return self._vertices

    @property
    def polygon(self) -> np.ndarray:
        """Numpy polygon array cho OpenCV."""
        return self._polygon
