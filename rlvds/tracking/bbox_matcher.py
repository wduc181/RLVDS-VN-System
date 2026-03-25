"""
Bounding Box Matching Utilities
================================

Mục đích:
    Cung cấp hàm tính IOU (Intersection over Union) giữa bounding boxes.
    Là building block cho OCR cache matching và tracking module.

Thư viện sử dụng:
    - typing: Type annotations
"""

from __future__ import annotations

from typing import Tuple


def compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """Tính Intersection over Union giữa hai bounding boxes.

    Args:
        box_a: Bounding box thứ nhất ``(x1, y1, x2, y2)``.
        box_b: Bounding box thứ hai ``(x1, y1, x2, y2)``.

    Returns:
        Giá trị IOU trong khoảng ``[0.0, 1.0]``.
        Trả về ``0.0`` nếu không có phần giao.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Tọa độ intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    # Diện tích từng box
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area
