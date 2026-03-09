"""
Visualization Utilities
=======================

Mục đích:
    Vẽ bounding boxes, text, polygon zones, và annotations lên frame
    để hiển thị kết quả detection/violation trên video output.

Tham chiếu sample code:
    - .github/sample/utils/helper.py::draw_text (dòng 20-26)
    - .github/sample/utils/helper.py::set_hd_resolution (dòng 9-18)
    - .github/sample/camera.py (dòng 59, 70, 77, 122, 124) — drawing calls

Thư viện sử dụng:
    - opencv-python (cv2): Drawing functions
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color Scheme (BGR — OpenCV dùng BGR, không phải RGB)
# ---------------------------------------------------------------------------
COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
COLOR_YELLOW: Tuple[int, int, int] = (0, 255, 255)
COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)

_LIGHT_COLOR_MAP = {
    "RED": COLOR_RED,
    "GREEN": COLOR_GREEN,
    "YELLOW": COLOR_YELLOW,
}


def draw_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int] = (0, 0),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1,
    font_thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Vẽ text lên frame (in-place).

    Args:
        img: Frame ảnh gốc sẽ bị thay đổi trực tiếp.
        text: Chuỗi nội dung cần vẽ.
        pos: Toạ độ ``(x, y)`` — góc trái dưới dòng text.
        font: OpenCV font constant.
        font_scale: Hệ số co giãn font.
        font_thickness: Độ dày nét chữ.
        text_color: Màu chữ BGR.
    """
    cv2.putText(
        img, text, pos, font, font_scale,
        text_color, font_thickness, cv2.LINE_AA,
    )


def set_hd_resolution(image: np.ndarray, width: int = 1280) -> np.ndarray:
    """Resize ảnh giữ nguyên tỷ lệ theo chiều rộng mong muốn.

    Args:
        image: Frame ảnh gốc ``(H, W, C)``.
        width: Chiều rộng đích (pixel).

    Returns:
        Ảnh đã resize.
    """
    height_orig, width_orig = image.shape[:2]
    ratio = height_orig / width_orig
    return cv2.resize(image, (width, int(width * ratio)))


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = COLOR_RED,
    thickness: int = 1,
    label: Optional[str] = None,
) -> np.ndarray:
    """Vẽ bounding box lên frame, kèm label nếu có.

    Args:
        frame: Frame ảnh gốc.
        bbox: ``(x1, y1, x2, y2)`` bounding box.
        color: Màu viền BGR.
        thickness: Độ dày viền.
        label: Nhãn hiển thị phía trên bbox (tuỳ chọn).

    Returns:
        Frame đã vẽ.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=thickness)
    if label:
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return frame


def draw_fps(frame: np.ndarray, fps: int) -> np.ndarray:
    """Vẽ chỉ số FPS lên góc trên bên trái frame.

    Args:
        frame: Frame ảnh gốc.
        fps: Giá trị FPS hiện tại.

    Returns:
        Frame đã vẽ.
    """
    cv2.putText(
        frame, str(fps), (7, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA,
    )
    return frame


def draw_light_status(
    frame: np.ndarray,
    state: str,
    position: Tuple[int, int] = (50, 150),
) -> np.ndarray:
    """Vẽ trạng thái đèn giao thông lên frame.

    Args:
        frame: Frame ảnh gốc.
        state: Trạng thái đèn — ``'RED'``, ``'GREEN'``, hoặc ``'YELLOW'``.
        position: Toạ độ ``(x, y)`` tâm hình tròn đèn.

    Returns:
        Frame đã vẽ.
    """
    color = _LIGHT_COLOR_MAP.get(state.upper(), COLOR_RED)
    cv2.circle(frame, position, 30, color, -1)
    cv2.putText(
        frame, state.upper(), (position[0] + 40, position[1] + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA,
    )
    return frame


def draw_violation_alert(
    frame: np.ndarray,
    text: str = "VIOLATION",
    color: Tuple[int, int, int] = COLOR_RED,
) -> np.ndarray:
    """Vẽ cảnh báo vi phạm nổi bật lên phần trên frame.

    Hiển thị banner bán trong suốt kèm text cảnh báo.

    Args:
        frame: Frame ảnh gốc.
        text: Nội dung cảnh báo.
        color: Màu text và viền BGR.

    Returns:
        Frame đã vẽ.
    """
    h, w = frame.shape[:2]
    # Banner bán trong suốt phía trên
    overlay = frame.copy()
    banner_h = 60
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    # Text cảnh báo canh giữa
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (banner_h + th) // 2
    cv2.putText(
        frame, text, (tx, ty),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return frame
