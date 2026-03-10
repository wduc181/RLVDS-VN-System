"""
Timing Utilities
================

Mục đích:
    Đồng bộ thời gian giữa video và system timer.

Thư viện sử dụng:
    - time: System time
    - datetime: Timestamp formatting
"""

from __future__ import annotations

from datetime import datetime


def frame_to_time(frame_num: int, fps: float) -> float:
    """Chuyển frame number sang thời gian video (giây).

    Args:
        frame_num: Số thứ tự frame (0-based).
        fps: Tốc độ khung hình của video (frames per second).

    Returns:
        Thời gian tính bằng giây tại frame đó.

    Raises:
        ValueError: Nếu ``fps <= 0``.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return frame_num / fps


def get_current_timestamp() -> str:
    """Lấy timestamp hiện tại theo format ``dd/mm/YYYY HH:MM:SS``.

    Format này khớp với camera.py sample:
        ``datetime.now().strftime("%d/%m/%Y %H:%M:%S")``

    Returns:
        Chuỗi timestamp đã format.
    """
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def calculate_video_offset(video_start: float) -> float:
    """Tính offset giữa video time và real time hiện tại.

    Dùng để đồng bộ timestamp khi video không phải live stream.

    Args:
        video_start: Epoch time (giây) tại thời điểm bắt đầu phát video.

    Returns:
        Số giây đã trôi kể từ ``video_start``.
    """
    import time

    return time.time() - video_start
