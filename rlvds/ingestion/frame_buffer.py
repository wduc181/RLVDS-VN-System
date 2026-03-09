"""
Frame Buffer
=============

Mục đích:
    Buffer frames để xử lý async hoặc skip frames tăng performance.

Lưu ý:
    Sample code (.github/sample/camera.py) KHÔNG dùng frame buffer.
    Module này là optional — implement nếu cần tối ưu performance.

Tham chiếu:
    - config/settings.py → VideoConfig.buffer_size (default=10)
    - config/settings.py → VideoConfig.fps (0 = không giới hạn)
"""

from __future__ import annotations

import collections
import threading
from typing import Generator, Optional

import numpy as np
from rlvds.core.base import BaseVideoSource


class FrameBuffer:
    """Thread-safe frame buffer using deque."""

    def __init__(self, max_size: int = 10) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._buffer: collections.deque[np.ndarray] = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()

    def put(self, frame: np.ndarray) -> None:
        """Thêm frame vào buffer."""
        with self._lock:
            self._buffer.append(frame)

    def get(self) -> Optional[np.ndarray]:
        """Lấy frame mới nhất (bên phải deque)."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer.pop()

    def is_full(self) -> bool:
        """Kiểm tra buffer đã đầy chưa."""
        with self._lock:
            return len(self._buffer) >= self.max_size

    def clear(self) -> None:
        """Xóa toàn bộ buffer."""
        with self._lock:
            self._buffer.clear()

    @staticmethod
    def skip_frames(source: BaseVideoSource, skip: int) -> Generator[np.ndarray, None, None]:
        """Chỉ yield mỗi `skip` frame để giảm tải.
        
        Args:
            source: Nguồn video (kế thừa BaseVideoSource)
            skip: Số frame bỏ qua (ví dụ: 3 nghĩa là xử lý frame 0, bỏ qua 1, 2)
            
        Yields:
            Các frame thỏa mãn điều kiện skip.
        """
        if skip < 1:
            raise ValueError("skip must be >= 1")
            
        for i, frame in enumerate(source):
            if i % skip == 0:
                yield frame
