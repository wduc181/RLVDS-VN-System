"""
Video Source — OpenCV Video Capture Wrapper
============================================

Mục đích:
    Wrapper thống nhất cho ``cv2.VideoCapture``, hỗ trợ đọc frame từ:
    * Video file trên ổ đĩa
    * Webcam (device index)
    * Network / IP camera (RTSP/HTTP/RTMP/UDP)

Tính năng:
    - Tự động nhận diện loại source (file vs. stream)
    - Context-manager support (``with VideoSource(...) as src: ...``)
    - Frame iteration với tolerance cho read failures và auto-reconnect
    - Helper methods: FPS, frame size, frame count

Thư viện sử dụng:
    - opencv-python (cv2): Video capture & processing
    - numpy: Array operations (frame data)

Sử dụng::

    from rlvds.ingestion.video_source import VideoSource

    with VideoSource("video.mp4") as src:
        for frame in src:
            ...  # process frame
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np
from typing_extensions import override

from rlvds.core.base import BaseVideoSource
from rlvds.utils.logger import get_logger

Source = Union[str, int]

logger = get_logger(__name__)


class VideoSource(BaseVideoSource):
    """Wrapper around ``cv2.VideoCapture`` with robust frame iteration."""

    _STREAM_PREFIXES = (
        "rtsp://",
        "http://",
        "https://",
        "rtmp://",
        "udp://",
    )

    def __init__(
        self,
        source: Source,
        *,
        max_read_failures: int = 20,
        reconnect_interval_sec: float = 0.5,
        max_reconnect_attempts: int = 10,
    ) -> None:
        if max_read_failures < 1:
            raise ValueError("max_read_failures must be >= 1")
        if reconnect_interval_sec < 0:
            raise ValueError("reconnect_interval_sec must be >= 0")
        if max_reconnect_attempts < 1:
            raise ValueError("max_reconnect_attempts must be >= 1")

        self.source: Source = source
        try:
            self._resolved_source: Source = self._normalize_source(source)
        except FileNotFoundError:
            logger.error("Video source path does not exist: %r", source)
            raise
        self._is_stream: bool = self._detect_stream(self._resolved_source)
        self._max_read_failures = max_read_failures
        self._reconnect_interval_sec = reconnect_interval_sec
        self._max_reconnect_attempts = max_reconnect_attempts

        self.cap: cv2.VideoCapture = cv2.VideoCapture(self._resolved_source)
        if not self.cap.isOpened():
            self.release()
            logger.error("Failed to open video source: %r", source)
            raise RuntimeError(f"Cannot open video source: {source!r}")
        logger.info("Opened video source: %r (Stream: %s)", source, self._is_stream)

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames continuously with basic failure tolerance."""
        failures = 0
        reconnect_attempts = 0

        while True:
            ok, frame = self.read_frame()
            if ok and frame is not None:
                failures = 0
                reconnect_attempts = 0
                yield frame
                continue

            failures += 1

            # File input: stop at EOS immediately.
            if not self._is_stream:
                break

            # Stream input: force reopen after hitting max consecutive failures,
            # even if isOpened() still returns True (common with RTSP/HTTP glitches).
            if failures >= self._max_read_failures:
                reconnect_attempts += 1
                if reconnect_attempts > self._max_reconnect_attempts:
                    logger.error(
                        "Stream permanently lost for source %r after %d reconnect attempts",
                        self.source,
                        reconnect_attempts - 1,
                    )
                    break
                self._safe_reopen()
                if not self.is_opened():
                    break
                failures = 0
                continue

            if not self.is_opened():
                reconnect_attempts += 1
                if reconnect_attempts > self._max_reconnect_attempts:
                    logger.error(
                        "Stream permanently lost for source %r after %d reconnect attempts",
                        self.source,
                        reconnect_attempts - 1,
                    )
                    break
                self._safe_reopen()
                if not self.is_opened():
                    continue
            else:
                time.sleep(self._reconnect_interval_sec)

    @override
    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame from capture."""
        if not self.is_opened():
            return False, None

        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    @override
    def is_opened(self) -> bool:
        """Return True if capture handle is opened."""
        return hasattr(self, "cap") and self.cap is not None and self.cap.isOpened()

    def reopen(self) -> None:
        """Recreate capture from original source."""
        self.release()
        time.sleep(0.05)  # Brief delay to let OpenCV release the handle fully
        self.cap = cv2.VideoCapture(self._resolved_source)
        if not self.cap.isOpened():
            logger.error("Failed to reopen video source: %r", self.source)
            raise RuntimeError(f"Cannot reopen video source: {self.source!r}")
        logger.info("Successfully reopened video source: %r", self.source)

    def get_fps(self) -> float:
        if not self.is_opened():
            return 0.0
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    def get_frame_size(self) -> tuple[int, int]:
        if not self.is_opened():
            return (0, 0)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def get_frame_count(self) -> int:
        if not self.is_opened():
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @override
    def release(self) -> None:
        """Release capture resources. Safe to call multiple times."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self) -> None:
        """Ensure camera resources are released on garbage collection."""
        self.release()

    def _safe_reopen(self) -> None:
        """Best-effort reopen used inside frame iteration."""
        try:
            self.reopen()
        except RuntimeError as e:
            logger.warning(
                "Reconnection failed for source %r: %s. Retrying in %.1fs",
                self.source,
                e,
                self._reconnect_interval_sec,
            )
            time.sleep(self._reconnect_interval_sec)

    @classmethod
    def _normalize_source(cls, source: Source) -> Source:
        """Normalize source into webcam index, stream URL, or existing file path."""
        if isinstance(source, int):
            return source

        text = source.strip()
        if text.isdigit():
            return int(text)

        if text.lower().startswith(cls._STREAM_PREFIXES):
            return text

        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {text}")
        return str(path)

    @classmethod
    def _detect_stream(cls, source: Source) -> bool:
        """Return True for live stream/webcam sources."""
        if isinstance(source, int):
            return True
        if not isinstance(source, str):
            return False
        return source.lower().startswith(cls._STREAM_PREFIXES)
