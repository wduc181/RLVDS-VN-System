"""Video source wrapper for file/webcam/IP camera using OpenCV."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

Source = Union[str, int]


class VideoSource:
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
    ) -> None:
        if max_read_failures < 1:
            raise ValueError("max_read_failures must be >= 1")
        if reconnect_interval_sec < 0:
            raise ValueError("reconnect_interval_sec must be >= 0")

        self.source: Source = source
        self._resolved_source: Source = self._normalize_source(source)
        self._is_stream: bool = self._detect_stream(self._resolved_source)
        self._max_read_failures = max_read_failures
        self._reconnect_interval_sec = reconnect_interval_sec

        self.cap: cv2.VideoCapture = cv2.VideoCapture(self._resolved_source)
        if not self.cap.isOpened():
            self.release()
            raise RuntimeError(f"Cannot open video source: {source!r}")

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.iter_frames()

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames continuously with basic failure tolerance."""
        failures = 0

        while True:
            ok, frame = self.read_frame()
            if ok and frame is not None:
                failures = 0
                yield frame
                continue

            failures += 1

            # File input: stop at EOS immediately.
            if not self._is_stream:
                break

            # Stream input: try reconnect for transient glitches.
            if failures > self._max_read_failures:
                break

            if not self.is_opened():
                self._safe_reopen()
            else:
                time.sleep(self._reconnect_interval_sec)

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read one frame from capture."""
        if not self.is_opened():
            return False, None

        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def is_opened(self) -> bool:
        """Return True if capture handle is opened."""
        return hasattr(self, "cap") and self.cap is not None and self.cap.isOpened()

    def reopen(self) -> None:
        """Recreate capture from original source."""
        self.release()
        self.cap = cv2.VideoCapture(self._resolved_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot reopen video source: {self.source!r}")

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

    def release(self) -> None:
        """Release capture resources. Safe to call multiple times."""
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def _safe_reopen(self) -> None:
        """Best-effort reopen used inside frame iteration."""
        try:
            self.reopen()
        except RuntimeError:
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
        return source.lower().startswith(cls._STREAM_PREFIXES)
