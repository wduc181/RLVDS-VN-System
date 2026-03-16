"""
RLVDS Ingestion Package
=======================

Xử lý video input từ file hoặc camera.

Modules:
    - video_source.py: Đọc video/camera
    - frame_buffer.py: Buffer và preprocessing frames
"""

from rlvds.ingestion.frame_buffer import FrameBuffer
from rlvds.ingestion.video_source import VideoSource

__all__ = ["VideoSource", "FrameBuffer"]
