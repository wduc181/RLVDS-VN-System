"""
RLVDS Detection Package
=======================

Phát hiện biển số xe sử dụng YOLOv5.

Modules:
    - detector.py: YOLOv5 license plate detector
    - models.py: Detection result dataclasses
"""

from rlvds.core.base import Detection
from rlvds.detection.detector import LicensePlateDetector

__all__ = ["LicensePlateDetector", "Detection"]
