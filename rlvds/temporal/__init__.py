"""
RLVDS Temporal Package
======================

Xử lý logic thời gian: đèn giao thông, violation detection.

Modules:
    - traffic_light.py: Traffic light state machine
    - timing.py: Timing synchronization
    - violation.py: Violation detection logic
"""

from rlvds.temporal.timing import (
    calculate_video_offset,
    frame_to_time,
    get_current_timestamp,
)
from rlvds.temporal.traffic_light import LightState, TrafficLightFSM
from rlvds.temporal.violation import ViolationDetector

__all__ = [
    "LightState",
    "TrafficLightFSM",
    "ViolationDetector",
    "frame_to_time",
    "get_current_timestamp",
    "calculate_video_offset",
]
