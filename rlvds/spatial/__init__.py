"""
RLVDS Spatial Package
=====================

Xử lý logic không gian: point-in-polygon, zone definitions.

Modules:
    - polygon.py: Point-in-polygon algorithms
    - zones.py: Violation zone definitions
    - calibration.py: Camera calibration (optional)
"""

from rlvds.spatial.polygon import (
    create_mask,
    create_polygon,
    draw_polygon,
    point_distance_to_polygon,
    point_in_polygon,
)
from rlvds.spatial.zones import ViolationZone

__all__ = [
    "create_polygon",
    "create_mask",
    "draw_polygon",
    "point_in_polygon",
    "point_distance_to_polygon",
    "ViolationZone",
]
