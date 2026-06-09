"""
RLVDS Tracking Package
======================

Multi-object tracking để theo dõi biển số qua các frames.

Modules:
    - tracker.py: Main tracker implementation
    - track_state.py: Track lifecycle management
    - speed_estimator.py: License plate speed warning estimates
"""

from rlvds.tracking.speed_estimator import LicensePlateSpeedEstimator, SpeedEstimate

__all__ = ["LicensePlateSpeedEstimator", "SpeedEstimate"]
