"""Track state and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import numpy as np


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class KalmanBoxTracker:
    """Kalman-filter-based bounding box tracker.

    State vector: [cx, cy, area, aspect_ratio, vx, vy, v_area, v_ar].
    """

    bbox: Tuple[int, int, int, int]
    track_id: int
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: TrackState = TrackState.TENTATIVE
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._kf = self._init_kalman()
        self._update_state_from_bbox(self.bbox)

    @staticmethod
    def _init_kalman() -> "KalmanFilter":
        from filterpy.kalman import KalmanFilter

        kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0

        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ])

        # Measurement function (observe cx, cy, area, ar)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        # Measurement noise
        kf.R[2:, 2:] *= 10.0
        kf.R *= 5.0

        # Process noise
        kf.Q[4:, 4:] *= 0.01
        kf.Q *= 0.01

        # Initial covariance
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0

        return kf

    def _update_state_from_bbox(self, bbox: Tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        self._kf.x[:4] = np.array([
            x1 + w / 2.0,
            y1 + h / 2.0,
            float(w * h),
            w / float(h),
        ]).reshape(4, 1)

    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next bounding box from Kalman state."""
        self._kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox()

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """Update Kalman state with new measurement."""
        self._update_state_from_bbox(bbox)
        self._kf.update(self._kf.x[:4])
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)

        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED

    def _state_to_bbox(self) -> Tuple[int, int, int, int]:
        cx, cy, area = self._kf.x[0, 0], self._kf.x[1, 0], self._kf.x[2, 0]
        ar = self._kf.x[3, 0]
        w = max(1, int(np.sqrt(max(1.0, area) * ar)))
        h = max(1, int(np.sqrt(max(1.0, area) / ar)))
        return (
            int(cx - w / 2.0),
            int(cy - h / 2.0),
            int(cx + w / 2.0),
            int(cy + h / 2.0),
        )

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_deleted(self) -> None:
        self.state = TrackState.DELETED

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self._kf.x[4, 0]), float(self._kf.x[5, 0]))

    @property
    def center(self) -> Tuple[float, float]:
        return (float(self._kf.x[0, 0]), float(self._kf.x[1, 0]))
