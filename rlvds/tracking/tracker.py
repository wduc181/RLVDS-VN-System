"""SORT-style multi-object tracker with Kalman filter and Hungarian matching."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rlvds.core.base import BaseTracker, Detection, Track
from rlvds.tracking.bbox_matcher import compute_iou
from rlvds.tracking.track_state import KalmanBoxTracker, TrackState
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class ObjectTracker(BaseTracker):
    """SORT (Simple Online Realtime Tracking) implementation.

    Uses Kalman filters for motion prediction and Hungarian algorithm
    for data association based on IOU.

    Args:
        max_age: Max frames to keep a track without detection match.
        min_hits: Min consecutive detections before a track is confirmed.
        iou_threshold: Minimum IOU for a detection-track match.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._tracks: List[KalmanBoxTracker] = []
        self._next_id = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections, return active tracks.

        Args:
            detections: Detections from the current frame.

        Returns:
            List of confirmed Track objects.
        """
        # Predict new locations of existing tracks
        for track in self._tracks:
            track.predict()

        # Associate detections to tracks via Hungarian matching
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)

        # Update matched tracks
        for track_idx, det_idx in matched:
            det = detections[det_idx]
            self._tracks[track_idx].update(det.bbox)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._tracks.append(
                KalmanBoxTracker(
                    bbox=det.bbox,
                    track_id=self._next_id,
                )
            )
            self._next_id += 1

        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].mark_lost()

        # Delete dead tracks
        self._tracks = [
            t for t in self._tracks
            if not (t.time_since_update > self._max_age)
        ]

        confirmed = [
            self._to_track(t)
            for t in self._tracks
            if t.state == TrackState.CONFIRMED
        ]
        return confirmed

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
        logger.debug("Tracker reset")

    def get_active_tracks(self) -> List[Track]:
        return [
            self._to_track(t)
            for t in self._tracks
            if t.state in (TrackState.TENTATIVE, TrackState.CONFIRMED)
        ]

    def _associate(
        self,
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to existing tracks using IOU + Hungarian.

        Returns:
            (matched_pairs, unmatched_det_indices, unmatched_track_indices)
        """
        if not self._tracks:
            return [], list(range(len(detections))), []

        if not detections:
            return [], [], list(range(len(self._tracks)))

        # Build IOU cost matrix
        iou_matrix = np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(self._tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(track.bbox, det.bbox)

        # Hungarian algorithm (minimize cost = 1 - IOU)
        cost_matrix = 1.0 - iou_matrix

        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            logger.warning("scipy not available — using greedy matching")
            row_indices, col_indices = self._greedy_match(iou_matrix)

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self._tracks)))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self._iou_threshold:
                matched.append((int(r), int(c)))
                unmatched_dets.remove(int(c))
                unmatched_tracks.remove(int(r))

        return matched, unmatched_dets, unmatched_tracks

    @staticmethod
    def _greedy_match(
        iou_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback greedy matching when scipy is unavailable."""
        rows, cols = [], []
        used_cols = set()
        for r in range(iou_matrix.shape[0]):
            best_c = -1
            best_iou = 0.0
            for c in range(iou_matrix.shape[1]):
                if c not in used_cols and iou_matrix[r, c] > best_iou:
                    best_iou = iou_matrix[r, c]
                    best_c = c
            if best_c >= 0:
                rows.append(r)
                cols.append(best_c)
                used_cols.add(best_c)
        return np.array(rows), np.array(cols)

    @staticmethod
    def _to_track(kt: KalmanBoxTracker) -> Track:
        return Track(
            track_id=kt.track_id,
            bbox=kt.bbox,
            age=kt.age,
            hits=kt.hits,
            time_since_update=kt.time_since_update,
            state=kt.state.value,
            velocity=kt.velocity,
            history=list(kt.history),
        )
