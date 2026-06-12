"""License plate speed estimation from bbox movement."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Sequence

from rlvds.core.base import Detection
from rlvds.tracking.bbox_matcher import compute_iou


@dataclass
class SpeedEstimate:
    """Speed estimate aligned with one detection."""

    track_id: int | None
    speed_kmh: float | None
    is_speeding: bool
    samples: int = 0


@dataclass
class _SpeedTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    anchor: tuple[float, float]
    last_frame_idx: int
    samples: int = 1
    speed_samples: list[float] = field(default_factory=list)


class LicensePlateSpeedEstimator:
    """Estimate plate speed from anchor displacement between detections.

    The estimator expects a fixed-camera scene and a manual
    ``meters_per_pixel`` calibration for the monitored road area.
    """

    def __init__(
        self,
        *,
        fps: float,
        meters_per_pixel: float,
        speed_limit_kmh: float,
        min_track_frames: int = 5,
        smoothing_window: int = 5,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        anchor: str = "bottom_center",
        enabled: bool = True,
    ) -> None:
        if meters_per_pixel <= 0:
            raise ValueError("meters_per_pixel must be > 0")
        if speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh must be > 0")
        if min_track_frames < 2:
            raise ValueError("min_track_frames must be >= 2")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_age < 1:
            raise ValueError("max_age must be >= 1")
        if anchor not in {"bottom_center", "center"}:
            raise ValueError("anchor must be 'bottom_center' or 'center'")

        self._fps = float(fps)
        self._meters_per_pixel = float(meters_per_pixel)
        self._speed_limit_kmh = float(speed_limit_kmh)
        self._min_track_frames = min_track_frames
        self._smoothing_window = smoothing_window
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._anchor = anchor
        self._enabled = enabled
        self._tracks: list[_SpeedTrack] = []
        self._next_id = 0
        self._frame_idx = 0

    def estimate(
        self,
        detections: Sequence[Detection],
        *,
        frame_idx: int | None = None,
        fps: float | None = None,
    ) -> list[SpeedEstimate]:
        """Estimate speed for each detection in the same order."""
        if not self._enabled:
            return [
                SpeedEstimate(track_id=None, speed_kmh=None, is_speeding=False)
                for _ in detections
            ]

        current_frame_idx = self._resolve_frame_idx(frame_idx)
        current_fps = float(self._fps if fps is None else fps)

        if not detections:
            self._cleanup(current_frame_idx)
            return []

        matched_track_indices: set[int] = set()
        estimates: list[SpeedEstimate] = []

        for det in detections:
            track_idx = self._match_track(det, matched_track_indices)
            if track_idx is None:
                track = self._create_track(det, current_frame_idx)
                estimates.append(
                    SpeedEstimate(
                        track_id=track.track_id,
                        speed_kmh=None,
                        is_speeding=False,
                        samples=track.samples,
                    )
                )
                matched_track_indices.add(len(self._tracks) - 1)
                continue

            matched_track_indices.add(track_idx)
            track = self._tracks[track_idx]
            speed_kmh = self._update_track(track, det, current_frame_idx, current_fps)
            estimates.append(
                SpeedEstimate(
                    track_id=track.track_id,
                    speed_kmh=speed_kmh,
                    is_speeding=(
                        speed_kmh is not None and speed_kmh > self._speed_limit_kmh
                    ),
                    samples=track.samples,
                )
            )

        self._cleanup(current_frame_idx)
        return estimates

    def reset(self) -> None:
        """Drop all active tracks."""
        self._tracks.clear()
        self._next_id = 0
        self._frame_idx = 0

    def _resolve_frame_idx(self, frame_idx: int | None) -> int:
        if frame_idx is None:
            self._frame_idx += 1
            return self._frame_idx
        self._frame_idx = max(self._frame_idx, int(frame_idx))
        return int(frame_idx)

    def _match_track(
        self,
        det: Detection,
        matched_track_indices: set[int],
    ) -> int | None:
        best_idx: int | None = None
        best_iou = 0.0
        for idx, track in enumerate(self._tracks):
            if idx in matched_track_indices:
                continue
            iou = compute_iou(track.bbox, det.bbox)
            if iou >= self._iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = idx
        return best_idx

    def _create_track(self, det: Detection, frame_idx: int) -> _SpeedTrack:
        track = _SpeedTrack(
            track_id=self._next_id,
            bbox=det.bbox,
            anchor=self._anchor_point(det),
            last_frame_idx=frame_idx,
        )
        self._tracks.append(track)
        self._next_id += 1
        return track

    def _update_track(
        self,
        track: _SpeedTrack,
        det: Detection,
        frame_idx: int,
        fps: float,
    ) -> float | None:
        current_anchor = self._anchor_point(det)
        frame_delta = max(1, frame_idx - track.last_frame_idx)
        speed_kmh: float | None = None

        if fps > 0:
            px_distance = hypot(
                current_anchor[0] - track.anchor[0],
                current_anchor[1] - track.anchor[1],
            )
            px_per_second = px_distance * fps / frame_delta
            raw_speed_kmh = px_per_second * self._meters_per_pixel * 3.6
            track.speed_samples.append(raw_speed_kmh)
            track.speed_samples = track.speed_samples[-self._smoothing_window:]

        track.bbox = det.bbox
        track.anchor = current_anchor
        track.last_frame_idx = frame_idx
        track.samples += 1

        if track.samples >= self._min_track_frames and track.speed_samples:
            speed_kmh = sum(track.speed_samples) / len(track.speed_samples)
        return speed_kmh

    def _cleanup(self, frame_idx: int) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if frame_idx - track.last_frame_idx <= self._max_age
        ]

    def _anchor_point(self, det: Detection) -> tuple[float, float]:
        if self._anchor == "center":
            return det.center()
        return det.get_anchor_point()
