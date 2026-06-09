"""Week-3 mini pipeline: Video -> Detect -> OCR -> Violation(mock)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence

import numpy as np

from rlvds.core.base import Detection
from rlvds.ingestion.video_source import VideoSource
from rlvds.temporal.violation import ViolationDetector
from rlvds.tracking.speed_estimator import SpeedEstimate
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class DetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...

    def crop_plate(
        self,
        detection: Detection,
        frame: np.ndarray,
        expand_ratio: float = 0.15,
    ) -> np.ndarray:
        ...


class OCRLike(Protocol):
    def recognize(self, image: np.ndarray) -> str:
        ...


class SpeedEstimatorLike(Protocol):
    def estimate(
        self,
        detections: Sequence[Detection],
        *,
        frame_idx: int | None = None,
        fps: float | None = None,
    ) -> list[SpeedEstimate]:
        ...


@dataclass
class MiniPipelineResult:
    plate_text: str
    detection: Detection
    is_violation: bool
    track_id: int | None = None
    speed_kmh: float | None = None
    is_speeding: bool = False


class MiniPipeline:
    """Simple frame-by-frame integration used for module-level testing."""

    def __init__(
        self,
        detector: DetectorLike,
        ocr: OCRLike,
        violation_detector: ViolationDetector,
        crop_expand_ratio: float = 0.15,
        speed_estimator: SpeedEstimatorLike | None = None,
    ) -> None:
        self._detector = detector
        self._ocr = ocr
        self._violation_detector = violation_detector
        self._crop_expand_ratio = crop_expand_ratio
        self._speed_estimator = speed_estimator
        self._frame_idx = 0

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int | None = None,
        fps: float | None = None,
    ) -> List[MiniPipelineResult]:
        """Run detect -> crop -> OCR -> mock violation check for a frame."""
        current_frame_idx = self._resolve_frame_idx(frame_idx)
        detections = self._detector.detect(frame)
        speed_estimates = self._estimate_speeds(detections, current_frame_idx, fps)
        results: List[MiniPipelineResult] = []

        for det, speed in zip(detections, speed_estimates):
            crop = self._detector.crop_plate(
                det,
                frame,
                expand_ratio=self._crop_expand_ratio,
            )
            plate_text = self._ocr.recognize(crop)
            is_violation = self._violation_detector.check_mock_violation(
                plate_text=plate_text,
                detection=det,
            )
            results.append(
                MiniPipelineResult(
                    plate_text=plate_text,
                    detection=det,
                    is_violation=is_violation,
                    track_id=speed.track_id,
                    speed_kmh=speed.speed_kmh,
                    is_speeding=speed.is_speeding,
                )
            )
        return results

    def run_video(
        self,
        source: str | int,
        max_frames: int | None = None,
    ) -> List[MiniPipelineResult]:
        """Run mini pipeline over video source and collect all frame results."""
        all_results: List[MiniPipelineResult] = []
        with VideoSource(source) as video:
            fps = video.get_fps()
            for idx, frame in enumerate(video):
                all_results.extend(
                    self.process_frame(frame, frame_idx=idx + 1, fps=fps)
                )
                if max_frames is not None and idx + 1 >= max_frames:
                    break
        logger.info("MiniPipeline processed %d results", len(all_results))
        return all_results

    def _resolve_frame_idx(self, frame_idx: int | None) -> int:
        if frame_idx is None:
            self._frame_idx += 1
            return self._frame_idx
        self._frame_idx = max(self._frame_idx, int(frame_idx))
        return int(frame_idx)

    def _estimate_speeds(
        self,
        detections: Sequence[Detection],
        frame_idx: int,
        fps: float | None,
    ) -> list[SpeedEstimate]:
        if self._speed_estimator is None:
            return [
                SpeedEstimate(track_id=None, speed_kmh=None, is_speeding=False)
                for _ in detections
            ]
        return self._speed_estimator.estimate(
            detections,
            frame_idx=frame_idx,
            fps=fps,
        )
