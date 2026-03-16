"""Week-3 mini pipeline: Video -> Detect -> OCR -> Violation(mock)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from rlvds.core.base import Detection
from rlvds.ingestion.video_source import VideoSource
from rlvds.temporal.violation import ViolationDetector
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


@dataclass
class MiniPipelineResult:
    plate_text: str
    detection: Detection
    is_violation: bool


class MiniPipeline:
    """Simple frame-by-frame integration used for module-level testing."""

    def __init__(
        self,
        detector: DetectorLike,
        ocr: OCRLike,
        violation_detector: ViolationDetector,
        crop_expand_ratio: float = 0.15,
    ) -> None:
        self._detector = detector
        self._ocr = ocr
        self._violation_detector = violation_detector
        self._crop_expand_ratio = crop_expand_ratio

    def process_frame(self, frame: np.ndarray) -> List[MiniPipelineResult]:
        """Run detect -> crop -> OCR -> mock violation check for a frame."""
        detections = self._detector.detect(frame)
        results: List[MiniPipelineResult] = []

        for det in detections:
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
            for idx, frame in enumerate(video):
                all_results.extend(self.process_frame(frame))
                if max_frames is not None and idx + 1 >= max_frames:
                    break
        logger.info("MiniPipeline processed %d results", len(all_results))
        return all_results
