"""
Cached Pipeline
================

Mục đích:
    Pipeline tích hợp OCR caching để tối ưu FPS.
    YOLO detection chạy mỗi frame, nhưng PaddleOCR chỉ gọi khi
    phát hiện biển số mới (cache miss). Các frame sau reuse kết quả
    OCR đã cache nhờ IOU-based bbox matching.

Thư viện sử dụng:
    - numpy: ndarray cho frame data
    - typing: Type annotations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from rlvds.core.base import Detection
from rlvds.ocr.plate_cache import CachedPlate, PlateTrackCache
from rlvds.temporal.violation import ViolationDetector
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class DetectorLike(Protocol):
    """Protocol cho detector module."""

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
    """Protocol cho OCR module."""

    def recognize(self, image: np.ndarray) -> str:
        ...


@dataclass
class CachedPipelineResult:
    """Kết quả xử lý frame từ CachedPipeline.

    Attributes:
        plate_text: Biển số đã nhận diện.
        detection: Detection gốc từ YOLO.
        is_violation: Cờ vi phạm.
        from_cache: True nếu plate_text lấy từ cache (skip OCR).
    """

    plate_text: str
    detection: Detection
    is_violation: bool
    from_cache: bool = False


class CachedPipeline:
    """Pipeline có OCR caching để tối ưu FPS.

    Flow mỗi frame:
        1. YOLO detect(frame) → list[Detection]          (luôn chạy, ~10-30ms)
        2. Với mỗi detection:
           a. cache.match(bbox) → hit?
              - HIT:  reuse cached plate_text              (~0ms)
              - MISS: crop → preprocess → OCR → cache.add  (~100-200ms)
        3. Violation check
        4. cache.cleanup()

    Args:
        detector: YOLO detector instance.
        ocr: PaddleOCR engine instance.
        violation_detector: Violation logic instance.
        cache: PlateTrackCache instance.
        crop_expand_ratio: Tỷ lệ mở rộng bbox khi crop.
        ocr_quality_frames: Số lần OCR tối đa cho cùng plate.
    """

    def __init__(
        self,
        detector: DetectorLike,
        ocr: OCRLike,
        violation_detector: ViolationDetector,
        cache: PlateTrackCache,
        crop_expand_ratio: float = 0.15,
        ocr_quality_frames: int = 3,
    ) -> None:
        self._detector = detector
        self._ocr = ocr
        self._violation_detector = violation_detector
        self._cache = cache
        self._crop_expand_ratio = crop_expand_ratio
        self._ocr_quality_frames = ocr_quality_frames
        self._frame_idx: int = 0

    def process_frame(self, frame: np.ndarray) -> List[CachedPipelineResult]:
        """Xử lý một frame với OCR caching.

        YOLO detection luôn chạy. OCR chỉ gọi khi cache miss
        hoặc chưa đủ ocr_quality_frames.

        Args:
            frame: BGR image từ OpenCV ``(H, W, C)``.

        Returns:
            Danh sách kết quả cho từng detection trong frame.
        """
        self._frame_idx += 1
        detections = self._detector.detect(frame)
        results: List[CachedPipelineResult] = []

        for det in detections:
            plate_text, from_cache = self._resolve_plate_text(det, frame)

            is_violation = self._violation_detector.check_mock_violation(
                plate_text=plate_text,
                detection=det,
            )

            results.append(
                CachedPipelineResult(
                    plate_text=plate_text,
                    detection=det,
                    is_violation=is_violation,
                    from_cache=from_cache,
                )
            )

        # Dọn dẹp entry hết hạn
        self._cache.cleanup(self._frame_idx)

        return results

    def _resolve_plate_text(
        self,
        det: Detection,
        frame: np.ndarray,
    ) -> tuple[str, bool]:
        """Quyết định dùng cache hay gọi OCR mới.

        Args:
            det: Detection hiện tại.
            frame: Frame gốc để crop nếu cần OCR.

        Returns:
            Tuple ``(plate_text, from_cache)``.
        """
        bbox = det.bbox

        # Tìm trong cache
        cached: Optional[CachedPlate] = self._cache.match(bbox, self._frame_idx)

        if cached is not None:
            # Cache HIT — kiểm tra xem đã đủ quality frames chưa
            if cached.ocr_count < self._ocr_quality_frames:
                # Chạy thêm OCR để cải thiện confidence
                plate_text = self._run_ocr(det, frame)
                if plate_text != "unknown":
                    self._cache.add_or_update(
                        bbox=bbox,
                        plate_text=plate_text,
                        confidence=det.confidence,
                        frame_idx=self._frame_idx,
                    )
                return cached.plate_text, True

            # Đã đủ OCR quality → reuse hoàn toàn
            return cached.plate_text, True

        # Cache MISS — chạy OCR
        plate_text = self._run_ocr(det, frame)
        if plate_text != "unknown":
            self._cache.add_or_update(
                bbox=bbox,
                plate_text=plate_text,
                confidence=det.confidence,
                frame_idx=self._frame_idx,
            )

        return plate_text, False

    def _run_ocr(self, det: Detection, frame: np.ndarray) -> str:
        """Crop và chạy OCR cho một detection.

        Args:
            det: Detection cần OCR.
            frame: Frame gốc.

        Returns:
            Chuỗi biển số hoặc ``"unknown"``.
        """
        crop = self._detector.crop_plate(
            det,
            frame,
            expand_ratio=self._crop_expand_ratio,
        )
        return self._ocr.recognize(crop)

    @property
    def cache(self) -> PlateTrackCache:
        """Trả về cache instance để truy cập stats."""
        return self._cache

    @property
    def frame_idx(self) -> int:
        """Frame index hiện tại."""
        return self._frame_idx

    def reset(self) -> None:
        """Reset pipeline state (cache + frame counter)."""
        self._cache.clear()
        self._frame_idx = 0
        logger.info("CachedPipeline reset")
