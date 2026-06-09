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
from typing import List, Optional, Protocol, Sequence

import numpy as np

from rlvds.core.base import Detection
from rlvds.ocr.plate_cache import CachedPlate, PlateTrackCache
from rlvds.ocr.recognizer import OCRResult
from rlvds.temporal.violation import ViolationDetector
from rlvds.tracking.speed_estimator import SpeedEstimate
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

    def recognize_with_confidence(self, image: np.ndarray) -> OCRResult:
        ...


class SpeedEstimatorLike(Protocol):
    """Protocol cho module ước lượng tốc độ."""

    def estimate(
        self,
        detections: Sequence[Detection],
        *,
        frame_idx: int | None = None,
        fps: float | None = None,
    ) -> list[SpeedEstimate]:
        ...


@dataclass
class CachedPipelineResult:
    """Kết quả xử lý frame từ CachedPipeline.

    Attributes:
        plate_text: Biển số đã nhận diện.
        detection: Detection gốc từ YOLO.
        is_violation: Cờ vi phạm.
        from_cache: True nếu plate_text lấy từ cache (skip OCR).
        track_id: ID track tốc độ nếu speed estimator bật.
        speed_kmh: Tốc độ ước lượng đã làm mượt.
        is_speeding: True nếu tốc độ vượt ngưỡng cảnh báo.
    """

    plate_text: str
    detection: Detection
    is_violation: bool
    from_cache: bool = False
    track_id: int | None = None
    speed_kmh: float | None = None
    is_speeding: bool = False


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
        async_ocr: bool = False,
        speed_estimator: SpeedEstimatorLike | None = None,
    ) -> None:
        self._detector = detector
        self._ocr = ocr
        self._violation_detector = violation_detector
        self._cache = cache
        self._crop_expand_ratio = crop_expand_ratio
        self._ocr_quality_frames = ocr_quality_frames
        self._frame_idx: int = 0
        self._speed_estimator = speed_estimator
        
        # Async OCR setup
        self._async_ocr = async_ocr
        if self._async_ocr:
            from concurrent.futures import ThreadPoolExecutor
            import threading
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._lock = threading.Lock()
            self._pending_jobs: set[int] = set()

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_idx: int | None = None,
        fps: float | None = None,
    ) -> List[CachedPipelineResult]:
        """Xử lý một frame với OCR caching.

        YOLO detection luôn chạy. OCR chỉ gọi khi cache miss
        hoặc chưa đủ ocr_quality_frames.

        Args:
            frame: BGR image từ OpenCV ``(H, W, C)``.

        Returns:
            Danh sách kết quả cho từng detection trong frame.
        """
        self._frame_idx += 1
        speed_frame_idx = self._frame_idx if frame_idx is None else int(frame_idx)
        detections = self._detector.detect(frame)
        speed_estimates = self._estimate_speeds(detections, speed_frame_idx, fps)
        results: List[CachedPipelineResult] = []

        for det, speed in zip(detections, speed_estimates):
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
                    track_id=speed.track_id,
                    speed_kmh=speed.speed_kmh,
                    is_speeding=speed.is_speeding,
                )
            )

        # Dọn dẹp entry hết hạn
        self._cache.cleanup(self._frame_idx)

        return results

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
            # Cache HIT
            if self._async_ocr:
                # Flow bất đồng bộ (Cache HIT)
                if cached.ocr_count < self._ocr_quality_frames:
                    with self._lock:
                        is_pending = id(cached) in self._pending_jobs
                    if not is_pending:
                        self._submit_ocr_job(cached, det, frame)
                    return cached.plate_text, False
                return cached.plate_text, True
            else:
                # Flow đồng bộ (Original Cache HIT)
                if cached.ocr_count < self._ocr_quality_frames:
                    # Chạy thêm OCR để cải thiện confidence
                    plate_text, ocr_conf = self._run_ocr(det, frame)
                    if plate_text != "unknown":
                        self._cache.add_or_update(
                            bbox=bbox,
                            plate_text=plate_text,
                            confidence=ocr_conf,
                            frame_idx=self._frame_idx,
                        )
                    else:
                        self._cache.add_or_update(
                            bbox=bbox,
                            plate_text=cached.plate_text,
                            confidence=cached.confidence,
                            frame_idx=self._frame_idx,
                        )
                    best_text = plate_text if plate_text != "unknown" else cached.plate_text
                    return best_text, False

                # Đã đủ OCR quality → reuse hoàn toàn
                return cached.plate_text, True

        # Cache MISS
        if self._async_ocr:
            # Flow bất đồng bộ (Cache MISS)
            # Thêm ngay entry tạm với text "unknown" để các frame sau match bbox qua IOU,
            # tránh sinh ra nhiều task chạy ngầm trùng lặp.
            cached = self._cache.add_or_update(
                bbox=bbox,
                plate_text="unknown",
                confidence=0.0,
                frame_idx=self._frame_idx,
            )
            # ocr_count khởi tạo là 1 cho lần chạy này
            cached.ocr_count = 1
            self._submit_ocr_job(cached, det, frame)
            return cached.plate_text, False
        else:
            # Flow đồng bộ (Original Cache MISS)
            plate_text, ocr_conf = self._run_ocr(det, frame)
            if plate_text != "unknown":
                self._cache.add_or_update(
                    bbox=bbox,
                    plate_text=plate_text,
                    confidence=ocr_conf,
                    frame_idx=self._frame_idx,
                )
            return plate_text, False

    def _submit_ocr_job(
        self,
        cached: CachedPlate,
        det: Detection,
        frame: np.ndarray,
    ) -> None:
        """Gửi tác vụ OCR chạy ngầm."""
        crop = self._detector.crop_plate(
            det,
            frame,
            expand_ratio=self._crop_expand_ratio,
        )
        if crop.size == 0:
            return

        cached_id = id(cached)
        with self._lock:
            self._pending_jobs.add(cached_id)

        future = self._executor.submit(self._async_ocr_worker, crop)

        def done_callback(f):
            try:
                plate_text, confidence = f.result()
                with self._lock:
                    if plate_text != "unknown":
                        # Chỉ cập nhật khi text tốt hơn hoặc text trước đó là placeholder "unknown"
                        if confidence > cached.confidence or cached.plate_text == "unknown":
                            cached.plate_text = plate_text
                            cached.confidence = confidence
                    # Luôn tăng ocr_count để tránh vòng lặp vô hạn
                    cached.ocr_count += 1
            except Exception as e:
                logger.error("Async OCR job failed: %s", e)
            finally:
                with self._lock:
                    self._pending_jobs.discard(cached_id)

        future.add_done_callback(done_callback)

    def _async_ocr_worker(self, crop: np.ndarray) -> tuple[str, float]:
        """Worker chạy ngầm cho việc crop và OCR."""
        try:
            result = self._ocr.recognize_with_confidence(crop)
            return result.text, result.confidence
        except Exception as e:
            logger.error("Error in async OCR worker: %s", e)
            return "unknown", 0.0

    def _run_ocr(self, det: Detection, frame: np.ndarray) -> tuple[str, float]:
        """Crop và chạy OCR cho một detection.

        Args:
            det: Detection cần OCR.
            frame: Frame gốc.

        Returns:
            Tuple ``(plate_text, ocr_confidence)``.
        """
        crop = self._detector.crop_plate(
            det,
            frame,
            expand_ratio=self._crop_expand_ratio,
        )
        if crop.size == 0:
            return "unknown", 0.0

        result = self._ocr.recognize_with_confidence(crop)
        return result.text, result.confidence

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
        if self._speed_estimator is not None and hasattr(self._speed_estimator, "reset"):
            self._speed_estimator.reset()
        if self._async_ocr:
            with self._lock:
                self._pending_jobs.clear()
        logger.info("CachedPipeline reset")

    def close(self) -> None:
        """Giải phóng tài nguyên executor khi dừng pipeline."""
        if self._async_ocr:
            logger.info("Shutting down CachedPipeline ThreadPoolExecutor...")
            self._executor.shutdown(wait=False)
