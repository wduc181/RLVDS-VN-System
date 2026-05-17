"""RLVDS Processing Pipeline — full orchestration of detection, OCR, and violation logic."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config.settings import Settings, get_settings
from rlvds.core.cached_pipeline import CachedPipeline
from rlvds.core.mini_pipeline import MiniPipeline
from rlvds.detection.detector import LicensePlateDetector
from rlvds.ingestion.video_source import VideoSource
from rlvds.ocr.plate_cache import PlateTrackCache
from rlvds.ocr.preprocessor import PlatePreprocessor
from rlvds.ocr.recognizer import LicensePlateOCR
from rlvds.persistence.database import Database
from rlvds.persistence.repository import ViolationRepository
from rlvds.spatial.zones import ViolationZone
from rlvds.temporal.traffic_light import TrafficLightFSM
from rlvds.temporal.violation import ViolationDetector
from rlvds.utils.logger import get_logger
from rlvds.utils.visualization import (
    draw_detections,
    draw_fps,
    draw_light_status,
    draw_zone_overlay,
)

logger = get_logger(__name__)


class Pipeline:
    """Full RLVDS pipeline orchestrating all components.

    Args:
        settings: Application settings. Loaded from config if not provided.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg = settings or get_settings()

        self.detector = LicensePlateDetector(
            model_path=self._cfg.detection.model_path,
            confidence_threshold=self._cfg.detection.confidence_threshold,
            iou_threshold=self._cfg.detection.iou_threshold,
            image_size=self._cfg.detection.image_size,
            device=self._cfg.detection.device,
        )
        self.ocr = LicensePlateOCR(
            lang=self._cfg.ocr.lang,
            use_gpu=self._cfg.ocr.use_gpu,
            confidence_threshold=self._cfg.ocr.confidence_threshold,
        )
        self.zone = ViolationZone(
            vertices=self._cfg.spatial.violation_zone,
            zone_id="default",
            color=self._cfg.spatial.zone_color,
            thickness=self._cfg.spatial.zone_thickness,
        )
        self.traffic_light = TrafficLightFSM(
            red_sec=self._cfg.temporal.red_duration_sec,
            green_sec=self._cfg.temporal.green_duration_sec,
            yellow_sec=self._cfg.temporal.yellow_duration_sec,
            initial_state=self._cfg.temporal.initial_state,
        )
        self.violation_detector = ViolationDetector(
            zone=self.zone,
            traffic_light=self.traffic_light,
            violations_dir=self._cfg.paths.violations_dir,
            zone_id=self.zone.zone_id,
        )
        self.video_source: Optional[VideoSource] = None
        self.db: Optional[Database] = None
        self.repo: Optional[ViolationRepository] = None
        self._running = False

        logger.info("Pipeline initialized (device=%s)", self._cfg.detection.device)

    def run(self, source: str | int, *, display: bool = True) -> None:
        """Run the pipeline on a video source.

        Args:
            source: Video file path or camera index.
            display: Show OpenCV window output.
        """
        self._start(source)
        self._running = True

        prev_time = time.perf_counter()
        violation_count = 0

        try:
            target_fps = self._cfg.video.fps
            frame_iter = (
                self.video_source.iter_frames_throttled(float(target_fps))
                if target_fps > 0
                else self.video_source
            )
            for frame in frame_iter:  # type: ignore[union-attr]
                if not self._running:
                    break

                now = time.perf_counter()
                fps = round(1 / (now - prev_time), 1) if now != prev_time else 0.0
                prev_time = now

                light_state = self.traffic_light.get_state().value

                draw_zone_overlay(
                    frame,
                    self.zone.polygon,
                    color=self._cfg.spatial.zone_color,
                    alpha=0.25,
                    thickness=self._cfg.spatial.zone_thickness,
                )

                detection_results = self._process_detections(frame)

                saved = self._persist_violations(frame, detection_results, light_state)
                violation_count += saved

                draw_light_status(frame, light_state)
                draw_fps(frame, fps)
                draw_detections(frame, detection_results)

                if display:
                    display_frame = cv2.resize(frame, (1280, 720))
                    cv2.imshow("RLVDS-VN", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            self.stop()

    def process_frame(self, frame: np.ndarray) -> List:
        """Process a single frame through detection and violation check."""
        return self._process_detections(frame)

    def stop(self) -> None:
        """Release all resources."""
        self._running = False
        if self.video_source is not None:
            self.video_source.release()
            self.video_source = None
        if self.db is not None:
            self.db.disconnect()
            self.db = None
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped")

    def _start(self, source: str | int) -> None:
        self.video_source = VideoSource(source)
        self.traffic_light.start()

        self.db = Database(self._cfg.database.url)
        self.db.connect()
        self.db.create_tables()
        self.db.migrate_schema()

        self.repo = ViolationRepository(
            database=self.db,
            violations_dir=self._cfg.paths.violations_dir,
        )

        # Build cached or mini pipeline
        preprocessor = PlatePreprocessor(self._cfg.preprocessing)
        if self._cfg.ocr_cache.enabled:
            cache = PlateTrackCache(
                iou_threshold=self._cfg.ocr_cache.iou_threshold,
                max_size=self._cfg.ocr_cache.max_cache_size,
                ttl_frames=self._cfg.ocr_cache.cache_ttl_frames,
            )
            self._pipeline = CachedPipeline(
                detector=self.detector,
                ocr=self.ocr,
                violation_detector=self.violation_detector,
                cache=cache,
                crop_expand_ratio=self._cfg.preprocessing.expand_ratio,
                ocr_quality_frames=self._cfg.ocr_cache.ocr_quality_frames,
            )
        else:
            self._pipeline = MiniPipeline(
                detector=self.detector,
                ocr=self.ocr,
                violation_detector=self.violation_detector,
                crop_expand_ratio=self._cfg.preprocessing.expand_ratio,
            )

        self._preprocessor = preprocessor
        logger.info("Pipeline started — source=%s", source)

    def _process_detections(self, frame: np.ndarray) -> List:
        if not self.detector.is_available():
            return []
        try:
            return self._pipeline.process_frame(frame)
        except Exception as exc:  # noqa: BLE001
            logger.error("Detection failed: %s", exc)
            return []

    def _persist_violations(
        self,
        frame: np.ndarray,
        results: List,
        light_state: str,
    ) -> int:
        if self.repo is None or not results:
            return 0
        saved = 0
        for result in results:
            if not result.is_violation or result.plate_text == "unknown":
                continue
            det = result.detection
            crop = det.crop(frame)
            processed_plate = None
            if hasattr(self, "_preprocessor") and crop.size > 0:
                processed = self._preprocessor.run_pipeline(crop)
                if processed.size > 0:
                    processed_plate = processed
            inserted_id = self.repo.record_violation(
                frame=frame,
                detection=det,
                plate_text=result.plate_text,
                light_state=light_state,
                preprocessed_plate=processed_plate,
                polygon=self.zone.polygon,
                zone_id=self.zone.zone_id,
                confidence=det.confidence,
            )
            if inserted_id is not None:
                saved += 1
        return saved
