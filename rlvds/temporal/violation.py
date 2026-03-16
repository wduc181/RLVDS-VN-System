"""Violation logic: spatial + temporal checks, with mock helper."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

from rlvds.core.base import BaseSpatialReasoner, Detection, Violation
from rlvds.temporal.traffic_light import LightState, TrafficLightFSM
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class ViolationDetector:
    """Combine spatial zone and traffic-light state for violation checks."""

    def __init__(
        self,
        zone: BaseSpatialReasoner,
        traffic_light: TrafficLightFSM,
        violations_dir: str = "data/violations",
        zone_id: str = "default",
    ) -> None:
        self._zone = zone
        self._traffic_light = traffic_light
        self._violations_dir = Path(violations_dir)
        self._zone_id = zone_id
        self._recorded_plates: Set[str] = set()

    def check_frame(self, detections: List[Detection]) -> List[Detection]:
        """Mark detections as violations when red light and anchor in zone."""
        for det in detections:
            det.is_violation = False

        if not self._traffic_light.is_red():
            return []

        violators: List[Detection] = []
        for det in detections:
            anchor = det.get_anchor_point()
            if self._zone.is_in_zone(anchor):
                det.is_violation = True
                violators.append(det)

        if violators:
            logger.info("Detected %d violation(s) during RED light", len(violators))
        return violators

    def check_mock_violation(self, plate_text: str, detection: Detection) -> bool:
        """Mock check used in week-3 mini pipeline.

        Condition: plate appears inside polygon while traffic light is RED.
        """
        if not plate_text or plate_text == "unknown":
            return False
        if not self._traffic_light.is_red():
            return False

        in_zone = self._zone.is_in_zone(detection.get_anchor_point())
        if in_zone:
            logger.info("[MOCK] Violation plate=%s zone=%s", plate_text, self._zone_id)
            return True
        return False

    def process_violation(
        self,
        detection: Detection,
        frame: np.ndarray,
        plate_text: str,
    ) -> Optional[Violation]:
        """Build violation record and save evidence image."""
        if self.is_duplicate(plate_text):
            logger.debug("Duplicate plate skipped: %s", plate_text)
            return None

        now = datetime.now()
        image_name = f"{plate_text}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = self._violations_dir / image_name

        self._violations_dir.mkdir(parents=True, exist_ok=True)
        import cv2

        success = cv2.imwrite(str(image_path), frame)
        if not success:
            logger.warning("Failed to save evidence image: %s", image_path)

        self._recorded_plates.add(plate_text)

        violation = Violation(
            plate_text=plate_text,
            timestamp=now,
            image_path=str(image_path),
            confidence=detection.confidence,
            bbox=detection.bbox,
            zone_id=self._zone_id,
            metadata={
                "light_state": self._traffic_light.get_light_state(),
                "anchor_point": detection.get_anchor_point(),
            },
        )

        logger.info("Violation recorded plate=%s zone=%s", plate_text, self._zone_id)
        return violation

    def is_duplicate(self, plate_text: str) -> bool:
        return plate_text in self._recorded_plates

    def get_light_state(self) -> LightState:
        return self._traffic_light.get_state()

    def clear_recorded_plates(self) -> None:
        self._recorded_plates.clear()
        logger.debug("Recorded plates cleared")


def mock_violation_check(
    *,
    plate_text: str,
    detection: Detection,
    zone: BaseSpatialReasoner,
    traffic_light: TrafficLightFSM,
) -> bool:
    """Standalone helper for quick testing without full class wiring."""
    if not plate_text or plate_text == "unknown":
        return False
    if not traffic_light.is_red():
        return False

    in_zone = zone.is_in_zone(detection.get_anchor_point())
    if in_zone:
        logger.info("[MOCK] Violation plate=%s at anchor=%s", plate_text, detection.get_anchor_point())
    return in_zone
