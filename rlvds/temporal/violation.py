"""
Violation Detection Logic
=========================

Mục đích:
    Kết hợp Spatial (zone) + Temporal (light) để xác định vi phạm.
    Đây là logic trung tâm — quyết định xe nào vượt đèn đỏ.

Violation Condition:
    (Light State == RED) AND (Detection anchor in Violation Zone) = VIOLATION

Thư viện sử dụng:
    - datetime: Timestamp
    - Các module nội bộ: spatial (BaseSpatialReasoner), temporal (TrafficLightFSM)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from rlvds.core.base import BaseSpatialReasoner, Detection, Violation
from rlvds.temporal.traffic_light import LightState, TrafficLightFSM
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class ViolationDetector:
    """Kết hợp Spatial zone + Temporal FSM để phát hiện vi phạm vượt đèn đỏ.

    Args:
        zone: Module suy luận không gian (implement ``BaseSpatialReasoner``).
        traffic_light: FSM quản lý chu kỳ đèn giao thông.
        violations_dir: Thư mục lưu ảnh bằng chứng vi phạm.
        zone_id: ID định danh vùng giám sát.
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_frame(
        self,
        detections: List[Detection],
    ) -> List[Detection]:
        """Kiểm tra các detection trong frame, đánh dấu vi phạm.

        Logic (theo camera.py):
            1. Nếu đèn KHÔNG đỏ → return rỗng
            2. Nếu đèn đỏ → kiểm tra anchor point của mỗi detection
               có nằm trong zone hay không

        Args:
            detections: Danh sách detection từ detector.

        Returns:
            Danh sách detection được xác nhận vi phạm (``is_violation=True``).
        """
        # Reset trạng thái vi phạm của tất cả detection trước khi kiểm tra
        # Tránh "nhớ nhầm" từ frame trước → đánh dấu oan xe bình thường
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
            logger.info(
                "Detected %d violation(s) during RED light", len(violators),
            )

        return violators

    def process_violation(
        self,
        detection: Detection,
        frame: np.ndarray,
        plate_text: str,
    ) -> Optional[Violation]:
        """Tạo bản ghi vi phạm và lưu ảnh bằng chứng.

        Args:
            detection: Detection object đã xác nhận vi phạm.
            frame: Frame ảnh gốc tại thời điểm vi phạm.
            plate_text: Biển số xe đã OCR.

        Returns:
            ``Violation`` dataclass nếu ghi nhận thành công,
            ``None`` nếu biển số trùng lặp (đã ghi nhận trước đó).
        """
        if self.is_duplicate(plate_text):
            logger.debug("Duplicate plate skipped: %s", plate_text)
            return None

        now = datetime.now()
        image_name = f"{plate_text}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = self._violations_dir / image_name

        # Lưu ảnh bằng chứng
        self._violations_dir.mkdir(parents=True, exist_ok=True)
        import cv2

        success = cv2.imwrite(str(image_path), frame)
        if not success:
            logger.warning(
                "Failed to save evidence image: %s", image_path,
            )

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

        logger.info(
            "Violation recorded — plate=%s zone=%s",
            plate_text, self._zone_id,
        )
        return violation

    def is_duplicate(self, plate_text: str) -> bool:
        """Kiểm tra biển số đã được ghi nhận trong session hiện tại chưa.

        Args:
            plate_text: Chuỗi biển số cần kiểm tra.

        Returns:
            ``True`` nếu đã ghi nhận trước đó.
        """
        return plate_text in self._recorded_plates

    def get_light_state(self) -> LightState:
        """Trả về trạng thái đèn hiện tại."""
        return self._traffic_light.get_state()

    def clear_recorded_plates(self) -> None:
        """Xóa danh sách biển số đã ghi nhận (cho cycle mới)."""
        self._recorded_plates.clear()
        logger.debug("Recorded plates cleared")
