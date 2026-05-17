"""
License Plate Detector
======================

YOLOv5-based license plate detection for Vietnamese plates.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from rlvds.core.base import BaseDetector, Detection

from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class LicensePlateDetector(BaseDetector):
    """YOLOv5-based license plate detector.

    Attributes:
        model: Loaded YOLOv5 model instance.
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IOU threshold for NMS.
        image_size: Input size for inference (default 640).
        device: torch device ("cuda", "cpu", or "auto").
    """

    def __init__(
        self,
        model_path: str = "weights/lp_vn_det_yolov5n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str = "auto",
    ) -> None:
        """Initialize detector with model path and inference settings.

        Args:
            model_path: Path to YOLOv5 weights file (.pt).
            confidence_threshold: Minimum confidence threshold.
            iou_threshold: IOU threshold for NMS.
            image_size: Input size for inference.
            device: Device to run inference ("cuda", "cpu", or "auto").
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.model = None

        self._load_model_safe(model_path)

    def _load_model_safe(self, path: str) -> None:
        """Load model with error handling.

        Args:
            path: Path to weights file.
        """
        try:
            import torch

            if not Path(path).exists():
                logger.warning(f"Model file not found: {path}")
                self.model = None
                return

            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=path,
                force_reload=False,
                trust_repo=True,  # security: arbitrary code from upstream — pin commit hash in production
            )

            # Configure model
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold

            # Set device
            if self.device == "auto":
                self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.model.to(self.device)

            logger.info(f"Model loaded successfully from {path}")

        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            self.model = None

    def load_model(self, path: str) -> None:
        """Reload model from new path (hot-swap).

        Args:
            path: Path to new weights file.
        """
        self.model_path = path
        self._load_model_safe(path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a frame.

        Args:
            frame: BGR image from OpenCV (H, W, C).

        Returns:
            List of Detection objects found in the frame.
        """
        if self.model is None:
            return []

        try:
            results = self.model(frame, size=self.image_size)
            detections = []

            # Parse results: [x1, y1, x2, y2, confidence, class_id, class_name]
            for row in results.pandas().xyxy[0].values.tolist():
                x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                confidence = float(row[4])
                class_id = int(row[5])
                class_name = str(row[6]) if len(row) > 6 else "license_plate"

                if confidence < self.confidence_threshold:
                    continue

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
                detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def crop_plate(
        self,
        detection: Detection,
        frame: np.ndarray,
        expand_ratio: float = 0.15,
    ) -> np.ndarray:
        """Crop license plate region with expanded margin.

        Args:
            detection: Detection object with bbox.
            frame: Original frame to crop from.
            expand_ratio: Ratio to expand bbox (default 0.15 = 15%).

        Returns:
            Cropped image of the license plate region.
        """
        x1, y1, x2, y2 = detection.bbox
        h, w = frame.shape[:2]

        # Calculate expansion
        width = x2 - x1
        height = y2 - y1
        expand_x = int(expand_ratio * width)
        expand_y = int(expand_ratio * height)

        # Apply expansion with bounds checking
        new_x1 = max(x1 - expand_x, 0)
        new_y1 = max(y1 - expand_y, 0)
        new_x2 = min(x2 + expand_x, w)
        new_y2 = min(y2 + expand_y, h)

        # Crop and return copy
        return frame[new_y1:new_y2, new_x1:new_x2].copy()

    def warmup(self) -> None:
        """Run dummy inference to warm up the model."""
        if self.model is None:
            return

        try:
            import numpy as np
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_frame, size=self.image_size)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def is_available(self) -> bool:
        """Check if model is loaded and ready.

        Returns:
            True if model is available for inference.
        """
        return self.model is not None
