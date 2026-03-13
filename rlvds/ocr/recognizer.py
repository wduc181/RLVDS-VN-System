"""License plate OCR engines (PaddleOCR primary, YOLO-char fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from rlvds.core.base import BaseOCR
from rlvds.ocr.postprocess import clean_plate_text, format_plate, preprocess_image
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float


class LicensePlateOCR(BaseOCR):
    """Primary OCR engine using PaddleOCR."""

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        confidence_threshold: float = 0.8,
        ocr_engine: Any | None = None,
    ) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._confidence_threshold = confidence_threshold
        self._ocr = ocr_engine if ocr_engine is not None else self._build_engine()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess_image(image)

    def recognize(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return "unknown"
        if self._ocr is None:
            logger.warning("PaddleOCR engine unavailable; returning unknown")
            return "unknown"

        processed = self.preprocess(image)
        result = self._ocr.ocr(processed)
        parsed = self._parse_paddle_result(result)
        if parsed is None:
            return "unknown"

        text = format_plate(parsed.text)
        return text if text else "unknown"

    def _build_engine(self) -> Any | None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot import PaddleOCR: %s", exc)
            return None

        try:
            return PaddleOCR(
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot initialize PaddleOCR: %s", exc)
            return None

    def _parse_paddle_result(self, result: Any) -> OCRResult | None:
        if not result:
            return None

        # PaddleOCR output: [ [ [box], (text, score) ], ... ]
        lines = result[0] if isinstance(result, Sequence) and len(result) > 0 else result
        if not lines:
            return None
        if _is_paddle_entry(lines):
            entries = [lines]
        else:
            entries = [line for line in lines if _is_paddle_entry(line)]

        texts: List[str] = []
        scores: List[float] = []
        for line in entries:
            payload = line[1]
            text = str(payload[0]).strip()
            score = float(payload[1])
            if score < self._confidence_threshold:
                return None
            normalized = clean_plate_text(text)
            if normalized:
                texts.append(normalized)
                scores.append(score)

        if not texts:
            return None

        merged = "-".join(texts) if len(texts) > 1 else texts[0]
        confidence = sum(scores) / len(scores)
        return OCRResult(text=merged, confidence=confidence)


class YOLOv5CharOCR(BaseOCR):
    """Fallback OCR engine based on character detection."""

    def __init__(self, model_path: str, model: Any | None = None) -> None:
        self._model_path = model_path
        self._model = model if model is not None else self._load_model(model_path)

    def recognize(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return "unknown"
        if self._model is None:
            return "unknown"

        try:
            results = self._model(image)
            rows = results.pandas().xyxy[0].values.tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO char OCR inference failed: %s", exc)
            return "unknown"

        if len(rows) < 7 or len(rows) > 10:
            return "unknown"

        chars = []
        for row in rows:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            label = str(row[6])
            chars.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, label))

        plate_text = self._assemble_text(chars)
        plate_text = format_plate(plate_text)
        return plate_text if plate_text else "unknown"

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess_image(image)

    def _load_model(self, model_path: str) -> Any | None:
        try:
            import torch

            return torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                force_reload=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot load YOLO char model %s: %s", model_path, exc)
            return None

    @staticmethod
    def _assemble_text(chars: list[tuple[float, float, str]]) -> str:
        if not chars:
            return ""

        left = min(chars, key=lambda p: p[0])
        right = max(chars, key=lambda p: p[0])

        two_line = False
        if left != right:
            for x, y, _ in chars:
                if not check_point_linear(x, y, left[0], left[1], right[0], right[1]):
                    two_line = True
                    break

        if not two_line:
            chars_sorted = sorted(chars, key=lambda p: p[0])
            return "".join(ch for _, _, ch in chars_sorted)

        y_mean = sum(y for _, y, _ in chars) / len(chars)
        line_1 = sorted([p for p in chars if p[1] <= y_mean], key=lambda p: p[0])
        line_2 = sorted([p for p in chars if p[1] > y_mean], key=lambda p: p[0])
        text_1 = "".join(ch for _, _, ch in line_1)
        text_2 = "".join(ch for _, _, ch in line_2)
        if not text_1 or not text_2:
            merged = sorted(chars, key=lambda p: p[0])
            return "".join(ch for _, _, ch in merged)
        return f"{text_1}-{text_2}"


def linear_equation(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Return (a, b) in y = ax + b, with vertical guard."""
    if abs(x2 - x1) < 1e-6:
        return float("inf"), x1
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


def check_point_linear(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    abs_tol: float = 3.0,
) -> bool:
    """Check if point approximately lies on the line through two points."""
    a, b = linear_equation(x1, y1, x2, y2)
    if a == float("inf"):
        return abs(x - b) <= abs_tol
    expected_y = a * x + b
    return abs(expected_y - y) <= abs_tol


def _is_paddle_entry(value: Any) -> bool:
    if not isinstance(value, Sequence) or len(value) < 2:
        return False
    payload = value[1]
    if not isinstance(payload, Sequence) or len(payload) < 2:
        return False
    return isinstance(payload[0], str)
