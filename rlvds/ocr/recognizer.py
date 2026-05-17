"""License plate OCR engines (PaddleOCR primary, YOLO-char fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from rlvds.core.base import BaseOCR
from config.settings import get_settings
from rlvds.ocr.postprocess import clean_plate_text, format_plate
from rlvds.ocr.preprocessor import PlatePreprocessor
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
        preprocessor: Optional[PlatePreprocessor] = None,
    ) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._confidence_threshold = confidence_threshold
        self._ocr = ocr_engine if ocr_engine is not None else self._build_engine()
        self._preprocessor = (
            preprocessor
            if preprocessor is not None
            else PlatePreprocessor(get_settings().preprocessing)
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return self._preprocessor.run_pipeline(image)

    def recognize(self, image: np.ndarray) -> str:
        result = self.recognize_with_confidence(image)
        return result.text

    def recognize_with_confidence(self, image: np.ndarray) -> OCRResult:
        """Nhận diện text từ ảnh biển số, trả về cả confidence.

        Args:
            image: Ảnh biển số ``(H, W, C)`` dạng BGR.

        Returns:
            ``OCRResult(text, confidence)``.
        """
        if image is None or image.size == 0:
            return OCRResult(text="unknown", confidence=0.0)
        if self._ocr is None:
            logger.warning("PaddleOCR engine unavailable; returning unknown")
            return OCRResult(text="unknown", confidence=0.0)

        processed = self.preprocess(image)
        result = self._ocr.ocr(processed)
        parsed = self._parse_paddle_result(result)
        if parsed is None:
            return OCRResult(text="unknown", confidence=0.0)

        text = format_plate(parsed.text)
        if not text:
            return OCRResult(text="unknown", confidence=0.0)
        return OCRResult(text=text, confidence=parsed.confidence)

    def _build_engine(self) -> Any | None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot import PaddleOCR: %s", exc)
            return None

        # HARDCODE use_gpu=False để tránh xung đột cuDNN
        # PyTorch (CUDA 12.4) kéo cuDNN 9.x, PaddlePaddle 2.6.2 chỉ tương thích cuDNN 8.x
        # OCR xử lý ảnh biển số nhỏ (~150x50px) nên CPU đủ nhanh, không cần GPU
        if self._use_gpu:
            logger.warning(
                "OCR use_gpu=True in config is overridden — PaddleOCR forced to CPU "
                "to avoid cuDNN 8.x/9.x conflict with PyTorch CUDA 12.4"
            )
        try:
            logger.info("Initializing PaddleOCR with lang='%s', use_gpu=False (CPU-only)...", self._lang)
            return PaddleOCR(
                lang=self._lang,
                use_gpu=False,  # LUÔN dùng CPU để tránh xung đột cuDNN
                show_log=False,
            )
        except Exception as e1:
            logger.warning("Failed to initialize with lang: %s - %s", type(e1).__name__, e1)

            # Try initializing fallback with CPU
            try:
                logger.info("Trying to initialize PaddleOCR default (CPU-only)...")
                return PaddleOCR(use_gpu=False, show_log=False)
            except Exception as e2:
                logger.error("Failed to initialize PaddleOCR: %s - %s", type(e2).__name__, e2)
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
                continue
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

    def __init__(
        self,
        model_path: str,
        model: Any | None = None,
    ) -> None:
        self._model_path = model_path
        self._model = model if model is not None else self._load_model(model_path)

    def recognize(self, image: np.ndarray) -> str:
        result = self.recognize_with_confidence(image)
        return result.text

    def recognize_with_confidence(self, image: np.ndarray) -> OCRResult:
        """Nhận diện text từ ảnh biển số, trả về cả confidence.

        Args:
            image: Ảnh biển số ``(H, W, C)`` dạng BGR.

        Returns:
            ``OCRResult(text, confidence)``.
        """
        if image is None or image.size == 0:
            return OCRResult(text="unknown", confidence=0.0)
        if self._model is None:
            return OCRResult(text="unknown", confidence=0.0)

        try:
            results = self._model(image)
            rows = results.pandas().xyxy[0].values.tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO char OCR inference failed: %s", exc)
            return OCRResult(text="unknown", confidence=0.0)

        if len(rows) < 7 or len(rows) > 10:
            return OCRResult(text="unknown", confidence=0.0)

        chars = []
        confidences: List[float] = []
        for row in rows:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            conf = float(row[4])
            label = str(row[6])
            chars.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, label))
            confidences.append(conf)

        plate_text = self._assemble_text(chars)
        plate_text = format_plate(plate_text)
        if not plate_text:
            return OCRResult(text="unknown", confidence=0.0)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(text=plate_text, confidence=avg_conf)

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
