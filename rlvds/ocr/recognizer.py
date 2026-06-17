"""License plate OCR engines (PaddleOCR primary, YOLO-char fallback)."""

from __future__ import annotations

from collections.abc import Iterator
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from rlvds.core.base import BaseOCR
from config.settings import get_settings
from rlvds.ocr.postprocess import check_valid_plate, clean_plate_text, format_plate
from rlvds.ocr.preprocessor import PlatePreprocessor, prepare_paddle_ocr_input
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)

OCR_SERVICE_URL = "http://127.0.0.1:8502"


def is_ocr_service_available(
    url: str = OCR_SERVICE_URL,
    timeout: float = 0.5,
) -> bool:
    """Return True only when the service on ``url`` speaks the OCR JSON API."""
    import json
    import urllib.error
    import urllib.request

    import cv2

    probe_image = np.ones((8, 16, 3), dtype=np.uint8) * 255
    success, encoded_img = cv2.imencode(".png", probe_image)
    if not success:
        return False

    request = urllib.request.Request(
        url,
        data=encoded_img.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False
    except Exception:  # noqa: BLE001
        return False

    return isinstance(payload, dict) and "raw_result" in payload


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
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        enable_mkldnn: bool | None = None,
        cpu_threads: int | None = None,
        use_angle_cls: bool | None = None,
        enhanced_fallback: bool | None = None,
    ) -> None:
        settings = get_settings()
        ocr_cfg = settings.ocr
        self._lang = lang
        self._use_gpu = use_gpu
        self._confidence_threshold = confidence_threshold
        self._det_model_dir = (
            det_model_dir if det_model_dir is not None else ocr_cfg.det_model_dir
        )
        self._rec_model_dir = (
            rec_model_dir if rec_model_dir is not None else ocr_cfg.rec_model_dir
        )
        self._enable_mkldnn = (
            enable_mkldnn
            if enable_mkldnn is not None
            else getattr(ocr_cfg, "enable_mkldnn", False)
        )
        self._cpu_threads = (
            cpu_threads
            if cpu_threads is not None
            else getattr(ocr_cfg, "cpu_threads", 2)
        )
        self._use_angle_cls = (
            use_angle_cls
            if use_angle_cls is not None
            else getattr(ocr_cfg, "use_angle_cls", False)
        )
        self._enhanced_fallback = (
            enhanced_fallback
            if enhanced_fallback is not None
            else getattr(ocr_cfg, "enhanced_fallback", False)
        )
        self._use_http = (ocr_engine is None)
        self._ocr = ocr_engine if ocr_engine is not None else self._build_engine()
        self._preprocessor = (
            preprocessor
            if preprocessor is not None
            else PlatePreprocessor(settings.preprocessing)
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

        # Thử gửi qua HTTP OCR Microservice trước. Raw được thử trước;
        # nếu chưa ra biển hợp lệ thì mới tốn thêm bước enhance/CLAHE.
        if self._use_http:
            import urllib.request
            import urllib.error
            import json
            import cv2

            for variant_name, ocr_image in self._ocr_input_variants(image):
                success, encoded_img = cv2.imencode(".png", ocr_image)
                if not success:
                    continue
                req_data = encoded_img.tobytes()
                try:
                    req = urllib.request.Request(
                        OCR_SERVICE_URL,
                        data=req_data,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    with urllib.request.urlopen(req, timeout=5) as response:
                        resp_data = response.read().decode("utf-8")
                        resp_json = json.loads(resp_data)
                        raw_result = resp_json.get("raw_result")
                        parsed = self._parse_paddle_result(raw_result)
                        formatted = self._format_valid_result(parsed)
                        if formatted is not None:
                            return formatted
                except Exception as e:
                    logger.debug(
                        "Failed to connect to OCR microservice on %s image, "
                        "using local engine: %s",
                        variant_name,
                        e,
                    )
                    break

        # Fallback về chạy cục bộ trong cùng tiến trình.
        if self._ocr is None:
            logger.warning("PaddleOCR engine unavailable; returning unknown")
            return OCRResult(text="unknown", confidence=0.0)

        for variant_name, processed in self._ocr_input_variants(image):
            try:
                result = self._run_paddle_ocr(processed)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PaddleOCR inference failed on %s image: %s",
                    variant_name,
                    exc,
                )
                continue
            formatted = self._format_valid_result(self._parse_paddle_result(result))
            if formatted is not None:
                return formatted

        return OCRResult(text="unknown", confidence=0.0)

    def _format_valid_result(self, parsed: OCRResult | None) -> OCRResult | None:
        if parsed is None:
            return None
        text = format_plate(parsed.text)
        if not text or not check_valid_plate(text):
            return None
        return OCRResult(text=text, confidence=parsed.confidence)

    def _ocr_input_variants(
        self,
        image: np.ndarray,
    ) -> Iterator[tuple[str, np.ndarray]]:
        raw = prepare_paddle_ocr_input(image)
        yield "raw", raw

        if not self._enhanced_fallback:
            return

        try:
            enhanced = prepare_paddle_ocr_input(self.preprocess(image))
        except Exception as exc:  # noqa: BLE001
            logger.debug("OCR preprocessing failed, using raw crop only: %s", exc)
            return

        if enhanced.shape != raw.shape or not np.array_equal(enhanced, raw):
            yield "enhanced", enhanced

    def _run_paddle_ocr(self, image: np.ndarray) -> Any:
        try:
            return self._ocr.ocr(image, cls=False)
        except TypeError:
            return self._ocr.ocr(image)

    def _build_engine(self) -> Any | None:
        # Kiểm tra xem OCR Microservice đã chạy chưa, nếu có thì không cần load model cục bộ
        if is_ocr_service_available(OCR_SERVICE_URL, timeout=0.5):
            logger.info(
                "Detected active OCR Microservice. "
                "Bypassing local engine initialization."
            )
            return None

        os.environ.setdefault("FLAGS_use_mkldnn", "0")

        # HARDCODE use_gpu=False để tránh xung đột cuDNN
        # PyTorch (CUDA 12.4) kéo cuDNN 9.x, PaddlePaddle 2.6.2 chỉ tương thích cuDNN 8.x
        # OCR xử lý ảnh biển số nhỏ (~150x50px) nên CPU đủ nhanh, không cần GPU
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cannot import PaddleOCR: %s", exc)
            return None

        kwargs = self._paddle_kwargs(include_lang=True)
        try:
            logger.info(
                "Initializing PaddleOCR with lang='%s', use_gpu=False, "
                "enable_mkldnn=%s, cpu_threads=%d...",
                self._lang,
                self._enable_mkldnn,
                self._cpu_threads,
            )
            return PaddleOCR(**kwargs)
        except Exception as e1:
            logger.warning(
                "Failed to initialize with lang: %s - %s",
                type(e1).__name__,
                e1,
            )

            # Try initializing fallback with CPU
            try:
                logger.info("Trying to initialize PaddleOCR default (CPU-only)...")
                return PaddleOCR(**self._paddle_kwargs(include_lang=False))
            except Exception as e2:
                logger.error(
                    "Failed to initialize PaddleOCR: %s - %s",
                    type(e2).__name__,
                    e2,
                )
                return None

    def _paddle_kwargs(self, *, include_lang: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "use_gpu": False,  # LUÔN dùng CPU để tránh xung đột cuDNN
            "show_log": False,
            "use_angle_cls": self._use_angle_cls,
            "enable_mkldnn": self._enable_mkldnn,
            "cpu_threads": self._cpu_threads,
        }
        if include_lang:
            kwargs["lang"] = self._lang
        if self._det_model_dir:
            kwargs["det_model_dir"] = self._det_model_dir
        if self._rec_model_dir:
            kwargs["rec_model_dir"] = self._rec_model_dir
        return kwargs

    def _parse_paddle_result(self, result: Any) -> OCRResult | None:
        if not result:
            return None

        entries = _collect_paddle_entries(result)
        if not entries:
            return None

        # Sort entries by the minimum Y coordinate to ensure top-to-bottom order (critical for 2-line plates)
        entries.sort(key=lambda line: min(p[1] for p in line[0]))

        raw_texts: List[str] = []
        cleaned_texts: List[str] = []
        scores: List[float] = []
        for line in entries:
            payload = line[1]
            raw_text = str(payload[0]).strip()
            score = float(payload[1])

            # Skip extreme low confidence noise boxes instead of rejecting the whole plate
            if score < 0.4:
                continue

            normalized = clean_plate_text(raw_text)
            # Skip junk characters
            if normalized and len(normalized) >= 2:
                raw_texts.append(raw_text)
                cleaned_texts.append(normalized)
                scores.append(score)

        if not raw_texts:
            return None

        valid_candidates: list[tuple[int, float, str]] = []
        for start in range(len(raw_texts)):
            for end in range(start + 1, len(raw_texts) + 1):
                candidate_scores = scores[start:end]
                candidate_confidence = sum(candidate_scores) / len(candidate_scores)
                if candidate_confidence < self._confidence_threshold:
                    continue
                for candidate_text in _candidate_text_variants(
                    raw_texts[start:end],
                    cleaned_texts[start:end],
                ):
                    formatted = format_plate(candidate_text)
                    if check_valid_plate(formatted):
                        valid_candidates.append(
                            (end - start, candidate_confidence, formatted)
                        )

        if valid_candidates:
            _, confidence, text = max(
                valid_candidates,
                key=lambda item: (item[0], item[1]),
            )
            return OCRResult(text=text, confidence=confidence)

        confidence = sum(scores) / len(scores)
        if confidence < self._confidence_threshold:
            return None

        merged = (
            "-".join(cleaned_texts)
            if len(cleaned_texts) > 1
            else cleaned_texts[0]
        )
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


def _collect_paddle_entries(value: Any) -> list[Any]:
    if _is_paddle_entry(value):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    entries: list[Any] = []
    for item in value:
        entries.extend(_collect_paddle_entries(item))
    return entries


def _candidate_text_variants(
    raw_texts: Sequence[str],
    cleaned_texts: Sequence[str],
) -> list[str]:
    has_explicit_separator = any("-" in raw_text for raw_text in raw_texts)
    if has_explicit_separator:
        candidates = [
            "-".join(raw_texts),
            "-".join(cleaned_texts),
            "".join(raw_texts),
            "".join(cleaned_texts),
        ]
    else:
        candidates = [
            "".join(raw_texts),
            "".join(cleaned_texts),
            "-".join(raw_texts),
            "-".join(cleaned_texts),
        ]
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered
