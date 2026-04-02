from __future__ import annotations

import numpy as np
import pytest

from rlvds.ocr.recognizer import (
    LicensePlateOCR,
    OCRResult,
    YOLOv5CharOCR,
    check_point_linear,
    linear_equation,
)


class _FakePreprocessor:
    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        return image


class _FakePaddleEngine:
    def __init__(self, result):
        self._result = result

    def ocr(self, _image):
        return self._result


class _FakeValues:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class _FakeTable:
    def __init__(self, rows):
        self.values = _FakeValues(rows)


class _FakePandas:
    def __init__(self, rows):
        self.xyxy = [_FakeTable(rows)]


class _FakeYOLOResults:
    def __init__(self, rows):
        self._rows = rows

    def pandas(self):
        return _FakePandas(self._rows)


class _FakeYOLOModel:
    def __init__(self, rows):
        self._rows = rows

    def __call__(self, _image):
        return _FakeYOLOResults(self._rows)


def test_license_plate_ocr_returns_unknown_for_empty_image() -> None:
    ocr = LicensePlateOCR(
        ocr_engine=_FakePaddleEngine(result=[]),
        preprocessor=_FakePreprocessor(),
    )
    result = ocr.recognize_with_confidence(np.empty((0, 0, 3), dtype=np.uint8))
    assert result == OCRResult(text="unknown", confidence=0.0)


def test_license_plate_ocr_returns_unknown_when_engine_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(LicensePlateOCR, "_build_engine", lambda self: None)
    ocr = LicensePlateOCR(preprocessor=_FakePreprocessor())
    image = np.ones((24, 80, 3), dtype=np.uint8) * 255
    result = ocr.recognize_with_confidence(image)
    assert result.text == "unknown"
    assert result.confidence == 0.0


def test_license_plate_ocr_rejects_low_confidence_text() -> None:
    fake_result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("30A12345", 0.6)],
        ]
    ]
    ocr = LicensePlateOCR(
        ocr_engine=_FakePaddleEngine(result=fake_result),
        confidence_threshold=0.8,
        preprocessor=_FakePreprocessor(),
    )
    image = np.ones((24, 80, 3), dtype=np.uint8) * 255
    result = ocr.recognize_with_confidence(image)
    assert result.text == "unknown"


def test_license_plate_ocr_parses_single_entry_structure() -> None:
    entry = [[[0, 0], [1, 0], [1, 1], [0, 1]], ("30A12345", 0.95)]
    ocr = LicensePlateOCR(
        ocr_engine=_FakePaddleEngine(result=[entry]),
        confidence_threshold=0.8,
        preprocessor=_FakePreprocessor(),
    )
    image = np.ones((24, 80, 3), dtype=np.uint8) * 255
    result = ocr.recognize_with_confidence(image)
    assert result.text == "30A-12345"
    assert result.confidence == pytest.approx(0.95)


def test_license_plate_ocr_merges_multi_line_output() -> None:
    fake_result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("30A1", 0.90)],
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("2345", 0.94)],
        ]
    ]
    ocr = LicensePlateOCR(
        ocr_engine=_FakePaddleEngine(result=fake_result),
        confidence_threshold=0.8,
        preprocessor=_FakePreprocessor(),
    )
    image = np.ones((30, 120, 3), dtype=np.uint8) * 255
    result = ocr.recognize_with_confidence(image)
    assert result.text == "30A-12345"
    assert result.confidence == pytest.approx((0.90 + 0.94) / 2.0)


def test_yolov5_char_ocr_returns_unknown_for_invalid_char_count() -> None:
    # 6 chars => outside expected [7..10] range
    rows = [
        [0, 0, 10, 10, 0.9, 0, "3"],
        [10, 0, 20, 10, 0.9, 0, "0"],
        [20, 0, 30, 10, 0.9, 0, "A"],
        [30, 0, 40, 10, 0.9, 0, "1"],
        [40, 0, 50, 10, 0.9, 0, "2"],
        [50, 0, 60, 10, 0.9, 0, "3"],
    ]
    ocr = YOLOv5CharOCR(model_path="unused.pt", model=_FakeYOLOModel(rows))
    result = ocr.recognize_with_confidence(np.ones((40, 120, 3), dtype=np.uint8))
    assert result.text == "unknown"
    assert result.confidence == 0.0


def test_yolov5_char_ocr_recognizes_plate_and_confidence() -> None:
    rows = [
        [0, 0, 10, 10, 0.91, 0, "3"],
        [12, 0, 22, 10, 0.92, 0, "0"],
        [24, 0, 34, 10, 0.93, 0, "A"],
        [36, 0, 46, 10, 0.94, 0, "1"],
        [48, 0, 58, 10, 0.95, 0, "2"],
        [60, 0, 70, 10, 0.96, 0, "3"],
        [72, 0, 82, 10, 0.97, 0, "4"],
        [84, 0, 94, 10, 0.98, 0, "5"],
    ]
    ocr = YOLOv5CharOCR(model_path="unused.pt", model=_FakeYOLOModel(rows))
    result = ocr.recognize_with_confidence(np.ones((50, 140, 3), dtype=np.uint8))
    assert result.text == "30A-12345"
    assert result.confidence == pytest.approx(sum(r[4] for r in rows) / len(rows))


def test_assemble_text_two_line_layout() -> None:
    chars = [(10, 10, "3"), (20, 10, "0"), (10, 20, "A"), (20, 20, "1")]
    text = YOLOv5CharOCR._assemble_text(chars)
    assert text == "30-A1"


def test_linear_helpers_cover_vertical_and_sloped_lines() -> None:
    a, b = linear_equation(2.0, 1.0, 2.0, 10.0)
    assert a == float("inf")
    assert b == 2.0
    assert check_point_linear(2.2, 100.0, 2.0, 1.0, 2.0, 10.0, abs_tol=0.3) is True
    assert check_point_linear(3.0, 100.0, 2.0, 1.0, 2.0, 10.0, abs_tol=0.3) is False

    a2, b2 = linear_equation(0.0, 0.0, 10.0, 10.0)
    assert a2 == pytest.approx(1.0)
    assert b2 == pytest.approx(0.0)
    assert check_point_linear(5.0, 5.2, 0.0, 0.0, 10.0, 10.0, abs_tol=0.3) is True
    assert check_point_linear(5.0, 6.0, 0.0, 0.0, 10.0, 10.0, abs_tol=0.3) is False
