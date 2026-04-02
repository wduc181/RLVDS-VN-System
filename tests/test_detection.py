from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from rlvds.detection.detector import LicensePlateDetector


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


class _FakeResults:
    def __init__(self, rows):
        self._rows = rows

    def pandas(self):
        return _FakePandas(self._rows)


class _FakeModel:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []
        self.conf = None
        self.iou = None
        self.device = None

    def __call__(self, frame, size=640):
        self.calls.append((frame.shape, size))
        return _FakeResults(self._rows)

    def to(self, device):
        self.device = device
        return self


class _BrokenModel:
    def __call__(self, _frame, size=640):
        raise RuntimeError("inference failed")


def test_detector_returns_empty_when_weights_missing(tmp_path: Path) -> None:
    detector = LicensePlateDetector(model_path=str(tmp_path / "missing.pt"))
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    assert detector.is_available() is False
    assert detector.detect(frame) == []


def test_load_model_safe_with_fake_torch(monkeypatch, tmp_path: Path) -> None:
    fake_model = _FakeModel(rows=[])

    class _FakeHub:
        def __init__(self, model):
            self.model = model
            self.called = False

        def load(self, *_args, **_kwargs):
            self.called = True
            return self.model

    fake_hub = _FakeHub(fake_model)
    fake_torch = types.SimpleNamespace(
        hub=fake_hub,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    weights = tmp_path / "dummy.pt"
    weights.write_bytes(b"fake-weights")
    detector = LicensePlateDetector(
        model_path=str(weights),
        confidence_threshold=0.42,
        iou_threshold=0.33,
        device="cpu",
    )

    assert fake_hub.called is True
    assert detector.is_available() is True
    assert detector.model is fake_model
    assert detector.model.conf == 0.42
    assert detector.model.iou == 0.33
    assert detector.model.device == "cpu"


def test_detect_parses_rows_and_filters_by_confidence(tmp_path: Path) -> None:
    detector = LicensePlateDetector(model_path=str(tmp_path / "missing.pt"), confidence_threshold=0.5)
    detector.model = _FakeModel(
        rows=[
            [10, 20, 60, 50, 0.92, 0, "license_plate"],
            [12, 24, 50, 44, 0.2, 0, "license_plate"],
        ]
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.bbox == (10, 20, 60, 50)
    assert det.confidence == 0.92
    assert det.class_id == 0
    assert det.class_name == "license_plate"


def test_detect_returns_empty_when_model_raises(tmp_path: Path) -> None:
    detector = LicensePlateDetector(model_path=str(tmp_path / "missing.pt"))
    detector.model = _BrokenModel()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_crop_plate_expands_and_clamps_bounds(tmp_path: Path) -> None:
    detector = LicensePlateDetector(model_path=str(tmp_path / "missing.pt"))
    frame = np.ones((40, 40, 3), dtype=np.uint8) * 255

    class _Det:
        bbox = (10, 10, 20, 20)

    crop = detector.crop_plate(_Det(), frame, expand_ratio=0.5)
    assert crop.shape == (20, 20, 3)


def test_warmup_calls_model(tmp_path: Path) -> None:
    detector = LicensePlateDetector(model_path=str(tmp_path / "missing.pt"))
    fake_model = _FakeModel(rows=[])
    detector.model = fake_model
    detector.warmup()
    assert len(fake_model.calls) == 1


def test_load_model_calls_loader(monkeypatch, tmp_path: Path) -> None:
    detector = LicensePlateDetector(model_path=str(tmp_path / "missing.pt"))
    captured = {}

    def _fake_loader(path: str) -> None:
        captured["path"] = path

    monkeypatch.setattr(detector, "_load_model_safe", _fake_loader)
    new_path = str(tmp_path / "new.pt")
    detector.load_model(new_path)
    assert captured["path"] == new_path
