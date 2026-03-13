from __future__ import annotations

import numpy as np

from rlvds.core.base import Detection
from rlvds.core.mini_pipeline import MiniPipeline
from rlvds.ocr.postprocess import check_valid_plate, format_plate, preprocess_image
from rlvds.ocr.recognizer import LicensePlateOCR
from rlvds.spatial.zones import ViolationZone
from rlvds.temporal.traffic_light import LightState, TrafficLightFSM
from rlvds.temporal.violation import ViolationDetector, mock_violation_check


class _FakePaddle:
    def __init__(self, result):
        self._result = result

    def ocr(self, _image):
        return self._result


class _FakeDetector:
    def detect(self, _frame):
        return [Detection(bbox=(10, 10, 60, 60), confidence=0.9)]

    def crop_plate(self, detection, frame, expand_ratio=0.15):
        x1, y1, x2, y2 = detection.bbox
        return frame[y1:y2, x1:x2]


def test_preprocess_image_upscale_and_gray() -> None:
    img = np.ones((20, 40, 3), dtype=np.uint8) * 128
    processed = preprocess_image(img)
    assert processed.ndim == 2
    assert processed.shape[0] >= 40
    assert processed.shape[1] >= 80


def test_format_and_validate_plate() -> None:
    text = format_plate("30a12345")
    assert text == "30A-12345"
    assert check_valid_plate(text) is True


def test_license_plate_ocr_with_fake_engine() -> None:
    fake_result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("30A12345", 0.95)],
        ]
    ]
    ocr = LicensePlateOCR(ocr_engine=_FakePaddle(fake_result), confidence_threshold=0.8)
    plate = ocr.recognize(np.ones((24, 80, 3), dtype=np.uint8) * 255)
    assert plate == "30A-12345"


def test_mock_violation_check_true_when_red_and_inside_zone() -> None:
    zone = ViolationZone(vertices=[[0, 0], [100, 0], [100, 100], [0, 100]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    det = Detection(bbox=(10, 10, 50, 60), confidence=0.9)

    ok = mock_violation_check(
        plate_text="30A-12345",
        detection=det,
        zone=zone,
        traffic_light=fsm,
    )
    assert ok is True


def test_mini_pipeline_detect_to_ocr_to_violation() -> None:
    zone = ViolationZone(vertices=[[0, 0], [100, 0], [100, 100], [0, 100]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    violation_detector = ViolationDetector(zone=zone, traffic_light=fsm)

    fake_result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("30A12345", 0.95)],
        ]
    ]

    ocr = LicensePlateOCR(ocr_engine=_FakePaddle(fake_result), confidence_threshold=0.8)
    pipeline = MiniPipeline(detector=_FakeDetector(), ocr=ocr, violation_detector=violation_detector)

    frame = np.ones((120, 120, 3), dtype=np.uint8) * 255
    out = pipeline.process_frame(frame)

    assert len(out) == 1
    assert out[0].plate_text == "30A-12345"
    assert out[0].is_violation is True


def test_mini_pipeline_not_violation_when_green() -> None:
    zone = ViolationZone(vertices=[[0, 0], [100, 0], [100, 100], [0, 100]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    fsm.set_state(LightState.GREEN)
    violation_detector = ViolationDetector(zone=zone, traffic_light=fsm)

    fake_result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("30A12345", 0.95)],
        ]
    ]

    ocr = LicensePlateOCR(ocr_engine=_FakePaddle(fake_result), confidence_threshold=0.8)
    pipeline = MiniPipeline(detector=_FakeDetector(), ocr=ocr, violation_detector=violation_detector)

    frame = np.ones((120, 120, 3), dtype=np.uint8) * 255
    out = pipeline.process_frame(frame)

    assert len(out) == 1
    assert out[0].is_violation is False
