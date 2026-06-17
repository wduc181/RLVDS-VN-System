from __future__ import annotations

import numpy as np
import pytest

from rlvds.core.base import Detection
from rlvds.core.mini_pipeline import MiniPipeline
from rlvds.ocr.postprocess import (
    check_valid_plate,
    clean_plate_text,
    format_plate,
    preprocess_image,
)
from rlvds.ocr.recognizer import LicensePlateOCR
from rlvds.spatial.zones import ViolationZone
from rlvds.temporal.traffic_light import LightState, TrafficLightFSM
from rlvds.temporal.violation import ViolationDetector, mock_violation_check
from rlvds.tracking.speed_estimator import LicensePlateSpeedEstimator


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


class _CountingDetector(_FakeDetector):
    def __init__(self) -> None:
        self.crop_count = 0

    def crop_plate(self, detection, frame, expand_ratio=0.15):
        self.crop_count += 1
        return super().crop_plate(detection, frame, expand_ratio=expand_ratio)


class _CountingOCR:
    def __init__(self, plate_text: str = "30A-12345") -> None:
        self.plate_text = plate_text
        self.call_count = 0

    def recognize(self, _image):
        self.call_count += 1
        return self.plate_text


class _MovingDetector:
    def __init__(self):
        self._boxes = [
            (10, 10, 60, 40),
            (20, 10, 70, 40),
        ]
        self._idx = 0

    def detect(self, _frame):
        bbox = self._boxes[min(self._idx, len(self._boxes) - 1)]
        self._idx += 1
        return [Detection(bbox=bbox, confidence=0.9)]

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


def test_format_and_validate_legacy_4_digit_plate() -> None:
    text = format_plate("30A1234")
    assert text == "30A-1234"
    assert check_valid_plate(text) is True


def test_format_and_validate_two_letter_series_plate() -> None:
    text = format_plate("29LD-001.43")
    assert text == "29LD-00143"
    assert check_valid_plate(text) is True


def test_format_and_validate_general_two_letter_series_plate() -> None:
    text = format_plate("90-AB 285.06")
    assert text == "90AB-28506"
    assert check_valid_plate(text) is True


def test_format_preserves_raw_candidates_before_aggressive_cleanup() -> None:
    assert format_plate("90AB28506") == "90AB-28506"
    assert format_plate("14PB2980") == "14PB-2980"
    assert format_plate("90A828506") == "90A8-28506"


def test_format_prefers_current_five_digit_tail_when_tail_starts_with_zero() -> None:
    assert format_plate("24A09019") == "24A-09019"
    assert format_plate("24A-09019") == "24A-09019"


def test_format_prefers_current_car_plate_for_a_series_compact_text() -> None:
    assert format_plate("47A40194") == "47A-40194"
    assert format_plate("47A\n40194") == "47A-40194"


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("14-P8 2980", "14P8-2980"),
        ("14P82980", "14P8-2980"),
        ("14P-82980", "14P8-2980"),
        ("18-E2 7988", "18E2-7988"),
        ("18E27988", "18E2-7988"),
        ("20-L3 4660", "20L3-4660"),
        ("20L34660", "20L3-4660"),
        ("20L-34660", "20L3-4660"),
        ("21V-78713", "21V7-8713"),
        ("21V78713", "21V7-8713"),
        ("90-B3 285.06", "90B3-28506"),
        ("90B328506", "90B3-28506"),
        ("90AB28506", "90AB-28506"),
    ],
)
def test_format_and_validate_two_line_motorbike_plates(
    raw_text: str,
    expected: str,
) -> None:
    text = format_plate(raw_text)
    assert text == expected
    assert check_valid_plate(text) is True


def test_clean_plate_text_series_a1_and_numeric_tail() -> None:
    # OCR commonly confuses B in numeric tail with digit 8.
    text = clean_plate_text("30A112B45")
    assert text == "30A112845"
    assert format_plate(text) == "30A1-12845"


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


def test_mock_violation_check_true_even_when_ocr_unknown() -> None:
    zone = ViolationZone(vertices=[[0, 0], [100, 0], [100, 100], [0, 100]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    det = Detection(bbox=(10, 10, 50, 60), confidence=0.9)

    ok = mock_violation_check(
        plate_text="unknown",
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


def test_mini_pipeline_attaches_speed_metadata() -> None:
    zone = ViolationZone(vertices=[[0, 0], [100, 0], [100, 100], [0, 100]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    violation_detector = ViolationDetector(zone=zone, traffic_light=fsm)

    fake_result = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("30A12345", 0.95)],
        ]
    ]
    speed_estimator = LicensePlateSpeedEstimator(
        fps=10.0,
        meters_per_pixel=0.1,
        speed_limit_kmh=30.0,
        min_track_frames=2,
    )

    ocr = LicensePlateOCR(ocr_engine=_FakePaddle(fake_result), confidence_threshold=0.8)
    pipeline = MiniPipeline(
        detector=_MovingDetector(),
        ocr=ocr,
        violation_detector=violation_detector,
        speed_estimator=speed_estimator,
    )

    frame = np.ones((120, 120, 3), dtype=np.uint8) * 255
    first = pipeline.process_frame(frame, frame_idx=1, fps=10.0)
    second = pipeline.process_frame(frame, frame_idx=2, fps=10.0)

    assert first[0].track_id == 0
    assert first[0].speed_kmh is None
    assert second[0].track_id == 0
    assert second[0].speed_kmh == pytest.approx(36.0)
    assert second[0].is_speeding is True


def test_mini_pipeline_can_skip_ocr_independently() -> None:
    zone = ViolationZone(vertices=[[0, 0], [100, 0], [100, 100], [0, 100]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    detector = _CountingDetector()
    ocr = _CountingOCR()
    pipeline = MiniPipeline(
        detector=detector,
        ocr=ocr,
        violation_detector=ViolationDetector(zone=zone, traffic_light=fsm),
    )

    frame = np.ones((120, 120, 3), dtype=np.uint8) * 255
    out = pipeline.process_frame(frame, run_ocr=False)

    assert out[0].plate_text == "unknown"
    assert out[0].is_violation is True
    assert detector.crop_count == 0
    assert ocr.call_count == 0


def test_mini_pipeline_can_skip_speed_independently() -> None:
    zone = ViolationZone(vertices=[[0, 0], [100, 0], [100, 100], [0, 100]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    speed_estimator = LicensePlateSpeedEstimator(
        fps=10.0,
        meters_per_pixel=0.1,
        speed_limit_kmh=30.0,
        min_track_frames=2,
    )
    pipeline = MiniPipeline(
        detector=_MovingDetector(),
        ocr=_CountingOCR(),
        violation_detector=ViolationDetector(zone=zone, traffic_light=fsm),
        speed_estimator=speed_estimator,
    )

    frame = np.ones((120, 120, 3), dtype=np.uint8) * 255
    first = pipeline.process_frame(frame, frame_idx=1, fps=10.0, estimate_speed=False)
    second = pipeline.process_frame(frame, frame_idx=2, fps=10.0, estimate_speed=False)

    assert first[0].speed_kmh is None
    assert first[0].is_speeding is False
    assert second[0].speed_kmh is None
    assert second[0].is_speeding is False
