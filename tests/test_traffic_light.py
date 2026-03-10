"""
Tests for Temporal Module
=========================

Covers:
    - rlvds/temporal/traffic_light.py  (LightState, TrafficLightFSM)
    - rlvds/temporal/timing.py         (frame_to_time, get_current_timestamp, ...)
    - rlvds/temporal/violation.py      (ViolationDetector)

Chạy: pytest tests/test_traffic_light.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from rlvds.core.base import Detection
from rlvds.spatial.zones import ViolationZone
from rlvds.temporal.timing import (
    calculate_video_offset,
    frame_to_time,
    get_current_timestamp,
)
from rlvds.temporal.traffic_light import LightState, TrafficLightFSM
from rlvds.temporal.violation import ViolationDetector


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def fsm() -> TrafficLightFSM:
    """FSM mặc định: R=30s, G=30s, Y=3s, bắt đầu ở RED."""
    f = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    f.start()
    return f


@pytest.fixture
def zone() -> ViolationZone:
    """Zone 100×100 tại gốc."""
    return ViolationZone(
        vertices=[[0, 0], [100, 0], [100, 100], [0, 100]],
        zone_id="test",
    )


@pytest.fixture
def detector(zone: ViolationZone, fsm: TrafficLightFSM) -> ViolationDetector:
    """ViolationDetector với zone + FSM mặc định."""
    return ViolationDetector(
        zone=zone,
        traffic_light=fsm,
        violations_dir="/tmp/rlvds_test_violations",
        zone_id="test",
    )


def _make_detection(
    x1: int, y1: int, x2: int, y2: int, conf: float = 0.9,
) -> Detection:
    """Helper tạo Detection nhanh."""
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf)


# =========================================================================
# test LightState enum
# =========================================================================

class TestLightState:
    """Test ``LightState`` enum values."""

    def test_values(self) -> None:
        assert LightState.RED.value == "RED"
        assert LightState.GREEN.value == "GREEN"
        assert LightState.YELLOW.value == "YELLOW"

    def test_from_string(self) -> None:
        assert LightState("RED") is LightState.RED


# =========================================================================
# test TrafficLightFSM
# =========================================================================

class TestTrafficLightFSM:
    """Test ``TrafficLightFSM``."""

    def test_initial_state_red(self, fsm: TrafficLightFSM) -> None:
        assert fsm.get_state() == LightState.RED

    def test_initial_state_green(self) -> None:
        f = TrafficLightFSM(initial_state="GREEN")
        f.start()
        assert f.get_state() == LightState.GREEN

    def test_initial_state_yellow(self) -> None:
        f = TrafficLightFSM(initial_state="YELLOW")
        f.start()
        assert f.get_state() == LightState.YELLOW

    def test_not_started_raises(self) -> None:
        f = TrafficLightFSM()
        with pytest.raises(RuntimeError, match="chưa start"):
            f.get_state()

    def test_is_started_property(self) -> None:
        f = TrafficLightFSM()
        assert f.is_started is False
        f.start()
        assert f.is_started is True

    def test_is_red(self, fsm: TrafficLightFSM) -> None:
        assert fsm.is_red() is True

    def test_transition_red_to_green(self, fsm: TrafficLightFSM) -> None:
        """Sau 30s (red_sec) → GREEN."""
        assert fsm_state_at_offset(fsm, 31) == LightState.GREEN

    def test_transition_green_to_yellow(self, fsm: TrafficLightFSM) -> None:
        """Sau 60s (red + green) → YELLOW."""
        assert fsm_state_at_offset(fsm, 61) == LightState.YELLOW

    def test_cycle_wraps_back_to_red(self, fsm: TrafficLightFSM) -> None:
        """Sau 63s (red + green + yellow) → RED lại."""
        assert fsm_state_at_offset(fsm, 64) == LightState.RED

    def test_time_remaining_at_start(self, fsm: TrafficLightFSM) -> None:
        remaining = fsm.get_time_remaining()
        # Vừa start, đang ở RED, nên ~30s remaining
        assert 29 < remaining <= 30

    def test_cycle_duration(self, fsm: TrafficLightFSM) -> None:
        assert fsm.cycle_duration == 63

    def test_set_state(self, fsm: TrafficLightFSM) -> None:
        fsm.set_state(LightState.GREEN)
        assert fsm.get_state() == LightState.GREEN

    def test_reset(self, fsm: TrafficLightFSM) -> None:
        fsm.set_state(LightState.GREEN)
        fsm.reset()
        assert fsm.get_state() == LightState.RED

    def test_get_light_state_returns_string(self, fsm: TrafficLightFSM) -> None:
        assert fsm.get_light_state() == "RED"

    def test_is_violation_when_red(self, fsm: TrafficLightFSM) -> None:
        from rlvds.core.base import Track

        track = Track(track_id=1, bbox=(10, 10, 50, 50))
        assert fsm.is_violation(track) is True

    def test_is_violation_when_green(self, fsm: TrafficLightFSM) -> None:
        from rlvds.core.base import Track

        fsm.set_state(LightState.GREEN)
        track = Track(track_id=1, bbox=(10, 10, 50, 50))
        assert fsm.is_violation(track) is False


# =========================================================================
# test timing.py
# =========================================================================

class TestTiming:
    """Test timing utility functions."""

    def test_frame_to_time(self) -> None:
        assert frame_to_time(0, 30.0) == 0.0
        assert frame_to_time(30, 30.0) == pytest.approx(1.0)
        assert frame_to_time(150, 30.0) == pytest.approx(5.0)

    def test_frame_to_time_invalid_fps(self) -> None:
        with pytest.raises(ValueError, match="fps must be > 0"):
            frame_to_time(10, 0)
        with pytest.raises(ValueError):
            frame_to_time(10, -1)

    def test_get_current_timestamp_format(self) -> None:
        ts = get_current_timestamp()
        # Format: dd/mm/YYYY HH:MM:SS
        assert len(ts) == 19
        assert ts[2] == "/" and ts[5] == "/"

    def test_calculate_video_offset(self) -> None:
        import time

        start = time.time()
        offset = calculate_video_offset(start)
        assert offset >= 0
        assert offset < 1  # Gần 0 vì vừa gọi


# =========================================================================
# test ViolationDetector
# =========================================================================

class TestViolationDetector:
    """Test ``ViolationDetector``."""

    def test_no_violation_when_green(
        self, detector: ViolationDetector, fsm: TrafficLightFSM,
    ) -> None:
        fsm.set_state(LightState.GREEN)
        det = _make_detection(20, 20, 60, 60)
        result = detector.check_frame([det])
        assert result == []

    def test_violation_when_red_in_zone(
        self, detector: ViolationDetector,
    ) -> None:
        # Anchor point = center-bottom = (40, 60) → trong zone [0..100]
        det = _make_detection(20, 20, 60, 60)
        result = detector.check_frame([det])
        assert len(result) == 1
        assert result[0].is_violation is True

    def test_no_violation_when_red_outside_zone(
        self, detector: ViolationDetector,
    ) -> None:
        # Anchor point = (160, 200) → ngoài zone [0..100]
        det = _make_detection(120, 150, 200, 200)
        result = detector.check_frame([det])
        assert result == []

    def test_multiple_detections_mixed(
        self, detector: ViolationDetector,
    ) -> None:
        in_zone = _make_detection(20, 20, 60, 60)     # anchor (40, 60) → in
        out_zone = _make_detection(120, 150, 200, 200) # anchor (160, 200) → out
        result = detector.check_frame([in_zone, out_zone])
        assert len(result) == 1
        assert result[0] is in_zone

    def test_is_duplicate(self, detector: ViolationDetector) -> None:
        assert detector.is_duplicate("29B1-12345") is False
        # Sau khi process_violation, plate được ghi nhận
        det = _make_detection(20, 20, 60, 60)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.process_violation(det, frame, "29B1-12345")
        assert detector.is_duplicate("29B1-12345") is True

    def test_process_violation_returns_violation(
        self, detector: ViolationDetector,
    ) -> None:
        det = _make_detection(20, 20, 60, 60)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        violation = detector.process_violation(det, frame, "30A-99999")
        assert violation is not None
        assert violation.plate_text == "30A-99999"
        assert violation.zone_id == "test"

    def test_process_violation_duplicate_returns_none(
        self, detector: ViolationDetector,
    ) -> None:
        det = _make_detection(20, 20, 60, 60)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.process_violation(det, frame, "51F-11111")
        result = detector.process_violation(det, frame, "51F-11111")
        assert result is None

    def test_clear_recorded_plates(
        self, detector: ViolationDetector,
    ) -> None:
        det = _make_detection(20, 20, 60, 60)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.process_violation(det, frame, "ABC-123")
        assert detector.is_duplicate("ABC-123") is True
        detector.clear_recorded_plates()
        assert detector.is_duplicate("ABC-123") is False

    def test_get_light_state(self, detector: ViolationDetector) -> None:
        assert detector.get_light_state() == LightState.RED

    def test_empty_detections(self, detector: ViolationDetector) -> None:
        result = detector.check_frame([])
        assert result == []


# =========================================================================
# Helper
# =========================================================================

def fsm_state_at_offset(fsm: TrafficLightFSM, offset_sec: float) -> LightState:
    """Giả lập trạng thái FSM sau ``offset_sec`` giây bằng cách dịch start_time."""
    import time

    fsm._start_time = time.time() - offset_sec
    return fsm.get_state()
