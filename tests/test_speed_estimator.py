from __future__ import annotations

import pytest

from rlvds.core.base import Detection
from rlvds.tracking.speed_estimator import LicensePlateSpeedEstimator


def _det(x1: int, y1: int, x2: int, y2: int) -> Detection:
    return Detection(bbox=(x1, y1, x2, y2), confidence=0.9)


def test_speed_estimator_waits_for_min_track_frames() -> None:
    estimator = LicensePlateSpeedEstimator(
        fps=10.0,
        meters_per_pixel=0.1,
        speed_limit_kmh=50.0,
        min_track_frames=3,
    )

    first = estimator.estimate([_det(10, 10, 60, 40)], frame_idx=1)
    second = estimator.estimate([_det(20, 10, 70, 40)], frame_idx=2)

    assert first[0].speed_kmh is None
    assert second[0].speed_kmh is None
    assert second[0].samples == 2


def test_speed_estimator_reports_zero_for_stationary_plate() -> None:
    estimator = LicensePlateSpeedEstimator(
        fps=10.0,
        meters_per_pixel=0.1,
        speed_limit_kmh=50.0,
        min_track_frames=2,
    )

    estimator.estimate([_det(10, 10, 60, 40)], frame_idx=1)
    out = estimator.estimate([_det(10, 10, 60, 40)], frame_idx=2)

    assert out[0].speed_kmh == pytest.approx(0.0)
    assert out[0].is_speeding is False


def test_speed_estimator_converts_pixel_motion_to_kmh() -> None:
    estimator = LicensePlateSpeedEstimator(
        fps=10.0,
        meters_per_pixel=0.1,
        speed_limit_kmh=30.0,
        min_track_frames=2,
    )

    estimator.estimate([_det(10, 10, 60, 40)], frame_idx=1)
    out = estimator.estimate([_det(20, 10, 70, 40)], frame_idx=2)

    assert out[0].speed_kmh == pytest.approx(36.0)
    assert out[0].is_speeding is True


def test_speed_estimator_accounts_for_frame_delta() -> None:
    estimator = LicensePlateSpeedEstimator(
        fps=30.0,
        meters_per_pixel=0.1,
        speed_limit_kmh=50.0,
        min_track_frames=2,
    )

    estimator.estimate([_det(10, 10, 60, 40)], frame_idx=1)
    out = estimator.estimate([_det(20, 10, 70, 40)], frame_idx=4)

    assert out[0].speed_kmh == pytest.approx(36.0)


def test_speed_estimator_expires_lost_tracks() -> None:
    estimator = LicensePlateSpeedEstimator(
        fps=10.0,
        meters_per_pixel=0.1,
        speed_limit_kmh=50.0,
        min_track_frames=2,
        max_age=1,
    )

    first = estimator.estimate([_det(10, 10, 60, 40)], frame_idx=1)
    estimator.estimate([], frame_idx=3)
    second = estimator.estimate([_det(10, 10, 60, 40)], frame_idx=4)

    assert first[0].track_id == 0
    assert second[0].track_id == 1
    assert second[0].speed_kmh is None
