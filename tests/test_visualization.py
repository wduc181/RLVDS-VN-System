from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rlvds.core.base import Detection
from rlvds.utils.visualization import draw_detections


def test_draw_detections_renders_speed_warning() -> None:
    frame = np.zeros((180, 260, 3), dtype=np.uint8)
    result = SimpleNamespace(
        detection=Detection(bbox=(40, 70, 140, 110), confidence=0.9),
        plate_text="30A-12345",
        is_violation=False,
        speed_kmh=72.5,
        is_speeding=True,
    )

    out = draw_detections(frame, [result])

    assert out is frame
    assert int(frame.sum()) > 0
