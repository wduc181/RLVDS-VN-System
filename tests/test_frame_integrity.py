from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np

from app import _process_stream_frame
from rlvds.core.base import Detection
from rlvds.core.pipeline import Pipeline


RAW_PIXEL = np.array([10, 20, 30], dtype=np.uint8)


class _OneFrameVideo:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.released = False

    def __iter__(self):
        yield self.frame

    def release(self) -> None:
        self.released = True


class _TrafficLight:
    def get_state(self):
        return SimpleNamespace(value="RED")

    def get_time_remaining(self) -> float:
        return 5.0


class _Detector:
    def __init__(self, detection: Detection) -> None:
        self.detection = detection
        self.calls: list[tuple[np.ndarray, float]] = []

    def is_available(self) -> bool:
        return True

    def crop_plate(
        self,
        detection: Detection,
        frame: np.ndarray,
        expand_ratio: float = 0.15,
    ) -> np.ndarray:
        self.calls.append((frame, expand_ratio))
        x1, y1, x2, y2 = detection.bbox
        pad_x = int((x2 - x1) * expand_ratio)
        pad_y = int((y2 - y1) * expand_ratio)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return frame[y1:y2, x1:x2].copy()


class _Pipeline:
    def __init__(self, detector: _Detector, expand_ratio: float) -> None:
        self.detector = detector
        self.expand_ratio = expand_ratio
        self.ocr_crop: np.ndarray | None = None
        self.process_frame_pixel: np.ndarray | None = None

    def process_frame(self, frame: np.ndarray):
        self.process_frame_pixel = frame[6, 6].copy()
        self.ocr_crop = self.detector.crop_plate(
            self.detector.detection,
            frame,
            expand_ratio=self.expand_ratio,
        )
        return [
            SimpleNamespace(
                plate_text="30A-12345",
                detection=self.detector.detection,
                is_violation=True,
                from_cache=False,
            )
        ]


class _Preprocessor:
    def __init__(self) -> None:
        self.crop: np.ndarray | None = None

    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        self.crop = image.copy()
        return image.copy()


class _Repo:
    def __init__(self) -> None:
        self.frame_pixel: np.ndarray | None = None
        self.raw_plate: np.ndarray | None = None
        self.preprocessed_plate: np.ndarray | None = None

    def record_violation(self, **kwargs):
        self.frame_pixel = kwargs["frame"][6, 6].copy()
        self.raw_plate = kwargs["raw_plate"].copy()
        self.preprocessed_plate = kwargs["preprocessed_plate"].copy()
        return 1


class _Zone:
    def __init__(self) -> None:
        self.zone_id = "test"
        self.polygon = np.array(
            [[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]],
            dtype=np.int32,
        )

    def draw(self, frame: np.ndarray) -> np.ndarray:
        frame[0:20, 0:20] = (0, 0, 255)
        return frame


def _settings(expand_ratio: float = 0.5):
    return SimpleNamespace(
        video=SimpleNamespace(fps=0),
        spatial=SimpleNamespace(zone_color=(0, 0, 255), zone_thickness=1),
        preprocessing=SimpleNamespace(expand_ratio=expand_ratio),
    )


def _frame() -> np.ndarray:
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    frame[4:10, 4:10] = RAW_PIXEL
    return frame


def test_pipeline_run_keeps_ocr_and_persistence_on_raw_expanded_crop() -> None:
    frame = _frame()
    det = Detection(bbox=(5, 5, 9, 9), confidence=0.9)
    detector = _Detector(det)
    settings = _settings()
    processing_pipeline = _Pipeline(detector, settings.preprocessing.expand_ratio)
    repo = _Repo()
    preprocessor = _Preprocessor()

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._cfg = settings
    pipeline.detector = detector
    pipeline._pipeline = processing_pipeline
    pipeline.zone = _Zone()
    pipeline.traffic_light = _TrafficLight()
    pipeline.video_source = _OneFrameVideo(frame)
    pipeline.db = None
    pipeline.repo = repo
    pipeline._preprocessor = preprocessor
    pipeline._running = False
    pipeline._start = MethodType(lambda self, source: None, pipeline)

    pipeline.run("unused", display=False)

    assert np.array_equal(processing_pipeline.process_frame_pixel, RAW_PIXEL)
    assert np.array_equal(repo.frame_pixel, RAW_PIXEL)
    assert processing_pipeline.ocr_crop is not None
    assert preprocessor.crop is not None
    assert repo.raw_plate is not None
    assert np.array_equal(preprocessor.crop, processing_pipeline.ocr_crop)
    assert np.array_equal(repo.raw_plate, processing_pipeline.ocr_crop)
    assert [call[1] for call in detector.calls] == [0.5, 0.5]
    assert np.array_equal(frame[6, 6], RAW_PIXEL)


def test_streamlit_frame_processing_uses_raw_frame_for_pipeline_and_persistence() -> None:
    frame = _frame()
    det = Detection(bbox=(5, 5, 9, 9), confidence=0.9)
    detector = _Detector(det)
    settings = _settings()
    processing_pipeline = _Pipeline(detector, settings.preprocessing.expand_ratio)
    repo = _Repo()
    preprocessor = _Preprocessor()

    display_frame, results, saved, light_state, time_remaining = _process_stream_frame(
        frame=frame,
        zone=_Zone(),
        traffic_light=_TrafficLight(),
        settings=settings,
        pipeline=processing_pipeline,
        detector=detector,
        repo=repo,
        preprocessor=preprocessor,
        show_detection=True,
        detection_available=True,
        show_zone_overlay=True,
        show_fps=False,
        fps=30.0,
    )

    assert len(results) == 1
    assert saved == 1
    assert light_state == "RED"
    assert time_remaining == 5.0
    assert np.array_equal(processing_pipeline.process_frame_pixel, RAW_PIXEL)
    assert np.array_equal(repo.frame_pixel, RAW_PIXEL)
    assert processing_pipeline.ocr_crop is not None
    assert repo.raw_plate is not None
    assert np.array_equal(repo.raw_plate, processing_pipeline.ocr_crop)
    assert [call[1] for call in detector.calls] == [0.5, 0.5]
    assert not np.array_equal(display_frame[6, 6], RAW_PIXEL)
