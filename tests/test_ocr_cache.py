"""Tests for bbox_matcher, plate_cache, and cached_pipeline modules."""

from __future__ import annotations

import numpy as np
import pytest

from rlvds.core.base import Detection
from rlvds.core.cached_pipeline import CachedPipeline, CachedPipelineResult
from rlvds.ocr.plate_cache import CachedPlate, PlateTrackCache
from rlvds.ocr.recognizer import OCRResult
from rlvds.spatial.zones import ViolationZone
from rlvds.temporal.traffic_light import TrafficLightFSM
from rlvds.temporal.violation import ViolationDetector
from rlvds.tracking.bbox_matcher import compute_iou


# =========================================================================
# IOU Tests
# =========================================================================

class TestComputeIOU:
    """Tests for compute_iou function."""

    def test_perfect_overlap(self) -> None:
        """Two identical boxes have IOU = 1.0."""
        iou = compute_iou((0, 0, 100, 100), (0, 0, 100, 100))
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Two non-overlapping boxes have IOU = 0.0."""
        iou = compute_iou((0, 0, 50, 50), (100, 100, 200, 200))
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Two partially overlapping boxes have expected IOU."""
        # box_a: (0,0)-(100,100) area=10000
        # box_b: (50,50)-(150,150) area=10000
        # intersection: (50,50)-(100,100) area=2500
        # union: 10000+10000-2500=17500
        # IOU = 2500/17500 ≈ 0.1429
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert iou == pytest.approx(2500.0 / 17500.0, rel=1e-4)

    def test_one_inside_other(self) -> None:
        """Inner box fully inside outer box."""
        # inner: (25,25)-(75,75) area=2500
        # outer: (0,0)-(100,100) area=10000
        # intersection=2500, union=10000, IOU=0.25
        iou = compute_iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert iou == pytest.approx(0.25)

    def test_touching_edge(self) -> None:
        """Two boxes touching at edge have IOU = 0.0."""
        iou = compute_iou((0, 0, 50, 50), (50, 0, 100, 50))
        assert iou == pytest.approx(0.0)

    def test_slight_displacement(self) -> None:
        """Small displacement still gives decent IOU (simulating frame-to-frame)."""
        # box_a: (100,100)-(200,200) area=10000
        # box_b: (110,105)-(210,205) area=10000
        # intersection: (110,105)-(200,200) = 90*95 = 8550
        # union: 10000+10000-8550 = 11450
        iou = compute_iou((100, 100, 200, 200), (110, 105, 210, 205))
        assert iou == pytest.approx(8550.0 / 11450.0, rel=1e-4)
        assert iou > 0.3  # Should be above default threshold


# =========================================================================
# PlateTrackCache Tests
# =========================================================================

class TestPlateTrackCache:
    """Tests for PlateTrackCache."""

    def test_add_and_match(self) -> None:
        """Added entry can be found via matching bbox."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=100)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.9, frame_idx=1)

        # Same bbox → should match
        result = cache.match((100, 100, 200, 200), frame_idx=2)
        assert result is not None
        assert result.plate_text == "30A-12345"

    def test_match_with_displacement(self) -> None:
        """Match works with slight bbox displacement (IOU > threshold)."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=100)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.9, frame_idx=1)

        # Slight displacement — IOU still > 0.3
        result = cache.match((110, 105, 210, 205), frame_idx=2)
        assert result is not None
        assert result.plate_text == "30A-12345"

    def test_miss_low_iou(self) -> None:
        """No match when bbox has low IOU (far displacement)."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=100)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.9, frame_idx=1)

        # Far away box → IOU < 0.3
        result = cache.match((500, 500, 600, 600), frame_idx=2)
        assert result is None

    def test_cleanup_ttl(self) -> None:
        """Entries expire after TTL frames."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=10)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.9, frame_idx=1)

        # Before TTL → still present
        assert cache.size == 1
        cache.cleanup(current_frame_idx=10)
        assert cache.size == 1

        # After TTL → expired
        cache.cleanup(current_frame_idx=12)
        assert cache.size == 0

    def test_max_size_eviction(self) -> None:
        """Cache evicts oldest entry when max_size exceeded."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=2, ttl_frames=100)
        cache.add_or_update((0, 0, 50, 50), "PLATE-1", 0.9, frame_idx=1)
        cache.add_or_update((200, 200, 300, 300), "PLATE-2", 0.9, frame_idx=2)
        assert cache.size == 2

        # Adding 3rd should evict PLATE-1 (oldest)
        cache.add_or_update((500, 500, 600, 600), "PLATE-3", 0.9, frame_idx=3)
        assert cache.size == 2

        # PLATE-1 should be gone
        result = cache.match((0, 0, 50, 50), frame_idx=4)
        assert result is None

    def test_update_better_confidence(self) -> None:
        """Higher confidence OCR updates the cached plate text."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=100)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.7, frame_idx=1)
        cache.add_or_update((100, 100, 200, 200), "30A-12346", 0.95, frame_idx=2)

        result = cache.match((100, 100, 200, 200), frame_idx=3)
        assert result is not None
        assert result.plate_text == "30A-12346"
        assert result.confidence == pytest.approx(0.95)

    def test_hit_miss_stats(self) -> None:
        """Hit/miss counters track correctly."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=100)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.9, frame_idx=1)

        cache.match((100, 100, 200, 200), frame_idx=2)  # hit
        cache.match((500, 500, 600, 600), frame_idx=3)  # miss

        assert cache.hit_count == 1
        assert cache.miss_count == 1
        assert 0 < cache.hit_rate < 1.0

    def test_expired_entry_not_revived(self) -> None:
        """Entry hết TTL không match dù IOU cao (Bug 1 fix)."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=5)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.9, frame_idx=1)

        # Frame 10: entry đã hết TTL (last_seen=1, ttl=5, 10-1=9 > 5)
        result = cache.match((100, 100, 200, 200), frame_idx=10)
        assert result is None, "Expired entry should NOT be revived"

    def test_add_or_update_no_stat_inflation(self) -> None:
        """add_or_update() không tăng hit/miss counters (Bug 2 fix)."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=100)

        # add_or_update lần 1: thêm mới
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.7, frame_idx=1)
        assert cache.hit_count == 0
        assert cache.miss_count == 0

        # add_or_update lần 2: cập nhật (same bbox)
        cache.add_or_update((100, 100, 200, 200), "30A-12346", 0.95, frame_idx=2)
        assert cache.hit_count == 0, "add_or_update should NOT inflate hit count"
        assert cache.miss_count == 0, "add_or_update should NOT inflate miss count"

    def test_clear(self) -> None:
        """Cache.clear() removes all entries and resets stats."""
        cache = PlateTrackCache(iou_threshold=0.3, max_size=10, ttl_frames=100)
        cache.add_or_update((100, 100, 200, 200), "30A-12345", 0.9, frame_idx=1)
        cache.clear()
        assert cache.size == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0


# =========================================================================
# CachedPipeline Tests
# =========================================================================

class _FakePaddle:
    """Fake PaddleOCR engine for testing."""

    def __init__(self, result):
        self._result = result

    def ocr(self, _image):
        return self._result


class _FakeDetector:
    """Fake detector returning fixed detections."""

    def __init__(self, detections=None):
        self._detections = detections or [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9)
        ]

    def detect(self, _frame):
        return list(self._detections)

    def crop_plate(self, detection, frame, expand_ratio=0.15):
        x1, y1, x2, y2 = detection.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2].copy()


class _CountingOCR:
    """OCR that counts how many times recognize() is called."""

    def __init__(self, plate_text: str = "30A-12345", confidence: float = 0.92):
        self._plate_text = plate_text
        self._confidence = confidence
        self.call_count = 0

    def recognize(self, _image):
        self.call_count += 1
        return self._plate_text

    def recognize_with_confidence(self, _image) -> OCRResult:
        self.call_count += 1
        return OCRResult(text=self._plate_text, confidence=self._confidence)


def _make_pipeline(
    ocr=None,
    detector=None,
    iou_threshold=0.3,
    ttl_frames=100,
    ocr_quality_frames=1,
):
    """Helper to create a CachedPipeline with test components."""
    zone = ViolationZone(vertices=[[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    fsm = TrafficLightFSM(red_sec=30, green_sec=30, yellow_sec=3, initial_state="RED")
    fsm.start()
    violation_detector = ViolationDetector(zone=zone, traffic_light=fsm)

    cache = PlateTrackCache(
        iou_threshold=iou_threshold,
        max_size=50,
        ttl_frames=ttl_frames,
    )

    return CachedPipeline(
        detector=detector or _FakeDetector(),
        ocr=ocr or _CountingOCR(),
        violation_detector=violation_detector,
        cache=cache,
        ocr_quality_frames=ocr_quality_frames,
    )


class TestCachedPipeline:
    """Tests for CachedPipeline."""

    def test_first_frame_calls_ocr(self) -> None:
        """First detection always triggers OCR (cache miss)."""
        ocr = _CountingOCR()
        pipeline = _make_pipeline(ocr=ocr)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        results = pipeline.process_frame(frame)
        assert len(results) == 1
        assert results[0].plate_text == "30A-12345"
        assert results[0].from_cache is False
        assert ocr.call_count == 1

    def test_second_frame_skips_ocr(self) -> None:
        """Same bbox in second frame uses cache (skip OCR)."""
        ocr = _CountingOCR()
        pipeline = _make_pipeline(ocr=ocr, ocr_quality_frames=1)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        # Frame 1: OCR runs
        pipeline.process_frame(frame)
        assert ocr.call_count == 1

        # Frame 2: Same detector → same bbox → cache hit
        results = pipeline.process_frame(frame)
        assert len(results) == 1
        assert results[0].plate_text == "30A-12345"
        assert results[0].from_cache is True
        assert ocr.call_count == 1  # No additional OCR call

    def test_many_frames_only_one_ocr(self) -> None:
        """Over many frames with same bbox, OCR is called only once."""
        ocr = _CountingOCR()
        pipeline = _make_pipeline(ocr=ocr, ocr_quality_frames=1)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        for _ in range(20):
            pipeline.process_frame(frame)

        assert ocr.call_count == 1  # Only first frame triggers OCR

    def test_different_bbox_triggers_ocr(self) -> None:
        """New plate (different bbox) triggers fresh OCR."""
        ocr = _CountingOCR()
        det1 = [Detection(bbox=(100, 100, 200, 200), confidence=0.9)]
        det2 = [Detection(bbox=(500, 500, 600, 600), confidence=0.85)]

        pipeline = _make_pipeline(ocr=ocr, ocr_quality_frames=1)
        frame = np.ones((700, 700, 3), dtype=np.uint8) * 128

        # Frame 1 with bbox1
        pipeline._detector = _FakeDetector(det1)
        pipeline.process_frame(frame)
        assert ocr.call_count == 1

        # Frame 2 with different bbox → cache miss → OCR
        pipeline._detector = _FakeDetector(det2)
        pipeline.process_frame(frame)
        assert ocr.call_count == 2

    def test_quality_frames_runs_extra_ocr(self) -> None:
        """ocr_quality_frames > 1 runs OCR multiple times for better accuracy."""
        ocr = _CountingOCR()
        pipeline = _make_pipeline(ocr=ocr, ocr_quality_frames=3)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        # First 3 frames should call OCR (quality improvement)
        for _ in range(3):
            pipeline.process_frame(frame)
        assert ocr.call_count == 3

        # Frame 4+ should reuse cache
        pipeline.process_frame(frame)
        assert ocr.call_count == 3

    def test_quality_frames_from_cache_false(self) -> None:
        """Quality-frames branch returns from_cache=False (Bug 4 fix)."""
        ocr = _CountingOCR()
        pipeline = _make_pipeline(ocr=ocr, ocr_quality_frames=3)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        # Frame 1: cache miss → from_cache=False
        results = pipeline.process_frame(frame)
        assert results[0].from_cache is False

        # Frame 2: cache hit nhưng ocr_count < 3 → chạy OCR → from_cache=False
        results = pipeline.process_frame(frame)
        assert results[0].from_cache is False, "Quality OCR branch should NOT be from_cache"

        # Frame 3: vẫn chạy OCR quality
        results = pipeline.process_frame(frame)
        assert results[0].from_cache is False

        # Frame 4: đủ quality → reuse → from_cache=True
        results = pipeline.process_frame(frame)
        assert results[0].from_cache is True

    def test_ocr_confidence_used_not_detection(self) -> None:
        """Cache stores OCR confidence, not YOLO detection confidence (Bug 3 fix)."""
        ocr = _CountingOCR(plate_text="30A-99999", confidence=0.88)
        pipeline = _make_pipeline(ocr=ocr, ocr_quality_frames=1)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        pipeline.process_frame(frame)
        # Detection confidence is 0.9 (from _FakeDetector),
        # but cache should store OCR confidence 0.88
        cached = pipeline.cache.match((100, 100, 200, 200), pipeline.frame_idx + 1)
        assert cached is not None
        assert cached.confidence == pytest.approx(0.88), (
            "Cache should store OCR confidence, not detection confidence"
        )

    def test_unknown_ocr_stops_after_quality_frames(self) -> None:
        """OCR returning 'unknown' still increments ocr_count (no infinite loop)."""
        ocr = _CountingOCR(plate_text="unknown", confidence=0.0)
        pipeline = _make_pipeline(ocr=ocr, ocr_quality_frames=3)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        # Frame 1: cache miss → OCR → "unknown" → no cache entry added
        pipeline.process_frame(frame)
        assert ocr.call_count == 1

        # Manually seed a cache entry (simulating a prior valid read)
        # ocr_count starts at 1 (default from add_or_update)
        pipeline.cache.add_or_update(
            bbox=(100, 100, 200, 200),
            plate_text="30A-12345",
            confidence=0.8,
            frame_idx=pipeline.frame_idx,
        )

        # Frames 2-3: cache hit + quality branch (ocr_count 1→2→3)
        # OCR returns "unknown" but ocr_count still increments
        for _ in range(2):
            pipeline.process_frame(frame)
        assert ocr.call_count == 3  # 1 (miss) + 2 (quality)

        # Frame 4+: ocr_count >= 3 → skip OCR entirely
        before = ocr.call_count
        pipeline.process_frame(frame)
        pipeline.process_frame(frame)
        assert ocr.call_count == before, "OCR should NOT be called after quality frames exhausted"

    def test_reset_clears_state(self) -> None:
        """Pipeline reset clears cache and frame counter."""
        ocr = _CountingOCR()
        pipeline = _make_pipeline(ocr=ocr)
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 128

        pipeline.process_frame(frame)
        assert pipeline.frame_idx == 1
        assert pipeline.cache.size == 1

        pipeline.reset()
        assert pipeline.frame_idx == 0
        assert pipeline.cache.size == 0
