"""
Plate Track Cache
==================

Mục đích:
    Quản lý bộ nhớ đệm kết quả OCR theo bounding box.
    Dùng IOU matching để quyết định reuse hay gọi OCR mới.

    Khi một biển số được phát hiện lần đầu → chạy OCR → cache kết quả.
    Các frame sau, nếu bbox mới có IOU >= threshold với entry cũ → reuse text.

Thư viện sử dụng:
    - dataclasses: Lightweight data containers
    - typing: Type annotations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rlvds.tracking.bbox_matcher import compute_iou
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CachedPlate:
    """Một entry trong cache chứa kết quả OCR đã nhận diện.

    Attributes:
        plate_text: Biển số đã OCR (chuỗi đã format).
        bbox: Bounding box lần cuối match ``(x1, y1, x2, y2)``.
        confidence: Độ tin cậy OCR cao nhất đã ghi nhận.
        first_seen_frame: Frame index đầu tiên phát hiện.
        last_seen_frame: Frame index cuối cùng được match.
        ocr_count: Số lần đã gọi OCR cho entry này.
    """

    plate_text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    ocr_count: int = 1


class PlateTrackCache:
    """Bộ nhớ đệm OCR sử dụng IOU matching cho bounding box.

    Cache giúp tránh gọi PaddleOCR lặp lại cho cùng một biển số
    khi nó xuất hiện liên tiếp trong nhiều frame.

    Args:
        iou_threshold: Ngưỡng IOU tối thiểu để match bbox.
        max_size: Số entry tối đa trong cache.
        ttl_frames: Số frame tối đa giữ entry (tự expire).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_size: int = 50,
        ttl_frames: int = 150,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_size = max_size
        self._ttl_frames = ttl_frames
        self._entries: List[CachedPlate] = []

        # Stats
        self._hits: int = 0
        self._misses: int = 0

    def _find_best_match(
        self,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
    ) -> Optional[CachedPlate]:
        """Tìm entry có IOU >= threshold, bỏ qua entry đã hết TTL.

        Hàm nội bộ chỉ làm nhiệm vụ tìm kiếm — **không** thay đổi
        state (bbox, last_seen) hay stats (hits/misses).

        Args:
            bbox: Bounding box cần tìm match ``(x1, y1, x2, y2)``.
            frame_idx: Index của frame hiện tại.

        Returns:
            ``CachedPlate`` nếu tìm thấy match, ``None`` nếu không.
        """
        best_entry: Optional[CachedPlate] = None
        best_iou: float = 0.0

        for entry in self._entries:
            # Skip entry đã hết TTL — tránh "revival bug"
            if (frame_idx - entry.last_seen_frame) > self._ttl_frames:
                continue
            iou = compute_iou(bbox, entry.bbox)
            if iou >= self._iou_threshold and iou > best_iou:
                best_iou = iou
                best_entry = entry

        return best_entry

    def match(
        self,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
    ) -> Optional[CachedPlate]:
        """Tìm entry có IOU >= threshold với bbox hiện tại.

        Cập nhật bbox, last_seen_frame và stats nếu tìm thấy.

        Args:
            bbox: Bounding box cần tìm match ``(x1, y1, x2, y2)``.
            frame_idx: Index của frame hiện tại.

        Returns:
            ``CachedPlate`` nếu tìm thấy match, ``None`` nếu cache miss.
        """
        best_entry = self._find_best_match(bbox, frame_idx)

        if best_entry is not None:
            # Cập nhật bbox và last_seen
            best_entry.bbox = bbox
            best_entry.last_seen_frame = frame_idx
            self._hits += 1
            logger.debug(
                "Cache HIT: plate=%s frame=%d",
                best_entry.plate_text,
                frame_idx,
            )
            return best_entry

        self._misses += 1
        return None

    def add_or_update(
        self,
        bbox: Tuple[int, int, int, int],
        plate_text: str,
        confidence: float,
        frame_idx: int,
    ) -> CachedPlate:
        """Thêm entry mới hoặc cập nhật nếu match IOU.

        Nếu đã có entry match → cập nhật plate_text nếu confidence
        cao hơn. Nếu chưa có → thêm entry mới.

        Args:
            bbox: Bounding box ``(x1, y1, x2, y2)``.
            plate_text: Chuỗi biển số đã OCR.
            confidence: Độ tin cậy OCR.
            frame_idx: Index frame hiện tại.

        Returns:
            ``CachedPlate`` đã thêm/cập nhật.
        """
        # Tìm entry hiện có — dùng _find_best_match để tránh
        # đếm nhầm stats (Bug 2: Stat Inflation)
        existing = self._find_best_match(bbox, frame_idx)
        if existing is not None:
            existing.bbox = bbox
            existing.last_seen_frame = frame_idx
            existing.ocr_count += 1
            # Cập nhật text nếu confidence mới cao hơn
            if confidence > existing.confidence:
                existing.plate_text = plate_text
                existing.confidence = confidence
                logger.debug(
                    "Cache UPDATE: plate=%s conf=%.2f (improved)",
                    plate_text,
                    confidence,
                )
            return existing

        # Kiểm tra max_size — xóa entry cũ nhất nếu đầy
        if len(self._entries) >= self._max_size:
            oldest = min(self._entries, key=lambda e: e.last_seen_frame)
            self._entries.remove(oldest)
            logger.debug("Cache EVICT: plate=%s (max_size reached)", oldest.plate_text)

        entry = CachedPlate(
            plate_text=plate_text,
            bbox=bbox,
            confidence=confidence,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
        )
        self._entries.append(entry)
        logger.debug("Cache ADD: plate=%s conf=%.2f frame=%d", plate_text, confidence, frame_idx)
        return entry

    def cleanup(self, current_frame_idx: int) -> int:
        """Xóa các entry đã hết TTL.

        Args:
            current_frame_idx: Frame index hiện tại.

        Returns:
            Số entry đã bị xóa.
        """
        before = len(self._entries)
        self._entries = [
            e
            for e in self._entries
            if (current_frame_idx - e.last_seen_frame) <= self._ttl_frames
        ]
        removed = before - len(self._entries)
        if removed > 0:
            logger.debug("Cache CLEANUP: removed %d expired entries", removed)
        return removed

    @property
    def size(self) -> int:
        """Số entry hiện có trong cache."""
        return len(self._entries)

    @property
    def hit_count(self) -> int:
        """Tổng số lần cache hit."""
        return self._hits

    @property
    def miss_count(self) -> int:
        """Tổng số lần cache miss."""
        return self._misses

    @property
    def hit_rate(self) -> float:
        """Tỉ lệ cache hit (0.0 – 1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def clear(self) -> None:
        """Xóa toàn bộ cache và reset stats."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        logger.debug("Cache CLEARED")
