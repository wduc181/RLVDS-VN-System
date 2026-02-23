"""
RLVDS Base Classes
==================

Mục đích:
    Định nghĩa abstract base classes (ABC) cho tất cả các components
    và dataclasses dùng chung trong pipeline.

    Đảm bảo mọi module tuân theo interface thống nhất (Interface
    Segregation) để dễ thay thế implementation sau này.

Thư viện sử dụng:
    - abc: Abstract Base Class
    - dataclasses: Lightweight data containers
    - typing: Type annotations
    - numpy: ndarray cho frame data

Abstract Base Classes:
    1. BaseDetector      – detect / load_model
    2. BaseTracker       – update / reset
    3. BaseSpatialReasoner – is_in_zone / set_zone
    4. BaseTemporalLogic – get_light_state / is_violation
    5. BaseOCR           – recognize / preprocess
    6. BaseRepository    – save / get_all / get_by_id / delete

Dataclasses:
    - Detection  – kết quả detect từ 1 frame
    - Track      – đối tượng đang được theo dõi qua nhiều frame
    - Violation  – bản ghi vi phạm hoàn chỉnh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =========================================================================
# Dataclasses
# =========================================================================

@dataclass
class Detection:
    """Kết quả phát hiện của một đối tượng trong frame.

    Attributes:
        bbox: Bounding box ``(x1, y1, x2, y2)`` theo pixel coords.
        confidence: Độ tin cậy của model, giá trị ``[0, 1]``.
        class_id: ID lớp đối tượng (0 = license_plate mặc định).
        class_name: Tên lớp đối tượng dạng string.
        timestamp: Thời điểm phát hiện (epoch seconds).
        is_violation: Cờ đánh dấu vi phạm (gán bởi logic layer).
    """

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    class_name: str = ""
    timestamp: Optional[float] = None
    is_violation: bool = False

    # -- Derived helpers ---------------------------------------------------

    def get_anchor_point(self) -> Tuple[float, float]:
        """Trả về điểm neo (giữa cạnh dưới bounding box).

        Đây là điểm thường dùng trong ``cv2.pointPolygonTest``
        để kiểm tra xe có nằm trong vùng vi phạm hay không.

        Returns:
            ``(cx, y2)`` — center-x và bottom-y của bbox.
        """
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        return (cx, float(y2))

    def center(self) -> Tuple[float, float]:
        """Trả về tâm bounding box ``(cx, cy)``."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def area(self) -> int:
        """Diện tích bounding box (pixel²)."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Cắt vùng ảnh tương ứng từ frame gốc.

        Args:
            frame: Ảnh gốc ``(H, W, C)`` dạng ``np.ndarray``.

        Returns:
            Vùng ảnh đã crop; mảng rỗng nếu bbox không hợp lệ.
        """
        x1, y1, x2, y2 = self.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2].copy()


@dataclass
class Track:
    """Đối tượng đang được theo dõi qua nhiều frame (tracker output).

    Attributes:
        track_id: Unique ID do tracker gán.
        bbox: Bounding box ``(x1, y1, x2, y2)`` ở frame hiện tại.
        age: Tổng số frame kể từ khi track được tạo.
        hits: Số frame liên tiếp track được match với detection.
        time_since_update: Số frame kể từ lần match cuối.
        state: Trạng thái lifecycle (``tentative`` / ``confirmed`` / ``deleted``).
        velocity: Vector vận tốc ước lượng ``(vx, vy)`` px/frame.
        history: Danh sách bbox qua các frame gần nhất.
    """

    track_id: int
    bbox: Tuple[int, int, int, int]
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    state: str = "tentative"
    velocity: Tuple[float, float] = (0.0, 0.0)
    history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def get_anchor_point(self) -> Tuple[float, float]:
        """Trả về điểm neo (giữa cạnh dưới bbox) — tương tự Detection."""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        return (cx, float(y2))

    def is_moving(self, threshold: float = 2.0) -> bool:
        """Kiểm tra track có đang di chuyển hay không.

        Args:
            threshold: Khoảng dịch pixel tối thiểu để coi là
                       đang di chuyển.

        Returns:
            ``True`` nếu tốc độ vượt ngưỡng.
        """
        vx, vy = self.velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5
        return speed > threshold


@dataclass
class Violation:
    """Bản ghi vi phạm vượt đèn đỏ.

    Attributes:
        plate_text: Chuỗi biển số xe đã OCR.
        timestamp: Thời điểm vi phạm.
        image_path: Đường dẫn ảnh bằng chứng.
        confidence: Độ tin cậy OCR ``[0, 1]``.
        bbox: Bounding box biển số tại thời điểm vi phạm.
        zone_id: ID vùng polygon vi phạm.
        track_id: ID track liên kết (nếu có).
        metadata: Dữ liệu bổ sung (tọa độ, v.v.).
    """

    plate_text: str
    timestamp: datetime
    image_path: str
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    zone_id: str = "default"
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================================
# Abstract Base Classes
# =========================================================================

class BaseDetector(ABC):
    """Interface cho module phát hiện đối tượng (YOLOv5, …)."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Nạp trọng số mô hình từ đường dẫn.

        Args:
            path: Đường dẫn tới file weights (``.pt``).
        """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Chạy inference trên 1 frame.

        Args:
            frame: Ảnh ``(H, W, C)`` dạng BGR (OpenCV).

        Returns:
            Danh sách ``Detection`` tìm được.
        """


class BaseTracker(ABC):
    """Interface cho module theo dõi đa đối tượng (SORT / ByteTrack)."""

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """Cập nhật tracker với detections mới và trả về tracks.

        Args:
            detections: Kết quả detect ở frame hiện tại.

        Returns:
            Danh sách ``Track`` đã cập nhật.
        """

    @abstractmethod
    def reset(self) -> None:
        """Xóa toàn bộ tracks, khởi tạo lại tracker."""


class BaseSpatialReasoner(ABC):
    """Interface cho module suy luận không gian (Point-in-Polygon)."""

    @abstractmethod
    def set_zone(self, polygon: List[Tuple[int, int]]) -> None:
        """Thiết lập vùng đa giác giám sát.

        Args:
            polygon: Danh sách đỉnh ``[(x, y), ...]`` theo thứ tự.
        """

    @abstractmethod
    def is_in_zone(self, point: Tuple[float, float]) -> bool:
        """Kiểm tra một điểm có nằm trong zone hay không.

        Args:
            point: Tọa độ ``(x, y)`` cần kiểm tra.

        Returns:
            ``True`` nếu điểm nằm trong hoặc trên cạnh polygon.
        """


class BaseTemporalLogic(ABC):
    """Interface cho module logic thời gian (Traffic Light FSM)."""

    @abstractmethod
    def get_light_state(self) -> str:
        """Trả về trạng thái đèn hiện tại.

        Returns:
            ``'RED'``, ``'GREEN'``, hoặc ``'YELLOW'``.
        """

    @abstractmethod
    def update(self, elapsed: float) -> None:
        """Cập nhật FSM theo thời gian đã trôi qua.

        Args:
            elapsed: Thời gian (giây) kể từ lần update trước.
        """

    @abstractmethod
    def is_violation(self, track: Track) -> bool:
        """Kiểm tra một track có đang vi phạm hay không.

        Điều kiện: đèn đỏ + track trong zone + track đang di chuyển.

        Args:
            track: Đối tượng ``Track`` cần kiểm tra.

        Returns:
            ``True`` nếu xác nhận vi phạm.
        """


class BaseOCR(ABC):
    """Interface cho module nhận diện ký tự (PaddleOCR, …)."""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        """Nhận diện text từ ảnh biển số đã crop.

        Args:
            image: Ảnh biển số ``(H, W, C)`` dạng BGR.

        Returns:
            Chuỗi biển số đã nhận diện, rỗng nếu thất bại.
        """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh trước khi đưa vào OCR engine.

        Mặc định trả về ảnh gốc. Subclass override để thêm
        grayscale, threshold, resize/padding, …

        Args:
            image: Ảnh biển số gốc.

        Returns:
            Ảnh đã tiền xử lý.
        """
        return image


class BaseRepository(ABC):
    """Interface cho tầng persistence (SQLite repository pattern)."""

    @abstractmethod
    def save(self, entity: Any) -> None:
        """Lưu một entity vào database.

        Args:
            entity: Đối tượng cần lưu (thường là ``Violation``).
        """

    @abstractmethod
    def get_all(self) -> List[Any]:
        """Lấy toàn bộ records.

        Returns:
            Danh sách entities.
        """

    @abstractmethod
    def get_by_id(self, entity_id: int) -> Optional[Any]:
        """Lấy một record theo ID.

        Args:
            entity_id: Primary key.

        Returns:
            Entity nếu tìm thấy, ``None`` nếu không.
        """

    @abstractmethod
    def delete(self, entity_id: int) -> bool:
        """Xóa record theo ID.

        Args:
            entity_id: Primary key.

        Returns:
            ``True`` nếu xóa thành công.
        """
