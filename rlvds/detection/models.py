"""
Detection Models
================

Mục đích:
    Re-export Detection dataclass từ core/base.py.
    Module này tồn tại để giữ cấu trúc package rõ ràng.

Lưu ý:
    Detection, Track, Violation đã được định nghĩa đầy đủ tại:
        rlvds/core/base.py

    KHÔNG cần tạo lại BoundingBox riêng — Detection đã có:
        - bbox: tuple[x1, y1, x2, y2]
        - get_anchor_point() -> (cx, y2)
        - center() -> (cx, cy)
        - area() -> int
        - crop(frame) -> np.ndarray

TODO:
    [ ] Import và re-export Detection từ core.base
    [ ] Nếu cần thêm dataclass riêng cho detection module, định nghĩa ở đây
    [ ] Ví dụ re-export:
        from rlvds.core.base import Detection, Track, Violation
        __all__ = ["Detection", "Track", "Violation"]
"""
