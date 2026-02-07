"""
Detection Models
================

Mục đích:
    Định nghĩa dataclasses cho detection results.

Thư viện sử dụng:
    - dataclasses: Python dataclasses

Dataclasses cần định nghĩa:
    1. BoundingBox
       - x1: int (top-left x)
       - y1: int (top-left y)
       - x2: int (bottom-right x)
       - y2: int (bottom-right y)
       
       Methods:
       - center() -> tuple[int, int]
       - area() -> int
       - crop(image: np.ndarray) -> np.ndarray
    
    2. Detection
       - bbox: BoundingBox
       - confidence: float
       - class_id: int = 0
       - timestamp: float = None

TODO:
    [ ] Sử dụng @dataclass decorator
    [ ] Implement helper methods cho BoundingBox
    [ ] Add __repr__ cho debug logging
"""
