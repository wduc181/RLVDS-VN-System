"""
RLVDS Base Classes
==================

Mục đích:
    Định nghĩa abstract base classes (ABC) cho tất cả các components.
    Đảm bảo các module tuân theo interface thống nhất.

Thư viện sử dụng:
    - abc: Abstract Base Class

Classes cần implement:
    1. BaseDetector(ABC)
       - detect(frame: np.ndarray) -> list[Detection]
       - load_model(path: str) -> None
    
    2. BaseTracker(ABC)
       - update(detections: list[Detection]) -> list[Track]
       - reset() -> None
    
    3. BaseSpatialReasoner(ABC)
       - is_in_zone(point: tuple) -> bool
       - set_zone(polygon: list[tuple]) -> None
    
    4. BaseTemporalLogic(ABC)
       - get_light_state() -> str
       - is_violation(track: Track) -> bool
    
    5. BaseOCR(ABC)
       - recognize(image: np.ndarray) -> str
    
    6. BaseRepository(ABC)
       - save(entity) -> None
       - get_all() -> list

Dataclasses cần định nghĩa:
    - Detection: bbox, confidence, class_id
    - Track: track_id, bbox, age, state
    - Violation: plate_text, timestamp, image_path

TODO:
    [ ] Import ABC từ abc module
    [ ] Định nghĩa các abstract methods với @abstractmethod
    [ ] Tạo dataclasses cho Detection, Track, Violation
    [ ] Document input/output types rõ ràng
"""
