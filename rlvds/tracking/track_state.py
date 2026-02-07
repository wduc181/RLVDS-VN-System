"""
Track State Management
======================

Mục đích:
    Quản lý trạng thái và lifecycle của từng track.

Track Lifecycle:
    TENTATIVE -> CONFIRMED -> LOST -> DELETED
    
    - TENTATIVE: Track mới, chưa đủ hits để confirm
    - CONFIRMED: Track đã được xác nhận (min_hits đạt)
    - LOST: Track không match được trong n frames
    - DELETED: Track bị xóa (age > max_age)

Dataclasses/Classes cần implement:
    1. TrackState (Enum)
       - TENTATIVE
       - CONFIRMED
       - LOST
       - DELETED
    
    2. Track
       - track_id: int
       - bbox: BoundingBox
       - state: TrackState
       - age: int (total frames)
       - hits: int (consecutive matches)
       - time_since_update: int (frames since last match)
       
       Methods:
       - update(detection: Detection) -> None
       - predict() -> BoundingBox  # Cho Kalman Filter
       - mark_lost() -> None

TODO:
    [ ] Định nghĩa TrackState enum
    [ ] Implement Track class với state management
    [ ] Add timestamp tracking cho violation detection
"""
