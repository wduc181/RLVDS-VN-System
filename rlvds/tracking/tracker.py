"""
Object Tracker
==============

Mục đích:
    Theo dõi nhiều biển số xe qua các frame liên tiếp.
    Tránh việc đếm trùng cùng một xe nhiều lần.

Thư viện sử dụng:
    - scipy: Hungarian algorithm cho matching
    - numpy: Matrix operations
    
    Hoặc sử dụng thư viện có sẵn:
    - supervision (pip install supervision)
    - norfair (pip install norfair)

Input:
    - detections: list[Detection] từ detector
    - frame_id: int (optional)

Output:
    - list[Track]: Danh sách tracks đã cập nhật
      - Track.track_id: int (unique ID)
      - Track.bbox: BoundingBox
      - Track.age: int (số frame tồn tại)
      - Track.state: str (tentative/confirmed/lost)

Classes cần implement:
    1. ObjectTracker(BaseTracker)
       - __init__(max_age: int, min_hits: int, iou_threshold: float)
       - update(detections: list[Detection]) -> list[Track]
       - reset() -> None
       - get_active_tracks() -> list[Track]

Thuật toán tracking:
    1. Tính IOU matrix giữa predictions và detections
    2. Hungarian matching để assign detections cho tracks
    3. Update matched tracks
    4. Create new tracks cho unmatched detections
    5. Mark unmatched tracks as lost/delete

TODO:
    [ ] Implement IOU calculation
    [ ] Implement Hungarian matching (scipy.optimize.linear_sum_assignment)
    [ ] Manage track lifecycle (tentative -> confirmed -> lost)
    [ ] Consider using Kalman Filter cho prediction (advanced)
"""
