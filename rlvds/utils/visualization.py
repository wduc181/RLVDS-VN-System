"""
Visualization Utilities
=======================

Mục đích:
    Vẽ bounding boxes, zones, và annotations lên frame.

Thư viện sử dụng:
    - opencv-python (cv2): Drawing functions

Input:
    - frame: np.ndarray (BGR image)
    - detections, tracks, zones, etc.

Output:
    - annotated_frame: np.ndarray

Hàm cần implement:
    1. draw_bbox(frame, bbox, color, label) -> np.ndarray
       - Vẽ bounding box với label
    
    2. draw_polygon(frame, vertices, color, fill) -> np.ndarray
       - Vẽ polygon (violation zone)
    
    3. draw_track(frame, track, color) -> np.ndarray
       - Vẽ track với ID
    
    4. draw_light_status(frame, state, position) -> np.ndarray
       - Vẽ trạng thái đèn
    
    5. draw_violation_alert(frame, violation) -> np.ndarray
       - Vẽ cảnh báo vi phạm

Color Scheme (BGR):
    - RED: (0, 0, 255) - Violation/Alert
    - GREEN: (0, 255, 0) - Safe/Normal
    - YELLOW: (0, 255, 255) - Warning
    - BLUE: (255, 0, 0) - Detection

TODO:
    [ ] Implement các hàm draw_*
    [ ] Add transparency support
    [ ] Add text với font đẹp (cv2.putText)
    [ ] Support dark mode overlay
"""
