"""
Visualization Utilities
=======================

Mục đích:
    Vẽ bounding boxes, text, polygon zones, và annotations lên frame
    để hiển thị kết quả detection/violation trên video output.

Tham chiếu sample code:
    - .github/sample/utils/helper.py::draw_text (dòng 20-26)
    - .github/sample/utils/helper.py::set_hd_resolution (dòng 9-18)
    - .github/sample/camera.py (dòng 59, 70, 77, 122, 124) — drawing calls

Thư viện sử dụng:
    - opencv-python (cv2): Drawing functions

===========================================================================
Hàm cần implement:
===========================================================================

1. draw_text(img: np.ndarray, text: str, pos: tuple = (0, 0),
             font=cv2.FONT_HERSHEY_SIMPLEX, font_scale: float = 1,
             font_thickness: int = 2, text_color: tuple = (255,255,255)) -> None
   
   Từ helper.py dòng 20-26:
     cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
   
   Dùng cho: hiển thị biển số đọc được lên frame (camera.py dòng 77)

2. set_hd_resolution(image: np.ndarray, width: int = 1280) -> np.ndarray
   
   Từ helper.py dòng 9-18:
     height, width_orig, _ = image.shape
     ratio = height / width_orig
     image = cv2.resize(image, (width, int(width * ratio)))
     return image
   
   Dùng cho: resize frame khi hiển thị (camera.py dòng 124)

3. draw_bbox(frame: np.ndarray, bbox: tuple, color: tuple = (0,0,225),
             thickness: int = 1, label: str = None) -> np.ndarray
   
   Từ camera.py dòng 70:
     cv2.rectangle(frame, (x1,y1), (x2,y2), color=color, thickness=thickness)
   
   Nếu có label:
     cv2.putText(frame, label, (x1, y1-10), ...)

4. draw_fps(frame: np.ndarray, fps: int) -> np.ndarray
   
   Từ camera.py dòng 122:
     cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                 (100, 255, 0), 3, cv2.LINE_AA)

5. draw_light_status(frame: np.ndarray, state: str,
                     position: tuple = (50, 150)) -> np.ndarray
   Vẽ trạng thái đèn (RED/GREEN/YELLOW) lên frame:
     color_map = {"RED": (0,0,255), "GREEN": (0,255,0), "YELLOW": (0,255,255)}
     cv2.circle(frame, position, 30, color_map[state], -1)  # filled circle
     cv2.putText(frame, state, (position[0]+40, position[1]+10), ...)

6. draw_violation_alert(frame: np.ndarray, text: str = "VIOLATION",
                        color: tuple = (0, 0, 255)) -> np.ndarray
   Vẽ cảnh báo vi phạm dạng nổi bật lên frame

Color Scheme (BGR — OpenCV dùng BGR, không phải RGB):
    - RED:    (0, 0, 255) — Violation/Alert
    - GREEN:  (0, 255, 0) — Safe/Normal, text biển số
    - YELLOW: (0, 255, 255) — Warning, polygon zone
    - BLUE:   (255, 0, 0) — Detection bbox

TODO:
    [ ] Import cv2, numpy
    [ ] Implement draw_text() — copy logic từ helper.py
    [ ] Implement set_hd_resolution() — copy logic từ helper.py
    [ ] Implement draw_bbox()
    [ ] Implement draw_fps()
    [ ] Implement draw_light_status()
    [ ] Implement draw_violation_alert()
    [ ] Test: tạo blank frame → vẽ bbox + text → verify ảnh output
"""
