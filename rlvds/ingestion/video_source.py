"""
Video Source Handler
====================

Mục đích:
    Đọc video từ file hoặc camera stream, trả về từng frame.

Tham chiếu sample code:
    - .github/sample/camera.py (dòng 30-31, 49-51) — VideoCapture + read loop
    - .github/sample/utils/helper.py::set_hd_resolution (dòng 9-18) — resize display

    camera.py:
      vid = cv2.VideoCapture("./test_video/tra.mp4")
      # vid = cv2.VideoCapture(0)  # webcam
      
      while True:
          ret, frame = vid.read()
          if not ret:
              break
          # ... process frame ...
      
      vid.release()
      cv2.destroyAllWindows()

Thư viện sử dụng:
    - opencv-python (cv2): Video capture

===========================================================================
Class cần implement:
===========================================================================

1. VideoSource
   - __init__(source: str | int)
     + source có thể là:
       - Đường dẫn file video: "data/samples/test.mp4"
       - Camera index: 0 (webcam), 1, 2...
     + Mở video: self.cap = cv2.VideoCapture(source)
     + Kiểm tra cap.isOpened(), raise lỗi nếu không mở được

   - __iter__() -> Generator[np.ndarray]
     + Yield từng frame BGR:
       while True:
           ret, frame = self.cap.read()
           if not ret:
               break
           yield frame

   - __enter__ / __exit__
     + Hỗ trợ context manager: with VideoSource("video.mp4") as source:

   - read_frame() -> tuple[bool, np.ndarray | None]
     + Đọc 1 frame: return self.cap.read()

   - get_fps() -> float
     + return self.cap.get(cv2.CAP_PROP_FPS)

   - get_frame_size() -> tuple[int, int]
     + width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     + height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     + return (width, height)

   - get_frame_count() -> int
     + return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

   - release() -> None
     + self.cap.release()
     + cv2.destroyAllWindows()

Ví dụ sử dụng:
    source = VideoSource("data/samples/test.mp4")
    for frame in source:
        detections = detector.detect(frame)
        # ... process ...
    source.release()

    # Hoặc dùng context manager:
    with VideoSource(0) as source:  # webcam
        for frame in source:
            process(frame)

TODO:
    [ ] Import cv2, numpy
    [ ] Implement class VideoSource
    [ ] Implement __init__ với cv2.VideoCapture
    [ ] Implement __iter__ yielding frames
    [ ] Implement __enter__ / __exit__ cho context manager
    [ ] Implement get_fps(), get_frame_size(), get_frame_count()
    [ ] Implement release()
    [ ] Handle lỗi: file không tồn tại, camera không mở được
    [ ] Test: mở video file → đếm frames → so sánh với get_frame_count()
"""
