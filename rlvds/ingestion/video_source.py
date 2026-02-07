"""
Video Source Handler
====================

Mục đích:
    Đọc video từ file hoặc camera stream.

Thư viện sử dụng:
    - opencv-python (cv2): Video capture

Input:
    - source: str | int
      - Đường dẫn file video (ví dụ: "data/samples/test.mp4")
      - Camera ID (ví dụ: 0 cho webcam)

Output:
    - Generator[np.ndarray]: Từng frame BGR

Classes cần implement:
    1. VideoSource
       - __init__(source: str | int)
       - __iter__() -> Generator[np.ndarray]
       - get_fps() -> float
       - get_frame_size() -> tuple[int, int]
       - release() -> None

Ví dụ sử dụng:
    source = VideoSource("video.mp4")
    for frame in source:
        process(frame)
    source.release()

TODO:
    [ ] Sử dụng cv2.VideoCapture
    [ ] Xử lý lỗi khi không mở được video
    [ ] Hỗ trợ resize frame nếu cần
    [ ] Hỗ trợ skip frames để tăng performance
"""
