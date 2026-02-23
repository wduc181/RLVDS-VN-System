"""
Frame Buffer
=============

Mục đích:
    Buffer frames để xử lý async hoặc skip frames tăng performance.

Lưu ý:
    Sample code (.github/sample/camera.py) KHÔNG dùng frame buffer.
    Module này là optional — implement nếu cần tối ưu performance.

Tham chiếu:
    - config/settings.py → VideoConfig.buffer_size (default=10)
    - config/settings.py → VideoConfig.fps (0 = không giới hạn)

===========================================================================
Class cần implement (optional):
===========================================================================

1. FrameBuffer
   - __init__(max_size: int = 10)
     + Dùng collections.deque(maxlen=max_size)

   - put(frame: np.ndarray) -> None
     + Thêm frame vào buffer

   - get() -> np.ndarray | None
     + Lấy frame mới nhất (hoặc cũ nhất tùy strategy)

   - skip_frames(source: VideoSource, skip: int) -> Generator
     + Chỉ yield mỗi N frame để giảm tải:
       for i, frame in enumerate(source):
           if i % skip == 0:
               yield frame

   - is_full() -> bool
   - clear() -> None

Khi nào dùng:
    - Video FPS cao (60fps) nhưng chỉ cần xử lý 10-15fps
    - Camera realtime + detector chậm → buffer giữ frame mới nhất

TODO:
    [ ] Import collections.deque
    [ ] Implement class FrameBuffer
    [ ] Implement skip_frames()
    [ ] (Optional) Thread-safe buffer với threading.Lock
"""
