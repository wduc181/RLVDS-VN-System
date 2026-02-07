"""
Frame Buffer
=============

Mục đích:
    Buffer frames cho smooth processing và preprocessing.

Thư viện sử dụng:
    - opencv-python (cv2): Image processing
    - numpy: Array operations

Input:
    - frames từ VideoSource
    - preprocessing config (resize, normalize, etc.)

Output:
    - Processed frames ready for detection

Classes cần implement:
    1. FrameBuffer
       - __init__(buffer_size: int = 30)
       - add(frame: np.ndarray) -> None
       - get_latest() -> np.ndarray | None
       - preprocess(frame: np.ndarray) -> np.ndarray

Preprocessing steps:
    1. Resize to model input size (nếu cần)
    2. Color space conversion (BGR -> RGB nếu model yêu cầu)
    3. Normalization

TODO:
    [ ] Implement circular buffer với collections.deque
    [ ] Preprocessing cho YOLOv5 input
    [ ] Thread-safe nếu dùng multi-threading
"""
