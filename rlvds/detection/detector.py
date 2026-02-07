"""
License Plate Detector
======================

Mục đích:
    Phát hiện vị trí biển số xe trong frame sử dụng YOLOv5.

Thư viện sử dụng:
    - torch: PyTorch framework
    - yolov5 (torch.hub hoặc ultralytics): Object detection

Input:
    - frame: np.ndarray (BGR image từ OpenCV)
    - confidence_threshold: float (ngưỡng confidence)

Output:
    - list[Detection]: Danh sách các biển số phát hiện được
      - Detection.bbox: tuple[x1, y1, x2, y2]
      - Detection.confidence: float
      - Detection.class_id: int

Classes cần implement:
    1. LicensePlateDetector(BaseDetector)
       - __init__(model_path: str, device: str = "cuda:0")
       - load_model(path: str) -> None
       - detect(frame: np.ndarray) -> list[Detection]
       - warmup() -> None  # Warm up model

Cách load YOLOv5:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
TODO:
    [ ] Load pretrained model cho license plate detection
    [ ] Implement detect() method
    [ ] Handle batch inference (optional)
    [ ] Add warmup để giảm latency lần đầu
    [ ] Crop detected license plate region để pass cho OCR
"""
