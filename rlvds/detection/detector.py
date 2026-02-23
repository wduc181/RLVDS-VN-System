"""
License Plate Detector
======================

Mục đích:
    Phát hiện vị trí biển số xe trong frame sử dụng YOLOv5.

Tham chiếu sample code:
    - Xem: .github/sample/camera.py (dòng 18, 60-68)
    - Xem: .github/sample/image.py (dòng 18, 23-25)
    - Xem: .github/sample/utils/helper.py::crop_expanded_plate (dòng 35-56)

Thư viện sử dụng:
    - torch: PyTorch framework
    - yolov5 via torch.hub (hoặc ultralytics): Object detection
    - opencv-python (cv2): Image handling

Input:
    - frame: np.ndarray (BGR image từ OpenCV)
    - confidence_threshold: float (ngưỡng confidence, default 0.5)

Output:
    - list[Detection]: Danh sách các biển số phát hiện được
      (Detection dataclass đã có sẵn tại core/base.py)

Classes cần implement:
    1. LicensePlateDetector(BaseDetector)
       - __init__(model_path: str, device: str = "auto")
         + Load model: torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
         + Lưu ý pathlib fix cho Windows: xem sample camera.py dòng 13-15
       
       - load_model(path: str) -> None
         + Hỗ trợ reload model hot-swap

       - detect(frame: np.ndarray) -> list[Detection]
         + Chạy inference: results = self.model(frame, size=640)
         + Parse results: results.pandas().xyxy[0].values.tolist()
         + Mỗi plate trong list: [x1, y1, x2, y2, confidence, class_id, class_name]
         + Convert sang Detection dataclass (từ core/base.py)
         + Lọc theo confidence_threshold

       - crop_plate(detection: Detection, frame: np.ndarray, expand_ratio: float = 0.15) -> np.ndarray
         + Crop vùng biển số với mở rộng viền
         + Logic từ helper.py::crop_expanded_plate:
           ```python
           width = x2 - x1
           height = y2 - y1
           expand_x = int(expand_ratio * width)
           expand_y = int(expand_ratio * height)
           new_x1 = max(x1 - expand_x, 0)
           new_y1 = max(y1 - expand_y, 0)
           new_x2 = min(x2 + expand_x, img.shape[1])
           new_y2 = min(y2 + expand_y, img.shape[0])
           cropped = img[new_y1:new_y2, new_x1:new_x2, :]
           ```

       - warmup() -> None
         + Chạy inference dummy 1 lần để warm up GPU

Cách load model (tham khảo camera.py dòng 18):
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='weights/lp_vn_det_yolov5n.pt',
                           force_reload=True)

Weights có sẵn:
    - weights/lp_vn_det_yolov5n.pt  (nano - nhanh)
    - weights/lp_vn_det_yolov5s.pt  (small - cân bằng)

TODO:
    [ ] Import torch, numpy, cv2, Detection từ core.base, BaseDetector từ core.base
    [ ] Implement class LicensePlateDetector(BaseDetector)
    [ ] Implement __init__ với torch.hub.load
    [ ] Implement detect() — parse YOLO results thành list[Detection]
    [ ] Implement crop_plate() — crop và expand vùng biển số
    [ ] Implement warmup()
    [ ] Handle lỗi khi model file không tồn tại
    [ ] Test với image.py flow: load model → detect → crop → return
"""
