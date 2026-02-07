"""
License Plate OCR Recognizer
============================

Mục đích:
    Đọc text từ ảnh biển số xe đã crop.

Thư viện sử dụng:
    - paddleocr: PaddleOCR library
    - paddlepaddle: PaddlePaddle framework

Input:
    - license_plate_image: np.ndarray (cropped license plate BGR)

Output:
    - text: str (recognized license plate text)
    - confidence: float

Classes cần implement:
    1. LicensePlateOCR(BaseOCR)
       - __init__(lang: str = "en", use_gpu: bool = True)
       - load_model() -> None
       - recognize(image: np.ndarray) -> tuple[str, float]
       - recognize_batch(images: list[np.ndarray]) -> list[tuple[str, float]]

Cách sử dụng PaddleOCR:
    from paddleocr import PaddleOCR
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    result = ocr.ocr(image, cls=True)
    # result format: [[[box], (text, confidence)], ...]

Vietnamese License Plate Format:
    - Xe máy: XX-YY ZZZZZ (ví dụ: 29-B1 12345)
    - Ô tô: XX-Y ZZZZZ (ví dụ: 30A-12345)

TODO:
    [ ] Initialize PaddleOCR với config phù hợp
    [ ] Implement recognize() method
    [ ] Handle multiple text boxes trong plate
    [ ] Combine text từ nhiều dòng (nếu có)
"""
