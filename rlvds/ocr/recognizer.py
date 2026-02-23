"""
License Plate OCR Recognizer
============================

Mục đích:
    Đọc text từ ảnh biển số xe đã crop.
    Hỗ trợ 2 phương pháp OCR:
      1. PaddleOCR (ppOCRv4) — phương pháp chính
      2. YOLOv5 character detection — phương pháp dự phòng

Tham chiếu sample code:
    - Phương pháp 1: .github/sample/utils/helper.py::read_plate_ppocr (dòng 95-131)
    - Phương pháp 2: .github/sample/utils/helper.py::read_plate (dòng 133-165)

Thư viện sử dụng:
    - paddleocr: PaddleOCR library (cho PP1)
    - torch: PyTorch (cho PP2 - YOLOv5 char detect)

===========================================================================
PHƯƠNG PHÁP 1: PaddleOCR (read_plate_ppocr)
===========================================================================

Input:
    - plate_image: np.ndarray (ảnh biển số đã crop + preprocess)

Output:
    - text: str (biển số đã đọc, "unknown" nếu thất bại)

Logic chính (từ helper.py dòng 97-131):
    1. Chạy OCR:          result = ocr.ocr(plate_image)[0]
    2. Nếu result None:   return "unknown"
    3. Duyệt từng box:
       - Lấy score = r[1][1]
       - Nếu score > 80%:  nối text (box đầu tiên không có "-", các box sau thêm "-")
       - Nếu score <= 80%: return "unknown"
    4. Post-process text:
       - Xóa "???"
       - Thay "O" thành "0"
       - Fix lỗi OCR phổ biến:
         + Vị trí text[2] == '8' → đổi thành 'B'
         + Vị trí text[2] == '6' → đổi thành 'G'
         + Tương tự cho vị trí text[3] nếu text[2] == '-'
    5. Return text

===========================================================================
PHƯƠNG PHÁP 2: YOLOv5 Character Detection (read_plate)
===========================================================================

Cần thêm model weights: weights/lp_vn_ocr_yolov5s_final.pt

Input:
    - yolo_license_plate: YOLOv5 model (đã load)
    - im: np.ndarray (ảnh biển số)

Output:
    - text: str (biển số đã đọc, "unknown" nếu thất bại)

Logic chính (từ helper.py dòng 133-165):
    1. Detect từng ký tự:   results = yolo_license_plate(im)
    2. Lấy bounding box:    bb_list = results.pandas().xyxy[0].values.tolist()
    3. Nếu số ký tự < 7 hoặc > 10: return "unknown"
    4. Tính center mỗi box: [(x1+x2)/2, (y1+y2)/2, class_name]
    5. Phân loại biển số 1 dòng hay 2 dòng:
       - Tìm điểm trái nhất (l_point) và phải nhất (r_point)
       - Kiểm tra các điểm có nằm trên 1 đường thẳng không
       - Nếu có điểm lệch → LP_type = "2" (2 dòng, xe máy)
    6. Ghép ký tự theo thứ tự x:
       - LP 1 dòng: sort all by x → nối text
       - LP 2 dòng: split line_1 (y <= y_mean) + line_2 (y > y_mean)
                     → sort mỗi line by x → nối bằng "-"

Hàm phụ trợ (cũng từ helper.py):
    - linear_equation(x1,y1,x2,y2):    Tính a, b của y = ax + b
    - check_point_linear(x,y,...):      Kiểm tra điểm nằm trên đường thẳng (abs_tol=3)

===========================================================================
Classes cần implement:
===========================================================================

1. LicensePlateOCR(BaseOCR)
   - __init__(lang="en", use_gpu=True)
     + Init PaddleOCR: ocr = PaddleOCR(lang=lang)
   
   - recognize(image: np.ndarray) -> str
     + Gọi read_plate_ppocr logic
     + Trả về biển số string hoặc "unknown"

   - preprocess(image: np.ndarray) -> np.ndarray
     + Gọi preprocess_image() từ ocr/postprocess.py
     + Trước khi recognize

2. YOLOv5CharOCR(BaseOCR)  [Phương pháp dự phòng]
   - __init__(model_path: str)
     + Load YOLOv5 OCR model
   
   - recognize(image: np.ndarray) -> str
     + Gọi read_plate logic

Vietnamese License Plate Format:
    - Xe máy (2 dòng): XX-YY ZZZZZ  (ví dụ: 29-B1 12345)
    - Ô tô (1 dòng):   XXY-ZZZZZ   (ví dụ: 30A-12345)

TODO:
    [ ] Import PaddleOCR, numpy, BaseOCR từ core.base
    [ ] Implement class LicensePlateOCR(BaseOCR) — ppOCR method
    [ ] Implement recognize() với logic read_plate_ppocr
    [ ] Implement post-process: fix O→0, 8→B, 6→G
    [ ] Implement class YOLOv5CharOCR(BaseOCR) — char detect method
    [ ] Implement recognize() với logic read_plate
    [ ] Implement linear_equation() và check_point_linear() helpers
    [ ] Handle 2-line plates (xe máy) vs 1-line plates (ô tô)
    [ ] Test: crop_plate → preprocess → recognize → return text
"""
