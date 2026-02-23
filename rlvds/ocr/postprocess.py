"""
OCR Postprocessing & Image Preprocessing
=========================================

Mục đích:
    1. Tiền xử lý ảnh biển số trước khi đưa vào OCR
    2. Clean up và validate text biển số sau OCR

Tham chiếu sample code:
    - .github/sample/utils/helper.py (dòng 167-198) — image preprocessing
    - .github/sample/utils/helper.py (dòng 69-91)  — check_valid_plate
    - .github/sample/clean_data.py                   — data cleaning logic

Thư viện sử dụng:
    - opencv-python (cv2): Image processing
    - numpy: Array operations

===========================================================================
IMAGE PREPROCESSING (từ helper.py dòng 167-198)
===========================================================================

Hàm cần implement:

1. upscale_image(image: np.ndarray, scale: float = 2.0) -> np.ndarray
   Logic:
     height, width = image.shape[:2]
     new_dimensions = (int(width * scale), int(height * scale))
     upscaled = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
     return upscaled

2. denoise_image(image: np.ndarray) -> np.ndarray
   Logic:
     - Nếu ảnh màu (3 channels): convert sang grayscale
     - Apply: cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
     - Return denoised image

3. adjust_contrast(image: np.ndarray) -> np.ndarray
   Logic:
     - Nếu ảnh màu: convert sang grayscale
     - Apply CLAHE: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
     - Return contrast-adjusted image

4. preprocess_image(image: np.ndarray) -> np.ndarray
   Pipeline tổng hợp — gọi theo thứ tự:
     upscaled = upscale_image(image)
     denoised = denoise_image(upscaled)
     contrast = adjust_contrast(denoised)
     return contrast

===========================================================================
PLATE VALIDATION (từ helper.py dòng 69-91)
===========================================================================

5. check_valid_plate(plate: str) -> bool
   Logic kiểm tra format biển số Việt Nam:
     - len(plate) <= 7: return False
     - Split bằng '-': parts = plate.split('-')
     - len(parts) <= 1 hoặc len(parts[0]) < 2: return False
     - 2 ký tự đầu phải là số
     - Không thuộc mã tỉnh không tồn tại: ["13","42","44","45","46","87","91","96"]
     
     Nếu 2 phần (ô tô: "XXY-ZZZZZ"):
       - parts[0] phải có 3 ký tự, ký tự thứ 3 là chữ cái
     
     Nếu 3 phần (xe máy: "XX-YY-ZZZZZ"):
       - parts[0] phải đúng 2 ký tự
     
     Phần cuối (số):
       - Nếu 4 ký tự: tất cả phải là digit
       - Nếu 6 ký tự và ký tự thứ 4 != '.': tất cả (trừ vị trí 3) phải là digit
       - Nếu < 4 hoặc > 6: return False

===========================================================================
TEXT CLEANUP
===========================================================================

6. clean_plate_text(raw_text: str) -> str
   - Xóa ký tự không hợp lệ
   - Uppercase toàn bộ
   - Xóa spaces thừa
   - Fix lỗi OCR phổ biến:
     + O <-> 0 (letter O vs zero) — tùy vị trí
     + B <-> 8 (ở vị trí chữ cái series)
     + G <-> 6 (ở vị trí chữ cái series)
     + I <-> 1
     + S <-> 5

7. format_plate(text: str) -> str
   - Format lại cho đẹp khi hiển thị
   - Ví dụ: "29B112345" -> "29B1-12345"

TODO:
    [ ] Import cv2, numpy
    [ ] Implement upscale_image()
    [ ] Implement denoise_image()
    [ ] Implement adjust_contrast()
    [ ] Implement preprocess_image() — pipeline
    [ ] Implement check_valid_plate() — copy logic từ helper.py
    [ ] Implement clean_plate_text()
    [ ] Implement format_plate()
    [ ] Test: ảnh biển số raw → preprocess → kiểm tra output grayscale, kích thước đúng
"""
