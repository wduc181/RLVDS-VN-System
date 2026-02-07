"""
OCR Postprocessing
==================

Mục đích:
    Clean up và validate text biển số sau OCR.

Input:
    - raw_text: str (text từ OCR, có thể có noise)

Output:
    - cleaned_text: str (text đã clean)
    - is_valid: bool (có phải format biển số VN không)

Hàm cần implement:
    1. clean_plate_text(raw_text: str) -> str
       - Xóa ký tự không hợp lệ
       - Uppercase
       - Xóa spaces thừa
    
    2. validate_vn_plate(text: str) -> bool
       - Check format biển số Việt Nam
       - Patterns:
         - Xe máy: XX-YY ZZZZZ
         - Ô tô: XXY-ZZZZZ
    
    3. format_plate(text: str) -> str
       - Format lại cho đẹp khi hiển thị

Common OCR Errors & Fixes:
    - O <-> 0 (letter O vs zero)
    - I <-> 1 (letter I vs one)
    - B <-> 8
    - S <-> 5

TODO:
    [ ] Implement clean_plate_text() với regex
    [ ] Implement validate_vn_plate() patterns
    [ ] Add common OCR error corrections
    [ ] Handle 2-line plates (xe máy)
"""
