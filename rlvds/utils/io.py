"""
File I/O Utilities
==================

Mục đích:
    Helper functions cho file operations.

Thư viện sử dụng:
    - os, pathlib: Path operations
    - opencv-python (cv2): Image I/O
    - json, yaml: Config files

Hàm cần implement:
    1. ensure_dir(path: str) -> Path
       - Tạo directory nếu chưa tồn tại
    
    2. save_image(image: np.ndarray, path: str) -> str
       - Lưu image, return full path
    
    3. load_image(path: str) -> np.ndarray
       - Load image từ file
    
    4. save_violation_image(image: np.ndarray, plate_text: str) -> str
       - Lưu ảnh vi phạm với naming convention
       - Example: data/violations/29B1_12345_20240101_120000.jpg
    
    5. load_yaml(path: str) -> dict
       - Load YAML file
    
    6. save_json(data: dict, path: str) -> None
       - Save data to JSON file

TODO:
    [ ] Implement các file operations
    [ ] Add error handling
    [ ] Support relative và absolute paths
"""
