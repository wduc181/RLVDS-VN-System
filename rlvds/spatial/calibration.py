"""
Camera Calibration (Optional/Advanced)
======================================

Mục đích:
    Chuyển đổi tọa độ pixel sang tọa độ thực tế.
    Dùng cho advanced use cases.

Thư viện sử dụng:
    - opencv-python (cv2): Camera calibration

Input:
    - camera_matrix: np.ndarray (intrinsic matrix)
    - distortion_coeffs: np.ndarray
    - homography_matrix: np.ndarray (pixel to world)

Output:
    - Real-world coordinates

Classes cần implement:
    1. CameraCalibrator
       - __init__()
       - calibrate_from_points(image_points, world_points) -> None
       - pixel_to_world(pixel_point: tuple) -> tuple
       - world_to_pixel(world_point: tuple) -> tuple

TODO:
    [ ] (Optional) Implement nếu cần tọa độ thực
    [ ] Đây là advanced feature, có thể skip cho MVP
"""
