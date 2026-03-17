"""
RLVDS OCR Preprocessor Module
==============================

Mục đích:
    Cung cấp class ``PlatePreprocessor`` thực hiện pipeline tiền xử lý
    ảnh biển số xe trước khi đưa vào OCR engine (PaddleOCR / YOLOv5CharOCR).

Pipeline:
    frame + Detection
        → crop_plate_region (bbox expand + clip)
        → upscale           (INTER_CUBIC × scale)
        → denoise           (fastNlMeansDenoising, grayscale)
        → apply_clahe       (CLAHE local contrast enhancement)
        → grayscale image   → sẵn sàng cho OCR

Cách sử dụng:
    Gọi ``run(frame, detection)`` khi có frame gốc + Detection từ YOLO::

        preprocessor = PlatePreprocessor(settings.preprocessing)
        enhanced = preprocessor.run(frame, detection)
        text = ocr_engine.recognize(enhanced)

    Hoặc ``run_pipeline(cropped_image)`` khi đã có ảnh biển số riêng::

        enhanced = preprocessor.run_pipeline(cropped_bgr)

Thư viện sử dụng:
    - OpenCV (cv2): resize, denoising, CLAHE, color conversion
    - NumPy: ndarray operations
    - Pydantic: PreprocessingConfig (type-safe parameters)

Ghi chú:
    Tất cả tham số xử lý đều được đọc từ ``PreprocessingConfig``
    (config/settings.py) — không hardcode bất kỳ giá trị số nào.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config.settings import PreprocessingConfig
from rlvds.core.base import Detection
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class PlatePreprocessor:
    """Tiền xử lý ảnh biển số trước khi OCR.

    Thực hiện pipeline 4 bước có thể cấu hình:

    1. **crop_plate_region** — Cắt ROI từ frame gốc, mở rộng bbox
       theo ``expand_ratio`` để tránh cắt mất ký tự ở rìa.
    2. **upscale** — Phóng to ảnh bằng ``cv2.INTER_CUBIC``,
       cải thiện độ phân giải cho biển số nhỏ (camera xa).
    3. **denoise** — Khử noise camera bằng ``fastNlMeansDenoising``
       trên ảnh grayscale.
    4. **apply_clahe** — Tăng tương phản cục bộ (CLAHE) để làm nổi
       bật ký tự trên nền biển số.

    Args:
        config: Instance ``PreprocessingConfig`` chứa toàn bộ
            tham số xử lý. Nếu ``None``, dùng giá trị mặc định
            từ Pydantic (tương đương ``default.yaml``).

    Example:
        Inject config từ settings::

            from config.settings import get_settings
            settings = get_settings()
            preprocessor = PlatePreprocessor(settings.preprocessing)

        Full pipeline từ frame gốc::

            enhanced = preprocessor.run(frame, detection)
            text = ocr_engine.recognize(enhanced)

        Chỉ chạy pipeline xử lý (ảnh đã crop sẵn)::

            enhanced = preprocessor.run_pipeline(cropped_bgr)
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self._cfg: PreprocessingConfig = (
            config if config is not None else PreprocessingConfig()
        )
        # Khởi tạo CLAHE một lần duy nhất — tránh khởi tạo lại mỗi frame
        self._clahe = cv2.createCLAHE(
            clipLimit=self._cfg.clahe_clip_limit,
            tileGridSize=self._cfg.clahe_tile_grid_size,
        )
        logger.debug(
            "PlatePreprocessor ready  |  scale=%.1f  denoise_h=%.1f  "
            "clahe_clip=%.1f  expand=%.0f%%",
            self._cfg.upscale_factor,
            self._cfg.denoise_h,
            self._cfg.clahe_clip_limit,
            self._cfg.expand_ratio * 100,
        )

    # ------------------------------------------------------------------
    # Step 1: Crop
    # ------------------------------------------------------------------

    def crop_plate_region(
        self,
        frame: np.ndarray,
        detection: Detection,
        expand_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """Cắt vùng biển số từ frame, có mở rộng bbox về 4 phía.

        Mở rộng bbox theo ``expand_ratio`` (% width/height) trước khi
        crop để tránh cắt mất ký tự ở cạnh biển số. Kết quả được clip
        trong giới hạn kích thước frame gốc.

        Args:
            frame: Ảnh gốc ``(H, W, C)`` dạng BGR từ VideoSource.
            detection: Đối tượng ``Detection`` chứa bbox
                ``(x1, y1, x2, y2)`` theo pixel coordinates.
            expand_ratio: Tỷ lệ mở rộng bbox (0.0–1.0).
                Nếu ``None``, dùng ``config.expand_ratio``.

        Returns:
            Ảnh crop BGR ``(H', W', C)`` — bản sao độc lập của frame.
            Trả về mảng rỗng ``shape=(0, 0, 3)`` nếu bbox không hợp lệ
            sau khi expand (thường do bbox quá nhỏ hoặc ngoài frame).
        """
        ratio = expand_ratio if expand_ratio is not None else self._cfg.expand_ratio
        x1, y1, x2, y2 = detection.bbox
        img_h, img_w = frame.shape[:2]

        pad_x = int((x2 - x1) * ratio)
        pad_y = int((y2 - y1) * ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            logger.debug(
                "crop_plate_region: bbox invalid after expand "
                "(x1=%d y1=%d x2=%d y2=%d), returning empty",
                x1, y1, x2, y2,
            )
            return np.empty((0, 0, 3), dtype=frame.dtype)

        return frame[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # Step 2: Upscale
    # ------------------------------------------------------------------

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """Phóng to ảnh bằng ``cv2.INTER_CUBIC``.

        Tăng độ phân giải biển số nhỏ (từ camera xa) để OCR engine
        nhận diện ký tự chính xác hơn. ``INTER_CUBIC`` cho chất lượng
        tốt hơn ``INTER_LINEAR`` với chi phí tính toán chấp nhận được.

        Args:
            image: Ảnh đầu vào ``(H, W)`` hoặc ``(H, W, C)``
                (grayscale hoặc BGR).

        Returns:
            Ảnh đã phóng to theo ``config.upscale_factor``.
            Trả về ảnh gốc không đổi nếu ``image`` rỗng.
        """
        if image.size == 0:
            return image
        h, w = image.shape[:2]
        new_w = max(1, int(w * self._cfg.upscale_factor))
        new_h = max(1, int(h * self._cfg.upscale_factor))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # ------------------------------------------------------------------
    # Step 3: Denoise
    # ------------------------------------------------------------------

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Khử noise bằng ``fastNlMeansDenoising`` trên ảnh grayscale.

        Chuyển sang grayscale trước (nếu cần) vì ``fastNlMeansDenoising``
        yêu cầu ảnh 1 kênh và hiệu quả hơn so với phiên bản màu.
        ``h`` (filter strength) là tham số quan trọng nhất:
        cao → mịn hơn nhưng có thể xóa chi tiết ký tự.

        Args:
            image: Ảnh BGR ``(H, W, 3)`` hoặc grayscale ``(H, W)``.

        Returns:
            Ảnh grayscale đã khử noise ``(H, W)``.
            Trả về ảnh gốc không đổi nếu rỗng.
        """
        if image.size == 0:
            return image
        gray = self._to_gray(image)
        return cv2.fastNlMeansDenoising(
            gray,
            None,
            self._cfg.denoise_h,
            self._cfg.denoise_template_window,
            self._cfg.denoise_search_window,
        )

    # ------------------------------------------------------------------
    # Step 4: CLAHE
    # ------------------------------------------------------------------

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Tăng tương phản cục bộ bằng CLAHE.

        CLAHE (Contrast Limited Adaptive Histogram Equalization) cải thiện
        độ tương phản giữa ký tự (tối/sáng) và nền biển số, đặc biệt
        hiệu quả trong điều kiện ánh sáng không đồng đều (bóng đổ, đêm,
        phản sáng đèn pha).

        Instance ``cv2.CLAHE`` được tái sử dụng (tạo một lần trong
        ``__init__``) để tối ưu hiệu năng.

        Args:
            image: Ảnh grayscale ``(H, W)`` hoặc BGR ``(H, W, 3)``.

        Returns:
            Ảnh grayscale đã tăng tương phản ``(H, W)``.
            Trả về ảnh gốc không đổi nếu rỗng.
        """
        if image.size == 0:
            return image
        gray = self._to_gray(image)
        return self._clahe.apply(gray)

    # ------------------------------------------------------------------
    # Composite pipelines
    # ------------------------------------------------------------------

    def run_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Chạy pipeline xử lý: upscale → denoise → CLAHE.

        Dùng khi đã có ảnh biển số riêng lẻ (không cần crop từ frame).
        Trả về ảnh grayscale đã xử lý, sẵn sàng cho OCR engine.

        Args:
            image: Ảnh biển số ``(H, W, C)`` dạng BGR, hoặc
                ``(H, W)`` dạng grayscale.

        Returns:
            Ảnh grayscale đã xử lý ``(H', W')``.
            Trả về mảng rỗng ``shape=(0, 0)`` nếu ``image`` rỗng / None.
        """
        if image is None or image.size == 0:
            return np.empty((0, 0), dtype=np.uint8)

        upscaled = self.upscale(image)
        denoised = self.denoise(upscaled)
        return self.apply_clahe(denoised)

    def run(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """Pipeline đầy đủ: crop → upscale → denoise → CLAHE.

        Entry point chính, nhận frame gốc và ``Detection`` từ YOLO detector,
        trả về ảnh biển số đã xử lý sẵn cho OCR engine.

        Bỏ qua (trả về mảng rỗng) nếu crop có kích thước nhỏ hơn
        ``config.min_plate_width`` × ``config.min_plate_height`` để tránh
        lãng phí tài nguyên xử lý biển số bị cắt sai hoặc nhiễu.

        Args:
            frame: Frame gốc ``(H, W, C)`` dạng BGR từ VideoSource.
            detection: Kết quả từ ``BaseDetector.detect()``
                chứa bbox ``(x1, y1, x2, y2)``.

        Returns:
            Ảnh grayscale đã xử lý ``(H', W')``.
            Trả về mảng rỗng ``shape=(0, 0)`` nếu:
            - crop không hợp lệ, hoặc
            - kích thước crop nhỏ hơn ``min_plate_width/height``.
        """
        cropped = self.crop_plate_region(frame, detection)
        if cropped.size == 0:
            return np.empty((0, 0), dtype=np.uint8)

        h, w = cropped.shape[:2]
        if w < self._cfg.min_plate_width or h < self._cfg.min_plate_height:
            logger.debug(
                "Plate crop too small (%dx%d px), skip  |  "
                "min=(%dx%d)",
                w, h,
                self._cfg.min_plate_width,
                self._cfg.min_plate_height,
            )
            return np.empty((0, 0), dtype=np.uint8)

        return self.run_pipeline(cropped)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Chuyển ảnh BGR sang grayscale; bỏ qua nếu đã là grayscale.

        Args:
            image: Ảnh ``(H, W)`` hoặc ``(H, W, C)``.

        Returns:
            Ảnh grayscale ``(H, W)``.
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
