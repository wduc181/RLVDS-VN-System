"""
RLVDS-VN Settings Module
========================

Mục đích:
    Định nghĩa cấu hình type-safe cho toàn bộ hệ thống sử dụng Pydantic.

Thư viện sử dụng:
    - pydantic / pydantic-settings: Type validation & env override
    - pyyaml: Đọc file YAML

Cách sử dụng:
    from config import settings
    print(settings.detection.confidence_threshold)

Override bằng env:
    RLVDS_DETECTION__CONFIDENCE_THRESHOLD=0.7
    RLVDS_DEBUG=true
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_YAML = PROJECT_ROOT / "config" / "local.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Đọc file YAML và trả về dict.

    Args:
        path: Đường dẫn tuyệt đối đến file YAML.

    Returns:
        Dictionary chứa dữ liệu cấu hình. Trả về ``{}`` nếu file
        không tồn tại hoặc rỗng.
    """
    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------
class VideoConfig(BaseModel):
    """Cấu hình nguồn video / camera."""

    source: str = Field(
        default="data/samples/sample.mp4",
        description="Đường dẫn video hoặc camera index (số nguyên dạng str)",
    )
    width: int = Field(default=1280, ge=1)
    height: int = Field(default=720, ge=1)
    fps: int = Field(default=30, ge=0, description="0 = không giới hạn")
    buffer_size: int = Field(default=10, ge=1)


class DetectionConfig(BaseModel):
    """Cấu hình YOLOv5 detector."""

    model_path: str = Field(default="weights/license_plate.pt")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32)
    device: str = Field(
        default="auto",
        description="'auto' | 'cuda:0' | 'cpu'",
    )


class TrackingConfig(BaseModel):
    """Cấu hình multi-object tracking."""

    enabled: bool = True
    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class SpatialConfig(BaseModel):
    """Cấu hình vùng vi phạm (polygon zones)."""

    violation_zone: List[List[int]] = Field(
        default_factory=list,
        description="Polygon vertices: [[x1,y1], [x2,y2], ...]",
    )
    zone_color: Tuple[int, int, int] = Field(default=(0, 0, 255))
    zone_thickness: int = Field(default=2, ge=1)

    @field_validator("zone_color", mode="before")
    @classmethod
    def _coerce_color(cls, v: Any) -> Tuple[int, int, int]:
        """YAML trả về list — chuyển thành tuple rõ ràng."""
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return tuple(int(c) for c in v)  # type: ignore[return-value]
        raise ValueError(f"zone_color phải là [B, G, R] có 3 phần tử, nhận {v!r}")


class TemporalConfig(BaseModel):
    """Cấu hình chu kỳ đèn giao thông (FSM)."""

    red_duration_sec: int = Field(default=30, ge=1)
    green_duration_sec: int = Field(default=30, ge=1)
    yellow_duration_sec: int = Field(default=3, ge=1)
    initial_state: Literal["RED", "GREEN", "YELLOW"] = "RED"


class OCRConfig(BaseModel):
    """Cấu hình PaddleOCR."""

    lang: str = "en"
    use_gpu: bool = True
    det_model_dir: str = ""
    rec_model_dir: str = ""
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class OCRCacheConfig(BaseModel):
    """Cấu hình OCR caching để tối ưu FPS.

    Khi enabled, chỉ gọi PaddleOCR cho detection mới (cache miss).
    Các frame sau reuse kết quả OCR nếu bbox match theo IOU.

    Attributes:
        enabled: Bật/tắt OCR caching.
        iou_threshold: Ngưỡng IOU tối thiểu để match bbox cũ-mới.
            Mặc định 0.3 vì FPS thấp (≤5) gây displacement lớn giữa frames.
        max_cache_size: Số plate tối đa trong cache.
        cache_ttl_frames: Số frame tối đa giữ cache entry trước khi expire.
        ocr_quality_frames: Số lần OCR tối đa cho cùng một plate
            (lấy kết quả confidence cao nhất).
    """

    enabled: bool = True
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_cache_size: int = Field(default=50, ge=1)
    cache_ttl_frames: int = Field(default=150, ge=1)
    ocr_quality_frames: int = Field(default=3, ge=1)


class PreprocessingConfig(BaseModel):
    """Cấu hình tiền xử lý ảnh biển số trước OCR.

    Tất cả tham số đều có thể override qua ``config/local.yaml``
    hoặc env var với prefix ``RLVDS_PREPROCESSING__``.

    Attributes:
        upscale_factor: Hệ số phóng to ảnh (cv2.INTER_CUBIC). Mặc định 2x.
        denoise_h: Cường độ lọc noise cho fastNlMeansDenoising.
            Giá trị cao hơn → mịn hơn nhưng mất chi tiết.
        denoise_template_window: Kích thước cửa sổ template (px, lẻ).
        denoise_search_window: Kích thước cửa sổ tìm kiếm (px, lẻ).
        clahe_clip_limit: Clip Limit của CLAHE; cao hơn → tương phản mạnh hơn.
        clahe_tile_grid_size: Kích thước tile cho CLAHE ``(W, H)``.
        expand_ratio: Tỷ lệ mở rộng bbox khi crop (0.0–1.0). Mặc định 15%.
        min_plate_width: Chiều rộng tối thiểu (px) của crop hợp lệ.
        min_plate_height: Chiều cao tối thiểu (px) của crop hợp lệ.
    """

    upscale_factor: float = Field(
        default=2.0, gt=0.0, description="Hệ số phóng to ảnh (cv2.INTER_CUBIC)"
    )
    denoise_h: float = Field(
        default=30.0, ge=1.0, description="Cường độ lọc noise (fastNlMeansDenoising h)"
    )
    denoise_template_window: int = Field(
        default=7, ge=1, description="Kích thước cửa sổ template cho NLM (px, lẻ)"
    )
    denoise_search_window: int = Field(
        default=21, ge=1, description="Kích thước cửa sổ tìm kiếm cho NLM (px, lẻ)"
    )
    clahe_clip_limit: float = Field(
        default=2.0, gt=0.0, description="Clip limit của CLAHE"
    )
    clahe_tile_grid_size: Tuple[int, int] = Field(
        default=(8, 8), description="Tile grid size cho CLAHE (W, H)"
    )
    expand_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Tỷ lệ mở rộng bbox khi crop biển số",
    )
    min_plate_width: int = Field(
        default=20, ge=1, description="Chiều rộng tối thiểu của crop hợp lệ (px)"
    )
    min_plate_height: int = Field(
        default=10, ge=1, description="Chiều cao tối thiểu của crop hợp lệ (px)"
    )

    @field_validator("clahe_tile_grid_size", mode="before")
    @classmethod
    def _coerce_tile_grid(cls, v: Any) -> Tuple[int, int]:
        """YAML trả về list — chuyển thành tuple (W, H)."""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return tuple(int(x) for x in v)  # type: ignore[return-value]
        raise ValueError(
            f"clahe_tile_grid_size phải là [W, H] với 2 phần tử, nhận {v!r}"
        )

    @field_validator("denoise_template_window", "denoise_search_window", mode="after")
    @classmethod
    def _ensure_odd_window(cls, v: int) -> int:
        """OpenCV fastNlMeansDenoising yêu cầu window size là số lẻ."""
        if v % 2 == 0:
            raise ValueError(f"Window size phải là số lẻ, nhận {v}")
        return v


class DatabaseConfig(BaseModel):
    """Cấu hình SQLite persistence."""

    url: str = "sqlite:///data/rlvds.db"

    @field_validator("url", mode="after")
    @classmethod
    def _resolve_sqlite_url(cls, v: str) -> str:
        """Chuyển đường dẫn SQLite tương đối thành tuyệt đối theo PROJECT_ROOT.

        Ví dụ:
            ``sqlite:///data/rlvds.db`` → ``sqlite:////abs/path/data/rlvds.db``
        """
        prefix = "sqlite:///"
        if v.startswith(prefix):
            relative = v[len(prefix):]
            p = Path(relative)
            if not p.is_absolute():
                absolute = PROJECT_ROOT / p
                absolute.parent.mkdir(parents=True, exist_ok=True)
                return f"{prefix}{absolute}"
        return v


class PathsConfig(BaseModel):
    """Cấu hình các đường dẫn lưu trữ.

    Tất cả đường dẫn tương đối sẽ được tự động resolve
    thành absolute path dựa trên PROJECT_ROOT.
    """

    violations_dir: str = "data/violations"
    weights_dir: str = "weights"
    samples_dir: str = "data/samples"

    @field_validator("violations_dir", "weights_dir", "samples_dir", mode="after")
    @classmethod
    def _resolve_dir(cls, v: str) -> str:
        """Auto-resolve đường dẫn tương đối → tuyệt đối."""
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return str(p)


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Cấu hình tổng hợp cho toàn bộ hệ thống RLVDS-VN.

    Thứ tự ưu tiên (cao → thấp):
        1. Environment variables  (prefix ``RLVDS_``)
        2. ``config/local.yaml``  (gitignored, dùng cho dev)
        3. ``config/default.yaml``
        4. Giá trị mặc định trong code
    """

    model_config = SettingsConfigDict(
        env_prefix="RLVDS_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Sub configs --
    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    ocr_cache: OCRCacheConfig = Field(default_factory=OCRCacheConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # -- Global flags --
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        """Trả về đường dẫn gốc của project."""
        return PROJECT_ROOT

    def resolve_path(self, relative: str) -> Path:
        """Chuyển đường dẫn tương đối thành tuyệt đối dựa trên project root.

        Args:
            relative: Đường dẫn tương đối (vd: ``weights/model.pt``).

        Returns:
            ``Path`` tuyệt đối.
        """
        p = Path(relative)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Tạo và cache instance ``Settings``.

    Merge config theo thứ tự: default.yaml → local.yaml → env vars.

    Returns:
        Instance ``Settings`` đã được validate.
    """
    # 1. Load YAML files
    yaml_data: Dict[str, Any] = {}
    yaml_data.update(load_yaml_config(DEFAULT_YAML))

    # local.yaml override (gitignored, cho development)
    if LOCAL_YAML.exists():
        local_data = load_yaml_config(LOCAL_YAML)
        _deep_merge(yaml_data, local_data)

    # 2. Build Settings — env vars tự động override nhờ pydantic-settings
    settings = Settings(**yaml_data)

    logger.info(
        "Settings loaded  |  device=%s  confidence=%.2f  debug=%s",
        settings.detection.device,
        settings.detection.confidence_threshold,
        settings.debug,
    )
    return settings


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Merge ``override`` vào ``base`` in-place (recursive).

    Args:
        base: Dict gốc sẽ bị thay đổi.
        override: Dict chứa giá trị mới cần ghi đè.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
