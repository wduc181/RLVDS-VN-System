"""
Logging Configuration
=====================

Mục đích:
    Setup logging chuẩn cho toàn bộ application RLVDS-VN.

Cách sử dụng::

    from rlvds.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.debug("Frame #%d processed", frame_id)

Log Format:
    [2026-02-15 12:00:00] | INFO  | module_name | Message here
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_FORMAT = (
    "[%(asctime)s] | %(levelname)-5s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3

# Project root — resolved once, avoids circular import with config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"

# Sentinel to ensure setup_logger runs only once per process
_ROOT_CONFIGURED = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def setup_logger(
    name: str = "rlvds",
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_file_handler: bool = True,
) -> logging.Logger:
    """Tạo và cấu hình logger với Console + File handler.

    Args:
        name: Tên logger (thường là ``"rlvds"`` cho root logger của app).
        level: Log level dạng string — ``"DEBUG"``, ``"INFO"``, …
        log_file: Tên file log. Mặc định ``"rlvds.log"`` trong thư mục
            ``logs/``. Truyền ``None`` để dùng giá trị mặc định.
        enable_file_handler: ``False`` để chỉ log ra console (cho testing).

    Returns:
        ``logging.Logger`` đã được cấu hình.
    """
    global _ROOT_CONFIGURED  # noqa: PLW0603

    logger = logging.getLogger(name)

    # Tránh thêm handler trùng lặp khi gọi nhiều lần
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console handler (stderr) ---
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Rotating file handler ---
    if enable_file_handler:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = _LOG_DIR / (log_file or "rlvds.log")

        file_handler = RotatingFileHandler(
            filename=str(file_path),
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # File luôn ghi DEBUG trở lên
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Không propagate lên root logger để tránh duplicate
    logger.propagate = False

    _ROOT_CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Lấy logger con kế thừa cấu hình từ root ``rlvds`` logger.

    Nếu root logger chưa được setup, tự động gọi ``setup_logger()``
    với giá trị mặc định (hoặc từ ``config.settings`` nếu có).

    Args:
        name: Tên module — thường truyền ``__name__``.

    Returns:
        ``logging.Logger`` sẵn sàng sử dụng.

    Example::

        logger = get_logger(__name__)
        logger.info("Detection started")
    """
    # Đảm bảo root logger đã được khởi tạo
    root = logging.getLogger("rlvds")
    if not root.handlers:
        _auto_setup()

    # Trả về child logger kế thừa handlers từ root
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _auto_setup() -> None:
    """Tự động setup root logger từ config settings (nếu có)."""
    level = "INFO"
    try:
        from config import settings
        level = settings.log_level
    except Exception:  # noqa: BLE001
        pass  # Config chưa sẵn sàng — dùng mặc định

    setup_logger(name="rlvds", level=level)
