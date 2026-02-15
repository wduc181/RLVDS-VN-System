"""RLVDS-VN Configuration Package.

Mục đích:
    Quản lý cấu hình toàn hệ thống sử dụng Pydantic.

Cách sử dụng::

    from config import settings
    print(settings.detection.confidence_threshold)

    from config import get_settings
    s = get_settings()
"""

from __future__ import annotations

import logging as _logging

from config.settings import Settings, get_settings

_logger = _logging.getLogger(__name__)

try:
    settings: Settings = get_settings()
except Exception as _exc:  # noqa: BLE001
    _logger.error(
        "Không thể load settings — kiểm tra config/default.yaml: %s", _exc
    )
    raise

__all__ = ["Settings", "get_settings", "settings"]
