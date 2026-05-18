"""File I/O utilities for RLVDS-VN."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return resolved Path."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_image(image: np.ndarray, path: str | Path) -> str:
    """Save image to disk, creating parent directories as needed.

    Returns:
        Absolute path to the saved image.

    Raises:
        RuntimeError: If cv2.imwrite fails.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), image)
    if not ok:
        raise RuntimeError(f"Cannot save image to {p}")
    return str(p)


def load_image(path: str | Path) -> np.ndarray:
    """Load image from file.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    image = cv2.imread(str(p))
    if image is None:
        raise RuntimeError(f"Cannot read image: {p}")
    return image


def save_violation_image(
    image: np.ndarray,
    plate_text: str,
    base_dir: str | Path = "data/violations",
) -> str:
    """Save a violation evidence image with naming convention.

    Format: ``{base_dir}/{plate}_{timestamp}.jpg``

    Args:
        image: Frame or crop image (BGR).
        plate_text: Recognized plate text for filename.
        base_dir: Root directory for violation images.

    Returns:
        Absolute path to saved image.
    """
    safe_plate = "".join(c for c in plate_text if c.isalnum() or c in "_-") or "unknown"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{safe_plate}_{ts}.jpg"
    return save_image(image, Path(base_dir) / filename)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file and return as dict. Returns {} if file missing or empty."""
    p = Path(path)
    if not p.exists():
        logger.warning("YAML file not found: %s", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save data as JSON file, creating parent directories as needed."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.debug("JSON saved: %s", p)
