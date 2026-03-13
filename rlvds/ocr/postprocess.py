"""OCR preprocessing and plate text normalization utilities."""

from __future__ import annotations

import re

import cv2
import numpy as np

_INVALID_PROVINCE_CODES = {"13", "42", "44", "45", "46", "87", "91", "96"}


def upscale_image(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale image before OCR."""
    if image.size == 0:
        return image
    if scale <= 0:
        raise ValueError("scale must be > 0")
    height, width = image.shape[:2]
    new_dimensions = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Denoise image using NLM on grayscale image."""
    if image.size == 0:
        return image
    gray = _to_gray(image)
    return cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to improve local contrast for OCR."""
    if image.size == 0:
        return image
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Standard OCR preprocessing pipeline."""
    upscaled = upscale_image(image)
    denoised = denoise_image(upscaled)
    return adjust_contrast(denoised)


def clean_plate_text(raw_text: str) -> str:
    """Normalize OCR text and fix common confusion pairs."""
    if not raw_text:
        return ""

    text = re.sub(r"[^A-Za-z0-9.-]", "", raw_text).upper().replace(".", "")
    text = text.replace(" ", "")
    if len(text) < 2:
        return text

    chars = list(text)
    chars[0] = _to_digit(chars[0])
    chars[1] = _to_digit(chars[1])

    for i in range(2, len(chars)):
        if chars[i].isalpha():
            continue
        if i == 2:
            chars[i] = _to_alpha(chars[i])
        else:
            chars[i] = _to_digit(chars[i])

    return "".join(chars)


def format_plate(text: str) -> str:
    """Format normalized text to VN-style plate representation."""
    cleaned = clean_plate_text(text)
    if "-" in cleaned:
        return cleaned

    candidates: list[str] = []
    if len(cleaned) >= 8:
        candidates.append(f"{cleaned[:3]}-{cleaned[3:]}")
    if len(cleaned) >= 9:
        candidates.append(f"{cleaned[:4]}-{cleaned[4:]}")

    for candidate in candidates:
        if check_valid_plate(candidate):
            return candidate

    return cleaned


def check_valid_plate(plate: str) -> bool:
    """Validate a formatted VN plate candidate."""
    if not plate:
        return False

    text = plate.upper().strip()
    if len(text) <= 7:
        return False

    parts = text.split("-")
    if len(parts) <= 1 or len(parts[0]) < 2:
        return False

    province_code = parts[0][:2]
    if not province_code.isdigit():
        return False
    if province_code in _INVALID_PROVINCE_CODES:
        return False

    if len(parts) == 2:
        prefix = parts[0]
        if len(prefix) < 3 or not prefix[2].isalpha():
            return False
    elif len(parts) == 3:
        if len(parts[0]) != 2:
            return False
    else:
        return False

    tail = parts[-1]
    if len(tail) < 4 or len(tail) > 6:
        return False
    if not tail.isdigit():
        return False

    return True


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_digit(char: str) -> str:
    return {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }.get(char, char)


def _to_alpha(char: str) -> str:
    return {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
    }.get(char, char)
