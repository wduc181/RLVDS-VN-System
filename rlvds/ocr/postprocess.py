"""OCR preprocessing and plate text normalization utilities."""

from __future__ import annotations

import re

import cv2
import numpy as np

_INVALID_PROVINCE_CODES = {"13", "42", "44", "45", "46", "87", "91", "96"}
_TWO_LETTER_SERIES = {
    "CD",
    "CV",
    "DA",
    "HC",
    "KT",
    "LD",
    "MD",
    "MK",
    "NG",
    "NN",
    "QT",
    "TD",
}
_DIRECT_LEGACY_SERIES_LETTERS = {"B", "E", "L", "P", "V"}


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
    gray = to_gray(image)
    return cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)


def adjust_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to improve local contrast for OCR."""
    if image.size == 0:
        return image
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Standard OCR preprocessing pipeline."""
    denoised = denoise_image(image)
    upscaled = upscale_image(denoised)
    return adjust_contrast(upscaled)


def clean_plate_text(raw_text: str) -> str:
    """Normalize OCR text and fix common confusion pairs."""
    if not raw_text:
        return ""

    text = re.sub(r"[^A-Za-z0-9.-]", "", raw_text).upper().replace(".", "")
    # Strip all hyphens early - they are unreliable from OCR and can be in wrong positions
    text = text.replace("-", "")
    if len(text) < 2:
        return text

    chars = list(text)
    # First 2 chars are always province code digits
    chars[0] = _to_digit(chars[0])
    chars[1] = _to_digit(chars[1])

    # Heuristic to determine prefix end:
    # - 2-digit province + 1 alpha series => prefix len 3
    # - 2-digit province + alpha+digit series (e.g. A1) => prefix len 4
    # - 2-digit province + two alpha series (e.g. AB/LD/NN) => prefix len 4
    if len(chars) >= 4 and chars[2].isalpha() and chars[3].isalpha():
        prefix_end = 4
        two_letter_prefix = True
    elif len(chars) >= 9 and chars[3].isdigit():
        prefix_end = 4
        two_letter_prefix = False
    else:
        prefix_end = 3
        two_letter_prefix = False

    # Fix chars in prefix (after province code)
    for i in range(2, min(prefix_end, len(chars))):
        if two_letter_prefix:
            chars[i] = _to_alpha(chars[i])
        elif i == 2:
            chars[i] = _to_alpha(chars[i])
        else:
            chars[i] = _to_digit(chars[i])

    # Fix chars in tail (all should be digits)
    for i in range(prefix_end, len(chars)):
        chars[i] = _to_digit(chars[i])

    return "".join(chars)


def format_plate(text: str) -> str:
    """Format normalized text to VN-style plate representation."""
    direct = re.sub(r"[^A-Za-z0-9.-]", "", text or "").upper().replace(".", "")
    candidates: list[str] = []
    candidates.extend(_direct_plate_candidates(direct))
    candidates.extend(_separated_plate_candidates(text))
    candidates.extend(_compact_plate_candidates(re.sub(r"[^A-Z0-9]", "", direct)))

    # Aggressive cleanup is a fallback source, not the single source of truth.
    # This keeps raw OCR text such as "90AB" from being rewritten to "90A8".
    cleaned = clean_plate_text(text)
    candidates.extend(_direct_plate_candidates(cleaned))
    candidates.extend(_compact_plate_candidates(cleaned.replace("-", "")))

    for candidate in _dedupe(candidates):
        if check_valid_plate(candidate):
            return candidate

    return cleaned


def _format_separated_plate(text: str) -> str:
    for candidate in _separated_plate_candidates(text):
        if check_valid_plate(candidate):
            return candidate
    return ""


def _direct_plate_candidates(text: str) -> list[str]:
    candidates = []
    legacy = _format_direct_legacy_motorbike_plate(text)
    if legacy:
        candidates.append(legacy)
    if len(text.split("-")) == 2:
        candidates.append(text)
    return candidates


def _separated_plate_candidates(text: str) -> list[str]:
    normalized = re.sub(r"[^A-Za-z0-9.\-\s]", "", text or "").upper()
    normalized = normalized.replace(".", "")
    tokens = re.findall(r"[A-Z0-9]+", normalized)
    if len(tokens) < 2:
        return []

    candidates: list[str] = []
    if len(tokens) >= 3:
        prefix = _normalize_prefix_parts(tokens[0], tokens[1])
        tail = _normalize_tail("".join(tokens[2:]))
        if prefix and tail:
            candidates.append(f"{prefix}-{tail}")

    tail = _normalize_tail("".join(tokens[1:]))
    ambiguous_legacy_split = len(tokens) == 2 and len(tokens[0]) == 4 and len(tail) == 4
    if not ambiguous_legacy_split:
        prefix = _normalize_prefix_token(tokens[0])
        if prefix and tail:
            candidates.append(f"{prefix}-{tail}")

    return candidates


def _compact_plate_candidates(text: str) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []
    if re.fullmatch(r"\d{2}[A-Z]{2}\d{4,5}", text):
        candidates.append(f"{text[:4]}-{text[4:]}")
    if _prefer_legacy_motorbike_split(text):
        candidates.append(f"{text[:4]}-{text[4:]}")
    if len(text) >= 7:
        candidates.append(f"{text[:3]}-{text[3:]}")
    if len(text) >= 9:
        candidates.append(f"{text[:4]}-{text[4:]}")
    return candidates


def _dedupe(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _normalize_prefix_parts(province: str, series: str) -> str:
    if len(province) != 2:
        return ""
    province = "".join(_to_digit(char) for char in province)
    normalized_series = _normalize_series(series)
    if not province.isdigit() or not normalized_series:
        return ""
    return f"{province}{normalized_series}"


def _normalize_prefix_token(prefix: str) -> str:
    if len(prefix) < 3:
        return ""
    province = "".join(_to_digit(char) for char in prefix[:2])
    series = _normalize_series(prefix[2:])
    if not province.isdigit() or not series:
        return ""
    return f"{province}{series}"


def _normalize_series(series: str) -> str:
    if not series or len(series) > 2:
        return ""
    if len(series) == 1:
        return _to_alpha(series[0])

    first = _to_alpha(series[0])
    if series[1].isalpha():
        return f"{first}{_to_alpha(series[1])}"
    return f"{first}{_to_digit(series[1])}"


def _normalize_tail(tail: str) -> str:
    return "".join(_to_digit(char) for char in tail)


def _prefer_legacy_motorbike_split(cleaned: str) -> bool:
    return bool(
        re.fullmatch(r"\d{2}[A-Z]\d\d{4}", cleaned)
        and cleaned[2] in _DIRECT_LEGACY_SERIES_LETTERS
        and cleaned[3] not in {"0", "1"}
    )


def _format_direct_legacy_motorbike_plate(text: str) -> str:
    parts = text.split("-")
    if len(parts) != 2:
        return ""
    prefix, tail = parts
    if not re.fullmatch(r"\d{2}[A-Z]", prefix):
        return ""
    if not re.fullmatch(r"\d{5}", tail):
        return ""
    if prefix[2] not in _DIRECT_LEGACY_SERIES_LETTERS:
        return ""
    if tail[0] in {"0", "1"}:
        return ""
    candidate = f"{prefix}{tail[0]}-{tail[1:]}"
    return candidate if check_valid_plate(candidate) else ""


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
    province_int = int(province_code)
    if province_int < 11 or province_int > 99:
        return False
    if province_code in _INVALID_PROVINCE_CODES:
        return False

    if len(parts) == 2:
        prefix = parts[0]
        if not _valid_plate_prefix(prefix):
            return False
    elif len(parts) == 3:
        if len(parts[0]) != 2 or not parts[0].isdigit():
            return False
        if not _valid_plate_series(parts[1]):
            return False
    else:
        return False

    tail = parts[-1]
    # Support both legacy 4-digit tail and current 5-digit tail.
    if len(tail) < 4 or len(tail) > 5:
        return False
    if not tail.isdigit():
        return False

    return True


def _valid_plate_prefix(prefix: str) -> bool:
    if re.fullmatch(r"\d{2}[A-Z]\d?", prefix):
        return True
    return bool(re.fullmatch(r"\d{2}[A-Z]{2}", prefix))


def _valid_plate_series(series: str) -> bool:
    return bool(re.fullmatch(r"[A-Z](?:\d|[A-Z])?", series))


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale; pass through if already single-channel."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Keep alias for external callers that used the private name
_to_gray = to_gray


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
