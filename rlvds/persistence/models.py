"""Pydantic models for persistence layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rlvds.core.base import Violation
from rlvds.ocr.postprocess import check_valid_plate, format_plate

_VALID_LIGHT_STATES = {"RED", "GREEN", "YELLOW", "UNKNOWN"}
_DEFAULT_STATUS = "VIOLATION"


def _to_iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)


def _normalize_plate(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return format_plate(text)


class ViolationRecord(BaseModel):
    """Database row model for a violation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    plate_text: str = Field(min_length=1)
    violation_time: str
    light_state: str
    status: str = _DEFAULT_STATUS
    full_image_path: str = ""
    plate_image_path: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    zone_id: str = "default"
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("plate_text", mode="before")
    @classmethod
    def _validate_plate_text(cls, value: Any) -> str:
        normalized = _normalize_plate(str(value or ""))
        if not normalized:
            raise ValueError("plate_text must not be empty")
        return normalized

    @field_validator("violation_time", mode="before")
    @classmethod
    def _validate_violation_time(cls, value: Any) -> str:
        if value is None:
            raise ValueError("violation_time is required")
        return _to_iso(value)

    @field_validator("light_state", mode="before")
    @classmethod
    def _validate_light_state(cls, value: Any) -> str:
        state = str(value or "UNKNOWN").strip().upper()
        if state not in _VALID_LIGHT_STATES:
            return "UNKNOWN"
        return state

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: Any) -> str:
        text = str(value or _DEFAULT_STATUS).strip().upper()
        return text or _DEFAULT_STATUS

    @field_validator("zone_id", mode="before")
    @classmethod
    def _validate_zone_id(cls, value: Any) -> str:
        text = str(value or "default").strip()
        return text or "default"

    @classmethod
    def from_row(cls, row: Any) -> "ViolationRecord":
        raw_conf = float(row["confidence"] or 0.0)
        normalized_conf = min(1.0, max(0.0, raw_conf))
        payload = {
            "id": row["id"],
            "plate_text": row["plate_text"],
            "violation_time": row["violation_time"],
            "light_state": row["light_state"],
            "status": row["status"],
            "full_image_path": row["full_image_path"] or "",
            "plate_image_path": row["plate_image_path"] or "",
            "confidence": normalized_conf,
            "zone_id": row["zone_id"] or "default",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        return cls.model_validate(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_violation(cls, v: Violation) -> "ViolationRecord":
        light_state = str(v.metadata.get("light_state", "RED"))
        plate_image_path = str(v.metadata.get("plate_image_path", ""))
        return cls(
            plate_text=v.plate_text,
            violation_time=_to_iso(v.timestamp),
            light_state=light_state,
            status=_DEFAULT_STATUS,
            full_image_path=v.image_path,
            plate_image_path=plate_image_path,
            confidence=v.confidence,
            zone_id=v.zone_id,
        )


class ViolationCreate(ViolationRecord):
    """Strict input model used before creating a DB row."""

    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @model_validator(mode="after")
    def _validate_plate_format(self) -> "ViolationCreate":
        if self.plate_text == "UNKNOWN" or not check_valid_plate(self.plate_text):
            raise ValueError(f"invalid Vietnamese plate format: {self.plate_text}")
        return self

    def to_record(self) -> ViolationRecord:
        return ViolationRecord.model_validate(self.model_dump())


class ViolationUpdate(BaseModel):
    """Partial update model for violation records."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    plate_text: str | None = None
    violation_time: str | datetime | None = None
    light_state: str | None = None
    status: str | None = None
    full_image_path: str | None = None
    plate_image_path: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    zone_id: str | None = None

    @field_validator("plate_text", mode="before")
    @classmethod
    def _validate_plate_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = _normalize_plate(str(value))
        if not normalized or not check_valid_plate(normalized):
            raise ValueError("plate_text must be a valid Vietnamese plate")
        return normalized

    @field_validator("violation_time", mode="before")
    @classmethod
    def _validate_violation_time(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _to_iso(value)

    @field_validator("light_state", mode="before")
    @classmethod
    def _validate_light_state(cls, value: Any) -> str | None:
        if value is None:
            return None
        state = str(value).strip().upper()
        if state not in _VALID_LIGHT_STATES:
            raise ValueError(f"unsupported light_state: {state}")
        return state

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: Any) -> str | None:
        if value is None:
            return None
        status = str(value).strip().upper()
        return status or None

    @field_validator("zone_id", mode="before")
    @classmethod
    def _validate_zone_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        zone = str(value).strip()
        return zone or None

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DailyStat(BaseModel):
    """Daily aggregation row for dashboard charts."""

    date: str
    count: int = Field(ge=0)


class ViolationStatistics(BaseModel):
    """Dashboard statistics snapshot."""

    total: int = Field(ge=0)
    by_status: dict[str, int] = Field(default_factory=dict)
    by_light_state: dict[str, int] = Field(default_factory=dict)
    by_zone: dict[str, int] = Field(default_factory=dict)
    daily: list[DailyStat] = Field(default_factory=list)
