"""Data models for persistence layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from rlvds.core.base import Violation


@dataclass
class ViolationRecord:
    """Database row model for a violation."""

    plate_text: str
    violation_time: str
    light_state: str
    status: str = "VIOLATION"
    full_image_path: str = ""
    plate_image_path: str = ""
    confidence: float = 0.0
    zone_id: str = "default"
    id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: Any) -> "ViolationRecord":
        return cls(
            id=row["id"],
            plate_text=row["plate_text"],
            violation_time=row["violation_time"],
            light_state=row["light_state"],
            status=row["status"],
            full_image_path=row["full_image_path"] or "",
            plate_image_path=row["plate_image_path"] or "",
            confidence=float(row["confidence"] or 0.0),
            zone_id=row["zone_id"] or "default",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_violation(cls, v: Violation) -> "ViolationRecord":
        light_state = str(v.metadata.get("light_state", "RED"))
        plate_image_path = str(v.metadata.get("plate_image_path", ""))
        return cls(
            plate_text=v.plate_text,
            violation_time=_to_iso(v.timestamp),
            light_state=light_state,
            status="VIOLATION",
            full_image_path=v.image_path,
            plate_image_path=plate_image_path,
            confidence=v.confidence,
            zone_id=v.zone_id,
        )


def _to_iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)
