"""Violation repository: CRUD + image persistence."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config.settings import get_settings
from rlvds.core.base import BaseRepository, Detection, Violation
from rlvds.ocr.postprocess import check_valid_plate
from rlvds.persistence.database import Database
from rlvds.persistence.models import ViolationRecord
from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class ViolationRepository(BaseRepository):
    """Data access layer for violations in SQLite."""

    def __init__(self, database: Database, violations_dir: str | None = None) -> None:
        self._db = database
        if violations_dir is None:
            violations_dir = get_settings().paths.violations_dir
        self._base_dir = Path(violations_dir)
        self._scene_dir = self._base_dir / "scene"
        self._plate_dir = self._base_dir / "plate"
        self._scene_dir.mkdir(parents=True, exist_ok=True)
        self._plate_dir.mkdir(parents=True, exist_ok=True)
        self._db.connect()
        self._db.create_tables()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, entity: Any) -> int | None:
        record = self._normalize_entity(entity)
        cur = self._db.execute(
            """
            INSERT INTO violations (
                plate_text, violation_time, light_state, status,
                full_image_path, plate_image_path, confidence, zone_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(plate_text) DO NOTHING;
            """,
            (
                record.plate_text,
                record.violation_time,
                record.light_state,
                record.status,
                record.full_image_path,
                record.plate_image_path,
                record.confidence,
                record.zone_id,
            ),
        )
        if cur.rowcount == 0:
            logger.debug("Skip duplicate violation for plate: %s", record.plate_text)
            return None
        violation_id = int(cur.lastrowid)
        logger.info(
            "Violation inserted id=%d plate=%s light=%s",
            violation_id,
            record.plate_text,
            record.light_state,
        )
        return violation_id

    def get_by_id(self, entity_id: int) -> ViolationRecord | None:
        row = self._db.query_one(
            "SELECT * FROM violations WHERE id = ?;",
            (entity_id,),
        )
        return ViolationRecord.from_row(row) if row else None

    def get_all(self, limit: int = 100, offset: int = 0) -> list[ViolationRecord]:
        rows = self._db.query_all(
            """
            SELECT * FROM violations
            ORDER BY violation_time DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        return [ViolationRecord.from_row(r) for r in rows]

    def get_by_plate(self, plate_text: str) -> ViolationRecord | None:
        row = self._db.query_one(
            """
            SELECT * FROM violations
            WHERE plate_text = ?
            LIMIT 1;
            """,
            (plate_text,),
        )
        return ViolationRecord.from_row(row) if row else None

    def get_by_date_range(self, start: str, end: str) -> list[ViolationRecord]:
        rows = self._db.query_all(
            """
            SELECT * FROM violations
            WHERE violation_time BETWEEN ? AND ?
            ORDER BY violation_time DESC;
            """,
            (start, end),
        )
        return [ViolationRecord.from_row(r) for r in rows]

    def update_status(self, violation_id: int, status: str) -> bool:
        cur = self._db.execute(
            "UPDATE violations SET status = ? WHERE id = ?;",
            (status, violation_id),
        )
        return cur.rowcount > 0

    def delete(self, entity_id: int) -> bool:
        row = self._db.query_one(
            "SELECT full_image_path, plate_image_path FROM violations WHERE id = ?;",
            (entity_id,),
        )
        cur = self._db.execute("DELETE FROM violations WHERE id = ?;", (entity_id,))
        if cur.rowcount <= 0:
            return False
        if row:
            self._safe_remove_file(row["full_image_path"])
            self._safe_remove_file(row["plate_image_path"])
        return True

    def count(self) -> int:
        row = self._db.query_one("SELECT COUNT(*) AS c FROM violations;")
        return int(row["c"]) if row else 0

    def exists_plate(self, plate_text: str) -> bool:
        row = self._db.query_one(
            "SELECT 1 FROM violations WHERE plate_text = ? LIMIT 1;",
            (plate_text,),
        )
        return row is not None

    def clean_data(self) -> int:
        """Remove invalid plate formats from DB."""
        rows = self._db.query_all("SELECT id, plate_text FROM violations;")
        invalid_ids = [int(r["id"]) for r in rows if not check_valid_plate(str(r["plate_text"]))]
        for vid in invalid_ids:
            self.delete(vid)
        return len(invalid_ids)

    def export_csv(self, filepath: str) -> None:
        output = Path(filepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self._db.query_all(
            """
            SELECT *
            FROM violations
            ORDER BY violation_time DESC;
            """
        )
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "plate_text",
                    "violation_time",
                    "light_state",
                    "status",
                    "full_image_path",
                    "plate_image_path",
                    "confidence",
                    "zone_id",
                ]
            )
            for row in rows:
                r = ViolationRecord.from_row(row)
                writer.writerow(
                    [
                        r.id,
                        r.plate_text,
                        r.violation_time,
                        r.light_state,
                        r.status,
                        r.full_image_path,
                        r.plate_image_path,
                        r.confidence,
                        r.zone_id,
                    ]
                )

    # ------------------------------------------------------------------
    # Violation asset management
    # ------------------------------------------------------------------

    def save_violation_images(
        self,
        *,
        frame: np.ndarray,
        detection: Detection | None,
        plate_text: str,
        light_state: str,
        preprocessed_plate: np.ndarray | None = None,
        polygon: np.ndarray | None = None,
        event_time: datetime | None = None,
    ) -> tuple[str, str]:
        """Save scene image and processed plate image, return absolute paths."""
        event_time = event_time or datetime.now()
        safe_plate = self._sanitize_plate(plate_text)
        suffix = event_time.strftime("%Y%m%d_%H%M%S_%f")

        scene_path = (self._scene_dir / f"{safe_plate}_{suffix}.jpg").resolve()
        plate_path = (self._plate_dir / f"{safe_plate}_{suffix}.png").resolve()

        scene = frame.copy()
        if polygon is not None and polygon.size > 0:
            cv2.polylines(scene, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        if detection is not None:
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                scene,
                plate_text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        cv2.putText(
            scene,
            f"LIGHT: {light_state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        self._safe_write(scene_path, scene)

        plate_img = self._build_plate_image(frame, detection, preprocessed_plate)
        self._safe_write(plate_path, plate_img)
        return str(scene_path), str(plate_path)

    def record_violation(
        self,
        *,
        frame: np.ndarray,
        detection: Detection,
        plate_text: str,
        light_state: str,
        preprocessed_plate: np.ndarray | None = None,
        polygon: np.ndarray | None = None,
        zone_id: str = "default",
        status: str = "VIOLATION",
        confidence: float | None = None,
        event_time: datetime | None = None,
    ) -> int | None:
        """Atomic flow: reserve DB row -> save images -> update paths."""
        if not plate_text or plate_text == "unknown":
            return None

        event_time = event_time or datetime.now()
        rec = ViolationRecord(
            plate_text=plate_text,
            violation_time=event_time.isoformat(timespec="seconds"),
            light_state=light_state,
            status=status,
            full_image_path="",
            plate_image_path="",
            confidence=float(confidence if confidence is not None else detection.confidence),
            zone_id=zone_id,
        )
        violation_id = self.save(rec)
        if violation_id is None:
            logger.debug("record_violation skipped duplicated plate: %s", plate_text)
            return None

        full_image_path: str | None = None
        plate_image_path: str | None = None
        try:
            full_image_path, plate_image_path = self.save_violation_images(
                frame=frame,
                detection=detection,
                plate_text=plate_text,
                light_state=light_state,
                preprocessed_plate=preprocessed_plate,
                polygon=polygon,
                event_time=event_time,
            )
            self._db.execute(
                """
                UPDATE violations
                SET full_image_path = ?, plate_image_path = ?
                WHERE id = ?;
                """,
                (full_image_path, plate_image_path, violation_id),
            )
        except Exception:
            # Rollback row when image persistence fails, avoid orphan DB records.
            self._db.execute("DELETE FROM violations WHERE id = ?;", (violation_id,))
            self._safe_remove_file(full_image_path)
            self._safe_remove_file(plate_image_path)
            raise

        return violation_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_plate(plate_text: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", plate_text.strip())
        return safe or "unknown"

    @staticmethod
    def _normalize_entity(entity: Any) -> ViolationRecord:
        if isinstance(entity, ViolationRecord):
            return entity
        if isinstance(entity, Violation):
            return ViolationRecord.from_violation(entity)
        raise TypeError(f"Unsupported entity type for save(): {type(entity)!r}")

    @staticmethod
    def _safe_write(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), image)
        if not ok:
            raise RuntimeError(f"Cannot save image to {path}")

    def _safe_remove_file(self, path: str | None) -> None:
        if not path:
            return
        p = Path(path).expanduser().resolve()
        if not self._is_under_base_dir(p):
            logger.warning("Skip unsafe delete outside violations_dir: %s", p)
            return
        try:
            if p.exists():
                p.unlink()
        except OSError:
            logger.warning("Cannot delete image file: %s", p)

    def _is_under_base_dir(self, path: Path) -> bool:
        base = self._base_dir.resolve()
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    @staticmethod
    def _build_plate_image(
        frame: np.ndarray,
        detection: Detection | None,
        preprocessed_plate: np.ndarray | None,
    ) -> np.ndarray:
        if preprocessed_plate is not None and preprocessed_plate.size > 0:
            return preprocessed_plate
        if detection is None:
            return frame
        x1, y1, x2, y2 = detection.bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2].copy()
