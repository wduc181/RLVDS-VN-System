"""Violation repository: CRUD + image persistence."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from config.settings import get_settings
from rlvds.core.base import BaseRepository, Detection, Violation
from rlvds.ocr.postprocess import check_valid_plate, format_plate
from rlvds.persistence.database import Database
from rlvds.persistence.models import (
    DailyStat,
    ViolationCreate,
    ViolationRecord,
    ViolationStatistics,
    ViolationUpdate,
)
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
        self._db.migrate_schema()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, entity: Any) -> int | None:
        record = self._normalize_entity(entity)
        if not check_valid_plate(record.plate_text):
            logger.warning("Skip invalid plate_text: %s", record.plate_text)
            return None
        cur = self._db.execute(
            """
            INSERT INTO violations (
                plate_text, violation_time, light_state, status,
                full_image_path, plate_image_path, confidence, zone_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
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
            logger.warning("Insert affected 0 rows for plate: %s", record.plate_text)
            return None
        violation_id = int(cur.lastrowid)
        logger.info(
            "Violation inserted id=%d plate=%s light=%s",
            violation_id,
            record.plate_text,
            record.light_state,
        )
        return violation_id

    def create(self, payload: ViolationCreate | ViolationRecord | Violation | dict[str, Any]) -> int | None:
        if isinstance(payload, dict):
            return self.save(ViolationCreate.model_validate(payload))
        return self.save(payload)

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
        rows = self.get_all_by_plate(plate_text=plate_text, limit=1)
        return rows[0] if rows else None

    def get_all_by_plate(self, plate_text: str, limit: int = 100) -> list[ViolationRecord]:
        normalized = self._normalize_plate_text(plate_text)
        if not normalized:
            return []
        rows = self._db.query_all(
            """
            SELECT * FROM violations
            WHERE plate_text = ?
            ORDER BY violation_time DESC
            LIMIT ?;
            """,
            (normalized, limit),
        )
        return [ViolationRecord.from_row(r) for r in rows]

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

    def update(self, violation_id: int, patch: ViolationUpdate | dict[str, Any]) -> bool:
        payload = patch if isinstance(patch, ViolationUpdate) else ViolationUpdate.model_validate(patch)
        updates = payload.to_update_dict()
        if not updates:
            return False

        if "plate_text" in updates and self.exists_plate(updates["plate_text"], exclude_id=violation_id):
            logger.debug("Skip update due to duplicate plate_text: %s", updates["plate_text"])
            return False

        columns = list(updates.keys())
        values = [updates[col] for col in columns]
        set_clause = ", ".join(f"{col} = ?" for col in columns)
        cur = self._db.execute(
            f"UPDATE violations SET {set_clause} WHERE id = ?;",
            (*values, violation_id),
        )
        return cur.rowcount > 0

    def update_status(self, violation_id: int, status: str) -> bool:
        return self.update(violation_id, {"status": status})

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

    def count(
        self,
        *,
        status: str | None = None,
        light_state: str | None = None,
        zone_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> int:
        where_sql, params = self._build_filters(
            status=status,
            light_state=light_state,
            zone_id=zone_id,
            start=start,
            end=end,
        )
        row = self._db.query_one(
            f"SELECT COUNT(*) AS c FROM violations{where_sql};",
            params,
        )
        return int(row["c"]) if row else 0

    def exists_plate(self, plate_text: str, exclude_id: int | None = None) -> bool:
        normalized = self._normalize_plate_text(plate_text)
        if not normalized:
            return False
        if exclude_id is None:
            row = self._db.query_one(
                "SELECT 1 FROM violations WHERE plate_text = ? LIMIT 1;",
                (normalized,),
            )
            return row is not None

        row = self._db.query_one(
            "SELECT 1 FROM violations WHERE plate_text = ? AND id != ? LIMIT 1;",
            (normalized, exclude_id),
        )
        return row is not None

    def clean_data(self) -> int:
        """Normalize plates, remove invalid records and deduplicate by (plate, time)."""
        rows = self._db.query_all(
            """
            SELECT id, plate_text, violation_time
            FROM violations
            ORDER BY id ASC;
            """
        )
        seen: set[tuple[str, str]] = set()
        ids_to_remove: list[int] = []
        updates: list[tuple[str, int]] = []

        for row in rows:
            raw_plate = str(row["plate_text"] or "")
            try:
                normalized = self._normalize_plate_text(raw_plate)
            except Exception:
                normalized = raw_plate.strip().upper()

            if not check_valid_plate(normalized):
                ids_to_remove.append(int(row["id"]))
                continue

            violation_time = str(row["violation_time"] or "")
            dedup_key = (normalized, violation_time)
            if dedup_key in seen:
                ids_to_remove.append(int(row["id"]))
                continue

            seen.add(dedup_key)
            if normalized != raw_plate:
                updates.append((normalized, int(row["id"])))

        for row_id in ids_to_remove:
            self.delete(row_id)
        for plate_text, row_id in updates:
            self._db.execute(
                "UPDATE violations SET plate_text = ? WHERE id = ?;",
                (plate_text, row_id),
            )
        return len(ids_to_remove)

    def export_csv(
        self,
        filepath: str,
        *,
        status: str | None = None,
        light_state: str | None = None,
        zone_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> int:
        output = Path(filepath)
        output.parent.mkdir(parents=True, exist_ok=True)
        where_sql, params = self._build_filters(
            status=status,
            light_state=light_state,
            zone_id=zone_id,
            start=start,
            end=end,
        )
        rows = self._db.query_all(
            f"""
            SELECT *
            FROM violations
            {where_sql}
            ORDER BY violation_time DESC;
            """,
            params,
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
        return len(rows)

    def get_statistics(
        self,
        *,
        status: str | None = None,
        light_state: str | None = None,
        zone_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        recent_days: int = 7,
    ) -> ViolationStatistics:
        where_sql, params = self._build_filters(
            status=status,
            light_state=light_state,
            zone_id=zone_id,
            start=start,
            end=end,
        )
        recent_days = max(1, int(recent_days))

        total = self.count(
            status=status,
            light_state=light_state,
            zone_id=zone_id,
            start=start,
            end=end,
        )
        by_status = self._group_count("status", where_sql, params)
        by_light = self._group_count("light_state", where_sql, params)
        by_zone = self._group_count("zone_id", where_sql, params)

        daily_rows = self._db.query_all(
            f"""
            SELECT substr(violation_time, 1, 10) AS day, COUNT(*) AS c
            FROM violations
            {where_sql}
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?;
            """,
            (*params, recent_days),
        )
        daily = [DailyStat(date=str(r["day"]), count=int(r["c"])) for r in daily_rows]

        return ViolationStatistics(
            total=total,
            by_status=by_status,
            by_light_state=by_light,
            by_zone=by_zone,
            daily=daily,
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
            logger.debug("record_violation skipped invalid plate: %s", plate_text)
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
        if isinstance(entity, ViolationCreate):
            return entity.to_record()
        if isinstance(entity, ViolationRecord):
            return entity
        if isinstance(entity, Violation):
            return ViolationRecord.from_violation(entity)
        if isinstance(entity, dict):
            return ViolationCreate.model_validate(entity).to_record()
        raise TypeError(f"Unsupported entity type for save(): {type(entity)!r}")

    @staticmethod
    def _normalize_plate_text(plate_text: str) -> str | None:
        """
        Lightweight normalization/validation for plate text used in lookups.

        Avoids the overhead of constructing a full ViolationRecord model and instead
        relies on the OCR postprocess helper to validate/normalize the plate.
        """
        if plate_text is None:
            return None
        candidate = plate_text.strip()
        if not candidate:
            return None

        # Delegate to OCR postprocessing helpers to validate and format.
        normalized = format_plate(candidate)
        if not normalized or not check_valid_plate(normalized):
            return None
        return normalized
    def _build_filters(
        self,
        *,
        status: str | None = None,
        light_state: str | None = None,
        zone_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            clauses.append("status = ?")
            params.append(str(status).strip().upper())
        if light_state is not None:
            clauses.append("light_state = ?")
            params.append(str(light_state).strip().upper())
        if zone_id is not None:
            clauses.append("zone_id = ?")
            params.append(str(zone_id).strip())
        if start is not None and end is not None:
            clauses.append("violation_time BETWEEN ? AND ?")
            params.extend([str(start), str(end)])
        elif start is not None:
            clauses.append("violation_time >= ?")
            params.append(str(start))
        elif end is not None:
            clauses.append("violation_time <= ?")
            params.append(str(end))

        if not clauses:
            return "", tuple()
        return " WHERE " + " AND ".join(clauses), tuple(params)

    def _group_count(
        self,
        column: str,
        where_sql: str,
        params: Sequence[Any],
    ) -> dict[str, int]:
        if column not in {"status", "light_state", "zone_id"}:
            raise ValueError(f"Unsupported group column: {column}")
        rows = self._db.query_all(
            f"""
            SELECT {column} AS key, COUNT(*) AS c
            FROM violations
            {where_sql}
            GROUP BY {column}
            ORDER BY c DESC;
            """,
            params,
        )
        return {str(r["key"]): int(r["c"]) for r in rows}

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
