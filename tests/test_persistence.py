from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from rlvds.core.base import Detection
from rlvds.persistence.database import Database
from rlvds.persistence.models import ViolationRecord
from rlvds.persistence.repository import ViolationRepository


def _build_repo(tmp_path: Path) -> ViolationRepository:
    db_path = tmp_path / "rlvds_test.db"
    db = Database(str(db_path))
    return ViolationRepository(db, violations_dir=str(tmp_path / "violations"))


def test_crud_and_dedup_by_plate(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    record = ViolationRecord(
        plate_text="30A-12345",
        violation_time="2026-03-25T10:00:00",
        light_state="RED",
        status="VIOLATION",
        confidence=0.91,
        zone_id="z1",
    )
    first_id = repo.save(record)
    second_id = repo.save(record)

    assert first_id is not None
    assert second_id is None
    assert repo.count() == 1

    stored = repo.get_by_id(first_id)
    assert stored is not None
    assert stored.plate_text == "30A-12345"
    assert stored.light_state == "RED"

    updated = repo.update_status(first_id, "CONFIRMED")
    assert updated is True
    assert repo.get_by_id(first_id).status == "CONFIRMED"  # type: ignore[union-attr]

    deleted = repo.delete(first_id)
    assert deleted is True
    assert repo.count() == 0


def test_record_violation_saves_both_images_and_paths(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    det = Detection(bbox=(50, 60, 150, 120), confidence=0.88)
    preprocessed_plate = np.ones((40, 120), dtype=np.uint8) * 255
    polygon = np.array([[20, 20], [280, 20], [280, 180], [20, 180]], dtype=np.int32).reshape(
        (-1, 1, 2)
    )
    event_time = datetime(2026, 3, 25, 12, 0, 0)

    violation_id = repo.record_violation(
        frame=frame,
        detection=det,
        plate_text="51F-11111",
        light_state="RED",
        preprocessed_plate=preprocessed_plate,
        polygon=polygon,
        zone_id="cam01",
        event_time=event_time,
    )

    assert violation_id is not None
    stored = repo.get_by_id(violation_id)
    assert stored is not None
    assert stored.full_image_path
    assert stored.plate_image_path
    assert Path(stored.full_image_path).exists()
    assert Path(stored.plate_image_path).exists()
    assert stored.light_state == "RED"
    assert stored.zone_id == "cam01"
    scene_files_before = list((tmp_path / "violations" / "scene").glob("*"))
    plate_files_before = list((tmp_path / "violations" / "plate").glob("*"))

    # dedup check: same plate should not insert again
    second = repo.record_violation(
        frame=frame,
        detection=det,
        plate_text="51F-11111",
        light_state="RED",
        preprocessed_plate=preprocessed_plate,
    )
    assert second is None
    assert repo.count() == 1
    scene_files_after = list((tmp_path / "violations" / "scene").glob("*"))
    plate_files_after = list((tmp_path / "violations" / "plate").glob("*"))
    assert len(scene_files_after) == len(scene_files_before)
    assert len(plate_files_after) == len(plate_files_before)


def test_delete_does_not_remove_files_outside_violations_dir(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("keep-me", encoding="utf-8")

    cur = repo._db.execute(
        """
        INSERT INTO violations (
            plate_text, violation_time, light_state, status,
            full_image_path, plate_image_path, confidence, zone_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "29A-99999",
            "2026-03-25T15:00:00",
            "RED",
            "VIOLATION",
            str(outside_file),
            str(outside_file),
            0.5,
            "z1",
        ),
    )
    violation_id = int(cur.lastrowid)
    assert repo.delete(violation_id) is True
    assert outside_file.exists()
