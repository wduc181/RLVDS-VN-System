from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rlvds.core.base import Detection
from rlvds.persistence.database import Database
from rlvds.persistence.models import ViolationCreate, ViolationRecord, ViolationUpdate
from rlvds.persistence.repository import ViolationRepository


def _build_repo(tmp_path: Path) -> ViolationRepository:
    db_path = tmp_path / "rlvds_test.db"
    db = Database(str(db_path))
    return ViolationRepository(db, violations_dir=str(tmp_path / "violations"))


def test_database_supports_sqlite_memory() -> None:
    db = Database(":memory:")
    db.connect()
    db.create_tables()
    db.execute(
        """
        INSERT INTO violations (
            plate_text, violation_time, light_state, status,
            full_image_path, plate_image_path, confidence, zone_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        ("11A-11111", "2026-03-25T10:00:00", "RED", "VIOLATION", "", "", 0.9, "z"),
    )
    row = db.query_one("SELECT COUNT(*) AS c FROM violations;")
    assert row is not None and int(row["c"]) == 1
    db.disconnect()


def test_crud_and_multiple_saves(tmp_path: Path) -> None:
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
    assert second_id is not None  # same plate can have multiple violations
    assert repo.count() == 2

    stored = repo.get_by_id(first_id)
    assert stored is not None
    assert stored.plate_text == "30A-12345"
    assert stored.light_state == "RED"

    updated = repo.update_status(first_id, "CONFIRMED")
    assert updated is True
    assert repo.get_by_id(first_id).status == "CONFIRMED"  # type: ignore[union-attr]

    deleted = repo.delete(first_id)
    assert deleted is True
    assert repo.count() == 1  # second record still exists


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

    # same plate can violate again — should create a second record
    second = repo.record_violation(
        frame=frame,
        detection=det,
        plate_text="51F-11111",
        light_state="RED",
        preprocessed_plate=preprocessed_plate,
    )
    assert second is not None
    assert repo.count() == 2


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


def test_export_csv_exports_all_rows(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    for idx in range(5):
        rec = ViolationRecord(
            plate_text=f"30A-12{idx:03d}",
            violation_time=f"2026-03-25T10:00:0{idx}",
            light_state="RED",
            status="VIOLATION",
            confidence=0.8,
            zone_id="z1",
        )
        repo.save(rec)

    output = tmp_path / "violations.csv"
    repo.export_csv(str(output))
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6  # header + 5 rows


def test_create_and_update_full_record(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    violation_id = repo.create(
        ViolationCreate(
            plate_text="43a12345",
            violation_time="2026-03-25T10:00:00",
            light_state="red",
            status="violation",
            confidence=0.81,
            zone_id="z1",
        )
    )
    assert violation_id is not None

    ok = repo.update(
        violation_id,
        ViolationUpdate(
            status="confirmed",
            light_state="yellow",
            confidence=0.95,
            zone_id="cam02",
        ),
    )
    assert ok is True

    updated = repo.get_by_id(violation_id)
    assert updated is not None
    assert updated.status == "CONFIRMED"
    assert updated.light_state == "YELLOW"
    assert updated.confidence == pytest.approx(0.95)
    assert updated.zone_id == "cam02"


def test_create_rejects_invalid_plate(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    with pytest.raises(ValueError):
        repo.create(
            ViolationCreate(
                plate_text="INVALID",
                violation_time="2026-03-25T10:00:00",
                light_state="RED",
            )
        )

    assert repo.count() == 0


def test_count_filters_and_statistics(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    samples = [
        ("30A-12345", "2026-03-25T10:00:00", "RED", "VIOLATION", "z1"),
        ("30A-12346", "2026-03-25T10:01:00", "GREEN", "CONFIRMED", "z1"),
        ("30A-12347", "2026-03-26T10:00:00", "RED", "CONFIRMED", "z2"),
        ("30A-12348", "2026-03-27T10:00:00", "YELLOW", "VIOLATION", "z2"),
    ]
    for plate, ts, light, status, zone in samples:
        repo.save(
            ViolationRecord(
                plate_text=plate,
                violation_time=ts,
                light_state=light,
                status=status,
                confidence=0.9,
                zone_id=zone,
            )
        )

    assert repo.count() == 4
    assert repo.count(status="CONFIRMED") == 2
    assert repo.count(light_state="RED") == 2
    assert repo.count(zone_id="z2") == 2
    assert repo.count(start="2026-03-26T00:00:00", end="2026-03-27T23:59:59") == 2

    stats = repo.get_statistics(recent_days=3)
    assert stats.total == 4
    assert stats.by_status["CONFIRMED"] == 2
    assert stats.by_light_state["RED"] == 2
    assert stats.by_zone["z1"] == 2
    assert len(stats.daily) == 3


def test_clean_data_removes_invalid_and_normalized_duplicates(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    repo._db.execute(
        """
        INSERT INTO violations (
            plate_text, violation_time, light_state, status,
            full_image_path, plate_image_path, confidence, zone_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        ("30A12345", "2026-03-25T10:00:00", "RED", "VIOLATION", "", "", 0.9, "z1"),
    )
    repo._db.execute(
        """
        INSERT INTO violations (
            plate_text, violation_time, light_state, status,
            full_image_path, plate_image_path, confidence, zone_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        ("30A-12345", "2026-03-25T10:01:00", "RED", "VIOLATION", "", "", 0.9, "z1"),
    )
    repo._db.execute(
        """
        INSERT INTO violations (
            plate_text, violation_time, light_state, status,
            full_image_path, plate_image_path, confidence, zone_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        ("INVALID", "2026-03-25T10:02:00", "RED", "VIOLATION", "", "", 0.9, "z1"),
    )

    removed = repo.clean_data()
    # Only INVALID plate removed; the two 30A entries have different violation_times
    assert removed == 1
    assert repo.count() == 2


def test_export_csv_with_filters_returns_written_rows(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    repo.save(
        ViolationRecord(
            plate_text="30A-22345",
            violation_time="2026-03-25T10:00:00",
            light_state="RED",
            status="VIOLATION",
            confidence=0.8,
            zone_id="z1",
        )
    )
    repo.save(
        ViolationRecord(
            plate_text="30A-22346",
            violation_time="2026-03-25T10:00:01",
            light_state="GREEN",
            status="CONFIRMED",
            confidence=0.8,
            zone_id="z1",
        )
    )

    output = tmp_path / "violations_filtered.csv"
    rows_written = repo.export_csv(str(output), light_state="RED")
    content = output.read_text(encoding="utf-8").strip().splitlines()
    assert rows_written == 1
    assert len(content) == 2
    assert "30A-22345" in content[1]


def test_save_returns_none_when_rowcount_zero(tmp_path: Path, monkeypatch) -> None:
    repo = _build_repo(tmp_path)
    rec = ViolationRecord(
        plate_text="30A-32345",
        violation_time="2026-03-25T10:00:00",
        light_state="RED",
        status="VIOLATION",
        confidence=0.8,
        zone_id="z1",
    )

    monkeypatch.setattr(repo, "exists_plate", lambda plate_text, exclude_id=None: False)
    monkeypatch.setattr(repo._db, "execute", lambda *args, **kwargs: SimpleNamespace(rowcount=0, lastrowid=999))
    assert repo.save(rec) is None


def test_get_by_plate_and_exists_plate_handle_empty_input(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    assert repo.get_by_plate("") is None
    assert repo.exists_plate("") is False


def test_from_row_clamps_legacy_confidence_out_of_range(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path)
    cur = repo._db.execute(
        """
        INSERT INTO violations (
            plate_text, violation_time, light_state, status,
            full_image_path, plate_image_path, confidence, zone_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        ("30A-42345", "2026-03-25T10:00:00", "RED", "VIOLATION", "", "", 1.7, "z1"),
    )
    row_id = int(cur.lastrowid)
    record = repo.get_by_id(row_id)
    assert record is not None
    assert record.confidence == pytest.approx(1.0)
