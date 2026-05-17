"""SQLite database manager for violation persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from rlvds.utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    """Thin wrapper around sqlite3 connection with schema bootstrap."""

    def __init__(self, db_path_or_url: str = "sqlite:///data/rlvds.db") -> None:
        self.connect_target, self.db_path = self._resolve_db_path(db_path_or_url)
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection | None = None
        self._lock = RLock()

    @staticmethod
    def _resolve_db_path(value: str) -> tuple[str, Path | None]:
        prefix = "sqlite:///"
        if value.startswith(prefix):
            value = value[len(prefix):]
        if value == ":memory:":
            return value, None
        resolved = Path(value).expanduser().resolve()
        return str(resolved), resolved

    def connect(self) -> None:
        with self._lock:
            if self.conn is not None:
                return
            self.conn = sqlite3.connect(self.connect_target, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA synchronous = NORMAL;")

    def disconnect(self) -> None:
        with self._lock:
            if self.conn is None:
                return
            self.conn.close()
            self.conn = None

    def migrate_schema(self) -> None:
        """Remove UNIQUE constraint on plate_text for existing databases.

        SQLite does not support ALTER TABLE DROP CONSTRAINT, so we
        rebuild the table without the UNIQUE clause and copy all rows.
        """
        self._ensure_connected()
        assert self.conn is not None

        with self._lock:
            cur = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='violations';"
            )
            row = cur.fetchone()
            if row is None:
                return

            create_sql = row[0]
            if "plate_text TEXT NOT NULL UNIQUE" not in create_sql:
                return  # already migrated or never had the constraint

            logger.info("Migration: rebuilding violations table to drop UNIQUE constraint")

            self.conn.execute("""
                CREATE TABLE violations_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    violation_time TEXT NOT NULL,
                    light_state TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'VIOLATION',
                    full_image_path TEXT,
                    plate_image_path TEXT,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    zone_id TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.execute("""
                INSERT INTO violations_new
                SELECT id, plate_text, violation_time, light_state, status,
                       full_image_path, plate_image_path, confidence, zone_id,
                       created_at, updated_at
                FROM violations;
            """)
            self.conn.execute("DROP TABLE violations;")
            self.conn.execute("ALTER TABLE violations_new RENAME TO violations;")

            # Recreate indexes and trigger
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_plate_text ON violations(plate_text);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_time ON violations(violation_time);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_light_state ON violations(light_state);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status);"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_violations_zone ON violations(zone_id);"
            )
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_violations_updated_at
                AFTER UPDATE ON violations
                FOR EACH ROW
                BEGIN
                    UPDATE violations
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = OLD.id;
                END;
            """)
            self.conn.commit()
        logger.info("Migration: UNIQUE constraint removed from plate_text")

    def create_tables(self) -> None:
        self._ensure_connected()
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                violation_time TEXT NOT NULL,
                light_state TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'VIOLATION',
                full_image_path TEXT,
                plate_image_path TEXT,
                confidence REAL NOT NULL DEFAULT 0.0,
                zone_id TEXT NOT NULL DEFAULT 'default',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_violations_plate_text ON violations(plate_text);"
        )
        self.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_violations_updated_at
            AFTER UPDATE ON violations
            FOR EACH ROW
            BEGIN
                UPDATE violations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = OLD.id;
            END;
            """
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_violations_time ON violations(violation_time);"
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_violations_light_state ON violations(light_state);"
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status);"
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_violations_zone ON violations(zone_id);"
        )
        logger.info("SQLite schema ready: %s", self.connect_target)

    def execute(self, query: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            self._ensure_connected()
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.execute(query, tuple(params))
            self.conn.commit()
            return cur

    def executemany(self, query: str, rows: Iterable[Iterable[Any]]) -> sqlite3.Cursor:
        with self._lock:
            self._ensure_connected()
            assert self.conn is not None
            cur = self.conn.cursor()
            cur.executemany(query, [tuple(r) for r in rows])
            self.conn.commit()
            return cur

    def query_one(self, query: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._execute_read(query, params)
            return cur.fetchone()

    def query_all(self, query: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._execute_read(query, params)
            return cur.fetchall()

    def _execute_read(self, query: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        self._ensure_connected()
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute(query, tuple(params))
        return cur

    def _ensure_connected(self) -> None:
        if self.conn is None:
            self.connect()

    def __enter__(self) -> "Database":
        self.connect()
        self.create_tables()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()
