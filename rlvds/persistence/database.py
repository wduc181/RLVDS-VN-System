"""SQLite database manager for violation persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path
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
        if self.conn is not None:
            return
        self.conn = sqlite3.connect(self.connect_target)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")

    def disconnect(self) -> None:
        if self.conn is None:
            return
        self.conn.close()
        self.conn = None

    def create_tables(self) -> None:
        self._ensure_connected()
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL UNIQUE,
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
        self._ensure_connected()
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.execute(query, tuple(params))
        self.conn.commit()
        return cur

    def executemany(self, query: str, rows: Iterable[Iterable[Any]]) -> sqlite3.Cursor:
        self._ensure_connected()
        assert self.conn is not None
        cur = self.conn.cursor()
        cur.executemany(query, [tuple(r) for r in rows])
        self.conn.commit()
        return cur

    def query_one(self, query: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        cur = self._execute_read(query, params)
        return cur.fetchone()

    def query_all(self, query: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
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
