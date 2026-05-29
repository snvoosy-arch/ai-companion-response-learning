from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from predictive_bot.runtime_state import SharedRuntimeStateStore


class _FakeConnection:
    def __init__(self) -> None:
        self.row_factory = None
        self.calls: list[str] = []

    def execute(self, sql: str, *args, **kwargs):
        self.calls.append(sql)
        if sql == "PRAGMA journal_mode = WAL":
            raise sqlite3.OperationalError("unable to open database file")
        return self

    def executescript(self, script: str):
        self.calls.append("SCRIPT")
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RuntimeStateStoreTests(unittest.TestCase):
    def test_store_creates_and_heartbeats_sqlite_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "runtime.sqlite3"
            store = SharedRuntimeStateStore(db_path)
            store.heartbeat(
                bot_name="black",
                project_name="predictive-discord-bot",
                discord_user_id="123",
                display_name="black",
            )
            self.assertTrue(db_path.exists())

    def test_wal_failure_falls_back_to_delete_journal(self) -> None:
        fake = _FakeConnection()
        with patch("predictive_bot.runtime_state.sqlite3.connect", return_value=fake):
            SharedRuntimeStateStore("/tmp/runtime_state_fallback.sqlite3")
        self.assertIn("PRAGMA journal_mode = WAL", fake.calls)
        self.assertIn("PRAGMA journal_mode = DELETE", fake.calls)


if __name__ == "__main__":
    unittest.main()
