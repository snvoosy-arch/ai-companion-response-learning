from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from predictive_bot.startup_lock import StartupLockError, acquire_startup_lock


class StartupLockTests(unittest.TestCase):
    def test_disabled_returns_noop_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "noop.lock"
            lock = acquire_startup_lock(lock_path, enabled=False)
            self.assertFalse(lock.enabled)
            self.assertFalse(lock.acquired)
            self.assertEqual(lock.path, lock_path)
            lock.release()

    def test_second_acquire_fails_until_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "singleton.lock"
            first = acquire_startup_lock(lock_path, enabled=True, bot_name="black")
            try:
                self.assertTrue(first.enabled)
                self.assertTrue(first.acquired)
                with self.assertRaises(StartupLockError):
                    acquire_startup_lock(lock_path, enabled=True, bot_name="black")
            finally:
                first.release()

            second = acquire_startup_lock(lock_path, enabled=True, bot_name="black")
            try:
                self.assertTrue(second.acquired)
            finally:
                second.release()


if __name__ == "__main__":
    unittest.main()
