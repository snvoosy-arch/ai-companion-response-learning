from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_lmstudio_bot.startup_lock import SingletonLockError, SingletonStartupLock


class StartupLockTests(unittest.TestCase):
    def test_disabled_lock_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock = SingletonStartupLock(Path(tmp) / "noop.lock", enabled=False)
            lock.acquire()
            self.assertFalse(lock.acquired)
            lock.close()

    def test_second_acquire_fails_until_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "white.lock"
            first = SingletonStartupLock(path, enabled=True)
            first.acquire()
            try:
                self.assertTrue(first.acquired)
                second = SingletonStartupLock(path, enabled=True)
                with self.assertRaises(SingletonLockError):
                    second.acquire()
            finally:
                first.close()

            third = SingletonStartupLock(path, enabled=True)
            try:
                third.acquire()
                self.assertTrue(third.acquired)
            finally:
                third.close()


if __name__ == "__main__":
    unittest.main()
