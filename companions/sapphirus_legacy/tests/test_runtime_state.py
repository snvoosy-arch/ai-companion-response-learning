from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from discord_lmstudio_bot.runtime_state import CLAIM_MODE_MANUAL, SharedRuntimeStateStore


class RuntimeStateTests(unittest.TestCase):
    def test_manual_claim_is_serialized_across_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runtime.sqlite3"
            store = SharedRuntimeStateStore(db_path)
            results: list[bool] = []
            lock = threading.Lock()
            barrier = threading.Barrier(2)

            def _claim(owner_id: str) -> None:
                barrier.wait()
                result = store.claim(
                    bot_name="white",
                    project_name="duo",
                    discord_user_id="123",
                    display_name="white",
                    owner_id=owner_id,
                    owner_name=owner_id,
                    channel_id="10",
                    guild_id="20",
                    reason="manual",
                    ttl_seconds=60,
                    mode=CLAIM_MODE_MANUAL,
                )
                with lock:
                    results.append(result.acquired)

            t1 = threading.Thread(target=_claim, args=("u1",))
            t2 = threading.Thread(target=_claim, args=("u2",))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            self.assertEqual(results.count(True), 1)
            self.assertEqual(results.count(False), 1)


if __name__ == "__main__":
    unittest.main()
