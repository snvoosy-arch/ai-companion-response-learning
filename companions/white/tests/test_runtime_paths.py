from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from bot_shared.runtime_paths import (
    render_shared_runtime_state_env_value,
    resolve_default_shared_runtime_state_db_path,
    resolve_private_shared_runtime_root,
)


class RuntimePathsTests(unittest.TestCase):
    def test_shared_root_prefers_userprofile_for_cross_runtime_sharing(self) -> None:
        with patch.dict(
            os.environ,
            {
                "USERPROFILE": "/mnt/c/Users/example",
                "BOT_PRIVATE_SHARED_RUNTIME_ROOT": "",
            },
            clear=False,
        ):
            self.assertEqual(
                resolve_private_shared_runtime_root(),
                Path("/mnt/c/Users/example/.bot-runtime/shared"),
            )
            self.assertEqual(
                resolve_default_shared_runtime_state_db_path(),
                Path("/mnt/c/Users/example/.bot-runtime/shared/runtime/bot_runtime_state.sqlite3"),
            )

    def test_explicit_shared_root_override_wins(self) -> None:
        with patch.dict(
            os.environ,
            {
                "USERPROFILE": "/mnt/c/Users/example",
                "BOT_PRIVATE_SHARED_RUNTIME_ROOT": "/secure/shared-root",
            },
            clear=False,
        ):
            self.assertEqual(
                resolve_private_shared_runtime_root(),
                Path("/secure/shared-root"),
            )
            self.assertEqual(
                render_shared_runtime_state_env_value(),
                "/secure/shared-root/runtime/bot_runtime_state.sqlite3",
            )

    @unittest.skipIf(os.name == "nt", "WSL/Linux normalization regression only")
    def test_windows_userprofile_is_normalized_under_wsl(self) -> None:
        with patch.dict(
            os.environ,
            {
                "USERPROFILE": r"C:\Users\example",
                "BOT_PRIVATE_SHARED_RUNTIME_ROOT": "",
            },
            clear=False,
        ):
            self.assertEqual(
                resolve_private_shared_runtime_root(),
                Path("/mnt/c/Users/example/.bot-runtime/shared"),
            )
            self.assertEqual(
                resolve_default_shared_runtime_state_db_path(),
                Path("/mnt/c/Users/example/.bot-runtime/shared/runtime/bot_runtime_state.sqlite3"),
            )


if __name__ == "__main__":
    unittest.main()
