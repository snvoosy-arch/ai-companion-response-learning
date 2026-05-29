from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

with mock.patch.dict(
    sys.modules,
    {
        "flask": SimpleNamespace(
            Flask=lambda *args, **kwargs: object(),
            abort=lambda *args, **kwargs: None,
            jsonify=lambda *args, **kwargs: None,
            request=SimpleNamespace(),
            send_file=lambda *args, **kwargs: None,
        )
    },
):
    from scripts.xtts_v2_server import (
        _resolve_speaker_wav_paths,
        _safe_request_id,
        _validate_bind_safety,
    )


class XTTSServerSafetyTests(unittest.TestCase):
    def test_safe_request_id_strips_path_traversal(self) -> None:
        safe = _safe_request_id("../../escape-this")
        self.assertNotIn("/", safe)
        self.assertNotIn(".", safe)
        self.assertTrue(safe)

    def test_validate_bind_safety_requires_token_for_non_loopback(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_bind_safety(host="0.0.0.0", server_token="")

    def test_resolve_speaker_wav_paths_blocks_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "voices"
            root.mkdir()
            outside = Path(tmp) / "outside.wav"
            outside.write_bytes(b"RIFF")
            with self.assertRaises(ValueError):
                _resolve_speaker_wav_paths([str(outside)], allowed_root=root)

    def test_resolve_speaker_wav_paths_accepts_allowed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "voices"
            root.mkdir()
            inside = root / "sample.wav"
            inside.write_bytes(b"RIFF")
            resolved = _resolve_speaker_wav_paths([str(inside)], allowed_root=root)
            self.assertEqual(resolved, [str(inside.resolve())])


if __name__ == "__main__":
    unittest.main()
