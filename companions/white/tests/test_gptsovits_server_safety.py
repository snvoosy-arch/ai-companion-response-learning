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
    from scripts.gptsovits_v2_server import (
        GPTSoVITSConfig,
        GPTSoVITSService,
        _resolve_reference_audio_path,
        _safe_request_id,
        _validate_bind_safety,
    )


class GPTSoVITSServerSafetyTests(unittest.TestCase):
    def test_safe_request_id_strips_path_traversal(self) -> None:
        safe = _safe_request_id("../../escape-this")
        self.assertNotIn("/", safe)
        self.assertNotIn(".", safe)
        self.assertTrue(safe)

    def test_validate_bind_safety_requires_token_for_non_loopback(self) -> None:
        with self.assertRaises(SystemExit):
            _validate_bind_safety(host="0.0.0.0", server_token="")

    def test_resolve_reference_audio_path_blocks_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "voices"
            root.mkdir()
            outside = Path(tmp) / "outside.wav"
            outside.write_bytes(b"RIFF")
            with self.assertRaises(ValueError):
                _resolve_reference_audio_path(str(outside), allowed_root=root)

    def test_resolve_reference_audio_path_accepts_allowed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "voices"
            root.mkdir()
            inside = root / "sample.wav"
            inside.write_bytes(b"RIFF")
            resolved = _resolve_reference_audio_path(str(inside), allowed_root=root)
            self.assertEqual(resolved, str(inside.resolve()))

    def test_mock_backend_creates_wav_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "artifacts"
            config = GPTSoVITSConfig(
                backend="mock",
                model_label="mock",
                output_dir=output_dir,
                default_language="ko",
                default_speaker="white",
                server_token="",
                max_text_chars=600,
                reference_audio_root=None,
                synth_command_template="",
            )
            service = GPTSoVITSService(config)
            result = service.synthesize(text="안녕. 지금은 GPT-SoVITS mock synth 테스트 중이야.")
            artifact_path = output_dir / f"{result['request_id']}.wav"
            self.assertTrue(artifact_path.exists())
            self.assertEqual(result["artifact_extension"], ".wav")
            self.assertGreater(artifact_path.stat().st_size, 44)


if __name__ == "__main__":
    unittest.main()
