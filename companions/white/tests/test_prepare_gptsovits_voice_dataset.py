from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from scripts.prepare_gptsovits_voice_dataset import build_manifest


class PrepareGPTSoVITSDatasetTests(unittest.TestCase):
    def test_build_manifest_copies_audio_and_builds_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "raw"
            output_dir = root / "processed"
            input_dir.mkdir()
            (input_dir / "sample01.wav").write_bytes(b"RIFF")
            (input_dir / "sample01.txt").write_text("안녕. 이건 화이트 샘플이야.", encoding="utf-8")

            entries, summary = build_manifest(
                input_dir=input_dir,
                output_dir=output_dir,
                speaker_name="white",
                language="ko",
                prefix="",
                missing_transcript="error",
                copy_mode="copy",
            )

            self.assertEqual(len(entries), 1)
            self.assertIn("|white|ko|안녕. 이건 화이트 샘플이야.", entries[0])
            self.assertEqual(summary["prepared_entries"], 1)
            self.assertTrue(any(output_dir.iterdir()))

    def test_build_manifest_can_skip_missing_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "raw"
            output_dir = root / "processed"
            input_dir.mkdir()
            (input_dir / "sample01.wav").write_bytes(b"RIFF")

            entries, summary = build_manifest(
                input_dir=input_dir,
                output_dir=output_dir,
                speaker_name="black",
                language="ko",
                prefix="",
                missing_transcript="skip",
                copy_mode="copy",
            )

            self.assertEqual(entries, [])
            self.assertEqual(summary["prepared_entries"], 0)
            self.assertEqual(summary["missing_transcripts"], ["sample01.wav"])


if __name__ == "__main__":
    unittest.main()
