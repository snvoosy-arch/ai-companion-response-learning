from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.gptsovits_official_api_client import (
    build_tts_payload,
    load_speaker_manifest,
    resolve_voice_payload,
)


class GPTSoVITSOfficialApiClientTests(unittest.TestCase):
    def test_load_speaker_manifest_reads_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "speakers.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "white": {
                            "ref_audio_path": str(Path(tmpdir) / "white.wav"),
                            "prompt_text": "오늘은 조금 조용히 있고 싶네.",
                            "prompt_lang": "ko",
                        }
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_speaker_manifest(str(manifest_path))
            self.assertEqual(loaded["white"]["prompt_lang"], "ko")

    def test_resolve_voice_payload_prefers_manifest_entry(self) -> None:
        speaker_manifest = {
            "white": {
                "ref_audio_path": "E:\\voices\\white.wav",
                "prompt_text": "천천히 말해도 괜찮아.",
                "prompt_lang": "ko",
            }
        }
        payload = resolve_voice_payload(
            voice_id="white",
            prompt_text="",
            prompt_lang="",
            ref_audio_path="",
            speaker_manifest=speaker_manifest,
            default_language="ko",
        )
        self.assertEqual(payload["prompt_text"], "천천히 말해도 괜찮아.")
        self.assertEqual(payload["prompt_lang"], "ko")
        self.assertTrue(payload["ref_audio_path"].endswith("white.wav"))

    def test_resolve_voice_payload_requires_reference_audio(self) -> None:
        with self.assertRaises(ValueError):
            resolve_voice_payload(
                voice_id="white",
                prompt_text="",
                prompt_lang="ko",
                ref_audio_path="",
                speaker_manifest={},
                default_language="ko",
            )

    def test_build_tts_payload_contains_official_api_fields(self) -> None:
        payload = build_tts_payload(
            text="안녕. 지금 테스트 중이야.",
            language="ko",
            speed=1.0,
            voice_payload={
                "ref_audio_path": "E:\\voices\\white.wav",
                "prompt_text": "안녕.",
                "prompt_lang": "ko",
            },
            text_split_method="cut5",
            media_type="wav",
        )
        self.assertEqual(payload["text_lang"], "ko")
        self.assertEqual(payload["ref_audio_path"], "E:\\voices\\white.wav")
        self.assertEqual(payload["prompt_lang"], "ko")
        self.assertEqual(payload["text_split_method"], "cut5")
        self.assertFalse(payload["streaming_mode"])


if __name__ == "__main__":
    unittest.main()
