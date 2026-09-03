from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DISCODEBOT_SRC = WORKSPACE_ROOT / "discodebot" / "src"
for candidate in (WORKSPACE_ROOT, DISCODEBOT_SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.append(candidate_text)

from bot_shared.speech import (
    default_gptsovits_client_python,
    default_gptsovits_client_script,
    default_gptsovits_speaker_manifest,
    resolve_gptsovits_command_template,
)


class GPTSoVITSCommandTemplateTests(unittest.TestCase):
    def test_explicit_template_wins(self) -> None:
        template = resolve_gptsovits_command_template(
            explicit_template="custom command",
            server_url="http://127.0.0.1:8022",
            workspace_root=WORKSPACE_ROOT,
        )
        self.assertEqual(template, "custom command")

    def test_empty_without_server_url_returns_empty(self) -> None:
        template = resolve_gptsovits_command_template(
            explicit_template="",
            server_url="",
            workspace_root=WORKSPACE_ROOT,
        )
        self.assertEqual(template, "")

    def test_server_url_builds_default_gptsovits_client_command(self) -> None:
        template = resolve_gptsovits_command_template(
            explicit_template="",
            server_url="http://127.0.0.1:9980",
            workspace_root=WORKSPACE_ROOT,
            language="ko",
        )
        self.assertIn(default_gptsovits_client_python(WORKSPACE_ROOT), template)
        self.assertIn(default_gptsovits_client_script(WORKSPACE_ROOT), template)
        self.assertIn(default_gptsovits_speaker_manifest(WORKSPACE_ROOT), template)
        self.assertIn("--server-url", template)
        self.assertIn("http://127.0.0.1:9980", template)
        self.assertIn("--speaker-manifest", template)
        self.assertIn("--text-file {text_path}", template)
        self.assertIn("--output-path {output_path}", template)
        self.assertIn("--voice-id {voice_id}", template)
        self.assertIn("--speed {speed}", template)


if __name__ == "__main__":
    unittest.main()
