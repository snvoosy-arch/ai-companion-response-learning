from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "desktop_chat_bridge.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("desktop_chat_bridge", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["desktop_chat_bridge"] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_module()


class DesktopChatBridgeWhiteQualityTests(unittest.TestCase):
    def test_white_local_observed_issues_flags_identity_meta_leak(self) -> None:
        issues = bridge._white_local_observed_issues(
            answer="나는 그 요청을 바로 받아들일 수 있어.",
            prompt_text="두 문장 이내로 자기소개해줘.",
        )

        self.assertIn("white_meta_leak", issues)
        self.assertIn("white_identity_weak_answer", issues)

    def test_white_local_observed_issues_accepts_clean_identity(self) -> None:
        issues = bridge._white_local_observed_issues(
            answer="나는 White/화이트라고 부르는 봇이야.",
            prompt_text="네 이름이 뭐야? 과하게 설명하지 말고 짧게 답해.",
        )

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
