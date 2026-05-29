from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_dual_bridge_live_probe.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


probe_script = _load_module(SCRIPT_PATH, "run_dual_bridge_live_probe")


class RunDualBridgeLiveProbeTests(unittest.TestCase):
    def test_black_issue_codes_flag_malformed_repeated_fragment(self) -> None:
        issues = probe_script.issue_codes_for(
            speaker="black",
            prompt="좋아졌다고 말하긴 애매한데, 아까보단 숨이 좀 붙는다.",
            reply="좋아졌다고졌다고 말하긴 애매한 쪽이야.",
            payload={
                "ok": True,
                "llm_used": True,
                "render_source": "llm",
                "reason_code": "feeling.share.reflective",
            },
            error="",
        )

        self.assertIn("black_malformed_or_stock_reply", issues)

    def test_black_issue_codes_flag_social_return_fragment(self) -> None:
        issues = probe_script.issue_codes_for(
            speaker="black",
            prompt="오랜만에 다시 말 걸어본다.",
            reply="다시 왔. 오랜만이네.",
            payload={
                "ok": True,
                "llm_used": True,
                "render_source": "llm",
                "reason_code": "conversation.continue.light_smalltalk",
            },
            error="",
        )

        self.assertIn("black_malformed_or_stock_reply", issues)

    def test_black_issue_codes_flag_stock_tail_even_with_anchor(self) -> None:
        issues = probe_script.issue_codes_for(
            speaker="black",
            prompt="별일 없었는데도 괜히 지친다.",
            reply="별일 없었는데도 괜히 지치는 날은 있지. 지금 결이 너무 선명하잖아.",
            payload={
                "ok": True,
                "llm_used": True,
                "render_source": "llm",
                "reason_code": "feeling.share.reflective",
            },
            error="",
        )

        self.assertIn("black_malformed_or_stock_reply", issues)

    def test_black_issue_codes_flag_draft_verification_issue(self) -> None:
        issues = probe_script.issue_codes_for(
            speaker="black",
            prompt="바다에서 무엇을 하고 놀면 좋을까?",
            reply="그런 결은 있지.",
            payload={
                "ok": True,
                "llm_used": True,
                "render_source": "llm",
                "reason_code": "opinion.share.activity_recommendation",
                "verification_issues": ["draft_anchor_missing"],
            },
            error="",
        )

        self.assertIn("black_draft_verification_issue", issues)

    def test_black_issue_codes_flag_polite_style_violation(self) -> None:
        issues = probe_script.issue_codes_for(
            speaker="black",
            prompt="첫 만남 느낌으로 너무 과하지 않게 안부 물어봐줘.",
            reply="오늘 컨디션이 좋을 것 같아요! 오늘은 어떠신가요?",
            payload={
                "ok": True,
                "llm_used": True,
                "render_source": "llm",
                "reason_code": "conversation.continue.light_smalltalk",
            },
            error="",
        )

        self.assertIn("black_polite_style_violation", issues)

    def test_black_issue_codes_do_not_treat_needed_as_polite_ending(self) -> None:
        issues = probe_script.issue_codes_for(
            speaker="black",
            prompt="오늘은 비가 오네. 그냥 조용히 있고 싶은 쪽이야.",
            reply="비 오는 날엔 굳이 텐션 올릴 필요 없지. 오늘은 조용한 쪽으로 가도 돼.",
            payload={
                "ok": True,
                "llm_used": True,
                "render_source": "llm",
                "reason_code": "weather.statement.feeling_reflect",
            },
            error="",
        )

        self.assertNotIn("black_polite_style_violation", issues)

    def test_black_issue_codes_flag_unsupported_news_addition(self) -> None:
        issues = probe_script.issue_codes_for(
            speaker="black",
            prompt="AI 뉴스 뭐 있어? 너무 길지 않게 말해줘.",
            reply="AI 뉴스는 확인된 헤드라인 기준으로만 짧게 볼게. 오늘날 `AI` 쪽으로 딱 맞는 헤드라인이 적어서, 전체",
            payload={
                "ok": True,
                "llm_used": True,
                "render_source": "llm",
                "action": "news_answer",
                "reason_code": "knowledge.news.answer",
                "draft_utterance": {"draft_reply": "AI 뉴스는 확인된 헤드라인 기준으로만 짧게 볼게."},
            },
            error="",
        )

        self.assertIn("black_unsupported_news_addition", issues)

    def test_build_draft_failure_rows_exports_training_record(self) -> None:
        report = {
            "results": [
                {
                    "id": "p001",
                    "speaker": "black",
                    "prompt": "바다에서 무엇을 하고 놀면 좋을까?",
                    "reply": "그런 결은 있지.",
                    "action": "share_opinion",
                    "reason_code": "opinion.share.activity_recommendation",
                    "issue_codes": ["black_draft_verification_issue"],
                    "verification_issues": ["draft_anchor_missing"],
                    "draft_utterance": {"draft_reply": "바다면 물놀이가 무난해."},
                }
            ]
        }

        rows = probe_script.build_draft_failure_rows(report, source_path=Path("out.json"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt"], "바다에서 무엇을 하고 놀면 좋을까?")
        self.assertEqual(rows[0]["failure_kind"], "draft_verification")
        self.assertEqual(rows[0]["draft_utterance"]["draft_reply"], "바다면 물놀이가 무난해.")
        self.assertEqual(rows[0]["source_report"], "out.json")

    def test_build_draft_failure_rows_keeps_generation_failures_with_draft(self) -> None:
        report = {
            "results": [
                {
                    "id": "p002",
                    "speaker": "black",
                    "prompt": "오늘은 좀 마음이 가라앉아.",
                    "reply": "응, 그 결은 오래 가도 돼.",
                    "pass": False,
                    "action": "share_feeling",
                    "reason_code": "feeling.share.reflective",
                    "issue_codes": ["black_llm_generation_issue"],
                    "verification_issues": [],
                    "draft_utterance": {"draft_reply": "오늘은 좀 낮게 가도 돼."},
                }
            ]
        }

        rows = probe_script.build_draft_failure_rows(report, source_path=Path("out.json"))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["failure_kind"], "black_quality")
        self.assertEqual(rows[0]["issue_codes"], ["black_llm_generation_issue"])


if __name__ == "__main__":
    unittest.main()
