from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_black_rejected_generation_sft_20260427.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_black_rejected_generation_sft_20260427", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_black_rejected_generation_sft_20260427"] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_module()


class BlackRejectedGenerationSftExporterTests(unittest.TestCase):
    def test_malformed_internal_label_draft_goes_to_review(self) -> None:
        row = {
            "speaker": "black",
            "input_text": "먼저 연락하기 전에 뭘 생각해봐야 할까?",
            "action": "share_opinion",
            "decision": "opinion.ask.soft_decision_advice",
            "draft_utterance": {
                "draft_reply": "opinion_decision_request. 먼저 연락하기 전에 뭘 생각해봐야는 부담이 너무 크지 않으면 해볼 만해.",
            },
        }

        status, _, reasons = exporter.classify_row(row)

        self.assertEqual(status, "review")
        self.assertIn("malformed_draft_reply", reasons)

    def test_clean_contact_draft_is_trainable(self) -> None:
        row = {
            "speaker": "black",
            "input_text": "먼저 연락하기 전에 뭘 생각해봐야 할까?",
            "action": "share_opinion",
            "decision": "opinion.ask.soft_decision_advice",
            "draft_utterance": {
                "draft_reply": "먼저 연락하기 전엔 목적이랑 상대 부담을 먼저 보면 돼. 가볍게 안부만 두는 쪽이 무난해.",
            },
        }

        status, target, reasons = exporter.classify_row(row)

        self.assertEqual(status, "train")
        self.assertIn("먼저 연락", target)
        self.assertIn("draft_rewrite_target", reasons)
        self.assertLess(
            exporter._target_copy_score(
                target=target,
                draft_reply=row["draft_utterance"]["draft_reply"],
            ),
            0.96,
        )

    def test_plain_words_containing_yo_are_not_polite_review(self) -> None:
        row = {
            "speaker": "black",
            "input_text": "지금 사과하기 말고 다른 선택지도 있을까?",
            "action": "share_opinion",
            "decision": "opinion.ask.soft_decision_advice",
            "draft_utterance": {
                "draft_reply": "사과하기만 고집할 필요는 없어. 부담이 크면 먼저 정리하고 나중에 말해도 돼.",
            },
        }

        status, target, reasons = exporter.classify_row(row)

        self.assertEqual(status, "train")
        self.assertIn("필욘", target)
        self.assertNotIn("polite_target_review", reasons)

    def test_stale_draft_missing_anchor_goes_to_review(self) -> None:
        row = {
            "speaker": "black",
            "input_text": "AI 뉴스 뭐 있어? 너무 길지 않게 말해줘.",
            "action": "news_answer",
            "decision": "knowledge.news.answer",
            "issues": ["draft_anchor_missing"],
            "draft_utterance": {
                "anchor": "ai",
                "draft_reply": "뉴스는 확인된 헤드라인 기준으로만 짧게 볼게.",
            },
        }

        status, _, reasons = exporter.classify_row(row)

        self.assertEqual(status, "review")
        self.assertIn("draft_anchor_missing_target", reasons)

    def test_repeated_category_recommendation_goes_to_review(self) -> None:
        row = {
            "speaker": "black",
            "input_text": "친구랑 볼 코미디를 찾는 사람한테 짧게 말해줘.",
            "action": "recommend",
            "decision": "recommend.request.media",
            "draft_utterance": {
                "anchor": "코미디",
                "draft_reply": "브루클린 나인-나인. 친구랑 볼 코미디면 코미디처럼 가볍게 웃기는 쪽이 좋아.",
            },
        }

        status, _, reasons = exporter.classify_row(row)

        self.assertEqual(status, "review")
        self.assertIn("bad_repeated_category", reasons)


if __name__ == "__main__":
    unittest.main()
