from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_black_relation_priority_resolver.py"
SCRIPTS_ROOT = SCRIPT_PATH.parent


def _load_module():
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location("evaluate_black_relation_priority_resolver", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"unable to load module: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_black_relation_priority_resolver"] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_module()


def _axis(label: str | None, confidence: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(label=label, confidence=confidence, scores={label: confidence} if label else {})


class _FakeClassifier:
    def predict(self, text: str) -> SimpleNamespace:
        if "퇴사 사표" in text:
            return SimpleNamespace(
                coarse_intent=_axis("smalltalk_opinion"),
                domain=_axis("daily_life"),
                schema=_axis("practical_advice"),
                speech_act=_axis("ask"),
                extra_axes={
                    "emotion": _axis("anxious"),
                    "state_hint": _axis("practical_focus"),
                    "action_hint": _axis("share_opinion"),
                    "draft_frame_family": _axis("practical_guidance"),
                    "draft_frame": _axis("judgment_quit_impulse_after_feedback"),
                    "relation_type": _axis(None),
                    "relation_priority": _axis(None),
                },
            )
        return SimpleNamespace(
            coarse_intent=_axis("smalltalk_feeling"),
            domain=_axis("ai_companion"),
            schema=_axis("emotional_support"),
            speech_act=_axis("ask"),
            extra_axes={
                "emotion": _axis("anxious"),
                "state_hint": _axis("emotional_context"),
                "action_hint": _axis("share_feeling"),
                "draft_frame_family": _axis("emotional_support"),
                "draft_frame": _axis("ai_comfort_before_emotion_proof"),
                "relation_type": _axis("ally_loneliness_emotion_first"),
                "relation_priority": _axis("emotion_stabilize"),
            },
        )


class EvaluateBlackRelationPriorityResolverTests(unittest.TestCase):
    def test_evaluate_rows_scores_resolver_against_model_priority_head(self) -> None:
        rows = [
            {
                "id": "judgment_1",
                "text": "상사 피드백 받고 퇴사 사표 충동이 올라와, 지금 결정하면 안 되지?",
                "targets": {
                    "relation_type": "quit_after_feedback_impulse",
                    "relation_priority": "judgment",
                },
                "meta": {"source_reason": "sample_quit_impulse"},
            },
            {
                "id": "none_1",
                "text": "AI 감정이 진짜인지 궁금한데 지금은 내가 불안해서 위로부터 받고 싶어.",
                "targets": {
                    "relation_type": "__none__",
                    "relation_priority": "__none__",
                },
                "meta": {"source_reason": "sample_ai_comfort"},
            },
        ]

        result = evaluator.evaluate_rows(_FakeClassifier(), rows, sample_errors=5)

        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["accuracy"]["resolver_relation_priority"], 1.0)
        self.assertEqual(result["accuracy"]["model_relation_priority"], 0.0)
        self.assertEqual(result["delta_vs_model_relation_priority"], 1.0)
        self.assertEqual(len(result["improved_examples"]), 2)


if __name__ == "__main__":
    unittest.main()
