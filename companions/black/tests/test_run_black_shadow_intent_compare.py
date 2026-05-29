from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_black_shadow_intent_compare.py"
_SPEC = importlib.util.spec_from_file_location("run_black_shadow_intent_compare", SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"unable to load shadow compare script: {SCRIPT_PATH}")
shadow_compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shadow_compare)


class BlackShadowIntentCompareTests(unittest.TestCase):
    def test_load_shadow_cases_supports_wrapped_payload(self) -> None:
        payload = {
            "name": "black-shadow",
            "version": "1",
            "purpose": "shadow compare",
            "cases": [
                {
                    "id": "BHC001",
                    "cluster": "quiet_day",
                    "turns": [
                        {"speaker": "user", "text": "오늘은 좀 말수가 적은 날 같아."},
                        {"speaker": "white", "text": "말수가 적은 날 같아."},
                        {"speaker": "black", "text": "그럴 수 있어."},
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shadow.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            meta, cases = shadow_compare.load_shadow_cases(path)

        self.assertEqual(meta["name"], "black-shadow")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["id"], "BHC001")
        self.assertEqual(cases[0]["turns"][1]["speaker"], "white")

    def test_extract_replay_turns_excludes_final_black_turn(self) -> None:
        case = {
            "id": "BHC001",
            "turns": [
                {"speaker": "user", "text": "오늘은 좀 말수가 적은 날 같아."},
                {"speaker": "white", "text": "말수가 적은 날 같아."},
                {"speaker": "black", "text": "그럴 수 있어."},
            ],
        }

        replay_turns, observed_black_reply = shadow_compare.extract_replay_turns(case)

        self.assertEqual(
            replay_turns,
            [
                {"speaker": "user", "text": "오늘은 좀 말수가 적은 날 같아."},
                {"speaker": "white", "text": "말수가 적은 날 같아."},
            ],
        )
        self.assertEqual(observed_black_reply, "그럴 수 있어.")

    def test_extract_replay_turns_keeps_user_carryover_after_mid_black_turn(self) -> None:
        case = {
            "id": "BHC006",
            "turns": [
                {"speaker": "user", "text": "대화할 때 자꾸 어색해져."},
                {"speaker": "black", "text": "그럴 수 있어."},
                {"speaker": "user", "text": "맞아. 집에 와서도 그 장면이 계속 맴돌아."},
            ],
        }

        replay_turns, observed_black_reply = shadow_compare.extract_replay_turns(case)

        self.assertEqual(
            replay_turns,
            [
                {"speaker": "user", "text": "대화할 때 자꾸 어색해져."},
                {"speaker": "user", "text": "맞아. 집에 와서도 그 장면이 계속 맴돌아."},
            ],
        )
        self.assertIsNone(observed_black_reply)

    def test_summarize_case_comparison_marks_intent_and_action_changes(self) -> None:
        summary = shadow_compare.summarize_case_comparison(
            baseline_type="heuristic",
            model_results={
                "heuristic": {
                    "final": {
                        "intent": "smalltalk_generic",
                        "question_schema": None,
                        "speech_act": "inform",
                        "action": "continue_conversation",
                        "classifier_source": "heuristic",
                    }
                },
                "charngram": {
                    "final": {
                        "intent": "smalltalk_feeling",
                        "question_schema": "relational_interpretation",
                        "speech_act": "ask",
                        "action": "share_feeling",
                        "classifier_source": "charngram",
                    }
                },
            },
        )

        self.assertTrue(summary["charngram"]["intent_changed"])
        self.assertTrue(summary["charngram"]["schema_changed"])
        self.assertTrue(summary["charngram"]["speech_act_changed"])
        self.assertTrue(summary["charngram"]["action_changed"])
        self.assertTrue(summary["charngram"]["source_changed"])

    def test_build_engine_env_supports_modernbert_without_alias_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "modernbert"
            char_path = Path(tmpdir) / "char.json"
            kcbert_path = Path(tmpdir) / "kcbert"
            model_path.mkdir()
            kcbert_path.mkdir()
            char_path.write_text("{}", encoding="utf-8")

            overrides, skip_reason = shadow_compare._build_engine_env(
                env_values={"BLACK_MODEL_ALIAS": "black.active", "GENERATION_BACKEND": "kobart"},
                model_type="modernbert",
                charngram_model_path=char_path,
                kcbert_model_path=kcbert_path,
                modernbert_model_path=model_path,
            )

        self.assertIsNone(skip_reason)
        self.assertIsNotNone(overrides)
        assert overrides is not None
        self.assertEqual(overrides["BLACK_MODEL_ALIAS"], "")
        self.assertEqual(overrides["INTENT_MODEL_TYPE"], "modernbert_meaning")
        self.assertEqual(overrides["KCBERT_MODEL_PATH"], str(model_path))
        self.assertEqual(overrides["GENERATION_BACKEND"], "template")


if __name__ == "__main__":
    unittest.main()
