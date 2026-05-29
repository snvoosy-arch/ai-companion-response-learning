from __future__ import annotations

import unittest
from uuid import uuid4
from pathlib import Path

from predictive_bot.core.classifier import HeuristicIntentClassifier, HybridIntentClassifier
from predictive_bot.core.intent_model import CharNgramCentroidModel
from predictive_bot.core.models import ConversationState, Intent


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class IntentModelTests(unittest.TestCase):
    def test_centroid_model_predicts_trained_examples(self) -> None:
        rows = [
            {"text": "안녕", "intent": "greeting"},
            {"text": "하이", "intent": "greeting"},
            {"text": "안뇽", "intent": "greeting"},
            {"text": "설명해줘", "intent": "help"},
            {"text": "설명 좀", "intent": "help"},
            {"text": "기능 알려줘", "intent": "help"},
            {"text": "심심하다", "intent": "smalltalk_generic"},
            {"text": "뭐해", "intent": "smalltalk_generic"},
        ]
        model = CharNgramCentroidModel.train(rows, top_features_per_intent=500)

        self.assertEqual(model.predict("설명 좀").intent, "help")
        self.assertEqual(model.predict("안뇽").intent, "greeting")

    def test_save_and_load_preserves_predictions(self) -> None:
        rows = [
            {"text": "안녕", "intent": "greeting"},
            {"text": "고마워", "intent": "thanks"},
            {"text": "뭐해", "intent": "smalltalk_generic"},
        ]
        model = CharNgramCentroidModel.train(rows, top_features_per_intent=500)

        path = TEST_TMP_ROOT / f"intent_model_{uuid4().hex}.json"
        try:
            model.save(path)
            loaded = CharNgramCentroidModel.load(path)
            self.assertEqual(loaded.predict("안녕").intent, "greeting")
            self.assertEqual(loaded.predict("고마워").intent, "thanks")
        finally:
            path.unlink(missing_ok=True)

    def test_hybrid_classifier_can_upgrade_unknown_to_help(self) -> None:
        rows = [
            {"text": "설명해줘", "intent": "help"},
            {"text": "사용법 알려줘", "intent": "help"},
            {"text": "뭐해", "intent": "smalltalk_generic"},
            {"text": "심심하다", "intent": "smalltalk_generic"},
        ]
        model = CharNgramCentroidModel.train(rows, top_features_per_intent=500)
        classifier = HybridIntentClassifier(
            heuristic=HeuristicIntentClassifier(),
            learned_model=model,
            min_confidence=0.10,
        )

        result = classifier.classify("설명 좀", ConversationState(user_id="u-1"))
        self.assertEqual(result.intent, Intent.HELP)


if __name__ == "__main__":
    unittest.main()
