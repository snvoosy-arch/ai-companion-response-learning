from __future__ import annotations

import unittest

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    Intent,
    MessageFeatures,
    WorldState,
)
from predictive_bot.core.phrasing import build_phrasing_plan


class PhrasingPlanTests(unittest.TestCase):
    def _features(self, content: str, intent: Intent) -> MessageFeatures:
        return MessageFeatures(
            content=content,
            normalized=content,
            intent=intent,
            sentiment="neutral",
            is_question="?" in content,
        )

    def test_boundary_continue_conversation_disables_followup(self) -> None:
        plan = build_phrasing_plan(
            features=self._features("그냥 와봤어", Intent.SMALLTALK_GENERIC),
            decision=ActionDecision(action=ActionType.CONTINUE_CONVERSATION, reason="follow up", goals=[]),
            state=ConversationState(user_id="u1", turn_count=5),
            world_state=WorldState(
                user_id="u1",
                dominant_intent=Intent.SMALLTALK_GENERIC,
                user_emotion="guarded",
                conversation_mode="daily_chat",
                turn_count_bucket="mid",
                tension_bucket="low",
                rapport_bucket="cool",
                boundary_history="active_boundary",
                user_directness_style="indirect",
                last_intent_hint=None,
                last_action_hint=None,
                unresolved_need=None,
                factuality_required=False,
                risk_level="low",
                memory_summary="",
            ),
        )

        self.assertFalse(plan.asks_followup)
        self.assertIn("boundary_active", plan.notes)
        self.assertIn("indirect_user", plan.notes)
        self.assertIn("mid_conversation", plan.notes)

    def test_share_feeling_without_boundary_keeps_followup_open(self) -> None:
        plan = build_phrasing_plan(
            features=self._features("오늘 좀 우울해", Intent.SMALLTALK_FEELING),
            decision=ActionDecision(action=ActionType.SHARE_FEELING, reason="emotion reply", goals=[]),
            state=ConversationState(user_id="u1", turn_count=1),
            world_state=None,
        )

        self.assertTrue(plan.asks_followup)
        self.assertIn("emotion_followup", plan.notes)


if __name__ == "__main__":
    unittest.main()
