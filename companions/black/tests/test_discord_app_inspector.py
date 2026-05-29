from __future__ import annotations

import unittest
from types import SimpleNamespace

from predictive_bot.core.memory import DurableMemoryBucket, DurableMemoryEntry
from predictive_bot.core.models import (
    ActionType,
    ConversationState,
    DecisionTrace,
    Intent,
    StateInferenceEntry,
    TurnRecord,
)
from predictive_bot.discord_app.inspector import (
    format_black_dashboard,
    format_black_recall,
    format_black_state,
    format_black_summary,
)


class BlackInspectorTests(unittest.TestCase):
    def test_black_recall_selects_relevant_memory(self) -> None:
        state = ConversationState(
            user_id="user-1",
            turn_count=24,
            durable_memory=[
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.COMPARISON,
                    text="친구 잘되는 거 보고 조금 씁쓸했다.",
                    captured_turn=20,
                ),
                DurableMemoryEntry(
                    bucket=DurableMemoryBucket.PREFERENCE,
                    text="맵고 자극적인 음식은 별로 안 좋아한다.",
                    captured_turn=8,
                ),
            ],
        )

        report = format_black_recall(state=state, query="비교되고 씁쓸한 기분")

        self.assertIn("comparison", report)
        self.assertIn("씁쓸", report)

    def test_black_dashboard_and_state_include_world_snapshot(self) -> None:
        state = ConversationState(
            user_id="user-1",
            turn_count=12,
            last_intent=Intent.SMALLTALK_FEELING,
            last_action=ActionType.SHARE_FEELING,
            durable_memory=[
                DurableMemoryEntry(bucket=DurableMemoryBucket.OPEN_LOOP, text="면접 결과를 기다리고 있다.")
            ],
            recent_turns=[
                TurnRecord(
                    user_text="결과가 아직 안 왔어",
                    bot_text="조금 더 기다리는 중이구나.",
                    action=ActionType.SHARE_FEELING,
                    decision_reason="support",
                )
            ],
        )
        latest_trace = DecisionTrace(
            decision_id="trace-1",
            user_id="user-1",
            input_text="결과가 아직 안 왔어",
            input_intent=Intent.SMALLTALK_FEELING,
            input_sentiment="negative",
            selected_action=ActionType.SHARE_FEELING,
            selected_reason="support",
            world_state_snapshot={
                "user_emotion": "negative",
                "conversation_mode": "support",
                "tension_bucket": "elevated",
                "rapport_bucket": "warm",
                "boundary_history": "soft",
                "user_directness_style": "mixed",
                "memory_summary": "면접 결과를 기다리는 open loop가 있다.",
                "relevant_open_loops": "면접 결과를 기다리고 있다.",
            },
            state_inference_trace=[
                StateInferenceEntry(
                    field="user_emotion",
                    value="negative",
                    reasons=["negative sentiment and waiting language were both present"],
                )
            ],
        )

        config = SimpleNamespace(
            runtime_state_enabled=True,
            tts_enabled=True,
            tts_mode="discord_voice",
            kobart_input_mode="slim",
        )

        dashboard = format_black_dashboard(
            config=config,
            runtime_report="black runtime alive",
            state=state,
            latest_trace=latest_trace,
        )
        summary = format_black_summary(state=state, latest_trace=latest_trace)
        state_report = format_black_state(state=state, latest_trace=latest_trace)

        self.assertIn("emotion: negative", dashboard)
        self.assertIn("memory_summary", summary)
        self.assertIn("open_loops", state_report)
        self.assertIn("user_emotion=negative", state_report)


if __name__ == "__main__":
    unittest.main()
