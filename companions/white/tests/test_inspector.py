from __future__ import annotations

import unittest

from discord_lmstudio_bot.inspector import (
    analyze_recent_signals,
    build_white_recall_report,
    build_white_signal_report,
)
from discord_lmstudio_bot.memory_store import ConversationSummary, DurableMemory, StoredMessage


class WhiteInspectorTests(unittest.TestCase):
    def test_analyze_recent_signals_marks_heavy_when_negative_markers_dominate(self) -> None:
        messages = [
            StoredMessage(
                id=1,
                guild_id=10,
                channel_id=20,
                user_id=30,
                user_name="tester",
                role="user",
                content="요즘 너무 불안하고 답답해서 잠도 잘 못 자.",
                created_at="2026-04-17 10:00:00",
            ),
            StoredMessage(
                id=2,
                guild_id=10,
                channel_id=20,
                user_id=30,
                user_name="tester",
                role="user",
                content="계속 무겁고 지친 느낌이야.",
                created_at="2026-04-17 10:05:00",
            ),
        ]
        snapshot = analyze_recent_signals(messages)
        self.assertEqual(snapshot.tone, "heavy")
        self.assertGreater(snapshot.negative_hits, snapshot.positive_hits)

    def test_build_white_reports_include_memory_and_summary_hint(self) -> None:
        summary = ConversationSummary(
            id=1,
            guild_id=10,
            channel_id=20,
            summary_text="사용자는 면접 결과를 기다리며 긴장하고 있다.",
            source_until_message_id=4,
            updated_at="2026-04-17 11:00:00",
        )
        memories = [
            DurableMemory(
                id=1,
                guild_id=10,
                channel_id=20,
                user_id=30,
                user_name="tester",
                scope_key="user:30",
                source_kind="user_note",
                memory_kind="open_loop",
                memory_text="면접 결과가 아직 안 나와서 계속 불안하다.",
                source_message_id=3,
                updated_at="2026-04-17 10:55:00",
            )
        ]
        recall = build_white_recall_report(query="면접 결과", summary=summary, memories=memories)
        signals = build_white_signal_report(recent_messages=[], memories=memories)

        self.assertIn("summary_hint", recall)
        self.assertIn("면접 결과", recall)
        self.assertIn("memory_focus", signals)


if __name__ == "__main__":
    unittest.main()
