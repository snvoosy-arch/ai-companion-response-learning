from __future__ import annotations

import unittest

from discord_lmstudio_bot.performance import WhiteOutputPacket, WhitePerformanceBrain, WhiteRuntimeEvent
from discord_lmstudio_bot.web_search import SearchContext, SearchResult


class WhitePerformanceBrainTests(unittest.TestCase):
    def test_soft_prompt_builds_support_packet(self) -> None:
        brain = WhitePerformanceBrain()
        packet = brain.build_turn_packet(
            event=WhiteRuntimeEvent(
                kind="chat_message",
                prompt="오늘 너무 힘들고 지쳤어",
                user_name="viewer",
            ),
            reply="오늘은 조금 천천히 가도 돼.",
        )

        self.assertEqual(packet.speaker, "white")
        self.assertEqual(packet.brain, "white.llm_event_loop")
        self.assertEqual(packet.emotion_state, "soft")
        self.assertEqual(packet.action_intent, "support_reply")
        self.assertEqual(packet.facial_expression, "soft")
        self.assertEqual(packet.voice_style, "white_soft")
        utterance = packet.to_utterance()
        self.assertEqual(utterance.mood, "soft")
        self.assertEqual(utterance.intent, "support_reply")
        self.assertEqual(utterance.metadata["attention_target"], "viewer")
        self.assertEqual(utterance.metadata["white_output_schema"], "white.output.v1")
        self.assertEqual(utterance.metadata["avatar_action"], "slow_nod")
        self.assertEqual(utterance.metadata["mouth_mode"], "speech")

    def test_search_event_builds_grounded_packet(self) -> None:
        brain = WhitePerformanceBrain()
        search_context = SearchContext(
            query="latest model news",
            results=(
                SearchResult(
                    title="Model News",
                    url="https://example.invalid/model",
                    content="short result",
                    published_date=None,
                ),
            ),
            prompt_context="source context",
        )

        packet = brain.build_turn_packet(
            event=WhiteRuntimeEvent(
                kind="slash_search",
                prompt="최신 모델 뉴스 찾아줘",
                user_name="viewer",
                reply_mode="interaction",
                search_used=True,
            ),
            reply="찾아보니 이런 흐름이 보여.",
            search_context=search_context,
        )

        self.assertEqual(packet.emotion_state, "focused")
        self.assertEqual(packet.action_intent, "grounded_search_reply")
        self.assertEqual(packet.facial_expression, "focused")
        self.assertIn("search_query:latest model news", packet.evidence_used)
        self.assertIn("source:Model News", packet.evidence_used)

    def test_duo_event_is_partner_focused_and_not_interruptible(self) -> None:
        brain = WhitePerformanceBrain(max_recent_events=2)
        brain.observe_event(WhiteRuntimeEvent(kind="chat_message", prompt="안녕", user_name="a"))
        brain.observe_event(WhiteRuntimeEvent(kind="chat_message", prompt="고마워", user_name="b"))
        packet = brain.build_turn_packet(
            event=WhiteRuntimeEvent(
                kind="duo_message",
                prompt="ㅋㅋ 이거 봐",
                user_name="black",
                reply_mode="send",
                duo=True,
            ),
            reply="그건 좀 웃겼다.",
        )

        self.assertEqual(packet.action_intent, "duo_reply")
        self.assertEqual(packet.emotion_state, "playful")
        self.assertEqual(packet.metadata["attention_target"], "partner")
        self.assertFalse(packet.can_interrupt)
        self.assertEqual(len(brain.recent_events), 2)

    def test_build_output_packet_exposes_white_runtime_contract(self) -> None:
        brain = WhitePerformanceBrain()
        packet = brain.build_output_packet(
            event=WhiteRuntimeEvent(
                kind="chat_message",
                prompt="안녕. 오늘은 가볍게 인사해줘",
                user_name="viewer",
            ),
            reply="안녕. 오늘은 천천히 있어도 돼.",
        )

        self.assertIsInstance(packet, WhiteOutputPacket)
        self.assertEqual(packet.speaker, "white")
        self.assertEqual(packet.schema_version, "white.output.v1")
        self.assertEqual(packet.emotion, "warm")
        self.assertEqual(packet.avatar_action, "small_bounce")
        self.assertEqual(packet.mouth_mode, "speech")
        self.assertEqual(packet.action_intent, "chat_reply")


if __name__ == "__main__":
    unittest.main()
