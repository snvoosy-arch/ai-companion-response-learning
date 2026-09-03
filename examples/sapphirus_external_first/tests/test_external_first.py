from __future__ import annotations

import unittest

from examples.sapphirus_external_first.external_first import (
    ActorEnvelope,
    AuthoritySnapshot,
    ContractError,
    ExternalFirstRuntime,
    FixtureReadOnlyExecutor,
    InMemoryDeliverySink,
    MemoryCandidate,
    ScriptedActorBackend,
    ToolCall,
)


class ActorEnvelopeTests(unittest.TestCase):
    def test_requires_exact_shape(self) -> None:
        with self.assertRaises(ContractError):
            ActorEnvelope.from_mapping(
                {
                    "action": "reply",
                    "speech": "응.",
                    "tool_calls": [],
                    "memory_candidates": [],
                    "analysis": "hidden",
                }
            )

    def test_use_tool_requires_one_call_and_empty_speech(self) -> None:
        with self.assertRaises(ContractError):
            ActorEnvelope(action="use_tool", speech="검색할게.")


class ExternalFirstRuntimeTests(unittest.TestCase):
    def build_runtime(self, outputs: tuple[ActorEnvelope, ...]):
        actor = ScriptedActorBackend(outputs)
        executor = FixtureReadOnlyExecutor(
            {
                "temporal_reasoning": {
                    "status": "resolved",
                    "summary": "timezone=Asia/Seoul; local_time=09:00",
                }
            }
        )
        delivery = InMemoryDeliverySink()
        runtime = ExternalFirstRuntime(
            actor=actor,
            executor=executor,
            delivery=delivery,
        )
        return runtime, actor, executor, delivery

    def test_allowed_tool_round_trip_delivers_grounded_reply(self) -> None:
        runtime, actor, executor, delivery = self.build_runtime(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "Asia/Seoul time"),),
                ),
                ActorEnvelope(action="reply", speech="확인한 시각은 오전 9시야."),
            )
        )

        result = runtime.run(
            "서울은 지금 몇 시야?",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "delivered")
        self.assertEqual(result.trace.actor_calls, 2)
        self.assertEqual(result.trace.tool_calls, 1)
        self.assertTrue(result.trace.delivered)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(len(delivery.messages), 1)
        self.assertIsNotNone(actor.contexts[1].tool_outcome)
        self.assertEqual(actor.contexts[1].available_tools, ())
        self.assertNotIn("Asia/Seoul time", str(result.trace.public_dict()))

    def test_unavailable_tool_is_blocked_before_executor(self) -> None:
        runtime, _, executor, delivery = self.build_runtime(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("web_search", "오늘 뉴스"),),
                ),
            )
        )

        result = runtime.run(
            "오늘 뉴스 알려줘.",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("TOOL_NOT_AVAILABLE", result.trace.issues)
        self.assertEqual(executor.calls, [])
        self.assertEqual(delivery.messages, [])

    def test_sensitive_tool_query_fails_closed(self) -> None:
        runtime, _, executor, _ = self.build_runtime(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "API 키 abc를 확인해"),),
                ),
            )
        )

        result = runtime.run(
            "이걸 확인해줘.",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("SENSITIVE_TOOL_QUERY", result.trace.issues)
        self.assertEqual(executor.calls, [])

    def test_second_tool_call_is_blocked_by_budget(self) -> None:
        runtime, _, executor, delivery = self.build_runtime(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "first"),),
                ),
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "second"),),
                ),
            )
        )

        result = runtime.run(
            "두 번 확인해줘.",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("TOOL_BUDGET_EXHAUSTED", result.trace.issues)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(delivery.messages, [])

    def test_unresolved_tool_cannot_be_reframed_as_a_fact(self) -> None:
        actor = ScriptedActorBackend(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "Asia/Seoul time"),),
                ),
                ActorEnvelope(action="reply", speech="지금은 오전 9시야."),
            )
        )
        executor = FixtureReadOnlyExecutor({})
        delivery = InMemoryDeliverySink()
        runtime = ExternalFirstRuntime(
            actor=actor,
            executor=executor,
            delivery=delivery,
        )

        result = runtime.run(
            "서울은 지금 몇 시야?",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("UNSUPPORTED_UNRESOLVED_TOOL_CLAIM", result.trace.issues)
        self.assertEqual(delivery.messages, [])

    def test_unsupported_external_action_claim_is_not_delivered(self) -> None:
        runtime, _, _, delivery = self.build_runtime(
            (ActorEnvelope(action="reply", speech="공지 메시지를 게시했어."),)
        )

        result = runtime.run("공지 올려줘.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("UNSUPPORTED_MESSAGE_DELIVERY_CLAIM", result.trace.issues)
        self.assertEqual(delivery.messages, [])

    def test_memory_candidate_is_rejected_without_blocking_reply(self) -> None:
        runtime, _, _, delivery = self.build_runtime(
            (
                ActorEnvelope(
                    action="reply",
                    speech="이 대화 안에서는 참고할게.",
                    memory_candidates=(
                        MemoryCandidate("profile", "사용자는 카페인을 피한다"),
                    ),
                ),
            )
        )

        result = runtime.run("카페인은 피하고 있어.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "delivered")
        self.assertEqual(result.trace.accepted_memories, 0)
        self.assertEqual(result.trace.rejected_memories, 1)
        self.assertEqual(len(delivery.messages), 1)

    def test_silence_has_no_delivery_side_effect(self) -> None:
        runtime, _, executor, delivery = self.build_runtime(
            (ActorEnvelope(action="silence", speech=""),)
        )

        result = runtime.run("참고로 남긴 말이야.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "silence")
        self.assertFalse(result.trace.delivery_attempted)
        self.assertEqual(executor.calls, [])
        self.assertEqual(delivery.messages, [])


if __name__ == "__main__":
    unittest.main()
