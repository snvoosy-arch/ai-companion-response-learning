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
    ToolOutcome,
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

    def test_rejects_non_string_json_scalars_instead_of_coercing_them(self) -> None:
        invalid_payloads = (
            {
                "action": "reply",
                "speech": 123,
                "tool_calls": [],
                "memory_candidates": [],
            },
            {
                "action": "use_tool",
                "speech": "",
                "tool_calls": [{"name": 123, "arguments": {"query": "time"}}],
                "memory_candidates": [],
            },
            {
                "action": "use_tool",
                "speech": "",
                "tool_calls": [
                    {"name": "temporal_reasoning", "arguments": {"query": False}}
                ],
                "memory_candidates": [],
            },
            {
                "action": "reply",
                "speech": "응.",
                "tool_calls": [],
                "memory_candidates": [{"kind": "profile", "text": ["value"]}],
            },
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                ActorEnvelope.from_mapping(payload)


class ToolOutcomeTests(unittest.TestCase):
    def test_requires_known_status_and_nonempty_identity(self) -> None:
        invalid_values = (
            {"tool_name": "", "status": "resolved", "evidence_id": "e-1"},
            {"tool_name": "clock", "status": "other", "evidence_id": "e-1"},
            {"tool_name": "clock", "status": "resolved", "evidence_id": ""},
            {"tool_name": 123, "status": "resolved", "evidence_id": "e-1"},
            {
                "tool_name": "clock",
                "status": "resolved",
                "evidence_id": "private token text",
            },
        )

        for values in invalid_values:
            with self.subTest(values=values), self.assertRaises(ContractError):
                ToolOutcome(summary="", **values)


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

    def test_executor_outcome_must_match_requested_tool_identity(self) -> None:
        class MismatchedExecutor:
            def execute(self, call: ToolCall) -> ToolOutcome:
                return ToolOutcome("different_tool", "resolved", "tool-1", "wrong")

        actor = ScriptedActorBackend(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "time"),),
                ),
            )
        )
        delivery = InMemoryDeliverySink()
        runtime = ExternalFirstRuntime(
            actor=actor,
            executor=MismatchedExecutor(),
            delivery=delivery,
        )

        result = runtime.run(
            "시간을 확인해줘.",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("TOOL_OUTCOME_IDENTITY_MISMATCH", result.trace.issues)
        self.assertEqual(result.trace.tool_calls, 1)
        self.assertEqual(result.trace.ledger[-1].status, "identity_mismatch")
        self.assertEqual(delivery.messages, [])

    def test_tool_execution_creates_a_reply_obligation(self) -> None:
        runtime, _, executor, delivery = self.build_runtime(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "time"),),
                ),
                ActorEnvelope(action="silence", speech=""),
            )
        )

        result = runtime.run(
            "시간을 확인해줘.",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("TOOL_RESULT_DISCLOSURE_REQUIRED", result.trace.issues)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(delivery.messages, [])

    def test_actor_exception_becomes_a_structured_block(self) -> None:
        class RaisingActor:
            def generate(self, context):
                raise RuntimeError("private actor detail")

        runtime = ExternalFirstRuntime(
            actor=RaisingActor(),
            executor=FixtureReadOnlyExecutor({}),
            delivery=InMemoryDeliverySink(),
        )

        result = runtime.run("안녕.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "blocked")
        self.assertEqual(result.trace.actor_calls, 1)
        self.assertEqual(result.trace.issues, ["ACTOR_ERROR"])
        self.assertNotIn("private actor detail", str(result.trace.public_dict()))

    def test_non_envelope_actor_output_becomes_a_structured_block(self) -> None:
        class InvalidActor:
            def generate(self, context):
                return {"action": "reply", "speech": "unvalidated"}

        runtime = ExternalFirstRuntime(
            actor=InvalidActor(),
            executor=FixtureReadOnlyExecutor({}),
            delivery=InMemoryDeliverySink(),
        )

        result = runtime.run("안녕.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "blocked")
        self.assertEqual(result.trace.issues, ["ACTOR_INVALID_OUTPUT"])

    def test_executor_exception_becomes_a_structured_block(self) -> None:
        class RaisingExecutor:
            def execute(self, call: ToolCall) -> ToolOutcome:
                raise RuntimeError("private executor detail")

        actor = ScriptedActorBackend(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "time"),),
                ),
            )
        )
        runtime = ExternalFirstRuntime(
            actor=actor,
            executor=RaisingExecutor(),
            delivery=InMemoryDeliverySink(),
        )

        result = runtime.run(
            "시간을 확인해줘.",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertEqual(result.trace.tool_calls, 1)
        self.assertIn("EXECUTOR_ERROR", result.trace.issues)
        self.assertEqual(result.trace.ledger[-1].status, "error")
        self.assertNotIn("private executor detail", str(result.trace.public_dict()))

    def test_non_outcome_executor_result_becomes_a_structured_block(self) -> None:
        class InvalidExecutor:
            def execute(self, call: ToolCall):
                return {"tool_name": call.name, "status": "resolved"}

        actor = ScriptedActorBackend(
            (
                ActorEnvelope(
                    action="use_tool",
                    speech="",
                    tool_calls=(ToolCall("temporal_reasoning", "time"),),
                ),
            )
        )
        runtime = ExternalFirstRuntime(
            actor=actor,
            executor=InvalidExecutor(),
            delivery=InMemoryDeliverySink(),
        )

        result = runtime.run(
            "시간을 확인해줘.",
            AuthoritySnapshot(available_tools=("temporal_reasoning",)),
        )

        self.assertEqual(result.outcome, "blocked")
        self.assertIn("EXECUTOR_INVALID_OUTCOME", result.trace.issues)
        self.assertEqual(result.trace.ledger[-1].status, "invalid")

    def test_delivery_exception_becomes_a_structured_unknown_outcome(self) -> None:
        class RaisingDelivery:
            def deliver(self, speech: str) -> bool:
                raise RuntimeError("private delivery detail")

        runtime = ExternalFirstRuntime(
            actor=ScriptedActorBackend(
                (ActorEnvelope(action="reply", speech="응답이야."),)
            ),
            executor=FixtureReadOnlyExecutor({}),
            delivery=RaisingDelivery(),
        )

        result = runtime.run("답해줘.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "delivery_unknown")
        self.assertTrue(result.trace.delivery_attempted)
        self.assertFalse(result.trace.delivered)
        self.assertIn("DELIVERY_ERROR", result.trace.issues)
        self.assertEqual(result.trace.ledger[-1].status, "unknown")
        self.assertNotIn("private delivery detail", str(result.trace.public_dict()))

    def test_non_boolean_delivery_result_becomes_unknown_outcome(self) -> None:
        class InvalidDelivery:
            def deliver(self, speech: str):
                return "yes"

        runtime = ExternalFirstRuntime(
            actor=ScriptedActorBackend(
                (ActorEnvelope(action="reply", speech="응답이야."),)
            ),
            executor=FixtureReadOnlyExecutor({}),
            delivery=InvalidDelivery(),
        )

        result = runtime.run("답해줘.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "delivery_unknown")
        self.assertIn("DELIVERY_INVALID_RESULT", result.trace.issues)
        self.assertEqual(result.trace.ledger[-1].status, "unknown")

    def test_explicit_delivery_rejection_is_a_confirmed_failure(self) -> None:
        class FailedDelivery:
            def deliver(self, speech: str) -> bool:
                return False

        runtime = ExternalFirstRuntime(
            actor=ScriptedActorBackend(
                (ActorEnvelope(action="reply", speech="응답이야."),)
            ),
            executor=FixtureReadOnlyExecutor({}),
            delivery=FailedDelivery(),
        )

        result = runtime.run("답해줘.", AuthoritySnapshot())

        self.assertEqual(result.outcome, "delivery_failed")
        self.assertNotIn("DELIVERY_ERROR", result.trace.issues)
        self.assertEqual(result.trace.ledger[-1].status, "failed")

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
        self.assertEqual(result.trace.authorized_memory_candidates, 0)
        self.assertEqual(result.trace.rejected_memory_candidates, 1)
        self.assertEqual(len(delivery.messages), 1)

    def test_memory_authorization_is_not_reported_as_persistence(self) -> None:
        runtime, _, _, _ = self.build_runtime(
            (
                ActorEnvelope(
                    action="reply",
                    speech="기억 후보로 검토할게.",
                    memory_candidates=(
                        MemoryCandidate("profile", "사용자는 카페인을 피한다"),
                    ),
                ),
            )
        )

        result = runtime.run(
            "카페인은 피하고 있어.",
            AuthoritySnapshot(
                persistent_memory_allowed=True,
                allowed_memory_kinds=("profile",),
            ),
        )

        self.assertEqual(result.trace.authorized_memory_candidates, 1)
        self.assertEqual(result.trace.rejected_memory_candidates, 0)
        self.assertEqual(result.trace.ledger[0].status, "authorized")
        self.assertNotIn("persisted_memories", result.trace.public_dict())

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
