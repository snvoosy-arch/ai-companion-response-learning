from __future__ import annotations

import unittest

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    EngineResult,
    Intent,
    MessageFeatures,
    VerificationReport,
    WeatherReport,
)
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder
from predictive_bot.evaluation.runtime_logs import snapshot_to_expectation
from predictive_bot.evaluation.soak import replay_sessions, snapshot_soak_result


class _FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=18.0,
            description="맑음",
            wind_kph=7.0,
        )


class _FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="14:32",
            formatted_date="2026-04-13",
            timezone_name="Asia/Seoul",
            source="fake_clock",
        )


class _FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        items = [
            NewsHeadline(title="첫 번째 헤드라인: AI 반도체 경쟁 심화", source="연합뉴스"),
            NewsHeadline(title="두 번째 헤드라인: 코스피 장중 상승", source="한겨레"),
            NewsHeadline(title="세 번째 헤드라인: LCK 결승 임박", source="한국경제"),
        ]
        return items[:limit]


def _build_engine() -> PredictiveEngine:
    action_selector = ActionSelector(default_location=None)
    return PredictiveEngine(
        classifier=HeuristicIntentClassifier(),
        goal_manager=GoalManager(default_location=None),
        action_selector=action_selector,
        world_state_builder=WorldStateBuilder(),
        policy=HierarchicalPolicy(action_selector=action_selector),
        renderer=ResponseRenderer(llm_client=None),
        verifier=ResponseVerifier(),
        weather_service=_FakeWeatherService(),
        time_service=_FakeTimeService(),
        news_service=_FakeNewsService(),
        state_store=MemoryStateStore(),
    )


class _BadEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def respond(self, user_id: str, text: str) -> EngineResult:
        self.calls += 1
        if self.calls == 1:
            return EngineResult(
                reply="같은 답",
                decision=ActionDecision(action=ActionType.CONTINUE_CONVERSATION, reason="test", goals=[]),
                features=MessageFeatures(
                    content=text,
                    normalized=text,
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    is_question=False,
                ),
                verification=VerificationReport(ok=True),
            )
        return EngineResult(
            reply="같은 답",
            decision=ActionDecision(action=ActionType.EXPLAIN_REASON, reason="test", goals=[]),
            features=MessageFeatures(
                content=text,
                normalized=text,
                intent=Intent.WHY,
                sentiment="neutral",
                is_question=True,
            ),
            verification=VerificationReport(ok=True),
        )


class RuntimeSoakEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = _build_engine()

    async def test_replay_sessions_reports_success_for_mixed_runtime_flow(self) -> None:
        sessions = [
            {
                "session_id": "runtime-weather",
                "category": "grounded",
                "turns": [
                    {"input": "오늘 날씨 어때?", "expect": {"action": "ask_location"}},
                    {"input": "서울", "expect": {"action": "weather_lookup", "reply_contains": ["서울"]}},
                    {"input": "왜?", "expect": {"action": "explain_reason"}},
                ],
            },
            {
                "session_id": "runtime-news-rec",
                "category": "mixed",
                "turns": [
                    {"input": "AI 뉴스 알려줘", "expect": {"action": "news_answer", "news_topic": "ai"}},
                    {"input": "공포영화 좋아해", "expect": {"action": "recommend"}},
                    {"input": "볼 거 추천해줘", "expect": {"action": "recommend", "reply_contains": ["공포영화"]}},
                    {"input": "왜?", "expect": {"action": "explain_reason"}},
                ],
            },
        ]

        report = await replay_sessions(self.engine, sessions)

        self.assertEqual(report["session_count"], 2)
        self.assertEqual(report["session_passed"], 2)
        self.assertEqual(report["failed_sessions"], [])
        self.assertEqual(report["hard_failure_count"], 0)
        self.assertEqual(report["warning_count"], 0)

    async def test_replay_sessions_flags_missing_explanation_trace_and_duplicate_reply(self) -> None:
        sessions = [
            {
                "session_id": "bad-runtime",
                "category": "failure",
                "turns": [
                    {"input": "첫 턴"},
                    {"input": "왜?"},
                ],
            }
        ]

        report = await replay_sessions(_BadEngine(), sessions)

        self.assertEqual(report["session_passed"], 0)
        self.assertIn("bad-runtime", report["failed_sessions"])
        hard_issues = {item["issue"]: item["count"] for item in report["hard_failure_metrics"]}
        warning_issues = {item["issue"]: item["count"] for item in report["warning_metrics"]}
        self.assertEqual(hard_issues.get("missing_explanation_trace"), 1)
        self.assertEqual(warning_issues.get("duplicate_reply"), 1)

    def test_snapshot_soak_result_marks_grounded_and_explanation_health(self) -> None:
        result = EngineResult(
            reply="지금 시간은 14:32이야.",
            decision=ActionDecision(
                action=ActionType.TELL_TIME,
                reason="grounded time",
                goals=[],
                slots={"knowledge_grounded": "true", "knowledge_source": "fake_clock"},
            ),
            features=MessageFeatures(
                content="지금 몇시야?",
                normalized="지금 몇시야?",
                intent=Intent.TIME_DATE,
                sentiment="neutral",
                is_question=True,
            ),
            verification=VerificationReport(ok=True),
        )

        snapshot = snapshot_soak_result(result, previous_input=None, previous_reply=None)

        self.assertTrue(snapshot["knowledge_grounded"])
        self.assertEqual(snapshot["knowledge_source"], "fake_clock")
        self.assertEqual(snapshot["hard_failures"], [])
        self.assertEqual(snapshot["warnings"], [])

    def test_snapshot_to_expectation_keeps_structural_fields(self) -> None:
        snapshot = {
            "intent": "weather",
            "action": "weather_lookup",
            "speech_act": "ask",
            "topic_hint": "weather",
            "news_topic": None,
            "conversation_mode": "tool_grounded",
            "unresolved_need": None,
            "risk_level": "medium",
            "boundary_history": "clear",
            "user_directness_style": "balanced",
            "rapport_bucket": "neutral",
            "decision_module": "weather",
            "explanation_mode": "short",
            "verification_ok": True,
            "response_needs": ["grounding"],
            "pragmatic_cues": ["casual_probe"],
            "constraints": ["do_not_guess_facts"],
            "counterfactual_actions": ["ask_location"],
            "reason_codes": ["intent_weather", "policy_candidates_considered"],
            "logic_rule_ids": ["decision.weather_lookup"],
        }

        expect = snapshot_to_expectation(snapshot)

        self.assertEqual(expect["intent"], "weather")
        self.assertEqual(expect["action"], "weather_lookup")
        self.assertEqual(expect["decision_module"], "weather")
        self.assertEqual(expect["verification_ok"], True)
        self.assertEqual(expect["response_needs"], ["grounding"])
        self.assertEqual(expect["pragmatic_cues"], ["casual_probe"])
        self.assertEqual(expect["constraints"], ["do_not_guess_facts"])
        self.assertEqual(expect["counterfactual_actions"], ["ask_location"])
        self.assertEqual(expect["reason_code_prefixes"], ["intent_weather", "policy_candidates_considered"])
        self.assertEqual(expect["logic_rule_prefixes"], ["decision.weather_lookup"])


if __name__ == "__main__":
    unittest.main()
