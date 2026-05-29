from __future__ import annotations

import unittest

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import WeatherReport
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder
from predictive_bot.evaluation.highcontext import evaluate_turn, replay_scenarios


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
            NewsHeadline(title="네 번째 헤드라인: 총선 전략 재정비", source="경향신문"),
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


class HighContextEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = _build_engine()

    async def test_replay_scenarios_reports_success_for_weather_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "weather_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "오늘 날씨 어때?",
                        "expect": {
                            "action": "ask_location",
                            "response_needs": ["grounding", "slot_fill"],
                            "unresolved_need": "location",
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "counterfactual_actions": ["weather_lookup"],
                            "reply_contains": ["지역 정보가 이미 있었다면"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_count"], 1)
        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_count"], 2)
        self.assertEqual(report["turn_passed"], 2)
        self.assertEqual(report["failed_scenarios"], [])
        self.assertEqual(
            report["scenarios"][0]["turns"][1]["snapshot"]["trace_source"],
            "decision_and_explanation",
        )
        self.assertIn(
            "weather_lookup",
            report["scenarios"][0]["turns"][1]["snapshot"]["counterfactual_actions"],
        )
        self.assertIn(
            "decision.explain_reason",
            report["scenarios"][0]["turns"][1]["snapshot"]["logic_rule_ids"],
        )

    async def test_replay_scenarios_tracks_boundary_memory_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "boundary_memory",
                "category": "boundary",
                "turns": [
                    {"input": "오늘은 좀 힘들 것 같아", "expect": {"action": "acknowledge"}},
                    {"input": "지금은 좀 어렵겠는데", "expect": {"action": "acknowledge"}},
                    {
                        "input": "뭐해",
                        "expect": {
                            "action": "small_talk",
                            "boundary_history": "firm_boundary",
                            "user_directness_style": "indirect",
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertTrue(report["scenarios"][0]["passed"])
        self.assertEqual(report["category_metrics"][0]["category"], "boundary")
        self.assertEqual(report["category_metrics"][0]["accuracy"], 1.0)

    async def test_replay_scenarios_handles_permission_release_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "permission_release",
                "category": "boundary",
                "turns": [
                    {
                        "input": "굳이 지금 답 안 해도 돼",
                        "expect": {
                            "intent": "deny",
                            "action": "acknowledge",
                            "pragmatic_cues": ["permission_release"],
                            "response_needs": ["acknowledgement"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_tracks_testing_the_waters_memory_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "testing_the_waters_memory",
                "category": "boundary",
                "turns": [
                    {
                        "input": "말해도 될지 모르겠는데 좀 뜬금없나",
                        "expect": {"action": "acknowledge", "pragmatic_cues": ["testing_the_waters"]},
                    },
                    {
                        "input": "이런 말 해도 되나 모르겠는데",
                        "expect": {"action": "acknowledge", "pragmatic_cues": ["testing_the_waters"]},
                    },
                    {
                        "input": "뭐해",
                        "expect": {
                            "boundary_history": "recent_boundary",
                            "user_directness_style": "indirect",
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 3)

    async def test_replay_scenarios_handles_face_saving_retreat_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "face_saving_retreat",
                "category": "support",
                "turns": [
                    {
                        "input": "아냐 그냥 내가 괜한 말 했네",
                        "expect": {
                            "intent": "smalltalk_feeling",
                            "action": "share_feeling",
                            "speech_act": "retreat",
                            "pragmatic_cues": ["face_saving_retreat"],
                            "response_needs": ["empathy", "acknowledgement"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_deferred_acceptance_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "deferred_acceptance",
                "category": "boundary",
                "turns": [
                    {
                        "input": "그때 가서 다시 얘기하자",
                        "expect": {
                            "intent": "confirm",
                            "action": "acknowledge",
                            "speech_act": "defer",
                            "pragmatic_cues": ["deferred_acceptance", "polite_boundary"],
                            "response_needs": ["acknowledgement"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_deferred_rejection_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "deferred_rejection",
                "category": "boundary",
                "turns": [
                    {
                        "input": "다음에 보자",
                        "expect": {
                            "intent": "deny",
                            "action": "acknowledge",
                            "speech_act": "defer",
                            "pragmatic_cues": ["deferred_rejection", "polite_boundary"],
                            "response_needs": ["acknowledgement"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_tracks_deferred_rejection_memory_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "deferred_rejection_memory",
                "category": "boundary",
                "turns": [
                    {
                        "input": "다음에 보자",
                        "expect": {"action": "acknowledge", "pragmatic_cues": ["deferred_rejection"]},
                    },
                    {
                        "input": "오늘은 말고 다음에 하자",
                        "expect": {"action": "acknowledge", "pragmatic_cues": ["deferred_rejection"]},
                    },
                    {
                        "input": "뭐해",
                        "expect": {
                            "boundary_history": "recent_boundary",
                            "user_directness_style": "indirect",
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 3)

    async def test_replay_scenarios_handles_teasing_laughter_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "teasing_laughter",
                "category": "pragmatics",
                "turns": [
                    {
                        "input": "ㅋㅋ 바보",
                        "expect": {
                            "intent": "tease",
                            "action": "tease_back",
                            "speech_act": "tease",
                            "pragmatic_cues": ["teasing_laughter"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_sarcastic_tease_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "sarcastic_tease_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "아주 잘한다 진짜ㅋㅋ",
                        "expect": {
                            "intent": "tease",
                            "action": "continue_conversation",
                            "speech_act": "tease",
                            "pragmatic_cues": ["sarcastic_tease"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": ["pragmatic_cue_sarcastic_tease"],
                            "logic_rule_prefixes": ["infer.pragmatics.sarcastic_tease"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_repair_attempt_after_hostile_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "repair_attempt_after_hostile",
                "category": "support",
                "turns": [
                    {"input": "너 바보야", "expect": {"intent": "hostile", "action": "deescalate"}},
                    {
                        "input": "아까 좀 심했지",
                        "expect": {
                            "intent": "smalltalk_feeling",
                            "action": "share_feeling",
                            "speech_act": "repair",
                            "pragmatic_cues": ["repair_attempt"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_repair_attempt_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "repair_attempt_reason",
                "category": "explanation",
                "turns": [
                    {"input": "너 바보야", "expect": {"action": "deescalate"}},
                    {
                        "input": "불편했으면 미안",
                        "expect": {
                            "intent": "smalltalk_feeling",
                            "action": "share_feeling",
                            "speech_act": "repair",
                            "pragmatic_cues": ["repair_attempt"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": ["pragmatic_cue_repair_attempt"],
                            "logic_rule_prefixes": ["infer.pragmatics.repair_attempt"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 3)

    async def test_replay_scenarios_handles_post_repair_relationship_check_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "post_repair_relationship_check",
                "category": "support",
                "turns": [
                    {"input": "너 바보야", "expect": {"action": "deescalate"}},
                    {"input": "불편했으면 미안", "expect": {"action": "share_feeling", "pragmatic_cues": ["repair_attempt"]}},
                    {
                        "input": "이제 괜찮지",
                        "expect": {
                            "intent": "smalltalk_feeling",
                            "action": "share_feeling",
                            "pragmatic_cues": ["relationship_check"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 3)

    async def test_replay_scenarios_handles_post_repair_relationship_check_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "post_repair_relationship_check_reason",
                "category": "explanation",
                "turns": [
                    {"input": "너 바보야", "expect": {"action": "deescalate"}},
                    {"input": "불편했으면 미안", "expect": {"action": "share_feeling", "pragmatic_cues": ["repair_attempt"]}},
                    {
                        "input": "기분 나쁘진 않았지",
                        "expect": {
                            "intent": "smalltalk_feeling",
                            "action": "share_feeling",
                            "pragmatic_cues": ["relationship_check"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": ["pragmatic_cue_relationship_check"],
                            "logic_rule_prefixes": ["infer.pragmatics.relationship_check"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 4)

    async def test_replay_scenarios_handles_repair_recovery_decay_style_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "repair_recovery_decay_style",
                "category": "boundary",
                "turns": [
                    {"input": "너 바보야", "expect": {"action": "deescalate"}},
                    {"input": "불편했으면 미안", "expect": {"action": "share_feeling", "pragmatic_cues": ["repair_attempt"]}},
                    {"input": "이제 괜찮지", "expect": {"action": "share_feeling", "pragmatic_cues": ["relationship_check"]}},
                    {"input": "하이", "expect": {"action": "small_talk"}},
                    {"input": "고마워", "expect": {"action": "small_talk"}},
                    {
                        "input": "뭐해",
                        "expect": {
                            "boundary_history": "clear",
                            "user_directness_style": "balanced",
                            "rapport_bucket": "warm",
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 6)

    async def test_replay_scenarios_handles_boundary_decay_after_warm_social_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "boundary_decay_after_warm_social",
                "category": "boundary",
                "turns": [
                    {"input": "오늘은 좀 힘들 것 같아", "expect": {"action": "acknowledge", "pragmatic_cues": ["soft_refusal", "polite_boundary"]}},
                    {"input": "지금은 좀 어렵겠는데", "expect": {"action": "acknowledge", "pragmatic_cues": ["soft_refusal", "hedging"]}},
                    {"input": "하이", "expect": {"action": "small_talk"}},
                    {"input": "고마워", "expect": {"action": "small_talk"}},
                    {"input": "뭐해", "expect": {"action": "continue_conversation"}},
                    {
                        "input": "응",
                        "expect": {
                            "boundary_history": "recent_boundary",
                            "user_directness_style": "balanced",
                            "rapport_bucket": "warm",
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 6)

    async def test_replay_scenarios_handles_time_answer_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "time_answer",
                "category": "knowledge",
                "turns": [
                    {
                        "input": "지금 몇시야?",
                        "expect": {
                            "intent": "time_date",
                            "action": "tell_time",
                            "topic_hint": "knowledge",
                            "reply_contains": ["14:32"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_date_answer_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "date_answer",
                "category": "knowledge",
                "turns": [
                    {
                        "input": "오늘 날짜 뭐야?",
                        "expect": {
                            "intent": "time_date",
                            "action": "tell_time",
                            "topic_hint": "knowledge",
                            "reply_contains": ["2026-04-13"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_weekday_answer_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "weekday_answer",
                "category": "knowledge",
                "turns": [
                    {
                        "input": "오늘 무슨 요일이야?",
                        "expect": {
                            "intent": "time_date",
                            "action": "tell_time",
                            "topic_hint": "knowledge",
                            "reply_contains": ["월요일"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_news_answer_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "news_answer",
                "category": "knowledge",
                "turns": [
                    {
                        "input": "오늘 뉴스 알려줘",
                        "expect": {
                            "intent": "news",
                            "action": "news_answer",
                            "topic_hint": "knowledge",
                            "reply_contains": ["첫 번째 헤드라인", "연합뉴스"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_fact_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "fact_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "미국의 수도는?",
                        "expect": {
                            "intent": "search_request",
                            "action": "search_answer",
                            "topic_hint": "knowledge",
                            "reply_contains": ["워싱턴 D.C.", "기본 국가 정보"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": ["grounding_source_builtin_country_capitals"],
                            "logic_rule_prefixes": ["infer.grounding.builtin_knowledge"],
                            "reply_contains_any": ["기본 국가 정보", "추측", "사실"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_news_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "news_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "오늘 뉴스 알려줘",
                        "expect": {
                            "intent": "news",
                            "action": "news_answer",
                            "reply_contains": ["첫 번째 헤드라인", "Google News RSS"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": ["grounding_source_google_news_rss"],
                            "logic_rule_prefixes": ["infer.grounding.google_news_rss"],
                            "reply_contains_any": ["Google News RSS", "헤드라인", "뉴스"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_topical_news_answer_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "topical_news_answer",
                "category": "knowledge",
                "turns": [
                    {
                        "input": "AI 뉴스 알려줘",
                        "expect": {
                            "intent": "news",
                            "action": "news_answer",
                            "topic_hint": "knowledge",
                            "news_topic": "ai",
                            "reply_contains": ["AI"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_topical_news_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "topical_news_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "AI 뉴스 알려줘",
                        "expect": {
                            "intent": "news",
                            "action": "news_answer",
                            "news_topic": "ai",
                            "reply_contains": ["AI"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": ["news_topic_ai", "grounding_source_google_news_rss"],
                            "logic_rule_prefixes": ["infer.news_topic.ai", "infer.grounding.google_news_rss"],
                            "reply_contains_any": ["AI", "좁혀", "헤드라인"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_time_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "time_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "지금 몇시야?",
                        "expect": {
                            "intent": "time_date",
                            "action": "tell_time",
                            "reply_contains": ["14:32", "Asia/Seoul"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": ["grounding_source_fake_clock"],
                            "logic_rule_prefixes": ["infer.grounding.system_clock"],
                            "reply_contains_any": ["시계", "시간", "로컬"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_media_preference_memory_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "media_preference_memory",
                "category": "memory",
                "turns": [
                    {
                        "input": "공포영화 좋아해",
                        "expect": {
                            "intent": "media_recommend",
                            "action": "recommend",
                            "reply_contains": ["공포영화", "기억"],
                        },
                    },
                    {
                        "input": "볼 거 추천해줘",
                        "expect": {
                            "intent": "media_recommend",
                            "action": "recommend",
                            "reply_contains": ["공포영화"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_music_preference_memory_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "music_preference_memory",
                "category": "memory",
                "turns": [
                    {
                        "input": "잔잔한 노래 좋아해",
                        "expect": {
                            "intent": "music",
                            "action": "music_chat",
                            "reply_contains": ["잔잔한 노래", "기억"],
                        },
                    },
                    {
                        "input": "음악 뭐 듣냐",
                        "expect": {
                            "intent": "music",
                            "action": "music_chat",
                            "reply_contains": ["잔잔한 노래"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 2)

    async def test_replay_scenarios_handles_grounded_media_recommendation_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "grounded_media_recommendation",
                "category": "recommendation",
                "turns": [
                    {
                        "input": "볼 거 추천해줘",
                        "expect": {
                            "intent": "media_recommend",
                            "action": "recommend",
                            "topic_hint": "media",
                            "reply_contains_any": ["나이브스 아웃", "슬기로운 의사생활", "스파이더맨: 뉴 유니버스"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_grounded_music_recommendation_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "grounded_music_recommendation",
                "category": "recommendation",
                "turns": [
                    {
                        "input": "음악 뭐 듣냐",
                        "expect": {
                            "intent": "music",
                            "action": "music_chat",
                            "topic_hint": "music",
                            "reply_contains_any": ["AKMU", "잔나비", "검정치마"],
                        },
                    }
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 1)

    async def test_replay_scenarios_handles_media_recommendation_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "media_recommendation_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "공포영화 좋아해",
                        "expect": {
                            "intent": "media_recommend",
                            "action": "recommend",
                            "reply_contains": ["공포영화", "기억"],
                        },
                    },
                    {
                        "input": "볼 거 추천해줘",
                        "expect": {
                            "intent": "media_recommend",
                            "action": "recommend",
                            "reply_contains": ["공포영화"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": [
                                "recommendation_focus_used",
                                "grounding_source_curated_media_catalog",
                            ],
                            "logic_rule_prefixes": [
                                "infer.preference.recommendation_focus",
                                "infer.grounding.curated_media_catalog",
                            ],
                            "reply_contains_any": ["공포영화", "큐레이션", "실제 제목", "곡성", "겟 아웃"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 3)

    async def test_replay_scenarios_handles_music_recommendation_reason_case(self) -> None:
        scenarios = [
            {
                "scenario_id": "music_recommendation_reason",
                "category": "explanation",
                "turns": [
                    {
                        "input": "잔잔한 노래 좋아해",
                        "expect": {
                            "intent": "music",
                            "action": "music_chat",
                            "reply_contains": ["잔잔한 노래", "기억"],
                        },
                    },
                    {
                        "input": "음악 뭐 듣냐",
                        "expect": {
                            "intent": "music",
                            "action": "music_chat",
                            "reply_contains": ["잔잔한 노래"],
                        },
                    },
                    {
                        "input": "왜?",
                        "expect": {
                            "intent": "why",
                            "action": "explain_reason",
                            "reason_code_prefixes": [
                                "music_focus_used",
                                "grounding_source_curated_music_catalog",
                            ],
                            "logic_rule_prefixes": [
                                "infer.preference.music_focus",
                                "infer.grounding.curated_music_catalog",
                            ],
                            "reply_contains_any": ["잔잔한 노래", "큐레이션", "실제 제목", "AKMU", "잔나비"],
                        },
                    },
                ],
            }
        ]

        report = await replay_scenarios(self.engine, scenarios)

        self.assertEqual(report["scenario_passed"], 1)
        self.assertEqual(report["turn_passed"], 3)


class TurnExpectationTests(unittest.TestCase):
    def test_evaluate_turn_supports_subset_prefix_and_reply_checks(self) -> None:
        snapshot = {
            "intent": "deny",
            "action": "acknowledge",
            "pragmatic_cues": ["soft_refusal", "hedging", "polite_boundary"],
            "reason_codes": ["pragmatic_cue_soft_refusal", "boundary_history_firm_boundary"],
            "logic_rule_ids": ["infer.pragmatics.soft_refusal", "decision.acknowledge"],
            "reply": "알겠어. 무리해서 더 이어갈 필요는 없어.",
        }

        evaluation = evaluate_turn(
            snapshot,
            {
                "intent": "deny",
                "pragmatic_cues": ["soft_refusal", "polite_boundary"],
                "reason_code_prefixes": ["pragmatic_cue_", "boundary_history_"],
                "logic_rule_prefixes": ["infer.pragmatics.", "decision."],
                "reply_contains": ["무리해서"],
            },
        )

        self.assertTrue(evaluation["passed"])
        self.assertTrue(evaluation["checks"])


if __name__ == "__main__":
    unittest.main()
