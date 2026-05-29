from __future__ import annotations

import unittest
from urllib.parse import parse_qs, urlparse

import httpx

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import WeatherReport
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.tools import BasicKnowledgeService, KnowledgeAnswer, WikidataKnowledgeService
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder


class _FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=20.0,
            description="맑음",
            wind_kph=4.0,
        )


class _StubFallbackKnowledgeService:
    def answer(self, question: str) -> KnowledgeAnswer:
        return KnowledgeAnswer(
            question=question,
            query_type="capital",
            subject="룩셈부르크",
            answer="룩셈부르크",
            source="stub_fallback",
        )


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
        knowledge_service=BasicKnowledgeService(),
        state_store=MemoryStateStore(),
    )


class BasicKnowledgeServiceTests(unittest.TestCase):
    def test_capital_lookup_returns_grounded_answer(self) -> None:
        service = BasicKnowledgeService()

        answer = service.answer("미국의 수도는?")

        self.assertEqual(answer.subject, "미국")
        self.assertEqual(answer.answer, "워싱턴 D.C.")
        self.assertEqual(answer.source, "builtin_country_capitals")

    def test_flag_lookup_returns_grounded_answer(self) -> None:
        service = BasicKnowledgeService()

        answer = service.answer("일본의 국기는?")

        self.assertEqual(answer.query_type, "flag")
        self.assertEqual(answer.subject, "일본")
        self.assertEqual(answer.answer, "🇯🇵")

    def test_location_lookup_returns_grounded_answer(self) -> None:
        service = BasicKnowledgeService()

        answer = service.answer("한국의 위치는?")

        self.assertEqual(answer.query_type, "location")
        self.assertEqual(answer.subject, "한국")
        self.assertIn("동아시아", answer.answer)

    def test_unknown_fact_uses_fallback_service(self) -> None:
        service = BasicKnowledgeService(fallback_service=_StubFallbackKnowledgeService())

        answer = service.answer("룩셈부르크의 수도는?")

        self.assertEqual(answer.subject, "룩셈부르크")
        self.assertEqual(answer.answer, "룩셈부르크")
        self.assertEqual(answer.source, "stub_fallback")


class WikidataKnowledgeServiceTests(unittest.IsolatedAsyncioTestCase):
    def _transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            parsed = urlparse(str(request.url))
            query = parse_qs(parsed.query)

            if parsed.path == "/w/api.php" and query.get("action") == ["wbsearchentities"]:
                search = query.get("search", [""])[0]
                if search == "룩셈부르크":
                    payload = {
                        "search": [
                            {
                                "id": "Q32",
                                "label": "룩셈부르크",
                                "description": "서유럽의 내륙국",
                                "display": {
                                    "label": {"value": "룩셈부르크"},
                                    "description": {"value": "서유럽의 내륙국"},
                                },
                            }
                        ]
                    }
                    return httpx.Response(200, json=payload)

                payload = {
                    "search": [
                        {
                            "id": "Q16",
                            "label": "캐나다",
                            "description": "북아메리카에 있는 나라",
                            "display": {
                                "label": {"value": "캐나다"},
                                "description": {"value": "북아메리카에 있는 나라"},
                            },
                        }
                    ]
                }
                return httpx.Response(200, json=payload)

            if parsed.path == "/wiki/Special:EntityData/Q16.json":
                payload = {
                    "entities": {
                        "Q16": {
                            "claims": {
                                "P36": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "value": {"id": "Q1930"}
                                            }
                                        }
                                    }
                                ]
                            ,
                                "P1082": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "value": {"amount": "+40000000"}
                                            }
                                        }
                                    }
                                ],
                                "P2046": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "value": {"amount": "+9984670"}
                                            }
                                        }
                                    }
                                ],
                                "P35": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "value": {"id": "Q110407609"}
                                            }
                                        }
                                    }
                                ],
                                "P6": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "value": {"id": "Q123456"}
                                            }
                                        }
                                    }
                                ],
                            }
                        }
                    }
                }
                return httpx.Response(200, json=payload)

            if parsed.path == "/wiki/Special:EntityData/Q32.json":
                payload = {
                    "entities": {
                        "Q32": {
                            "claims": {
                                "P36": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "value": {"id": "Q1842"}
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
                return httpx.Response(200, json=payload)

            if parsed.path == "/w/api.php" and query.get("action") == ["wbgetentities"]:
                entity_id = query["ids"][0]
                if entity_id == "Q16":
                    payload = {
                        "entities": {
                            entity_id: {
                                "labels": {
                                    "ko": {"value": "캐나다"},
                                    "en": {"value": "Canada"},
                                },
                                "descriptions": {
                                    "ko": {"value": "북아메리카에 있는 나라"},
                                    "en": {"value": "country in North America"},
                                },
                            }
                        }
                    }
                elif entity_id == "Q32":
                    payload = {
                        "entities": {
                            entity_id: {
                                "labels": {
                                    "ko": {"value": "룩셈부르크"},
                                    "en": {"value": "Luxembourg"},
                                },
                                "descriptions": {
                                    "ko": {"value": "서유럽의 내륙국"},
                                    "en": {"value": "landlocked country in Western Europe"},
                                },
                            }
                        }
                    }
                elif entity_id == "Q1842":
                    payload = {
                        "entities": {
                            entity_id: {
                                "labels": {
                                    "ko": {"value": "룩셈부르크"},
                                    "en": {"value": "Luxembourg"},
                                },
                                "descriptions": {
                                    "ko": {"value": "룩셈부르크의 수도"},
                                    "en": {"value": "capital of Luxembourg"},
                                },
                            }
                        }
                    }
                elif entity_id == "Q110407609":
                    payload = {
                        "entities": {
                            entity_id: {
                                "labels": {
                                    "ko": {"value": "찰스 3세"},
                                    "en": {"value": "Charles III"},
                                },
                                "descriptions": {
                                    "ko": {"value": "캐나다의 국가원수"},
                                    "en": {"value": "head of state of Canada"},
                                },
                            }
                        }
                    }
                elif entity_id == "Q123456":
                    payload = {
                        "entities": {
                            entity_id: {
                                "labels": {
                                    "ko": {"value": "마크 카니"},
                                    "en": {"value": "Mark Carney"},
                                },
                                "descriptions": {
                                    "ko": {"value": "캐나다의 정부수반"},
                                    "en": {"value": "head of government of Canada"},
                                },
                            }
                        }
                    }
                else:
                    payload = {
                        "entities": {
                            entity_id: {
                                "labels": {
                                    "ko": {"value": "오타와"},
                                    "en": {"value": "Ottawa"},
                                },
                                "descriptions": {
                                    "ko": {"value": "캐나다의 수도"},
                                    "en": {"value": "capital of Canada"},
                                },
                            }
                        }
                    }
                return httpx.Response(200, json=payload)

            return httpx.Response(404, json={"error": "not found"})

        return httpx.MockTransport(handler)

    def test_capital_lookup_uses_wikidata_backend(self) -> None:
        service = WikidataKnowledgeService(
            user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
            transport=self._transport(),
        )

        answer = service.answer("캐나다의 수도는?")

        self.assertEqual(answer.query_type, "capital")
        self.assertEqual(answer.subject, "캐나다")
        self.assertEqual(answer.answer, "오타와")
        self.assertEqual(answer.source, "wikidata_capital")

    def test_description_lookup_uses_wikidata_backend(self) -> None:
        service = WikidataKnowledgeService(
            user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
            transport=self._transport(),
        )

        answer = service.answer("캐나다는 뭐야?")

        self.assertEqual(answer.query_type, "description")
        self.assertIn("북아메리카에 있는 나라", answer.answer)
        self.assertEqual(answer.answer, "캐나다는 북아메리카에 있는 나라야.")

    def test_population_lookup_uses_wikidata_backend(self) -> None:
        service = WikidataKnowledgeService(
            user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
            transport=self._transport(),
        )

        answer = service.answer("캐나다의 인구는?")

        self.assertEqual(answer.query_type, "population")
        self.assertIn("40,000,000명", answer.answer)

    def test_area_lookup_uses_wikidata_backend(self) -> None:
        service = WikidataKnowledgeService(
            user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
            transport=self._transport(),
        )

        answer = service.answer("캐나다의 면적은?")

        self.assertEqual(answer.query_type, "area")
        self.assertIn("9,984,670km²", answer.answer)

    def test_president_lookup_uses_wikidata_backend(self) -> None:
        service = WikidataKnowledgeService(
            user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
            transport=self._transport(),
        )

        answer = service.answer("캐나다의 대통령은?")

        self.assertEqual(answer.query_type, "head_of_state")
        self.assertIn("찰스 3세", answer.answer)

    def test_prime_minister_lookup_uses_wikidata_backend(self) -> None:
        service = WikidataKnowledgeService(
            user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
            transport=self._transport(),
        )

        answer = service.answer("캐나다의 총리는?")

        self.assertEqual(answer.query_type, "head_of_government")
        self.assertIn("마크 카니", answer.answer)

    async def test_engine_can_use_fallback_service_for_unknown_capital(self) -> None:
        action_selector = ActionSelector(default_location=None)
        engine = PredictiveEngine(
            classifier=HeuristicIntentClassifier(),
            goal_manager=GoalManager(default_location=None),
            action_selector=action_selector,
            world_state_builder=WorldStateBuilder(),
            policy=HierarchicalPolicy(action_selector=action_selector),
            renderer=ResponseRenderer(llm_client=None),
            verifier=ResponseVerifier(),
            weather_service=_FakeWeatherService(),
            knowledge_service=BasicKnowledgeService(
                fallback_service=WikidataKnowledgeService(
                    user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
                    transport=self._transport(),
                )
            ),
            state_store=MemoryStateStore(),
        )

        result = await engine.respond("knowledge-fallback", "룩셈부르크의 수도는?")

        self.assertEqual(result.decision.action.value, "search_answer")
        self.assertEqual(result.decision.slots.get("knowledge_source"), "wikidata_capital")
        self.assertIn("룩셈부르크", result.reply)


class KnowledgeSearchEngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = _build_engine()

    async def test_fact_question_returns_grounded_capital_answer(self) -> None:
        result = await self.engine.respond("knowledge-user", "미국의 수도는?")

        self.assertEqual(result.features.intent.value, "search_request")
        self.assertEqual(result.decision.action.value, "search_answer")
        self.assertEqual(result.decision.slots.get("knowledge_grounded"), "true")
        self.assertEqual(result.decision.slots.get("knowledge_answer"), "워싱턴 D.C.")
        self.assertIn("워싱턴 D.C.", result.reply)
        self.assertIn("기본 국가 정보", result.reply)
        self.assertTrue(result.verification.ok)

    async def test_flag_question_returns_grounded_flag_answer(self) -> None:
        result = await self.engine.respond("knowledge-flag", "일본의 국기는?")

        self.assertEqual(result.features.intent.value, "search_request")
        self.assertEqual(result.decision.action.value, "search_answer")
        self.assertEqual(result.decision.slots.get("knowledge_query_type"), "flag")
        self.assertIn("🇯🇵", result.reply)

    async def test_location_question_returns_grounded_location_answer(self) -> None:
        result = await self.engine.respond("knowledge-location", "한국의 위치는?")

        self.assertEqual(result.features.intent.value, "search_request")
        self.assertEqual(result.decision.action.value, "search_answer")
        self.assertEqual(result.decision.slots.get("knowledge_query_type"), "location")
        self.assertIn("동아시아", result.reply)


if __name__ == "__main__":
    unittest.main()
