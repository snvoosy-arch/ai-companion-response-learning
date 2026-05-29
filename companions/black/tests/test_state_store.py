from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.memory import DurableMemoryBucket
from predictive_bot.core.models import ActionType, DecisionModule, ExplanationMode, Intent, WeatherReport
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import SQLiteStateStore


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=18.0,
            description="맑음",
            wind_kph=7.0,
        )


class SQLiteStateStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_state_redacts_sensitive_text_before_persisting(self) -> None:
        db_path = TEST_TMP_ROOT / f"state_sensitive_{uuid4().hex}.sqlite3"
        store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
        try:
            first_engine = PredictiveEngine(
                classifier=HeuristicIntentClassifier(),
                goal_manager=GoalManager(default_location=None),
                action_selector=ActionSelector(default_location=None),
                renderer=ResponseRenderer(llm_client=None),
                weather_service=FakeWeatherService(),
                state_store=store,
            )
            await first_engine.respond("user-sensitive", "내 token=abc123 이고 이메일은 test@example.com 이야")

            connection = sqlite3.connect(db_path)
            try:
                row = connection.execute(
                    "SELECT user_text FROM message_log WHERE user_id = ? ORDER BY rowid DESC LIMIT 1",
                    ("user-sensitive",),
                ).fetchone()
                self.assertIsNotNone(row)
                user_text = row[0]
                self.assertNotIn("abc123", user_text)
                self.assertNotIn("test@example.com", user_text)
                self.assertIn("[redacted:secret]", user_text)
                self.assertIn("[redacted:email]", user_text)
            finally:
                connection.close()
        finally:
            store.close()
            for suffix in ("", "-wal", "-shm"):
                (Path(f"{db_path}{suffix}") if suffix else db_path).unlink(missing_ok=True)

    async def test_sqlite_state_tolerates_malformed_state_and_trace_rows(self) -> None:
        db_path = TEST_TMP_ROOT / f"state_malformed_{uuid4().hex}.sqlite3"
        bootstrap_store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
        bootstrap_store.close()
        store = None
        try:
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    INSERT INTO conversation_state (
                      user_id,
                      turn_count,
                      tension,
                      rapport,
                      boundary_pressure,
                      directness_score,
                      last_intent,
                      last_action,
                      preference_memory_json,
                      durable_memory_json,
                      recent_turns_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "broken-user",
                        3,
                        0.4,
                        0.7,
                        0.1,
                        0.5,
                        "not_a_real_intent",
                        "not_a_real_action",
                        "{broken",
                        "{also_broken",
                        "{still_broken",
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO decision_trace (
                      decision_id,
                      user_id,
                      input_text,
                      input_intent,
                      input_sentiment,
                      selected_action,
                      selected_reason,
                      decision_module,
                      explanation_mode,
                      classifier_evidence_json,
                      reason_trace_json,
                      evidence_json,
                      constraints_json,
                      world_state_snapshot_json,
                      state_inference_trace_json,
                      policy_candidates_json,
                      counterfactuals_json,
                      logic_chain_json,
                      output_text,
                      llm_used,
                      llm_fallback_reason,
                      verification_issues_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "broken-trace",
                        "broken-user",
                        "친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸해",
                        "bad_intent",
                        "mixed",
                        "bad_action",
                        "broken selected reason",
                        "bad_module",
                        "bad_mode",
                        "{broken",
                        "{broken",
                        "{broken",
                        "{broken",
                        "{broken",
                        "{broken",
                        "{broken",
                        "{broken",
                        "{broken",
                        "output",
                        1,
                        "llm_exception:RuntimeError",
                        "{broken",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
            restored_state = store.get_or_create("broken-user")
            self.assertEqual(restored_state.last_intent, Intent.UNKNOWN)
            self.assertEqual(restored_state.last_action, ActionType.CONTINUE_CONVERSATION)
            self.assertEqual(restored_state.preference_memory, {})
            self.assertEqual(restored_state.durable_memory, [])
            self.assertEqual(restored_state.recent_turns, [])

            latest_trace = store.get_latest_decision_trace("broken-user")
            self.assertIsNotNone(latest_trace)
            assert latest_trace is not None
            self.assertEqual(latest_trace.input_intent, Intent.UNKNOWN)
            self.assertEqual(latest_trace.selected_action, ActionType.CONTINUE_CONVERSATION)
            self.assertEqual(latest_trace.decision_module, DecisionModule.DAILY_CHAT)
            self.assertEqual(latest_trace.explanation_mode, ExplanationMode.ON_REQUEST_ONLY)
            self.assertEqual(latest_trace.reason_trace, [])
            self.assertEqual(latest_trace.policy_candidates, [])
            self.assertEqual(latest_trace.counterfactuals, [])
            self.assertEqual(latest_trace.logic_chain, [])
            self.assertEqual(latest_trace.verification_issues, [])
        finally:
            if store is not None:
                store.close()
            for suffix in ("", "-wal", "-shm"):
                (Path(f"{db_path}{suffix}") if suffix else db_path).unlink(missing_ok=True)

    async def test_sqlite_state_persists_across_engine_restart(self) -> None:
        db_path = TEST_TMP_ROOT / f"state_{uuid4().hex}.sqlite3"
        first_store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
        second_store = None
        try:
            first_engine = PredictiveEngine(
                classifier=HeuristicIntentClassifier(),
                goal_manager=GoalManager(default_location=None),
                action_selector=ActionSelector(default_location=None),
                renderer=ResponseRenderer(llm_client=None),
                weather_service=FakeWeatherService(),
                state_store=first_store,
            )

            first_result = await first_engine.respond("user-1", "오늘 날씨 어때?")
            self.assertEqual(first_result.decision.action, ActionType.ASK_LOCATION)
            self.assertIsNotNone(first_result.decision_trace)
            self.assertEqual(first_result.decision_trace.decision_module, DecisionModule.WEATHER)
            self.assertEqual(first_result.decision_trace.explanation_mode, ExplanationMode.SHORT)
            first_trace_id = first_result.decision_trace.decision_id
            first_store.close()

            second_store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
            second_engine = PredictiveEngine(
                classifier=HeuristicIntentClassifier(),
                goal_manager=GoalManager(default_location=None),
                action_selector=ActionSelector(default_location=None),
                renderer=ResponseRenderer(llm_client=None),
                weather_service=FakeWeatherService(),
                state_store=second_store,
            )

            second_result = await second_engine.respond("user-1", "서울")
            self.assertEqual(second_result.decision.action, ActionType.WEATHER_LOOKUP)
            self.assertIn("서울", second_result.reply)

            restored_state = second_store.get_or_create("user-1")
            self.assertEqual(restored_state.known_location, "서울")
            self.assertIsNone(restored_state.awaiting_slot)
            self.assertEqual(len(restored_state.recent_turns), 2)
            self.assertIsNotNone(restored_state.last_decision_id)
            self.assertGreaterEqual(restored_state.rapport, 0.5)
            self.assertGreaterEqual(restored_state.directness_score, 0.44)

            latest_trace = second_store.get_latest_decision_trace("user-1")
            self.assertIsNotNone(latest_trace)
            self.assertEqual(latest_trace.input_text, "서울")
            self.assertEqual(latest_trace.selected_action, ActionType.WEATHER_LOOKUP)
            self.assertEqual(latest_trace.decision_module, DecisionModule.WEATHER)
            self.assertEqual(latest_trace.explanation_mode, ExplanationMode.SHORT)
            self.assertIsNotNone(latest_trace.classifier_evidence)
            self.assertTrue(latest_trace.state_inference_trace)
            self.assertTrue(latest_trace.policy_candidates)
            self.assertTrue(latest_trace.logic_chain)
            self.assertIsNotNone(latest_trace.response_plan)
            self.assertNotEqual(latest_trace.decision_id, first_trace_id)

            connection = sqlite3.connect(db_path)
            try:
                logged_turns = connection.execute(
                    "SELECT COUNT(*) FROM message_log WHERE user_id = ?",
                    ("user-1",),
                ).fetchone()[0]
                logged_traces = connection.execute(
                    "SELECT COUNT(*) FROM decision_trace WHERE user_id = ?",
                    ("user-1",),
                ).fetchone()[0]
                logged_decision_module, logged_explanation_mode = connection.execute(
                    """
                    SELECT decision_module, explanation_mode
                    FROM decision_trace
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    ("user-1",),
                ).fetchone()
                first_counterfactuals_json = connection.execute(
                    """
                    SELECT counterfactuals_json
                    FROM decision_trace
                    WHERE decision_id = ?
                    """,
                    (first_trace_id,),
                ).fetchone()[0]
                first_logic_chain_json = connection.execute(
                    """
                    SELECT logic_chain_json
                    FROM decision_trace
                    WHERE decision_id = ?
                    """,
                    (first_trace_id,),
                ).fetchone()[0]
                first_response_plan_json = connection.execute(
                    """
                    SELECT response_plan_json
                    FROM decision_trace
                    WHERE decision_id = ?
                    """,
                    (first_trace_id,),
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(logged_turns, 2)
            self.assertEqual(logged_traces, 2)
            self.assertEqual(logged_decision_module, DecisionModule.WEATHER.value)
            self.assertEqual(logged_explanation_mode, ExplanationMode.SHORT.value)
            self.assertTrue(json.loads(first_counterfactuals_json))
            self.assertTrue(json.loads(first_logic_chain_json))
            self.assertTrue(json.loads(first_response_plan_json))

            connection = sqlite3.connect(db_path)
            try:
                rapport, boundary_pressure, directness_score = connection.execute(
                    """
                    SELECT rapport, boundary_pressure, directness_score
                    FROM conversation_state
                    WHERE user_id = ?
                    """,
                    ("user-1",),
                ).fetchone()
            finally:
                connection.close()

            self.assertIsNotNone(rapport)
            self.assertIsNotNone(boundary_pressure)
            self.assertIsNotNone(directness_score)
        finally:
            first_store.close()
            if second_store is not None:
                second_store.close()
            for suffix in ("", "-wal", "-shm"):
                (Path(f"{db_path}{suffix}") if suffix else db_path).unlink(missing_ok=True)

    async def test_sqlite_state_persists_preference_memory(self) -> None:
        db_path = TEST_TMP_ROOT / f"state_pref_{uuid4().hex}.sqlite3"
        first_store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
        second_store = None
        try:
            first_engine = PredictiveEngine(
                classifier=HeuristicIntentClassifier(),
                goal_manager=GoalManager(default_location=None),
                action_selector=ActionSelector(default_location=None),
                renderer=ResponseRenderer(llm_client=None),
                weather_service=FakeWeatherService(),
                state_store=first_store,
            )

            first_result = await first_engine.respond("pref-user", "공포영화 좋아해")
            self.assertEqual(first_result.decision.action, ActionType.RECOMMEND)
            first_store.close()

            second_store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
            second_engine = PredictiveEngine(
                classifier=HeuristicIntentClassifier(),
                goal_manager=GoalManager(default_location=None),
                action_selector=ActionSelector(default_location=None),
                renderer=ResponseRenderer(llm_client=None),
                weather_service=FakeWeatherService(),
                state_store=second_store,
            )

            restored_state = second_store.get_or_create("pref-user")
            self.assertEqual(restored_state.preference_memory.get("media_like"), "공포영화")

            second_result = await second_engine.respond("pref-user", "볼 거 추천해줘")
            self.assertEqual(second_result.decision.action, ActionType.RECOMMEND)
            self.assertIn("공포영화", second_result.reply)

            connection = sqlite3.connect(db_path)
            try:
                preference_memory_json = connection.execute(
                    """
                    SELECT preference_memory_json
                    FROM conversation_state
                    WHERE user_id = ?
                    """,
                    ("pref-user",),
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(json.loads(preference_memory_json)["media_like"], "공포영화")
        finally:
            first_store.close()
            if second_store is not None:
                second_store.close()
            for suffix in ("", "-wal", "-shm"):
                (Path(f"{db_path}{suffix}") if suffix else db_path).unlink(missing_ok=True)

    async def test_sqlite_state_persists_durable_memory(self) -> None:
        db_path = TEST_TMP_ROOT / f"state_durable_{uuid4().hex}.sqlite3"
        first_store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
        second_store = None
        try:
            first_engine = PredictiveEngine(
                classifier=HeuristicIntentClassifier(),
                goal_manager=GoalManager(default_location=None),
                action_selector=ActionSelector(default_location=None),
                renderer=ResponseRenderer(llm_client=None),
                weather_service=FakeWeatherService(),
                state_store=first_store,
            )

            await first_engine.respond("durable-user", "친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸해")
            first_store.close()

            second_store = SQLiteStateStore(db_path=db_path, max_recent_turns=6)
            restored_state = second_store.get_or_create("durable-user")

            self.assertIn(
                "친구 잘되는 거 축하해주고 왔는데 이상하게 조금 씁쓸해",
                [entry.text for entry in restored_state.durable_memory],
            )
            self.assertEqual(restored_state.durable_memory[0].bucket, DurableMemoryBucket.COMPARISON)
            self.assertIsNotNone(restored_state.durable_memory[0].captured_turn)

            connection = sqlite3.connect(db_path)
            try:
                durable_memory_json = connection.execute(
                    """
                    SELECT durable_memory_json
                    FROM conversation_state
                    WHERE user_id = ?
                    """,
                    ("durable-user",),
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertIn("씁쓸", json.loads(durable_memory_json)[0]["text"])
            self.assertEqual(json.loads(durable_memory_json)[0]["bucket"], DurableMemoryBucket.COMPARISON.value)
            self.assertEqual(json.loads(durable_memory_json)[0]["captured_turn"], 1)
        finally:
            first_store.close()
            if second_store is not None:
                second_store.close()
            for suffix in ("", "-wal", "-shm"):
                (Path(f"{db_path}{suffix}") if suffix else db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
