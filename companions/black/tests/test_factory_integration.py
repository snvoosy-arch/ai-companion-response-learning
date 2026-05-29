from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from predictive_bot.config import DEFAULT_KCBERT_MODEL_PATH, DEFAULT_POLICY_ACTION_MODEL_PATH, AppConfig
from predictive_bot.factory import build_engine


def _integration_config() -> AppConfig:
    return AppConfig(
        discord_bot_token=None,
        bot_trigger_prefix="!predict",
        default_location=None,
        bot_persona="black",
        generation_backend="template",
        kobart_model_name_or_path="gogamza/kobart-base-v2",
        kobart_device="auto",
        kobart_max_new_tokens=24,
        kobart_num_beams=1,
        state_backend="memory",
        state_db_path=":memory:",
        state_max_recent_turns=6,
        intent_model_type="kcbert",
        intent_model_path=None,
        kcbert_model_path=str(DEFAULT_KCBERT_MODEL_PATH),
        intent_model_min_confidence=0.10,
        policy_action_model_path=str(DEFAULT_POLICY_ACTION_MODEL_PATH),
        knowledge_backend="builtin",
        wikidata_user_agent="predictive-discord-bot-test/0.1 (https://example.invalid/tests)",
        wikidata_timeout_seconds=10.0,
        openai_api_key=None,
        openai_model=None,
        openai_base_url="https://api.openai.com/v1",
        openai_timeout_seconds=20.0,
        log_message_content=False,
        runtime_state_enabled=False,
        runtime_state_db_path="runtime/bot_runtime_state.sqlite3",
        runtime_bot_name="black",
        runtime_heartbeat_seconds=15.0,
        runtime_online_timeout_seconds=45.0,
        runtime_auto_claim_ttl_seconds=300,
        runtime_manual_claim_ttl_seconds=1800,
        duo_mode_enabled=False,
        duo_partner_bot_id=None,
        duo_channel_id=None,
        duo_max_turns_per_bot=6,
        duo_autostart_enabled=False,
        duo_autostart_channel_id=None,
        duo_autostart_prompt=None,
        startup_lock_enabled=False,
        startup_lock_path="runtime/predictive_bot_black.startup.lock",
        tts_enabled=False,
        tts_mode="off",
        tts_provider="noop",
        tts_command_template="",
        tts_play_command_template="",
        tts_local_player="auto",
        tts_ffmpeg_executable="ffmpeg",
        tts_output_dir="runtime/tts/black",
        tts_obs_output_dir="runtime/obs-live/black",
        tts_audio_format="ogg",
        tts_max_chars=240,
        tts_elevenlabs_api_key="",
        tts_elevenlabs_model_id="eleven_multilingual_v2",
        tts_elevenlabs_base_url="https://api.elevenlabs.io",
        tts_elevenlabs_request_timeout_seconds=60.0,
        tts_white_voice_id="white_default",
        tts_black_voice_id="black_default",
        tts_white_speed=0.94,
        tts_black_speed=1.02,
        tts_white_style="soft",
        tts_black_style="clear",
        tts_xtts_server_url="",
        tts_xtts_server_token="",
        tts_xtts_client_python="",
        tts_xtts_client_script="",
        tts_xtts_language="ko",
        tts_gptsovits_server_url="",
        tts_gptsovits_server_token="",
        tts_gptsovits_client_python="",
        tts_gptsovits_client_script="",
        tts_gptsovits_language="ko",
    )


class _FakeMultiHeadMeaningClassifier:
    def __init__(self, *, model_dir, device: str = "auto", max_length: int = 128) -> None:
        self.model_dir = model_dir
        self.device = device
        self.max_length = max_length


class FactoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = build_engine(_integration_config())

    async def asyncTearDown(self) -> None:
        self.engine.state_store.close()

    async def test_runtime_hybrid_engine_handles_contextual_reason_question(self) -> None:
        await self.engine.respond("integration-user", "오늘 날씨 어때")
        result = await self.engine.respond("integration-user", "그렇게 말한 근거는?")
        self.assertEqual(result.features.intent.value, "why")
        self.assertEqual(result.decision.action.value, "explain_reason")

    async def test_runtime_hybrid_engine_handles_fact_query(self) -> None:
        result = await self.engine.respond("integration-fact", "미국의 수도는?")
        self.assertEqual(result.features.intent.value, "search_request")
        self.assertEqual(result.decision.action.value, "search_answer")
        self.assertIn("워싱턴 D.C.", result.reply)

    async def test_runtime_hybrid_engine_handles_flag_query(self) -> None:
        result = await self.engine.respond("integration-flag", "일본의 국기는?")
        self.assertEqual(result.features.intent.value, "search_request")
        self.assertEqual(result.decision.action.value, "search_answer")
        self.assertIn("🇯🇵", result.reply)

    async def test_runtime_hybrid_engine_keeps_slang_greeting_as_small_talk(self) -> None:
        result = await self.engine.respond("integration-greeting", "와썹")
        self.assertEqual(result.features.intent.value, "greeting")
        self.assertEqual(result.decision.action.value, "small_talk")


class FactoryMeaningGateTests(unittest.TestCase):
    def test_build_engine_passes_trusted_axes_to_modernbert_meaning_classifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _integration_config()
            config.intent_model_type = "modernbert_meaning"
            config.kcbert_model_path = tmp_dir
            config.meaning_trusted_axes = ("schema", "tone", "draft_frame_family")

            with patch(
                "predictive_bot.core.meaning_classifier.MultiHeadMeaningClassifier",
                _FakeMultiHeadMeaningClassifier,
            ):
                engine = build_engine(config)

            try:
                self.assertEqual(
                    engine.classifier.meaning_trusted_axes,
                    frozenset(("schema", "tone", "draft_frame_family")),
                )
            finally:
                engine.state_store.close()


if __name__ == "__main__":
    unittest.main()
