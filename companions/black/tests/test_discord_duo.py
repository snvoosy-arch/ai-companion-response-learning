from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from predictive_bot.config import AppConfig
from predictive_bot.discord_app.bot import PredictiveDiscordClient


class _StubEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.state_store = SimpleNamespace(close=lambda: None)

    async def respond(self, user_id: str, text: str):
        self.calls.append((user_id, text))
        return SimpleNamespace(
            reply="테스트 응답",
            decision=SimpleNamespace(action=SimpleNamespace(value="continue_conversation"), reason="stub"),
        )


class _FakeChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id
        self.sent_messages: list[str] = []

    async def send(self, text: str) -> None:
        self.sent_messages.append(text)


class _FakeAuthor:
    def __init__(self, author_id: int, *, bot: bool = True) -> None:
        self.id = author_id
        self.bot = bot
        self.name = f"bot-{author_id}"
        self.display_name = self.name


class _FakeMessage:
    def __init__(self, *, channel_id: int, author_id: int, content: str) -> None:
        self.channel = _FakeChannel(channel_id)
        self.author = _FakeAuthor(author_id, bot=True)
        self.content = content
        self.role_mentions: list[object] = []
        self.raw_mentions: list[int] = []
        self.reference = None
        self.guild = None


def _build_config(runtime_dir: Path) -> AppConfig:
    return AppConfig(
        discord_bot_token=None,
        bot_trigger_prefix="!predict",
        default_location=None,
        bot_persona="black",
        generation_backend="template",
        kobart_model_name_or_path="gogamza/kobart-base-v2",
        kobart_device="cpu",
        kobart_max_new_tokens=24,
        kobart_num_beams=1,
        state_backend="memory",
        state_db_path=str(runtime_dir / "state.sqlite3"),
        state_max_recent_turns=6,
        intent_model_type="kcbert",
        intent_model_path=None,
        kcbert_model_path=str(runtime_dir / "kcbert"),
        intent_model_min_confidence=0.35,
        policy_action_model_path=None,
        knowledge_backend="builtin",
        wikidata_user_agent="test-agent",
        wikidata_timeout_seconds=5.0,
        openai_api_key=None,
        openai_model=None,
        openai_base_url="https://api.openai.com/v1",
        openai_timeout_seconds=20.0,
        log_message_content=False,
        runtime_state_enabled=False,
        runtime_state_db_path=str(runtime_dir / "runtime.sqlite3"),
        runtime_bot_name="black",
        runtime_heartbeat_seconds=15.0,
        runtime_online_timeout_seconds=45.0,
        runtime_auto_claim_ttl_seconds=300,
        runtime_manual_claim_ttl_seconds=1800,
        duo_mode_enabled=True,
        duo_partner_bot_id="123456789012345678",
        duo_channel_id="123456789012345678",
        duo_max_turns_per_bot=6,
        duo_autostart_enabled=False,
        duo_autostart_channel_id=None,
        duo_autostart_prompt=None,
        startup_lock_enabled=False,
        startup_lock_path=str(runtime_dir / "startup.lock"),
        tts_enabled=False,
        tts_mode="off",
        tts_provider="noop",
        tts_command_template="",
        tts_play_command_template="",
        tts_local_player="auto",
        tts_ffmpeg_executable="ffmpeg",
        tts_output_dir=str(runtime_dir / "tts"),
        tts_obs_output_dir=str(runtime_dir / "obs-live"),
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


class PredictiveDiscordClientDuoTests(unittest.IsolatedAsyncioTestCase):
    async def test_partner_message_is_accepted_even_before_session_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = PredictiveDiscordClient(config=_build_config(Path(temp_dir)), engine=_StubEngine())
            message = _FakeMessage(
                channel_id=123456789012345678,
                author_id=123456789012345678,
                content="안녕. 오늘 기분은 어때?",
            )
            self.assertTrue(client._should_accept_partner_bot_message(message))
            await client.close()

    async def test_partner_message_implicitly_starts_session_and_replies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = _StubEngine()
            client = PredictiveDiscordClient(config=_build_config(Path(temp_dir)), engine=engine)
            message = _FakeMessage(
                channel_id=123456789012345678,
                author_id=123456789012345678,
                content="안녕. 오늘 기분은 어때?",
            )
            await client._handle_partner_bot_message(message)
            self.assertTrue(client._duo_session.active)
            self.assertEqual(client._duo_session.channel_id, 123456789012345678)
            self.assertEqual(client._duo_session.turns_sent, 1)
            self.assertEqual(message.channel.sent_messages, ["테스트 응답"])
            self.assertEqual(len(engine.calls), 1)
            await client.close()

    async def test_close_closes_engine_state_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            close_calls: list[str] = []
            engine = _StubEngine()
            engine.state_store = SimpleNamespace(close=lambda: close_calls.append("closed"))
            client = PredictiveDiscordClient(config=_build_config(Path(temp_dir)), engine=engine)

            await client.close()

            self.assertEqual(close_calls, ["closed"])

    async def test_run_discord_bot_closes_client_when_start_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _build_config(Path(temp_dir))
            config.discord_bot_token = "test-token"
            engine = _StubEngine()

            with patch("predictive_bot.discord_app.bot.PredictiveDiscordClient.start", new=AsyncMock(side_effect=RuntimeError("boom"))), patch(
                "predictive_bot.discord_app.bot.PredictiveDiscordClient.close",
                new=AsyncMock(),
            ) as close_mock:
                from predictive_bot.discord_app.bot import run_discord_bot

                with self.assertRaises(RuntimeError):
                    await run_discord_bot(config, engine)

            self.assertEqual(close_mock.await_count, 1)


if __name__ == "__main__":
    unittest.main()
