from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

from discord_lmstudio_bot.config import Settings
from discord_lmstudio_bot.llm_client import LMStudioClient, _build_chat_messages, _build_memory_prompt
from discord_lmstudio_bot.memory_store import DurableMemory
from discord_lmstudio_bot.output_guard import GuardIssue, GuardResult


def _make_settings() -> Settings:
    return Settings(
        discord_token="token",
        lm_studio_model="model",
        discord_guild_id=None,
        lm_studio_base_url="http://127.0.0.1:8000/v1",
        lm_studio_api_key="local-token",
        system_prompt="system prompt",
        disable_thinking=True,
        chat_reply_enabled=True,
        reply_when_mentioned=True,
        chat_channel_ids=(),
        chat_history_messages=8,
        memory_db_path="data/discord_memory.sqlite3",
        memory_recent_messages=16,
        memory_summary_trigger_messages=24,
        memory_summary_batch_messages=20,
        temperature=0.5,
        max_output_tokens=160,
        request_timeout_seconds=120,
        output_guard_trace_path="",
        output_guard_no_canned_fallback=False,
        web_search_mode="off",
        tavily_api_key="",
        tavily_max_results=5,
        tavily_search_depth="basic",
        tavily_country="south korea",
        tavily_news_time_range="month",
        duo_enabled=False,
        duo_partner_bot_id=None,
        duo_channel_ids=(),
        duo_max_replies=8,
        duo_reply_delay_seconds=1.5,
        runtime_state_enabled=True,
        runtime_state_db_path="runtime/bot_runtime_state.sqlite3",
        runtime_bot_name="white",
        runtime_heartbeat_seconds=15,
        runtime_online_timeout_seconds=45,
        runtime_auto_claim_ttl_seconds=300,
        runtime_manual_claim_ttl_seconds=1800,
        startup_singleton_enabled=True,
        startup_singleton_lock_path="data/white_startup.lock",
        tts_enabled=False,
        tts_mode="off",
        tts_provider="noop",
        tts_command_template="",
        tts_play_command_template="",
        tts_local_player="auto",
        tts_ffmpeg_executable="ffmpeg",
        tts_output_dir="runtime/tts/white",
        tts_obs_output_dir="runtime/obs-live/white",
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


def _context_memory(kind: str = "profile") -> DurableMemory:
    return DurableMemory(
        id=1,
        guild_id=10,
        channel_id=20,
        user_id=30,
        user_name="tester",
        scope_key="user:30",
        source_kind="user_note",
        memory_kind=kind,
        memory_text="요즘 면접 준비 중인데 자꾸 불안해.",
        source_message_id=100,
        updated_at="2026-04-25 00:00:00",
        relevance_score=5.0,
        matched_terms=("면접", "불안"),
        retrieval_rank=0,
    )


class _RepairFirstGuard:
    def __init__(self) -> None:
        self.build_response_instruction_calls = 0
        self.build_retry_instruction_calls = 0
        self.build_repair_instruction_calls = 0
        self.should_prefer_repair_first_calls = 0
        self.check_calls = 0

    def build_response_instruction(self, *, user_prompt: str, recent_replies: list[str], retry: bool = False) -> str:
        self.build_response_instruction_calls += 1
        return "response-instruction"

    def check(self, reply: str, *, user_prompt: str, recent_replies: list[str]) -> GuardResult:
        self.check_calls += 1
        if self.check_calls == 1:
            return GuardResult(
                reply=reply,
                issues=(GuardIssue(code="checkin_vague_reply", detail="test", blocking=True),),
            )
        return GuardResult(reply="수정된 답변", issues=())

    def should_prefer_repair_first(self, *, user_prompt: str, issues: tuple[GuardIssue, ...]) -> bool:
        self.should_prefer_repair_first_calls += 1
        return True

    def build_retry_instruction(self, *, user_prompt: str, issues: tuple[GuardIssue, ...], recent_replies: list[str] | None = None) -> str:
        self.build_retry_instruction_calls += 1
        return "retry-instruction"

    def build_repair_instruction(
        self,
        *,
        user_prompt: str,
        issues: tuple[GuardIssue, ...],
        rejected_replies: list[str] | None = None,
        recent_replies: list[str] | None = None,
    ) -> str:
        self.build_repair_instruction_calls += 1
        return "repair-instruction"

    def build_fallback_reply(
        self,
        *,
        user_prompt: str,
        issues: tuple[GuardIssue, ...],
        recent_replies: list[str] | None = None,
        rejected_replies: list[str] | None = None,
    ) -> str:
        raise AssertionError("fallback should not be reached in this test")


class LMStudioClientRepairFirstTests(unittest.IsolatedAsyncioTestCase):
    def test_build_chat_messages_includes_context_packet_before_memory(self) -> None:
        messages = _build_chat_messages(
            settings=_make_settings(),
            prompt="오늘은 좀 울적해.",
            user_name="viewer",
            history=[],
            images=None,
            web_context=None,
            memory_summary="요약",
            durable_memories=[_context_memory("profile")],
        )

        self.assertGreaterEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("[white_context_packet]", str(messages[1]["content"]))
        self.assertIn("scene=comfort_support", str(messages[1]["content"]))
        self.assertIn("[관련 장기 기억]", str(messages[2]["content"]))

    def test_memory_prompt_groups_typed_entries(self) -> None:
        prompt = _build_memory_prompt(
            memory_summary="요약",
            durable_memories=[
                DurableMemory(
                    id=1,
                    guild_id=10,
                    channel_id=20,
                    user_id=30,
                    user_name="tester",
                    scope_key="user:30",
                    source_kind="user_note",
                    memory_kind="profile",
                    memory_text="나는 매운 음식을 좋아해.",
                    source_message_id=100,
                    updated_at="2026-04-15 00:00:00",
                    relevance_score=4.25,
                    matched_terms=("좋아해",),
                    retrieval_rank=0,
                ),
                DurableMemory(
                    id=2,
                    guild_id=10,
                    channel_id=20,
                    user_id=30,
                    user_name="tester",
                    scope_key="user:30",
                    source_kind="user_note",
                    memory_kind="open_loop",
                    memory_text="요즘 면접 준비 중인데 자꾸 불안해.",
                    source_message_id=101,
                    updated_at="2026-04-15 00:01:00",
                    relevance_score=5.5,
                    matched_terms=("면접", "불안"),
                    retrieval_rank=1,
                ),
            ],
        )

        self.assertIn("[프로필]", prompt)
        self.assertIn("[열린 루프]", prompt)
        self.assertIn('- "나는 매운 음식을 좋아해."', prompt)
        self.assertIn('- "요즘 면접 준비 중인데 자꾸 불안해."', prompt)
        self.assertIn("[채널 요약]", prompt)

    async def test_checkin_like_failure_is_observed_without_repair(self) -> None:
        client = LMStudioClient(_make_settings())
        guard = _RepairFirstGuard()
        client._output_guard = guard  # type: ignore[assignment]
        client._generate_chat_text = AsyncMock(side_effect=["오늘은 조금 덜 뻔하게 남아볼게.", "수정된 답변"])  # type: ignore[assignment]

        reply = await client.ask("안녕. 오늘 기분은 어때?", "tester")

        self.assertEqual(reply, "오늘은 조금 덜 뻔하게 남아볼게.")
        self.assertEqual(guard.should_prefer_repair_first_calls, 0)
        self.assertEqual(guard.build_retry_instruction_calls, 0)
        self.assertEqual(guard.build_repair_instruction_calls, 0)
        self.assertEqual(client._generate_chat_text.await_count, 1)

    async def test_no_canned_fallback_returns_raw_reply_without_repair(self) -> None:
        client = LMStudioClient(_make_settings())
        client._settings = replace(client._settings, output_guard_no_canned_fallback=True)

        class _NoFallbackGuard(_RepairFirstGuard):
            def __init__(self) -> None:
                super().__init__()
                self.build_fallback_reply_calls = 0

            def check(self, reply: str, *, user_prompt: str, recent_replies: list[str]) -> GuardResult:
                self.check_calls += 1
                if self.check_calls == 1:
                    return GuardResult(
                        reply="원본 답변",
                        issues=(GuardIssue(code="checkin_vague_reply", detail="test", blocking=True),),
                    )
                if self.check_calls == 2:
                    return GuardResult(
                        reply="재시도 답변",
                        issues=(GuardIssue(code="repeated_reply", detail="test", blocking=True),),
                    )
                return GuardResult(
                    reply="수정된 답변",
                    issues=(GuardIssue(code="repeated_reply", detail="test", blocking=True),),
                )

            def should_prefer_repair_first(self, *, user_prompt: str, issues: tuple[GuardIssue, ...]) -> bool:
                self.should_prefer_repair_first_calls += 1
                return False

            def build_fallback_reply(
                self,
                *,
                user_prompt: str,
                issues: tuple[GuardIssue, ...],
                recent_replies: list[str] | None = None,
                rejected_replies: list[str] | None = None,
            ) -> str:
                self.build_fallback_reply_calls += 1
                raise AssertionError("fallback should not be reached in no-fallback mode")

        guard = _NoFallbackGuard()
        client._output_guard = guard  # type: ignore[assignment]
        client._generate_chat_text = AsyncMock(side_effect=["오늘은 조금 덜 뻔하게 남아볼게.", "수정된 답변", "최종 보정 답변"])  # type: ignore[assignment]

        reply = await client.ask("안녕. 오늘 기분은 어때?", "tester")

        self.assertEqual(reply, "오늘은 조금 덜 뻔하게 남아볼게.")
        self.assertEqual(guard.should_prefer_repair_first_calls, 0)
        self.assertEqual(guard.build_retry_instruction_calls, 0)
        self.assertEqual(guard.build_repair_instruction_calls, 0)
        self.assertEqual(guard.build_fallback_reply_calls, 0)
        self.assertEqual(client._generate_chat_text.await_count, 1)

    async def test_no_canned_fallback_trace_marks_final_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            settings = replace(
                _make_settings(),
                output_guard_trace_path=str(trace_path),
                output_guard_no_canned_fallback=True,
            )
            client = LMStudioClient(settings)

            class _TracingGuard(_RepairFirstGuard):
                def check(self, reply: str, *, user_prompt: str, recent_replies: list[str]) -> GuardResult:
                    self.check_calls += 1
                    if self.check_calls == 1:
                        return GuardResult(
                            reply="원본 답변",
                            issues=(GuardIssue(code="checkin_vague_reply", detail="test", blocking=True),),
                        )
                    if self.check_calls == 2:
                        return GuardResult(
                            reply="재시도 답변",
                            issues=(GuardIssue(code="repeated_reply", detail="test", blocking=True),),
                        )
                    return GuardResult(
                        reply="수정된 답변",
                        issues=(GuardIssue(code="repeated_reply", detail="test", blocking=True),),
                    )

                def should_prefer_repair_first(self, *, user_prompt: str, issues: tuple[GuardIssue, ...]) -> bool:
                    self.should_prefer_repair_first_calls += 1
                    return False

                def build_fallback_reply(
                    self,
                    *,
                    user_prompt: str,
                    issues: tuple[GuardIssue, ...],
                    recent_replies: list[str] | None = None,
                    rejected_replies: list[str] | None = None,
                ) -> str:
                    raise AssertionError("fallback should not be reached in no-fallback mode")

            guard = _TracingGuard()
            client._output_guard = guard  # type: ignore[assignment]
            client._generate_chat_text = AsyncMock(side_effect=["오늘은 조금 덜 뻔하게 남아볼게.", "수정된 답변", "최종 보정 답변"])  # type: ignore[assignment]

            reply = await client.ask(
                "안녕. 오늘 기분은 어때?",
                "tester",
                durable_memories=[
                    DurableMemory(
                        id=1,
                        guild_id=10,
                        channel_id=20,
                        user_id=30,
                        user_name="tester",
                        scope_key="user:30",
                        source_kind="user_note",
                        memory_kind="ongoing",
                        memory_text="요즘 면접 준비 중인데 자꾸 불안해.",
                        source_message_id=100,
                        updated_at="2026-04-15 00:00:00",
                        relevance_score=5.25,
                        matched_terms=("면접", "불안"),
                        retrieval_rank=0,
                    )
                ],
            )

            self.assertEqual(reply, "오늘은 조금 덜 뻔하게 남아볼게.")
            trace = trace_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(trace), 1)
            payload = json.loads(trace[0])
            self.assertEqual(payload["final_source"], "raw_observed")
            self.assertTrue(payload["no_canned_fallback"])
            self.assertEqual(payload["final_reply"], "오늘은 조금 덜 뻔하게 남아볼게.")
            self.assertEqual(payload["final_generation_stage"], "raw")
            self.assertEqual(payload["durable_memory_count"], 1)
            self.assertEqual(payload["durable_memory_kinds"], ["ongoing"])
            self.assertEqual(payload["durable_memory_scopes"], ["user:30"])

    async def test_guard_trace_redacts_sensitive_prompt_and_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            settings = replace(
                _make_settings(),
                output_guard_trace_path=str(trace_path),
            )
            client = LMStudioClient(settings)

            class _PassGuard(_RepairFirstGuard):
                def check(self, reply: str, *, user_prompt: str, recent_replies: list[str]) -> GuardResult:
                    return GuardResult(reply=reply, issues=())

            client._output_guard = _PassGuard()  # type: ignore[assignment]
            client._generate_chat_text = AsyncMock(return_value="답장은 test@example.com 으로 보내고 password=hunter2 라고 적어둬.")  # type: ignore[assignment]

            reply = await client.ask("내 이메일은 test@example.com 이고 비밀번호는 hunter2 야.", "tester")

            self.assertIn("test@example.com", reply)
            payload = json.loads(trace_path.read_text(encoding="utf-8").strip())
            self.assertNotIn("test@example.com", payload["user_prompt"])
            self.assertNotIn("hunter2", payload["user_prompt"])
            self.assertNotIn("test@example.com", payload["raw_reply"])
            self.assertNotIn("hunter2", payload["raw_reply"])
            self.assertIn("[redacted:email]", payload["user_prompt"])
            self.assertIn("[redacted:secret]", payload["raw_reply"])

    async def test_guard_trace_records_runtime_response_model_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            settings = replace(
                _make_settings(),
                output_guard_trace_path=str(trace_path),
            )
            client = LMStudioClient(settings)

            class _PassGuard(_RepairFirstGuard):
                def check(self, reply: str, *, user_prompt: str, recent_replies: list[str]) -> GuardResult:
                    return GuardResult(reply=reply, issues=())

            async def _fake_generate_chat_text(*, messages, temperature, max_tokens):  # type: ignore[no-untyped-def]
                client._last_generation_metadata = {
                    "response_model": "white_respfix3_runtime",
                    "system_fingerprint": "abc123fingerprint",
                    "response_id": "chatcmpl-test",
                }
                return "괜찮아. 여기서는 천천히 말해도 돼."

            client._output_guard = _PassGuard()  # type: ignore[assignment]
            client._generate_chat_text = AsyncMock(side_effect=_fake_generate_chat_text)  # type: ignore[assignment]

            reply = await client.ask("오늘은 좀 조용히 있고 싶어.", "tester")

            self.assertEqual(reply, "괜찮아. 여기서는 천천히 말해도 돼.")
            payload = json.loads(trace_path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["request_model"], "model")
            self.assertEqual(payload["raw_response_model"], "white_respfix3_runtime")
            self.assertEqual(payload["raw_system_fingerprint"], "abc123fingerprint")
            self.assertEqual(payload["raw_response_id"], "chatcmpl-test")
            self.assertEqual(payload["final_generation_stage"], "raw")
            self.assertEqual(payload["final_response_model"], "white_respfix3_runtime")
            self.assertEqual(payload["final_system_fingerprint"], "abc123fingerprint")


if __name__ == "__main__":
    unittest.main()
