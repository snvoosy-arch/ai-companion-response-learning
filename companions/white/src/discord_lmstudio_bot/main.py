from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import mimetypes
import os
import re
import sys
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from .config import Settings, load_settings
from .inspector import (
    analyze_recent_signals,
    build_white_dashboard_report,
    build_white_recall_report,
    build_white_signal_report,
    build_white_summary_report,
)
from .llm_client import LMStudioClient, chunk_for_discord
from .memory_store import (
    DurableMemory,
    MemoryStore,
    classify_durable_memory_kind,
    prepare_durable_memory_capture,
)
from .performance import WhitePerformanceBrain, WhiteRuntimeEvent
from .runtime_state import ClaimResult, RuntimePresenceClient
from .startup_lock import SingletonLockError, SingletonStartupLock
from .web_search import (
    SearchContext,
    TavilySearchClient,
    build_search_request,
    build_search_request_from_query,
    extract_explicit_search_request,
    format_source_links,
)

LOGGER = logging.getLogger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from bot_shared.speech import (
    Utterance,
    VoiceProfile,
    build_speech_runtime,
    disconnect_all_client_voice_connections,
    play_artifact_to_member_voice,
)
from bot_shared.vtuber import VTuberTurnPacket


def _user_visible_error_message(operation: str) -> str:
    return f"{operation} 처리 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요."


def _write_runtime_pid_file() -> Path | None:
    raw_path = os.getenv("BOT_RUNTIME_PID_FILE_PATH", "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def _cleanup_runtime_pid_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            current = path.read_text(encoding="utf-8").strip()
            if current == str(os.getpid()):
                path.unlink()
    except OSError:
        pass


@dataclass(slots=True)
class DuoSession:
    replies_sent: int = 0
    active: bool = True
    stop_reason: str | None = None
    last_partner_message_id: int | None = None
    last_partner_message_fingerprint: str | None = None
    last_bot_reply_fingerprint: str | None = None


class DiscordLMStudioBot(commands.Bot):
    def __init__(
        self,
        settings: Settings,
        llm_client: LMStudioClient,
        memory_store: MemoryStore,
        web_search_client: TavilySearchClient | None = None,
        speech_runtime=None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = (
            settings.chat_reply_enabled
            or settings.reply_when_mentioned
            or settings.duo_enabled
        )
        super().__init__(command_prefix="!", intents=intents)
        self.settings = settings
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.web_search_client = web_search_client
        self.speech_runtime = speech_runtime
        self.performance_brain = WhitePerformanceBrain()
        self.duo_sessions: dict[int, DuoSession] = {}
        self._seen_duo_message_ids: set[int] = set()
        self.runtime_presence = RuntimePresenceClient(
            enabled=settings.runtime_state_enabled,
            db_path=settings.runtime_state_db_path,
            bot_name=settings.runtime_bot_name,
            project_name="discodebot",
            heartbeat_interval_seconds=settings.runtime_heartbeat_seconds,
            online_timeout_seconds=settings.runtime_online_timeout_seconds,
            auto_claim_ttl_seconds=settings.runtime_auto_claim_ttl_seconds,
            manual_claim_ttl_seconds=settings.runtime_manual_claim_ttl_seconds,
        )
        self._runtime_heartbeat_task: asyncio.Task[None] | None = None
        self._summary_refresh_tasks: dict[int, asyncio.Task[None]] = {}
        self._summary_refresh_pending_channels: set[int] = set()

    async def setup_hook(self) -> None:
        if self.settings.discord_guild_id:
            guild = discord.Object(id=self.settings.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            LOGGER.info("Synced %s command(s) to guild %s", len(synced), guild.id)
            return

        synced = await self.tree.sync()
        LOGGER.info("Synced %s global command(s)", len(synced))

    async def on_ready(self) -> None:
        if self.user is None:
            return
        if self.runtime_presence.enabled:
            self.runtime_presence.activate(
                discord_user_id=self.user.id,
                display_name=self.user.name,
            )
            if self._runtime_heartbeat_task is None or self._runtime_heartbeat_task.done():
                self._runtime_heartbeat_task = asyncio.create_task(
                    _runtime_heartbeat_loop(self),
                    name=f"{self.runtime_presence.bot_name}-runtime-heartbeat",
                )
        LOGGER.info(
            "Logged in as %s (%s) | model=%s",
            self.user.name,
            self.user.id,
            self.settings.lm_studio_model,
        )
        if self.settings.duo_enabled:
            scope = (
                ",".join(str(channel_id) for channel_id in self.settings.duo_channel_ids)
                if self.settings.duo_channel_ids
                else "all-visible-channels"
            )
            LOGGER.info(
                "Duo mode enabled | partner_bot_id=%s | channels=%s | max_replies=%s | delay=%.2fs",
                self.settings.duo_partner_bot_id,
                scope,
                self.settings.duo_max_replies,
                self.settings.duo_reply_delay_seconds,
            )

        guilds = sorted(self.guilds, key=lambda guild: guild.name.lower())
        if not guilds:
            LOGGER.info("Connected guilds: none")
            return

        LOGGER.info("Connected guilds: %s", len(guilds))
        for guild in guilds:
            LOGGER.info("Guild | %s (%s)", guild.name, guild.id)

    async def close(self) -> None:
        summary_refresh_tasks = list(self._summary_refresh_tasks.values())
        for task in summary_refresh_tasks:
            task.cancel()
        if summary_refresh_tasks:
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*summary_refresh_tasks, return_exceptions=True)
        self._summary_refresh_tasks.clear()
        self._summary_refresh_pending_channels.clear()
        if self._runtime_heartbeat_task is not None:
            self._runtime_heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runtime_heartbeat_task
            self._runtime_heartbeat_task = None
        self.runtime_presence.mark_offline()
        await self.llm_client.close()
        if self.web_search_client is not None:
            await self.web_search_client.close()
        if self.speech_runtime is not None:
            await self.speech_runtime.close()
        await disconnect_all_client_voice_connections(self)
        await super().close()

    async def on_message(self, message: discord.Message) -> None:
        if self.user is not None and message.author.id == self.user.id:
            return

        if self.settings.duo_enabled and _is_duo_channel(self, message.channel.id) and not message.author.bot:
            _reset_duo_session(self, message.channel.id)

        has_attachments = bool(message.attachments)
        duo_prompt = _extract_duo_prompt(self, message, has_attachments=has_attachments)
        if duo_prompt:
            if not _should_handle_duo_message(self, message):
                return
            _note_runtime_message_activity(self, message=message, reason="duo_reply")
            answer = await _respond_to_message(
                self,
                message=message,
                prompt=duo_prompt,
                reply_mode="send",
                capture_reply=True,
                duo_guard_message=message,
            )
            if answer is None:
                return
            _mark_duo_reply_sent(self, message.channel.id, answer=answer)
            return

        if message.author.bot:
            return

        prompt = _extract_chat_prompt(self, message, has_attachments=has_attachments)
        if not prompt:
            return

        _note_runtime_message_activity(self, message=message, reason="chat_reply")
        await _respond_to_message(
            self,
            message=message,
            prompt=prompt,
            reply_mode="reply",
        )


async def _respond_to_message(
    bot: DiscordLMStudioBot,
    *,
    message: discord.Message,
    prompt: str,
    reply_mode: str,
    capture_reply: bool = False,
    duo_guard_message: discord.Message | None = None,
) -> str | None:
    channel_id = message.channel.id
    guild_id = message.guild.id if message.guild else None
    user_name = message.author.display_name
    memory_summary, history, durable_memories = _load_memory_context(
        bot,
        channel_id=channel_id,
        user_id=message.author.id,
        prompt=prompt,
    )
    images = await _load_image_inputs(message)
    search_context = await _maybe_search_web(bot, prompt, user_name=user_name)

    try:
        if reply_mode == "send" and bot.settings.duo_reply_delay_seconds > 0:
            await asyncio.sleep(bot.settings.duo_reply_delay_seconds)

        async with _typing_context(message, enabled=(reply_mode == "reply")):
            answer = await bot.llm_client.ask(
                prompt=prompt,
                user_name=user_name,
                history=history,
                images=images,
                web_context=search_context.prompt_context if search_context else None,
                memory_summary=memory_summary,
                durable_memories=durable_memories,
                reply_mode=reply_mode,
                duo=reply_mode == "send",
            )
    except Exception:
        LOGGER.exception("Failed to handle chat message")
        if reply_mode == "reply":
            await message.reply(_user_visible_error_message("LLM 요청"), mention_author=False)
        else:
            await message.channel.send(_user_visible_error_message("LLM 요청"))
        return None

    if search_context:
        answer = _append_search_sources(answer, search_context)

    if duo_guard_message is not None and _should_block_duo_answer(
        bot,
        channel_id=channel_id,
        message=duo_guard_message,
        answer=answer,
    ):
        return None

    chunks = list(chunk_for_discord(answer))
    if reply_mode == "reply":
        await message.reply(chunks[0], mention_author=False)
    else:
        await message.channel.send(chunks[0])
    for chunk in chunks[1:]:
        await message.channel.send(chunk)
    performance_packet = bot.performance_brain.build_turn_packet(
        event=WhiteRuntimeEvent(
            kind="duo_message" if reply_mode == "send" else "chat_message",
            prompt=prompt,
            user_name=user_name,
            reply_mode=reply_mode,
            has_images=bool(images),
            search_used=search_context is not None,
            duo=reply_mode == "send",
        ),
        reply=answer,
        search_context=search_context,
    )
    await _maybe_send_tts_for_message(
        bot,
        channel=message.channel,
        text=answer,
        speaker="white",
        voice_member=message.author,
        performance_packet=performance_packet,
    )

    _persist_conversation_turn(
        bot,
        guild_id=guild_id,
        channel_id=channel_id,
        user_id=message.author.id,
        user_name=user_name,
        prompt=prompt,
        answer=answer,
        discord_message_id=message.id,
    )
    _schedule_summary_refresh(bot, guild_id=guild_id, channel_id=channel_id)
    return answer if capture_reply else None


@asynccontextmanager
async def _typing_context(message: discord.Message, *, enabled: bool):
    if not enabled:
        yield
        return

    try:
        async with message.channel.typing():
            yield
            return
    except Exception as exc:
        LOGGER.warning(
            "Typing indicator skipped in channel %s: %s",
            message.channel.id,
            exc,
        )

    yield


def _is_duo_channel(bot: DiscordLMStudioBot, channel_id: int) -> bool:
    if not bot.settings.duo_enabled:
        return False
    if not bot.settings.duo_channel_ids:
        return True
    return channel_id in bot.settings.duo_channel_ids


def _is_partner_bot_message(bot: DiscordLMStudioBot, message: discord.Message) -> bool:
    return bool(
        bot.settings.duo_enabled
        and bot.settings.duo_partner_bot_id is not None
        and message.author.bot
        and message.author.id == bot.settings.duo_partner_bot_id
        and _is_duo_channel(bot, message.channel.id)
    )


def _reset_duo_session(bot: DiscordLMStudioBot, channel_id: int) -> None:
    if not _is_duo_channel(bot, channel_id):
        return
    bot.duo_sessions[channel_id] = DuoSession()


def _extract_duo_prompt(
    bot: DiscordLMStudioBot,
    message: discord.Message,
    *,
    has_attachments: bool,
) -> str | None:
    if not _is_partner_bot_message(bot, message):
        return None

    content = message.content.strip()
    if not content and not has_attachments:
        return None

    return content or "상대 봇이 방금 보낸 이미지를 보고 답해 줘."


def _should_handle_duo_message(bot: DiscordLMStudioBot, message: discord.Message) -> bool:
    if message.id in bot._seen_duo_message_ids:
        return False

    session = bot.duo_sessions.get(message.channel.id)
    if session is None:
        session = DuoSession()
        bot.duo_sessions[message.channel.id] = session

    if not session.active:
        return False

    if session.stop_reason:
        return False

    if session.replies_sent >= bot.settings.duo_max_replies:
        session.active = False
        session.stop_reason = "max_replies_reached"
        return False

    fingerprint = _duo_message_fingerprint(message)
    if session.last_partner_message_id == message.id:
        session.active = False
        session.stop_reason = "repeated_partner_message_id"
        return False
    if session.last_partner_message_fingerprint == fingerprint:
        session.active = False
        session.stop_reason = "repeated_partner_message_content"
        return False

    bot._seen_duo_message_ids.add(message.id)
    if len(bot._seen_duo_message_ids) > 4096:
        bot._seen_duo_message_ids.clear()
    return True


def _mark_duo_reply_sent(bot: DiscordLMStudioBot, channel_id: int, *, answer: str | None = None) -> None:
    session = bot.duo_sessions.get(channel_id)
    if session is None:
        session = DuoSession()
        bot.duo_sessions[channel_id] = session

    session.replies_sent += 1
    if answer is not None:
        session.last_bot_reply_fingerprint = _duo_text_fingerprint(answer)
    if session.replies_sent >= bot.settings.duo_max_replies:
        session.active = False
        session.stop_reason = "max_replies_reached"
        LOGGER.info(
            "Duo session paused in channel %s after %s replies",
            channel_id,
            session.replies_sent,
        )


def _stop_duo_session(bot: DiscordLMStudioBot, channel_id: int, *, reason: str) -> None:
    if not _is_duo_channel(bot, channel_id):
        return
    session = bot.duo_sessions.get(channel_id)
    if session is None:
        session = DuoSession()
        bot.duo_sessions[channel_id] = session
    session.active = False
    session.stop_reason = reason


def _should_block_duo_answer(
    bot: DiscordLMStudioBot,
    channel_id: int,
    *,
    message: discord.Message,
    answer: str,
) -> bool:
    session = bot.duo_sessions.get(channel_id)
    if session is None:
        session = DuoSession()
        bot.duo_sessions[channel_id] = session
    session.last_partner_message_id = message.id
    session.last_partner_message_fingerprint = _duo_message_fingerprint(message)
    answer_fingerprint = _duo_text_fingerprint(answer)
    if session.last_bot_reply_fingerprint == answer_fingerprint:
        session.active = False
        session.stop_reason = "repeated_bot_reply"
        LOGGER.info("Duo session paused in channel %s due to repeated bot reply", channel_id)
        return True
    return False


def _duo_message_fingerprint(message: discord.Message) -> str:
    content = (message.content or "").strip()
    if not content and message.attachments:
        content = "attachments:" + ",".join(attachment.filename for attachment in message.attachments)
    return _fingerprint_text(content)


def _duo_text_fingerprint(text: str) -> str:
    return _fingerprint_text(text)


def _fingerprint_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    normalized = re.sub(r"<@!?\d+>", "<mention>", normalized)
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


async def _runtime_heartbeat_loop(bot: DiscordLMStudioBot) -> None:
    try:
        while True:
            await asyncio.sleep(bot.runtime_presence.heartbeat_interval_seconds)
            try:
                bot.runtime_presence.heartbeat()
            except Exception:
                LOGGER.exception("Failed to update runtime heartbeat")
    except asyncio.CancelledError:
        raise


def _note_runtime_message_activity(
    bot: DiscordLMStudioBot,
    *,
    message: discord.Message,
    reason: str,
) -> None:
    result = bot.runtime_presence.note_auto_activity(
        owner_id=message.author.id,
        owner_name=getattr(message.author, "display_name", message.author.name),
        channel_id=message.channel.id,
        guild_id=message.guild.id if message.guild else None,
        reason=reason,
    )
    if result is not None and result.conflict:
        LOGGER.info(
            "Runtime activity for %s skipped due to manual claim held by %s in channel %s",
            bot.runtime_presence.bot_name,
            result.holder_name,
            result.holder_channel_id,
        )


def _note_runtime_interaction_activity(
    bot: DiscordLMStudioBot,
    *,
    interaction: discord.Interaction,
    reason: str,
) -> None:
    result = bot.runtime_presence.note_auto_activity(
        owner_id=interaction.user.id,
        owner_name=interaction.user.display_name,
        channel_id=interaction.channel_id,
        guild_id=interaction.guild_id,
        reason=reason,
    )
    if result is not None and result.conflict:
        LOGGER.info(
            "Runtime interaction for %s skipped due to manual claim held by %s in channel %s",
            bot.runtime_presence.bot_name,
            result.holder_name,
            result.holder_channel_id,
        )


def _format_runtime_claim_result(
    bot_name: str,
    result: ClaimResult | None,
    *,
    manual_ttl_seconds: int,
) -> str:
    if result is None:
        return "공유 runtime 상태 기능이 꺼져 있어요."
    if result.acquired:
        ttl_minutes = max(1, manual_ttl_seconds // 60)
        return (
            f"`{bot_name}` 점유를 등록했어요.\n"
            f"- mode: manual\n"
            f"- lease: {ttl_minutes}분"
        )
    if result.conflict:
        holder_name = result.holder_name or "다른 사용자"
        holder_channel = result.holder_channel_id or "unknown"
        holder_reason = result.holder_reason or "manual_claim"
        return (
            f"`{bot_name}`은 이미 사용 중이에요.\n"
            f"- holder: {holder_name}\n"
            f"- channel: {holder_channel}\n"
            f"- reason: {holder_reason}"
        )
    return f"`{bot_name}` 점유 등록에 실패했어요."


def _format_duo_stop_reason(session: DuoSession | None) -> str:
    if session is None:
        return "none"
    return session.stop_reason or "none"


async def _maybe_send_tts_for_message(
    bot: DiscordLMStudioBot,
    *,
    channel: discord.abc.Messageable,
    text: str,
    speaker: str,
    voice_member: discord.abc.User | discord.Member | None = None,
    performance_packet: VTuberTurnPacket | None = None,
) -> None:
    result = await _maybe_dispatch_tts(
        bot,
        text=text,
        speaker=speaker,
        performance_packet=performance_packet,
    )
    if result is None or result.artifact is None:
        return
    if result.mode == "discord_voice":
        if voice_member is None:
            return
        await _maybe_play_tts_in_voice(bot, artifact=result.artifact, voice_member=voice_member)
        return
    if result.mode != "discord_file":
        return
    await channel.send(file=discord.File(str(result.artifact.path), filename=result.artifact.path.name))


async def _maybe_send_tts_for_interaction(
    bot: DiscordLMStudioBot,
    *,
    interaction: discord.Interaction,
    text: str,
    speaker: str,
    voice_member: discord.abc.User | discord.Member | None = None,
    performance_packet: VTuberTurnPacket | None = None,
) -> None:
    result = await _maybe_dispatch_tts(
        bot,
        text=text,
        speaker=speaker,
        performance_packet=performance_packet,
    )
    if result is None or result.artifact is None:
        return
    if result.mode == "discord_voice":
        if voice_member is None:
            return
        await _maybe_play_tts_in_voice(bot, artifact=result.artifact, voice_member=voice_member)
        return
    if result.mode != "discord_file":
        return
    await interaction.followup.send(
        file=discord.File(str(result.artifact.path), filename=result.artifact.path.name)
    )


async def _maybe_dispatch_tts(
    bot: DiscordLMStudioBot,
    *,
    text: str,
    speaker: str,
    performance_packet: VTuberTurnPacket | None = None,
):
    runtime = getattr(bot, "speech_runtime", None)
    if runtime is None or not getattr(runtime, "enabled", False):
        return None
    try:
        utterance = (
            performance_packet.to_utterance()
            if performance_packet is not None
            else Utterance(
                speaker=speaker,
                text=text,
                mood="neutral",
                intent="reply",
            )
        )
        return await runtime.dispatch(
            utterance=utterance,
        )
    except Exception:
        LOGGER.exception("TTS synthesis failed")
        return None


async def _maybe_play_tts_in_voice(
    bot: DiscordLMStudioBot,
    *,
    artifact,
    voice_member: discord.abc.User | discord.Member,
) -> None:
    try:
        await play_artifact_to_member_voice(
            client=bot,
            member=voice_member,
            artifact=artifact,
            ffmpeg_executable=bot.settings.tts_ffmpeg_executable,
        )
    except Exception:
        LOGGER.exception("TTS discord voice playback failed")


def build_bot(settings: Settings) -> DiscordLMStudioBot:
    llm_client = LMStudioClient(settings)
    memory_store = MemoryStore(settings.memory_db_path)
    web_search_client = TavilySearchClient(settings) if settings.web_search_mode != "off" else None
    speech_runtime = build_speech_runtime(
        enabled=settings.tts_enabled,
        mode=settings.tts_mode,
        provider_name=settings.tts_provider,
        output_dir=settings.tts_output_dir,
        command_template=settings.tts_command_template,
        play_command_template=settings.tts_play_command_template,
        audio_format=settings.tts_audio_format,
        max_chars=settings.tts_max_chars,
        local_player_name=settings.tts_local_player,
        obs_output_dir=settings.tts_obs_output_dir,
        xtts_server_url=settings.tts_xtts_server_url,
        xtts_server_token=settings.tts_xtts_server_token,
        xtts_language=settings.tts_xtts_language,
        gptsovits_server_url=settings.tts_gptsovits_server_url,
        gptsovits_server_token=settings.tts_gptsovits_server_token,
        gptsovits_language=settings.tts_gptsovits_language,
        profiles={
            "white": VoiceProfile(
                name="white",
                voice_id=settings.tts_white_voice_id,
                speed=settings.tts_white_speed,
                style=settings.tts_white_style,
            ),
            "black": VoiceProfile(
                name="black",
                voice_id=settings.tts_black_voice_id,
                speed=settings.tts_black_speed,
                style=settings.tts_black_style,
            ),
        },
        elevenlabs_api_key=settings.tts_elevenlabs_api_key,
        elevenlabs_model_id=settings.tts_elevenlabs_model_id,
        elevenlabs_base_url=settings.tts_elevenlabs_base_url,
        elevenlabs_request_timeout_seconds=settings.tts_elevenlabs_request_timeout_seconds,
    )
    bot = DiscordLMStudioBot(settings, llm_client, memory_store, web_search_client, speech_runtime)

    @bot.tree.command(name="ask", description="현재 설정된 vLLM/OpenAI 호환 모델에게 질문합니다.")
    @app_commands.describe(prompt="모델에 보낼 질문")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        await interaction.response.defer(thinking=True)

        channel_id = interaction.channel_id
        guild_id = interaction.guild_id
        user_name = interaction.user.display_name
        _note_runtime_interaction_activity(bot, interaction=interaction, reason="slash_ask")
        if channel_id is not None and _is_duo_channel(bot, channel_id):
            _reset_duo_session(bot, channel_id)
        memory_summary, history, durable_memories = _load_memory_context(
            bot,
            channel_id=channel_id,
            user_id=interaction.user.id,
            prompt=prompt,
        )

        try:
            search_context = await _maybe_search_web(bot, prompt, user_name=user_name)
            answer = await bot.llm_client.ask(
                prompt=prompt,
                user_name=user_name,
                history=history,
                web_context=search_context.prompt_context if search_context else None,
                memory_summary=memory_summary,
                durable_memories=durable_memories,
                reply_mode="interaction",
            )
            if search_context:
                answer = _append_search_sources(answer, search_context)
            chunks = list(chunk_for_discord(answer))
            await interaction.followup.send(chunks[0])
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)
            performance_packet = bot.performance_brain.build_turn_packet(
                event=WhiteRuntimeEvent(
                    kind="slash_ask",
                    prompt=prompt,
                    user_name=user_name,
                    reply_mode="interaction",
                    search_used=search_context is not None,
                ),
                reply=answer,
                search_context=search_context,
            )
            await _maybe_send_tts_for_interaction(
                bot,
                interaction=interaction,
                text=answer,
                speaker="white",
                voice_member=interaction.user,
                performance_packet=performance_packet,
            )

            if channel_id is not None:
                _persist_conversation_turn(
                    bot,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=interaction.user.id,
                    user_name=user_name,
                    prompt=prompt,
                    answer=answer,
                    discord_message_id=interaction.id,
                )
                _schedule_summary_refresh(bot, guild_id=guild_id, channel_id=channel_id)
        except Exception:
            LOGGER.exception("Failed to handle /ask")
            await interaction.followup.send(_user_visible_error_message("LLM 요청"))

    @bot.tree.command(name="search", description="답변 전에 Tavily 웹검색을 먼저 실행합니다.")
    @app_commands.describe(prompt="웹검색과 함께 답할 질문")
    async def search(interaction: discord.Interaction, prompt: str) -> None:
        await interaction.response.defer(thinking=True)

        if bot.web_search_client is None:
            await interaction.followup.send(
                "웹검색 기능이 꺼져 있어요. `.env`에서 `WEB_SEARCH_MODE`와 `TAVILY_API_KEY`를 먼저 설정해 주세요."
            )
            return

        channel_id = interaction.channel_id
        guild_id = interaction.guild_id
        user_name = interaction.user.display_name
        _note_runtime_interaction_activity(bot, interaction=interaction, reason="slash_search")
        if channel_id is not None and _is_duo_channel(bot, channel_id):
            _reset_duo_session(bot, channel_id)
        memory_summary, history, durable_memories = _load_memory_context(
            bot,
            channel_id=channel_id,
            user_id=interaction.user.id,
            prompt=prompt,
        )

        try:
            search_context = await _maybe_search_web(bot, prompt, user_name=user_name, force=True)
            if search_context is None:
                await interaction.followup.send("웹검색 결과를 찾지 못했어요.")
                return

            answer = await bot.llm_client.ask(
                prompt=prompt,
                user_name=user_name,
                history=history,
                web_context=search_context.prompt_context,
                memory_summary=memory_summary,
                durable_memories=durable_memories,
                reply_mode="interaction",
            )
            answer = _append_search_sources(answer, search_context)
            chunks = list(chunk_for_discord(answer))
            await interaction.followup.send(chunks[0])
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)
            performance_packet = bot.performance_brain.build_turn_packet(
                event=WhiteRuntimeEvent(
                    kind="slash_search",
                    prompt=prompt,
                    user_name=user_name,
                    reply_mode="interaction",
                    search_used=True,
                ),
                reply=answer,
                search_context=search_context,
            )
            await _maybe_send_tts_for_interaction(
                bot,
                interaction=interaction,
                text=answer,
                speaker="white",
                voice_member=interaction.user,
                performance_packet=performance_packet,
            )

            if channel_id is not None:
                _persist_conversation_turn(
                    bot,
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=interaction.user.id,
                    user_name=user_name,
                    prompt=prompt,
                    answer=answer,
                    discord_message_id=interaction.id,
                )
                _schedule_summary_refresh(bot, guild_id=guild_id, channel_id=channel_id)
        except Exception:
            LOGGER.exception("Failed to handle /search")
            await interaction.followup.send(_user_visible_error_message("웹검색"))

    @bot.tree.command(name="ping", description="Discord와 vLLM/OpenAI 호환 endpoint 연결 상태를 확인합니다.")
    async def ping(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)

        discord_latency_ms = round(bot.latency * 1000)
        try:
            models = await bot.llm_client.list_models()
            loaded_text = ", ".join(models) if models else "없음"
            await interaction.followup.send(
                f"Discord 지연시간: {discord_latency_ms}ms\nLLM endpoint: 정상\n모델: {loaded_text}",
                ephemeral=True,
            )
        except Exception:
            LOGGER.exception("Failed to handle /ping")
            await interaction.followup.send(
                f"Discord 지연시간: {discord_latency_ms}ms\nLLM endpoint 연결 확인 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요.",
                ephemeral=True,
            )

    @bot.tree.command(name="model", description="현재 설정된 모델과 endpoint 모델 목록을 확인합니다.")
    async def model(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)

        try:
            models = await bot.llm_client.list_models()
            model_lines = "\n".join(f"- {model_id}" for model_id in models) if models else "- 없음"
            await interaction.followup.send(
                "현재 설정\n"
                f"- VLLM_MODEL: {bot.settings.lm_studio_model}\n"
                f"- WEB_SEARCH_MODE: {bot.settings.web_search_mode}\n"
                f"- DUO_ENABLED: {bot.settings.duo_enabled}\n"
                f"- DUO_PARTNER_BOT_ID: {bot.settings.duo_partner_bot_id}\n\n"
                "Endpoint 모델 목록\n"
                f"{model_lines}",
                ephemeral=True,
            )
        except Exception:
            LOGGER.exception("Failed to handle /model")
            await interaction.followup.send(_user_visible_error_message("모델 목록 조회"), ephemeral=True)

    @bot.tree.command(name="summary", description="현재 채널의 장기 요약과 주요 기억을 확인합니다.")
    async def summary(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel_id = interaction.channel_id
        if channel_id is None:
            await interaction.followup.send("채널에서만 쓸 수 있어요.", ephemeral=True)
            return
        summary_entry = bot.memory_store.load_latest_summary(channel_id=channel_id)
        memories = bot.memory_store.search_durable_memories(
            channel_id=channel_id,
            user_id=interaction.user.id,
            prompt="최근 이야기 요약",
            limit=3,
        )
        await interaction.followup.send(
            build_white_summary_report(summary=summary_entry, memories=memories),
            ephemeral=True,
        )

    @bot.tree.command(name="recall", description="관련 장기기억을 다시 꺼내 봅니다.")
    @app_commands.describe(query="되짚어볼 주제나 키워드")
    async def recall(interaction: discord.Interaction, query: str | None = None) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel_id = interaction.channel_id
        if channel_id is None:
            await interaction.followup.send("채널에서만 쓸 수 있어요.", ephemeral=True)
            return
        recall_query = (query or "").strip() or "최근 이야기"
        summary_entry = bot.memory_store.load_latest_summary(channel_id=channel_id)
        memories = bot.memory_store.search_durable_memories(
            channel_id=channel_id,
            user_id=interaction.user.id,
            prompt=recall_query,
            limit=4,
        )
        await interaction.followup.send(
            build_white_recall_report(
                query=recall_query,
                summary=summary_entry,
                memories=memories,
            ),
            ephemeral=True,
        )

    @bot.tree.command(name="dashboard", description="현재 채널의 memory/runtime/TTS 상태를 확인합니다.")
    async def dashboard(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel_id = interaction.channel_id
        if channel_id is None:
            await interaction.followup.send("채널에서만 쓸 수 있어요.", ephemeral=True)
            return
        summary_entry = bot.memory_store.load_latest_summary(channel_id=channel_id)
        recent_user_messages = bot.memory_store.load_recent_messages(
            channel_id=channel_id,
            user_id=interaction.user.id,
            role="user",
            limit=6,
        )
        report = build_white_dashboard_report(
            model_name=bot.settings.lm_studio_model,
            web_search_mode=bot.settings.web_search_mode,
            tts_enabled=bot.settings.tts_enabled,
            tts_mode=bot.settings.tts_mode,
            runtime_report=bot.runtime_presence.format_status_report(),
            message_count=bot.memory_store.count_messages(channel_id=channel_id),
            user_memory_counts=bot.memory_store.count_durable_memories_for_scope(
                scope_key=f"user:{interaction.user.id}"
            ),
            channel_memory_counts=bot.memory_store.count_durable_memories_for_scope(
                scope_key=f"channel:{channel_id}"
            ),
            summary=summary_entry,
            signals=analyze_recent_signals(recent_user_messages),
        )
        await interaction.followup.send(report, ephemeral=True)

    @bot.tree.command(name="signals", description="최근 대화에서 읽히는 감정 신호를 확인합니다.")
    async def signals(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        channel_id = interaction.channel_id
        if channel_id is None:
            await interaction.followup.send("채널에서만 쓸 수 있어요.", ephemeral=True)
            return
        recent_user_messages = bot.memory_store.load_recent_messages(
            channel_id=channel_id,
            user_id=interaction.user.id,
            role="user",
            limit=6,
        )
        memories = bot.memory_store.search_durable_memories(
            channel_id=channel_id,
            user_id=interaction.user.id,
            prompt="최근 기분과 상태",
            limit=3,
        )
        await interaction.followup.send(
            build_white_signal_report(
                recent_messages=recent_user_messages,
                memories=memories,
            ),
            ephemeral=True,
        )

    if settings.runtime_state_enabled:
        @bot.tree.command(name="runtime_status", description="공용 runtime 상태를 확인합니다.")
        async def runtime_status(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(
                bot.runtime_presence.format_status_report(),
                ephemeral=True,
            )

        @bot.tree.command(name="runtime_claim", description="현재 채널에서 이 봇 점유를 등록합니다.")
        async def runtime_claim(interaction: discord.Interaction) -> None:
            result = bot.runtime_presence.claim_manual(
                owner_id=interaction.user.id,
                owner_name=interaction.user.display_name,
                channel_id=interaction.channel_id,
                guild_id=interaction.guild_id,
                reason="manual_claim",
            )
            await interaction.response.send_message(
                _format_runtime_claim_result(
                    bot.runtime_presence.bot_name,
                    result,
                    manual_ttl_seconds=bot.runtime_presence.manual_claim_ttl_seconds,
                ),
                ephemeral=True,
            )

        @bot.tree.command(name="runtime_release", description="현재 봇 점유를 해제합니다.")
        async def runtime_release(interaction: discord.Interaction) -> None:
            released = bot.runtime_presence.release(owner_id=interaction.user.id)
            message = (
                f"`{bot.runtime_presence.bot_name}` 점유를 해제했어요."
                if released
                else f"`{bot.runtime_presence.bot_name}` 점유를 해제하지 못했어요."
            )
            await interaction.response.send_message(message, ephemeral=True)

    if settings.duo_enabled:
        @bot.tree.command(name="duo_start", description="듀오 모드에서 이 봇이 먼저 말을 시작합니다.")
        @app_commands.describe(prompt="듀오 대화를 시작할 첫 문장")
        async def duo_start(interaction: discord.Interaction, prompt: str) -> None:
            await interaction.response.defer(thinking=True)

            channel_id = interaction.channel_id
            if channel_id is None:
                await interaction.followup.send("이 명령은 채널에서만 사용할 수 있어요.", ephemeral=True)
                return
            if not _is_duo_channel(bot, channel_id):
                await interaction.followup.send(
                    "이 채널은 듀오 모드 대상이 아니에요. `.env`의 `DUO_CHANNEL_IDS`에 채널 ID를 넣거나 비워 두세요.",
                    ephemeral=True,
                )
                return

            _reset_duo_session(bot, channel_id)
            _note_runtime_interaction_activity(bot, interaction=interaction, reason="duo_start")
            await interaction.followup.send(prompt)

        @bot.tree.command(name="duo_stop", description="현재 채널의 듀오 모드를 중지합니다.")
        @app_commands.describe(reason="중지 사유")
        async def duo_stop(interaction: discord.Interaction, reason: str = "manual_stop") -> None:
            await interaction.response.defer(ephemeral=True, thinking=True)

            channel_id = interaction.channel_id
            if channel_id is None:
                await interaction.followup.send("이 명령은 채널에서만 사용할 수 있어요.", ephemeral=True)
                return
            if not _is_duo_channel(bot, channel_id):
                await interaction.followup.send(
                    "이 채널은 듀오 모드 대상이 아니에요. `.env`의 `DUO_CHANNEL_IDS`를 확인해 주세요.",
                    ephemeral=True,
                )
                return

            _stop_duo_session(bot, channel_id, reason=reason.strip() or "manual_stop")
            await interaction.followup.send(
                f"듀오 모드를 중지했어요.\n- reason: {reason.strip() or 'manual_stop'}",
                ephemeral=True,
            )

        @bot.tree.command(name="duo_status", description="현재 채널의 듀오 모드 상태를 확인합니다.")
        async def duo_status(interaction: discord.Interaction) -> None:
            channel_id = interaction.channel_id
            session = bot.duo_sessions.get(channel_id) if channel_id is not None else None
            scope = (
                ", ".join(str(item) for item in bot.settings.duo_channel_ids)
                if bot.settings.duo_channel_ids
                else "보이는 모든 채널"
            )
            await interaction.response.send_message(
                "듀오 모드 상태\n"
                f"- 활성화 여부: {bot.settings.duo_enabled}\n"
                f"- 상대 봇 ID: {bot.settings.duo_partner_bot_id}\n"
                f"- 동작 채널 범위: {scope}\n"
                f"- 봇당 최대 응답 횟수: {bot.settings.duo_max_replies}\n"
                f"- 응답 지연: {bot.settings.duo_reply_delay_seconds}초\n"
                f"- 현재 채널 활성 상태: {session.active if session else False}\n"
                f"- 현재 채널 응답 횟수: {session.replies_sent if session else 0}\n"
                f"- 현재 채널 stop reason: {_format_duo_stop_reason(session)}",
                ephemeral=True,
            )

    return bot


def _extract_chat_prompt(
    bot: DiscordLMStudioBot,
    message: discord.Message,
    has_attachments: bool,
) -> str | None:
    content = message.content.strip()
    if not content and not has_attachments:
        return None

    if content.startswith("/"):
        return None

    if bot.user and bot.settings.reply_when_mentioned and bot.user in message.mentions:
        cleaned = re.sub(r"<@!?\d+>", "", content).strip()
        return cleaned or ("이 이미지를 분석해 줘." if has_attachments else "방금 보낸 메시지에 답해 줘.")

    if bot.settings.chat_reply_enabled and message.channel.id in bot.settings.chat_channel_ids:
        return content or "이 이미지를 분석해 줘."

    return None


def _load_memory_context(
    bot: DiscordLMStudioBot,
    *,
    channel_id: int | None,
    user_id: int | None,
    prompt: str,
) -> tuple[str | None, list[dict[str, str]], list[DurableMemory]]:
    if channel_id is None:
        return None, [], []

    summary = bot.memory_store.load_latest_summary(channel_id=channel_id)
    history = bot.memory_store.load_recent_history(
        channel_id=channel_id,
        limit=bot.settings.memory_recent_messages,
    )
    durable_memories = bot.memory_store.search_durable_memories(
        channel_id=channel_id,
        user_id=user_id,
        prompt=prompt,
        limit=4,
    )
    return (summary.summary_text if summary else None), history, durable_memories


def _persist_conversation_turn(
    bot: DiscordLMStudioBot,
    *,
    guild_id: int | None,
    channel_id: int,
    user_id: int | None,
    user_name: str,
    prompt: str,
    answer: str,
    discord_message_id: int | None,
) -> None:
    user_message_id = bot.memory_store.save_message(
        discord_message_id=discord_message_id,
        guild_id=guild_id,
        channel_id=channel_id,
        user_id=user_id,
        user_name=user_name,
        role="user",
        content=prompt,
    )
    if user_id is not None:
        durable_note = _extract_durable_user_memory_note(prompt)
        if durable_note:
            bot.memory_store.save_user_memory_note(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                user_name=user_name,
                memory_text=durable_note,
                source_message_id=user_message_id,
                memory_kind=classify_durable_memory_kind(durable_note),
            )
    if _should_store_assistant_answer(answer):
        bot.memory_store.save_message(
            discord_message_id=None,
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=bot.user.id if bot.user else None,
            user_name=bot.user.name if bot.user else "assistant",
            role="assistant",
            content=answer,
        )
    else:
        LOGGER.warning("Skipping assistant memory save for suspicious output in channel %s", channel_id)


def _should_store_assistant_answer(answer: str) -> bool:
    normalized = answer.strip()
    if not normalized:
        return False
    if normalized == "응답이 비어 있어요.":
        return False
    if "<think>" in normalized.lower() or "</think>" in normalized.lower():
        return False

    latin1_like = len(re.findall(r"[À-ÿ]", normalized))
    hangul = len(re.findall(r"[가-힣]", normalized))
    if latin1_like >= 2 and latin1_like >= hangul:
        return False

    return True


def _schedule_summary_refresh(bot: DiscordLMStudioBot, *, guild_id: int | None, channel_id: int) -> None:
    bot._summary_refresh_pending_channels.add(channel_id)
    existing_task = bot._summary_refresh_tasks.get(channel_id)
    if existing_task is not None and not existing_task.done():
        return

    async def _runner() -> None:
        try:
            while True:
                bot._summary_refresh_pending_channels.discard(channel_id)
                try:
                    await _maybe_refresh_summary(bot, guild_id=guild_id, channel_id=channel_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception("Background summary refresh failed for channel %s", channel_id)
                if channel_id not in bot._summary_refresh_pending_channels:
                    break
        finally:
            current_task = bot._summary_refresh_tasks.get(channel_id)
            if current_task is task:
                bot._summary_refresh_tasks.pop(channel_id, None)
            bot._summary_refresh_pending_channels.discard(channel_id)

    task = asyncio.create_task(_runner(), name=f"white-summary-refresh-{channel_id}")
    bot._summary_refresh_tasks[channel_id] = task


def _extract_durable_user_memory_note(prompt: str) -> str | None:
    normalized = prompt.strip()
    if not normalized:
        return None
    if normalized.endswith("?"):
        return None
    if len(normalized) < 10:
        return None
    if normalized.startswith("/"):
        return None

    durable_markers = (
        "요즘",
        "최근",
        "자꾸",
        "계속",
        "아직",
        "오랜만",
        "준비",
        "면접",
        "시험",
        "프로젝트",
        "취준",
        "취업",
        "이직",
        "퇴사",
        "회사",
        "학교",
        "이사",
        "가족",
        "친구",
        "연락",
        "병원",
        "약",
        "불안",
        "우울",
        "공황",
        "잠",
        "불면",
        "아프",
        "수술",
        "치료",
    )
    if not any(marker in normalized for marker in durable_markers):
        return None

    return prepare_durable_memory_capture(normalized, max_length=140)


async def _maybe_refresh_summary(bot: DiscordLMStudioBot, *, guild_id: int | None, channel_id: int) -> None:
    latest_summary = bot.memory_store.load_latest_summary(channel_id=channel_id)
    after_message_id = latest_summary.source_until_message_id if latest_summary else 0
    unsummarized_count = bot.memory_store.count_messages_after(channel_id=channel_id, after_message_id=after_message_id)

    if unsummarized_count <= bot.settings.memory_summary_trigger_messages:
        return

    available_to_summarize = unsummarized_count - bot.settings.memory_recent_messages
    if available_to_summarize <= 0:
        return

    batch_size = min(available_to_summarize, bot.settings.memory_summary_batch_messages)
    messages = bot.memory_store.load_messages_after(
        channel_id=channel_id,
        after_message_id=after_message_id,
        limit=batch_size,
    )
    if not messages:
        return

    updated_summary = await bot.llm_client.summarize_conversation(
        existing_summary=latest_summary.summary_text if latest_summary else None,
        messages=messages,
    )
    if not updated_summary:
        return

    bot.memory_store.save_summary(
        guild_id=guild_id,
        channel_id=channel_id,
        summary_text=updated_summary,
        source_until_message_id=messages[-1].id,
    )
    LOGGER.info("Updated long-term summary for channel %s up to message %s", channel_id, messages[-1].id)


async def _load_image_inputs(message: discord.Message) -> list[dict[str, object]]:
    image_inputs: list[dict[str, object]] = []

    for attachment in message.attachments:
        if not _is_supported_image(attachment):
            continue

        image_bytes = await attachment.read()
        media_type = attachment.content_type or mimetypes.guess_type(attachment.filename)[0] or "image/png"
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        image_inputs.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_base64}",
                },
            }
        )

    return image_inputs


def _is_supported_image(attachment: discord.Attachment) -> bool:
    if attachment.content_type and attachment.content_type.startswith("image/"):
        return True

    guessed_type, _ = mimetypes.guess_type(attachment.filename)
    return bool(guessed_type and guessed_type.startswith("image/"))


async def _maybe_search_web(
    bot: DiscordLMStudioBot,
    prompt: str,
    user_name: str,
    *,
    force: bool = False,
) -> SearchContext | None:
    if bot.web_search_client is None:
        return None

    if force:
        request = build_search_request_from_query(prompt, forced=True)
    else:
        request = extract_explicit_search_request(prompt)
        if request is None:
            if bot.settings.web_search_mode == "auto":
                decision = await bot.llm_client.decide_web_search(prompt=prompt, user_name=user_name)
                if not decision.should_search:
                    return None
                request = build_search_request_from_query(decision.query or prompt, forced=False)
            else:
                request = build_search_request(prompt, mode=bot.settings.web_search_mode)
    if request is None:
        return None

    try:
        return await bot.web_search_client.search(request)
    except Exception:
        LOGGER.exception("Web search failed")
        if force or request.forced:
            raise
        return None


def _append_search_sources(answer: str, search_context: SearchContext) -> str:
    source_block = format_source_links(search_context.results)
    if not source_block:
        return answer
    return f"{answer.rstrip()}{source_block}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = load_settings()
    startup_lock = SingletonStartupLock(
        settings.startup_singleton_lock_path,
        enabled=settings.startup_singleton_enabled,
        label="white",
    )
    try:
        startup_lock.acquire()
    except SingletonLockError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc

    runtime_pid_file = _write_runtime_pid_file()
    bot = build_bot(settings)
    try:
        bot.run(settings.discord_token, log_handler=None)
    finally:
        _cleanup_runtime_pid_file(runtime_pid_file)
        startup_lock.close()
