from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
import sys
import traceback
from pathlib import Path

import discord

from predictive_bot.config import AppConfig
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.runtime_state import ClaimResult, RuntimePresenceClient
from predictive_bot.discord_app.inspector import (
    format_black_dashboard,
    format_black_recall,
    format_black_state,
    format_black_summary,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from bot_shared.speech import Utterance, disconnect_all_client_voice_connections, play_artifact_to_member_voice
from bot_shared.vtuber import VTuberTurnPacket


@dataclass(slots=True)
class DuoSessionState:
    active: bool = False
    session_token: int = 0
    channel_id: int | None = None
    partner_bot_id: str | None = None
    prompt: str | None = None
    turns_sent: int = 0
    last_partner_signature: str | None = None
    last_my_reply_signature: str | None = None
    stopped_reason: str | None = None


@dataclass(slots=True)
class DuoCommand:
    kind: str
    prompt: str | None = None


@dataclass(slots=True)
class RuntimeCommand:
    kind: str
    argument: str | None = None


class PredictiveDiscordClient(discord.Client):
    def __init__(self, *, config: AppConfig, engine: PredictiveEngine, speech_runtime=None) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        super().__init__(intents=intents)
        self.config = config
        self.engine = engine
        self.speech_runtime = speech_runtime
        self.runtime_presence = RuntimePresenceClient(
            enabled=config.runtime_state_enabled,
            db_path=config.runtime_state_db_path,
            bot_name=config.runtime_bot_name,
            project_name="predictive-discord-bot",
            heartbeat_interval_seconds=config.runtime_heartbeat_seconds,
            online_timeout_seconds=config.runtime_online_timeout_seconds,
            auto_claim_ttl_seconds=config.runtime_auto_claim_ttl_seconds,
            manual_claim_ttl_seconds=config.runtime_manual_claim_ttl_seconds,
        )
        self._runtime_heartbeat_task: asyncio.Task[None] | None = None
        self._runtime_heartbeat_observed = False
        self._duo_session = DuoSessionState()
        self._duo_autostart_attempted = False

    def _close_engine_state_store(self) -> None:
        state_store = getattr(self.engine, "state_store", None)
        if state_store is None:
            return
        try:
            state_store.close()
        except Exception:
            print("[runtime] state_store close failed.")
            traceback.print_exc()

    async def on_ready(self) -> None:
        if self.user is not None and self.runtime_presence.enabled:
            self.runtime_presence.activate(
                discord_user_id=self.user.id,
                display_name=self.user.name,
            )
            if self._runtime_heartbeat_task is None or self._runtime_heartbeat_task.done():
                self._runtime_heartbeat_task = asyncio.create_task(
                    self._runtime_heartbeat_loop(),
                    name=f"{self.runtime_presence.bot_name}-runtime-heartbeat",
                )
            print(
                f"[runtime] presence activated bot={self.runtime_presence.bot_name!r} "
                f"heartbeat_interval={self.runtime_presence.heartbeat_interval_seconds}"
            )
            print(
                f"[runtime] heartbeat task started interval={self.runtime_presence.heartbeat_interval_seconds}s"
            )
        print(f"Logged in as {self.user} (id={self.user.id})")
        print(
            f"[startup] prefix={self.config.bot_trigger_prefix!r} "
            "DM, 멘션, prefix 메시지에 응답합니다."
        )
        if self.config.duo_mode_enabled:
            print(
                f"[startup] duo_mode enabled partner={self.config.duo_partner_bot_id!r} "
                f"channel={self.config.duo_channel_id!r} max_turns={self.config.duo_max_turns_per_bot}"
            )
            print(
                f"[startup] duo_autostart enabled={self.config.duo_autostart_enabled} "
                f"channel={self.config.duo_autostart_channel_id!r}"
            )
            await self._maybe_autostart_duo_session()

    async def close(self) -> None:
        if self._runtime_heartbeat_task is not None:
            self._runtime_heartbeat_task.cancel()
            try:
                await self._runtime_heartbeat_task
            except asyncio.CancelledError:
                pass
            self._runtime_heartbeat_task = None
        try:
            self.runtime_presence.mark_offline()
        except Exception:
            print("[runtime] mark_offline failed.")
            traceback.print_exc()
        finally:
            self._close_engine_state_store()
            if self.speech_runtime is not None:
                await self.speech_runtime.close()
            await disconnect_all_client_voice_connections(self)
        if not self.is_closed():
            await super().close()

    async def on_message(self, message: discord.Message) -> None:
        try:
            if message.author.bot:
                if not self._should_accept_partner_bot_message(message):
                    return
                await self._handle_partner_bot_message(message)
                return

            duo_command = self._extract_duo_command(message)
            if duo_command is not None:
                await self._handle_duo_command(message, duo_command)
                return

            if self._should_reset_duo_on_human_message(message):
                self._reset_duo_session("human_message_reset")

            should_respond = self._should_respond(message)
            content_for_log = message.content if self.config.log_message_content else f"<len={len(message.content)}>"
            print(
                "[message] "
                f"author={message.author} "
                f"guild={getattr(message.guild, 'name', 'DM')} "
                f"channel={getattr(message.channel, 'name', type(message.channel).__name__)} "
                f"mentioned={self._is_directly_mentioned(message)} "
                f"role_mention_match={self._matches_name_role_mention(message)} "
                f"raw_mentions={message.raw_mentions} "
                f"should_respond={should_respond} "
                f"content={content_for_log!r}"
            )

            if message.guild is not None and not message.content.strip():
                print(
                    "[hint] guild 메시지 내용이 비어 있습니다. "
                    "Discord Developer Portal에서 Message Content Intent가 켜져 있는지 확인하세요."
                )

            if not should_respond:
                return

            user_text = self._extract_user_text(message)
            print(f"[parsed] user_text={user_text!r}")
            if not user_text:
                await message.channel.send("멘션 뒤에 한 줄만 붙여주면 그 기준으로 답할게.")
                return
            runtime_command = self._extract_runtime_command(message, user_text)
            if runtime_command is not None:
                await self._handle_runtime_command(message, runtime_command)
                return

            self._note_runtime_activity(message, reason="predictive_reply")

            result = await self.engine.respond(user_id=str(message.author.id), text=user_text)
            await self._send_text_with_optional_tts(
                message.channel,
                result.reply,
                speaker="black",
                voice_member=message.author,
                performance_packet=getattr(result, "performance_packet", None),
            )
            print(f"[decision] user={message.author.id} {result.audit_record.format_for_log(include_reply=True)}")
        except Exception:
            print("[error] on_message 처리 중 예외가 발생했습니다.")
            traceback.print_exc()

    def _should_respond(self, message: discord.Message) -> bool:
        if message.guild is None:
            return True
        if self._is_directly_mentioned(message):
            return True
        if self._matches_name_role_mention(message):
            return True
        if message.reference and message.reference.resolved:
            resolved = message.reference.resolved
            if isinstance(resolved, discord.Message) and self.user and resolved.author.id == self.user.id:
                return True
        if self.config.bot_trigger_prefix and message.content.strip().startswith(self.config.bot_trigger_prefix):
            return True
        return False

    def _extract_user_text(self, message: discord.Message) -> str:
        content = message.content.strip()
        if self.user:
            mention_patterns = [
                re.escape(self.user.mention),
                rf"<@!?{self.user.id}>",
            ]
            for pattern in mention_patterns:
                content = re.sub(pattern, "", content).strip()
        for role in message.role_mentions:
            content = content.replace(role.mention, "").strip()
        if self.config.bot_trigger_prefix and content.startswith(self.config.bot_trigger_prefix):
            content = content[len(self.config.bot_trigger_prefix) :].strip()
        return content

    def _is_directly_mentioned(self, message: discord.Message) -> bool:
        if not self.user:
            return False
        return self.user.mentioned_in(message) or self.user.id in message.raw_mentions

    def _matches_name_role_mention(self, message: discord.Message) -> bool:
        if not self.user or not message.guild or not message.role_mentions:
            return False

        candidate_names = {self.user.name.casefold()}
        global_name = getattr(self.user, "global_name", None)
        if global_name:
            candidate_names.add(global_name.casefold())

        member = message.guild.get_member(self.user.id)
        if member and member.display_name:
            candidate_names.add(member.display_name.casefold())

        for role in message.role_mentions:
            if role.name.casefold() in candidate_names:
                return True
        return False

    def _extract_runtime_command(self, message: discord.Message, user_text: str) -> RuntimeCommand | None:
        if not self.config.bot_trigger_prefix:
            return None
        if not message.content.strip().startswith(self.config.bot_trigger_prefix):
            return None

        normalized = " ".join(user_text.casefold().split())
        command_map = {
            "runtime status": RuntimeCommand(kind="status"),
            "runtime claim": RuntimeCommand(kind="claim"),
            "runtime release": RuntimeCommand(kind="release"),
            "dashboard": RuntimeCommand(kind="dashboard"),
            "summary": RuntimeCommand(kind="summary"),
            "state": RuntimeCommand(kind="state"),
            "mood": RuntimeCommand(kind="state"),
        }
        direct = command_map.get(normalized)
        if direct is not None:
            return direct
        tokens = user_text.split(None, 1)
        if not tokens:
            return None
        if tokens[0].casefold() == "recall":
            argument = tokens[1].strip() if len(tokens) > 1 else ""
            return RuntimeCommand(kind="recall", argument=argument or None)
        return None

    def _extract_duo_command(self, message: discord.Message) -> DuoCommand | None:
        if not self.config.duo_mode_enabled or not self.config.bot_trigger_prefix:
            return None
        content = message.content.strip()
        if not content.startswith(self.config.bot_trigger_prefix):
            return None

        user_text = self._extract_user_text(message)
        tokens = user_text.split(None, 2)
        if len(tokens) < 2:
            return None
        if tokens[0].casefold() != "duo":
            return None

        command = tokens[1].casefold()
        if command == "status":
            return DuoCommand(kind="status")
        if command == "stop":
            return DuoCommand(kind="stop")
        if command == "start":
            prompt = tokens[2].strip() if len(tokens) >= 3 else ""
            return DuoCommand(kind="start", prompt=prompt or None)
        return None

    async def _handle_duo_command(self, message: discord.Message, command: DuoCommand) -> None:
        if not self.config.duo_mode_enabled:
            await self._send_private_notice(message, "duo mode가 꺼져 있어.")
            return
        if not self._is_duo_channel(message):
            await self._send_private_notice(message, "duo channel이 아니야.")
            return

        if command.kind == "start":
            if not command.prompt:
                await self._send_private_notice(message, "사용법: `!predict duo start <prompt>`")
                return
            self._begin_duo_session(
                channel_id=message.channel.id,
                partner_bot_id=self.config.duo_partner_bot_id,
                prompt=command.prompt,
                reason="manual_start",
            )
            await message.channel.send(command.prompt)
            print(
                f"[duo] started channel={message.channel.id} partner={self.config.duo_partner_bot_id!r} "
                f"prompt={command.prompt!r}"
            )
            return

        if command.kind == "status":
            await self._send_private_notice(message, self._format_duo_status())
            return

        if command.kind == "stop":
            stopped = self._stop_duo_session("manual_stop")
            await self._send_private_notice(
                message,
                "duo session을 중지했어." if stopped else "duo session이 이미 꺼져 있어.",
            )
            return

    async def _maybe_autostart_duo_session(self) -> None:
        if self._duo_autostart_attempted:
            return
        self._duo_autostart_attempted = True

        if not self.config.duo_mode_enabled or not self.config.duo_autostart_enabled:
            return
        if self._duo_session.active:
            print("[duo] autostart skipped: session already active")
            return
        if not self.config.duo_partner_bot_id:
            print("[duo] autostart skipped: partner bot id is missing")
            return
        if not self.config.duo_channel_id:
            print("[duo] autostart skipped: duo channel id is missing")
            return
        if not self.config.duo_autostart_channel_id:
            print("[duo] autostart skipped: autostart channel id is missing")
            return
        if self.config.duo_channel_id != self.config.duo_autostart_channel_id:
            print(
                "[duo] autostart skipped: autostart channel does not match duo channel "
                f"duo={self.config.duo_channel_id!r} autostart={self.config.duo_autostart_channel_id!r}"
            )
            return

        prompt = (self.config.duo_autostart_prompt or "").strip()
        if not prompt:
            print("[duo] autostart skipped: prompt is empty")
            return

        try:
            channel_id = int(self.config.duo_autostart_channel_id)
        except (TypeError, ValueError):
            print(
                f"[duo] autostart skipped: invalid channel id={self.config.duo_autostart_channel_id!r}"
            )
            return

        channel = self.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(channel_id)
            except Exception:
                print(f"[duo] autostart skipped: channel not found channel_id={channel_id}")
                traceback.print_exc()
                return

        self._begin_duo_session(
            channel_id=channel_id,
            partner_bot_id=self.config.duo_partner_bot_id,
            prompt=prompt,
            reason="autostart",
        )
        try:
            await channel.send(prompt)
        except Exception:
            print(f"[duo] autostart failed: unable to send prompt channel_id={channel_id}")
            traceback.print_exc()
            self._stop_duo_session("autostart_send_failed")
            return

        print(
            f"[duo] autostarted channel={channel_id} partner={self.config.duo_partner_bot_id!r} "
            f"prompt={prompt!r}"
        )

    async def _handle_partner_bot_message(self, message: discord.Message) -> None:
        if not self._is_duo_channel(message):
            return
        if not self._is_duo_partner(message):
            return
        if not self._duo_session.active:
            self._begin_duo_session(
                channel_id=message.channel.id,
                partner_bot_id=self.config.duo_partner_bot_id,
                prompt=None,
                reason="implicit_partner_start",
            )
            print(
                f"[duo] implicitly started from partner message "
                f"channel={message.channel.id} partner={self.config.duo_partner_bot_id!r}"
            )

        user_text = self._extract_user_text(message)
        if not user_text:
            self._stop_duo_session("empty_partner_message")
            return

        partner_signature = self._normalize_duo_signature(user_text)
        if self._duo_session.last_partner_signature == partner_signature:
            self._stop_duo_session("repeated_partner_message")
            print("[duo] stopped: repeated partner message")
            return

        if self._duo_session.turns_sent >= self.config.duo_max_turns_per_bot:
            self._stop_duo_session("max_turns_reached")
            print("[duo] stopped: max turns reached")
            return

        duo_user_id = self._duo_session_user_id(message)
        self._note_runtime_activity(message, reason="duo_partner_reply")
        result = await self.engine.respond(user_id=duo_user_id, text=user_text)
        reply_signature = self._normalize_duo_signature(result.reply)
        if self._duo_session.last_my_reply_signature == reply_signature:
            self._stop_duo_session("repeated_bot_reply")
            print("[duo] stopped: repeated bot reply")
            return

        await self._send_text_with_optional_tts(
            message.channel,
            result.reply,
            speaker="black",
            voice_member=None,
            performance_packet=getattr(result, "performance_packet", None),
        )
        self._duo_session.turns_sent += 1
        self._duo_session.last_partner_signature = partner_signature
        self._duo_session.last_my_reply_signature = reply_signature
        audit_record = getattr(result, "audit_record", None)
        audit_text = (
            audit_record.format_for_log(include_reply=True)
            if audit_record is not None
            else f"reply={result.reply!r}"
        )
        print(f"[duo] turns_sent={self._duo_session.turns_sent} {audit_text}")

    def _should_reset_duo_on_human_message(self, message: discord.Message) -> bool:
        if not self.config.duo_mode_enabled:
            return False
        return self._duo_session.active and self._is_duo_channel(message)

    def _should_accept_partner_bot_message(self, message: discord.Message) -> bool:
        return self.config.duo_mode_enabled and self._is_duo_channel(message) and self._is_duo_partner(message)

    def _is_duo_channel(self, message: discord.Message) -> bool:
        if not self.config.duo_mode_enabled or not self.config.duo_channel_id:
            return False
        return str(message.channel.id) == self.config.duo_channel_id

    def _is_duo_partner(self, message: discord.Message) -> bool:
        if not self.config.duo_partner_bot_id:
            return False
        return str(message.author.id) == self.config.duo_partner_bot_id

    def _duo_session_user_id(self, message: discord.Message) -> str:
        channel_id = self._duo_session.channel_id or message.channel.id
        partner_id = self._duo_session.partner_bot_id or self.config.duo_partner_bot_id or "partner"
        return f"duo:{channel_id}:{partner_id}:{self._duo_session.session_token}"

    def _reset_duo_session(self, reason: str) -> None:
        self._duo_session.active = False
        self._duo_session.session_token += 1
        self._duo_session.channel_id = None
        self._duo_session.partner_bot_id = None
        self._duo_session.prompt = None
        self._duo_session.turns_sent = 0
        self._duo_session.last_partner_signature = None
        self._duo_session.last_my_reply_signature = None
        self._duo_session.stopped_reason = reason

    def _stop_duo_session(self, reason: str) -> bool:
        if not self._duo_session.active:
            self._duo_session.stopped_reason = reason
            return False
        self._reset_duo_session(reason)
        return True

    def _begin_duo_session(
        self,
        *,
        channel_id: int,
        partner_bot_id: str | None,
        prompt: str | None,
        reason: str,
    ) -> None:
        self._reset_duo_session(reason)
        self._duo_session.active = True
        self._duo_session.channel_id = channel_id
        self._duo_session.partner_bot_id = partner_bot_id
        self._duo_session.prompt = prompt
        self._duo_session.turns_sent = 0
        self._duo_session.stopped_reason = None

    def _format_duo_status(self) -> str:
        session = self._duo_session
        return (
            "duo status\n"
            f"- enabled: {self.config.duo_mode_enabled}\n"
            f"- active: {session.active}\n"
            f"- channel: {session.channel_id}\n"
            f"- partner_bot_id: {session.partner_bot_id or self.config.duo_partner_bot_id}\n"
            f"- turns_sent: {session.turns_sent}\n"
            f"- max_turns_per_bot: {self.config.duo_max_turns_per_bot}\n"
            f"- autostart_enabled: {self.config.duo_autostart_enabled}\n"
            f"- autostart_channel_id: {self.config.duo_autostart_channel_id}\n"
            f"- stopped_reason: {session.stopped_reason}"
        )

    async def _send_private_notice(self, message: discord.Message, text: str) -> None:
        try:
            await message.author.send(text)
        except Exception:
            await message.channel.send(text)

    async def _send_text_with_optional_tts(
        self,
        channel: discord.abc.Messageable,
        text: str,
        *,
        speaker: str,
        voice_member: discord.abc.User | discord.Member | None = None,
        performance_packet: VTuberTurnPacket | None = None,
    ) -> None:
        await channel.send(text)
        result = await self._maybe_dispatch_tts(
            text=text,
            speaker=speaker,
            performance_packet=performance_packet,
        )
        if result is None or result.artifact is None:
            return
        if result.mode == "discord_voice":
            if voice_member is None:
                return
            await self._maybe_play_tts_in_voice(artifact=result.artifact, voice_member=voice_member)
            return
        if result.mode != "discord_file":
            return
        await channel.send(file=discord.File(str(result.artifact.path), filename=result.artifact.path.name))

    async def _maybe_dispatch_tts(
        self,
        *,
        text: str,
        speaker: str,
        performance_packet: VTuberTurnPacket | None = None,
    ):
        runtime = getattr(self, "speech_runtime", None)
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
            print("[tts] synthesis failed")
            traceback.print_exc()
            return None

    async def _maybe_play_tts_in_voice(self, *, artifact, voice_member: discord.abc.User | discord.Member) -> None:
        try:
            await play_artifact_to_member_voice(
                client=self,
                member=voice_member,
                artifact=artifact,
                ffmpeg_executable=self.config.tts_ffmpeg_executable,
            )
        except Exception:
            print("[tts] discord voice playback failed")
            traceback.print_exc()

    @staticmethod
    def _normalize_duo_signature(text: str) -> str:
        return " ".join(text.casefold().split())

    async def _handle_runtime_command(self, message: discord.Message, command: RuntimeCommand) -> None:
        if command.kind == "status":
            if not self.runtime_presence.enabled:
                await message.channel.send("공유 runtime 상태 기능이 꺼져 있어.")
                return
            await message.channel.send(self.runtime_presence.format_status_report())
            return

        if command.kind == "claim":
            if not self.runtime_presence.enabled:
                await message.channel.send("공유 runtime 상태 기능이 꺼져 있어.")
                return
            result = self.runtime_presence.claim_manual(
                owner_id=message.author.id,
                owner_name=getattr(message.author, "display_name", message.author.name),
                channel_id=message.channel.id,
                guild_id=message.guild.id if message.guild else None,
                reason="manual_claim",
            )
            await message.channel.send(
                _format_runtime_claim_result(
                    self.runtime_presence.bot_name,
                    result,
                    manual_ttl_seconds=self.runtime_presence.manual_claim_ttl_seconds,
                )
            )
            return

        if command.kind == "release":
            if not self.runtime_presence.enabled:
                await message.channel.send("공유 runtime 상태 기능이 꺼져 있어.")
                return
            released = self.runtime_presence.release(owner_id=message.author.id)
            reply = (
                f"`{self.runtime_presence.bot_name}` 점유를 해제했어."
                if released
                else f"`{self.runtime_presence.bot_name}` 점유를 해제하지 못했어."
            )
            await message.channel.send(reply)
            return

        user_id = str(message.author.id)
        state = self.engine.state_store.get_or_create(user_id)
        latest_trace = self.engine.state_store.get_latest_decision_trace(user_id)
        if command.kind == "dashboard":
            await self._send_private_notice(
                message,
                format_black_dashboard(
                    config=self.config,
                    runtime_report=self.runtime_presence.format_status_report()
                    if self.runtime_presence.enabled
                    else "runtime disabled",
                    state=state,
                    latest_trace=latest_trace,
                ),
            )
            return
        if command.kind == "summary":
            await self._send_private_notice(
                message,
                format_black_summary(state=state, latest_trace=latest_trace),
            )
            return
        if command.kind == "state":
            await self._send_private_notice(
                message,
                format_black_state(state=state, latest_trace=latest_trace),
            )
            return
        if command.kind == "recall":
            query = (command.argument or "").strip()
            if not query:
                await self._send_private_notice(message, "사용법: `!predict recall <키워드>`")
                return
            await self._send_private_notice(
                message,
                format_black_recall(state=state, query=query),
            )

    def _note_runtime_activity(self, message: discord.Message, *, reason: str) -> None:
        result = self.runtime_presence.note_auto_activity(
            owner_id=message.author.id,
            owner_name=getattr(message.author, "display_name", message.author.name),
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            reason=reason,
        )
        if result is not None and result.conflict:
            print(
                "[runtime] "
                f"auto activity skipped for {self.runtime_presence.bot_name} "
                f"holder={result.holder_name!r} channel={result.holder_channel_id!r}"
            )

    async def _runtime_heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.runtime_presence.heartbeat_interval_seconds)
                try:
                    self.runtime_presence.heartbeat()
                    if not self._runtime_heartbeat_observed:
                        self._runtime_heartbeat_observed = True
                        print(
                            f"[runtime] heartbeat observed bot={self.runtime_presence.bot_name!r} "
                            f"interval={self.runtime_presence.heartbeat_interval_seconds}s"
                        )
                except Exception:
                    print("[runtime] heartbeat update failed.")
                    traceback.print_exc()
        except asyncio.CancelledError:
            raise


async def run_discord_bot(config: AppConfig, engine: PredictiveEngine, speech_runtime=None) -> None:
    client = PredictiveDiscordClient(config=config, engine=engine, speech_runtime=speech_runtime)
    try:
        await client.start(config.discord_bot_token)
    finally:
        await client.close()


def _format_runtime_claim_result(
    bot_name: str,
    result: ClaimResult | None,
    *,
    manual_ttl_seconds: int,
) -> str:
    if result is None:
        return "공유 runtime 상태 기능이 꺼져 있어."
    if result.acquired:
        ttl_minutes = max(1, manual_ttl_seconds // 60)
        return (
            f"`{bot_name}` 점유를 등록했어.\n"
            f"- mode: manual\n"
            f"- lease: {ttl_minutes}분"
        )
    if result.conflict:
        holder_name = result.holder_name or "다른 사용자"
        holder_channel = result.holder_channel_id or "unknown"
        holder_reason = result.holder_reason or "manual_claim"
        return (
            f"`{bot_name}`은 이미 사용 중이야.\n"
            f"- holder: {holder_name}\n"
            f"- channel: {holder_channel}\n"
            f"- reason: {holder_reason}"
        )
    return f"`{bot_name}` 점유 등록에 실패했어."
