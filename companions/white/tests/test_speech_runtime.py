from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DISCODEBOT_SRC = WORKSPACE_ROOT / "discodebot" / "src"
for candidate in (WORKSPACE_ROOT, DISCODEBOT_SRC):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.append(candidate_text)

from bot_shared.speech import Utterance, VoiceProfile, build_speech_runtime
from bot_shared.speech.models import SpeechArtifact
from bot_shared.speech.runtime import (
    AudioPlayback,
    SpeechRuntime,
    normalize_for_speech,
    prepare_text_for_speech,
    rewrite_for_spoken_delivery,
)
from bot_shared.vtuber import VTuberActionCue, VTuberTurnPacket


class _FakeProvider:
    provider_name = "fake"

    async def synthesize(self, *, utterance: Utterance, profile: VoiceProfile, output_path: Path) -> SpeechArtifact | None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(utterance.text, encoding="utf-8")
        return SpeechArtifact(
            path=output_path,
            mime_type="audio/ogg",
            provider=self.provider_name,
            voice_profile=profile.name,
            text=utterance.text,
        )


class _FakePlayback:
    def __init__(self, *, sleep_seconds: float) -> None:
        self._stopped = False
        self._task = asyncio.create_task(self._run(sleep_seconds))

    async def _run(self, sleep_seconds: float) -> int:
        try:
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            return -1
        return 0

    async def wait(self) -> int | None:
        return await self._task

    async def stop(self) -> None:
        self._stopped = True
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    @property
    def is_playing(self) -> bool:
        return (not self._stopped) and (not self._task.done())


class _FakePlayer:
    player_name = "fake-player"

    def __init__(self, *, sleep_seconds: float = 0.2) -> None:
        self.sleep_seconds = sleep_seconds
        self.played_texts: list[str] = []

    async def play(self, *, artifact: SpeechArtifact, utterance: Utterance, profile: VoiceProfile) -> AudioPlayback:
        self.played_texts.append(utterance.text)
        return _FakePlayback(sleep_seconds=self.sleep_seconds)


class SpeechRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_noop_runtime_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_speech_runtime(
                enabled=True,
                mode="discord_file",
                provider_name="noop",
                output_dir=tmp,
                command_template="",
                play_command_template="",
                audio_format="ogg",
                max_chars=120,
                profiles={
                    "white": VoiceProfile(name="white", voice_id="white_default"),
                },
            )

            artifact = await runtime.synthesize(
                utterance=Utterance(
                    speaker="white",
                    text="안녕, 오늘은 조금 차분하게 말해볼게.",
                )
            )

            self.assertIsNone(artifact)

    def test_normalize_for_speech_strips_markup(self) -> None:
        normalized = normalize_for_speech("  **안녕**  <@12345>  `테스트`  ", max_chars=80)
        self.assertEqual(normalized, "안녕 테스트")

    def test_normalize_for_speech_truncates(self) -> None:
        normalized = normalize_for_speech(
            "이 문장은 음성 출력용으로 너무 길어서 적당한 길이에서 잘려야 해.",
            max_chars=18,
        )
        self.assertTrue(normalized.endswith("..."))
        self.assertLessEqual(len(normalized), 21)

    def test_rewrite_for_spoken_delivery_shortens_white_analysis_tail(self) -> None:
        rewritten = rewrite_for_spoken_delivery(
            Utterance(
                speaker="white",
                text="그 말은 괜찮은 거야. 오늘은 조금 느린 리듬이 필요하다는 뜻이기도 하니까.",
            )
        )
        self.assertIn("그 말은 괜찮아", rewritten)
        self.assertNotIn("뜻이기도", rewritten)

    def test_rewrite_for_spoken_delivery_cleans_black_tail_and_typos(self) -> None:
        rewritten = rewrite_for_spoken_delivery(
            Utterance(
                speaker="black",
                text="말이 많지어도 같이 있는 기분은 더렷해질 수 있지. 그런 날이 있더라.",
            )
        )
        self.assertIn("많지 않아도", rewritten)
        self.assertIn("또렷", rewritten)
        self.assertIn("그런 날도 있어", rewritten)

    def test_prepare_text_for_speech_collapses_similar_sentences(self) -> None:
        prepared = prepare_text_for_speech(
            utterance=Utterance(
                speaker="black",
                text="같이 있다가 돌아왔는데 말이 더 고프더라. 같이 있다가 돌아오면 말이 더 쉬워질 때가 있어.",
            ),
            max_chars=120,
        )
        self.assertEqual(prepared.count("같이"), 1)

    def test_command_provider_requires_template(self) -> None:
        with self.assertRaises(ValueError):
            build_speech_runtime(
                enabled=True,
                mode="discord_file",
                provider_name="command",
                output_dir=".",
                command_template="",
                play_command_template="",
                audio_format="ogg",
                max_chars=120,
                profiles={
                    "white": VoiceProfile(name="white", voice_id="white_default"),
                },
            )

    def test_elevenlabs_provider_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            build_speech_runtime(
                enabled=True,
                mode="discord_file",
                provider_name="elevenlabs",
                output_dir=".",
                command_template="",
                play_command_template="",
                audio_format="mp3_44100_128",
                max_chars=120,
                profiles={
                    "white": VoiceProfile(name="white", voice_id="voice-id"),
                },
                elevenlabs_api_key="",
            )

    def test_xtts_http_provider_requires_server_url(self) -> None:
        with self.assertRaises(ValueError):
            build_speech_runtime(
                enabled=True,
                mode="discord_file",
                provider_name="xtts_http",
                output_dir=".",
                command_template="",
                play_command_template="",
                audio_format="wav",
                max_chars=120,
                profiles={
                    "white": VoiceProfile(name="white", voice_id="voice-id"),
                },
                xtts_server_url="",
            )

    def test_gptsovits_http_provider_requires_server_url(self) -> None:
        with self.assertRaises(ValueError):
            build_speech_runtime(
                enabled=True,
                mode="discord_file",
                provider_name="gptsovits_http",
                output_dir=".",
                command_template="",
                play_command_template="",
                audio_format="wav",
                max_chars=120,
                profiles={
                    "white": VoiceProfile(name="white", voice_id="voice-id"),
                },
                gptsovits_server_url="",
            )

    def test_local_live_requires_playback_template(self) -> None:
        with mock.patch("bot_shared.speech.runtime.shutil.which", return_value=None):
            with self.assertRaises(ValueError):
                build_speech_runtime(
                    enabled=True,
                    mode="local_live",
                    provider_name="command",
                    output_dir=".",
                    command_template="echo synth",
                    play_command_template="",
                    audio_format="ogg",
                    max_chars=120,
                    profiles={
                        "white": VoiceProfile(name="white", voice_id="white_default"),
                    },
                )

    def test_local_live_auto_detects_ffplay(self) -> None:
        def _fake_which(binary: str) -> str | None:
            return "/usr/bin/ffplay" if binary == "ffplay" else None

        with mock.patch("bot_shared.speech.runtime.shutil.which", side_effect=_fake_which):
            runtime = build_speech_runtime(
                enabled=True,
                mode="local_live",
                provider_name="command",
                output_dir=".",
                command_template="echo synth",
                play_command_template="",
                audio_format="ogg",
                max_chars=120,
                profiles={
                    "white": VoiceProfile(name="white", voice_id="white_default"),
                },
            )
        self.assertEqual(runtime.player.player_name, "ffplay")

    def test_local_live_honors_preferred_player(self) -> None:
        def _fake_which(binary: str) -> str | None:
            return "/usr/bin/mpv" if binary == "mpv" else None

        with mock.patch("bot_shared.speech.runtime.shutil.which", side_effect=_fake_which):
            runtime = build_speech_runtime(
                enabled=True,
                mode="local_live",
                provider_name="command",
                output_dir=".",
                command_template="echo synth",
                play_command_template="",
                audio_format="ogg",
                max_chars=120,
                local_player_name="mpv",
                profiles={
                    "white": VoiceProfile(name="white", voice_id="white_default"),
                },
            )
        self.assertEqual(runtime.player.player_name, "mpv")

    def test_local_live_rejects_unknown_preferred_player(self) -> None:
        with self.assertRaises(ValueError):
            build_speech_runtime(
                enabled=True,
                mode="local_live",
                provider_name="command",
                output_dir=".",
                command_template="echo synth",
                play_command_template="",
                audio_format="ogg",
                max_chars=120,
                local_player_name="unknown-player",
                profiles={
                    "white": VoiceProfile(name="white", voice_id="white_default"),
                },
            )

    async def test_local_live_interrupts_lower_priority_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            player = _FakePlayer(sleep_seconds=0.2)
            runtime = SpeechRuntime(
                enabled=True,
                mode="local_live",
                provider=_FakeProvider(),
                output_dir=Path(tmp),
                profiles={"white": VoiceProfile(name="white", voice_id="white_default")},
                max_chars=120,
                player=player,
            )
            try:
                first = await runtime.speak(
                    utterance=Utterance(
                        speaker="white",
                        text="첫 번째 말",
                        priority="normal",
                        can_interrupt=True,
                    )
                )
                await asyncio.sleep(0.05)
                second = await runtime.speak(
                    utterance=Utterance(
                        speaker="white",
                        text="두 번째 말",
                        priority="high",
                        can_interrupt=True,
                    )
                )

                await second.wait()
                await asyncio.sleep(0.05)

                self.assertTrue(first.interrupted)
                self.assertTrue(second.started)
                self.assertEqual(player.played_texts[-1], "두 번째 말")
            finally:
                await runtime.close()

    async def test_dispatch_returns_artifact_for_discord_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = SpeechRuntime(
                enabled=True,
                mode="discord_file",
                provider=_FakeProvider(),
                output_dir=Path(tmp),
                profiles={"white": VoiceProfile(name="white", voice_id="white_default")},
                max_chars=120,
            )
            result = await runtime.dispatch(
                utterance=Utterance(
                    speaker="white",
                    text="파일 첨부용 발화",
                )
            )

            self.assertEqual(result.mode, "discord_file")
            self.assertIsNotNone(result.artifact)
            self.assertIsNone(result.handle)

    async def test_dispatch_returns_handle_for_local_live_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            player = _FakePlayer(sleep_seconds=0.01)
            runtime = SpeechRuntime(
                enabled=True,
                mode="local_live",
                provider=_FakeProvider(),
                output_dir=Path(tmp),
                profiles={"white": VoiceProfile(name="white", voice_id="white_default")},
                max_chars=120,
                player=player,
            )
            try:
                result = await runtime.dispatch(
                    utterance=Utterance(
                        speaker="white",
                        text="로컬 재생용 발화",
                    )
                )
                self.assertEqual(result.mode, "local_live")
                self.assertIsNone(result.artifact)
                self.assertIsNotNone(result.handle)
                await result.handle.wait()
                self.assertEqual(player.played_texts[-1], "로컬 재생용 발화")
            finally:
                await runtime.close()

    async def test_dispatch_writes_obs_live_cue_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tts"
            obs_dir = Path(tmp) / "obs"
            runtime = SpeechRuntime(
                enabled=True,
                mode="obs_live",
                provider=_FakeProvider(),
                output_dir=output_dir,
                obs_output_dir=obs_dir,
                profiles={"white": VoiceProfile(name="white", voice_id="white_default", style="soft")},
                max_chars=120,
            )
            result = await runtime.dispatch(
                utterance=Utterance(
                    speaker="white",
                    text="OBS로 보낼 발화",
                    mood="calm",
                    intent="reply",
                )
            )

            self.assertEqual(result.mode, "obs_live")
            self.assertIsNotNone(result.artifact)
            self.assertIsNotNone(result.cue)
            self.assertIsNotNone(result.cue_manifest_path)
            self.assertTrue(result.cue_manifest_path.exists())
            payload = json.loads(result.cue_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["speaker"], "white")
            self.assertEqual(payload["expression"], "soft")
            self.assertEqual(payload["audio_path"], str(result.artifact.path))
            self.assertEqual(result.cue_manifest_path.parent.name, "white")
            self.assertGreater(payload["duration_sec"], 0)
            self.assertEqual(payload["metadata"]["mouth_mode"], "viseme")
            self.assertTrue(payload["visemes"])
            self.assertLessEqual(payload["visemes"][-1]["end_ms"], round(payload["duration_sec"] * 1000))

    async def test_dispatch_writes_overlay_cue_manifest_for_discord_voice_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tts"
            obs_dir = Path(tmp) / "obs"
            runtime = SpeechRuntime(
                enabled=True,
                mode="discord_voice",
                provider=_FakeProvider(),
                output_dir=output_dir,
                obs_output_dir=obs_dir,
                profiles={"black": VoiceProfile(name="black", voice_id="black_default", style="dry")},
                max_chars=120,
            )
            result = await runtime.dispatch(
                utterance=Utterance(
                    speaker="black",
                    text="디스코드 음성 재생과 함께 오버레이 cue도 남겨야 해.",
                    mood="neutral",
                    intent="reply",
                )
            )

            self.assertEqual(result.mode, "discord_voice")
            self.assertIsNotNone(result.artifact)
            self.assertIsNotNone(result.cue)
            self.assertIsNotNone(result.cue_manifest_path)
            self.assertTrue(result.cue_manifest_path.exists())
            latest_path = obs_dir / "latest" / "black.json"
            self.assertTrue(latest_path.exists())
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["speaker"], "black")
            self.assertEqual(payload["audio_path"], str(result.artifact.path))
            self.assertTrue(payload["visemes"])

    async def test_dispatch_preserves_vtuber_action_cues_for_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "tts"
            obs_dir = Path(tmp) / "obs"
            runtime = SpeechRuntime(
                enabled=True,
                mode="obs_live",
                provider=_FakeProvider(),
                output_dir=output_dir,
                obs_output_dir=obs_dir,
                profiles={"black": VoiceProfile(name="black", voice_id="black_default", style="clear")},
                max_chars=120,
            )
            packet = VTuberTurnPacket(
                speaker="black",
                text="근거 확인하고 말하는 중이야.",
                brain="black.predictive_policy",
                emotion_state="grounded",
                action_intent="weather_lookup",
                facial_expression="focused",
                voice_style="black_grounded",
                action_cues=[
                    VTuberActionCue(kind="speak", intensity=1.0),
                    VTuberActionCue(kind="posture:focused", intensity=0.72),
                    VTuberActionCue(kind="gesture:small_nod", intensity=0.42),
                ],
            )

            result = await runtime.dispatch(utterance=packet.to_utterance())

            self.assertIsNotNone(result.cue_manifest_path)
            payload = json.loads(result.cue_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["expression"], "focused")
            self.assertEqual(payload["mood"], "grounded")
            self.assertEqual(payload["intent"], "weather_lookup")
            self.assertEqual(payload["metadata"]["voice_style"], "black_grounded")
            self.assertEqual(
                [item["kind"] for item in payload["action_cues"]],
                ["speak", "posture:focused", "gesture:small_nod"],
            )
            self.assertTrue(payload["visemes"])

    async def test_command_provider_removes_temporary_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = build_speech_runtime(
                enabled=True,
                mode="discord_file",
                provider_name="command",
                output_dir=tmp,
                command_template="cp {text_path} {output_path}",
                play_command_template="",
                audio_format="wav",
                max_chars=120,
                profiles={
                    "white": VoiceProfile(name="white", voice_id="white_default"),
                },
            )
            artifact = await runtime.synthesize(
                utterance=Utterance(
                    speaker="white",
                    text="임시 텍스트 파일은 남지 않아야 해.",
                )
            )

            self.assertIsNotNone(artifact)
            leftovers = list(Path(tmp).glob("tts-command-*.txt"))
            self.assertEqual(leftovers, [])

    async def test_synthesize_applies_spoken_rewrite_before_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = SpeechRuntime(
                enabled=True,
                mode="discord_file",
                provider=_FakeProvider(),
                output_dir=Path(tmp),
                profiles={"white": VoiceProfile(name="white", voice_id="white_default")},
                max_chars=120,
            )
            artifact = await runtime.synthesize(
                utterance=Utterance(
                    speaker="white",
                    text="그 말은 괜찮은 거야. 오늘은 조금 느린 리듬이 필요하다는 뜻이기도 하니까.",
                )
            )

            self.assertIsNotNone(artifact)
            payload = artifact.path.read_text(encoding="utf-8")
            self.assertIn("그 말은 괜찮아", payload)
            self.assertNotIn("뜻이기도", payload)
