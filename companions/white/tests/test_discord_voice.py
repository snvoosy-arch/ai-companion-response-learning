from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from bot_shared.speech.discord_voice import (
    disconnect_all_client_voice_connections,
    play_artifact_to_member_voice,
)
from bot_shared.speech.models import SpeechArtifact


class _FakeFFmpegOpusAudio:
    def __init__(self, path: str, *, executable: str = "ffmpeg") -> None:
        self.path = path
        self.executable = executable


class _FakeVoiceClient:
    def __init__(self, *, channel=None, connected: bool = True) -> None:
        self.channel = channel
        self._connected = connected
        self.played_source = None
        self.stopped = False
        self.disconnected = False

    def is_playing(self) -> bool:
        return self.played_source is not None and not self.stopped

    def stop(self) -> None:
        self.stopped = True

    def play(self, source) -> None:
        self.played_source = source
        self.stopped = False

    def is_connected(self) -> bool:
        return self._connected

    async def disconnect(self, *, force: bool = False) -> None:
        self.disconnected = force
        self._connected = False

    async def move_to(self, channel) -> None:
        self.channel = channel


class _FakeVoiceChannel:
    def __init__(self, guild) -> None:
        self.guild = guild
        self.connected_client = None

    async def connect(self, *, self_deaf: bool = True):
        client = _FakeVoiceClient(channel=self)
        self.guild.voice_client = client
        self.connected_client = client
        return client


class _FakeGuild:
    def __init__(self) -> None:
        self.voice_client = None


class _FakeVoiceState:
    def __init__(self, channel) -> None:
        self.channel = channel


class _FakeMember:
    def __init__(self, channel) -> None:
        self.voice = _FakeVoiceState(channel)


class _FakeClient:
    def __init__(self, voice_clients) -> None:
        self.voice_clients = voice_clients


class DiscordVoiceHelperTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        fake_discord = types.SimpleNamespace(FFmpegOpusAudio=_FakeFFmpegOpusAudio)
        self._patcher = mock.patch.dict(sys.modules, {"discord": fake_discord})
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()

    async def test_play_artifact_connects_and_plays(self) -> None:
        guild = _FakeGuild()
        channel = _FakeVoiceChannel(guild)
        member = _FakeMember(channel)
        artifact = SpeechArtifact(
            path=Path("/tmp/fake.mp3"),
            mime_type="audio/mpeg",
            provider="elevenlabs",
            voice_profile="white",
            text="테스트",
        )

        played = await play_artifact_to_member_voice(
            client=object(),
            member=member,
            artifact=artifact,
            ffmpeg_executable="ffmpeg-custom",
        )

        self.assertTrue(played)
        self.assertIsNotNone(guild.voice_client)
        self.assertEqual(guild.voice_client.played_source.path, str(artifact.path))
        self.assertEqual(guild.voice_client.played_source.executable, "ffmpeg-custom")

    async def test_play_artifact_converts_wsl_path_for_windows_ffmpeg(self) -> None:
        guild = _FakeGuild()
        channel = _FakeVoiceChannel(guild)
        member = _FakeMember(channel)
        artifact = SpeechArtifact(
            path=Path("~/.bot-runtime/tts/white/test.wav"),
            mime_type="audio/wav",
            provider="command",
            voice_profile="white",
            text="테스트",
        )

        played = await play_artifact_to_member_voice(
            client=object(),
            member=member,
            artifact=artifact,
            ffmpeg_executable="/mnt/e/tools/ffmpeg.exe",
        )

        self.assertTrue(played)
        self.assertEqual(guild.voice_client.played_source.path, "E:\\bot\\runtime\\tts\\white\\test.wav")
        self.assertEqual(guild.voice_client.played_source.executable, "/mnt/e/tools/ffmpeg.exe")

    async def test_disconnect_all_client_voice_connections_disconnects_everything(self) -> None:
        voice_client_a = _FakeVoiceClient()
        voice_client_b = _FakeVoiceClient()
        client = _FakeClient([voice_client_a, voice_client_b])

        await disconnect_all_client_voice_connections(client)

        self.assertTrue(voice_client_a.disconnected)
        self.assertTrue(voice_client_b.disconnected)
