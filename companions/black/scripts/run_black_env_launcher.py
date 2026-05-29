#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.append(candidate_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch black bot from an explicit env file.")
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--tts-enabled", default="")
    parser.add_argument("--tts-mode", default="")
    parser.add_argument("--tts-provider", default="")
    parser.add_argument("--tts-audio-format", default="")
    parser.add_argument("--tts-xtts-server-url", default="")
    parser.add_argument("--tts-xtts-language", default="")
    parser.add_argument("--tts-gptsovits-server-url", default="")
    parser.add_argument("--tts-gptsovits-server-token", default="")
    parser.add_argument("--tts-gptsovits-language", default="")
    parser.add_argument("--tts-ffmpeg-executable", default="")
    parser.add_argument("--tts-black-voice-id", default="")
    return parser.parse_args()


WINDOWS_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def _looks_like_windows_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    return WINDOWS_PATH_RE.match(value) is not None


def _windows_to_wsl_path(value: str) -> str:
    match = WINDOWS_PATH_RE.match(value)
    if not match:
        return value
    drive = match.group("drive").lower()
    rest = match.group("rest").replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _normalize_env_value(name: str, value: str) -> str:
    normalized_name = name.upper()
    is_path_like = (
        normalized_name.endswith(("_PATH", "_DIR", "_FILE", "_SCRIPT", "_PYTHON", "_EXECUTABLE"))
        or normalized_name in {
            "KOBART_MODEL_NAME_OR_PATH",
            "TTS_WHITE_VOICE_ID",
            "TTS_BLACK_VOICE_ID",
        }
    )
    if is_path_like and _looks_like_windows_path(value):
        return _windows_to_wsl_path(value)
    return value


def _apply_override(name: str, value: str) -> None:
    if value.strip():
        os.environ[name] = _normalize_env_value(name, value.strip())


def _load_env_file(path: str) -> None:
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        os.environ[key] = _normalize_env_value(key, value)


def main() -> None:
    args = parse_args()
    _load_env_file(args.env_file)

    _apply_override("TTS_ENABLED", args.tts_enabled)
    _apply_override("TTS_MODE", args.tts_mode)
    _apply_override("TTS_PROVIDER", args.tts_provider)
    _apply_override("TTS_AUDIO_FORMAT", args.tts_audio_format)
    _apply_override("TTS_XTTS_SERVER_URL", args.tts_xtts_server_url)
    _apply_override("TTS_XTTS_LANGUAGE", args.tts_xtts_language)
    _apply_override("TTS_GPTSOVITS_SERVER_URL", args.tts_gptsovits_server_url)
    _apply_override("TTS_GPTSOVITS_SERVER_TOKEN", args.tts_gptsovits_server_token)
    _apply_override("TTS_GPTSOVITS_LANGUAGE", args.tts_gptsovits_language)
    _apply_override("TTS_FFMPEG_EXECUTABLE", args.tts_ffmpeg_executable)
    _apply_override("TTS_BLACK_VOICE_ID", args.tts_black_voice_id)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONPATH", str(SRC_ROOT))
    sys.argv = [sys.argv[0]]

    from predictive_bot.main import main as run_main

    run_main()


if __name__ == "__main__":
    main()
