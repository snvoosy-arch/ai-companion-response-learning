from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PRIVATE_RUNTIME_ROOT = Path.home() / ".bot-runtime"
WHITE_PRIVATE_RUNTIME_ROOT = PRIVATE_RUNTIME_ROOT / "white"
DEFAULT_MEMORY_DB_PATH = WHITE_PRIVATE_RUNTIME_ROOT / "discord_memory.sqlite3"
DEFAULT_STARTUP_SINGLETON_LOCK_PATH = WHITE_PRIVATE_RUNTIME_ROOT / "white_startup.lock"
DEFAULT_TTS_OUTPUT_DIR = WHITE_PRIVATE_RUNTIME_ROOT / "tts"
DEFAULT_TTS_OBS_OUTPUT_DIR = WHITE_PRIVATE_RUNTIME_ROOT / "obs-live"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from bot_shared.runtime_paths import resolve_default_shared_runtime_state_db_path
from bot_shared.speech import resolve_gptsovits_command_template, resolve_xtts_command_template

DEFAULT_RUNTIME_STATE_DB_PATH = resolve_default_shared_runtime_state_db_path()


@dataclass(slots=True)
class Settings:
    discord_token: str
    lm_studio_model: str
    discord_guild_id: int | None
    lm_studio_base_url: str
    lm_studio_api_key: str
    system_prompt: str
    disable_thinking: bool
    chat_reply_enabled: bool
    reply_when_mentioned: bool
    chat_channel_ids: tuple[int, ...]
    chat_history_messages: int
    memory_db_path: str
    memory_recent_messages: int
    memory_summary_trigger_messages: int
    memory_summary_batch_messages: int
    temperature: float
    max_output_tokens: int
    request_timeout_seconds: float
    output_guard_trace_path: str
    output_guard_no_canned_fallback: bool
    web_search_mode: str
    tavily_api_key: str
    tavily_max_results: int
    tavily_search_depth: str
    tavily_country: str
    tavily_news_time_range: str
    duo_enabled: bool
    duo_partner_bot_id: int | None
    duo_channel_ids: tuple[int, ...]
    duo_max_replies: int
    duo_reply_delay_seconds: float
    runtime_state_enabled: bool
    runtime_state_db_path: str
    runtime_bot_name: str
    runtime_heartbeat_seconds: float
    runtime_online_timeout_seconds: float
    runtime_auto_claim_ttl_seconds: int
    runtime_manual_claim_ttl_seconds: int
    startup_singleton_enabled: bool
    startup_singleton_lock_path: str
    tts_enabled: bool
    tts_mode: str
    tts_provider: str
    tts_command_template: str
    tts_play_command_template: str
    tts_local_player: str
    tts_ffmpeg_executable: str
    tts_output_dir: str
    tts_obs_output_dir: str
    tts_audio_format: str
    tts_max_chars: int
    tts_elevenlabs_api_key: str
    tts_elevenlabs_model_id: str
    tts_elevenlabs_base_url: str
    tts_elevenlabs_request_timeout_seconds: float
    tts_white_voice_id: str
    tts_black_voice_id: str
    tts_white_speed: float
    tts_black_speed: float
    tts_white_style: str
    tts_black_style: str
    tts_xtts_server_url: str
    tts_xtts_server_token: str
    tts_xtts_client_python: str
    tts_xtts_client_script: str
    tts_xtts_language: str
    tts_gptsovits_server_url: str
    tts_gptsovits_server_token: str
    tts_gptsovits_client_python: str
    tts_gptsovits_client_script: str
    tts_gptsovits_language: str


def _read_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"필수 환경변수가 비어 있어요: {name}")
    return value


def _read_alias(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _read_required_alias(names: tuple[str, ...]) -> str:
    value = _read_alias(names)
    if not value:
        joined = ", ".join(names)
        raise ValueError(f"필수 환경변수가 비어 있어요: {joined}")
    return value


def _read_optional_int(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None
    return int(raw_value)


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}의 불리언 값이 올바르지 않아요: {raw_value}")


def _read_choice(name: str, default: str, choices: set[str]) -> str:
    raw_value = os.getenv(name, default).strip().lower()
    if raw_value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} 값이 올바르지 않아요: {raw_value}. 가능한 값: {allowed}")
    return raw_value


def _read_int_list(name: str) -> tuple[int, ...]:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return ()

    values: list[int] = []
    for part in raw_value.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    return tuple(values)


def _expand_runtime_path(raw: str) -> str:
    return str(Path(os.path.expandvars(raw)).expanduser())


def load_settings() -> Settings:
    env_path = os.getenv("DISCORD_BOT_ENV_FILE", "").strip()
    if env_path:
        load_dotenv(env_path, override=False)
    else:
        load_dotenv()

    chat_history_messages = int(os.getenv("CHAT_HISTORY_MESSAGES", "8"))
    memory_recent_messages = int(os.getenv("MEMORY_RECENT_MESSAGES", str(chat_history_messages * 2)))
    memory_summary_trigger_messages = int(
        os.getenv("MEMORY_SUMMARY_TRIGGER_MESSAGES", str(max(memory_recent_messages + 8, 24)))
    )
    memory_summary_batch_messages = int(os.getenv("MEMORY_SUMMARY_BATCH_MESSAGES", "20"))
    tts_provider = os.getenv("TTS_PROVIDER", "noop").strip().lower()
    tts_xtts_server_url = os.getenv("TTS_XTTS_SERVER_URL", "").strip()
    tts_xtts_server_token = os.getenv("TTS_XTTS_SERVER_TOKEN", "").strip()
    tts_xtts_client_python = os.getenv("TTS_XTTS_CLIENT_PYTHON", "").strip()
    tts_xtts_client_script = os.getenv("TTS_XTTS_CLIENT_SCRIPT", "").strip()
    tts_xtts_language = os.getenv("TTS_XTTS_LANGUAGE", "ko").strip() or "ko"
    tts_gptsovits_server_url = os.getenv("TTS_GPTSOVITS_SERVER_URL", "").strip()
    tts_gptsovits_server_token = os.getenv("TTS_GPTSOVITS_SERVER_TOKEN", "").strip()
    tts_gptsovits_client_python = os.getenv("TTS_GPTSOVITS_CLIENT_PYTHON", "").strip()
    tts_gptsovits_client_script = os.getenv("TTS_GPTSOVITS_CLIENT_SCRIPT", "").strip()
    tts_gptsovits_speaker_manifest = os.getenv("TTS_GPTSOVITS_SPEAKER_MANIFEST", "").strip()
    tts_gptsovits_language = os.getenv("TTS_GPTSOVITS_LANGUAGE", "ko").strip() or "ko"
    tts_command_template = resolve_gptsovits_command_template(
        explicit_template=os.getenv("GPTSOVITS_SYNTH_COMMAND_TEMPLATE", "").strip()
        or os.getenv("TTS_COMMAND_TEMPLATE", "").strip(),
        server_url=tts_gptsovits_server_url,
        workspace_root=WORKSPACE_ROOT,
        client_python=tts_gptsovits_client_python,
        client_script=tts_gptsovits_client_script,
        speaker_manifest=tts_gptsovits_speaker_manifest,
        language=tts_gptsovits_language,
        server_token=tts_gptsovits_server_token,
    )
    if not tts_command_template:
        tts_command_template = resolve_xtts_command_template(
            explicit_template=os.getenv("TTS_COMMAND_TEMPLATE", "").strip(),
            server_url=tts_xtts_server_url,
            workspace_root=WORKSPACE_ROOT,
            client_python=tts_xtts_client_python,
            client_script=tts_xtts_client_script,
            language=tts_xtts_language,
            server_token=tts_xtts_server_token,
        )

    settings = Settings(
        discord_token=_read_required("DISCORD_TOKEN"),
        lm_studio_model=_read_required_alias(("VLLM_MODEL", "LM_STUDIO_MODEL")),
        discord_guild_id=_read_optional_int("DISCORD_GUILD_ID"),
        lm_studio_base_url=_read_alias(("VLLM_BASE_URL", "LM_STUDIO_BASE_URL"), "http://127.0.0.1:8000/v1"),
        lm_studio_api_key=_read_required_alias(("VLLM_API_KEY", "LM_STUDIO_API_KEY")),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "너는 한국어 Discord AI 도우미다. 답변은 짧고 정확하게 하고, 모르면 모른다고 말해라.",
        ).strip(),
        disable_thinking=_read_bool("DISABLE_THINKING", default=True),
        chat_reply_enabled=_read_bool("CHAT_REPLY_ENABLED", default=True),
        reply_when_mentioned=_read_bool("REPLY_WHEN_MENTIONED", default=True),
        chat_channel_ids=_read_int_list("DISCORD_CHAT_CHANNEL_IDS"),
        chat_history_messages=chat_history_messages,
        memory_db_path=_expand_runtime_path(os.getenv("MEMORY_DB_PATH", str(DEFAULT_MEMORY_DB_PATH)).strip()),
        memory_recent_messages=memory_recent_messages,
        memory_summary_trigger_messages=memory_summary_trigger_messages,
        memory_summary_batch_messages=memory_summary_batch_messages,
        temperature=float(os.getenv("TEMPERATURE", "0.5")),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "160")),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        output_guard_trace_path=_expand_runtime_path(os.getenv("OUTPUT_GUARD_TRACE_PATH", "").strip())
        if os.getenv("OUTPUT_GUARD_TRACE_PATH", "").strip()
        else "",
        # Product policy: bad generations are observed/traced, not replaced by
        # canned replies. Keep the env var for compatibility, but do not let it
        # re-enable canned fallback behavior.
        output_guard_no_canned_fallback=True,
        web_search_mode=_read_choice("WEB_SEARCH_MODE", "off", {"off", "auto", "always", "heuristic"}),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        tavily_max_results=int(os.getenv("TAVILY_MAX_RESULTS", "5")),
        tavily_search_depth=_read_choice(
            "TAVILY_SEARCH_DEPTH",
            "basic",
            {"advanced", "basic", "fast", "ultra-fast"},
        ),
        tavily_country=os.getenv("TAVILY_COUNTRY", "south korea").strip().lower(),
        tavily_news_time_range=_read_choice(
            "TAVILY_NEWS_TIME_RANGE",
            "month",
            {"day", "week", "month", "year", "d", "w", "m", "y"},
        ),
        duo_enabled=_read_bool("DUO_ENABLED", default=False),
        duo_partner_bot_id=_read_optional_int("DUO_PARTNER_BOT_ID"),
        duo_channel_ids=_read_int_list("DUO_CHANNEL_IDS"),
        duo_max_replies=int(os.getenv("DUO_MAX_REPLIES", "8")),
        duo_reply_delay_seconds=float(os.getenv("DUO_REPLY_DELAY_SECONDS", "1.5")),
        runtime_state_enabled=_read_bool("BOT_RUNTIME_ENABLED", default=True),
        runtime_state_db_path=_expand_runtime_path(
            os.getenv("BOT_RUNTIME_STATE_DB_PATH", str(DEFAULT_RUNTIME_STATE_DB_PATH)).strip()
        ),
        runtime_bot_name=(os.getenv("BOT_RUNTIME_BOT_NAME", "white").strip().lower() or "white"),
        runtime_heartbeat_seconds=float(os.getenv("BOT_RUNTIME_HEARTBEAT_SECONDS", "15")),
        runtime_online_timeout_seconds=float(os.getenv("BOT_RUNTIME_ONLINE_TIMEOUT_SECONDS", "45")),
        runtime_auto_claim_ttl_seconds=int(os.getenv("BOT_RUNTIME_AUTO_CLAIM_TTL_SECONDS", "300")),
        runtime_manual_claim_ttl_seconds=int(os.getenv("BOT_RUNTIME_MANUAL_CLAIM_TTL_SECONDS", "1800")),
        startup_singleton_enabled=_read_bool("BOT_STARTUP_SINGLETON_ENABLED", default=True),
        startup_singleton_lock_path=_expand_runtime_path(
            os.getenv(
                "BOT_STARTUP_SINGLETON_LOCK_PATH",
                str(DEFAULT_STARTUP_SINGLETON_LOCK_PATH),
            ).strip()
        ),
        tts_enabled=_read_bool("TTS_ENABLED", default=False),
        tts_mode=_read_choice("TTS_MODE", "off", {"off", "discord_file", "local_live", "discord_voice", "obs_live"}),
        tts_provider=tts_provider,
        tts_command_template=tts_command_template,
        tts_play_command_template=os.getenv("TTS_PLAY_COMMAND_TEMPLATE", "").strip(),
        tts_local_player=_read_choice("TTS_LOCAL_PLAYER", "auto", {"auto", "ffplay", "mpv", "afplay", "paplay", "pw-play"}),
        tts_ffmpeg_executable=os.getenv("TTS_FFMPEG_EXECUTABLE", "ffmpeg").strip() or "ffmpeg",
        tts_output_dir=_expand_runtime_path(os.getenv("TTS_OUTPUT_DIR", str(DEFAULT_TTS_OUTPUT_DIR)).strip()),
        tts_obs_output_dir=_expand_runtime_path(
            os.getenv(
                "TTS_OBS_OUTPUT_DIR",
                str(DEFAULT_TTS_OBS_OUTPUT_DIR),
            ).strip()
        ),
        tts_audio_format=os.getenv("TTS_AUDIO_FORMAT", "ogg").strip().lower() or "ogg",
        tts_max_chars=int(os.getenv("TTS_MAX_CHARS", "240")),
        tts_elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
        tts_elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip(),
        tts_elevenlabs_base_url=os.getenv("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").strip(),
        tts_elevenlabs_request_timeout_seconds=float(os.getenv("ELEVENLABS_REQUEST_TIMEOUT_SECONDS", "60")),
        tts_white_voice_id=os.getenv("TTS_WHITE_VOICE_ID", "white_default").strip() or "white_default",
        tts_black_voice_id=os.getenv("TTS_BLACK_VOICE_ID", "black_default").strip() or "black_default",
        tts_white_speed=float(os.getenv("TTS_WHITE_SPEED", "0.94")),
        tts_black_speed=float(os.getenv("TTS_BLACK_SPEED", "1.02")),
        tts_white_style=os.getenv("TTS_WHITE_STYLE", "soft").strip() or "soft",
        tts_black_style=os.getenv("TTS_BLACK_STYLE", "clear").strip() or "clear",
        tts_xtts_server_url=tts_xtts_server_url,
        tts_xtts_server_token=tts_xtts_server_token,
        tts_xtts_client_python=tts_xtts_client_python,
        tts_xtts_client_script=tts_xtts_client_script,
        tts_xtts_language=tts_xtts_language,
        tts_gptsovits_server_url=tts_gptsovits_server_url,
        tts_gptsovits_server_token=tts_gptsovits_server_token,
        tts_gptsovits_client_python=tts_gptsovits_client_python,
        tts_gptsovits_client_script=tts_gptsovits_client_script,
        tts_gptsovits_language=tts_gptsovits_language,
    )
    if settings.memory_recent_messages < 1:
        raise ValueError("MEMORY_RECENT_MESSAGES는 1 이상이어야 해요.")
    if settings.memory_summary_trigger_messages < 1:
        raise ValueError("MEMORY_SUMMARY_TRIGGER_MESSAGES는 1 이상이어야 해요.")
    if settings.memory_summary_batch_messages < 1:
        raise ValueError("MEMORY_SUMMARY_BATCH_MESSAGES는 1 이상이어야 해요.")
    if settings.web_search_mode != "off" and not settings.tavily_api_key:
        raise ValueError("WEB_SEARCH_MODE가 off가 아닐 때는 TAVILY_API_KEY가 필요해요.")
    if settings.duo_enabled and settings.duo_partner_bot_id is None:
        raise ValueError("DUO_ENABLED=true일 때는 DUO_PARTNER_BOT_ID가 필요해요.")
    if settings.duo_max_replies < 1:
        raise ValueError("DUO_MAX_REPLIES는 1 이상이어야 해요.")
    if settings.duo_reply_delay_seconds < 0:
        raise ValueError("DUO_REPLY_DELAY_SECONDS는 0 이상이어야 해요.")
    if settings.runtime_heartbeat_seconds <= 0:
        raise ValueError("BOT_RUNTIME_HEARTBEAT_SECONDS는 0보다 커야 해요.")
    if settings.runtime_online_timeout_seconds <= 0:
        raise ValueError("BOT_RUNTIME_ONLINE_TIMEOUT_SECONDS는 0보다 커야 해요.")
    if settings.runtime_auto_claim_ttl_seconds <= 0:
        raise ValueError("BOT_RUNTIME_AUTO_CLAIM_TTL_SECONDS는 0보다 커야 해요.")
    if settings.runtime_manual_claim_ttl_seconds <= 0:
        raise ValueError("BOT_RUNTIME_MANUAL_CLAIM_TTL_SECONDS는 0보다 커야 해요.")
    if not settings.startup_singleton_lock_path:
        raise ValueError("BOT_STARTUP_SINGLETON_LOCK_PATH는 비워 둘 수 없어요.")
    if settings.tts_max_chars < 32:
        raise ValueError("TTS_MAX_CHARS는 32 이상이어야 해요.")
    if settings.tts_enabled and settings.tts_provider == "elevenlabs" and not settings.tts_elevenlabs_api_key:
        raise ValueError("TTS_PROVIDER=elevenlabs일 때는 ELEVENLABS_API_KEY가 필요해요.")
    return settings
