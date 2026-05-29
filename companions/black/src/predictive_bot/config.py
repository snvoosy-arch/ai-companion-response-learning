from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
PRIVATE_RUNTIME_ROOT = Path.home() / ".bot-runtime"
BLACK_PRIVATE_RUNTIME_ROOT = PRIVATE_RUNTIME_ROOT / "black"
DEFAULT_STATE_DB_PATH = BLACK_PRIVATE_RUNTIME_ROOT / "predictive_bot_state.sqlite3"
DEFAULT_KCBERT_MODEL_PATH = WORKSPACE_ROOT / "models" / "runtime" / "black" / "intent" / "kcbert_daily_intent_final"
DEFAULT_POLICY_ACTION_MODEL_PATH = (
    WORKSPACE_ROOT / "models" / "runtime" / "black" / "policy" / "policy_action_daily_centroid.json"
)
DEFAULT_STARTUP_LOCK_PATH = BLACK_PRIVATE_RUNTIME_ROOT / "predictive_bot_black.startup.lock"
DEFAULT_TTS_OUTPUT_DIR = BLACK_PRIVATE_RUNTIME_ROOT / "tts"
DEFAULT_TTS_OBS_OUTPUT_DIR = BLACK_PRIVATE_RUNTIME_ROOT / "obs-live"
DEFAULT_MODEL_ALIAS_FILE = WORKSPACE_ROOT / "models" / "active_model_aliases.json"
DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES = (
    "schema",
    "emotion",
    "state_hint",
    "action_hint",
    "draft_frame_family",
    "tone",
)
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from bot_shared.runtime_paths import resolve_default_shared_runtime_state_db_path
from bot_shared.speech import resolve_gptsovits_command_template, resolve_xtts_command_template

DEFAULT_RUNTIME_STATE_DB_PATH = resolve_default_shared_runtime_state_db_path()


@dataclass(slots=True)
class AppConfig:
    discord_bot_token: str | None
    bot_trigger_prefix: str
    default_location: str | None
    bot_persona: str
    generation_backend: str
    kobart_model_name_or_path: str
    kobart_device: str
    kobart_max_new_tokens: int
    kobart_num_beams: int
    state_backend: str
    state_db_path: str
    state_max_recent_turns: int
    intent_model_type: str  # "kcbert", "modernbert_meaning", or "charngram"
    intent_model_path: str | None
    kcbert_model_path: str
    intent_model_min_confidence: float
    policy_action_model_path: str | None
    knowledge_backend: str
    wikidata_user_agent: str
    wikidata_timeout_seconds: float
    openai_api_key: str | None
    openai_model: str | None
    openai_base_url: str
    openai_timeout_seconds: float
    log_message_content: bool
    runtime_state_enabled: bool
    runtime_state_db_path: str
    runtime_bot_name: str
    runtime_heartbeat_seconds: float
    runtime_online_timeout_seconds: float
    runtime_auto_claim_ttl_seconds: int
    runtime_manual_claim_ttl_seconds: int
    duo_mode_enabled: bool
    duo_partner_bot_id: str | None
    duo_channel_id: str | None
    duo_max_turns_per_bot: int
    duo_autostart_enabled: bool
    duo_autostart_channel_id: str | None
    duo_autostart_prompt: str | None
    startup_lock_enabled: bool
    startup_lock_path: str
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
    kcbert_device: str = "auto"
    kobart_input_mode: str = "full"
    strict_llm_only: bool = False
    black_draft_only: bool = False
    llm_output_guard_enabled: bool = True
    black_model_alias_file: str = ""
    black_model_alias: str = ""
    black_model_alias_status: str = ""
    causal_lm_model_name_or_path: str = "google/gemma-3-1b-it"
    causal_lm_device: str = "auto"
    causal_lm_max_new_tokens: int = 96
    causal_lm_temperature: float = 0.35
    causal_lm_top_p: float = 0.9
    causal_lm_quantization: str = ""
    meaning_trusted_axes: tuple[str, ...] | None = DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES

    @classmethod
    def from_env(cls) -> "AppConfig":
        black_model_alias_file = _normalize_cross_os_path(
            _clean_env("BLACK_MODEL_ALIAS_FILE") or str(DEFAULT_MODEL_ALIAS_FILE)
        )
        black_model_alias = _clean_env("BLACK_MODEL_ALIAS") or ""
        black_alias = _load_black_model_alias(black_model_alias_file, black_model_alias)

        tts_provider = (_clean_env("TTS_PROVIDER") or "noop").lower()
        tts_xtts_server_url = _clean_env("TTS_XTTS_SERVER_URL") or ""
        tts_xtts_server_token = _clean_env("TTS_XTTS_SERVER_TOKEN") or ""
        tts_xtts_client_python = _normalize_cross_os_path(_clean_env("TTS_XTTS_CLIENT_PYTHON") or "")
        tts_xtts_client_script = _normalize_cross_os_path(_clean_env("TTS_XTTS_CLIENT_SCRIPT") or "")
        tts_xtts_language = _clean_env("TTS_XTTS_LANGUAGE") or "ko"
        tts_gptsovits_server_url = _clean_env("TTS_GPTSOVITS_SERVER_URL") or ""
        tts_gptsovits_server_token = _clean_env("TTS_GPTSOVITS_SERVER_TOKEN") or ""
        tts_gptsovits_client_python = _normalize_cross_os_path(_clean_env("TTS_GPTSOVITS_CLIENT_PYTHON") or "")
        tts_gptsovits_client_script = _normalize_cross_os_path(_clean_env("TTS_GPTSOVITS_CLIENT_SCRIPT") or "")
        tts_gptsovits_speaker_manifest = _normalize_cross_os_path(
            _clean_env("TTS_GPTSOVITS_SPEAKER_MANIFEST") or ""
        )
        tts_gptsovits_language = _clean_env("TTS_GPTSOVITS_LANGUAGE") or "ko"
        tts_command_template = resolve_gptsovits_command_template(
            explicit_template=(_clean_env("GPTSOVITS_SYNTH_COMMAND_TEMPLATE") or "")
            or (_clean_env("TTS_COMMAND_TEMPLATE") or ""),
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
                explicit_template=_clean_env("TTS_COMMAND_TEMPLATE") or "",
                server_url=tts_xtts_server_url,
                workspace_root=WORKSPACE_ROOT,
                client_python=tts_xtts_client_python,
                client_script=tts_xtts_client_script,
                language=tts_xtts_language,
                server_token=tts_xtts_server_token,
            )
        llm_output_guard_enabled = _read_bool_env(
            "BLACK_OUTPUT_GUARD_ENABLED",
            _read_bool_env("LLM_OUTPUT_GUARD_ENABLED", True),
        )
        return cls(
            discord_bot_token=_clean_env("DISCORD_BOT_TOKEN"),
            bot_trigger_prefix=_clean_env("BOT_TRIGGER_PREFIX") or "!predict",
            default_location=_clean_env("DEFAULT_LOCATION"),
            bot_persona=(_clean_env("BOT_PERSONA") or "black").lower(),
            generation_backend=(_clean_env("GENERATION_BACKEND") or "template").lower(),
            kobart_model_name_or_path=_normalize_cross_os_path(
                _alias_value(black_alias, "kobart_model")
                or _clean_env("KOBART_MODEL_NAME_OR_PATH")
                or "gogamza/kobart-base-v2"
            ),
            kobart_device=(_clean_env("KOBART_DEVICE") or "auto").lower(),
            kobart_max_new_tokens=int(_clean_env("KOBART_MAX_NEW_TOKENS") or "24"),
            kobart_num_beams=int(_clean_env("KOBART_NUM_BEAMS") or "1"),
            state_backend=(_clean_env("STATE_BACKEND") or "sqlite").lower(),
            state_db_path=_expand_runtime_path(_clean_env("STATE_DB_PATH") or str(DEFAULT_STATE_DB_PATH)),
            state_max_recent_turns=int(_clean_env("STATE_MAX_RECENT_TURNS") or "6"),
            intent_model_type=(_alias_value(black_alias, "intent_model_type") or _clean_env("INTENT_MODEL_TYPE") or "kcbert").lower(),
            intent_model_path=_normalize_optional_cross_os_path(
                _alias_value(black_alias, "intent_centroid") or _clean_env("INTENT_MODEL_PATH")
            ),
            kcbert_model_path=_normalize_cross_os_path(
                _alias_value(black_alias, "intent_model")
                or _clean_env("KCBERT_MODEL_PATH")
                or str(DEFAULT_KCBERT_MODEL_PATH)
            ),
            kcbert_device=(_clean_env("KCBERT_DEVICE") or "auto").lower(),
            intent_model_min_confidence=float(_clean_env("INTENT_MODEL_MIN_CONFIDENCE") or "0.10"),
            policy_action_model_path=_normalize_cross_os_path(
                _alias_value(black_alias, "policy_action_model")
                or _clean_env("POLICY_ACTION_MODEL_PATH")
                or str(DEFAULT_POLICY_ACTION_MODEL_PATH)
            ),
            knowledge_backend=(_clean_env("KNOWLEDGE_BACKEND") or "wikidata").lower(),
            wikidata_user_agent=_clean_env("WIKIDATA_USER_AGENT")
            or "predictive-discord-bot/0.1 (https://example.invalid/predictive-discord-bot)",
            wikidata_timeout_seconds=float(_clean_env("WIKIDATA_TIMEOUT_SECONDS") or "10"),
            openai_api_key=_clean_env("OPENAI_API_KEY"),
            openai_model=_clean_env("OPENAI_MODEL"),
            openai_base_url=_clean_env("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            openai_timeout_seconds=float(_clean_env("OPENAI_TIMEOUT_SECONDS") or "20"),
            log_message_content=(_clean_env("LOG_MESSAGE_CONTENT") or "false").lower() in {"1", "true", "yes", "on"},
            runtime_state_enabled=(_clean_env("BOT_RUNTIME_ENABLED") or "true").lower() in {"1", "true", "yes", "on"},
            runtime_state_db_path=_expand_runtime_path(
                _clean_env("BOT_RUNTIME_STATE_DB_PATH") or str(DEFAULT_RUNTIME_STATE_DB_PATH)
            ),
            runtime_bot_name=(_clean_env("BOT_RUNTIME_BOT_NAME") or "black").lower(),
            runtime_heartbeat_seconds=float(_clean_env("BOT_RUNTIME_HEARTBEAT_SECONDS") or "15"),
            runtime_online_timeout_seconds=float(_clean_env("BOT_RUNTIME_ONLINE_TIMEOUT_SECONDS") or "45"),
            runtime_auto_claim_ttl_seconds=int(_clean_env("BOT_RUNTIME_AUTO_CLAIM_TTL_SECONDS") or "300"),
            runtime_manual_claim_ttl_seconds=int(_clean_env("BOT_RUNTIME_MANUAL_CLAIM_TTL_SECONDS") or "1800"),
            duo_mode_enabled=(_clean_env("BOT_DUO_ENABLED") or "false").lower() in {"1", "true", "yes", "on"},
            duo_partner_bot_id=_clean_env("BOT_DUO_PARTNER_BOT_ID"),
            duo_channel_id=_clean_env("BOT_DUO_CHANNEL_ID"),
            duo_max_turns_per_bot=int(_clean_env("BOT_DUO_MAX_TURNS_PER_BOT") or "6"),
            duo_autostart_enabled=(_clean_env("BOT_DUO_AUTOSTART_ENABLED") or "false").lower()
            in {"1", "true", "yes", "on"},
            duo_autostart_channel_id=_clean_env("BOT_DUO_AUTOSTART_CHANNEL_ID"),
            duo_autostart_prompt=_clean_env("BOT_DUO_AUTOSTART_PROMPT"),
            startup_lock_enabled=(_clean_env("BOT_STARTUP_LOCK_ENABLED") or "true").lower()
            in {"1", "true", "yes", "on"},
            startup_lock_path=_expand_runtime_path(_clean_env("BOT_STARTUP_LOCK_PATH") or str(DEFAULT_STARTUP_LOCK_PATH)),
            tts_enabled=(_clean_env("TTS_ENABLED") or "false").lower() in {"1", "true", "yes", "on"},
            tts_mode=(_clean_env("TTS_MODE") or "off").lower(),
            tts_provider=tts_provider,
            tts_command_template=tts_command_template,
            tts_play_command_template=_clean_env("TTS_PLAY_COMMAND_TEMPLATE") or "",
            tts_local_player=_read_choice("TTS_LOCAL_PLAYER", "auto", {"auto", "ffplay", "mpv", "afplay", "paplay", "pw-play"}),
            tts_ffmpeg_executable=_clean_env("TTS_FFMPEG_EXECUTABLE") or "ffmpeg",
            tts_output_dir=_expand_runtime_path(_clean_env("TTS_OUTPUT_DIR") or str(DEFAULT_TTS_OUTPUT_DIR)),
            tts_obs_output_dir=_expand_runtime_path(
                _clean_env("TTS_OBS_OUTPUT_DIR") or str(DEFAULT_TTS_OBS_OUTPUT_DIR)
            ),
            tts_audio_format=(_clean_env("TTS_AUDIO_FORMAT") or "ogg").lower(),
            tts_max_chars=int(_clean_env("TTS_MAX_CHARS") or "240"),
            tts_elevenlabs_api_key=_clean_env("ELEVENLABS_API_KEY") or "",
            tts_elevenlabs_model_id=_clean_env("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2",
            tts_elevenlabs_base_url=_clean_env("ELEVENLABS_BASE_URL") or "https://api.elevenlabs.io",
            tts_elevenlabs_request_timeout_seconds=float(_clean_env("ELEVENLABS_REQUEST_TIMEOUT_SECONDS") or "60"),
            tts_white_voice_id=_clean_env("TTS_WHITE_VOICE_ID") or "white_default",
            tts_black_voice_id=_clean_env("TTS_BLACK_VOICE_ID") or "black_default",
            tts_white_speed=float(_clean_env("TTS_WHITE_SPEED") or "0.94"),
            tts_black_speed=float(_clean_env("TTS_BLACK_SPEED") or "1.02"),
            tts_white_style=_clean_env("TTS_WHITE_STYLE") or "soft",
            tts_black_style=_clean_env("TTS_BLACK_STYLE") or "clear",
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
            kobart_input_mode=_read_choice("KOBART_INPUT_MODE", "full", {"full", "slim"}),
            strict_llm_only=(_clean_env("STRICT_LLM_ONLY") or "false").lower() in {"1", "true", "yes", "on"},
            black_draft_only=(_clean_env("BLACK_DRAFT_ONLY") or "false").lower() in {"1", "true", "yes", "on"},
            llm_output_guard_enabled=llm_output_guard_enabled,
            black_model_alias_file=black_model_alias_file,
            black_model_alias=black_model_alias,
            black_model_alias_status=str(black_alias.get("status", "")) if black_alias else "",
            causal_lm_model_name_or_path=_normalize_optional_model_or_path(
                _alias_value(black_alias, "causal_lm_model")
                or _clean_env("CAUSAL_LM_MODEL_NAME_OR_PATH")
                or "google/gemma-3-1b-it"
            ),
            causal_lm_device=(_clean_env("CAUSAL_LM_DEVICE") or _clean_env("KOBART_DEVICE") or "auto").lower(),
            causal_lm_max_new_tokens=int(_clean_env("CAUSAL_LM_MAX_NEW_TOKENS") or "96"),
            causal_lm_temperature=float(_clean_env("CAUSAL_LM_TEMPERATURE") or "0.35"),
            causal_lm_top_p=float(_clean_env("CAUSAL_LM_TOP_P") or "0.9"),
            causal_lm_quantization=(_clean_env("CAUSAL_LM_QUANTIZATION") or "").lower(),
            meaning_trusted_axes=_read_axis_list(
                _alias_value(black_alias, "meaning_trusted_axes")
                or _clean_env("INTENT_MEANING_TRUSTED_AXES")
                or _clean_env("MEANING_TRUSTED_AXES"),
                default=DEFAULT_MODERNBERT_MEANING_TRUSTED_AXES,
            ),
        )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)


def _clean_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _read_bool_env(name: str, default: bool) -> bool:
    value = (_clean_env(name) or "").lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _read_axis_list(raw: str | None, *, default: tuple[str, ...] | None) -> tuple[str, ...] | None:
    value = str(raw or "").strip()
    if not value:
        return default
    lowered = value.lower()
    if lowered in {"*", "all", "open", "ungated"}:
        return None
    if lowered in {"none", "empty", "blocked"}:
        return ()
    axes = [part.strip() for part in re.split(r"[\s,;|]+", value) if part.strip()]
    return tuple(dict.fromkeys(axes))


def _load_black_model_alias(alias_file: str, alias_name: str) -> dict[str, str]:
    if not alias_name:
        return {}

    alias_path = Path(alias_file)
    if not alias_path.exists():
        raise FileNotFoundError(f"BLACK_MODEL_ALIAS_FILE does not exist: {alias_path}")

    try:
        registry = json.loads(alias_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"BLACK_MODEL_ALIAS_FILE is not valid json: {alias_path}: {exc}") from exc

    aliases = registry.get("aliases")
    if not isinstance(aliases, dict):
        raise ValueError(f"BLACK_MODEL_ALIAS_FILE has no aliases object: {alias_path}")

    entry = aliases.get(alias_name)
    if not isinstance(entry, dict):
        available = ", ".join(sorted(str(key) for key in aliases.keys()))
        raise KeyError(f"BLACK_MODEL_ALIAS not found: {alias_name}; available: {available}")

    role = str(entry.get("role", ""))
    if role and role != "black-runtime":
        raise ValueError(f"BLACK_MODEL_ALIAS must point to role=black-runtime: {alias_name} role={role}")

    resolved: dict[str, str] = {}
    for key in (
        "kobart_model",
        "causal_lm_model",
        "intent_model",
        "intent_centroid",
        "policy_action_model",
        "intent_model_type",
        "meaning_trusted_axes",
    ):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            if key == "intent_model_type":
                resolved[key] = value.strip().lower()
            elif key == "meaning_trusted_axes":
                resolved[key] = value.strip()
            else:
                normalized = _normalize_optional_model_or_path(value.strip())
                if _looks_like_filesystem_path(normalized) and not Path(normalized).exists():
                    raise FileNotFoundError(f"BLACK_MODEL_ALIAS path does not exist: {alias_name}.{key}={normalized}")
                resolved[key] = normalized
    status = entry.get("status")
    if isinstance(status, str):
        resolved["status"] = status
    return resolved


def _alias_value(alias: dict[str, str], key: str) -> str | None:
    return alias.get(key) or None


def _expand_runtime_path(raw: str) -> str:
    return str(Path(_normalize_cross_os_path(raw)))


def _normalize_optional_model_or_path(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if _looks_like_hf_repo_id(value):
        return value
    return _normalize_cross_os_path(value)


def _looks_like_hf_repo_id(value: str) -> bool:
    if value.startswith(("/", ".", "~")):
        return False
    if _WINDOWS_ABS_PATH.match(value):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._/-]*", value))


def _looks_like_filesystem_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", ".", "~")):
        return True
    if _WINDOWS_ABS_PATH.match(value):
        return True
    if "\\" in value:
        return True
    return False


_WINDOWS_ABS_PATH = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
_WSL_MOUNT_PATH = re.compile(r"^/mnt/(?P<drive>[A-Za-z])/(?P<rest>.*)$")


def _normalize_cross_os_path(raw: str) -> str:
    if not raw:
        return raw
    expanded = os.path.expandvars(raw)
    expanded = os.path.expanduser(expanded)
    if os.name != "nt":
        match = _WINDOWS_ABS_PATH.match(expanded)
        if match:
            drive = match.group("drive").lower()
            rest = match.group("rest").replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return expanded
    match = _WSL_MOUNT_PATH.match(expanded)
    if match:
        drive = match.group("drive").upper()
        rest = match.group("rest").replace("/", "\\")
        return f"{drive}:\\{rest}"
    return expanded


def _normalize_optional_cross_os_path(raw: str | None) -> str | None:
    if raw is None:
        return None
    return _normalize_cross_os_path(raw)


def _read_choice(name: str, default: str, choices: set[str]) -> str:
    raw = (_clean_env(name) or default).lower()
    if raw not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} 값이 올바르지 않아: {raw}. 가능한 값: {allowed}")
    return raw
