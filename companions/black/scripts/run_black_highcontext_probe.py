from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import dotenv_values
except ModuleNotFoundError:  # pragma: no cover - fallback for lean runtime envs
    def dotenv_values(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from bot_shared.runtime_paths import resolve_default_shared_runtime_state_db_path
from scripts.model_aliases import DEFAULT_ALIAS_FILE, apply_black_runtime_alias_env

DEFAULT_ENV_FILE = ROOT / ".env.black.duo.local"
DEFAULT_PROBE_FILE = ROOT / "reports" / "black_highcontext_probe_sessions_20260415.json"
DEFAULT_OUT_JSON = ROOT / "reports" / "black_highcontext_probe_20260415.json"
DEFAULT_OUT_MEMO = ROOT / "reports" / "black_highcontext_probe_20260415.md"
DEFAULT_KOBART_MODEL_PATH = (
    WORKSPACE_ROOT / "models" / "runtime" / "black" / "generation" / "kobart_black_broad_phrasing_rebuild_v2_20260422"
)
DEFAULT_KCBERT_MODEL_PATH = WORKSPACE_ROOT / "models" / "runtime" / "black" / "intent" / "kcbert_daily_intent_final"
DEFAULT_POLICY_ACTION_MODEL_PATH = (
    WORKSPACE_ROOT / "models" / "runtime" / "black" / "policy" / "policy_action_daily_centroid.json"
)
DEFAULT_STATE_DB_PATH = ROOT / "data" / "predictive_bot_state.sqlite3"
DEFAULT_RUNTIME_STATE_DB_PATH = resolve_default_shared_runtime_state_db_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="black multi-turn high-context probe를 실행합니다.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE_FILE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-memo", type=Path, default=DEFAULT_OUT_MEMO)
    parser.add_argument("--kobart-model-path", type=Path, default=None)
    parser.add_argument("--kcbert-model-path", type=Path, default=None)
    parser.add_argument("--policy-action-model-path", type=Path, default=None)
    parser.add_argument("--alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--state-db-path", type=Path, default=None)
    parser.add_argument("--runtime-state-db-path", type=Path, default=None)
    parser.add_argument("--persona", default="black")
    parser.add_argument("--session-limit", type=int, default=0)
    parser.add_argument("--turn-limit", type=int, default=0)
    return parser.parse_args()


def _apply_env(items: dict[str, str | None]) -> None:
    for key, value in items.items():
        if value is None:
            continue
        os.environ[key] = value


def build_probe_env(args: argparse.Namespace, env_values: dict[str, str]) -> dict[str, str]:
    merged = dict(env_values)
    merged.setdefault("BOT_RUNTIME_ENABLED", "false")
    merged.setdefault("BOT_DUO_ENABLED", "false")
    merged.setdefault("BOT_STARTUP_LOCK_ENABLED", "false")
    merged.setdefault("STATE_BACKEND", "memory")
    merged.setdefault("BOT_TRIGGER_PREFIX", "!predict")
    merged.setdefault("BOT_PERSONA", args.persona)
    merged.setdefault("GENERATION_BACKEND", "kobart")
    if args.kobart_model_path is not None:
        merged["KOBART_MODEL_NAME_OR_PATH"] = str(args.kobart_model_path)
    else:
        merged.setdefault("KOBART_MODEL_NAME_OR_PATH", str(DEFAULT_KOBART_MODEL_PATH))
    if args.kcbert_model_path is not None:
        merged["KCBERT_MODEL_PATH"] = str(args.kcbert_model_path)
    else:
        merged.setdefault("KCBERT_MODEL_PATH", str(DEFAULT_KCBERT_MODEL_PATH))
    if args.policy_action_model_path is not None:
        merged["POLICY_ACTION_MODEL_PATH"] = str(args.policy_action_model_path)
    else:
        merged.setdefault("POLICY_ACTION_MODEL_PATH", str(DEFAULT_POLICY_ACTION_MODEL_PATH))
    if args.state_db_path is not None:
        merged["STATE_DB_PATH"] = str(args.state_db_path)
    else:
        merged.setdefault("STATE_DB_PATH", str(DEFAULT_STATE_DB_PATH))
    if args.runtime_state_db_path is not None:
        merged["BOT_RUNTIME_STATE_DB_PATH"] = str(args.runtime_state_db_path)
    else:
        merged.setdefault("BOT_RUNTIME_STATE_DB_PATH", str(DEFAULT_RUNTIME_STATE_DB_PATH))
    apply_black_runtime_alias_env(
        merged,
        alias_file=args.alias_file,
        alias_name=args.model_alias,
    )
    return merged


def _runtime_metadata_from_config(config) -> dict[str, Any]:
    return {
        "generation_backend": config.generation_backend,
        "intent_model_type": config.intent_model_type,
        "state_backend": config.state_backend,
        "kobart_model_name_or_path": config.kobart_model_name_or_path,
        "kcbert_model_path": config.kcbert_model_path,
        "policy_action_model_path": config.policy_action_model_path,
        "black_model_alias_file": config.black_model_alias_file,
        "black_model_alias": config.black_model_alias,
        "black_model_alias_status": config.black_model_alias_status,
        "state_db_path": config.state_db_path,
        "runtime_state_db_path": config.runtime_state_db_path,
    }


def _load_sessions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"high-context probe file must be a JSON list: {path}")
    sessions: list[dict[str, Any]] = []
    for raw_session in payload:
        if not isinstance(raw_session, dict):
            continue
        session_id = str(raw_session.get("id") or raw_session.get("session_id") or "").strip()
        if not session_id:
            continue
        raw_turns = raw_session.get("turns")
        if not isinstance(raw_turns, list):
            continue
        turns = [str(turn).strip() for turn in raw_turns if str(turn).strip()]
        if not turns:
            continue
        sessions.append(
            {
                "id": session_id,
                "category": str(raw_session.get("category", "")).strip() or "uncategorized",
                "description": str(raw_session.get("description", "")).strip(),
                "turns": turns,
            }
        )
    return sessions


def _quality_flags(reply: str, action: str) -> dict[str, bool]:
    lowered = reply.lower()
    template_like_markers = (
        "한 줄만 더",
        "조금만 더 풀어",
        "말해봐",
        "설명해줘",
        "기능",
        "취향",
        "내 기준",
        "응답은",
        "지금 말만으론",
        "어느 쪽",
        "다시 한 번",
    )
    awkward_markers = (
        "그럴 수 있지. 괜찮아.",
        "말수가 적어질 것 같아.",
        "조금 더",
        "꽤 잘",
        "꽤",
        "애매해.",
        "말이 좋다",
        "느낌이다.",
        "느낌이 든다.",
        "공간",
        "정리되는",
    )
    over_clarify = action == "ask_clarification" or any(token in lowered for token in ("한 줄만 더", "조금만 더 풀어", "어느 쪽", "무엇을 말", "뭘 말", "설명해줘"))
    template_like = any(token in reply for token in template_like_markers)
    awkward_korean = any(token in reply for token in awkward_markers)
    topic_switch_ok = not template_like and not over_clarify and not awkward_korean
    return {
        "template_like": template_like,
        "over_clarify": over_clarify,
        "awkward_korean": awkward_korean,
        "topic_switch_ok": topic_switch_ok,
    }


def _render_source(llm_used: bool, llm_fallback_reason: str | None) -> str:
    if llm_used:
        return "kobart"
    if llm_fallback_reason:
        return "template_fallback"
    return "template_only"


def _preview_text(text: str, *, limit: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


async def _run_probe_sessions(
    sessions: list[dict[str, Any]],
    out_json: Path,
    out_memo: Path,
    *,
    probe_file: Path,
) -> None:
    from predictive_bot.config import AppConfig
    from predictive_bot.factory import build_engine

    config = AppConfig.from_env()
    engine = build_engine(config)
    runtime = _runtime_metadata_from_config(config)

    session_results: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    render_source_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    llm_used_count = 0
    template_like_count = 0
    over_clarify_count = 0
    awkward_count = 0
    topic_switch_ok_count = 0
    turn_count = 0

    try:
        for session in sessions:
            session_id = str(session["id"])
            category = str(session.get("category", "uncategorized"))
            category_counts[category] = category_counts.get(category, 0) + 1
            user_id = f"highcontext::{session_id}"
            turn_results: list[dict[str, Any]] = []

            for turn_index, prompt in enumerate(session["turns"], start=1):
                result = await engine.respond(user_id=user_id, text=prompt)
                reply = result.reply
                action = result.decision.action.value
                llm_used = bool(result.llm_used)
                llm_fallback_reason = result.llm_fallback_reason
                render_source = _render_source(llm_used, llm_fallback_reason)
                flags = _quality_flags(reply, action)

                turn_count += 1
                action_counts[action] = action_counts.get(action, 0) + 1
                render_source_counts[render_source] = render_source_counts.get(render_source, 0) + 1
                if llm_fallback_reason:
                    fallback_reason_counts[llm_fallback_reason] = fallback_reason_counts.get(llm_fallback_reason, 0) + 1
                llm_used_count += int(llm_used)
                template_like_count += int(flags["template_like"])
                over_clarify_count += int(flags["over_clarify"])
                awkward_count += int(flags["awkward_korean"])
                topic_switch_ok_count += int(flags["topic_switch_ok"])

                turn_results.append(
                    {
                        "turn_index": turn_index,
                        "prompt": prompt,
                        "reply": reply,
                        "action": action,
                        "intent": result.features.intent.value,
                        "decision_reason": result.decision.reason,
                        "response_style": result.decision.response_style,
                        "render_source": render_source,
                        "llm_used": llm_used,
                        "llm_fallback_reason": llm_fallback_reason,
                        "flags": flags,
                        "verification_issues": list(result.verification.issues),
                    }
                )

            session_results.append(
                {
                    "id": session_id,
                    "category": category,
                    "description": str(session.get("description", "")),
                    "turn_count": len(turn_results),
                    "turns": turn_results,
                }
            )
    finally:
        state_store = getattr(engine, "state_store", None)
        if state_store is not None:
            try:
                state_store.close()
            except Exception:
                pass

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "persona": config.bot_persona,
        "kobart_model_name_or_path": runtime["kobart_model_name_or_path"],
        "probe_file": str(probe_file),
        "session_count": len(session_results),
        "turn_count": turn_count,
        "category_counts": category_counts,
        "action_counts": action_counts,
        "render_source_counts": render_source_counts,
        "llm_used_count": llm_used_count,
        "llm_used_ratio": round((llm_used_count / turn_count) if turn_count else 0.0, 3),
        "template_fallback_count": render_source_counts.get("template_fallback", 0),
        "template_only_count": render_source_counts.get("template_only", 0),
        "fallback_reason_counts": fallback_reason_counts,
        "flag_counts": {
            "template_like": template_like_count,
            "over_clarify": over_clarify_count,
            "awkward_korean": awkward_count,
            "topic_switch_ok": topic_switch_ok_count,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "probe_type": "multi_turn_high_context",
                "source": str(probe_file),
                "runtime": runtime,
                "summary": summary,
                "sessions": session_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    memo_lines = [
        "# Black multi-turn high-context probe",
        "",
        f"- probe file: `{probe_file}`",
        f"- result file: `{out_json}`",
        f"- persona: `{summary['persona']}`",
        f"- KoBART model: `{runtime['kobart_model_name_or_path']}`",
        f"- model alias: `{runtime.get('black_model_alias') or ''}`",
        f"- sessions: `{summary['session_count']}`",
        f"- turns: `{summary['turn_count']}`",
        "",
        "## Quick read",
        f"- KoBART used: `{summary['llm_used_count']}`",
        f"- KoBART ratio: `{summary['llm_used_ratio']}`",
        f"- template fallback: `{summary['template_fallback_count']}`",
        f"- template only: `{summary['template_only_count']}`",
        f"- template-like: `{template_like_count}`",
        f"- over-clarify: `{over_clarify_count}`",
        f"- awkward Korean: `{awkward_count}`",
        f"- topic-switch OK: `{topic_switch_ok_count}`",
        "",
        "## Category mix",
    ]
    for category, count in sorted(category_counts.items()):
        memo_lines.append(f"- `{category}`: `{count}`")
    memo_lines.extend(["", "## Render source mix"])
    for source, count in sorted(render_source_counts.items()):
        memo_lines.append(f"- `{source}`: `{count}`")
    memo_lines.extend(["", "## Fallback reasons"])
    if fallback_reason_counts:
        for reason, count in sorted(fallback_reason_counts.items(), key=lambda item: (-item[1], item[0])):
            memo_lines.append(f"- `{reason}`: `{count}`")
    else:
        memo_lines.append("- none")
    memo_lines.extend(["", "## Session previews"])
    for session in session_results:
        memo_lines.append(f"- `{session['id']}` `{session['category']}`: `{session['turn_count']}` turns")
        if session["turns"]:
            first_turn = session["turns"][0]
            memo_lines.append(
                f"  - first turn: `{_preview_text(first_turn['prompt'])}` -> `{_preview_text(first_turn['reply'])}`"
            )
    out_memo.write_text("\n".join(memo_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.env_file.exists():
        raise SystemExit(f"env file not found: {args.env_file}")
    if not args.probe_file.exists():
        raise SystemExit(f"probe file not found: {args.probe_file}")

    env_values = dict(dotenv_values(args.env_file))
    _apply_env(build_probe_env(args, env_values))

    sessions = _load_sessions(args.probe_file)
    if args.session_limit > 0:
        sessions = sessions[: args.session_limit]
    if args.turn_limit > 0:
        sessions = [
            {**session, "turns": list(session["turns"][: args.turn_limit])}
            for session in sessions
        ]

    asyncio.run(_run_probe_sessions(sessions, args.out_json, args.out_memo, probe_file=args.probe_file))
    print(f"saved high-context probe result to {args.out_json}")
    print(f"saved memo to {args.out_memo}")


if __name__ == "__main__":
    main()
