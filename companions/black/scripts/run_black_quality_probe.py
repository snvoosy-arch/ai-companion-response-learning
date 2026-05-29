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
DEFAULT_PROBE_FILE = ROOT.parent / "reports" / "white_black_quality_probe_prompts_20260415.json"
DEFAULT_OUT_JSON = ROOT.parent / "reports" / "black_quality_probe_20260415.json"
DEFAULT_OUT_MEMO = ROOT.parent / "reports" / "black_quality_probe_20260415.md"
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
    parser = argparse.ArgumentParser(description="black conversational quality probe를 실행합니다.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE_FILE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-memo", type=Path, default=DEFAULT_OUT_MEMO)
    parser.add_argument("--kobart-model-path", "--kobart-model", dest="kobart_model_path", type=Path, default=None)
    parser.add_argument("--kcbert-model-path", type=Path, default=None)
    parser.add_argument("--policy-action-model-path", type=Path, default=None)
    parser.add_argument("--alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--state-db-path", type=Path, default=None)
    parser.add_argument("--runtime-state-db-path", type=Path, default=None)
    parser.add_argument("--persona", default="black")
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def _apply_env(items: dict[str, str | None]) -> None:
    for key, value in items.items():
        if value is None:
            continue
        os.environ[key] = value


def _load_probe_items(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"probe file must be a JSON list: {path}")
    items: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        probe_id = str(item.get("id", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if probe_id and prompt:
            items.append({"id": probe_id, "prompt": prompt})
    return items


def _quality_flags(prompt: str, reply: str, action: str) -> dict[str, bool]:
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
    stock_tail_markers = (
        "그런 날이 있더라",
        "그런 날이 있지",
        "그런 날은 처음인 거구나",
        "처음인 거구나",
    )
    awkward_closure_markers = (
        "같이 있던 돌아왔",
        "몸이 마음이",
        "민폐에 빠질 수 있어",
        "그 날은 말이 더 길게 느껴질 때가 있어",
    )
    over_clarify = action == "ask_clarification" or any(token in lowered for token in ("한 줄만 더", "조금만 더 풀어", "어느 쪽", "무엇을 말", "뭘 말", "설명해줘"))
    template_like = any(token in reply for token in template_like_markers)
    awkward_korean = any(token in reply for token in awkward_markers)
    stock_tail = any(token in reply for token in stock_tail_markers)
    awkward_closure = any(token in reply for token in awkward_closure_markers)
    topic_switch_ok = not template_like and not over_clarify and not awkward_korean and not stock_tail and not awkward_closure
    return {
        "template_like": template_like,
        "over_clarify": over_clarify,
        "awkward_korean": awkward_korean,
        "stock_tail": stock_tail,
        "awkward_closure": awkward_closure,
        "topic_switch_ok": topic_switch_ok,
    }


def _render_source(llm_used: bool, llm_fallback_reason: str | None, *, generation_backend: str) -> str:
    if llm_used:
        return generation_backend or "llm"
    if llm_fallback_reason:
        return "template_fallback"
    return "template_only"


def _preview_text(text: str, *, limit: int = 72) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


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
        "causal_lm_model_name_or_path": config.causal_lm_model_name_or_path,
        "kcbert_model_path": config.kcbert_model_path,
        "policy_action_model_path": config.policy_action_model_path,
        "black_model_alias_file": config.black_model_alias_file,
        "black_model_alias": config.black_model_alias,
        "black_model_alias_status": config.black_model_alias_status,
        "state_db_path": config.state_db_path,
        "runtime_state_db_path": config.runtime_state_db_path,
    }


async def _run_probe(
    items: list[dict[str, str]],
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

    results: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    render_source_counts: dict[str, int] = {}
    fallback_reason_counts: dict[str, int] = {}
    generation_issue_counts: dict[str, int] = {}
    template_like_count = 0
    over_clarify_count = 0
    awkward_count = 0
    topic_switch_ok_count = 0
    llm_used_count = 0

    try:
        for index, item in enumerate(items, start=1):
            probe_id = item["id"]
            prompt = item["prompt"]
            result = await engine.respond(user_id=f"probe-{index:02d}", text=prompt)
            reply = result.reply
            action = result.decision.action.value
            llm_used = bool(result.llm_used)
            llm_fallback_reason = result.llm_fallback_reason
            llm_generation_issue = getattr(result, "llm_generation_issue", None)
            render_source = _render_source(
                llm_used,
                llm_fallback_reason,
                generation_backend=config.generation_backend,
            )
            flags = _quality_flags(prompt, reply, action)

            action_counts[action] = action_counts.get(action, 0) + 1
            render_source_counts[render_source] = render_source_counts.get(render_source, 0) + 1
            if llm_fallback_reason:
                fallback_reason_counts[llm_fallback_reason] = fallback_reason_counts.get(llm_fallback_reason, 0) + 1
            if llm_generation_issue:
                generation_issue_counts[llm_generation_issue] = generation_issue_counts.get(llm_generation_issue, 0) + 1
            template_like_count += int(flags["template_like"])
            over_clarify_count += int(flags["over_clarify"])
            awkward_count += int(flags["awkward_korean"])
            topic_switch_ok_count += int(flags["topic_switch_ok"])
            llm_used_count += int(llm_used)

            results.append(
                {
                    "id": probe_id,
                    "prompt": prompt,
                    "reply": reply,
                    "action": action,
                    "intent": result.features.intent.value,
                    "decision_reason": result.decision.reason,
                    "response_style": result.decision.response_style,
                    "render_source": render_source,
                    "llm_used": llm_used,
                    "llm_fallback_reason": llm_fallback_reason,
                    "llm_generation_issue": llm_generation_issue,
                    "flags": flags,
                    "verification_issues": list(result.verification.issues),
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
        "generation_backend": config.generation_backend,
        "kobart_model_name_or_path": runtime["kobart_model_name_or_path"],
        "causal_lm_model_name_or_path": runtime["causal_lm_model_name_or_path"],
        "probe_file": str(probe_file),
        "probe_count": len(results),
        "action_counts": action_counts,
        "render_source_counts": render_source_counts,
        "llm_used_count": llm_used_count,
        "llm_used_ratio": round((llm_used_count / len(results)) if results else 0.0, 3),
        "template_fallback_count": render_source_counts.get("template_fallback", 0),
        "template_only_count": render_source_counts.get("template_only", 0),
        "fallback_reason_counts": fallback_reason_counts,
        "generation_issue_counts": generation_issue_counts,
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
                "probe_type": "single_turn",
                "source": str(probe_file),
                "runtime": runtime,
                "summary": summary,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    memo_lines = [
        "# Black 30-문장 품질 probe",
        "",
        f"- probe file: `{probe_file}`",
        f"- result file: `{out_json}`",
        f"- persona: `{summary['persona']}`",
        f"- generation backend: `{summary['generation_backend']}`",
        f"- KoBART model: `{runtime['kobart_model_name_or_path']}`",
        f"- causal LM model: `{runtime['causal_lm_model_name_or_path']}`",
        f"- model alias: `{runtime.get('black_model_alias') or ''}`",
        f"- probe count: `{summary['probe_count']}`",
        "",
        "## How to read",
        "- `kobart`/`causal_lm`: 해당 생성기가 실제로 사용된 경우",
        "- `template_fallback`: 실패 감지용 지표. Black strict runtime에서는 0이어야 함",
        "- `template_only`: 애초에 KoBART-first 대상이 아닌 액션이라 템플릿만 쓴 경우",
        "",
        "## Quick read",
        f"- LLM used: `{llm_used_count}`",
        f"- LLM ratio: `{summary['llm_used_ratio']}`",
        f"- template fallback: `{summary['template_fallback_count']}`",
        f"- template only: `{summary['template_only_count']}`",
        f"- template-like: `{template_like_count}`",
        f"- over-clarify: `{over_clarify_count}`",
        f"- awkward Korean: `{awkward_count}`",
        f"- topic-switch OK: `{topic_switch_ok_count}`",
        "",
        "## Render source mix",
    ]
    for source, count in sorted(render_source_counts.items()):
        memo_lines.append(f"- `{source}`: `{count}`")
    memo_lines.extend(
        [
            "",
            "## Fallback reasons",
        ]
    )
    if fallback_reason_counts:
        for reason, count in sorted(
            fallback_reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            memo_lines.append(f"- `{reason}`: `{count}`")
    else:
        memo_lines.append("- none")
    memo_lines.extend(
        [
            "",
            "## Generation issues",
        ]
    )
    if generation_issue_counts:
        for issue, count in sorted(
            generation_issue_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            memo_lines.append(f"- `{issue}`: `{count}`")
    else:
        memo_lines.append("- none")
    memo_lines.extend(
        [
            "",
            "## Fallback examples",
        ]
    )
    fallback_examples: list[tuple[str, str, str]] = []
    seen_reasons: set[str] = set()
    for item in results:
        reason = item.get("llm_fallback_reason")
        if not reason or reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        fallback_examples.append(
            (
                str(item.get("id", "")),
                str(reason),
                _preview_text(str(item.get("prompt", ""))),
            )
        )
    if fallback_examples:
        for probe_id, reason, prompt in fallback_examples:
            memo_lines.append(f"- `{probe_id}` `{reason}`: `{prompt}`")
    else:
        memo_lines.append("- none")
    memo_lines.extend(
        [
            "",
        "## Action mix",
        ]
    )
    for action, count in sorted(action_counts.items()):
        memo_lines.append(f"- `{action}`: `{count}`")
    memo_lines.extend(
        [
            "",
            "## Notes",
            "- Black strict runtime expects fallback counts to remain 0",
            "- generation issues are logged separately from fallback reasons",
            "- probe is single-turn and uses the common prompt file as requested",
            "- detailed per-item flags are in the JSON result",
        ]
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

    items = _load_probe_items(args.probe_file)
    if args.limit > 0:
        items = items[: args.limit]

    asyncio.run(
        _run_probe(
            items,
            args.out_json,
            args.out_memo,
            probe_file=args.probe_file,
        )
    )
    print(f"saved probe result to {args.out_json}")
    print(f"saved memo to {args.out_memo}")


if __name__ == "__main__":
    main()
