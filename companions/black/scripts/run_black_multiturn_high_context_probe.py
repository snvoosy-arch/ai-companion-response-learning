from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_black_quality_probe import (
    DEFAULT_ENV_FILE,
    DEFAULT_KCBERT_MODEL_PATH,
    DEFAULT_KOBART_MODEL_PATH,
    DEFAULT_POLICY_ACTION_MODEL_PATH,
    DEFAULT_RUNTIME_STATE_DB_PATH,
    DEFAULT_STATE_DB_PATH,
    _apply_env,
    _preview_text,
    _quality_flags,
    _render_source,
    build_probe_env,
    dotenv_values,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_FILE = ROOT.parent / "reports" / "black_multiturn_high_context_probe_20260415.json"
DEFAULT_OUT_JSON = ROOT.parent / "reports" / "black_multiturn_high_context_probe_results_20260415.json"
DEFAULT_OUT_MEMO = ROOT.parent / "reports" / "black_multiturn_high_context_probe_results_20260415.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="black multi-turn high-context probe를 실행합니다.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE_FILE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-memo", type=Path, default=DEFAULT_OUT_MEMO)
    parser.add_argument("--kobart-model-path", "--kobart-model", dest="kobart_model_path", type=Path, default=DEFAULT_KOBART_MODEL_PATH)
    parser.add_argument("--kcbert-model-path", type=Path, default=DEFAULT_KCBERT_MODEL_PATH)
    parser.add_argument("--policy-action-model-path", type=Path, default=DEFAULT_POLICY_ACTION_MODEL_PATH)
    parser.add_argument("--state-db-path", type=Path, default=DEFAULT_STATE_DB_PATH)
    parser.add_argument("--runtime-state-db-path", type=Path, default=DEFAULT_RUNTIME_STATE_DB_PATH)
    parser.add_argument("--persona", default="black")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def load_conversations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"probe file must be a JSON list: {path}")
    conversations: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        probe_id = str(item.get("id", f"ctx_{index:02d}")).strip()
        turns = item.get("turns")
        if not probe_id or not isinstance(turns, list):
            continue
        normalized_turns = [str(turn).strip() for turn in turns if str(turn).strip()]
        if not normalized_turns:
            continue
        conversations.append(
            {
                "id": probe_id,
                "theme": str(item.get("theme", "")).strip(),
                "turns": normalized_turns,
            }
        )
    return conversations


def _state_snapshot(state: Any) -> dict[str, Any]:
    return {
        "turn_count": getattr(state, "turn_count", None),
        "tension": getattr(state, "tension", None),
        "rapport": getattr(state, "rapport", None),
        "boundary_pressure": getattr(state, "boundary_pressure", None),
        "directness_score": getattr(state, "directness_score", None),
        "last_intent": getattr(getattr(state, "last_intent", None), "value", None),
        "last_action": getattr(getattr(state, "last_action", None), "value", None),
        "awaiting_slot": getattr(state, "awaiting_slot", None),
        "recent_turn_count": len(getattr(state, "recent_turns", []) or []),
    }


async def run_probe(
    conversations: list[dict[str, Any]],
    out_json: Path,
    out_memo: Path,
    *,
    probe_file: Path,
    kobart_model_path: Path,
) -> None:
    from predictive_bot.config import AppConfig
    from predictive_bot.factory import build_engine

    config = AppConfig.from_env()
    engine = build_engine(config)

    results: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    render_source_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    llm_used_count = 0
    total_turns = 0

    try:
        for index, conversation in enumerate(conversations, start=1):
            user_id = f"context-probe-{index:02d}"
            turn_results: list[dict[str, Any]] = []
            for turn_index, prompt in enumerate(conversation["turns"], start=1):
                total_turns += 1
                result = await engine.respond(user_id=user_id, text=prompt)
                reply = result.reply
                action = result.decision.action.value
                llm_used = bool(result.llm_used)
                llm_fallback_reason = result.llm_fallback_reason
                render_source = _render_source(llm_used, llm_fallback_reason)
                flags = _quality_flags(prompt, reply, action)
                state = engine.state_store.get_or_create(user_id)

                action_counts[action] += 1
                render_source_counts[render_source] += 1
                if llm_fallback_reason:
                    fallback_reason_counts[llm_fallback_reason] += 1
                llm_used_count += int(llm_used)

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
                        "state_after_turn": _state_snapshot(state),
                    }
                )
            results.append(
                {
                    "id": conversation["id"],
                    "theme": conversation["theme"],
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
        "kobart_model_name_or_path": str(kobart_model_path),
        "probe_file": str(probe_file),
        "conversation_count": len(results),
        "turn_count": total_turns,
        "action_counts": dict(action_counts),
        "render_source_counts": dict(render_source_counts),
        "llm_used_count": llm_used_count,
        "llm_used_ratio": round((llm_used_count / total_turns) if total_turns else 0.0, 3),
        "template_fallback_count": render_source_counts.get("template_fallback", 0),
        "template_only_count": render_source_counts.get("template_only", 0),
        "fallback_reason_counts": dict(fallback_reason_counts),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "source": str(probe_file),
                "summary": summary,
                "conversations": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    memo_lines = [
        "# Black Multi-turn High-context Probe",
        "",
        f"- probe file: `{probe_file}`",
        f"- result file: `{out_json}`",
        f"- KoBART model: `{kobart_model_path}`",
        f"- conversation count: `{summary['conversation_count']}`",
        f"- turn count: `{summary['turn_count']}`",
        "",
        "## Quick read",
        f"- KoBART used: `{summary['llm_used_count']}`",
        f"- KoBART ratio: `{summary['llm_used_ratio']}`",
        f"- template fallback: `{summary['template_fallback_count']}`",
        f"- template only: `{summary['template_only_count']}`",
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
        for reason, count in sorted(fallback_reason_counts.items(), key=lambda item: (-item[1], item[0])):
            memo_lines.append(f"- `{reason}`: `{count}`")
    else:
        memo_lines.append("- none")
    memo_lines.extend(
        [
            "",
            "## Conversation snapshots",
        ]
    )
    for conversation in results[:5]:
        memo_lines.append(f"### {conversation['id']} ({conversation['theme']})")
        for turn in conversation["turns"][:3]:
            memo_lines.append(f"- user: {_preview_text(turn['prompt'])}")
            memo_lines.append(
                "  - "
                f"{turn['render_source']} / {turn['action']} / "
                f"state(turn={turn['state_after_turn']['turn_count']}, "
                f"rapport={turn['state_after_turn']['rapport']}, "
                f"tension={turn['state_after_turn']['tension']})"
            )
            memo_lines.append(f"  - bot: {_preview_text(turn['reply'])}")
    out_memo.write_text("\n".join(memo_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.env_file.exists():
        raise SystemExit(f"env file not found: {args.env_file}")
    if not args.probe_file.exists():
        raise SystemExit(f"probe file not found: {args.probe_file}")

    env_values = dict(dotenv_values(args.env_file))
    _apply_env(build_probe_env(args, env_values))

    conversations = load_conversations(args.probe_file)
    if args.limit > 0:
        conversations = conversations[: args.limit]

    asyncio.run(
        run_probe(
            conversations,
            args.out_json,
            args.out_memo,
            probe_file=args.probe_file,
            kobart_model_path=args.kobart_model_path,
        )
    )
    print(f"saved multi-turn probe result to {args.out_json}")
    print(f"saved memo to {args.out_memo}")


if __name__ == "__main__":
    main()
