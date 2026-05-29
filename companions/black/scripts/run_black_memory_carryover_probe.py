from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from run_black_quality_probe import (
    DEFAULT_ENV_FILE,
    DEFAULT_KCBERT_MODEL_PATH,
    DEFAULT_KOBART_MODEL_PATH,
    DEFAULT_POLICY_ACTION_MODEL_PATH,
    _apply_env,
    _preview_text,
    _render_source,
    build_probe_env,
    dotenv_values,
)


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_PROBE_FILE = ROOT.parent / "reports" / "black_memory_carryover_probe_20260416.json"
DEFAULT_OUT_JSON = ROOT.parent / "reports" / "black_memory_carryover_probe_results_20260416.json"
DEFAULT_OUT_MEMO = ROOT.parent / "reports" / "black_memory_carryover_probe_results_20260416.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="black durable memory carry-over probe를 실행합니다.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE_FILE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-memo", type=Path, default=DEFAULT_OUT_MEMO)
    parser.add_argument("--kobart-model-path", type=Path, default=DEFAULT_KOBART_MODEL_PATH)
    parser.add_argument("--kcbert-model-path", type=Path, default=DEFAULT_KCBERT_MODEL_PATH)
    parser.add_argument("--policy-action-model-path", type=Path, default=DEFAULT_POLICY_ACTION_MODEL_PATH)
    parser.add_argument("--state-db-path", type=Path, default=None)
    parser.add_argument("--runtime-state-db-path", type=Path, default=None)
    parser.add_argument("--persona", default="black")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"probe file must be a JSON list: {path}")
    scenarios: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        scenario_id = str(item.get("id") or f"memory_case_{index:02d}").strip()
        seed_turns = [str(turn).strip() for turn in item.get("seed_turns") or [] if str(turn).strip()]
        filler_turns = [str(turn).strip() for turn in item.get("filler_turns") or [] if str(turn).strip()]
        followup = str(item.get("followup") or "").strip()
        if not scenario_id or not seed_turns or not followup:
            continue
        scenarios.append(
            {
                "id": scenario_id,
                "theme": str(item.get("theme") or "").strip(),
                "seed_turns": seed_turns,
                "filler_turns": filler_turns,
                "followup": followup,
            }
        )
    return scenarios


def _durable_memory_snapshot(state: Any) -> list[dict[str, str]]:
    snapshot: list[dict[str, str]] = []
    for entry in getattr(state, "durable_memory", []) or []:
        snapshot.append(
            {
                "bucket": getattr(getattr(entry, "bucket", None), "value", str(getattr(entry, "bucket", ""))),
                "text": getattr(entry, "text", ""),
                "source": getattr(entry, "source", ""),
            }
        )
    return snapshot


def _recent_turn_snapshot(state: Any) -> list[str]:
    snapshot: list[str] = []
    for turn in getattr(state, "recent_turns", []) or []:
        text = getattr(turn, "user_text", "")
        if text:
            snapshot.append(text)
    return snapshot


def _world_memory_snapshot(world_state: Any) -> dict[str, Any]:
    if world_state is None:
        return {}
    return {
        "memory_summary": getattr(world_state, "memory_summary", ""),
        "stable_preferences": list(getattr(world_state, "stable_preferences", []) or []),
        "relevant_relationship_notes": list(getattr(world_state, "relevant_relationship_notes", []) or []),
        "relevant_stress_signals": list(getattr(world_state, "relevant_stress_signals", []) or []),
        "relevant_open_loops": list(getattr(world_state, "relevant_open_loops", []) or []),
        "evidence": list(getattr(world_state, "evidence", []) or []),
    }


def _carryover_flags(world_memory: dict[str, Any], restored_durable_memory: list[dict[str, str]]) -> dict[str, bool]:
    return {
        "has_durable_memory": bool(restored_durable_memory),
        "has_relationship_memory": bool(world_memory.get("relevant_relationship_notes")),
        "has_stress_memory": bool(world_memory.get("relevant_stress_signals")),
        "has_open_loop_memory": bool(world_memory.get("relevant_open_loops")),
        "has_any_world_memory": any(
            bool(world_memory.get(key))
            for key in (
                "relevant_relationship_notes",
                "relevant_stress_signals",
                "relevant_open_loops",
                "stable_preferences",
            )
        ),
    }


async def run_probe(args: argparse.Namespace) -> None:
    from predictive_bot.config import AppConfig
    from predictive_bot.factory import build_engine

    scenarios = load_scenarios(args.probe_file)
    if args.limit > 0:
        scenarios = scenarios[: args.limit]

    env_values = dict(dotenv_values(args.env_file))
    results: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    render_source_counts: Counter[str] = Counter()
    memory_flag_counts: Counter[str] = Counter()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        state_db_path = tmp_path / "carryover_state.sqlite3"
        runtime_db_path = tmp_path / "carryover_runtime.sqlite3"

        merged_env = build_probe_env(args, env_values)
        merged_env["STATE_BACKEND"] = "sqlite"
        merged_env["STATE_DB_PATH"] = str(state_db_path)
        merged_env["BOT_RUNTIME_STATE_DB_PATH"] = str(runtime_db_path)
        _apply_env(merged_env)

        config = AppConfig.from_env()

        engine_seed = build_engine(config)
        try:
            seeded_snapshots: dict[str, dict[str, Any]] = {}
            for index, scenario in enumerate(scenarios, start=1):
                user_id = f"carryover-probe-{index:02d}"
                for turn in scenario["seed_turns"]:
                    await engine_seed.respond(user_id=user_id, text=turn)
                for turn in scenario["filler_turns"]:
                    await engine_seed.respond(user_id=user_id, text=turn)
                seed_state = engine_seed.state_store.get_or_create(user_id)
                seeded_snapshots[user_id] = {
                    "durable_memory": _durable_memory_snapshot(seed_state),
                    "recent_turns": _recent_turn_snapshot(seed_state),
                }
        finally:
            state_store = getattr(engine_seed, "state_store", None)
            if state_store is not None:
                try:
                    state_store.close()
                except Exception:
                    pass

        engine_followup = build_engine(config)
        try:
            for index, scenario in enumerate(scenarios, start=1):
                user_id = f"carryover-probe-{index:02d}"
                restored_state = engine_followup.state_store.get_or_create(user_id)
                restored_snapshot = {
                    "durable_memory": _durable_memory_snapshot(restored_state),
                    "recent_turns": _recent_turn_snapshot(restored_state),
                }
                result = await engine_followup.respond(user_id=user_id, text=scenario["followup"])
                render_source = _render_source(bool(result.llm_used), result.llm_fallback_reason)
                world_memory = _world_memory_snapshot(result.world_state)
                flags = _carryover_flags(world_memory, restored_snapshot["durable_memory"])

                action_counts[result.decision.action.value] += 1
                render_source_counts[render_source] += 1
                for key, value in flags.items():
                    if value:
                        memory_flag_counts[key] += 1

                results.append(
                    {
                        "id": scenario["id"],
                        "theme": scenario["theme"],
                        "seed_turns": list(scenario["seed_turns"]),
                        "filler_turns": list(scenario["filler_turns"]),
                        "followup": scenario["followup"],
                        "seed_state": seeded_snapshots[user_id],
                        "restored_state_before_followup": restored_snapshot,
                        "followup_result": {
                            "reply": result.reply,
                            "action": result.decision.action.value,
                            "intent": result.features.intent.value,
                            "decision_reason": result.decision.reason,
                            "response_style": result.decision.response_style,
                            "render_source": render_source,
                            "llm_used": bool(result.llm_used),
                            "llm_fallback_reason": result.llm_fallback_reason,
                            "world_memory": world_memory,
                            "verification_issues": list(result.verification.issues) if result.verification else [],
                        },
                        "carryover_flags": flags,
                    },
                )
        finally:
            state_store = getattr(engine_followup, "state_store", None)
            if state_store is not None:
                try:
                    state_store.close()
                except Exception:
                    pass

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "persona": args.persona,
        "probe_file": str(args.probe_file),
        "scenario_count": len(results),
        "action_counts": dict(action_counts),
        "render_source_counts": dict(render_source_counts),
        "memory_flag_counts": dict(memory_flag_counts),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "scenarios": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    memo_lines = [
        "# Black Memory Carry-over Probe",
        "",
        f"- probe file: `{args.probe_file}`",
        f"- result file: `{args.out_json}`",
        f"- KoBART model: `{args.kobart_model_path}`",
        f"- scenarios: `{summary['scenario_count']}`",
        "",
        "## Summary",
    ]
    for key, value in summary["memory_flag_counts"].items():
        memo_lines.append(f"- `{key}`: `{value}`")
    memo_lines.extend(["", "## Action mix"])
    for key, value in sorted(summary["action_counts"].items()):
        memo_lines.append(f"- `{key}`: `{value}`")
    memo_lines.extend(["", "## Scenario snapshots"])
    for scenario in results:
        followup = scenario["followup_result"]
        memo_lines.append(f"### {scenario['id']} ({scenario['theme']})")
        memo_lines.append(f"- followup: {_preview_text(scenario['followup'])}")
        memo_lines.append(
            f"- action/render: `{followup['action']}` / `{followup['render_source']}`"
        )
        memo_lines.append(
            f"- carried durable memory: `{len(scenario['restored_state_before_followup']['durable_memory'])}` entries"
        )
        world_memory = followup["world_memory"]
        if world_memory.get("relevant_open_loops"):
            memo_lines.append(f"- open loops: {world_memory['relevant_open_loops']}")
        if world_memory.get("relevant_relationship_notes"):
            memo_lines.append(f"- relationship notes: {world_memory['relevant_relationship_notes']}")
        if world_memory.get("relevant_stress_signals"):
            memo_lines.append(f"- stress signals: {world_memory['relevant_stress_signals']}")
        memo_lines.append(f"- bot: {_preview_text(followup['reply'])}")
    args.out_memo.write_text("\n".join(memo_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.env_file.exists():
        raise SystemExit(f"env file not found: {args.env_file}")
    if not args.probe_file.exists():
        raise SystemExit(f"probe file not found: {args.probe_file}")
    asyncio.run(run_probe(args))
    print(f"saved carry-over probe result to {args.out_json}")
    print(f"saved memo to {args.out_memo}")


if __name__ == "__main__":
    main()
