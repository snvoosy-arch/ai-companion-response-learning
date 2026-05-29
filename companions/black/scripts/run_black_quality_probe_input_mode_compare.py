from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_black_quality_probe.py"
DEFAULT_ENV_FILE = ROOT / ".env.black.duo.local"
DEFAULT_PROBE_FILE = ROOT.parent / "reports" / "white_black_quality_probe_prompts_alt_20260415.json"
DEFAULT_OUTPUT_DIR = ROOT.parent / "reports" / "black_quality_probe_input_mode_compare_20260417"


def _resolve_python() -> str:
    env_python = os.environ.get("BLACK_TRAIN_PYTHON")
    if env_python:
        return env_python
    wsl_runtime_python = Path("/root/.bot-runtime/black/venv/bin/python")
    if wsl_runtime_python.exists():
        return str(wsl_runtime_python)
    repo_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if repo_python.exists():
        return str(repo_python)
    return sys.executable


def _normalize_arg_for_python(arg: str, python_bin: str) -> str:
    if not python_bin.lower().endswith(".exe"):
        return arg
    if not arg.startswith("/"):
        return arg
    return subprocess.check_output(["wslpath", "-w", arg], text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="black single-turn quality probe를 full/slim 입력 모드로 비교합니다.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE_FILE)
    parser.add_argument("--kobart-model-path", "--kobart-model", dest="kobart_model_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args()


def _run_single_mode(*, mode: str, args: argparse.Namespace, out_json: Path, out_memo: Path) -> None:
    python_bin = _resolve_python()
    env = dict(os.environ)
    env["KOBART_INPUT_MODE"] = mode
    cmd = [
        python_bin,
        _normalize_arg_for_python(str(RUNNER), python_bin),
        "--env-file",
        _normalize_arg_for_python(str(args.env_file), python_bin),
        "--probe-file",
        _normalize_arg_for_python(str(args.probe_file), python_bin),
        "--kobart-model",
        _normalize_arg_for_python(str(args.kobart_model_path), python_bin),
        "--out-json",
        _normalize_arg_for_python(str(out_json), python_bin),
        "--out-memo",
        _normalize_arg_for_python(str(out_memo), python_bin),
        "--limit",
        str(args.limit),
    ]
    subprocess.run(cmd, check=True, env=env)


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _count_deltas(full_counts: dict[str, int], slim_counts: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(full_counts) | set(slim_counts))
    return {key: slim_counts.get(key, 0) - full_counts.get(key, 0) for key in keys}


def _join_item_deltas(full_payload: dict[str, Any], slim_payload: dict[str, Any]) -> list[dict[str, Any]]:
    full_results = {item["id"]: item for item in full_payload.get("results", [])}
    slim_results = {item["id"]: item for item in slim_payload.get("results", [])}
    rows: list[dict[str, Any]] = []
    for probe_id in sorted(set(full_results) | set(slim_results)):
        full_item = full_results.get(probe_id)
        slim_item = slim_results.get(probe_id)
        if full_item is None or slim_item is None:
            rows.append({"id": probe_id, "missing_in_one_side": True})
            continue
        rows.append(
            {
                "id": probe_id,
                "prompt": full_item["prompt"],
                "same_reply": full_item["reply"] == slim_item["reply"],
                "full_action": full_item["action"],
                "slim_action": slim_item["action"],
                "full_render_source": full_item["render_source"],
                "slim_render_source": slim_item["render_source"],
                "full_reply": full_item["reply"],
                "slim_reply": slim_item["reply"],
                "full_flags": full_item["flags"],
                "slim_flags": slim_item["flags"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_json = args.output_dir / "full.json"
    full_memo = args.output_dir / "full.md"
    slim_json = args.output_dir / "slim.json"
    slim_memo = args.output_dir / "slim.md"
    compare_json = args.output_dir / "compare.json"
    compare_memo = args.output_dir / "compare.md"

    _run_single_mode(mode="full", args=args, out_json=full_json, out_memo=full_memo)
    _run_single_mode(mode="slim", args=args, out_json=slim_json, out_memo=slim_memo)

    full_payload = _load_summary(full_json)
    slim_payload = _load_summary(slim_json)
    full_summary = full_payload["summary"]
    slim_summary = slim_payload["summary"]

    compare_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "probe_file": str(args.probe_file),
        "kobart_model_name_or_path": str(args.kobart_model_path),
        "limit": args.limit,
        "full_summary": full_summary,
        "slim_summary": slim_summary,
        "summary_deltas": {
            "render_source_counts": _count_deltas(full_summary.get("render_source_counts", {}), slim_summary.get("render_source_counts", {})),
            "fallback_reason_counts": _count_deltas(full_summary.get("fallback_reason_counts", {}), slim_summary.get("fallback_reason_counts", {})),
            "flag_counts": _count_deltas(full_summary.get("flag_counts", {}), slim_summary.get("flag_counts", {})),
            "llm_used_ratio_delta": round(float(slim_summary.get("llm_used_ratio", 0.0)) - float(full_summary.get("llm_used_ratio", 0.0)), 3),
        },
        "item_deltas": _join_item_deltas(full_payload, slim_payload),
        "artifacts": {
            "full_json": str(full_json),
            "slim_json": str(slim_json),
            "full_memo": str(full_memo),
            "slim_memo": str(slim_memo),
        },
    }
    compare_json.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    memo_lines = [
        "# Black Quality Probe Input Mode Compare",
        "",
        f"- probe file: `{args.probe_file}`",
        f"- KoBART model: `{args.kobart_model_path}`",
        f"- limit: `{args.limit}`",
        "",
        "## Quick read",
        f"- full llm_used_ratio: `{full_summary.get('llm_used_ratio')}`",
        f"- slim llm_used_ratio: `{slim_summary.get('llm_used_ratio')}`",
        f"- llm_used_ratio delta: `{compare_payload['summary_deltas']['llm_used_ratio_delta']}`",
        "",
        "## Render source deltas (slim - full)",
    ]
    for key, value in compare_payload["summary_deltas"]["render_source_counts"].items():
        memo_lines.append(f"- `{key}`: `{value}`")
    memo_lines.extend(["", "## Fallback deltas (slim - full)"])
    fallback_deltas = compare_payload["summary_deltas"]["fallback_reason_counts"]
    if fallback_deltas:
        for key, value in fallback_deltas.items():
            memo_lines.append(f"- `{key}`: `{value}`")
    else:
        memo_lines.append("- none")
    memo_lines.extend(["", "## Flag deltas (slim - full)"])
    for key, value in compare_payload["summary_deltas"]["flag_counts"].items():
        memo_lines.append(f"- `{key}`: `{value}`")
    memo_lines.extend(["", "## Per-item changes"])
    changed_rows = [row for row in compare_payload["item_deltas"] if not row.get("same_reply", False)]
    if changed_rows:
        for row in changed_rows[:10]:
            memo_lines.append(f"### {row['id']}")
            memo_lines.append(f"- prompt: `{row['prompt']}`")
            memo_lines.append(f"- full: `{row['full_render_source']}` / `{row['full_action']}` / `{row['full_reply']}`")
            memo_lines.append(f"- slim: `{row['slim_render_source']}` / `{row['slim_action']}` / `{row['slim_reply']}`")
    else:
        memo_lines.append("- no reply differences")
    compare_memo.write_text("\n".join(memo_lines) + "\n", encoding="utf-8")

    print(f"saved full probe to {full_json}")
    print(f"saved slim probe to {slim_json}")
    print(f"saved compare result to {compare_json}")


if __name__ == "__main__":
    main()
