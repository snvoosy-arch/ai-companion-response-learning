from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
MODEL_TRAINING_ROOT = WORKSPACE_ROOT / "model training"

DEFAULT_ALIAS_PATH = WORKSPACE_ROOT / "models" / "active_model_aliases.json"
DEFAULT_ALIAS = "black.candidate.qwen2_5_0_5b_instruct"
DEFAULT_TRAIN_PATH = (
    ROOT
    / "data"
    / "black_rejected_generation_rewrite_qwen100_honest_after_stylecheck_20260427_train_messages.jsonl"
)
DEFAULT_EVAL_PATH = (
    ROOT
    / "data"
    / "black_rejected_generation_rewrite_qwen100_honest_after_stylecheck_20260427_eval_messages.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT
    / "models"
    / "candidates"
    / "black"
    / "causal_lm"
    / "qwen2_5_0_5b_rejected_rewrite_20260427"
)
DEFAULT_REPORT_PATH = ROOT / "reports" / "black_qwen_rejected_rewrite_20260427_train_report.json"
DEFAULT_TRAIN_SCRIPT = MODEL_TRAINING_ROOT / "train_sft.py"
DEFAULT_MODEL_TRAINING_PYTHON = Path(
    os.getenv("BOT_MODEL_TRAINING_PYTHON")
    or os.getenv("BOT_MODEL_TRAINING_VENV", "/root/bot-model-training-venv") + "/bin/python"
)

REQUIRED_TRAINING_MODULES = ("torch", "transformers", "datasets", "peft", "trl", "accelerate")
INTERNAL_LABEL_FRAGMENTS = {
    "activityinvite",
    "activitycontext",
    "activitydetail",
    "activityplace",
    "activitycondition",
    "aftereffecthold",
    "capability",
    "identity",
    "knowledge",
    "media",
    "opiniondecisionrequest",
    "opinionhabitpreference",
    "opinionpreferencelike",
    "quietmode",
}
POLITE_ENDING_RE = re.compile(
    r"(?:해요|돼요|좋아요|있어요|없어요|봐요|가요|와요|먹어요|할게요|볼게요|"
    r"드릴게요|해드릴게요|세요|입니다|습니다|습니까|시다면)\s*(?:[.!?。]|$)"
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_for_echo(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_file"] = str(path)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def resolve_model_name(*, alias_path: Path, alias: str, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    payload = json.loads(alias_path.read_text(encoding="utf-8"))
    aliases = payload.get("aliases") if isinstance(payload, dict) else {}
    entry = aliases.get(alias) if isinstance(aliases, dict) else None
    if not isinstance(entry, dict):
        raise RuntimeError(f"model alias not found: {alias}")
    model_name = _compact(entry.get("causal_lm_model") or entry.get("model_dir") or entry.get("model_id"))
    if not model_name:
        raise RuntimeError(f"model alias has no causal_lm_model/model_dir/model_id: {alias}")
    return model_name


def _row_completion(row: dict[str, Any]) -> str:
    completion = _compact(row.get("completion"))
    messages = row.get("messages")
    if not completion and isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            completion = _compact(last.get("content"))
    return completion


def _row_prompt(row: dict[str, Any]) -> str:
    prompt = _compact(row.get("prompt"))
    if prompt:
        return prompt
    messages = row.get("messages")
    if isinstance(messages, list):
        return _compact(" ".join(_compact(item.get("content")) for item in messages if isinstance(item, dict)))
    return ""


def validate_message_row(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        issues.append("messages_missing_or_too_short")
    else:
        roles = [item.get("role") for item in messages if isinstance(item, dict)]
        if not roles or roles[0] != "system" or roles[-1] != "assistant":
            issues.append("messages_role_contract")

    prompt = _row_prompt(row)
    completion = _row_completion(row)
    if not prompt:
        issues.append("missing_prompt")
    if not completion:
        issues.append("missing_completion")
    if completion and POLITE_ENDING_RE.search(completion):
        issues.append("polite_completion")

    normalized_completion = _normalize_for_echo(completion)
    if any(label in normalized_completion for label in INTERNAL_LABEL_FRAGMENTS):
        issues.append("internal_label_completion")
    if "```" in completion or re.search(r"^\s*(?:action|decision|draft_reply|Final reply)\s*:", completion, flags=re.I):
        issues.append("structured_artifact_completion")

    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    input_text = _compact(meta.get("input_text"))
    normalized_input = _normalize_for_echo(input_text)
    if normalized_input and len(normalized_input) >= 8 and normalized_input in normalized_completion:
        issues.append("prompt_echo_completion")

    messages_completion = ""
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            messages_completion = _compact(last.get("content"))
    if completion and messages_completion and completion != messages_completion:
        issues.append("completion_messages_mismatch")
    return list(dict.fromkeys(issues))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    completion_lengths: list[int] = []
    examples: list[dict[str, Any]] = []
    seen_pairs: Counter[tuple[str, str]] = Counter()

    for row in rows:
        issues = validate_message_row(row)
        for issue in issues:
            issue_counts[issue] += 1
        if issues and len(examples) < 5:
            examples.append(
                {
                    "source_file": row.get("_source_file", ""),
                    "line_no": row.get("_line_no", 0),
                    "issues": issues,
                    "completion": _row_completion(row),
                    "input_text": (row.get("meta") or {}).get("input_text", "") if isinstance(row.get("meta"), dict) else "",
                }
            )
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        action_counts[_compact(meta.get("action")) or "unknown"] += 1
        completion = _row_completion(row)
        prompt = _row_prompt(row)
        if completion:
            completion_lengths.append(len(completion))
        seen_pairs[(prompt, completion)] += 1

    duplicate_pairs = sum(1 for count in seen_pairs.values() if count > 1)
    duplicate_rows = sum(count for count in seen_pairs.values() if count > 1)
    return {
        "rows": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": examples,
        "duplicate_prompt_completion_groups": duplicate_pairs,
        "duplicate_prompt_completion_rows": duplicate_rows,
        "completion_length": {
            "min": min(completion_lengths) if completion_lengths else 0,
            "max": max(completion_lengths) if completion_lengths else 0,
            "avg": round(sum(completion_lengths) / len(completion_lengths), 2) if completion_lengths else 0.0,
        },
    }


def check_python_modules(python_bin: Path, modules: tuple[str, ...] = REQUIRED_TRAINING_MODULES) -> dict[str, Any]:
    if not python_bin.exists():
        return {
            "python_bin": str(python_bin),
            "available": {module: False for module in modules},
            "missing": list(modules),
            "error": "python_not_found",
        }
    probe_code = (
        "import importlib.util, json; "
        f"mods={json.dumps(list(modules))}; "
        "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
    )
    try:
        result = subprocess.run(
            [str(python_bin), "-c", probe_code],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover - defensive system boundary
        return {
            "python_bin": str(python_bin),
            "available": {module: False for module in modules},
            "missing": list(modules),
            "error": repr(exc),
        }
    if result.returncode != 0:
        return {
            "python_bin": str(python_bin),
            "available": {module: False for module in modules},
            "missing": list(modules),
            "error": _compact(result.stderr or result.stdout)[:500],
        }
    available = json.loads(result.stdout)
    missing = [module for module, ok in available.items() if not ok]
    return {
        "python_bin": str(python_bin),
        "available": available,
        "missing": missing,
        "error": "",
    }


def current_python_modules(modules: tuple[str, ...] = REQUIRED_TRAINING_MODULES) -> dict[str, Any]:
    available = {module: importlib.util.find_spec(module) is not None for module in modules}
    return {
        "python_bin": sys.executable,
        "available": available,
        "missing": [module for module, ok in available.items() if not ok],
        "error": "",
    }


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def build_training_command(args: argparse.Namespace, model_name_or_path: str) -> list[str]:
    return [
        str(args.python_bin),
        str(args.train_script),
        "--model_name_or_path",
        model_name_or_path,
        "--dataset_path",
        str(args.train_path.resolve()),
        "--eval_dataset_path",
        str(args.eval_path.resolve()),
        "--output_dir",
        str(args.output_dir.resolve()),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--max_length",
        str(args.max_length),
        "--dataset-format",
        args.dataset_format,
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--logging_steps",
        str(args.logging_steps),
        "--save_strategy",
        args.save_strategy,
        "--save_total_limit",
        str(args.save_total_limit),
        "--seed",
        str(args.seed),
        bool_flag("use_lora", args.use_lora),
        bool_flag("load_in_4bit", args.load_in_4bit),
        bool_flag("gradient_checkpointing", args.gradient_checkpointing),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
    ]


def build_report(args: argparse.Namespace, *, model_name_or_path: str, command: list[str]) -> dict[str, Any]:
    train_rows = load_jsonl(args.train_path)
    eval_rows = load_jsonl(args.eval_path)
    train_summary = summarize_rows(train_rows)
    eval_summary = summarize_rows(eval_rows)
    all_issue_counts = Counter(train_summary["issue_counts"])
    all_issue_counts.update(eval_summary["issue_counts"])
    pair_overlap = {
        (_row_prompt(row), _row_completion(row))
        for row in train_rows
    } & {
        (_row_prompt(row), _row_completion(row))
        for row in eval_rows
    }
    dependency_report = (
        current_python_modules()
        if args.check_current_python
        else check_python_modules(args.python_bin)
    )
    missing_deps = list(dependency_report.get("missing") or [])

    blockers: list[str] = []
    if not args.train_path.exists():
        blockers.append("train_path_missing")
    if not args.eval_path.exists():
        blockers.append("eval_path_missing")
    if not args.train_script.exists():
        blockers.append("train_script_missing")
    if all_issue_counts:
        blockers.append("dataset_contract_issues")
    if pair_overlap:
        blockers.append("train_eval_pair_overlap")
    if missing_deps:
        blockers.append("training_dependencies_missing")

    return {
        "status": "dry_run" if args.dry_run else "ready_to_train",
        "model_alias": args.alias,
        "model_name_or_path": model_name_or_path,
        "paths": {
            "alias_path": str(args.alias_path),
            "train_path": str(args.train_path),
            "eval_path": str(args.eval_path),
            "output_dir": str(args.output_dir),
            "report_out": str(args.report_out),
            "train_script": str(args.train_script),
        },
        "hyperparameters": {
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "dataset_format": args.dataset_format,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "use_lora": args.use_lora,
            "load_in_4bit": args.load_in_4bit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        "dataset": {
            "train": train_summary,
            "eval": eval_summary,
            "train_eval_pair_overlap": len(pair_overlap),
            "total_rows": len(train_rows) + len(eval_rows),
            "total_issue_counts": dict(sorted(all_issue_counts.items())),
        },
        "dependencies": dependency_report,
        "blockers": blockers,
        "training_command": command,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or dry-run Black Qwen rejected-generation rewrite LoRA.")
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--alias-path", type=Path, default=DEFAULT_ALIAS_PATH)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_MODEL_TRAINING_PYTHON)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-current-python",
        action="store_true",
        help="Check this interpreter's modules instead of the configured training python.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--dataset-format",
        choices=("prompt_completion", "chat_prompt_completion", "messages"),
        default="prompt_completion",
        help=(
            "Training input format for model training/train_sft.py. "
            "Use chat_prompt_completion for Qwen runtime-aligned rewrite LoRA."
        ),
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_name_or_path = resolve_model_name(
        alias_path=args.alias_path,
        alias=args.alias,
        explicit_model=args.model_name_or_path,
    )
    command = build_training_command(args, model_name_or_path)
    report = build_report(args, model_name_or_path=model_name_or_path, command=command)
    write_report(args.report_out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    hard_blockers = [item for item in report["blockers"] if item != "training_dependencies_missing"]
    if hard_blockers:
        raise SystemExit(f"training plan has blockers: {', '.join(hard_blockers)}")
    if args.dry_run:
        return 0
    if report["dependencies"].get("missing"):
        raise SystemExit(
            "training dependencies missing for configured python: "
            + ", ".join(report["dependencies"]["missing"])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=str(args.train_script.parent), check=False)
    report["status"] = "trained" if result.returncode == 0 else "train_failed"
    report["returncode"] = result.returncode
    write_report(args.report_out, report)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
