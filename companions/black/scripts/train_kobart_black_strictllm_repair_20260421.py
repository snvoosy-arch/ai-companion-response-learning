from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_black_kobart_phrase_stability_repair_sft.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_kobart_black.py"
SOURCE_PATH = ROOT / "data" / "kobart_strictllm_repair_seed_20260421.json"
ALL_PATH = ROOT / "data" / "kobart_strictllm_repair_all_20260421.jsonl"
TRAIN_PATH = ROOT / "data" / "kobart_strictllm_repair_train_20260421.jsonl"
EVAL_PATH = ROOT / "data" / "kobart_strictllm_repair_eval_20260421.jsonl"
SUMMARY_PATH = ROOT / "reports" / "kobart_strictllm_repair_summary_20260421.json"
OUTPUT_DIR = ROOT / "models" / "kobart_black_strictllm_repair_20260421"
REPORT_PATH = ROOT / "reports" / "kobart_black_strictllm_repair_report_20260421.json"


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


def _run_python_script(script: Path, *args: str) -> None:
    python_bin = _resolve_python()
    cmd = [
        python_bin,
        _normalize_arg_for_python(str(script), python_bin),
        *[_normalize_arg_for_python(arg, python_bin) for arg in args],
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="strict no-fallback black KoBART repair용 build + train 런처")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-source-length", type=int, default=320)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-name-or-path",
        default=str(ROOT / "models" / "kobart_black_phase_a_integrated_20260419"),
        help="현재 black KoBART 통합 모델 경로",
    )
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--all-path", type=Path, default=ALL_PATH)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_only:
        build_args = [
            "--source",
            str(args.source),
            "--all-path",
            str(args.all_path),
            "--train-path",
            str(args.train_path),
            "--eval-path",
            str(args.eval_path),
            "--summary-path",
            str(args.summary_path),
            "--eval-ratio",
            str(args.eval_ratio),
        ]
        _run_python_script(BUILD_SCRIPT, *build_args)

    if args.build_only:
        return

    train_args = [
        "--source",
        str(args.all_path),
        "--model-name-or-path",
        args.model_name_or_path,
        "--output-dir",
        str(args.output_dir),
        "--report-out",
        str(args.report_out),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--eval-ratio",
        str(args.eval_ratio),
        "--max-source-length",
        str(args.max_source_length),
        "--max-target-length",
        str(args.max_target_length),
    ]
    if args.dry_run:
        train_args.append("--dry-run")
    _run_python_script(TRAIN_SCRIPT, *train_args)


if __name__ == "__main__":
    main()
