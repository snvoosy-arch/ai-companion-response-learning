from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_kobart_black.py"
ALL_PATH = ROOT / "data" / "kobart_memory_carryover_phrase_repair_all_20260416.jsonl"
OUTPUT_DIR = ROOT / "models" / "kobart_black_memory_carryover_phrase_repair_20260416"
REPORT_PATH = ROOT / "reports" / "kobart_black_memory_carryover_phrase_repair_report_20260416.json"


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
    parser = argparse.ArgumentParser(
        description="black memory carry-over phrasing repair용 KoBART 미세보정 train 런처"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-source-length", type=int, default=320)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument(
        "--model-name-or-path",
        default=str(ROOT / "models" / "kobart_black_memory_carryover_repair_20260416"),
        help="기존 black KoBART 모델 경로",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def _run_python_script(script: Path, *args: str) -> None:
    python_bin = _resolve_python()
    cmd = [
        python_bin,
        _normalize_arg_for_python(str(script), python_bin),
        *[_normalize_arg_for_python(arg, python_bin) for arg in args],
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    train_args = [
        "--source",
        str(ALL_PATH),
        "--model-name-or-path",
        args.model_name_or_path,
        "--output-dir",
        str(args.output_dir),
        "--report-out",
        str(args.report_out),
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
