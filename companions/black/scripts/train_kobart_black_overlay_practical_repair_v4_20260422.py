from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_black_overlay_practical_repair_v4_20260422.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_kobart_black.py"
SOURCE_PATH = ROOT / "data" / "black_broad_plus_duo_overlay20_practical_repair_v4_all_20260422.jsonl"
OUTPUT_DIR = ROOT / "models" / "kobart_black_overlay_practical_repair_v4_20260422"
REPORT_PATH = ROOT / "reports" / "kobart_black_overlay_practical_repair_v4_report_20260422.json"


def _resolve_python() -> str:
    env_python = os.environ.get("BLACK_TRAIN_PYTHON")
    if env_python:
        return env_python
    training_python = Path("/root/bot-model-training-venv/bin/python")
    if training_python.exists():
        return str(training_python)
    runtime_python = Path("/root/.bot-runtime/black/venv/bin/python")
    if runtime_python.exists():
        return str(runtime_python)
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
    parser = argparse.ArgumentParser(description="overlay practical repair v4용 black KoBART build + train 런처")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-source-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-name-or-path",
        default=str(ROOT / "models" / "kobart_black_broad_plus_duo_overlay20_rebuild_v3_20260422"),
    )
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_only:
        _run_python_script(BUILD_SCRIPT)

    if args.build_only:
        return

    train_args = [
        "--source",
        str(args.source),
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
        "0.12",
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
