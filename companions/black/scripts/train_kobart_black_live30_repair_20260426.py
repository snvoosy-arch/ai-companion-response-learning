from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
BUILD_SCRIPT = ROOT / "scripts" / "build_black_live30_repair_20260426.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_kobart_black.py"
ALL_PATH = ROOT / "data" / "black_live30_repair_20260426_all.jsonl"
OUTPUT_DIR = WORKSPACE_ROOT / "models" / "candidates" / "black" / "generation" / "kobart_black_live30_repair_20260426"
REPORT_PATH = ROOT / "reports" / "kobart_black_live30_repair_20260426_train_report.json"
ALIAS_PATH = WORKSPACE_ROOT / "models" / "active_model_aliases.json"
FALLBACK_BASE_MODEL = WORKSPACE_ROOT / "models" / "runtime" / "black" / "generation" / "kobart_black_broad_phrasing_rebuild_v2_20260422"


def _resolve_python() -> str:
    env_python = os.environ.get("BLACK_TRAIN_PYTHON")
    if env_python:
        return env_python
    wsl_runtime_python = Path("/root/.bot-runtime/black/venv/bin/python")
    if wsl_runtime_python.exists():
        return str(wsl_runtime_python)
    repo_python = ROOT / ".venv" / "bin" / "python"
    if repo_python.exists():
        return str(repo_python)
    windows_repo_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if windows_repo_python.exists():
        return str(windows_repo_python)
    return sys.executable


def _normalize_arg_for_python(arg: str, python_bin: str) -> str:
    if not python_bin.lower().endswith(".exe"):
        return arg
    if not arg.startswith("/"):
        return arg
    return subprocess.check_output(["wslpath", "-w", arg], text=True).strip()


def _default_base_model() -> str:
    if ALIAS_PATH.exists():
        try:
            aliases = json.loads(ALIAS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            aliases = {}
        black_active = aliases.get("aliases", {}).get("black.active")
        if isinstance(black_active, dict):
            kobart_model = str(black_active.get("kobart_model") or "").strip()
            if kobart_model:
                return kobart_model
    return str(FALLBACK_BASE_MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black live30 repair용 KoBART 미세보정을 위한 build + train 런처"
    )
    parser.add_argument("--build-only", action="store_true", help="repair SFT만 만들고 학습은 건너뜀")
    parser.add_argument("--train-only", action="store_true", help="이미 생성된 SFT를 사용하고 build는 건너뜀")
    parser.add_argument("--dry-run", action="store_true", help="학습 스크립트의 dry-run만 실행")
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="train 스크립트에 넘길 eval 비율")
    parser.add_argument("--epochs", type=int, default=1, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=4, help="학습 배치 크기")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="평가 배치 크기")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="학습률")
    parser.add_argument("--max-source-length", type=int, default=512, help="입력 최대 길이")
    parser.add_argument("--max-target-length", type=int, default=96, help="출력 최대 길이")
    parser.add_argument(
        "--model-name-or-path",
        default=_default_base_model(),
        help="기존 black KoBART 모델 경로. 기본값은 models/active_model_aliases.json의 black.active.kobart_model",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="결과 모델 저장 경로")
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH, help="학습 리포트 저장 경로")
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

    if not args.train_only:
        _run_python_script(BUILD_SCRIPT)

    if args.build_only:
        return

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
