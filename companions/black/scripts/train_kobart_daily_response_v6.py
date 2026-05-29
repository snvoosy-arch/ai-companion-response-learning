from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    train_module = load_module(ROOT / "scripts" / "train_kobart_black.py", "train_kobart_black_v6")
    data_path = ROOT / "data" / "daily_response_rewritten_sft_v6_all.jsonl"
    output_dir = ROOT / "models" / "kobart_daily_response_rewritten_sft_v6"
    report_out = ROOT / "reports" / "kobart_daily_response_rewritten_sft_v6_report.json"

    args = train_module.parse_args(
        [
            "--source",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--report-out",
            str(report_out),
        ]
    )
    train_module.run(args)


if __name__ == "__main__":
    main()
