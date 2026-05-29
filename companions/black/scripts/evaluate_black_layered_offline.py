from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.evaluation.layered import (
    build_failure_rows,
    build_offline_draft_engine,
    load_layered_eval_items,
    replay_layered_items,
    write_jsonl,
)


DEFAULT_DATASET = ROOT / "data" / "evals" / "black_meaning_probe30_manual_direct_20260428.json"
DEFAULT_REPORT = ROOT / "reports" / "black_layered_offline_eval_report.json"
DEFAULT_FAILURES = ROOT / "data" / "black_layered_offline_failures.jsonl"
DEFAULT_V27_MEANING_MODEL = (
    ROOT.parents[1]
    / "models"
    / "candidates"
    / "black"
    / "intent"
    / "modernbert_meaning_gold_direct_v27_fullreview_pass_mix_20260504"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Black's draft-only layered offline eval: meaning -> state -> action -> draft -> rewrite boundary."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSON/JSONL eval dataset")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT, help="Layered report JSON path")
    parser.add_argument("--failures-out", type=Path, default=DEFAULT_FAILURES, help="Failure repair JSONL path")
    parser.add_argument("--suite-name", default="black_layered_offline_eval")
    parser.add_argument("--pass-threshold", type=float, default=0.82)
    parser.add_argument("--default-location", default=None)
    parser.add_argument(
        "--meaning-model-path",
        type=Path,
        default=None,
        help="Optional ModernBERT multi-head meaning model path. Leave unset for heuristic-only eval.",
    )
    parser.add_argument(
        "--use-v27-meaning",
        action="store_true",
        help="Shortcut for the current v27 ModernBERT meaning candidate.",
    )
    parser.add_argument("--meaning-device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--meaning-min-confidence", type=float, default=0.10)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict[str, object]:
    meaning_model_path = args.meaning_model_path
    if args.use_v27_meaning:
        meaning_model_path = DEFAULT_V27_MEANING_MODEL
    engine = build_offline_draft_engine(
        default_location=args.default_location,
        meaning_model_path=meaning_model_path,
        meaning_device=args.meaning_device,
        meaning_min_confidence=args.meaning_min_confidence,
    )
    try:
        items = load_layered_eval_items(args.dataset)
        report = await replay_layered_items(
            engine,
            items,
            suite_name=args.suite_name,
            pass_threshold=args.pass_threshold,
        )
    finally:
        engine.state_store.close()

    report["dataset"] = str(args.dataset)
    report["report_out"] = str(args.report_out)
    report["failures_out"] = str(args.failures_out)
    report["meaning_model_path"] = str(meaning_model_path) if meaning_model_path is not None else None
    report["meaning_device"] = args.meaning_device
    report["meaning_min_confidence"] = args.meaning_min_confidence
    report["notes"] = [
        "Qwen/rewrite is disabled; final_rewrite records the draft-only boundary.",
        "Schema/action labels are evaluated as supporting signals, not as the only quality target.",
    ]
    return report


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    failures = build_failure_rows(report)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(args.failures_out, failures)

    print(
        json.dumps(
            {
                "mode": report["mode"],
                "rewrite_enabled": report["rewrite_enabled"],
                "case_count": report["case_count"],
                "accuracy": report["accuracy"],
                "failed_count": report["failed_count"],
                "failure_rows": len(failures),
                "meaning_model_path": report["meaning_model_path"],
                "layer_metrics": report["layer_metrics"],
                "failure_targets": report["failure_targets"],
                "weak_layer_metrics": report["weak_layer_metrics"],
                "report_out": str(args.report_out),
                "failures_out": str(args.failures_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
