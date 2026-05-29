from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "meaning" / "black_draft_planner_fixed_eval_v1_20260509.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "black_draft_planner_heads_eval_v5_noframe_goldmix_on_fixed_eval_20260509.json"
DEFAULT_OUTPUT = ROOT / "data" / "meaning" / "black_draft_planner_fixed_eval_failures_v1_20260509.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "black_draft_planner_fixed_eval_failures_v1_20260509_summary.json"
DEFAULT_HEADS = ("emotion", "state_hint", "action_hint", "draft_frame", "tone", "followup_policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Black DraftPlanner eval failures as training-review rows.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--heads", default=",".join(DEFAULT_HEADS))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(",") if part.strip())


def label_of(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("label")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def row_targets(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("targets") if isinstance(row.get("targets"), dict) else {})


def export_failures(dataset: list[dict[str, Any]], report: dict[str, Any], heads: tuple[str, ...]) -> list[dict[str, Any]]:
    by_id = {str(row.get("id")): row for row in dataset}
    failures: list[dict[str, Any]] = []
    for prediction in report.get("row_predictions", []):
        if not isinstance(prediction, dict):
            continue
        row_id = str(prediction.get("id") or "")
        source_row = by_id.get(row_id)
        if source_row is None:
            continue
        targets = row_targets(source_row)
        actual_payload = prediction.get("actual") if isinstance(prediction.get("actual"), dict) else {}
        failed_heads: list[str] = []
        actual_labels: dict[str, Any] = {}
        actual_top: dict[str, Any] = {}
        for head in heads:
            expected = label_of(targets.get(head))
            if expected is None:
                continue
            actual_axis = actual_payload.get(head) if isinstance(actual_payload, dict) else {}
            actual = label_of(actual_axis)
            if isinstance(actual_axis, dict):
                actual_top[head] = actual_axis.get("top", {})
            actual_labels[head] = actual
            if actual != expected:
                failed_heads.append(head)
        if not failed_heads:
            continue
        row = dict(source_row)
        meta = dict(row.get("meta") if isinstance(row.get("meta"), dict) else {})
        meta.update(
            {
                "source_report": str(report.get("model_dir") or ""),
                "source_dataset": str(report.get("dataset") or ""),
                "failed_heads": failed_heads,
                "actual_labels": actual_labels,
                "actual_top": actual_top,
            }
        )
        row["meta"] = meta
        row["label_status"] = "planner_fixed_eval_failure"
        row["issues"] = [
            {
                "type": "planner_head_mismatch",
                "heads": failed_heads,
                "expected": {head: targets.get(head) for head in failed_heads},
                "actual": {head: actual_labels.get(head) for head in failed_heads},
            }
        ]
        failures.append(row)
    return failures


def main() -> None:
    args = parse_args()
    heads = split_csv(args.heads)
    dataset = load_jsonl(args.dataset)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = export_failures(dataset, report, heads)
    write_jsonl(args.output, failures)
    head_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    for row in failures:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        head_counter.update(meta.get("failed_heads", []))
        category_counter[str(meta.get("category") or "uncategorized")] += 1
    summary = {
        "dataset": str(args.dataset),
        "report": str(args.report),
        "output": str(args.output),
        "row_count": len(failures),
        "failed_heads": dict(head_counter.most_common()),
        "categories": dict(category_counter.most_common()),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
