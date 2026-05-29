from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = ROOT / "data" / "black_high_context_gold_v1_20260420.json"
DEFAULT_TRAIN_PATH = ROOT / "data" / "black_high_context_gold_v1_train_20260420.jsonl"
DEFAULT_EVAL_PATH = ROOT / "data" / "black_high_context_gold_v1_eval_20260420.jsonl"
DEFAULT_SUMMARY_PATH = ROOT / "reports" / "black_high_context_gold_v1_summary_20260420.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="black high-context gold JSON을 intent-seed JSONL로 export합니다."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported gold payload: {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"gold payload must include a 'cases' list: {path}")
    meta = {
        "name": str(payload.get("name", "")).strip(),
        "version": str(payload.get("version", "")).strip(),
        "purpose": str(payload.get("purpose", "")).strip(),
    }
    return meta, cases


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def render_context(turns: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for turn in turns:
        speaker = normalize_text(turn.get("speaker")).lower()
        text = normalize_text(turn.get("text"))
        if not speaker or not text:
            continue
        rendered.append(f"{speaker}: {text}")
    return " | ".join(rendered)


def build_record(case: dict[str, Any]) -> dict[str, Any]:
    turns = case.get("turns")
    if not isinstance(turns, list):
        turns = []
    final_input = normalize_text(case.get("final_input"))
    if not final_input:
        final_input = normalize_text(turns[-1].get("text")) if turns else ""
    if not final_input:
        raise ValueError(f"case is missing final input text: {case.get('id')}")

    context_text = render_context(turns)
    context_required = bool(case.get("context_required", False))
    return {
        "text": final_input,
        "intent": normalize_text(case.get("gold_intent")),
        "source": normalize_text(case.get("source_dataset")) or "black_high_context_gold",
        "assistant_reply": None,
        "meta": {
            "case_id": normalize_text(case.get("id")),
            "cluster": normalize_text(case.get("cluster")),
            "gold_action": normalize_text(case.get("gold_action")),
            "context_required": context_required,
            "source_bucket": normalize_text(case.get("source_bucket")),
            "gold_reply_shape": normalize_text(case.get("gold_reply_shape")),
            "label_notes": normalize_text(case.get("label_notes")),
            "context_text": context_text,
            "turn_count": len(turns),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    meta, raw_cases = load_cases(args.source)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    intent_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    context_required_counts: Counter[str] = Counter()

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        raw_case = dict(raw_case)
        raw_case.setdefault("source_dataset", meta.get("name") or "black_high_context_gold")
        record = build_record(raw_case)
        split = "eval" if normalize_text(raw_case.get("split")).lower() == "eval" else "train"
        if split == "eval":
            eval_rows.append(record)
        else:
            train_rows.append(record)
        intent_counts[record["intent"]] += 1
        split_counts[split] += 1
        cluster_counts[record["meta"]["cluster"]] += 1
        context_required_counts["true" if record["meta"]["context_required"] else "false"] += 1

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)

    summary = {
        "source_path": str(args.source),
        "train_path": str(args.train_out),
        "eval_path": str(args.eval_out),
        "total_records": len(train_rows) + len(eval_rows),
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "intent_counts": dict(intent_counts),
        "split_counts": dict(split_counts),
        "cluster_counts": dict(cluster_counts),
        "context_required_counts": dict(context_required_counts),
        "meta": meta,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("black high-context gold exported")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
