from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "meaning"
DEFAULT_REPORT_DIR = ROOT / "reports"
DEFAULT_BASE_TRAIN = DEFAULT_OUTPUT_DIR / "black_meaning_gold_direct_v11_daily30_repair_20260428_train.jsonl"
DEFAULT_BASE_EVAL = DEFAULT_OUTPUT_DIR / "black_meaning_gold_direct_v11_daily30_repair_20260428_eval.jsonl"
DEFAULT_PREFIX = "black_meaning_gold_direct_v14_balanced_trainonly_resolver_open_topic_20260429"
DEFAULT_REINFORCE_EXCLUDE_SCHEMAS = "preference_disclosure,habit_preference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic train-only balanced Black meaning dataset. "
            "The old eval split is preserved for promotion comparison, while "
            "new resolver rows and reinforced old-schema train rows are added to train only."
        )
    )
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--add-train", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--reinforce-factor", type=int, default=1)
    parser.add_argument(
        "--reinforce-exclude-schemas",
        default=DEFAULT_REINFORCE_EXCLUDE_SCHEMAS,
        help="Comma-separated schemas not duplicated during old-schema reinforcement.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"no rows found: {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def schema_of(row: dict[str, Any]) -> str:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    value = targets.get("schema", row.get("schema"))
    return str(value or "__none__")


def slots_of(row: dict[str, Any]) -> dict[str, str]:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    slots = targets.get("slots", row.get("slots", {}))
    if not isinstance(slots, dict):
        return {}
    return {str(key): str(value) for key, value in slots.items() if str(value).strip()}


def slot_label_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for label, raw_value in slots_of(row).items():
            if label in {"request", "schema"}:
                continue
            for value in str(raw_value).split("|"):
                if value.strip():
                    counts[label] += 1
    return counts


def add_meta(row: dict[str, Any], patch: dict[str, Any]) -> None:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.update(patch)
    row["meta"] = meta


def reinforced_copy(row: dict[str, Any], *, prefix: str, index: int, round_index: int) -> dict[str, Any]:
    copy = deepcopy(row)
    original_id = str(copy.get("id") or f"row_{index:05d}")
    copy["id"] = f"{prefix}_reinforce_r{round_index}_{index:05d}"
    add_meta(
        copy,
        {
            "balanced_copy_of": original_id,
            "balanced_copy_reason": "old_schema_reinforcement",
            "no_seed_expansion": True,
        },
    )
    return copy


def main() -> None:
    args = parse_args()
    excluded_schemas = {
        item.strip()
        for item in str(args.reinforce_exclude_schemas).split(",")
        if item.strip()
    }

    base_train = load_jsonl(args.base_train)
    base_eval = load_jsonl(args.base_eval)
    eval_texts = {str(row.get("text") or "") for row in base_eval}

    added_rows: list[dict[str, Any]] = []
    skipped_eval_overlap = 0
    skipped_duplicate_add = 0
    seen_train_texts = {str(row.get("text") or "") for row in base_train}
    for add_path in args.add_train:
        for row in load_jsonl(add_path):
            text = str(row.get("text") or "")
            if text in eval_texts:
                skipped_eval_overlap += 1
                continue
            if text in seen_train_texts:
                skipped_duplicate_add += 1
                continue
            seen_train_texts.add(text)
            added_rows.append(row)

    reinforced_rows: list[dict[str, Any]] = []
    if args.reinforce_factor > 0:
        reinforce_source = [
            row
            for row in base_train
            if schema_of(row) not in excluded_schemas
        ]
        for round_index in range(1, args.reinforce_factor + 1):
            for index, row in enumerate(reinforce_source, 1):
                reinforced_rows.append(
                    reinforced_copy(row, prefix=args.prefix, index=index, round_index=round_index)
                )

    train_rows = [*base_train, *added_rows, *reinforced_rows]
    eval_rows = base_eval
    all_rows = [*train_rows, *eval_rows]

    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    eval_path = args.output_dir / f"{args.prefix}_eval.jsonl"
    all_path = args.output_dir / f"{args.prefix}_all.jsonl"
    summary_path = args.report_dir / f"{args.prefix}_summary.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(all_path, all_rows)

    summary = {
        "prefix": args.prefix,
        "base_train": str(args.base_train),
        "base_eval": str(args.base_eval),
        "add_train": [str(path) for path in args.add_train],
        "reinforce_factor": args.reinforce_factor,
        "reinforce_exclude_schemas": sorted(excluded_schemas),
        "base_train_rows": len(base_train),
        "added_train_rows": len(added_rows),
        "reinforced_train_rows": len(reinforced_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "all_rows": len(all_rows),
        "skipped_eval_overlap": skipped_eval_overlap,
        "skipped_duplicate_add": skipped_duplicate_add,
        "schema_counts_train": dict(Counter(schema_of(row) for row in train_rows)),
        "schema_counts_eval": dict(Counter(schema_of(row) for row in eval_rows)),
        "slot_label_counts_train": dict(slot_label_counts(train_rows)),
        "outputs": {
            "train": str(train_path),
            "eval": str(eval_path),
            "all": str(all_path),
            "summary": str(summary_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

