from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.config import DEFAULT_STATE_DB_PATH
from predictive_bot.core.policy_features import (
    build_group_key,
    is_probably_clean_user_text,
    render_policy_feature_text,
)


DEFAULT_SOURCE_DB = DEFAULT_STATE_DB_PATH
DEFAULT_OUTPUT = ROOT / "data" / "policy_trace_dataset.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "policy_trace_dataset_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQLite decision_trace를 policy action 학습용 JSONL로 변환합니다."
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def load_rows(db_path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows: list[dict[str, str]] = []
    stats = {"loaded": 0, "dropped_unclean": 0, "dropped_duplicate_groups": 0}
    seen_groups: set[str] = set()

    try:
        cursor = conn.execute(
            """
            SELECT
              decision_id,
              input_text,
              input_intent,
              input_sentiment,
              selected_action,
              constraints_json,
              evidence_json,
              world_state_snapshot_json
            FROM decision_trace
            ORDER BY id ASC
            """
        )
        for row in cursor:
            stats["loaded"] += 1
            input_text = row["input_text"] or ""
            if not is_probably_clean_user_text(input_text):
                stats["dropped_unclean"] += 1
                continue

            group = build_group_key(
                input_text=input_text,
                input_intent=row["input_intent"],
                selected_action=row["selected_action"],
            )
            if group in seen_groups:
                stats["dropped_duplicate_groups"] += 1
                continue
            seen_groups.add(group)

            snapshot = json.loads(row["world_state_snapshot_json"] or "{}")
            constraints = json.loads(row["constraints_json"] or "[]")
            evidence = json.loads(row["evidence_json"] or "[]")

            text = render_policy_feature_text(
                input_text=input_text,
                input_intent=row["input_intent"],
                input_sentiment=row["input_sentiment"],
                conversation_mode=snapshot.get("conversation_mode"),
                user_emotion=snapshot.get("user_emotion"),
                risk_level=snapshot.get("risk_level"),
                unresolved_need=snapshot.get("unresolved_need"),
                factuality_required=snapshot.get("factuality_required"),
                turn_count_bucket=snapshot.get("turn_count_bucket"),
                tension_bucket=snapshot.get("tension_bucket"),
                last_intent_hint=snapshot.get("last_intent_hint"),
                last_action_hint=snapshot.get("last_action_hint"),
                constraints=constraints,
                evidence=evidence,
            )
            rows.append(
                {
                    "text": text,
                    "intent": row["selected_action"],
                    "decision_id": row["decision_id"],
                    "group": group,
                }
            )
    finally:
        conn.close()

    return rows, stats


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(rows: list[dict[str, str]], stats: dict[str, int]) -> dict[str, object]:
    action_counts = Counter(row["intent"] for row in rows)
    group_counts = Counter(row["group"] for row in rows)
    return {
        "loaded_records": stats["loaded"],
        "records": len(rows),
        "dropped_unclean": stats["dropped_unclean"],
        "dropped_duplicate_groups": stats["dropped_duplicate_groups"],
        "unique_groups": len(group_counts),
        "action_counts": dict(sorted(action_counts.items())),
        "duplicate_groups": [
            {"group": group, "count": count}
            for group, count in group_counts.items()
            if count > 1
        ][:20],
    }


def main() -> None:
    args = parse_args()
    rows, stats = load_rows(args.source_db)
    write_rows(args.output, rows)
    summary = build_summary(rows, stats)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
