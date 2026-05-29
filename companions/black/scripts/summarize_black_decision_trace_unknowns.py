from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.config import DEFAULT_STATE_DB_PATH


SHARED_REPORTS_DIR = ROOT.parent / "reports"
DEFAULT_JSON_OUT = SHARED_REPORTS_DIR / "black_decision_trace_unknowns_20260421.json"
DEFAULT_MD_OUT = SHARED_REPORTS_DIR / "black_decision_trace_unknowns_20260421.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="black decision_trace에서 heuristic unknown/ask_clarification 사례를 요약합니다."
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_STATE_DB_PATH)
    parser.add_argument("--classifier-source", default="heuristic")
    parser.add_argument("--intent", default="unknown")
    parser.add_argument("--action", default="ask_clarification")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    try:
        cursor = conn.execute(
            """
            SELECT
              id,
              created_at,
              user_id,
              input_text,
              input_intent,
              selected_action,
              selected_reason,
              classifier_evidence_json,
              output_text
            FROM decision_trace
            ORDER BY created_at ASC, id ASC
            """
        )
        for row in cursor:
            classifier_evidence = _load_json_object(row["classifier_evidence_json"])
            rows.append(
                {
                    "id": int(row["id"]),
                    "created_at": str(row["created_at"]),
                    "user_id": str(row["user_id"]),
                    "input_text": str(row["input_text"] or ""),
                    "input_intent": str(row["input_intent"] or ""),
                    "selected_action": str(row["selected_action"] or ""),
                    "selected_reason": str(row["selected_reason"] or ""),
                    "classifier_source": str(classifier_evidence.get("source") or "unknown"),
                    "classifier_reason": str(classifier_evidence.get("chosen_reason") or ""),
                    "reply": str(row["output_text"] or ""),
                }
            )
    finally:
        conn.close()
    return rows


def _load_json_object(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "null")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def summarize(
    rows: list[dict[str, Any]],
    *,
    classifier_source: str,
    intent: str,
    action: str,
) -> dict[str, Any]:
    filtered = [
        row
        for row in rows
        if row["classifier_source"] == classifier_source
        and row["input_intent"] == intent
        and row["selected_action"] == action
    ]

    top_inputs = Counter(row["input_text"] for row in filtered)
    top_replies = Counter(row["reply"] for row in filtered)
    top_classifier_reasons = Counter(row["classifier_reason"] for row in filtered)
    top_selected_reasons = Counter(row["selected_reason"] for row in filtered)

    return {
        "total_rows": len(rows),
        "filter": {
            "classifier_source": classifier_source,
            "intent": intent,
            "action": action,
        },
        "matched_rows": len(filtered),
        "top_inputs": [{"text": text, "count": count} for text, count in top_inputs.most_common(20)],
        "top_replies": [{"text": text, "count": count} for text, count in top_replies.most_common(20)],
        "top_classifier_reasons": [{"reason": text, "count": count} for text, count in top_classifier_reasons.most_common(10)],
        "top_selected_reasons": [{"reason": text, "count": count} for text, count in top_selected_reasons.most_common(10)],
        "rows": filtered,
    }


def write_outputs(summary: dict[str, Any], *, json_out: Path, md_out: Path, source_db: Path) -> None:
    summary["source_db"] = str(source_db)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Black Decision Trace Unknowns",
        "",
        f"- source db: `{source_db}`",
        f"- total rows: `{summary['total_rows']}`",
        f"- matched rows: `{summary['matched_rows']}`",
        f"- filter: `classifier={summary['filter']['classifier_source']}` / `intent={summary['filter']['intent']}` / `action={summary['filter']['action']}`",
        "",
        "## Top Inputs",
    ]
    if summary["top_inputs"]:
        for item in summary["top_inputs"]:
            lines.append(f"- `{item['count']}` · `{item['text']}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Top Replies"])
    if summary["top_replies"]:
        for item in summary["top_replies"]:
            lines.append(f"- `{item['count']}` · `{item['text']}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Rows"])
    if summary["rows"]:
        for row in summary["rows"]:
            lines.extend(
                [
                    f"### id {row['id']}",
                    f"- created_at: `{row['created_at']}`",
                    f"- input: `{row['input_text']}`",
                    f"- selected_reason: `{row['selected_reason']}`",
                    f"- classifier_reason: `{row['classifier_reason']}`",
                    f"- reply: `{row['reply']}`",
                    "",
                ]
            )
    else:
        lines.append("- none")

    md_out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.source_db)
    summary = summarize(
        rows,
        classifier_source=args.classifier_source,
        intent=args.intent,
        action=args.action,
    )
    write_outputs(summary, json_out=args.json_out, md_out=args.md_out, source_db=args.source_db)
    print(json.dumps(
        {
            "source_db": str(args.source_db),
            "total_rows": summary["total_rows"],
            "matched_rows": summary["matched_rows"],
            "json_out": str(args.json_out),
            "md_out": str(args.md_out),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
