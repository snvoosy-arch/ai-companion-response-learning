from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "runtime_soak_real_logs_report_kobart_kcbert_wikidata.json"
DEFAULT_OUTPUT = ROOT / "data" / "runtime_soak_duplicate_reply_followup.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "runtime_soak_duplicate_reply_followup_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract duplicate_reply follow-up examples from the real-log soak report.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--context-turns", type=int, default=2)
    return parser.parse_args()


def detect_pattern(reply: str) -> str:
    normalized = reply.strip()
    if any(token in normalized for token in ("위치", "도시")):
        return "location_clarification_loop"
    if any(token in normalized for token in ("다시 한 번", "잘 모르겠어")):
        return "unknown_rephrase_loop"
    if any(token in normalized for token in ("설명해줘", "한 줄만 더", "풀어줘", "애매해")):
        return "clarification_overuse"
    return "generic_duplicate_reply"


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()
    reply_counter: Counter[str] = Counter()

    for session in report.get("sessions", []):
        turns = session.get("turns", [])
        for index, turn in enumerate(turns):
            warnings = turn.get("warnings", [])
            if "duplicate_reply" not in warnings:
                continue

            snapshot = turn.get("snapshot", {})
            reply = str(snapshot.get("reply", "")).strip()
            pattern = detect_pattern(reply)
            context_start = max(0, index - args.context_turns)
            context: list[dict[str, Any]] = []
            for past_turn in turns[context_start:index]:
                past_snapshot = past_turn.get("snapshot", {})
                context.append(
                    {
                        "turn_index": past_turn.get("turn_index"),
                        "input": past_turn.get("input"),
                        "reply": past_snapshot.get("reply"),
                        "action": past_snapshot.get("action"),
                    }
                )

            row = {
                "session_id": session.get("session_id"),
                "category": session.get("category"),
                "turn_index": turn.get("turn_index"),
                "input": turn.get("input"),
                "reply": reply,
                "action": snapshot.get("action"),
                "decision_module": snapshot.get("decision_module"),
                "pattern": pattern,
                "warnings": warnings,
                "reason_codes": snapshot.get("reason_codes", []),
                "context": context,
            }
            rows.append(row)

            category_counter[str(session.get("category", "unknown"))] += 1
            action_counter[str(snapshot.get("action", "unknown"))] += 1
            pattern_counter[pattern] += 1
            reply_counter[reply] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_report": str(args.report),
        "output": str(args.output),
        "count": len(rows),
        "category_counts": dict(sorted(category_counter.items())),
        "action_counts": dict(sorted(action_counter.items())),
        "pattern_counts": dict(sorted(pattern_counter.items())),
        "top_replies": [{"reply": reply, "count": count} for reply, count in reply_counter.most_common(10)],
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
