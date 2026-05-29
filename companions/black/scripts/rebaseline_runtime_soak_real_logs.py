from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_SOURCE_DATASET = ROOT / "data" / "runtime_soak_real_logs_eval.jsonl"
DEFAULT_SOURCE_REPORT = ROOT / "reports" / "runtime_soak_real_logs_report_kobart_kcbert_wikidata_v3.json"
DEFAULT_OUTPUT_DATASET = ROOT / "data" / "runtime_soak_real_logs_eval_rebased.jsonl"
DEFAULT_SUMMARY = ROOT / "reports" / "runtime_soak_real_logs_rebaseline_summary.json"

from predictive_bot.evaluation.runtime_logs import snapshot_to_expectation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="runtime soak real-log dataset 기대값을 현재 보고서 스냅샷으로 재기준화합니다.")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output-dataset", type=Path, default=DEFAULT_OUTPUT_DATASET)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_expectations(old_expect: dict[str, Any], new_expect: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for key in sorted(set(old_expect) | set(new_expect)):
        if old_expect.get(key) != new_expect.get(key):
            changed.append(key)
    return changed


def build_rebased_dataset(
    source_sessions: list[dict[str, Any]],
    report_sessions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(source_sessions) != len(report_sessions):
        raise ValueError(
            f"session count mismatch: dataset={len(source_sessions)} report={len(report_sessions)}"
        )

    rebased_sessions: list[dict[str, Any]] = []
    changed_turns = 0
    changed_field_counts: Counter[str] = Counter()
    session_diffs: list[dict[str, Any]] = []

    for source_session, report_session in zip(source_sessions, report_sessions):
        source_session_id = str(source_session.get("session_id", ""))
        report_session_id = str(report_session.get("session_id", ""))
        if source_session_id != report_session_id:
            raise ValueError(
                f"session_id mismatch: dataset={source_session_id!r} report={report_session_id!r}"
            )

        source_turns = list(source_session.get("turns", []))
        report_turns = list(report_session.get("turns", []))
        if len(source_turns) != len(report_turns):
            raise ValueError(
                f"turn count mismatch for {source_session_id}: dataset={len(source_turns)} report={len(report_turns)}"
            )

        rebased_session = copy.deepcopy(source_session)
        rebased_turns: list[dict[str, Any]] = []
        turn_diffs: list[dict[str, Any]] = []

        for turn_index, (source_turn, report_turn) in enumerate(zip(source_turns, report_turns), start=1):
            snapshot = dict(report_turn.get("snapshot") or {})
            old_expect = dict(source_turn.get("expect") or {})
            new_expect = snapshot_to_expectation(snapshot)
            changed_fields = compare_expectations(old_expect, new_expect)

            if changed_fields:
                changed_turns += 1
                for field in changed_fields:
                    changed_field_counts[field] += 1
                turn_diffs.append(
                    {
                        "turn_index": turn_index,
                        "input": source_turn.get("input"),
                        "changed_fields": changed_fields,
                    }
                )

            rebased_turn = copy.deepcopy(source_turn)
            rebased_turn["expect"] = new_expect
            meta = dict(rebased_turn.get("meta") or {})
            meta.update(
                {
                    "rebaseline_source_report": str(report_session.get("session_id", "")),
                    "rebaseline_source_turn_index": turn_index,
                    "rebaseline_changed_fields": changed_fields,
                }
            )
            rebased_turn["meta"] = meta
            rebased_turns.append(rebased_turn)

        rebased_session["turns"] = rebased_turns
        rebased_sessions.append(rebased_session)

        if turn_diffs:
            session_diffs.append(
                {
                    "session_id": source_session_id,
                    "category": source_session.get("category", "uncategorized"),
                    "turn_count": len(turn_diffs),
                    "sample_turns": turn_diffs[:5],
                }
            )

    summary = {
        "source_dataset": "",
        "source_report": "",
        "rebased_sessions": len(rebased_sessions),
        "rebased_turns": sum(len(session.get("turns", [])) for session in rebased_sessions),
        "changed_turns": changed_turns,
        "unchanged_turns": sum(len(session.get("turns", [])) for session in rebased_sessions) - changed_turns,
        "changed_field_counts": dict(sorted(changed_field_counts.items())),
        "session_diffs": session_diffs[:20],
    }
    return rebased_sessions, summary


def main() -> None:
    args = parse_args()
    source_sessions = load_jsonl(args.source_dataset)
    report = load_report(args.source_report)
    report_sessions = list(report.get("sessions", []))

    rebased_sessions, summary = build_rebased_dataset(source_sessions, report_sessions)
    summary["source_dataset"] = str(args.source_dataset)
    summary["source_report"] = str(args.source_report)
    summary["output_dataset"] = str(args.output_dataset)

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    with args.output_dataset.open("w", encoding="utf-8") as handle:
        for session in rebased_sessions:
            handle.write(json.dumps(session, ensure_ascii=False) + "\n")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
