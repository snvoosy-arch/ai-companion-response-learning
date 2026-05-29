from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ORIGINAL_DATASET = ROOT / "data" / "runtime_soak_real_logs_eval.jsonl"
DEFAULT_REBASED_DATASET = ROOT / "data" / "runtime_soak_real_logs_eval_rebased.jsonl"
DEFAULT_ORIGINAL_REPORT = ROOT / "reports" / "runtime_soak_real_logs_report_kobart_kcbert_wikidata_v3.json"
DEFAULT_REBASED_REPORT = ROOT / "reports" / "runtime_soak_real_logs_report_rebased.json"
DEFAULT_REPORT_OUT = ROOT / "reports" / "runtime_soak_real_logs_drift_breakdown.json"
DEFAULT_MD_OUT = ROOT / "reports" / "runtime_soak_real_logs_drift_breakdown_2026-04-13.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="runtime soak real-log rebaseline drift를 요약합니다.")
    parser.add_argument("--original-dataset", type=Path, default=DEFAULT_ORIGINAL_DATASET)
    parser.add_argument("--rebased-dataset", type=Path, default=DEFAULT_REBASED_DATASET)
    parser.add_argument("--original-report", type=Path, default=DEFAULT_ORIGINAL_REPORT)
    parser.add_argument("--rebased-report", type=Path, default=DEFAULT_REBASED_REPORT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_turn_pairs(
    original_sessions: list[dict[str, Any]],
    rebased_sessions: list[dict[str, Any]],
):
    if len(original_sessions) != len(rebased_sessions):
        raise ValueError(
            f"session count mismatch: original={len(original_sessions)} rebased={len(rebased_sessions)}"
        )

    for original_session, rebased_session in zip(original_sessions, rebased_sessions):
        if original_session.get("session_id") != rebased_session.get("session_id"):
            raise ValueError(
                f"session_id mismatch: original={original_session.get('session_id')!r} "
                f"rebased={rebased_session.get('session_id')!r}"
            )

        original_turns = list(original_session.get("turns", []))
        rebased_turns = list(rebased_session.get("turns", []))
        if len(original_turns) != len(rebased_turns):
            raise ValueError(
                f"turn mismatch for {original_session.get('session_id')}: "
                f"original={len(original_turns)} rebased={len(rebased_turns)}"
            )

        for turn_index, (original_turn, rebased_turn) in enumerate(zip(original_turns, rebased_turns), start=1):
            yield {
                "session_id": original_session.get("session_id"),
                "category": original_session.get("category", "uncategorized"),
                "turn_index": turn_index,
                "input": original_turn.get("input"),
                "original_expect": dict(original_turn.get("expect") or {}),
                "rebased_expect": dict(rebased_turn.get("expect") or {}),
            }


def compare_fields(original_expect: dict[str, Any], rebased_expect: dict[str, Any]) -> dict[str, Any]:
    diffs: dict[str, Any] = {}
    all_keys = sorted(set(original_expect) | set(rebased_expect))
    for key in all_keys:
        original_value = original_expect.get(key)
        rebased_value = rebased_expect.get(key)
        if original_value != rebased_value:
            diffs[key] = {
                "original": original_value,
                "rebased": rebased_value,
            }
    return diffs


def summarize(original_sessions: list[dict[str, Any]], rebased_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    field_counts: collections.Counter[str] = collections.Counter()
    intent_pairs: collections.Counter[str] = collections.Counter()
    action_pairs: collections.Counter[str] = collections.Counter()
    decision_module_pairs: collections.Counter[str] = collections.Counter()
    reason_code_prefix_pairs: collections.Counter[str] = collections.Counter()
    logic_rule_prefix_pairs: collections.Counter[str] = collections.Counter()
    sample_drifts: list[dict[str, Any]] = []

    total_turns = 0
    changed_turns = 0

    for pair in iter_turn_pairs(original_sessions, rebased_sessions):
        total_turns += 1
        diffs = compare_fields(pair["original_expect"], pair["rebased_expect"])
        if not diffs:
            continue

        changed_turns += 1
        for field in diffs:
            field_counts[field] += 1

        original_intent = str(pair["original_expect"].get("intent", ""))
        rebased_intent = str(pair["rebased_expect"].get("intent", ""))
        original_action = str(pair["original_expect"].get("action", ""))
        rebased_action = str(pair["rebased_expect"].get("action", ""))
        original_module = str(pair["original_expect"].get("decision_module", ""))
        rebased_module = str(pair["rebased_expect"].get("decision_module", ""))

        if original_intent != rebased_intent:
            intent_pairs[f"{original_intent}->{rebased_intent}"] += 1
        if original_action != rebased_action:
            action_pairs[f"{original_action}->{rebased_action}"] += 1
        if original_module != rebased_module:
            decision_module_pairs[f"{original_module}->{rebased_module}"] += 1

        original_reason_prefixes = tuple(_prefixes(pair["original_expect"].get("reason_code_prefixes")))
        rebased_reason_prefixes = tuple(_prefixes(pair["rebased_expect"].get("reason_code_prefixes")))
        if original_reason_prefixes != rebased_reason_prefixes:
            reason_code_prefix_pairs[f"{_short_join(original_reason_prefixes)}->{_short_join(rebased_reason_prefixes)}"] += 1

        original_logic_prefixes = tuple(_prefixes(pair["original_expect"].get("logic_rule_prefixes")))
        rebased_logic_prefixes = tuple(_prefixes(pair["rebased_expect"].get("logic_rule_prefixes")))
        if original_logic_prefixes != rebased_logic_prefixes:
            logic_rule_prefix_pairs[f"{_short_join(original_logic_prefixes)}->{_short_join(rebased_logic_prefixes)}"] += 1

        if len(sample_drifts) < 25:
            sample_drifts.append(
                {
                    "session_id": pair["session_id"],
                    "category": pair["category"],
                    "turn_index": pair["turn_index"],
                    "input": pair["input"],
                    "changed_fields": sorted(diffs),
                    "original": {
                        "intent": pair["original_expect"].get("intent"),
                        "action": pair["original_expect"].get("action"),
                        "decision_module": pair["original_expect"].get("decision_module"),
                        "reason_code_prefixes": list(_prefixes(pair["original_expect"].get("reason_code_prefixes"))),
                    },
                    "rebased": {
                        "intent": pair["rebased_expect"].get("intent"),
                        "action": pair["rebased_expect"].get("action"),
                        "decision_module": pair["rebased_expect"].get("decision_module"),
                        "reason_code_prefixes": list(_prefixes(pair["rebased_expect"].get("reason_code_prefixes"))),
                    },
                }
            )

    return {
        "total_turns": total_turns,
        "changed_turns": changed_turns,
        "unchanged_turns": total_turns - changed_turns,
        "changed_turn_ratio": round(changed_turns / max(1, total_turns), 4),
        "field_counts": dict(sorted(field_counts.items())),
        "top_intent_pairs": _top_items(intent_pairs),
        "top_action_pairs": _top_items(action_pairs),
        "top_decision_module_pairs": _top_items(decision_module_pairs),
        "top_reason_code_prefix_pairs": _top_items(reason_code_prefix_pairs),
        "top_logic_rule_prefix_pairs": _top_items(logic_rule_prefix_pairs),
        "sample_drifts": sample_drifts,
    }


def _prefixes(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        text = values.strip()
        return [text] if text else []
    if not isinstance(values, (list, tuple, set)):
        text = str(values).strip()
        return [text] if text else []
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            result.append(text)
    return result


def _short_join(values: tuple[str, ...]) -> str:
    if not values:
        return "[]"
    if len(values) <= 3:
        return "|".join(values)
    return "|".join(values[:3]) + "|..."


def _top_items(counter: collections.Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [{"pair": key, "count": count} for key, count in counter.most_common(limit)]


def write_md_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# runtime soak rebaseline drift breakdown",
        "",
        f"- total turns: {summary['total_turns']}",
        f"- changed turns: {summary['changed_turns']}",
        f"- changed turn ratio: {summary['changed_turn_ratio']}",
        "",
        "## top field drift",
    ]
    for field, count in sorted(summary["field_counts"].items(), key=lambda item: (-item[1], item[0]))[:10]:
        lines.append(f"- {field}: {count}")
    lines.extend(
        [
            "",
            "## top patterns",
            _format_top("intent", summary["top_intent_pairs"]),
            _format_top("action", summary["top_action_pairs"]),
            _format_top("decision_module", summary["top_decision_module_pairs"]),
            _format_top("reason_code_prefixes", summary["top_reason_code_prefix_pairs"]),
            _format_top("logic_rule_prefixes", summary["top_logic_rule_prefix_pairs"]),
            "",
            "## sample drifts",
        ]
    )
    for item in summary["sample_drifts"][:8]:
        lines.append(
            f"- [{item['session_id']}#{item['turn_index']}] {item['input']} -> {', '.join(item['changed_fields'][:6])}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_top(name: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return f"- {name}: none"
    top = items[0]
    return f"- {name}: {top['pair']} ({top['count']})"


def main() -> None:
    args = parse_args()
    original_sessions = load_jsonl(args.original_dataset)
    rebased_sessions = load_jsonl(args.rebased_dataset)
    original_report = load_json(args.original_report)
    rebased_report = load_json(args.rebased_report)

    summary = summarize(original_sessions, rebased_sessions)
    summary["original_report"] = str(args.original_report)
    summary["rebased_report"] = str(args.rebased_report)
    summary["original_report_session_count"] = original_report.get("session_count")
    summary["rebased_report_session_count"] = rebased_report.get("session_count")
    summary["original_report_turn_accuracy"] = original_report.get("turn_accuracy")
    summary["rebased_report_turn_accuracy"] = rebased_report.get("turn_accuracy")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md_report(args.md_out, summary)
    print(json.dumps(
        {
            "changed_turns": summary["changed_turns"],
            "changed_turn_ratio": summary["changed_turn_ratio"],
            "top_action_pairs": summary["top_action_pairs"][:3],
            "report_out": str(args.report_out),
            "md_out": str(args.md_out),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
