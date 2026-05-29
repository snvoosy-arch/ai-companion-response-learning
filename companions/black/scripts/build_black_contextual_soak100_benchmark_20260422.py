#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("<repo>")
SOURCE = ROOT / "predictive-discord-bot" / "data" / "runtime_soak_real_logs_eval_rebased.jsonl"
OUT_JSON = ROOT / "predictive-discord-bot" / "data" / "black_contextual_soak100_benchmark_20260422.json"
OUT_MD = ROOT / "reports" / "black_contextual_soak100_benchmark_20260422.md"
TARGET_TOTAL = 100
TARGET_PER_COMBO = 5
SEED = 20260422


def _load_sessions(path: Path) -> list[dict]:
    sessions: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sessions.append(json.loads(line))
    return sessions


def _build_case(*, session: dict, turn_index: int) -> dict:
    turns = session["turns"]
    target = turns[turn_index]
    expect = target.get("expect", {})
    history = []
    for prior in turns[:turn_index]:
        history.append({"text": prior.get("input", "")})
    return {
        "case_id": "",
        "bucket": f"{expect.get('intent')}__{expect.get('action')}",
        "session_id": session.get("session_id"),
        "source_user_alias": session.get("source_user_alias"),
        "category": session.get("category"),
        "description": session.get("description"),
        "history_len": len(history),
        "history": history,
        "text": target.get("input", ""),
        "expected_intent": expect.get("intent"),
        "expected_action": expect.get("action"),
        "expected_speech_act": expect.get("speech_act"),
        "expected_topic_hint": expect.get("topic_hint"),
        "expected_reason_prefixes": list(expect.get("reason_code_prefixes") or []),
        "expected_logic_prefixes": list(expect.get("logic_rule_prefixes") or []),
        "meta": target.get("meta", {}),
    }


def main() -> None:
    rng = random.Random(SEED)
    sessions = _load_sessions(SOURCE)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for session in sessions:
        turns = session.get("turns") or []
        for turn_index, turn in enumerate(turns):
            expect = turn.get("expect", {})
            combo = (expect.get("intent"), expect.get("action"))
            grouped[combo].append(_build_case(session=session, turn_index=turn_index))

    selected: list[dict] = []
    for combo in sorted(grouped):
        candidates = list(grouped[combo])
        rng.shuffle(candidates)
        selected.extend(candidates[:TARGET_PER_COMBO])

    combo_counts = Counter((row["expected_intent"], row["expected_action"]) for row in selected)
    selected_keys = {
        (row["session_id"], row["history_len"], row["text"], row["expected_intent"], row["expected_action"])
        for row in selected
    }
    if len(selected) < TARGET_TOTAL:
        leftovers: list[dict] = []
        for combo in sorted(grouped):
            for row in grouped[combo]:
                key = (
                    row["session_id"],
                    row["history_len"],
                    row["text"],
                    row["expected_intent"],
                    row["expected_action"],
                )
                if key not in selected_keys:
                    leftovers.append(row)
        rng.shuffle(leftovers)
        while leftovers and len(selected) < TARGET_TOTAL:
            candidate = leftovers.pop()
            selected.append(candidate)
            combo = (candidate["expected_intent"], candidate["expected_action"])
            combo_counts[combo] += 1

    while len(selected) > TARGET_TOTAL:
        max_count = max(combo_counts.values())
        heavy_combos = [combo for combo, count in combo_counts.items() if count == max_count]
        chosen_combo = rng.choice(heavy_combos)
        for index in range(len(selected) - 1, -1, -1):
            row = selected[index]
            combo = (row["expected_intent"], row["expected_action"])
            if combo == chosen_combo:
                selected.pop(index)
                combo_counts[combo] -= 1
                if combo_counts[combo] <= 0:
                    del combo_counts[combo]
                break

    selected.sort(key=lambda item: (item["expected_intent"], item["history_len"], item["text"]))
    for index, row in enumerate(selected, start=1):
        row["case_id"] = f"BCS{index:03d}"

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history_counts = Counter(row["history_len"] for row in selected)
    combo_counts = Counter((row["expected_intent"], row["expected_action"]) for row in selected)
    lines = [
        "# Black Contextual Soak Benchmark 100 (2026-04-22)",
        "",
        f"- source: `{SOURCE}`",
        f"- seed: `{SEED}`",
        f"- selected: `{len(selected)}`",
        f"- target_per_combo: `{TARGET_PER_COMBO}`",
        "",
        "## History Length Counts",
    ]
    for history_len, count in sorted(history_counts.items()):
        lines.append(f"- `history_len={history_len}`: `{count}`")
    lines.extend(["", "## Combo Counts"])
    for (intent, action), count in sorted(combo_counts.items()):
        lines.append(f"- `{intent} -> {action}`: `{count}`")
    lines.extend(["", "## Sample Cases"])
    for row in selected[:15]:
        lines.append(
            f"- `{row['case_id']}` [{row['bucket']}] history=`{row['history_len']}` text=`{row['text']}`"
        )
    OUT_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "selected": len(selected),
                "history_counts": dict(history_counts),
                "combo_counts": {
                    f"{intent}::{action}": count
                    for (intent, action), count in sorted(combo_counts.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
