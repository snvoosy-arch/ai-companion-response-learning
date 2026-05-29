#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("<repo>")
SOURCE = ROOT / "predictive-discord-bot" / "data" / "examples" / "daily_conversation_examples_384.jsonl"
OUT_JSON = ROOT / "predictive-discord-bot" / "data" / "black_alt_question_benchmark_examples100_20260422.json"
OUT_MD = ROOT / "reports" / "black_alt_question_benchmark_examples100_20260422.md"
TARGET_PER_COMBO = 5
TARGET_TOTAL = 100
SEED = 20260422


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def main() -> None:
    rng = random.Random(SEED)
    rows = _load_rows(SOURCE)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        combo = (row["labels"]["intent"], row["labels"]["action"])
        grouped[combo].append(row)

    selected: list[dict] = []
    for combo in sorted(grouped):
        candidates = list(grouped[combo])
        rng.shuffle(candidates)
        for row in candidates[:TARGET_PER_COMBO]:
            selected.append(
                {
                    "case_id": "",
                    "text": row["input"],
                    "bucket": f"{combo[0]}__{combo[1]}",
                    "expected_intent": combo[0],
                    "expected_action": combo[1],
                    "source_state": row.get("state", {}),
                    "target_reply": row.get("target_reply", ""),
                }
            )

    combo_counts = Counter((row["expected_intent"], row["expected_action"]) for row in selected)
    while len(selected) > TARGET_TOTAL:
        removable_counts = sorted(combo_counts.values(), reverse=True)
        max_count = removable_counts[0]
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

    selected.sort(key=lambda item: (item["expected_intent"], item["text"]))
    for index, row in enumerate(selected, start=1):
        row["case_id"] = f"BAX{index:03d}"

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    combo_counts = Counter((row["expected_intent"], row["expected_action"]) for row in selected)
    lines = [
        "# Black Alt Question Benchmark From Examples (2026-04-22)",
        "",
        f"- source: `{SOURCE}`",
        f"- seed: `{SEED}`",
        f"- selected: `{len(selected)}`",
        f"- target_per_combo: `{TARGET_PER_COMBO}`",
        "",
        "## Combo Counts",
    ]
    for (intent, action), count in sorted(combo_counts.items()):
        lines.append(f"- `{intent} -> {action}`: `{count}`")
    lines.extend(["", "## Sample Cases"])
    for row in selected[:15]:
        lines.extend(
            [
                f"- `{row['case_id']}` [{row['bucket']}] `{row['text']}`",
                f"  expected: `{row['expected_intent']} -> {row['expected_action']}`",
            ]
        )

    OUT_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    printable_combo_counts = {
        f"{intent}::{action}": count
        for (intent, action), count in sorted(combo_counts.items())
    }
    print(json.dumps({"selected": len(selected), "combo_counts": printable_combo_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
