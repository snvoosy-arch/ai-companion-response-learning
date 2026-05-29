from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("<repo>/companions/black")
BASE_ALL = ROOT / "data" / "black_broad_plus_duo_overlay20_rebuild_all_20260422.jsonl"
PRACTICAL_CASES = ROOT / "data" / "black_overlay_practical_cases_20260420.json"
UNFORCED_REPORT = ROOT / "reports" / "black_overlay_practical_reeval_overlay20_v3_unforced_20260422.json"
OUT_ALL = ROOT / "data" / "black_broad_plus_duo_overlay20_practical_repair_v4_all_20260422.jsonl"
OUT_TRAIN = ROOT / "data" / "black_broad_plus_duo_overlay20_practical_repair_v4_train_20260422.jsonl"
OUT_EVAL = ROOT / "data" / "black_broad_plus_duo_overlay20_practical_repair_v4_eval_20260422.jsonl"
OUT_SUMMARY = ROOT / "reports" / "black_overlay_practical_repair_v4_summary_20260422.json"
OUT_MD = ROOT / "reports" / "black_overlay_practical_repair_v4_notes_20260422.md"


def _norm(text: str) -> str:
    return " ".join(str(text).strip().split())


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" in obj and "completion" in obj:
                rows.append(
                    {
                        "instruction": "",
                        "input": _norm(obj["prompt"]),
                        "output": _norm(obj["completion"]),
                        "category": obj.get("meta", {}).get("category", "unknown"),
                        "failure_tag": obj.get("failure_tag", "none"),
                        "source": obj.get("meta", {}).get("source_file", obj.get("source", "unknown")),
                        "split": obj.get("meta", {}).get("split", obj.get("split", "train")),
                    }
                )
            else:
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "prompt": _norm(row.get("input", "")),
                        "completion": _norm(row.get("output", "")),
                        "meta": {
                            "category": row.get("category", "unknown"),
                            "split": row.get("split", "train"),
                            "source_file": row.get("source", "unknown"),
                            "failure_tag": row.get("failure_tag", "none"),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _manual_row(*, prompt: str, chosen: str, split: str) -> dict:
    return {
        "instruction": (
            "task: discord_reply persona: black_casual intent: smalltalk_feeling "
            "action: share_feeling question_schema: practical_repair_v4 "
            "reason_code: feeling.share.reflective"
        ),
        "input": _norm(prompt),
        "output": _norm(chosen),
        "category": "overlay_practical_repair_v4",
        "failure_tag": "llm_topic_drift",
        "source": "overlay_practical_unforced_v3_20260422",
        "split": split,
    }


def build_manual_rows() -> list[dict]:
    cases = {item["label"]: item for item in json.loads(PRACTICAL_CASES.read_text(encoding="utf-8"))}
    report = {item["label"]: item for item in json.loads(UNFORCED_REPORT.read_text(encoding="utf-8"))["results"]}
    chosen_map = {
        "weather_feeling": ("비 오는 날엔 괜히 톤이 더 내려가긴 하지. 오늘은 조용한 쪽이 더 맞겠다.", "train"),
        "social_awkwardness": ("아 그거 은근 오래 남지. 한 번 어색해지면 괜히 계속 신경 쓰이잖아.", "train"),
        "low_energy_checkin": ("응, 오늘은 좀 조용한 쪽이 더 편하겠다.", "train"),
        "quiet_good_news": ("오, 생각보다 잘 풀려서 다행이네. 막 들뜨진 않아도 마음은 좀 놓였겠다.", "eval"),
    }
    rows: list[dict] = []
    for label, (chosen, split) in chosen_map.items():
        if label not in cases or label not in report:
            continue
        rows.append(_manual_row(prompt=cases[label]["text"], chosen=chosen, split=split))
    return rows


def main() -> None:
    base_rows = _load_jsonl(BASE_ALL)
    manual_rows = build_manual_rows()
    all_rows = list(base_rows) + manual_rows

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in all_rows:
        key = (_norm(row.get("input", "")), _norm(row.get("output", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    train_rows = [row for row in deduped if row.get("split", "train") != "eval"]
    eval_rows = [row for row in deduped if row.get("split") == "eval"]
    _write_jsonl(OUT_ALL, deduped)
    _write_jsonl(OUT_TRAIN, train_rows)
    _write_jsonl(OUT_EVAL, eval_rows)

    category_counts = Counter(str(row.get("category", "unknown")) for row in deduped)
    source_counts = Counter(str(row.get("source", "unknown")) for row in deduped)
    summary = {
        "total_rows": len(deduped),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "manual_rows_added": len(manual_rows),
        "category_counts": category_counts,
        "source_counts": source_counts,
        "all_path": str(OUT_ALL),
        "train_path": str(OUT_TRAIN),
        "eval_path": str(OUT_EVAL),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Black Overlay Practical Repair V4 Notes",
        "",
        f"- total rows: `{len(deduped)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        f"- manual rows added: `{len(manual_rows)}`",
        "",
        "## Category Counts",
        "",
    ]
    for key, value in sorted(category_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Source Counts", ""])
    for key, value in sorted(source_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    OUT_MD.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
