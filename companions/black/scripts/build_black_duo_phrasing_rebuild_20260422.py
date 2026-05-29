from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN = ROOT / "data" / "black_broad_phrasing_rebuild_train_20260422.jsonl"
BASE_EVAL = ROOT / "data" / "black_broad_phrasing_rebuild_eval_20260422.jsonl"
DUO_TRAIN = ROOT / "data" / "black_overlay_duo_phrasing_rebuild_train_20260422.jsonl"
DUO_EVAL = ROOT / "data" / "black_overlay_duo_phrasing_rebuild_eval_20260422.jsonl"
OUT_ALL = ROOT / "data" / "black_broad_plus_duo_phrasing_rebuild_all_20260422.jsonl"
OUT_TRAIN = ROOT / "data" / "black_broad_plus_duo_phrasing_rebuild_train_20260422.jsonl"
OUT_EVAL = ROOT / "data" / "black_broad_plus_duo_phrasing_rebuild_eval_20260422.jsonl"
OUT_SUMMARY = ROOT / "reports" / "black_broad_plus_duo_phrasing_rebuild_summary_20260422.json"
OUT_MD = ROOT / "reports" / "black_broad_plus_duo_phrasing_rebuild_notes_20260422.md"


def load_jsonl(path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            meta = dict(obj.get("meta") or {})
            meta["split"] = split
            meta["source_file"] = path.name
            rows.append(
                {
                    "prompt": str(obj["prompt"]).strip(),
                    "completion": str(obj["completion"]).strip(),
                    "meta": meta,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "prompt": row["prompt"],
                        "completion": row["completion"],
                        "meta": row["meta"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    rows = []
    rows.extend(load_jsonl(BASE_TRAIN, "train"))
    rows.extend(load_jsonl(BASE_EVAL, "eval"))
    rows.extend(load_jsonl(DUO_TRAIN, "train"))
    rows.extend(load_jsonl(DUO_EVAL, "eval"))

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["prompt"], row["completion"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    train_rows = [row for row in deduped if row["meta"].get("split") == "train"]
    eval_rows = [row for row in deduped if row["meta"].get("split") == "eval"]

    write_jsonl(OUT_TRAIN, train_rows)
    write_jsonl(OUT_EVAL, eval_rows)
    write_jsonl(OUT_ALL, train_rows + eval_rows)

    category_counts = Counter(str((row.get("meta") or {}).get("category") or "unknown") for row in deduped)
    failure_counts = Counter(str((row.get("meta") or {}).get("failure_type") or "none") for row in deduped)
    source_counts = Counter(str((row.get("meta") or {}).get("source_file") or "unknown") for row in deduped)
    summary = {
        "total_rows": len(deduped),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "category_counts": category_counts,
        "failure_counts": failure_counts,
        "source_counts": source_counts,
        "train_path": str(OUT_TRAIN),
        "eval_path": str(OUT_EVAL),
        "all_path": str(OUT_ALL),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Black Broad + Duo Phrasing Rebuild",
        "",
        "Merged broad phrasing rebuild v2 data with overlay duo-specific continue_conversation repairs.",
        "",
        f"- total rows: `{len(deduped)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        "",
        "## Category counts",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"- `{category}`: `{count}`")
    lines.append("")
    lines.append("## Failure counts")
    for failure, count in sorted(failure_counts.items()):
        lines.append(f"- `{failure}`: `{count}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
