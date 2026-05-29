from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "black_broad_phrasing_rebuild_all_20260422.jsonl"
LIVE30_PATH = ROOT / "data" / "black_live30_repair_20260426_all.jsonl"

ALL_PATH = ROOT / "data" / "black_live30_mixed_repair_20260426_all.jsonl"
TRAIN_PATH = ROOT / "data" / "black_live30_mixed_repair_20260426_train.jsonl"
EVAL_PATH = ROOT / "data" / "black_live30_mixed_repair_20260426_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "black_live30_mixed_repair_20260426_summary.json"
NOTES_PATH = ROOT / "reports" / "black_live30_mixed_repair_20260426_notes.md"

SEED = 42
EVAL_RATIO = 0.12
LIVE30_WEIGHT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black active broad phrasing 데이터와 live30 repair 데이터를 섞은 KoBART SFT 세트를 만듭니다."
    )
    parser.add_argument("--base-path", type=Path, default=BASE_PATH)
    parser.add_argument("--live30-path", type=Path, default=LIVE30_PATH)
    parser.add_argument("--live30-weight", type=int, default=LIVE30_WEIGHT)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, *, source_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = str(payload.get("prompt") or "").strip()
            completion = str(payload.get("completion") or "").strip()
            if not prompt or not completion:
                continue
            meta = dict(payload.get("meta") or {})
            meta["source_file"] = path.name
            meta["source_group"] = source_name
            meta.setdefault("source_row", index)
            rows.append({"prompt": prompt, "completion": completion, "meta": meta})
    if not rows:
        raise RuntimeError(f"no usable rows found: {path}")
    return rows


def build_rows(*, base_path: Path, live30_path: Path, live30_weight: int) -> list[dict[str, object]]:
    if live30_weight < 1:
        raise ValueError("live30_weight must be >= 1")

    base_rows = load_jsonl(base_path, source_name="broad_phrasing_active_base")
    live30_rows = load_jsonl(live30_path, source_name="live30_repair")

    rows: list[dict[str, object]] = list(base_rows)
    for copy_index in range(1, live30_weight + 1):
        for row in live30_rows:
            weighted = {
                "prompt": row["prompt"],
                "completion": row["completion"],
                "meta": dict(row["meta"]),
            }
            weighted["meta"]["live30_weight_copy"] = copy_index
            rows.append(weighted)
    return rows


def split_rows(
    rows: list[dict[str, object]], *, eval_ratio: float, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    live_eval_keys = {"p04", "p09"}
    eval_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []

    for row in rows:
        meta = row.get("meta") or {}
        if meta.get("source_group") == "live30_repair" and meta.get("probe_id") in live_eval_keys and meta.get("live30_weight_copy") == 1:
            eval_rows.append(row)
        else:
            train_rows.append(row)

    shuffled_train = list(train_rows)
    random.Random(seed).shuffle(shuffled_train)
    target_eval_count = max(len(eval_rows), int(len(rows) * eval_ratio))
    needed = max(0, target_eval_count - len(eval_rows))
    eval_rows.extend(shuffled_train[:needed])
    train_rows = shuffled_train[needed:]
    if not train_rows:
        raise RuntimeError("train split is empty")
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
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
    args = parse_args()
    rows = build_rows(
        base_path=args.base_path,
        live30_path=args.live30_path,
        live30_weight=args.live30_weight,
    )
    train_rows, eval_rows = split_rows(rows, eval_ratio=args.eval_ratio, seed=args.seed)

    source_counts = Counter(str((row["meta"] or {}).get("source_group") or "unknown") for row in rows)
    live30_probe_counts = Counter(
        str((row["meta"] or {}).get("probe_id") or "unknown")
        for row in rows
        if (row["meta"] or {}).get("source_group") == "live30_repair"
    )
    issue_tag_counts = Counter(
        str(tag)
        for row in rows
        for tag in ((row["meta"] or {}).get("issue_tags") or [])
    )
    summary = {
        "base_path": str(args.base_path),
        "live30_path": str(args.live30_path),
        "live30_weight": args.live30_weight,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "source_counts": dict(sorted(source_counts.items())),
        "live30_probe_counts": dict(sorted(live30_probe_counts.items())),
        "issue_tag_counts": dict(sorted(issue_tag_counts.items())),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    write_jsonl(ALL_PATH, rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    NOTES_PATH.write_text(
        "\n".join(
            [
                "# Black live30 mixed repair 20260426",
                "",
                f"- base: `{args.base_path}`",
                f"- live30: `{args.live30_path}`",
                f"- live30 weight: `{args.live30_weight}`",
                f"- rows: `{summary['rows']}`",
                f"- train/eval: `{summary['train_rows']}` / `{summary['eval_rows']}`",
                "",
                "## Why",
                "The live30-only candidate learned the social-return anchor but destabilized normal emotional replies. This mixed set keeps the active broad phrasing base visible while giving live30 repairs extra weight.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
