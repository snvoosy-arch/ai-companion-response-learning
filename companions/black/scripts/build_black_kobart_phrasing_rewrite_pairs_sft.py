from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = ROOT / "data" / "kobart_black_phrasing_rewrite_pairs_20260419.json"
ALL_PATH = ROOT / "data" / "kobart_black_phrasing_rewrite_pairs_all_20260419.jsonl"
TRAIN_PATH = ROOT / "data" / "kobart_black_phrasing_rewrite_pairs_train_20260419.jsonl"
EVAL_PATH = ROOT / "data" / "kobart_black_phrasing_rewrite_pairs_eval_20260419.jsonl"
SUMMARY_PATH = ROOT / "reports" / "kobart_black_phrasing_rewrite_pairs_summary_20260419.json"

SEED = 42
EVAL_RATIO = 0.25

ACTION_RULES = {
    "continue_conversation": "reply like brief ongoing small talk, grounded and natural, ask at most one light follow-up",
    "share_feeling": "reply with light emotional support, do not over-explain or evaluate choices",
    "share_opinion": "give a short direct opinion, do not comfort in a white-like way",
}


def build_runtime_style_prompt(*, facts: dict) -> str:
    user_text = str(facts.get("user_text", "")).strip()
    action = str(facts.get("action", "")).strip()
    intent = str(facts.get("intent", "")).strip()
    weak_completion = str(facts.get("weak_completion", "")).strip()
    rewrite_focus = [str(item).strip() for item in facts.get("rewrite_focus", []) if str(item).strip()]
    focus_text = ", ".join(rewrite_focus) if rewrite_focus else "natural black phrasing"
    action_rule = ACTION_RULES.get(action, "reply naturally and follow the action exactly")

    return (
        "task: discord_reply_rewrite\n"
        "persona: black_casual\n"
        f"intent: {intent}\n"
        f"action: {action}\n"
        f"action_rule: {action_rule}\n"
        f"user: {user_text}\n"
        f"avoid_this_reply: {weak_completion}\n"
        f"rewrite_focus: {focus_text}\n"
        "rules:\n"
        "- write natural Korean only\n"
        "- keep it short and grounded\n"
        "- keep at least one concrete topic word from the user message\n"
        "- do not reuse malformed wording from avoid_this_reply\n"
        "- avoid stock tail like '그런 날이 있더라' or '처음인 거구나'\n"
        "- avoid sounding too soft or white-like\n"
        "- prefer firm-but-warm black phrasing\n"
        "reply:\n"
    )


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build black phrasing rewrite-pair SFT JSONL.")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--all-path", type=Path, default=ALL_PATH)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--eval-ratio", type=float, default=EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    source_rows = payload["items"] if isinstance(payload, dict) else payload

    converted: list[dict] = []
    for row in source_rows:
        prompt = build_runtime_style_prompt(
            facts={
                "user_text": row["user_text"],
                "action": row["action"],
                "intent": row.get("intent") or "",
                "weak_completion": row["weak_completion"],
                "rewrite_focus": row.get("rewrite_focus") or [],
            }
        )
        converted.append(
            {
                "prompt": prompt,
                "completion": row["target_completion"].strip(),
                "meta": {
                    "id": row["id"],
                    "category": row.get("category") or "",
                    "user_text": row["user_text"],
                    "action": row["action"],
                    "intent": row.get("intent") or "",
                    "weak_completion": row["weak_completion"],
                    "target_completion": row["target_completion"],
                    "rewrite_focus": row.get("rewrite_focus") or [],
                    "source_type": "kobart_black_phrasing_rewrite_pairs",
                },
            }
        )

    train_rows, eval_rows = split_rows(converted, eval_ratio=args.eval_ratio, seed=args.seed)
    write_jsonl(args.all_path, converted)
    write_jsonl(args.train_path, train_rows)
    write_jsonl(args.eval_path, eval_rows)

    category_counts: dict[str, int] = {}
    for row in source_rows:
        category = str(row.get("category") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

    summary = {
        "source": str(args.source),
        "rows": len(converted),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "category_counts": category_counts,
        "all_path": str(args.all_path),
        "train_path": str(args.train_path),
        "eval_path": str(args.eval_path),
        "sample": converted[:2],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
