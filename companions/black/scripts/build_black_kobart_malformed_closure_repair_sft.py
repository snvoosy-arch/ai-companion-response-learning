from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = ROOT / "data" / "kobart_malformed_closure_repair_seed_20260417.json"
ALL_PATH = ROOT / "data" / "kobart_malformed_closure_repair_all_20260417.jsonl"
TRAIN_PATH = ROOT / "data" / "kobart_malformed_closure_repair_train_20260417.jsonl"
EVAL_PATH = ROOT / "data" / "kobart_malformed_closure_repair_eval_20260417.jsonl"
SUMMARY_PATH = ROOT / "reports" / "kobart_malformed_closure_repair_summary_20260417.json"

SEED = 42
EVAL_RATIO = 0.25

ACTION_RULES = {
    "continue_conversation": "reply like ongoing small talk, ask one light follow-up at most",
    "share_feeling": "reply with light emotional support, do not evaluate choices or preferences",
}


def build_runtime_style_prompt(*, facts: dict) -> str:
    user_text = str(facts.get("user_text", "")).strip()
    action = str(facts.get("action", "")).strip()
    reason = str(facts.get("reason", "")).strip()
    action_rule = ACTION_RULES.get(action, "reply naturally and follow the action exactly")

    world_state = facts.get("world_state") or {}
    intent = str(world_state.get("dominant_intent") or "").strip()
    context = str(world_state.get("memory_summary") or "").strip()
    raw_constraints = world_state.get("constraints") or []
    constraints = [str(item).strip() for item in raw_constraints if str(item).strip()]

    phrasing_plan = facts.get("phrasing_plan") or {}
    opener = str(phrasing_plan.get("opener") or "").strip()
    question_mode = str(phrasing_plan.get("question_mode") or "").strip()
    closer = str(phrasing_plan.get("closer") or "").strip()
    distance = str(phrasing_plan.get("distance") or "").strip()
    asks_followup = bool(phrasing_plan.get("asks_followup"))
    notes = phrasing_plan.get("notes") or []
    note_text = ", ".join(str(item).strip() for item in notes if str(item).strip()) or "none"

    phrasing_lines: list[str] = []
    if opener == "grounded":
        phrasing_lines.append("- open in a grounded, concrete way")
    elif opener == "brief":
        phrasing_lines.append("- keep the opening brief and plain")
    if question_mode == "none" or not asks_followup or "no_followup" in constraints:
        phrasing_lines.append("- do not add a follow-up question")
    if closer == "soft_close":
        phrasing_lines.append("- end softly without generic stock tail")
    if distance == "soft":
        phrasing_lines.append("- keep some distance and avoid sounding overly close")
    elif distance == "steady":
        phrasing_lines.append("- keep the tone steady and plain")

    rules_block = "\n".join(
        [
            "- write natural Korean only",
            "- one or two short sentences",
            "- keep at least one concrete topic word from the user message",
            "- avoid endings like '그런 날이 있더라' and '처음인 거구나'",
            "- avoid typo-like wording such as '기대를기보다', '많지어도', '더렷'",
            "- do not close with generic emotional filler",
            *phrasing_lines,
        ]
    )

    return (
        "task: discord_reply\n"
        "persona: black_casual\n"
        f"intent: {intent}\n"
        f"action: {action}\n"
        f"action_rule: {action_rule}\n"
        f"context: {context}\n"
        f"user: {user_text}\n"
        f"reason: {reason}\n"
        "phrasing_plan:"
        f" opener={opener},"
        f" question_mode={question_mode},"
        f" closer={closer},"
        f" distance={distance},"
        f" asks_followup={str(asks_followup).lower()},"
        f" notes={note_text}\n"
        "rules:\n"
        f"{rules_block}\n"
        "reply:"
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
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="malformed-closure focused black KoBART repair 데이터를 만든다.")
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
        facts = {
            "user_text": row["user_text"],
            "action": row["action"],
            "reason": row["reason"],
            "phrasing_plan": row.get("phrasing_plan") or {},
            "world_state": {
                "dominant_intent": row.get("intent") or "",
                "memory_summary": row.get("memory_summary") or "",
                "constraints": row.get("constraints") or [],
            },
        }
        prompt = build_runtime_style_prompt(facts=facts)
        converted.append(
            {
                "prompt": prompt,
                "completion": row["completion"].strip(),
                "meta": {
                    "id": row["id"],
                    "category": row.get("category") or "",
                    "user_text": row["user_text"],
                    "action": row["action"],
                    "intent": row.get("intent") or "",
                    "memory_summary": row.get("memory_summary") or "",
                    "constraints": row.get("constraints") or [],
                    "phrasing_plan": row.get("phrasing_plan") or {},
                    "source_type": "kobart_malformed_closure_repair",
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
