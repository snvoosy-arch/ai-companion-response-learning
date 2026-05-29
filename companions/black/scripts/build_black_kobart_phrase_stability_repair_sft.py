from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = ROOT / "data" / "kobart_phrase_stability_repair_seed_20260417.json"
ALL_PATH = ROOT / "data" / "kobart_phrase_stability_repair_all_20260417.jsonl"
TRAIN_PATH = ROOT / "data" / "kobart_phrase_stability_repair_train_20260417.jsonl"
EVAL_PATH = ROOT / "data" / "kobart_phrase_stability_repair_eval_20260417.jsonl"
SUMMARY_PATH = ROOT / "reports" / "kobart_phrase_stability_repair_summary_20260417.json"

SEED = 42
EVAL_RATIO = 0.25

ACTION_RULES = {
    "continue_conversation": "reply like ongoing small talk, ask one light follow-up at most",
    "share_feeling": "reply with light emotional support, do not evaluate choices or preferences",
    "share_opinion": "give a simple, direct opinion or judgment in one or two short sentences, do not comfort emotionally or ask how the user feels",
}

SCHEMA_RULES = {
    "preference_disclosure": [
        "- answer by revealing your own preference directly",
        "- keep one concrete topic anchor from the user's question",
    ],
    "habit_preference": [
        "- answer whether you tend to do that often or not",
        "- sound like a casual self-description rather than advice",
    ],
    "self_style": [
        "- if asked about your style, answer with one concrete opener or way you would actually start",
        "- do not drift into abstract mood talk",
    ],
    "reflective_judgment": [
        "- choose a side briefly and conditionally",
        "- do not mirror the whole question back",
    ],
    "process_advice": [
        "- start with the first thing to check or narrow down",
        "- keep the advice procedural, not emotional",
    ],
    "soft_decision_advice": [
        "- give conditional advice about whether to do it",
        "- keep the reply practical and do not turn it into comfort",
    ],
    "weather_conditioned_activity_opinion": [
        "- treat the weather phrase as the user's premise, not a factual weather lookup",
        "- keep the activity anchor and do not ask for location",
    ],
}


def build_runtime_style_prompt(*, facts: dict) -> str:
    user_text = str(facts.get("user_text", "")).strip()
    action = str(facts.get("action", "")).strip()
    reason = str(facts.get("reason", "")).strip()
    reason_code = str(facts.get("reason_code", "")).strip()
    question_schema = str(facts.get("question_schema", "")).strip()
    action_rule = ACTION_RULES.get(action, "reply naturally and follow the action exactly")

    world_state = facts.get("world_state") or {}
    intent = str(world_state.get("dominant_intent") or "").strip()
    context = str(world_state.get("memory_summary") or "").strip()
    raw_constraints = world_state.get("constraints") or []
    constraints = [str(item).strip() for item in raw_constraints if str(item).strip()]
    schema_rules = SCHEMA_RULES.get(question_schema, [])

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
    elif opener == "bridging":
        phrasing_lines.append("- bridge from the user's wording naturally")

    if question_mode == "none" or not asks_followup or "no_followup" in constraints:
        phrasing_lines.append("- do not add a follow-up question")
        phrasing_lines.append("- do not end the reply with a question mark")
    if closer == "soft_close":
        phrasing_lines.append("- end softly so the user can stop there")
    if distance == "soft":
        phrasing_lines.append("- keep some distance and avoid sounding overly close")
    elif distance == "steady":
        phrasing_lines.append("- keep the tone steady and plain")
    if "avoid_self_insertion" in constraints:
        phrasing_lines.append("- do not turn the reply into your own mood or state with phrases like '나는 ...'")
    if "direct_opinion_only" in constraints:
        phrasing_lines.append("- answer the opinion directly without hedging or bouncing it back")
    if "self_style_anchor" in constraints:
        phrasing_lines.append("- if asked about your own style, answer with one concrete opener you would actually say")
    if "avoid_repetition" in constraints:
        phrasing_lines.append("- avoid repeating the same subject or time word across both sentences")
    if "avoid_weather_restatement" in constraints:
        phrasing_lines.append("- do not waste the reply by merely restating that it is a weather day")
    phrasing_lines.extend(schema_rules)

    rules_block = "\n".join(
        [
            "- write natural Korean only",
            "- one or two short sentences",
            "- keep at least one concrete topic word from the user message",
            "- avoid stock opener like '그럴 때 있지' or '그럴 수 있지'",
            "- avoid repeated token fragments such as '몸이 몸이'",
            "- do not shift the topic into body/mind/metaphor unless the user already did",
            *phrasing_lines,
        ]
    )

    return (
        "task: discord_reply\n"
        "persona: black_casual\n"
        f"intent: {intent}\n"
        f"action: {action}\n"
        f"question_schema: {question_schema or 'none'}\n"
        f"reason_code: {reason_code or 'none'}\n"
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
    parser = argparse.ArgumentParser(description="phrase-stability focused black KoBART repair 데이터를 만든다.")
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
            "question_schema": (
                str(row.get("question_schema") or "").strip()
                or str((row.get("meta") or {}).get("schema") or "").strip()
                or str(row.get("category") or "").strip()
            ),
            "reason_code": (
                str(row.get("decision_reason_code") or "").strip()
                or str((row.get("meta") or {}).get("decision_reason_code") or "").strip()
            ),
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
                    "question_schema": (
                        str(row.get("question_schema") or "").strip()
                        or str((row.get("meta") or {}).get("schema") or "").strip()
                        or str(row.get("category") or "").strip()
                    ),
                    "decision_reason_code": (
                        str(row.get("decision_reason_code") or "").strip()
                        or str((row.get("meta") or {}).get("decision_reason_code") or "").strip()
                    ),
                    "memory_summary": row.get("memory_summary") or "",
                    "constraints": row.get("constraints") or [],
                    "phrasing_plan": row.get("phrasing_plan") or {},
                    "source_type": "kobart_phrase_stability_repair",
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
