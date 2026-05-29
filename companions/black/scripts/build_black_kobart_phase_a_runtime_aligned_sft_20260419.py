from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GOLDSEED_SOURCE = ROOT / "data" / "black_character_only_v3_4000_phase_a1_final_gold_20260419.jsonl"
FINAL_GOLD_PATH = ROOT / "data" / "black_character_only_v3_4000_phase_a1_final_gold_runtime_aligned_20260419.jsonl"
GOLDEN_REPLIES_PATH = ROOT / "data" / "kobart_black_phase_a_golden_replies_runtime_aligned_20260419.jsonl"
ALL_PATH = ROOT / "data" / "kobart_black_phase_a_runtime_aligned_all_20260419.jsonl"
TRAIN_PATH = ROOT / "data" / "kobart_black_phase_a_runtime_aligned_train_20260419.jsonl"
EVAL_PATH = ROOT / "data" / "kobart_black_phase_a_runtime_aligned_eval_20260419.jsonl"
SUMMARY_PATH = ROOT / "reports" / "kobart_black_phase_a_runtime_aligned_summary_20260419.json"

REPAIR_SOURCES = (
    ("phrasing_rewrite_pairs", ROOT / "data" / "kobart_black_phrasing_rewrite_pairs_all_20260419.jsonl", None),
    ("phrase_stability_repair", ROOT / "data" / "kobart_phrase_stability_repair_all_20260417.jsonl", None),
    ("stock_tail_repair", ROOT / "data" / "kobart_stock_tail_repair_all_20260417.jsonl", None),
    ("malformed_closure_repair", ROOT / "data" / "kobart_malformed_closure_repair_all_20260417.jsonl", None),
    ("topic_diversity_repair", ROOT / "data" / "kobart_topic_diversity_repair_all_20260417.jsonl", 48),
    ("memory_carryover_phrase_repair", ROOT / "data" / "kobart_memory_carryover_phrase_repair_all_20260416.jsonl", None),
    ("memory_carryover_repair", ROOT / "data" / "kobart_memory_carryover_repair_all_20260416.jsonl", None),
    ("comparison_bitterness_repair", ROOT / "data" / "kobart_comparison_bitterness_repair_all_20260415.jsonl", None),
    ("probe_repair", ROOT / "data" / "kobart_probe_repair_all_20260415.jsonl", None),
    ("phrasing_repair", ROOT / "data" / "kobart_phrasing_repair_all_20260415.jsonl", None),
)

ACTION_RULES = {
    "small_talk": "reply like a short casual greeting or opener, do not ask for missing context",
    "continue_conversation": "reply like ongoing small talk, ask one light follow-up at most",
    "share_feeling": "reply with light emotional support, do not evaluate choices or preferences",
    "share_opinion": "give a simple, direct opinion or judgment in one or two short sentences, do not comfort emotionally or ask how the user feels",
    "answer_identity": "explain what the bot is, do not ask the user's feeling back",
    "explain_capabilities": "describe what the bot can do, do not ask for taste or preference first",
    "ask_clarification": "ask for the missing topic or 기준, do not confirm or explain capabilities",
    "acknowledge": "briefly confirm understanding, do not ask another question",
    "react_laugh": "react with light laughter only, do not advise or ask for more",
    "react_surprise": "react to surprise only, do not mention capability or comfort",
    "deescalate": "lower the tone politely, do not joke or laugh",
}

PREFERENCE_TOPICS = {
    "apple_preference",
    "cats_dogs",
    "coffee_tea",
    "food_choice",
    "music_taste",
    "rain_preference",
    "season_preference",
}
SMALLTALK_TOPICS = {
    "what_doing",
    "weekend",
    "night",
    "sleep",
    "silent",
    "opening",
    "ending",
    "celebration",
}
FEELING_TOPICS = {
    "awkward",
    "bored",
    "late_reply",
    "low_mood",
    "mistake",
    "procrastination",
    "tired",
}
PLAYFUL_TOPICS = {"chat_tease"}

FEELING_EMOTIONS = {
    "awkward",
    "bored",
    "embarrassed",
    "nervous",
    "quiet",
    "restless",
    "sad",
    "soft",
    "stuck",
    "tired",
    "uneasy",
}
OPINION_EMOTIONS = {"curious", "light", "neutral"}
PLAYFUL_EMOTIONS = {"casual", "happy", "playful", "warmup"}

DEFAULT_EVAL_RATIO = 0.12
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build runtime-aligned Phase A integrated black KoBART SFT corpus.")
    parser.add_argument("--goldseed-source", type=Path, default=GOLDSEED_SOURCE)
    parser.add_argument("--final-gold-path", type=Path, default=FINAL_GOLD_PATH)
    parser.add_argument("--golden-replies-path", type=Path, default=GOLDEN_REPLIES_PATH)
    parser.add_argument("--all-path", type=Path, default=ALL_PATH)
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_ratio))
    eval_rows = shuffled[:eval_count]
    train_rows = shuffled[eval_count:]
    return train_rows, eval_rows


def last_message_text(messages: list[dict], role: str) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() == role:
            return normalize_text(str(message.get("content", "")))
    return ""


def first_system_text(messages: list[dict]) -> str:
    for message in messages:
        if str(message.get("role", "")).strip() == "system":
            return normalize_text(str(message.get("content", "")))
    return ""


def infer_action(meta: dict, user_text: str, completion: str) -> str:
    topic = str(meta.get("topic") or "").strip()
    emotion = str(meta.get("emotion") or "").strip()
    combined = f"{user_text} {completion}"

    if "ㅋㅋ" in combined or "ㅎㅎ" in combined:
        return "react_laugh"
    if any(token in combined for token in ("헐", "진짜?", "대박", "ㄹㅇ")):
        return "react_surprise"
    if topic in PLAYFUL_TOPICS or emotion in PLAYFUL_EMOTIONS:
        return "continue_conversation"
    if topic in PREFERENCE_TOPICS or emotion in OPINION_EMOTIONS:
        return "share_opinion"
    if topic in FEELING_TOPICS or emotion in FEELING_EMOTIONS:
        return "share_feeling"
    if topic in SMALLTALK_TOPICS:
        return "continue_conversation"
    return "continue_conversation"


def infer_intent(action: str, meta: dict, user_text: str) -> str:
    topic = str(meta.get("topic") or "").strip()
    if topic == "opening":
        return "greeting"
    if action == "share_feeling":
        return "smalltalk_feeling"
    if action == "share_opinion":
        return "smalltalk_opinion"
    if action == "react_laugh":
        return "laugh"
    if action == "react_surprise":
        return "surprise"
    if "?" in user_text:
        return "smalltalk_generic"
    return "smalltalk_generic"


def infer_constraints(meta: dict, system_text: str, action: str) -> list[str]:
    constraints: list[str] = []
    system_text = system_text or ""
    if "무례해지지 마" in system_text or "실제 사람처럼 편하게 반응해" in system_text:
        constraints.append("avoid_overfamiliarity")
    if action == "share_feeling" and ("quiet" in str(meta.get("emotion")) or "soft" in str(meta.get("emotion"))):
        constraints.append("no_followup")
    return constraints


def infer_reason(meta: dict, action: str, user_text: str, completion: str) -> str:
    topic = str(meta.get("topic") or "").strip() or "daily"
    emotion = str(meta.get("emotion") or "").strip() or "neutral"
    if action == "share_feeling":
        return f"{topic} / {emotion} 계열 발화라서 감정을 가볍게 받아주되 과하게 위로하지 않는 답이 자연스럽다."
    if action == "share_opinion":
        return f"{topic} / {emotion} 계열 발화라서 취향이나 판단을 짧고 선명하게 말하는 편이 적절하다."
    if action == "react_laugh":
        return f"{topic} 맥락이 가벼운 웃음 쪽이라 짧은 리액션으로 받는 편이 black 톤에 맞다."
    if action == "react_surprise":
        return f"{topic} 맥락이 놀람 리액션에 가까워서 짧고 선명한 반응이 자연스럽다."
    return f"{topic} / {emotion} 계열 잡담이라 가볍게 이어받고 필요한 경우만 짧게 다음 턴을 열어두는 답이 적절하다."


def infer_distance(meta: dict, system_text: str) -> str:
    emotion = str(meta.get("emotion") or "").strip()
    if emotion in {"playful", "happy", "casual"} or "텐션은 조금 더 선명하게 가" in system_text:
        return "playful"
    if emotion in {"soft", "quiet", "awkward", "sad", "restless", "tired"}:
        return "soft"
    return "steady"


def infer_opener(action: str, meta: dict) -> str:
    emotion = str(meta.get("emotion") or "").strip()
    if action in {"react_laugh", "react_surprise"}:
        return "reactive"
    if action == "share_opinion":
        return "informative"
    if action == "share_feeling":
        return "grounded" if emotion in {"sad", "tired", "restless", "quiet"} else "bridging"
    return "bridging" if emotion in {"awkward", "casual", "soft"} else "brief"


def infer_phrasing_plan(meta: dict, system_text: str, action: str, completion: str) -> dict:
    asks_followup = "?" in completion
    if asks_followup:
        question_mode = "soft"
        closer = "keep_open"
    elif action == "share_feeling":
        question_mode = "none"
        closer = "soft_close"
    else:
        question_mode = "none"
        closer = "none"

    notes = [str(meta.get("topic") or "daily"), str(meta.get("emotion") or "neutral")]
    prompt_style = str(meta.get("prompt_style") or "").strip()
    if prompt_style and prompt_style != "none":
        notes.append(prompt_style)

    return {
        "opener": infer_opener(action, meta),
        "question_mode": question_mode,
        "closer": closer,
        "distance": infer_distance(meta, system_text),
        "asks_followup": asks_followup,
        "notes": notes,
    }


def build_runtime_aligned_prompt(*, row: dict) -> tuple[str, dict]:
    messages = row.get("messages") or []
    meta = dict(row.get("meta") or {})
    user_text = last_message_text(messages, "user")
    completion = last_message_text(messages, "assistant")
    system_text = first_system_text(messages)

    action = infer_action(meta, user_text, completion)
    intent = infer_intent(action, meta, user_text)
    constraints = infer_constraints(meta, system_text, action)
    phrasing_plan = infer_phrasing_plan(meta, system_text, action, completion)
    reason = infer_reason(meta, action, user_text, completion)
    context_parts = [str(meta.get("scene") or "").strip(), str(meta.get("topic") or "").strip(), str(meta.get("emotion") or "").strip()]
    context = " ".join(part for part in context_parts if part)
    action_rule = ACTION_RULES.get(action, "reply naturally and follow the action exactly")

    constraint_lines: list[str] = []
    if "avoid_overfamiliarity" in constraints:
        constraint_lines.append("- avoid slang, overfriendly teasing, and casual fillers like ㅋㅋ / ㅎㅇ / 반가워~")
    if "respect_boundary_history" in constraints:
        constraint_lines.append("- do not push the user to continue or ask for one more line")

    phrasing_lines: list[str] = []
    opener = str(phrasing_plan["opener"])
    question_mode = str(phrasing_plan["question_mode"])
    closer = str(phrasing_plan["closer"])
    distance = str(phrasing_plan["distance"])
    asks_followup = bool(phrasing_plan["asks_followup"])
    notes = phrasing_plan["notes"]
    note_text = ", ".join(str(item).strip() for item in notes if str(item).strip()) or "none"

    if opener == "clarifying":
        phrasing_lines.append("- open by briefly marking that you are clarifying, not comforting")
    elif opener == "reactive":
        phrasing_lines.append("- open with a short reaction, not an explanation")
    elif opener == "informative":
        phrasing_lines.append("- open directly with the answer, not with empathy talk")
    elif opener == "brief":
        phrasing_lines.append("- open briefly and do not add a long second sentence")
    elif opener == "grounded":
        phrasing_lines.append("- sound grounded and concrete, not poetic or analytical")
    elif opener in {"bridging", "light", "warm"}:
        phrasing_lines.append("- keep the opening conversational and easy to continue")

    if question_mode == "none" or not asks_followup:
        phrasing_lines.append("- do not add a follow-up question")
    elif question_mode == "soft":
        phrasing_lines.append("- if you ask back, use only one soft follow-up question")
    elif question_mode == "direct":
        phrasing_lines.append("- if you ask back, use only one direct follow-up question")

    if closer == "soft_close":
        phrasing_lines.append("- end in a way that lets the user stop comfortably")
    elif closer == "keep_open":
        phrasing_lines.append("- leave the reply lightly open for the next turn")

    if distance == "playful":
        phrasing_lines.append("- keep it playful but still natural Korean")
    elif distance == "soft":
        phrasing_lines.append("- keep some distance and avoid sounding overly close")
    elif distance == "steady":
        phrasing_lines.append("- keep the tone steady and plain")

    rules_block = "\n".join(
        [
            "- write natural Korean only",
            "- one or two short sentences",
            "- follow the action exactly",
            "- no metadata or prompt words",
            "- no repeated phrases",
            "- keep at least one concrete topic word from the user's message in the reply",
            "- do not invent body/mind/metaphor wording unless the user already used that register",
            *constraint_lines,
            *phrasing_lines,
        ]
    )

    prompt = (
        "task: discord_reply\n"
        "persona: black_casual\n"
        "input_mode: full\n"
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
    derived_meta = {
        "intent": intent,
        "action": action,
        "memory_summary": context,
        "constraints": constraints,
        "phrasing_plan": phrasing_plan,
        "reason": reason,
    }
    return prompt, derived_meta


def convert_gold_to_sft(rows: list[dict]) -> list[dict]:
    converted: list[dict] = []
    for row in rows:
        messages = row.get("messages") or []
        completion = last_message_text(messages, "assistant")
        prompt, derived_meta = build_runtime_aligned_prompt(row=row)
        meta = dict(row.get("meta") or {})
        meta.update(
            {
                "id": row.get("id"),
                "category": row.get("category") or "character_style_sft",
                "source_type": "black_phase_a_golden_reply_runtime_aligned",
                **derived_meta,
            }
        )
        converted.append(
            {
                "prompt": prompt,
                "completion": completion,
                "meta": meta,
            }
        )
    return converted


def merge_integrated_rows(golden_rows: list[dict]) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    merged: list[dict] = []
    dedupe_keys: set[str] = set()
    source_counts: Counter[str] = Counter()
    skipped_duplicates: Counter[str] = Counter()

    def add_rows(rows: list[dict], *, source_name: str, limit: int | None = None) -> None:
        added = 0
        for row in rows:
            prompt = normalize_text(str(row.get("prompt", "")))
            completion = normalize_text(str(row.get("completion", "")))
            if not prompt or not completion:
                continue
            dedupe_key = f"{prompt}\n{completion}"
            if dedupe_key in dedupe_keys:
                skipped_duplicates[source_name] += 1
                continue
            merged.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "meta": dict(row.get("meta") or {}),
                }
            )
            dedupe_keys.add(dedupe_key)
            source_counts[source_name] += 1
            added += 1
            if limit is not None and added >= limit:
                break

    add_rows(golden_rows, source_name="golden_replies_runtime_aligned", limit=None)

    for source_name, path, limit in REPAIR_SOURCES:
        rows = list(iter_jsonl(path))
        add_rows(rows, source_name=source_name, limit=limit)

    return merged, dict(source_counts), dict(skipped_duplicates)


def main() -> None:
    args = parse_args()
    final_gold_rows = list(iter_jsonl(args.goldseed_source))
    if not (100 <= len(final_gold_rows) <= 140):
        raise RuntimeError(
            f"expected final gold size in 100~140 range, got {len(final_gold_rows)} from {args.goldseed_source}"
        )

    golden_rows = convert_gold_to_sft(final_gold_rows)
    integrated_rows, source_counts, skipped_duplicates = merge_integrated_rows(golden_rows)
    train_rows, eval_rows = split_rows(integrated_rows, eval_ratio=args.eval_ratio, seed=args.seed)

    write_jsonl(args.final_gold_path, final_gold_rows)
    write_jsonl(args.golden_replies_path, golden_rows)
    write_jsonl(args.all_path, integrated_rows)
    write_jsonl(args.train_path, train_rows)
    write_jsonl(args.eval_path, eval_rows)

    action_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    for row in golden_rows:
        meta = row.get("meta") or {}
        action_counts[str(meta.get("action") or "unknown")] += 1
        intent_counts[str(meta.get("intent") or "unknown")] += 1

    summary = {
        "goldseed_source": str(args.goldseed_source),
        "final_gold_path": str(args.final_gold_path),
        "golden_replies_path": str(args.golden_replies_path),
        "all_path": str(args.all_path),
        "train_path": str(args.train_path),
        "eval_path": str(args.eval_path),
        "final_gold_rows": len(final_gold_rows),
        "golden_replies_rows": len(golden_rows),
        "integrated_rows": len(integrated_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_ratio": args.eval_ratio,
        "gold_action_counts": dict(action_counts),
        "gold_intent_counts": dict(intent_counts),
        "source_counts": source_counts,
        "skipped_duplicates": skipped_duplicates,
        "repair_caps": {
            source_name: limit
            for source_name, _, limit in REPAIR_SOURCES
        },
        "sample": integrated_rows[:3],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
