from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "daily_response_rewritten_sft_v10_all.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "black_character_corpus_phase_a1_candidates_20260419.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "black_character_corpus_phase_a1_candidates_20260419.json"

ALLOWED_ACTION_SCORES = {
    "continue_conversation": 5,
    "share_feeling": 5,
    "share_opinion": 4,
    "small_talk": 3,
    "acknowledge": 3,
    "react_surprise": 3,
    "react_laugh": 2,
}

EXCLUDED_ACTIONS = {
    "explain_capabilities",
    "ask_location",
    "recommend",
    "search_answer",
    "weather_lookup",
    "music_chat",
    "answer_identity",
    "ask_clarification",
    "game_chat",
}

ALLOWED_INTENT_SCORES = {
    "smalltalk_generic": 4,
    "smalltalk_feeling": 4,
    "smalltalk_opinion": 3,
    "greeting": 2,
    "surprise": 2,
    "laugh": 1,
    "reply_request": 2,
}

EXCLUDED_INTENTS = {
    "help",
    "game_talk",
    "confirm",
    "weather",
    "media_recommend",
    "search_request",
    "provide_location",
    "music",
    "who_are_you",
    "hostile",
}

ALLOWED_REPLY_STYLES = {
    "casual_comment": 3,
    "light_reaction": 2,
    "brief_ack": 2,
    "plain_greeting": 1,
    "light_greeting": 1,
}

ALLOWED_REPLY_FOCUS = {"default", "preference_probe"}

BANNED_SUBSTRINGS = (
    "지금 받는다",
    "일단 받는다",
    "들어온 건 봤다",
    "바로 본다",
    "바로 보고 있다",
    "보고 있었지",
    "접속 확인",
    "지금부터 받는다",
    "설명해줄게",
    "검색해볼게",
    "위치 알려줘",
    "추천해줄게",
    "능력은",
)

SOFT_BANNED_SUBSTRINGS = (
    "오냐",
    "얘기 던져",
    "하나만 꺼내",
    "봤다",
)

BROKEN_SUFFIXES = ("거든", "라", "지", "야", "임")

EXCLUDED_TOPICS = {
    "building",
    "coop",
    "creeper",
    "death",
    "inventory",
    "low_food",
    "mining",
    "nether",
    "opening",
    "quiet_stream",
}

ALLOWED_PROMPT_STYLE_SCORES = {
    "none": 2,
    "light_system": 2,
    "scene_guided": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A-1용 black character corpus 후보 필터")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=160)
    parser.add_argument("--per-action-cap", type=int, default=40)
    parser.add_argument("--per-intent-cap", type=int, default=28)
    parser.add_argument("--per-topic-cap", type=int, default=8)
    parser.add_argument("--per-emotion-cap", type=int, default=16)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sentence_count(text: str) -> int:
    chunks = [chunk.strip() for chunk in re.split(r"[.!?…]+", text) if chunk.strip()]
    return max(1, len(chunks))


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() == "user":
            return normalize_text(str(message.get("content", "")))
    return ""


def last_assistant_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() == "assistant":
            return normalize_text(str(message.get("content", "")))
    return ""


def score_row(row: dict) -> tuple[int, list[str]] | None:
    meta = row.get("meta") or row.get("metadata") or {}
    completion = normalize_text(row.get("completion", ""))
    user_text = normalize_text(meta.get("user_text", ""))
    action = str(meta.get("action", "")).strip()
    intent = str(meta.get("intent", "")).strip()
    reply_style = str(meta.get("reply_style", "")).strip()
    reply_focus = str(meta.get("reply_focus", "default")).strip()

    if not completion or not user_text:
        return None
    if action in EXCLUDED_ACTIONS or intent in EXCLUDED_INTENTS:
        return None
    if action not in ALLOWED_ACTION_SCORES:
        return None
    if intent not in ALLOWED_INTENT_SCORES:
        return None

    reasons: list[str] = []
    score = 0

    score += ALLOWED_ACTION_SCORES[action]
    reasons.append(f"action:{action}")
    score += ALLOWED_INTENT_SCORES[intent]
    reasons.append(f"intent:{intent}")

    if reply_style in ALLOWED_REPLY_STYLES:
        score += ALLOWED_REPLY_STYLES[reply_style]
        reasons.append(f"style:{reply_style}")
    else:
        score -= 1

    if reply_focus in ALLOWED_REPLY_FOCUS:
        score += 1
    else:
        score -= 2

    length = len(completion)
    if 10 <= length <= 55:
        score += 3
    elif 56 <= length <= 72:
        score += 1
    else:
        score -= 4

    sent_count = sentence_count(completion)
    if sent_count <= 2:
        score += 2
    else:
        score -= 3

    if completion.count("?") + completion.count("？") <= 1:
        score += 1
    else:
        score -= 2

    if any(bad in completion for bad in BANNED_SUBSTRINGS):
        return None
    for soft_bad in SOFT_BANNED_SUBSTRINGS:
        if soft_bad in completion:
            score -= 3

    if re.search(r"[A-Za-z]{4,}", completion):
        score -= 2
    if re.search(r"\d{2,}", completion):
        score -= 1
    if "..." in completion or "ㅋㅋㅋㅋ" in completion:
        score -= 2
    if any(completion.endswith(suffix) for suffix in BROKEN_SUFFIXES):
        return None
    if re.search(r"(는데|했는데|었는데)\s+(거든|야|지|라|임)$", completion):
        return None

    if "너는" in completion or "너 쪽은" in completion or "어땠냐" in completion or "어때" in completion:
        score += 1
    if "마음" in completion and "너무" in completion:
        score -= 1

    return score, reasons


def score_character_messages_row(row: dict) -> tuple[int, list[str]] | None:
    meta = row.get("meta") or row.get("metadata") or {}
    messages = row.get("messages") or []
    completion = last_assistant_text(messages)
    user_text = last_user_text(messages)
    topic = str(meta.get("topic", "")).strip()
    emotion = str(meta.get("emotion", "")).strip()
    prompt_style = str(meta.get("prompt_style", "none")).strip()
    turn_count = int(meta.get("turn_count", 0) or 0)

    if not completion or not user_text:
        return None
    if str(meta.get("language", "")).strip() not in {"", "ko"}:
        return None
    if topic in EXCLUDED_TOPICS:
        return None

    reasons: list[str] = []
    score = 0

    if topic:
        score += 2
        reasons.append(f"topic:{topic}")
    if emotion:
        score += 1
        reasons.append(f"emotion:{emotion}")

    score += ALLOWED_PROMPT_STYLE_SCORES.get(prompt_style, 0)
    reasons.append(f"style:{prompt_style or 'none'}")

    if turn_count in {2, 3}:
        score += 2
    elif turn_count == 4:
        score += 1
    elif turn_count >= 5:
        score -= 1

    length = len(completion)
    if 10 <= length <= 58:
        score += 3
    elif 59 <= length <= 72:
        score += 1
    else:
        score -= 3

    sent_count = sentence_count(completion)
    if sent_count <= 2:
        score += 2
    else:
        score -= 2

    if completion.count("?") + completion.count("？") <= 1:
        score += 1
    else:
        score -= 2

    if any(bad in completion for bad in BANNED_SUBSTRINGS):
        return None
    for soft_bad in SOFT_BANNED_SUBSTRINGS:
        if soft_bad in completion:
            score -= 3

    if completion.startswith(("오케이", "좋아,")):
        score -= 1
    if re.search(r"[A-Za-z]{4,}", completion):
        score -= 2
    if re.search(r"\d{2,}", completion):
        score -= 1
    if "..." in completion or "ㅋㅋㅋㅋ" in completion:
        score -= 2
    if any(completion.endswith(suffix) for suffix in BROKEN_SUFFIXES):
        return None
    if re.search(r"(는데|했는데|었는데)\s+(거든|야|지|라|임)$", completion):
        return None
    if completion.endswith("네") and len(completion) <= 4:
        return None

    if "너" in completion or "넌" in completion:
        score += 1
    if completion.endswith(("지.", "지", "네.", "네", "거구나.", "거구나")):
        score += 1

    return score, reasons


def extract_row_payload(row: dict) -> dict | None:
    if "prompt" in row and "completion" in row:
        meta = row.get("meta") or row.get("metadata") or {}
        return {
            "kind": "legacy_prompt_completion",
            "prompt": row.get("prompt", ""),
            "completion": normalize_text(row.get("completion", "")),
            "meta": meta,
            "dedupe_key": normalize_text(row.get("completion", "")),
            "action": str(meta.get("action", "")).strip(),
            "intent": str(meta.get("intent", "")).strip(),
            "topic": str(meta.get("topic", "")).strip(),
            "emotion": str(meta.get("emotion", "")).strip(),
        }

    if "messages" in row and isinstance(row.get("messages"), list):
        messages = row.get("messages") or []
        prompt = last_user_text(messages)
        completion = last_assistant_text(messages)
        meta = row.get("meta") or row.get("metadata") or {}
        return {
            "kind": "messages_character_style",
            "messages": messages,
            "prompt": prompt,
            "completion": completion,
            "meta": meta,
            "dedupe_key": f"{prompt}\n{completion}",
            "action": "",
            "intent": "",
            "topic": str(meta.get("topic", "")).strip(),
            "emotion": str(meta.get("emotion", "")).strip(),
        }

    return None


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield line_no, row


def main() -> None:
    args = parse_args()
    candidates: list[dict] = []

    for line_no, row in iter_rows(args.input):
        payload = extract_row_payload(row)
        if not payload:
            continue

        if payload["kind"] == "legacy_prompt_completion":
            scored = score_row(row)
        else:
            scored = score_character_messages_row(row)
        if not scored:
            continue
        score, reasons = scored
        meta = payload["meta"]
        candidates.append(
            {
                "line_no": line_no,
                "score": score,
                "reasons": reasons,
                "prompt": payload["prompt"],
                "completion": payload["completion"],
                "meta": meta,
                "dedupe_key": payload["dedupe_key"],
                "kind": payload["kind"],
                "row": row,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["meta"].get("action", ""),
            item["meta"].get("intent", ""),
            item["meta"].get("topic", ""),
            item["completion"],
        )
    )

    seen_completion: set[str] = set()
    action_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    emotion_counts: Counter[str] = Counter()
    selected: list[dict] = []

    for item in candidates:
        completion_key = normalize_text(item["dedupe_key"])
        action = str(item["meta"].get("action", "")).strip()
        intent = str(item["meta"].get("intent", "")).strip()
        topic = str(item["meta"].get("topic", "")).strip()
        emotion = str(item["meta"].get("emotion", "")).strip()

        if completion_key in seen_completion:
            continue
        if item["kind"] == "legacy_prompt_completion":
            if action_counts[action] >= args.per_action_cap:
                continue
            if intent_counts[intent] >= args.per_intent_cap:
                continue
        else:
            if topic and topic_counts[topic] >= args.per_topic_cap:
                continue
            if emotion and emotion_counts[emotion] >= args.per_emotion_cap:
                continue

        seen_completion.add(completion_key)
        if item["kind"] == "legacy_prompt_completion":
            action_counts[action] += 1
            intent_counts[intent] += 1
        else:
            topic_counts[topic] += 1
            emotion_counts[emotion] += 1
        selected.append(item)
        if len(selected) >= args.limit:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            output_row = dict(item["row"])
            output_row["selection_meta"] = {
                "score": item["score"],
                "reasons": item["reasons"],
                "line_no": item["line_no"],
                "source": str(args.input),
                "filter_kind": item["kind"],
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.input),
        "candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "limit": args.limit,
        "per_action_cap": args.per_action_cap,
        "per_intent_cap": args.per_intent_cap,
        "per_topic_cap": args.per_topic_cap,
        "per_emotion_cap": args.per_emotion_cap,
        "selected_action_counts": dict(action_counts),
        "selected_intent_counts": dict(intent_counts),
        "selected_topic_counts": dict(topic_counts),
        "selected_emotion_counts": dict(emotion_counts),
        "top_examples": [
            {
                "score": item["score"],
                "action": item["meta"].get("action"),
                "intent": item["meta"].get("intent"),
                "topic": item["meta"].get("topic"),
                "emotion": item["meta"].get("emotion"),
                "user_text": item["meta"].get("user_text"),
                "prompt": item["prompt"],
                "completion": item["completion"],
            }
            for item in selected[:10]
        ],
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source={args.input}")
    print(f"candidate_rows={len(candidates)}")
    print(f"selected_rows={len(selected)}")
    print(f"saved_output={args.output}")
    print(f"saved_report={args.report}")


if __name__ == "__main__":
    main()
