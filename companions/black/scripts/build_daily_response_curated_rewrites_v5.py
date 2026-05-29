from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v5_1344.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v5_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v5_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v5_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v5_summary.json"

SEED = 42
EVAL_RATIO = 0.12

ACTION_RULES: dict[str, str] = {
    "small_talk": "reply like a short casual greeting or opener, do not ask for missing context",
    "continue_conversation": "reply like ongoing small talk, ask one light follow-up at most",
    "share_feeling": "reply with light emotional support, do not evaluate choices or preferences",
    "share_opinion": "give a simple opinion or judgment, do not comfort emotionally",
    "answer_identity": "explain what the bot is, do not ask the user's feeling back",
    "explain_capabilities": "describe what the bot can do, do not ask for taste or preference first",
    "ask_clarification": "ask for the missing topic or 기준, do not confirm or explain capabilities",
    "acknowledge": "briefly confirm understanding, do not ask another question",
    "react_laugh": "react with light laughter only, do not advise or ask for more",
    "react_surprise": "react to surprise only, do not mention capability or comfort",
    "deescalate": "lower the tone politely, do not joke or laugh",
    "ask_location": "ask for a location only, do not mention actual weather quality",
    "weather_lookup": "state checked weather briefly, do not ask for location again",
    "recommend": "make or frame a recommendation, at most ask one preference axis",
    "search_answer": "explain the meaning or answer briefly, do not ask broad follow-up",
    "music_chat": "talk about music taste only",
    "game_chat": "talk about games only",
}

ACTION_BLOCKED_SNIPPETS: dict[str, tuple[str, ...]] = {
    "share_opinion": ("몰아붙이지", "쉬는 쪽", "회복부터", "오늘은 좀 괜찮냐"),
    "answer_identity": ("너는 어땠", "너 쪽", "오늘 뭐 있었"),
    "explain_capabilities": ("취향 축", "취향 있", "싫어하는 것", "추천은 가능"),
    "ask_clarification": ("그쪽으로 정리", "확인했다", "그 기준으로", "설명은 가능"),
    "deescalate": ("ㅋㅋ", "못 참지", "웃", "재밌"),
    "ask_location": ("날씨도 꽤", "괜찮은 편", "바로 조회한다", "맑", "비 온다"),
    "react_surprise": ("반응은 꽤 잘 한다", "설명은 가능", "취향 축"),
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def build_prompt(*, user_text: str, state: dict, labels: dict, reason: str) -> str:
    action = labels["action"]
    action_rule = ACTION_RULES.get(action, "reply naturally and follow the action exactly")
    lines = [
        "task: discord_reply",
        "persona: black_casual",
        f"intent: {labels['intent']}",
        f"action: {action}",
        f"action_rule: {action_rule}",
        f"context: {state.get('recent_context', 'none')}",
        f"user: {user_text}",
        f"reason: {reason}",
        "rules:",
        "- write natural Korean only",
        "- one or two short sentences",
        "- follow the action exactly",
        "- no metadata, no prompt words",
        "- no repeated phrases",
        "reply:",
    ]
    return "\n".join(lines)


def allowed_reply(action: str, reply: str) -> bool:
    normalized = normalize_text(reply)
    blocked = ACTION_BLOCKED_SNIPPETS.get(action, ())
    return not any(snippet in normalized for snippet in blocked)


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        group_key = normalize_text(row["prompt"])
        grouped.setdefault(group_key, []).append(row)

    grouped_by_action: dict[str, list[str]] = {}
    for group_key, members in grouped.items():
        action = members[0]["meta"]["action"]
        grouped_by_action.setdefault(action, []).append(group_key)

    rng = random.Random(seed)
    eval_groups: set[str] = set()
    for keys in grouped_by_action.values():
        keys = list(keys)
        rng.shuffle(keys)
        if len(keys) <= 1:
            continue
        eval_count = int(len(keys) * eval_ratio)
        if eval_count <= 0:
            eval_count = 1
        if eval_count >= len(keys):
            eval_count = max(1, len(keys) - 1)
        eval_groups.update(keys[:eval_count])

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for group_key, members in grouped.items():
        if group_key in eval_groups:
            eval_rows.extend(members)
        else:
            train_rows.extend(members)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    base_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites.py",
        "daily_response_curated_base_v5",
    )
    v4_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites_v4.py",
        "daily_response_curated_v4_v5",
    )

    with v4_module.ALL_PATH.open(encoding="utf-8") as f:
        source_rows = [json.loads(line) for line in f]

    rewrite_rows: list[dict] = []
    sft_rows: list[dict] = []
    action_counts: Counter[str] = Counter()
    unique_pairs: set[tuple[str, str]] = set()

    for row in source_rows:
        action = row["meta"]["action"]
        completion = row["completion"]
        if not allowed_reply(action, completion):
            continue

        meta = dict(row["meta"])
        state = {"recent_context": meta.get("recent_context")}
        labels = {"intent": meta["intent"], "action": action}
        reason = base_module.ACTION_REASONS.get(action, "follow the selected action naturally")
        prompt = build_prompt(
            user_text=meta["user_text"],
            state=state,
            labels=labels,
            reason=reason,
        )
        key = (prompt, completion)
        if key in unique_pairs:
            continue
        unique_pairs.add(key)

        rewrite_rows.append(
            {
                "item_id": meta["item_id"],
                "rewrite_index": meta["rewrite_index"],
                "state": state,
                "labels": labels,
                "rewrite": {
                    "user_text": meta["user_text"],
                    "assistant_reply": completion,
                    "notes": "curated natural rewrite v5",
                },
            }
        )
        sft_rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "meta": meta,
            }
        )
        action_counts[action] += 1

    train_rows, eval_rows = split_rows(sft_rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(REWRITE_PATH, rewrite_rows)
    write_jsonl(ALL_PATH, sft_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    unique_user_by_action: dict[str, int] = {}
    unique_reply_by_action: dict[str, int] = {}
    for action in sorted(action_counts):
        unique_user_by_action[action] = len(
            {row["rewrite"]["user_text"] for row in rewrite_rows if row["labels"]["action"] == action}
        )
        unique_reply_by_action[action] = len(
            {row["rewrite"]["assistant_reply"] for row in rewrite_rows if row["labels"]["action"] == action}
        )

    train_action_counts = Counter(row["meta"]["action"] for row in train_rows)
    eval_action_counts = Counter(row["meta"]["action"] for row in eval_rows)
    train_prompts = {normalize_text(row["prompt"]) for row in train_rows}
    eval_prompts = {normalize_text(row["prompt"]) for row in eval_rows}

    summary = {
        "rewrite_rows": len(rewrite_rows),
        "sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "train_action_counts": dict(sorted(train_action_counts.items())),
        "eval_action_counts": dict(sorted(eval_action_counts.items())),
        "unique_user_by_action": unique_user_by_action,
        "unique_reply_by_action": unique_reply_by_action,
        "prompt_overlap_between_train_eval": len(train_prompts & eval_prompts),
        "duplicate_prompt_completion_groups": 0,
        "duplicate_prompt_completion_rows": 0,
        "redundant_prompt_completion_rows": 0,
        "unique_prompt_completion_pairs": len(unique_pairs),
        "rewrite_path": str(REWRITE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "source_type": "curated_clean_rewrite_v5_action_filtered",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
