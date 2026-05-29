from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v6_1344.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v6_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v6_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v6_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v6_summary.json"

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
    "share_opinion": ("모아붙이지", "되는 쪽", "더 공부", "오늘은 좀 괜찮"),
    "answer_identity": ("너는 이런", "어떤 쪽", "오늘 뭐했"),
    "explain_capabilities": ("취향 축", "취향 쪽", "무슨 하는 거", "추천은 가능"),
    "ask_clarification": ("기능 설명", "정리해준다", "그 기준으로", "설명은 가능"),
    "deescalate": ("그 톤면", "몇 번쯤", "ㅋㅋ", "농담"),
    "ask_location": ("날씨 쪽", "괜찮은 듯", "바로 조회한다", "맑", "비온다"),
    "react_surprise": ("반응을 봐준다", "설명은 가능", "취향 축"),
}

SOFT_HINTS = (
    "괜찮",
    "너무",
    "천천히",
    "몰아붙",
    "하루",
    "기분",
    "무리",
    "쉬어",
)
LIGHT_HINTS = ("ㅋㅋ", "ㄹㅇ", "헐", "오", "어이", "와", "웃기", "터짐")
DIRECT_HINTS = ("할 수", "가능", "정리", "답할", "먼저", "구조")
BOUNDARY_HINTS = ("차분", "한 줄", "다시", "필요한 말", "낮추", "가자")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def allowed_reply(action: str, reply: str) -> bool:
    normalized = normalize_text(reply)
    blocked = ACTION_BLOCKED_SNIPPETS.get(action, ())
    return not any(snippet in normalized for snippet in blocked)


def infer_reply_style(action: str, completion: str) -> str:
    text = normalize_text(completion)

    if action in {"react_laugh", "react_surprise"}:
        return "light_reaction"
    if action == "deescalate":
        if any(token in text for token in BOUNDARY_HINTS):
            return "calm_boundary"
        return "calm_reset"
    if action == "share_feeling":
        if any(token in text for token in SOFT_HINTS):
            return "soft_support"
        return "quiet_support"
    if action == "continue_conversation":
        if "?" in text:
            return "casual_followup"
        return "casual_comment"
    if action == "share_opinion":
        return "direct_opinion" if "?" not in text else "opinion_followup"
    if action == "explain_capabilities":
        if any(token in text for token in DIRECT_HINTS):
            return "direct_capability"
        return "brief_capability"
    if action == "answer_identity":
        return "identity_intro"
    if action == "ask_clarification":
        return "missing_topic_prompt"
    if action == "acknowledge":
        return "brief_ack"
    if action == "ask_location":
        return "location_only"
    if action == "weather_lookup":
        return "fact_weather"
    if action == "recommend":
        return "preference_nudge" if "?" in text else "light_recommend"
    if action == "search_answer":
        return "brief_explanation"
    if action == "music_chat":
        return "music_smalltalk"
    if action == "game_chat":
        return "game_smalltalk"
    if action == "small_talk":
        return "light_greeting" if any(token in text for token in LIGHT_HINTS) else "plain_greeting"
    return "default"


def build_prompt(*, user_text: str, state: dict, labels: dict, reason: str, reply_style: str) -> str:
    action = labels["action"]
    action_rule = ACTION_RULES.get(action, "reply naturally and follow the action exactly")
    lines = [
        "task: discord_reply",
        "persona: black_casual",
        f"intent: {labels['intent']}",
        f"action: {action}",
        f"reply_style: {reply_style}",
        f"action_rule: {action_rule}",
        f"context: {state.get('recent_context', 'none')}",
        f"user: {user_text}",
        f"reason: {reason}",
        "rules:",
        "- write natural Korean only",
        "- one or two short sentences",
        "- follow the action exactly",
        "- follow the reply_style hint",
        "- no metadata, no prompt words",
        "- no repeated phrases",
        "reply:",
    ]
    return "\n".join(lines)


def split_rows(rows: list[dict], *, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        meta = row.get("meta", {})
        group_key = "||".join(
            [
                normalize_text(meta.get("user_text", "")),
                str(meta.get("action", "")),
                str(meta.get("reply_style", "")),
            ]
        )
        grouped.setdefault(group_key, []).append(row)

    grouped_by_action: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for group_key, members in grouped.items():
        action = members[0]["meta"]["action"]
        grouped_by_action[action].append((group_key, len(members)))

    rng = random.Random(seed)
    eval_groups: set[str] = set()
    for action, groups in grouped_by_action.items():
        groups = list(groups)
        rng.shuffle(groups)
        total_rows = sum(size for _, size in groups)
        target_rows = max(1, round(total_rows * eval_ratio))
        selected: list[str] = []
        current = 0
        for group_key, size in sorted(groups, key=lambda item: item[1], reverse=True):
            if current >= target_rows:
                break
            if current + size <= target_rows or not selected:
                selected.append(group_key)
                current += size
        eval_groups.update(selected)

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
        "daily_response_curated_base_v6",
    )
    v5_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites_v5.py",
        "daily_response_curated_v5_v6",
    )

    with v5_module.ALL_PATH.open(encoding="utf-8") as f:
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
        reply_style = infer_reply_style(action, completion)
        prompt = build_prompt(
            user_text=meta["user_text"],
            state=state,
            labels=labels,
            reason=reason,
            reply_style=reply_style,
        )
        key = (prompt, completion)
        if key in unique_pairs:
            continue
        unique_pairs.add(key)

        meta["reply_style"] = reply_style
        rewrite_rows.append(
            {
                "item_id": meta["item_id"],
                "rewrite_index": meta["rewrite_index"],
                "state": state,
                "labels": labels,
                "rewrite": {
                    "user_text": meta["user_text"],
                    "assistant_reply": completion,
                    "reply_style": reply_style,
                    "notes": "curated natural rewrite v6 action+style conditioned",
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
    unique_prompt_by_action: dict[str, int] = {}
    unique_style_by_action: dict[str, int] = {}
    action_prompt_completion_counts: dict[str, list[int]] = {}

    prompt_to_completions: dict[str, set[str]] = defaultdict(set)
    for row in sft_rows:
        prompt = normalize_text(row["prompt"])
        completion = normalize_text(row["completion"])
        prompt_to_completions[prompt].add(completion)

    for action in sorted(action_counts):
        action_rows = [row for row in sft_rows if row["meta"]["action"] == action]
        unique_user_by_action[action] = len({row["meta"]["user_text"] for row in action_rows})
        unique_reply_by_action[action] = len({row["completion"] for row in action_rows})
        unique_prompt_by_action[action] = len({normalize_text(row["prompt"]) for row in action_rows})
        unique_style_by_action[action] = len({row["meta"]["reply_style"] for row in action_rows})
        counts = sorted(
            [len(prompt_to_completions[normalize_text(row["prompt"])]) for row in action_rows],
            reverse=True,
        )
        action_prompt_completion_counts[action] = counts[:5]

    train_action_counts = Counter(row["meta"]["action"] for row in train_rows)
    eval_action_counts = Counter(row["meta"]["action"] for row in eval_rows)
    train_prompts = {normalize_text(row["prompt"]) for row in train_rows}
    eval_prompts = {normalize_text(row["prompt"]) for row in eval_rows}

    ambiguous_prompt_groups = {
        prompt: len(completions)
        for prompt, completions in prompt_to_completions.items()
        if len(completions) > 1
    }

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
        "unique_prompt_by_action": unique_prompt_by_action,
        "unique_style_by_action": unique_style_by_action,
        "largest_prompt_completion_counts_by_action": action_prompt_completion_counts,
        "prompt_overlap_between_train_eval": len(train_prompts & eval_prompts),
        "duplicate_prompt_completion_groups": 0,
        "duplicate_prompt_completion_rows": 0,
        "redundant_prompt_completion_rows": 0,
        "unique_prompt_completion_pairs": len(unique_pairs),
        "ambiguous_prompt_groups": len(ambiguous_prompt_groups),
        "max_completions_for_single_prompt": max(ambiguous_prompt_groups.values(), default=1),
        "rewrite_path": str(REWRITE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "source_type": "curated_clean_rewrite_v6_action_style_conditioned",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
