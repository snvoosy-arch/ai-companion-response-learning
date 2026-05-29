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


REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v4_1344.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v4_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v4_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v4_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v4_summary.json"

SEED = 42
EVAL_RATIO = 0.12

CORE_ACTIONS = {
    "small_talk",
    "share_feeling",
    "share_opinion",
    "continue_conversation",
    "explain_capabilities",
    "ask_clarification",
    "acknowledge",
    "game_chat",
}

SECONDARY_ACTIONS = {
    "answer_identity",
    "search_answer",
    "recommend",
    "react_laugh",
    "react_surprise",
    "deescalate",
    "ask_location",
    "weather_lookup",
    "music_chat",
}

TARGET_COUNT_BY_ACTION = {action: 96 for action in CORE_ACTIONS} | {
    action: 64 for action in SECONDARY_ACTIONS
}

EXTRA_USER_POOLS_V4: dict[str, list[str]] = {
    "help": [
        "대충 뭐까지 되는데",
        "어디까지 받아줄 수 있음",
        "할 줄 아는 거 짧게 말해봐",
        "기능 범위가 어디까지냐",
        "지금 되는 거만 골라서 말해줘",
        "주력 기능이 뭐냐",
    ],
    "deescalate": [
        "말 너무 세게 나간다",
        "지금 톤 많이 올라간 거 아니냐",
        "표현 수위 좀 낮춰라",
        "그 말투면 싸움 난다",
        "지금 말 너무 거칠다",
        "그 톤은 좀 줄이자",
    ],
    "surprise": [
        "와 이건 생각 못 했는데",
        "이건 좀 세게 오네",
        "생각보다 파괴력 있네",
        "와 그건 진짜 의외다",
        "이건 확실히 튄다",
        "이건 반응 나올 만하다",
    ],
}

EXTRA_REPLY_POOLS_V4: dict[str, list[str]] = {
    "help": [
        "짧게 받자면, 일상 대화 받아주고 뜻 풀어주고 가벼운 추천까지는 된다.",
        "지금 강한 쪽은 일상 대화, 표현 설명, 간단한 추천이다.",
        "기본 대화는 바로 받고, 애매한 표현도 풀어줄 수 있다.",
        "엄청 넓진 않아도, 일상 질문이랑 짧은 설명은 안정적이다.",
        "가볍게 이어받고 설명하는 쪽이 주력이다.",
        "복잡한 판단보다 일상 대화와 짧은 설명이 먼저 된다.",
        "대화 받아주고, 의미 풀어주고, 추천 조금 하는 정도로 보면 된다.",
        "일상 반응, 짧은 설명, 간단한 추천은 바로 붙는다.",
        "할 수 있는 건 많지 않아도, 일상 대화 축은 꽤 단단하다.",
        "지금은 평소 대화, 짧은 질문 응답, 의미 설명 쪽이 메인이다.",
        "일단은 잡담하고 설명하는 데 강하다.",
        "대충 말하면 일상 대화형이다. 설명이랑 추천도 조금 된다.",
        "복잡한 건 다시 물어보지만, 기본 대화는 바로 이어간다.",
        "네가 짧게 던지면 그걸 받아서 이어가는 쪽이 제일 자연스럽다.",
        "지금 잘 받는 건 일상 대화와 설명 쪽이다.",
        "질문, 감정 반응, 뜻 설명은 바로 붙는다.",
    ],
    "deescalate": [
        "지금은 말 세기부터 낮추는 게 먼저다.",
        "그 톤이면 내용보다 감정만 남는다. 조금만 낮춰.",
        "하고 싶은 말은 받는데, 표현부터 정리해줘.",
        "그 말투 그대로 가면 대화가 깨진다. 톤만 줄이자.",
        "세게 가는 쪽보다 정리해서 말하는 게 낫다.",
        "감정 실린 건 알겠는데, 수위만 낮추면 바로 받는다.",
        "그 표현은 너무 직선적이다. 한 번만 누르고 다시 말해줘.",
        "대화는 이어갈 수 있다. 다만 그 톤은 조금 빼자.",
        "지금은 맞받아치기보다 정리해서 말하는 게 이득이다.",
        "표현만 낮추면 내용은 바로 볼 수 있다.",
        "말이 센 건 알겠는데, 지금은 한 번만 정리해줘.",
        "그 말투면 불필요하게 커진다. 수위만 낮추자.",
        "감정부터 줄이면 대화는 바로 된다.",
        "내용은 받는다. 표현만 덜 세게 해줘.",
        "그 톤으로는 대화보다 충돌이 먼저 난다.",
        "지금은 공격보다 정리가 먼저다.",
    ],
    "surprise": [
        "그건 확실히 튄다. 반응 나올 만하다.",
        "와, 그건 바로 놀랄 만했다.",
        "예상보다 세다. 눈에 확 들어온다.",
        "그건 좀 의외다. 바로 반응하게 된다.",
        "생각보다 강하게 들어오네.",
        "그건 놀라는 쪽이 정상이다.",
        "지금 건 예상 밖이라 바로 튄다.",
        "그건 한 번 더 보게 만드는 포인트가 있다.",
        "와, 그건 그냥 지나가기 어렵다.",
        "생각보다 파급이 크다. 놀랄 만하다.",
        "그건 반응부터 나오는 게 맞다.",
        "의외로 세게 온다. 바로 눈에 걸린다.",
        "그건 바로 놀람 쪽으로 간다.",
        "지금 건 생각보다 강하다.",
        "그건 튀는 포인트가 확실하다.",
        "놀라는 반응이 먼저 나올 만하다.",
    ],
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


def is_clean_reply(text: str, blocked_patterns: tuple[str, ...]) -> bool:
    normalized = " ".join(text.split())
    return not any(pattern in normalized for pattern in blocked_patterns)


def target_count_for_action(action: str, source_pair_count: int) -> int:
    if action in TARGET_COUNT_BY_ACTION:
        return TARGET_COUNT_BY_ACTION[action]
    return max(source_pair_count * 8, 32)


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


def build_pair_records(base_module, v2_module, v3_module) -> list[dict]:
    records = v3_module.build_pair_records(base_module, v2_module)
    for record in records:
        context = record["context"]
        record["user_pool"] = dedupe_keep_order(
            list(record["user_pool"]) + EXTRA_USER_POOLS_V4.get(context, [])
        )
        record["reply_pool"] = dedupe_keep_order(
            list(record["reply_pool"]) + EXTRA_REPLY_POOLS_V4.get(context, [])
        )
    return records


def build_candidates_for_record(
    *,
    record: dict,
    base_module,
    v3_module,
    blocked_patterns: tuple[str, ...],
) -> list[dict]:
    state = dict(record["state"])
    labels = dict(record["labels"])
    reason = base_module.ACTION_REASONS.get(labels["action"], "follow the selected action naturally")
    user_pool = dedupe_keep_order([record["source_user"]] + list(record["user_pool"]))
    reply_pool = [
        reply
        for reply in dedupe_keep_order(
            ([record["source_reply"]] if record["source_reply"] else []) + list(record["reply_pool"])
        )
        if is_clean_reply(reply, blocked_patterns)
    ]

    if not user_pool or not reply_pool:
        return []

    rotated_user_pool = user_pool[record["pair_index"] % len(user_pool):] + user_pool[: record["pair_index"] % len(user_pool)]
    rotated_reply_pool = reply_pool[record["group_index"] % len(reply_pool):] + reply_pool[: record["group_index"] % len(reply_pool)]

    candidates: list[dict] = []
    local_seen: set[tuple[str, str]] = set()
    candidate_index = 0
    for reply in rotated_reply_pool:
        for user_text in rotated_user_pool:
            prompt = v3_module.build_prompt(user_text=user_text, state=state, labels=labels, reason=reason)
            key = (prompt, reply)
            if key in local_seen:
                continue
            local_seen.add(key)
            candidates.append(
                {
                    "item_id": record["item_id"],
                    "rewrite_index": candidate_index,
                    "state": state,
                    "labels": labels,
                    "source_dialogue": {
                        "user_text": record["source_user"],
                        "assistant_reply": record["source_reply"],
                    },
                    "rewrite": {
                        "user_text": user_text,
                        "assistant_reply": reply,
                        "notes": "curated natural rewrite v4",
                    },
                    "prompt": prompt,
                    "completion": reply,
                }
            )
            candidate_index += 1
    return candidates


def build_rewrite_rows(base_module, v2_module, v3_module) -> list[dict]:
    records = build_pair_records(base_module, v2_module, v3_module)
    blocked_patterns = tuple(v3_module.BLOCKED_REPLY_SNIPPETS)
    by_action: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_action[record["labels"]["action"]].append(record)

    all_rows: list[dict] = []
    for action, action_records in by_action.items():
        target_count = target_count_for_action(action, len(action_records))
        candidate_lists = [
            build_candidates_for_record(
                record=record,
                base_module=base_module,
                v3_module=v3_module,
                blocked_patterns=blocked_patterns,
            )
            for record in action_records
        ]
        used_keys: set[tuple[str, str]] = set()
        unique_candidates: list[dict] = []
        for candidates in candidate_lists:
            for candidate in candidates:
                key = (candidate["prompt"], candidate["completion"])
                if key in used_keys:
                    continue
                used_keys.add(key)
                unique_candidates.append(candidate)

        first_pass: list[dict] = []
        rest: list[dict] = []
        seen_completions: set[str] = set()
        for candidate in unique_candidates:
            completion = candidate["completion"]
            if completion not in seen_completions:
                seen_completions.add(completion)
                first_pass.append(candidate)
            else:
                rest.append(candidate)

        ordered_candidates = first_pass + rest
        rows_for_action: list[dict] = []
        for candidate in ordered_candidates[:target_count]:
            candidate = dict(candidate)
            candidate["rewrite_index"] = len(rows_for_action)
            rows_for_action.append(candidate)

        all_rows.extend(rows_for_action)

    return all_rows


def build_sft_rows(rewrite_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in rewrite_rows:
        state = dict(row["state"])
        labels = dict(row["labels"])
        user_text = row["rewrite"]["user_text"]
        rows.append(
            {
                "prompt": row["prompt"],
                "completion": row["completion"],
                "meta": {
                    "item_id": row["item_id"],
                    "rewrite_index": row["rewrite_index"],
                    "recent_context": state.get("recent_context"),
                    "intent": labels["intent"],
                    "action": labels["action"],
                    "user_text": user_text,
                    "source_user_text": row["source_dialogue"]["user_text"],
                },
            }
        )
    return rows


def summarize_rows(rewrite_rows: list[dict], sft_rows: list[dict], train_rows: list[dict], eval_rows: list[dict]) -> dict:
    action_counts = Counter(row["labels"]["action"] for row in rewrite_rows)
    unique_user_by_action: dict[str, int] = {}
    unique_reply_by_action: dict[str, int] = {}
    for action in sorted(action_counts):
        unique_user_by_action[action] = len(
            {row["rewrite"]["user_text"] for row in rewrite_rows if row["labels"]["action"] == action}
        )
        unique_reply_by_action[action] = len(
            {row["rewrite"]["assistant_reply"] for row in rewrite_rows if row["labels"]["action"] == action}
        )

    prompt_completion_counter = Counter((row["prompt"], row["completion"]) for row in sft_rows)
    duplicate_groups = sum(1 for count in prompt_completion_counter.values() if count > 1)
    duplicate_rows = sum(count for count in prompt_completion_counter.values() if count > 1)
    redundant_rows = sum(count - 1 for count in prompt_completion_counter.values() if count > 1)

    return {
        "rewrite_rows": len(rewrite_rows),
        "sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "unique_user_by_action": unique_user_by_action,
        "unique_reply_by_action": unique_reply_by_action,
        "duplicate_prompt_completion_groups": duplicate_groups,
        "duplicate_prompt_completion_rows": duplicate_rows,
        "redundant_prompt_completion_rows": redundant_rows,
        "unique_prompt_completion_pairs": len(prompt_completion_counter),
        "rewrite_path": str(REWRITE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "source_type": "curated_clean_rewrite_v4",
    }


def main() -> None:
    base_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites.py",
        "daily_response_curated_base_v4",
    )
    v2_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites_v2.py",
        "daily_response_curated_v2_v4",
    )
    v3_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites_v3.py",
        "daily_response_curated_v3_v4",
    )

    rewrite_rows = build_rewrite_rows(base_module, v2_module, v3_module)
    sft_rows = build_sft_rows(rewrite_rows)
    train_rows, eval_rows = split_rows(sft_rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(REWRITE_PATH, rewrite_rows)
    write_jsonl(ALL_PATH, sft_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    summary = summarize_rows(rewrite_rows, sft_rows, train_rows, eval_rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
