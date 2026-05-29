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


REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v3_1344.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v3_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v3_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v3_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v3_summary.json"

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

BLOCKED_REPLY_SNIPPETS = (
    "다시 한 줄만 줘",
    "그래야 안 틀린다",
    "그쪽은 빼고 다시 본다",
    "나중에 보자",
    "덜 망가",
    "다음 것도 이어서 보자",
    "직전 맥락을 같이 주면",
    "대상만 조금 더 주면",
    "뭐에 대한 왜인지",
)

EXTRA_REPLY_POOLS: dict[str, list[str]] = {
    "greeting": [
        "안녕. 바로 보고 있다.",
        "왔네. 뭐부터 할까.",
        "하이. 오늘은 뭐 들고 왔냐.",
        "반갑다. 얘기 하나 던져.",
        "좋다. 바로 이어가자.",
        "접속 확인. 이제 말해.",
    ],
    "smalltalk": [
        "크게 달라진 건 없는데, 네 쪽 얘기부터 듣자.",
        "적당히 굴러가고 있다. 너는 어땠냐.",
        "지금은 한가한 편이다. 하나만 꺼내봐.",
        "무난하게 가는 중이다. 너는 뭐 있었냐.",
        "조용히 넘어가고 있었다. 너 쪽은?",
        "엄청 바쁘진 않다. 얘기 하나 던져.",
    ],
    "feeling": [
        "그럴 땐 억지로 끌어올리려 하지 말고 그냥 버텨도 된다.",
        "지금은 회복부터 보는 쪽이 맞다. 너무 몰아붙이지 마.",
        "그 상태면 쉬는 쪽으로 가는 게 낫다. 오늘은 덜 하자.",
        "하루 정도는 처져 있어도 된다. 네가 이상한 건 아니다.",
        "그런 날 있다. 오늘은 최소치만 해도 충분하다.",
        "오늘은 속도부터 줄이는 게 맞다. 너무 세게 밀지 마.",
    ],
    "opinion": [
        "내 기준에선 그쪽이 더 덜 후회할 선택처럼 보인다.",
        "지금 정보로는 그 방향이 제일 무난하다.",
        "무리수까지는 아닌데 한 번 더 보고 가면 더 좋다.",
        "당장 하나 고르라면 나는 그쪽으로 간다.",
        "보수적으로 가면 그 선택이 더 안정적이다.",
        "완벽하진 않아도 그쪽이 덜 흔들린다.",
    ],
    "identity": [
        "짧게 말하면, 먼저 판단하고 그다음 말하는 봇이다.",
        "대화형 디스코드 봇인데, 반응 전에 한 번 정리하고 간다.",
        "입력 보고 의도부터 잡고 거기에 맞춰 답한다.",
        "아무 말이나 하는 타입보단, 상황 보고 반응 고르는 쪽이다.",
        "예측 기반으로 반응하는 디스코드 봇이라고 보면 된다.",
        "말부터 치는 쪽보단 먼저 보고 그다음 답한다.",
    ],
    "help": [
        "지금은 잡담, 감정 반응, 짧은 설명, 간단한 추천 쪽이 안정적이다.",
        "일상 대화 받아주고 표현 뜻 풀어주고, 가벼운 추천 정도는 된다.",
        "기본 대화랑 설명, 간단한 날씨 흐름 정도는 처리할 수 있다.",
        "엄청 만능은 아니어도, 일상 대화 쪽은 꽤 잘 받는다.",
        "질문 응답, 감정 반응, 짧은 설명이 주력이다.",
        "가볍게 이어받고 설명하는 쪽은 바로 된다.",
    ],
    "reply_request": [
        "보고는 있다. 다만 뭘로 답하면 되는지 한 줄만 더 줘.",
        "응답은 가능하다. 기준만 조금 더 주면 바로 이어간다.",
        "무시한 건 아니다. 맥락이 비어서 한 줄만 더 필요하다.",
        "읽고 있다. 주제만 더 붙여주면 바로 간다.",
        "답은 한다. 지금은 정보가 조금 부족하다.",
        "받고는 있다. 뭘 원하는지만 더 짧게 줘.",
    ],
    "confirm": [
        "좋다. 그 방향으로 이해하고 이어간다.",
        "오케이. 그 기준으로 잡고 간다.",
        "확인했다. 그걸로 두자.",
        "좋아. 그렇게 보고 이어간다.",
        "받았다. 그러면 그쪽으로 정리한다.",
        "인정. 그 기준이면 충분하다.",
    ],
    "deny": [
        "좋다. 그건 빼고 다시 좁혀보자.",
        "확인. 그 방향은 아닌 걸로 두겠다.",
        "오케이. 그 해석은 버리고 다시 본다.",
        "알겠다. 그건 제외하고 이어가면 된다.",
        "좋다. 그건 아니라고 보고 다른 쪽으로 간다.",
        "받았다. 그 방향은 접고 다시 잡자.",
    ],
    "meaning": [
        "표현 자체 설명은 가능하다. 문맥까지 주면 더 정확하게 풀 수 있다.",
        "말뜻은 풀어줄 수 있다. 문장째로 주면 뉘앙스까지 더 잘 잡힌다.",
        "단어 설명은 되는데, 문맥이 있으면 훨씬 정확해진다.",
        "이 표현이 왜 그렇게 들리는지는 풀 수 있다. 앞뒤 문장 있으면 더 좋다.",
        "뜻 설명은 가능하다. 문장까지 붙이면 애매함이 줄어든다.",
        "표현 설명은 해줄 수 있다. 뉘앙스까지 보려면 문맥이 필요하다.",
    ],
    "recommend": [
        "좋다. 취향 축만 하나 주면 꽤 바로 좁혀줄 수 있다.",
        "무난하게 갈지 취향 타게 갈지부터 말해주면 된다.",
        "추천은 가능하다. 다만 취향 축 하나만 잡아주면 더 좋다.",
        "가볍게 갈 거면 바로 해도 된다. 취향 있으면 더 맞춘다.",
        "하나 찍어주는 건 되는데, 싫어하는 것만 말해줘도 훨씬 편하다.",
        "방향만 조금 주면 덜 빗나간다.",
    ],
    "laugh": [
        "그건 웃길 만했다 ㅋㅋ",
        "아 그 포인트는 인정이다 ㅋㅋ",
        "그건 나도 바로 웃었다 ㅋㅋ",
        "지금 건 좀 세게 웃겼다 ㅋㅋ",
        "그 장면은 진짜 못 참지 ㅋㅋ",
        "나도 그건 터질 만하다고 본다 ㅋㅋ",
    ],
    "surprise": [
        "그건 놀랄 만했다. 바로 튀는 포인트가 있다.",
        "생각보다 세서 눈에 바로 들어온다.",
        "그 반응이 나오는 게 이해된다. 좀 크다.",
        "그건 의외다. 바로 받아칠 만하다.",
        "와, 그건 예상 밖으로 강하다.",
        "그건 확실히 놀랄 만하다.",
    ],
    "deescalate": [
        "그 톤이면 대화가 깨진다. 조금만 낮춰서 다시 말해줘.",
        "지금은 세게 가는 쪽보다 정리해서 말하는 게 낫다.",
        "표현만 조금 낮추면 바로 이어갈 수 있다.",
        "그 말투로 가면 불필요하게 커진다. 톤만 낮추자.",
        "대화는 받을 건데, 표현을 조금만 덜 세게 해줘.",
        "지금은 감정부터 줄이는 쪽이 맞다. 한 번만 정리해서 줘.",
    ],
    "weather": [
        "위치만 주면 바로 볼 수 있다. 도시 이름만 던져.",
        "어느 지역인지 먼저 필요하다. 그다음은 바로 간다.",
        "지명만 있으면 바로 확인할 수 있다.",
        "위치 없이는 추측하게 된다. 지역부터 줘.",
        "도시 이름 하나만 주면 바로 날씨 쪽으로 간다.",
        "지역이 있어야 정확히 본다. 지명만 말해줘.",
    ],
    "weather_followup": [
        "좋다. 그 지역 기준으로 바로 조회한다.",
        "받았다. 그쪽 날씨로 바로 넘어간다.",
        "오케이. 그 지역 기준으로 본다.",
        "확인. 그 위치 날씨로 이어간다.",
        "좋아. 지금부터는 그 지역 기준이다.",
        "그 지명으로 바로 잡고 간다.",
    ],
    "music": [
        "요즘은 잔잔한 쪽이 더 편하게 들어온다.",
        "그때그때 다른데, 보통은 오래 듣기 편한 쪽을 잡는다.",
        "센 것보다 오래 가는 분위기 쪽을 더 듣는 편이다.",
        "너무 튀는 것보단 무난하게 오래 가는 쪽이 좋다.",
        "분위기 타는 편이라 상황 따라 바뀌는데 잔잔한 쪽이 많다.",
        "플레이리스트는 그날 텐션 따라 바뀌지만 편한 쪽이 기본이다.",
    ],
    "game": [
        "가볍게 오래 붙잡을 수 있는 쪽을 더 보게 된다.",
        "요즘은 빡세게 하기보다 부담 적은 쪽이 더 편하다.",
        "장르 안 가리진 않는데 오래 붙이는 건 결국 편한 쪽이다.",
        "빡집중하는 것보단 계속 굴리기 쉬운 쪽을 더 본다.",
        "그때그때 다른데, 무난하게 계속 하기 좋은 게 더 손간다.",
        "엄청 하드하게 가는 것보단 꾸준히 건드릴 수 있는 걸 더 본다.",
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


def is_clean_reply(text: str) -> bool:
    normalized = " ".join(text.split())
    return not any(pattern in normalized for pattern in BLOCKED_REPLY_SNIPPETS)


def target_count_for_action(action: str, source_pair_count: int) -> int:
    if action in TARGET_COUNT_BY_ACTION:
        return TARGET_COUNT_BY_ACTION[action]
    return max(source_pair_count * 8, 32)


def distribute_counts(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def build_prompt(*, user_text: str, state: dict, labels: dict, reason: str) -> str:
    lines = [
        "task: discord_reply",
        "persona: black_casual",
        f"intent: {labels['intent']}",
        f"action: {labels['action']}",
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


def build_pair_records(base_module, v2_module) -> list[dict]:
    records: list[dict] = []
    for group_index, group in enumerate(base_module.load_groups()):
        state = dict(group["state"])
        labels = dict(group["labels"])
        context = state["recent_context"]

        source_user_pool = [user_text for user_text, _ in group["pairs"]]
        source_reply_pool = [reply for _, reply in group["pairs"]]
        clean_source_replies = [reply for reply in source_reply_pool if is_clean_reply(reply)]

        user_pool = dedupe_keep_order(
            source_user_pool
            + base_module.EXTRA_USER_POOLS.get(context, [])
            + v2_module.SUPPLEMENTAL_USER_POOLS.get(context, [])
        )
        reply_pool = dedupe_keep_order(
            clean_source_replies
            + base_module.REPLY_POOLS.get(context, [])
            + v2_module.SUPPLEMENTAL_REPLY_POOLS.get(context, [])
            + EXTRA_REPLY_POOLS.get(context, [])
        )
        reply_pool = [reply for reply in reply_pool if is_clean_reply(reply)]

        for pair_index, (source_user, source_reply) in enumerate(group["pairs"]):
            records.append(
                {
                    "group_index": group_index,
                    "pair_index": pair_index,
                    "state": state,
                    "labels": labels,
                    "context": context,
                    "source_user": source_user,
                    "source_reply": source_reply if is_clean_reply(source_reply) else None,
                    "user_pool": user_pool,
                    "reply_pool": reply_pool,
                    "item_id": f"daily-v3-{group_index:02d}-{pair_index:02d}",
                }
            )
    return records


def build_rewrite_rows(base_module, v2_module) -> list[dict]:
    pair_records = build_pair_records(base_module, v2_module)
    by_action: dict[str, list[dict]] = defaultdict(list)
    for record in pair_records:
        by_action[record["labels"]["action"]].append(record)

    rows: list[dict] = []
    for action, records in by_action.items():
        counts = distribute_counts(
            target_count_for_action(action, len(records)),
            len(records),
        )
        for record, rewrite_count in zip(records, counts):
            user_pool = dedupe_keep_order([record["source_user"]] + record["user_pool"])
            reply_pool = dedupe_keep_order(
                ([record["source_reply"]] if record["source_reply"] else []) + record["reply_pool"]
            )

            for rewrite_index in range(rewrite_count):
                user_text = user_pool[
                    (record["pair_index"] + rewrite_index) % len(user_pool)
                ]
                assistant_reply = reply_pool[
                    (record["group_index"] * 5 + rewrite_index) % len(reply_pool)
                ]
                rows.append(
                    {
                        "item_id": record["item_id"],
                        "rewrite_index": rewrite_index,
                        "state": record["state"],
                        "labels": record["labels"],
                        "source_dialogue": {
                            "user_text": record["source_user"],
                            "assistant_reply": record["source_reply"],
                        },
                        "rewrite": {
                            "user_text": user_text,
                            "assistant_reply": assistant_reply,
                            "notes": "curated natural rewrite v3",
                        },
                    }
                )
    return rows


def build_sft_rows(base_module, rewrite_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in rewrite_rows:
        user_text = row["rewrite"]["user_text"]
        assistant_reply = row["rewrite"]["assistant_reply"]
        state = dict(row["state"])
        labels = dict(row["labels"])
        reason = base_module.ACTION_REASONS.get(
            labels["action"],
            "follow the selected action naturally",
        )
        rows.append(
            {
                "prompt": build_prompt(
                    user_text=user_text,
                    state=state,
                    labels=labels,
                    reason=reason,
                ),
                "completion": assistant_reply,
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


def main() -> None:
    base_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites.py",
        "daily_response_curated_base",
    )
    v2_module = load_module(
        ROOT / "scripts" / "build_daily_response_curated_rewrites_v2.py",
        "daily_response_curated_v2",
    )

    rewrite_rows = build_rewrite_rows(base_module, v2_module)
    sft_rows = build_sft_rows(base_module, rewrite_rows)
    train_rows, eval_rows = split_rows(sft_rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(REWRITE_PATH, rewrite_rows)
    write_jsonl(ALL_PATH, sft_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

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

    summary = {
        "rewrite_rows": len(rewrite_rows),
        "sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "unique_user_by_action": unique_user_by_action,
        "unique_reply_by_action": unique_reply_by_action,
        "rewrite_path": str(REWRITE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "source_type": "curated_clean_rewrite_v3",
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
