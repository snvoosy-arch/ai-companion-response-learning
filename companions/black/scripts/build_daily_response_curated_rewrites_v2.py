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


REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_v2_1344.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v2_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_v2_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_v2_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_v2_summary.json"

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

SUPPLEMENTAL_USER_POOLS: dict[str, list[str]] = {
    "greeting": [
        "안냐",
        "하이하이",
        "왔음",
        "접속했다",
        "나 왔다",
        "살아있냐",
    ],
    "smalltalk": [
        "요즘 뭐 하고 지내",
        "별일 없냐",
        "오늘은 어땠냐",
        "지금 바쁨?",
        "뭐 보면서 쉬는 중이냐",
        "오늘 컨디션 어떰",
    ],
    "feeling": [
        "오늘 기분이 너무 가라앉아",
        "계속 힘이 안 나",
        "괜히 축 처진다",
        "마음이 좀 무겁네",
        "하루 종일 기분이 별로야",
        "오늘따라 유독 처진다",
    ],
    "opinion": [
        "이 선택 괜찮아 보이냐",
        "이거 지금 해도 되겠냐",
        "내 생각 너무 이상하냐",
        "이 판단 무리수 같냐",
        "이쪽으로 가는 게 맞냐",
        "이 그림 어떠냐",
    ],
    "identity": [
        "너 정체가 뭐냐",
        "너는 뭐 하는 애냐",
        "대충 어떤 봇이냐",
        "너 누구 컨셉이냐",
        "너 역할이 뭔데",
        "뭐 하는 놈인지 말해봐",
    ],
    "help": [
        "할 수 있는 거 목록 줘",
        "기능 대충 정리해줘",
        "너 어디까지 되냐",
        "무슨 도움 줄 수 있냐",
        "사용법 알려줘",
        "어떤 식으로 반응하냐",
    ],
    "reply_request": [
        "왜 대답 안 하냐",
        "답 좀 해봐",
        "읽었으면 반응해",
        "지금 무시한 거냐",
        "왜 씹냐",
        "말 좀 받아줘",
    ],
    "confirm": [
        "오케이",
        "그럼 됐다",
        "맞음",
        "좋아 그걸로",
        "그렇지",
        "ㅇㅋ 인정",
    ],
    "deny": [
        "아니 그건 아님",
        "그렇게는 아냐",
        "그건 좀 아니지",
        "아냐아냐",
        "그건 빼자",
        "아닌데",
    ],
    "meaning": [
        "이 표현 무슨 뉘앙스냐",
        "이 말투가 어떻게 들리냐",
        "이 문장 뜻 좀 풀어줘",
        "저 말이 왜 그렇게 들리냐",
        "이 표현 설명 가능?",
        "이거 한국어 느낌이 뭐냐",
    ],
    "recommend": [
        "심심한데 뭐 볼까",
        "가볍게 즐길 거 추천해",
        "요즘 볼만한 거 있냐",
        "시간 때울 거 골라줘",
        "무난한 거 하나만 찍어줘",
        "취향 안 타는 걸로 추천해",
    ],
    "laugh": [
        "이건 진짜 웃기네",
        "개웃기다 ㅋㅋ",
        "방금 그건 좀 웃겼다",
        "나 터졌다",
        "그 포인트는 인정 ㅋㅋ",
        "웃겨서 말이 안 나온다",
    ],
    "surprise": [
        "이건 좀 놀랐는데",
        "와 이게 진짜냐",
        "생각보다 세네",
        "그건 진짜 의외다",
        "와 좀 충격인데",
        "이건 예상 못 했다",
    ],
    "deescalate": [
        "말이 너무 쎈데",
        "지금 좀 과하게 가는 거 아님?",
        "그 표현은 선 넘은 듯",
        "그 말은 좀 빼라",
        "굳이 그렇게까지 말해야 하냐",
        "톤 좀 낮추자",
    ],
    "weather": [
        "밖에 많이 추움?",
        "오늘 비 오냐",
        "우산 챙겨야 하냐",
        "기온 어떰",
        "패딩 입어야 되냐",
        "오늘 날씨 상태 어때",
    ],
    "weather_followup": [
        "서울 기준",
        "부산으로 봐줘",
        "인천은 어때",
        "대전 쪽",
        "대구로 보면?",
        "수원 기준으로",
    ],
    "music": [
        "요즘 뭐 듣냐",
        "플레이리스트 뭐 도냐",
        "잔잔한 곡 좋아하냐",
        "요즘 꽂힌 노래 있냐",
        "노래 추천도 하냐",
        "음악 취향 대충 말해봐",
    ],
    "game": [
        "요즘 하는 게임 뭐냐",
        "최근에 뭐 했냐",
        "게임 얘기 좀 해봐",
        "무슨 장르 좋아함",
        "지금도 게임 하냐",
        "요새 재밌는 거 있냐",
    ],
}

SUPPLEMENTAL_REPLY_POOLS: dict[str, list[str]] = {
    "greeting": [
        "왔네. 바로 본다.",
        "하이. 지금부터 받는다.",
        "어 왔냐. 얘기 던져.",
        "들어온 건 봤다. 뭐부터 갈래.",
        "좋다. 바로 이어가자.",
        "반갑지. 오늘은 뭐 들고 왔냐.",
    ],
    "smalltalk": [
        "크게 달라진 건 없는데, 지금은 네 얘기부터 듣는다.",
        "그냥 적당히 굴러가고 있다. 너는 어땠냐.",
        "조용히 넘기고 있었지. 너 쪽은 뭐 있었냐.",
        "지금은 한가한 편이다. 얘기 하나만 던져.",
        "엄청 바쁘진 않다. 하나만 꺼내보자.",
        "무난하게 가는 중이다. 너는 오늘 뭐 있었냐.",
    ],
    "feeling": [
        "그럴 땐 억지로 끌어올리려 하지 말고 그냥 버텨도 된다.",
        "지금은 회복부터 보는 쪽이 맞다. 너무 몰아붙이지 마.",
        "그 상태면 쉬는 쪽으로 가는 게 낫다. 오늘은 덜 하자.",
        "괜히 더 세게 밀지 말고, 오늘은 숨만 고르는 날로 둬.",
        "하루 정도는 처져 있어도 된다. 네가 이상한 건 아니다.",
        "그런 날 있다. 오늘은 최소치만 해도 충분하다.",
    ],
    "opinion": [
        "내 기준에선 그쪽이 더 덜 후회할 선택처럼 보인다.",
        "무리수까지는 아닌데, 한 번 더 보고 가면 더 좋다.",
        "지금 정보로는 그 방향이 제일 무난하다.",
        "완벽하진 않아도 그 선택이 덜 흔들릴 것 같다.",
        "조금 보수적으로 가면 그쪽이 낫다.",
        "당장 하나 고르라면 나는 그쪽으로 간다.",
    ],
    "identity": [
        "나는 입력 보고 의도 먼저 잡고, 그다음 답을 고르는 쪽에 가깝다.",
        "대충 말하면 대화형 디스코드 봇인데, 반응 전에 한 번 판단을 거친다.",
        "그냥 수다만 치는 봇보단, 상황 보고 답을 고르는 타입이다.",
        "질문, 감정, 요청 같은 걸 먼저 보고 거기에 맞춰 반응한다.",
        "예측 기반으로 반응하는 쪽에 가까운 디스코드 봇이라고 보면 된다.",
        "짧게 말하면, 먼저 판단하고 그다음 말하는 봇이다.",
    ],
    "help": [
        "지금은 잡담, 감정 반응, 간단한 설명, 추천 쪽이 제일 안정적이다.",
        "일상 대화 받아주고, 표현 의미 풀어주고, 가벼운 추천 정도는 된다.",
        "질문 응답, 기분 받아주기, 날씨 같은 간단한 흐름은 처리할 수 있다.",
        "엄청 만능은 아니어도, 기본 대화랑 설명은 꽤 잘 받는다.",
        "잡담하고 뜻 설명하고, 간단한 도움 주는 쪽으로 보면 된다.",
        "지금은 일상 대화 중심이다. 복잡한 건 풀어 묻는 쪽이 더 안정적이다.",
    ],
    "reply_request": [
        "받고는 있다. 다만 뭘로 답하면 되는지 한 줄만 더 줘.",
        "응답은 가능하다. 기준만 조금 더 주면 바로 이어간다.",
        "무시한 건 아니다. 맥락이 비어서 한 줄만 더 필요하다.",
        "보고 있다. 뭘 원한 건지만 짧게 붙여줘.",
        "답은 한다. 다만 지금은 정보가 조금 부족하다.",
        "읽고 있다. 주제만 더 붙여주면 바로 간다.",
    ],
    "confirm": [
        "좋다. 그 방향으로 이해하고 이어간다.",
        "오케이. 지금부터는 그 기준으로 본다.",
        "확인했다. 그걸로 잡고 가자.",
        "좋아. 그 해석으로 두고 이어간다.",
        "받았다. 그러면 그쪽으로 정리한다.",
        "인정. 그 기준이면 충분하다.",
    ],
    "deny": [
        "좋다. 그건 빼고 다시 좁혀보자.",
        "확인. 그 방향은 아닌 걸로 두겠다.",
        "오케이. 그 해석은 버리고 다시 본다.",
        "알겠다. 그건 제외하고 이어가면 된다.",
        "좋다. 그건 아니라고 보고 다시 잡는다.",
        "받았다. 그쪽은 접고 다른 방향으로 보자.",
    ],
    "meaning": [
        "표현 자체 설명은 가능하다. 문맥까지 주면 더 정확하게 풀 수 있다.",
        "말뜻은 풀어줄 수 있다. 다만 문장째로 주면 뉘앙스까지 더 잘 잡힌다.",
        "단어 설명은 되는데, 문맥이 있으면 훨씬 정확해진다.",
        "이 표현이 왜 그렇게 들리는지는 풀 수 있다. 앞뒤 문장 있으면 더 좋다.",
        "뜻 설명은 가능하다. 문장까지 붙이면 애매함이 줄어든다.",
        "표현 설명은 해줄 수 있다. 뉘앙스까지 보려면 문맥이 필요하다.",
    ],
    "recommend": [
        "좋다. 취향 축만 하나 주면 꽤 바로 좁혀줄 수 있다.",
        "무난하게 갈지, 취향 타게 갈지부터 말해주면 된다.",
        "추천은 가능하다. 다만 네 취향 축 하나만 잡아주면 더 좋다.",
        "가볍게 갈 거면 바로 해도 된다. 취향 있으면 더 맞춘다.",
        "하나 찍어주는 건 되는데, 싫어하는 것만 말해줘도 훨씬 편하다.",
        "추천은 할 수 있다. 방향만 조금 주면 덜 빗나간다.",
    ],
    "laugh": [
        "그건 웃길 만했다 ㅋㅋ",
        "아 그 포인트는 인정이다 ㅋㅋ",
        "그건 나도 바로 웃었다 ㅋㅋ",
        "지금 건 좀 세게 웃겼다 ㅋㅋ",
        "그 장면은 진짜 못 참지 ㅋㅋ",
        "그건 나도 터질 만하다고 본다 ㅋㅋ",
    ],
    "surprise": [
        "그건 놀랄 만했다. 바로 튀는 포인트가 있다.",
        "생각보다 세서 바로 눈에 들어온다.",
        "그 반응이 나오는 게 이해된다. 좀 크다.",
        "그건 의외다. 바로 받아칠 만한 포인트가 있다.",
        "와, 그건 예상 밖으로 강하다.",
        "그건 확실히 놀랄 만하다. 눈에 확 들어온다.",
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
        "요즘은 너무 튀는 것보단 무난하게 오래 가는 쪽이 좋다.",
        "분위기 타는 편이라, 상황 따라 바뀌는데 잔잔한 쪽이 많다.",
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


def load_base_module():
    script_path = ROOT / "scripts" / "build_daily_response_curated_rewrites.py"
    spec = importlib.util.spec_from_file_location("daily_response_curated_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def build_pair_records(base_module) -> list[dict]:
    records: list[dict] = []
    for group_index, group in enumerate(base_module.load_groups()):
        state = dict(group["state"])
        labels = dict(group["labels"])
        context = state["recent_context"]
        for pair_index, (source_user, source_reply) in enumerate(group["pairs"]):
            base_user_pool = [user_text for user_text, _ in group["pairs"]]
            base_reply_pool = [reply for _, reply in group["pairs"]]
            user_pool = dedupe_keep_order(
                base_user_pool
                + base_module.EXTRA_USER_POOLS.get(context, [])
                + SUPPLEMENTAL_USER_POOLS.get(context, [])
            )
            reply_pool = dedupe_keep_order(
                base_reply_pool
                + base_module.REPLY_POOLS.get(context, [])
                + SUPPLEMENTAL_REPLY_POOLS.get(context, [])
            )
            records.append(
                {
                    "group_index": group_index,
                    "pair_index": pair_index,
                    "state": state,
                    "labels": labels,
                    "context": context,
                    "source_user": source_user,
                    "source_reply": source_reply,
                    "user_pool": user_pool,
                    "reply_pool": reply_pool,
                    "item_id": f"daily-v2-{group_index:02d}-{pair_index:02d}",
                }
            )
    return records


def target_count_for_action(action: str, source_pair_count: int) -> int:
    if action in TARGET_COUNT_BY_ACTION:
        return TARGET_COUNT_BY_ACTION[action]
    return max(source_pair_count * 8, 32)


def distribute_counts(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def build_rewrite_rows(base_module) -> list[dict]:
    pair_records = build_pair_records(base_module)
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
            reply_pool = dedupe_keep_order([record["source_reply"]] + record["reply_pool"])

            for rewrite_index in range(rewrite_count):
                if rewrite_index == 0:
                    user_text = record["source_user"]
                    assistant_reply = record["source_reply"]
                else:
                    user_text = user_pool[
                        (record["pair_index"] + rewrite_index - 1) % len(user_pool)
                    ]
                    assistant_reply = reply_pool[
                        (record["group_index"] * 3 + rewrite_index - 1) % len(reply_pool)
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
                            "notes": "curated natural rewrite v2",
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
        rows.append(
            {
                "prompt": base_module.build_prompt(user_text=user_text, state=state, labels=labels),
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
    base_module = load_base_module()
    rewrite_rows = build_rewrite_rows(base_module)
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
        "source_type": "curated_clean_rewrite_v2",
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
