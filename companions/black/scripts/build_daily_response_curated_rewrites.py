from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive_bot.core.renderer import SYSTEM_PROMPTS
from predictive_bot.llm.kobart_client import KoBartGenerationClient


REWRITE_PATH = ROOT / "data" / "rewrite_results" / "daily_response_rewrites_curated_448.jsonl"
ALL_PATH = ROOT / "data" / "daily_response_rewritten_sft_all.jsonl"
TRAIN_PATH = ROOT / "data" / "daily_response_rewritten_sft_train.jsonl"
EVAL_PATH = ROOT / "data" / "daily_response_rewritten_sft_eval.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_response_rewritten_sft_summary.json"

SEED = 42
EVAL_RATIO = 0.12
REWRITES_PER_PAIR = 4


ACTION_REASONS = {
    "small_talk": "인사나 가벼운 반응에는 짧게 받아주는 게 자연스럽다.",
    "continue_conversation": "짧은 잡담은 부담 없이 받아주고 대화를 이어가는 게 맞다.",
    "share_feeling": "감정 섞인 말은 공감 쪽으로 짧게 받아주는 게 낫다.",
    "share_opinion": "의견을 묻는 말에는 한 줄 의견을 먼저 주는 게 자연스럽다.",
    "answer_identity": "정체를 묻는 말에는 짧고 직접적으로 소개하는 게 맞다.",
    "explain_capabilities": "기능 질문은 할 수 있는 범위를 짧게 설명하는 게 좋다.",
    "ask_clarification": "맥락이 비면 바로 단정하지 말고 한 줄 더 물어보는 게 안전하다.",
    "acknowledge": "확인/부정 응답은 짧게 수용하고 흐름만 이어주면 된다.",
    "search_answer": "뜻이나 의미 질문은 표현의 뜻을 짧게 풀어주는 게 맞다.",
    "recommend": "추천 요청은 취향을 조금 더 묻거나 가볍게 방향을 제시하면 된다.",
    "react_laugh": "웃긴 입력은 짧고 즉각적으로 같이 반응하는 게 자연스럽다.",
    "react_surprise": "놀란 입력은 짧은 감탄으로 바로 받는 편이 맞다.",
    "deescalate": "날 선 말은 감정을 낮추고 차분하게 다시 말해달라고 하는 게 우선이다.",
    "ask_location": "날씨는 지역이 있어야 정확히 답할 수 있어서 위치를 먼저 물어야 한다.",
    "weather_lookup": "위치가 주어졌으면 그 기준으로 바로 조회하겠다고 받는 게 맞다.",
    "music_chat": "음악 얘기는 취향 중심으로 가볍게 받아치는 게 자연스럽다.",
    "game_chat": "게임 얘기는 취향과 상황을 가볍게 이어받는 게 맞다.",
}


EXTRA_USER_POOLS: dict[str, list[str]] = {
    "greeting": ["들어왔다", "왔냐", "어이", "접속했다"],
    "smalltalk": ["지금 뭐 하는 중이냐", "요새 어찌 지내냐", "오늘 하루 어땠냐"],
    "feeling": ["오늘 마음이 좀 처진다", "기분이 좀 내려앉는다", "괜히 답답하다"],
    "opinion": ["네 기준엔 어떤 편이냐", "이 선택 어떻게 보냐", "솔직히 이거 별로냐"],
    "identity": ["정체가 뭐임", "무슨 타입 봇이냐", "넌 대체 뭐 하는 애냐"],
    "help": ["사용법 알려줘", "되는 기능 말해줘", "어디까지 할 수 있냐"],
    "reply_request": ["왜 답 안 하냐", "반응 좀 해봐", "보고 있으면 대답 좀"],
    "confirm": ["맞지", "그치", "오케이 그거지"],
    "deny": ["아닌데", "그건 좀 아니다", "그건 아닌 듯"],
    "meaning": ["이 말이 정확히 무슨 의미야", "그 표현 뜻 설명해줘", "이건 무슨 뜻으로 읽어야 해"],
    "recommend": ["가볍게 볼만한 거 있냐", "시간 때울 거 추천해줘", "하나만 골라줘"],
    "laugh": ["아 개웃기네", "이건 좀 웃긴데", "ㅋㅋ 이건 못 참지"],
    "surprise": ["뭐야 이건 또", "이건 예상 못 했네", "와 이건 좀 세다"],
    "deescalate": ["왜 이렇게 공격적이냐", "말을 너무 세게 하는데", "그 톤은 좀 낮춰라"],
    "weather": ["오늘 밖에 추움?", "비 올 각이냐", "지금 날씨 어떤 편이냐"],
    "weather_followup": ["서울 쪽", "부산인데", "인천 기준"],
    "music": ["요즘 듣는 음악 뭐냐", "노래 뭐 듣고 있었냐", "자주 트는 곡 있냐"],
    "game": ["요새 하는 게임 뭐냐", "롤 하냐", "게임 얘기 좀 해봐"],
    "meta": ["왜 그렇게 봤어", "그 판단 이유가 뭐야", "왜 그렇게 말한 건데"],
    "closing": ["도움 됐다", "땡큐", "덕분에 정리됐다"],
    "idle_break": ["갑자기 조용해졌네", "왜 말 끊겼냐", "뭐 생각하느라 멈췄냐"],
    "care": ["오늘 잠을 거의 못 잤다", "계속 피곤하다", "아무것도 하기 싫다"],
    "compliment": ["오늘 말 잘하네", "생각보다 괜찮은데", "오늘은 좀 믿음직하다"],
    "choice_help": ["이거 보내도 되겠냐", "지금 연락해도 될까", "집에 있을까 나갈까"],
}


REPLY_POOLS: dict[str, list[str]] = {
    "greeting": [
        "안녕. 보고 있었지.",
        "왔네. 지금 받는다.",
        "들어온 건 봤다. 얘기 던져.",
        "오냐. 바로 이어간다.",
        "반갑다. 오늘은 뭐 꺼낼 건데.",
    ],
    "smalltalk": [
        "그냥 이것저것 보고 있었다. 너는 뭐 하는 중인데.",
        "크게 바쁜 건 아니다. 너 쪽은 어때.",
        "그럭저럭 간다. 얘기 하나 던져봐.",
        "지금은 괜찮다. 뭐부터 말할래.",
        "별건 없었다. 너는 오늘 뭐 했냐.",
    ],
    "feeling": [
        "그럴 때는 억지로 올리려 하지 마.",
        "그런 날도 있다. 오늘은 속도 낮춰도 된다.",
        "지금은 버티는 기준부터 낮추는 게 낫다.",
        "그 상태면 무리하게 끌어올릴 필요 없다.",
        "오늘은 그냥 덜 망가지는 쪽으로 가도 된다.",
    ],
    "opinion": [
        "내 기준에선 나쁘지 않다.",
        "그쪽이 더 깔끔해 보이긴 한다.",
        "완전히 틀린 쪽은 아닌데 조금만 더 다듬으면 좋겠다.",
        "나라면 조금 덜 세게 갈 것 같다.",
        "무리수는 아닌데 약간만 정리하면 더 낫다.",
    ],
    "identity": [
        "예측 기반으로 반응하는 대화형 봇이야.",
        "채팅 보고 판단해서 답하는 쪽에 가까운 봇이지.",
        "대화 이어받고 설명하는 쪽으로 만든 봇이다.",
        "질문 받고 반응 고르는 봇이라고 보면 된다.",
        "짧게 말하면, 대화형으로 굴리는 봇이다.",
    ],
    "help": [
        "기본 잡담, 간단한 설명, 날씨 같은 건 가능하다.",
        "대화 이어받기랑 짧은 판단, 설명 쪽은 된다.",
        "일상대화, 감정 반응, 간단한 추천 정도는 바로 가능하다.",
        "질문 응답, 뜻 설명, 가벼운 추천은 받을 수 있다.",
        "지금은 대화, 설명, 추천 쪽이 중심이다.",
    ],
    "reply_request": [
        "응답은 하는데 뭘로 답하면 될지 한 줄만 더 줘.",
        "좋아. 근데 기준이 좀 비어 있다. 한 줄만 더 붙여.",
        "보고는 있었다. 이어갈 주제만 조금 더 줘.",
        "받긴 했는데 맥락이 비어서 한 줄 더 필요하다.",
        "반응은 한다. 뭘로 이어가면 될지만 더 줘.",
    ],
    "confirm": [
        "오케이. 그쪽으로 이해할게.",
        "좋다. 그 기준으로 이어간다.",
        "확인. 그럼 그쪽으로 본다.",
        "알겠다. 그걸로 두자.",
        "오케이. 그 해석으로 잡는다.",
    ],
    "deny": [
        "오케이. 그건 아닌 걸로 두자.",
        "좋아. 그쪽은 빼고 다시 본다.",
        "알겠다. 그 해석은 접자.",
        "그럼 다시 잡아보자.",
        "좋다. 그건 제외하고 이어간다.",
    ],
    "meaning": [
        "문장 전체를 주면 더 정확히 풀 수 있다.",
        "표현 자체 뜻은 설명할 수 있다. 문맥도 있으면 더 정확하다.",
        "지금은 뜻을 묻는 질문으로 읽힌다. 한 줄 더 주면 더 정확히 풀어준다.",
        "말 자체는 풀 수 있다. 문장째로 주면 더 잘 잡힌다.",
        "의미 설명은 가능하다. 문맥이 있으면 덜 헷갈린다.",
    ],
    "recommend": [
        "취향 한 줄만 더 주면 덜 헛나가게 좁힐 수 있다.",
        "가볍게 볼 건지 무거운 걸 원하는지부터 줘.",
        "추천은 가능하다. 원하는 결만 한 줄 더 던져.",
        "좋아. 근데 취향부터 조금 알아야 맞게 간다.",
        "방향만 주면 바로 몇 개로 좁힐 수 있다.",
    ],
    "laugh": [
        "그건 좀 웃기긴 한다.",
        "그 반응 나올 만하다 ㅋㅋ",
        "그래 그건 웃어도 된다.",
        "그건 못 참지 ㅋㅋ",
        "그 정도면 같이 웃을 만하다.",
    ],
    "surprise": [
        "그건 좀 놀랄 만하다.",
        "그 반응 나올 만하다.",
        "나도 그건 바로 눈에 들어온다.",
        "그건 확실히 튄다.",
        "그건 예상 밖으로 세게 들어온다.",
    ],
    "deescalate": [
        "세게 말하면 얘기가 안 풀리니까 한 톤만 낮춰줘.",
        "그렇게 가면 대화가 안 된다. 한 줄만 차분하게 다시 줘.",
        "답답한 건 알겠는데 조금만 정리해서 말해줘.",
        "지금은 감정부터 낮추는 게 맞다. 다시 한 줄만 줘.",
        "톤만 낮추면 바로 이어서 볼 수 있다.",
    ],
    "weather": [
        "어느 지역인데? 도시 이름만 주면 바로 볼게.",
        "지역부터 줘. 그래야 안 틀린다.",
        "위치가 있어야 정확히 본다. 도시 이름만 줘.",
        "어느 쪽 날씨를 볼지 먼저 줘.",
        "도시만 주면 바로 확인해본다.",
    ],
    "weather_followup": [
        "좋아. 그 기준으로 바로 볼게.",
        "오케이. 그 지역 기준으로 이어서 본다.",
        "알겠다. 그쪽 기준으로 확인해본다.",
        "좋다. 그 지역으로 조회해볼게.",
        "받았다. 그 위치 기준으로 간다.",
    ],
    "music": [
        "요즘은 잔잔한 쪽을 더 자주 듣는다.",
        "과한 것보단 오래 가는 쪽을 더 좋아한다.",
        "한 곡만 꽂히기보다 분위기 따라 바꾸는 편이다.",
        "그때그때 다르지만 오래 듣는 쪽은 몇 개 있다.",
        "요즘은 무난하게 오래 가는 쪽을 더 튼다.",
    ],
    "game": [
        "가볍게 게임하거나 영상 보는 쪽이 많다.",
        "그때그때 다르지만 오래 붙잡는 쪽은 몇 개 있다.",
        "완전히 빼놓진 못하지.",
        "장르 따라 다르지만 붙잡으면 오래 보는 편이다.",
        "있긴 한데 취향부터 좀 알아야 맞게 고른다.",
    ],
    "meta": [
        "뭐에 대한 왜인지 한 줄만 더 붙여줘.",
        "직전 맥락을 같이 주면 더 정확히 설명한다.",
        "대상만 조금 더 주면 바로 이어서 설명할게.",
        "어느 판단을 말하는지 먼저 붙여줘.",
        "맥락만 보이면 바로 이유를 풀 수 있다.",
    ],
    "closing": [
        "별말을. 필요하면 또 불러.",
        "오케이. 또 필요하면 던져.",
        "그럼 됐다. 다음 것도 이어서 보자.",
        "좋다. 필요할 때 또 부르면 된다.",
        "그 정도면 된 거지. 또 오면 된다.",
    ],
    "idle_break": [
        "잠깐 생각 정리하고 있었다.",
        "한 박자 쉬고 다시 붙는 중이었다.",
        "끊긴 건 아니고 흐름을 보고 있었다.",
        "조금 멈췄다가 다시 이어붙이는 중이었다.",
        "잠깐 정리하다가 다시 붙는 중이다.",
    ],
    "care": [
        "그럼 텐션 억지로 올리지 말고 천천히 가라.",
        "그럴 땐 무리해서 판단 빨리 안 하는 게 맞다.",
        "그 상태면 일단 쉬는 쪽이 먼저다.",
        "그런 날은 진짜 최소한만 해도 된다.",
        "오늘은 속도 자체를 낮추는 쪽이 낫다.",
    ],
    "compliment": [
        "갑자기 칭찬 들어오네. 나쁘지 않다.",
        "그 정도면 꽤 선방한 거지.",
        "오늘은 컨디션이 좀 괜찮은가 보다.",
        "의외라는 말은 조금 걸리지만 좋게 받겠다.",
        "좋다. 그럼 오늘은 덜 망한 편이네.",
    ],
    "choice_help": [
        "무난하게 가려면 덜 튀는 쪽이 낫다.",
        "지금 기운 없으면 무리해서 가는 쪽은 아니다.",
        "나중에 후회할 것 같으면 한 번 더 묵히는 게 낫다.",
        "급한 일 아니면 상대 시간대부터 보는 게 맞다.",
        "확신이 없으면 한 박자 늦추는 쪽이 안전하다.",
    ],
}


def load_groups():
    script_path = ROOT / "scripts" / "generate_daily_conversation_seed.py"
    spec = importlib.util.spec_from_file_location("daily_seed_source", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GROUPS


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


def build_facts(*, user_text: str, state: dict, labels: dict) -> dict:
    action = labels["action"]
    return {
        "action": action,
        "reason": ACTION_REASONS.get(action, "짧고 자연스럽게 현재 행동을 수행한다."),
        "style": "short",
        "user_text": user_text,
        "known_location": state.get("known_location"),
        "weather": None,
        "world_state": {
            "dominant_intent": labels["intent"],
            "user_emotion": "neutral",
            "conversation_mode": state.get("mode", "daily_chat"),
            "unresolved_need": state.get("awaiting_slot"),
            "factuality_required": action in {"ask_location", "weather_lookup", "search_answer"},
            "risk_level": "medium" if action == "deescalate" else "low",
            "memory_summary": state.get("recent_context", "none"),
            "constraints": [],
            "evidence": [f"recent_context={state.get('recent_context', 'none')}"],
        },
        "policy_trace": {
            "policy_name": "daily_response_rewritten_sft",
            "selected_action": action,
            "selected_reason": ACTION_REASONS.get(action, ""),
            "constraints": [],
            "candidates": [
                {
                    "action": action,
                    "score": 1.0,
                    "reason": ACTION_REASONS.get(action, ""),
                }
            ],
        },
        "explanation_trace": None,
    }


def build_prompt(*, user_text: str, state: dict, labels: dict) -> str:
    user_prompt = (
        "Turn this structured decision into the final Discord reply.\n"
        f"{json.dumps(build_facts(user_text=user_text, state=state, labels=labels), ensure_ascii=False, indent=2)}"
    )
    return KoBartGenerationClient._build_prompt(
        system_prompt=SYSTEM_PROMPTS["black"],
        user_prompt=user_prompt,
    )


def pick_variants(pool: list[str], *, count: int, offset: int) -> list[str]:
    if not pool:
        return []
    return [pool[(offset + step) % len(pool)] for step in range(count)]


def build_rewrite_rows() -> list[dict]:
    groups = load_groups()
    rows: list[dict] = []

    for group_index, group in enumerate(groups):
        state = dict(group["state"])
        labels = dict(group["labels"])
        context = state["recent_context"]

        source_user_pool = [user_text for user_text, _ in group["pairs"]]
        user_pool = dedupe_keep_order(source_user_pool + EXTRA_USER_POOLS.get(context, []))
        reply_pool = dedupe_keep_order(REPLY_POOLS.get(context, [reply for _, reply in group["pairs"]]))

        for pair_index, (source_user, source_reply) in enumerate(group["pairs"]):
            user_variants = [source_user]
            user_variants.extend(
                pick_variants(
                    [candidate for candidate in user_pool if candidate != source_user],
                    count=max(0, REWRITES_PER_PAIR - 1),
                    offset=pair_index + group_index,
                )
            )
            user_variants = dedupe_keep_order(user_variants)[:REWRITES_PER_PAIR]
            while len(user_variants) < REWRITES_PER_PAIR:
                user_variants.append(source_user)

            reply_variants = [source_reply]
            reply_variants.extend(
                pick_variants(
                    [candidate for candidate in reply_pool if candidate != source_reply],
                    count=max(0, REWRITES_PER_PAIR - 1),
                    offset=pair_index * 2 + group_index,
                )
            )
            reply_variants = dedupe_keep_order(reply_variants)[:REWRITES_PER_PAIR]
            while len(reply_variants) < REWRITES_PER_PAIR:
                reply_variants.append(source_reply)

            item_id = f"daily-{group_index:02d}-{pair_index:02d}"
            for rewrite_index, (user_text, assistant_reply) in enumerate(zip(user_variants, reply_variants)):
                rows.append(
                    {
                        "item_id": item_id,
                        "rewrite_index": rewrite_index,
                        "state": state,
                        "labels": labels,
                        "source_dialogue": {
                            "user_text": source_user,
                            "assistant_reply": source_reply,
                        },
                        "rewrite": {
                            "user_text": user_text,
                            "assistant_reply": assistant_reply,
                            "notes": "curated natural rewrite",
                        },
                    }
                )
    return rows


def build_sft_rows(rewrite_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in rewrite_rows:
        user_text = row["rewrite"]["user_text"]
        assistant_reply = row["rewrite"]["assistant_reply"]
        state = dict(row["state"])
        labels = dict(row["labels"])
        rows.append(
            {
                "prompt": build_prompt(user_text=user_text, state=state, labels=labels),
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
    rewrite_rows = build_rewrite_rows()
    sft_rows = build_sft_rows(rewrite_rows)
    train_rows, eval_rows = split_rows(sft_rows, eval_ratio=EVAL_RATIO, seed=SEED)

    write_jsonl(REWRITE_PATH, rewrite_rows)
    write_jsonl(ALL_PATH, sft_rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    unique_prompt_completion = {
        (row["meta"]["user_text"], row["completion"])
        for row in sft_rows
    }
    summary = {
        "rewrite_rows": len(rewrite_rows),
        "sft_rows": len(sft_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "unique_prompt_completion_pairs": len(unique_prompt_completion),
        "rewrite_path": str(REWRITE_PATH),
        "all_path": str(ALL_PATH),
        "train_path": str(TRAIN_PATH),
        "eval_path": str(EVAL_PATH),
        "source_type": "curated_clean_rewrite",
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
