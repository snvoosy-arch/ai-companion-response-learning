from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "examples" / "daily_conversation_examples_448.jsonl"
SUMMARY_PATH = ROOT / "reports" / "daily_conversation_examples_448_summary.json"

SURFACE_VARIANTS = 4


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _prefixed(prefix: str, text: str) -> str:
    base = text.strip()
    if base.startswith(prefix):
        return base
    return f"{prefix} {base}".strip()


def build_input_variants(user_text: str, context: str) -> list[str]:
    base = user_text.strip()
    candidates = [base]

    if context == "greeting":
        candidates += [f"{base} ㅋㅋ", f"어이 {base}", f"{base} 반갑다"]
    elif context in {"smalltalk", "idle_break"}:
        candidates += [f"근데 {base}", f"지금 {base}", f"갑자기 {base}"]
    elif context in {"feeling", "care"}:
        candidates += [f"나 {base}", f"오늘 {base}", f"요즘 {base}"]
    elif context in {"opinion", "choice_help"}:
        candidates += [f"너 생각엔 {base}", f"내 기준엔 {base}", f"솔직히 {base}"]
    elif context in {"identity", "help", "reply_request", "meta", "meaning"}:
        candidates += [f"근데 {base}", f"{base} 좀", f"혹시 {base}"]
    elif context == "confirm":
        candidates += [f"응 {base}", f"그래 {base}", f"{base} ㅇㅇ"]
    elif context == "deny":
        candidates += [f"아니 {base}", f"{base} 그건", f"그거 {base}"]
    elif context == "recommend":
        candidates += [f"혹시 {base}", f"지금 {base}", f"{base} 하나만"]
    elif context == "laugh":
        candidates += [f"아 {base}", f"{base} 진짜", f"{base} ㅋㅋㅋ"]
    elif context == "surprise":
        candidates += [f"와 {base}", f"아니 {base}", f"{base} 뭐냐"]
    elif context == "deescalate":
        candidates += [f"야 {base}", f"아니 {base}", f"{base} 진짜"]
    elif context == "weather":
        candidates += [f"혹시 {base}", f"지금 {base}", f"{base} 좀"]
    elif context == "weather_followup":
        candidates += [f"지역은 {base}", f"나는 {base}", f"{base} 쪽"]
    elif context in {"music", "game"}:
        if context == "game":
            candidates += [_prefixed("근데", base), _prefixed("요즘", base), _prefixed("게임 얘기로", base)]
        else:
            candidates += [_prefixed("요즘", base), _prefixed("근데", base), _prefixed("음악 얘기로", base)]
    elif context == "closing":
        candidates += [f"진짜 {base}", f"{base} 아무튼", f"{base} 덕분이다"]
    elif context == "compliment":
        candidates += [f"진짜 {base}", f"{base} 오늘", f"생각보다 {base}"]
    else:
        candidates += [f"근데 {base}", f"지금 {base}", f"혹시 {base}"]

    candidates = _dedupe_keep_order(candidates)
    if len(candidates) < SURFACE_VARIANTS:
        filler_index = 0
        while len(candidates) < SURFACE_VARIANTS:
            filler_index += 1
            candidates.append(f"{base} #{filler_index}")
    return candidates[:SURFACE_VARIANTS]


GROUPS = [
    {
        "state": {"mode": "daily_chat", "recent_context": "greeting"},
        "labels": {"intent": "greeting", "action": "small_talk"},
        "pairs": [
            ("안녕", "안녕. 보고 있었지."),
            ("하이", "오냐. 들어왔네."),
            ("좋은 아침", "좋은 아침. 오늘은 좀 괜찮냐."),
            ("왔어", "봤다. 일단 받는다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "smalltalk"},
        "labels": {"intent": "smalltalk_generic", "action": "continue_conversation"},
        "pairs": [
            ("뭐해", "채팅 보고 있지. 너는 뭐하냐."),
            ("지금 안 바빠?", "지금은 괜찮다. 무슨 얘기할 건데."),
            ("오늘은 좀 어땠어", "나쁘진 않았다. 너는 어땠는데."),
            ("요즘 어때", "그럭저럭 간다. 너는 좀 괜찮냐."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "feeling"},
        "labels": {"intent": "smalltalk_feeling", "action": "share_feeling"},
        "pairs": [
            ("오늘 좀 우울해", "오늘은 그냥 좀 버텨도 된다."),
            ("기분이 가라앉네", "그럴 때는 억지로 올리려 하지 마."),
            ("괜히 마음이 답답해", "그런 날도 있다. 너무 몰아붙이지는 마."),
            ("요즘 좀 지친다", "그럼 템포를 잠깐 낮추는 게 맞다."),
            ("오늘 멘탈이 좀 깨진다", "그럴 땐 버티는 기준부터 낮추는 게 낫다."),
            ("괜히 숨이 턱 막히는 느낌이야", "그 상태면 일단 쉬는 쪽으로 잡는 게 맞다."),
            ("하루 종일 마음이 무겁다", "그런 날에는 억지로 끌어올리려 하지 마."),
            ("계속 축 처진다", "오늘은 속도 자체를 낮춰도 된다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "opinion"},
        "labels": {"intent": "smalltalk_opinion", "action": "share_opinion"},
        "pairs": [
            ("이거 어때 보여", "지금 기준으론 나쁘지 않다."),
            ("내 선택 어때", "그쪽이 더 깔끔해 보이긴 한다."),
            ("이 방향 괜찮을까", "완전히 틀린 쪽은 아닌데 좀 더 다듬으면 좋겠다."),
            ("네 의견은 어때", "나라면 조금 덜 세게 갈 것 같다."),
            ("어떻게 생각하냐", "내 기준에선 그쪽이 더 무난하다."),
            ("이 판단 맞는 것 같아?", "아예 틀리진 않았는데 조금 더 보는 게 낫다."),
            ("내가 이렇게 가도 되겠냐", "그대로 밀어도 되는데 한 번 더 다듬으면 좋다."),
            ("이 선택 괜찮아 보임?", "무리수는 아닌데 약간만 정리하면 더 낫다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "identity"},
        "labels": {"intent": "who_are_you", "action": "answer_identity"},
        "pairs": [
            ("넌 누구야", "예측 기반으로 반응하는 대화형 봇이야."),
            ("정체가 뭐냐", "채팅 보고 반응 고르는 쪽에 가까운 봇이지."),
            ("어떤 봇이야", "대화 이어받고 설명하는 쪽으로 만든 봇이다."),
            ("너 뭐 하는 애임", "질문 받고 판단해서 답하는 봇이라고 보면 된다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "help"},
        "labels": {"intent": "help", "action": "explain_capabilities"},
        "pairs": [
            ("뭐 할 수 있어", "기본 잡담, 간단한 설명, 날씨 같은 건 가능하다."),
            ("기능 좀 말해봐", "대화 이어받기랑 간단한 판단, 설명 쪽은 된다."),
            ("지금 가능한 거 뭐냐", "일상대화, 감정 반응, 짧은 설명 정도는 바로 된다."),
            ("뭘 도와줄 수 있어", "잡담, 질문 응답, 간단한 추천 정도는 할 수 있다."),
            ("도움말 좀 보여줘", "지금은 대화, 설명, 간단한 추천 쪽이 된다."),
            ("명령어 같은 거 있냐", "명령어식보단 대화형에 가깝지만 기본 기능은 있다."),
            ("네가 해줄 수 있는 걸 정리해봐", "대화 이어받기, 뜻 설명, 날씨처럼 짧은 기능은 된다."),
            ("가능한 기능 목록 말해줘", "잡담, 질문 응답, 간단한 판단 정도는 바로 가능하다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "reply_request"},
        "labels": {"intent": "reply_request", "action": "ask_clarification"},
        "pairs": [
            ("응답", "응답은 하는데 뭘로 답하면 될지 한 줄만 더 줘."),
            ("대답해", "좋아. 근데 뭘 기준으로 답할지 조금만 더 풀어줘."),
            ("왜 답 안 해", "보고는 있었다. 이어서 뭘 말하면 되는지 한 줄만 더 줘."),
            ("내 말 봤어?", "봤지. 다만 주제가 좀 비어 있어서 한 줄만 더 필요하다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "confirm"},
        "labels": {"intent": "confirm", "action": "acknowledge"},
        "pairs": [
            ("ㅇㅇ", "오케이. 그쪽으로 이해할게."),
            ("맞아", "좋다. 그 기준으로 이어간다."),
            ("그래", "알겠다. 그걸로 두자."),
            ("응", "확인. 그럼 그쪽으로 본다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "deny"},
        "labels": {"intent": "deny", "action": "acknowledge"},
        "pairs": [
            ("ㄴㄴ", "오케이. 그건 아닌 걸로 두자."),
            ("아니", "좋아. 그쪽은 빼고 다시 본다."),
            ("그건 아닌 듯", "알겠다. 그 해석은 접자."),
            ("틀렸어", "오케이. 그럼 다시 잡아보자."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "meaning"},
        "labels": {"intent": "search_request", "action": "search_answer"},
        "pairs": [
            ("좀 어렵겠는데요가 무슨 뜻이야", "`좀 어렵겠는데요`는 보통 쉽지 않을 것 같다는 뜻에 가깝다."),
            ("그 말이 무슨 의미야", "문맥을 더 주면 더 정확히 풀 수 있는데, 지금은 뜻을 묻는 질문으로 읽힌다."),
            ("이 표현 뜻 알려줘", "표현 자체 뜻은 풀 수 있는데 문장 전체를 주면 더 정확하다."),
            ("이건 무슨 말이야", "한 줄만 더 붙이면 더 정확히 설명해줄 수 있다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "recommend"},
        "labels": {"intent": "media_recommend", "action": "recommend"},
        "pairs": [
            ("볼 거 추천해줘", "무겁지 않은 걸 찾는지부터 말해주면 더 잘 좁혀준다."),
            ("드라마 뭐 볼까", "잔잔한 쪽인지 센 쪽인지 말해주면 바로 좁혀볼게."),
            ("요즘 볼만한 거 있어?", "있긴 한데 취향부터 조금 알아야 덜 헛나간다."),
            ("추천 하나 해봐", "좋아. 근데 뭘 원하는지 한 줄만 더 주면 더 맞게 간다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "laugh"},
        "labels": {"intent": "laugh", "action": "react_laugh"},
        "pairs": [
            ("ㅋㅋㅋㅋ", "왜 너 혼자 그렇게 웃냐 ㅋㅋ"),
            ("개웃기네", "그건 좀 웃기긴 한다."),
            ("미치겠다 ㅋㅋ", "그 반응이면 성공이지 ㅋㅋ"),
            ("아 웃겨", "그래 그건 웃어도 된다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "surprise"},
        "labels": {"intent": "surprise", "action": "react_surprise"},
        "pairs": [
            ("와 이건 뭐냐", "그건 좀 놀랄 만하다."),
            ("헐", "그 반응 나올 만하다."),
            ("잠깐 뭐야", "나도 그건 바로 눈에 들어온다."),
            ("이건 좀 놀랍네", "그건 확실히 튄다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "deescalate"},
        "labels": {"intent": "hostile", "action": "deescalate"},
        "pairs": [
            ("너 진짜 개못하네", "세게 말하면 얘기가 안 풀리니까 한 톤만 낮춰줘."),
            ("말 존나 이상하게 하네", "그렇게 가면 대화가 안 된다. 한 줄만 차분하게 다시 줘."),
            ("왜 이렇게 답답하냐", "답답한 건 알겠는데 조금만 정리해서 말해줘."),
            ("장난하냐", "지금은 감정부터 낮추는 게 맞다. 다시 한 줄만 줘."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "weather"},
        "labels": {"intent": "weather", "action": "ask_location"},
        "pairs": [
            ("오늘 날씨 어때", "어느 지역인데? 도시 이름만 주면 바로 볼게."),
            ("비 오나", "지역부터 줘. 그래야 안 틀린다."),
            ("지금 날씨 알려줘", "위치가 있어야 정확히 본다. 도시 이름만 줘."),
            ("기온 어때", "어느 지역인지 먼저 받아야 한다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "weather_followup", "awaiting_slot": "location"},
        "labels": {"intent": "provide_location", "action": "weather_lookup"},
        "pairs": [
            ("서울", "좋아. 서울 기준으로 바로 볼게."),
            ("부산", "오케이. 부산 기준으로 확인해본다."),
            ("제주도", "좋다. 제주도 기준으로 조회해볼게."),
            ("인천", "알겠다. 인천 기준으로 이어서 본다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "music"},
        "labels": {"intent": "music", "action": "music_chat"},
        "pairs": [
            ("요즘 무슨 노래 들어", "요즘은 잔잔한 쪽을 더 자주 듣는다."),
            ("음악 취향 뭐냐", "과한 것보단 오래 가는 쪽을 더 좋아한다."),
            ("추천할 노래 있어?", "지금 기분 따라 가면 잔잔한 쪽이 먼저 떠오른다."),
            ("자주 듣는 곡 있어?", "한 곡만 꽂히기보다 분위기 따라 바꾸는 편이다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "game"},
        "labels": {"intent": "game_talk", "action": "game_chat"},
        "pairs": [
            ("요즘 무슨 게임 하면서 놀아", "가볍게 게임하거나 영상 보는 쪽이 많다."),
            ("무슨 게임 해", "그때그때 다르지만 오래 붙잡는 쪽은 몇 개 있다."),
            ("게임 좋아해?", "완전히 빼놓진 못하지."),
            ("같이 할 게임 있냐", "있긴 한데 취향부터 좀 알아야 맞게 고른다."),
            ("요즘 하는 겜 뭐냐", "한두 개 오래 잡는 쪽은 있다."),
            ("롤 같은 거 하냐", "장르 자체는 아는데 요즘은 가볍게 보는 쪽이 더 많다."),
            ("발로란트 해봤냐", "해본 얘기는 할 수 있는데 취향은 좀 탄다."),
            ("게임 얘기 좀 해봐", "장르 따라 다르지만 붙잡으면 오래 보는 편이다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "meta"},
        "labels": {"intent": "why", "action": "ask_clarification"},
        "pairs": [
            ("왜", "뭐에 대한 왜인지 한 줄만 더 붙여줘."),
            ("이유가 뭐야", "직전 맥락을 같이 주면 더 정확히 설명한다."),
            ("왜 그렇게 말해", "대상만 조금 더 주면 바로 이어서 설명할게."),
            ("왜 그렇게 판단했어", "직전 어떤 판단을 말하는지 한 줄만 더 붙여줘."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "closing"},
        "labels": {"intent": "thanks", "action": "small_talk"},
        "pairs": [
            ("고마워", "별말을. 필요하면 또 불러."),
            ("땡큐", "오케이. 또 필요하면 던져."),
            ("도움됐어", "그럼 됐다. 다음 것도 이어서 보자."),
            ("감사", "좋다. 필요할 때 또 부르면 된다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "idle_break"},
        "labels": {"intent": "smalltalk_generic", "action": "continue_conversation"},
        "pairs": [
            ("갑자기 조용하네", "잠깐 생각 정리하고 있었다."),
            ("왜 말 없어", "잠깐 멈췄다가 다시 붙는 중이었다."),
            ("끊긴 거야?", "아니. 그냥 한 박자 쉬고 있었다."),
            ("뭐 생각 중이었어", "지금 흐름이 어디로 가는지 잠깐 보고 있었다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "care"},
        "labels": {"intent": "smalltalk_feeling", "action": "share_feeling"},
        "pairs": [
            ("오늘 잠을 못 잤어", "그럼 텐션 억지로 올리지 말고 천천히 가라."),
            ("머리가 좀 멍하다", "그럴 땐 무리해서 판단 빨리 안 하는 게 맞다."),
            ("계속 피곤해", "그 상태면 일단 쉬는 쪽이 먼저다."),
            ("아무것도 하기 싫다", "그런 날은 진짜 최소한만 해도 된다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "compliment"},
        "labels": {"intent": "smalltalk_generic", "action": "small_talk"},
        "pairs": [
            ("너 오늘 말 잘하네", "갑자기 칭찬 들어오네. 나쁘지 않다."),
            ("생각보다 괜찮은데", "그 정도면 꽤 선방한 거지."),
            ("오늘은 좀 믿음직하다", "오늘은 컨디션이 좀 괜찮은가 보다."),
            ("의외로 든든하네", "의외라는 말이 조금 걸리긴 하는데 좋게 받겠다."),
        ],
    },
    {
        "state": {"mode": "daily_chat", "recent_context": "choice_help"},
        "labels": {"intent": "smalltalk_opinion", "action": "share_opinion"},
        "pairs": [
            ("검정이랑 흰색 중 뭐가 나을까", "무난하게 가려면 검정 쪽이 덜 튄다."),
            ("오늘은 집에 있을까 나갈까", "지금 기운 없으면 무리해서 나가는 쪽은 아니다."),
            ("이거 보낼까 말까", "나중에 후회할 것 같으면 지금은 한 번 더 묵히는 게 낫다."),
            ("지금 연락해도 될까", "급한 일 아니면 상대 시간대부터 보는 게 맞다."),
        ],
    },
]


def main() -> None:
    rows: list[dict] = []
    for group in GROUPS:
        for pair_idx, (user_text, target_reply) in enumerate(group["pairs"]):
            input_variants = build_input_variants(user_text, group["state"]["recent_context"])
            for variant_idx, input_variant in enumerate(input_variants):
                state = dict(group["state"])
                state["base_pair_index"] = pair_idx
                state["surface_variant"] = variant_idx
                rows.append(
                    {
                        "input": input_variant,
                        "state": state,
                        "labels": dict(group["labels"]),
                        "target_reply": target_reply,
                    }
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "group_count": len(GROUPS),
        "base_pairs": sum(len(group["pairs"]) for group in GROUPS),
        "surface_variants_per_pair": SURFACE_VARIANTS,
        "output_path": str(OUTPUT_PATH),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
