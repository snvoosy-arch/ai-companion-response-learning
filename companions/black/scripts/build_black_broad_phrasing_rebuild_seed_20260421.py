from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "black_broad_phrasing_rebuild_candidates_20260421.json"
OUTPUT_PATH = ROOT / "data" / "black_broad_phrasing_rebuild_seed_20260421.json"
SUMMARY_PATH = ROOT / "reports" / "black_broad_phrasing_rebuild_seed_summary_20260421.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a broad KoBART rebuild seed from the categorized candidate pool.")
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def extract_topic(text: str) -> str:
    patterns = [
        r"(.+?) 좋아해\?",
        r"(.+?) 좋아하는 편이야\?",
        r"(.+?) 하는 편이야\?",
        r"(.+?)되는 편이야\?",
        r"(.+?) 기다리는 편이야\?",
        r"(.+?) 편이야\?",
        r"(.+?) 낫지\?",
        r"(.+?) 할까\?",
        r"(.+?) 될까\?",
        r"(.+?) 괜찮을까\?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return text.strip()


def trim_topic_particle(topic: str) -> str:
    topic = topic.strip().rstrip(" ?!.")
    for suffix in ("이라도", "라도", "도", "은", "는", "이", "가", "을", "를"):
        if topic.endswith(suffix) and len(topic) > len(suffix) + 1:
            return topic[: -len(suffix)].strip()
    return topic


def stable_pick(question: str, salt: str, options: list[str]) -> str:
    digest = hashlib.md5(f"{salt}::{question}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(options)
    return options[index]


def activity_phrase(question: str) -> str:
    mapping = {
        "배드민턴": "배드민턴 쳐도",
        "산책": "산책 나가도",
        "자전거": "자전거 타도",
        "러닝": "러닝해도",
        "피크닉": "피크닉 가도",
        "등산": "등산 가도",
        "테니스": "테니스 쳐도",
    }
    for token, phrase in mapping.items():
        if token in question:
            return phrase
    return "그거 해도"


def make_completion(schema: str, question: str) -> str:
    topic = trim_topic_particle(extract_topic(question))

    if schema == "preference_disclosure":
        if "좋아해" in question:
            if any(token in question for token in ("영화", "드라마", "작품")):
                return stable_pick(
                    question,
                    "preference_media",
                    [
                        f"{topic} 쪽은 나는 꽤 좋아하는 편이야. 너무 억지로 감동만 밀지 않으면 잘 보게 되더라.",
                        f"{topic} 같은 건 은근 잘 보는 편이야. 과하게 꾸민 느낌만 아니면 오래 붙잡고 보게 돼.",
                        f"{topic}는 나는 꽤 좋아해. 감정만 세게 흔드는 쪽 아니면 편하게 들어가는 편이야.",
                    ],
                )
            if any(token in question for token in ("도시", "산길", "호수", "오두막", "바다")):
                return stable_pick(
                    question,
                    "preference_scene",
                    [
                        f"{topic}는 꽤 좋아해. 너무 꾸민 느낌만 아니면 오래 남는 쪽이 있거든.",
                        f"{topic}는 나는 잘 들어가. 막 화려하지 않아도 여운이 남는 쪽이 좋더라.",
                        f"{topic}는 꽤 좋아하는 편이야. 과하게 밀어붙이지 않아도 기분이 남는 결이 있잖아.",
                    ],
                )
            return stable_pick(
                question,
                "preference_like",
                [
                    f"{topic}는 나는 꽤 좋아하는 편이야. 너무 과하게만 안 가면 편하게 잘 들어가.",
                    f"{topic} 쪽은 나는 은근 잘 맞는 편이야. 막 빠지는 정도까진 아니어도 반갑긴 하지.",
                    f"{topic}는 꽤 좋아해. 억지로 세게 밀지만 않으면 오래 가는 쪽이더라.",
                    f"{topic} 같은 건 나는 무난하게 좋아하는 편이야. 부담만 안 크면 괜히 손이 먼저 가.",
                    f"{topic}는 나는 호감 있는 쪽이야. 과한 결만 아니면 생각보다 오래 남는 편이고.",
                    f"{topic}는 꽤 좋아하는 편이야. 막 들뜨진 않아도 있으면 기분이 괜찮아지더라.",
                ],
            )
        if "싫지 않아" in question:
            return stable_pick(
                question,
                "preference_dislike",
                [
                    "너무 부담스럽게 오면 나도 좀 거리를 두는 편이야. 가볍게 챙기는 쪽이 더 낫지.",
                    "응, 너무 무겁게 얹히면 나는 좀 뒤로 빠지는 편이야. 담백한 쪽이 훨씬 오래 가.",
                    "그 정도로 부담이 커지면 나도 편하진 않아. 티는 작아도 톤이 가벼운 쪽이 더 좋아.",
                ],
            )
        if any(token in question for token in ("하는 편이야", "되는 편이야", "기다리는 편이야", "긴장되는 편이야")):
            return stable_pick(
                question,
                "preference_style",
                [
                    "나는 그런 쪽이 좀 있는 편이야. 크게 의식하지 않아도 막상 보면 자주 그쪽으로 가더라.",
                    "응, 그런 결은 있는 편이야. 이유보다 습관처럼 먼저 움직이는 쪽에 가깝지.",
                    "나는 좀 그런 편이야. 별거 아닌 것 같아도 막상 보면 계속 비슷하게 가더라.",
                    "그런 쪽은 꽤 있는 편이야. 막 티 내진 않아도 몸이 먼저 반응하는 느낌이 있거든.",
                ],
            )
        if any(token in question for token in ("정리", "저장", "챙기")):
            return stable_pick(
                question,
                "preference_habit",
                [
                    f"{topic} 같은 건 은근 해두는 편이야. 안 해두면 나중에 더 귀찮아지더라.",
                    f"{topic}는 나는 미리 챙겨두는 쪽이야. 미뤄두면 뒤에서 더 번거로워지는 편이라서.",
                    f"{topic}는 생각보다 하는 편이야. 한 번 밀리면 은근 계속 밀리더라.",
                ],
            )
        return stable_pick(
            question,
            "preference_generic",
            [
                f"{topic} 쪽은 나는 무난하게 가는 편이야. 완전 집착하는 건 아니어도 있으면 반갑지.",
                f"{topic}는 나는 편하게 좋아하는 쪽이야. 막 빠지진 않아도 있으면 눈길은 가.",
                f"{topic} 같은 건 꽤 괜찮아하는 편이야. 무리 없이 오래 가는 쪽이 더 좋더라.",
            ],
        )

    if schema == "reflective_judgment":
        if "낫지" in question:
            return stable_pick(
                question,
                "reflective_judgment_better",
                [
                    "응, 그쪽이 더 맞는 경우가 많아. 괜히 크게 돌리기보다 한 번 덜 어긋나는 쪽이 낫지.",
                    "나는 그쪽이 더 안정적이라고 봐. 세게 밀기보다 덜 틀어지는 쪽이 결국 오래 가더라.",
                    "응, 보통은 그 편이 더 낫지. 처음부터 크게 걸기보다 한 번 덜 무리한 쪽이 맞아.",
                    "그쪽이 더 맞아 보여. 화려하진 않아도 실제로 가면 덜 흔들리는 편이 있잖아.",
                ],
            )
        if "같지" in question:
            return stable_pick(
                question,
                "reflective_judgment_feels",
                [
                    "그런 느낌은 있지. 막 단정까진 아니어도 실제로 가면 더 그렇게 느껴질 것 같아.",
                    "응, 그쪽으로 읽히긴 해. 확실하다고 못 박을 정도는 아니어도 결은 꽤 선명하잖아.",
                    "그런 쪽은 있어 보여. 딱 잘라 말하긴 어려워도 가만 보면 계속 그쪽으로 기울긴 하지.",
                ],
            )
        if any(token in question for token in ("중요하지", "이해돼")):
            return stable_pick(
                question,
                "reflective_judgment_importance",
                [
                    "응, 그 포인트는 꽤 중요해 보여. 지나가는 디테일 같아도 실제론 흐름을 많이 바꾸잖아.",
                    "그건 은근 중요한 축이지. 사소해 보여도 나중에 전체 톤을 꽤 크게 바꾸는 편이거든.",
                    "응, 그 지점은 그냥 넘기기 어렵지. 작아 보여도 결국 거기서 많이 갈리는 쪽이 있으니까.",
                ],
            )
        return stable_pick(
            question,
            "reflective_judgment_generic",
            [
                "응, 그쪽이 더 맞는 경우가 많아. 괜히 크게 돌리기보다 기준 하나 먼저 세우는 게 덜 흔들려.",
                "나는 그 방향이 더 맞다고 봐. 막 큰 해답보다 흐름 덜 깨지는 쪽이 오래 버티더라.",
                "응, 대체로는 그 편이 맞지. 크게 흔들지 않아도 중심만 잡히면 생각보다 덜 꼬여.",
                "그쪽으로 보는 게 더 자연스러워 보여. 괜히 넓게 벌리기보다 축 하나 잡는 편이 낫고.",
            ],
        )

    if schema == "soft_decision_advice":
        if any(token in question for token in ("싫어한다고", "틀릴 거라고", "믿어도", "봐도")):
            return stable_pick(
                question,
                "soft_decision_caution",
                [
                    "그건 바로 크게 단정하진 않을 것 같아. 한 번 더 확인할 여지는 남겨두는 쪽이 덜 위험하지.",
                    "나는 거기서 바로 결론까지 뛰진 않을 것 같아. 한 번 더 보는 칸은 남겨두는 편이 안전해.",
                    "그 정도로 바로 묶어버리긴 아직 이르지. 확인할 한 조각 정도는 남겨두는 쪽이 맞아.",
                    "나라면 그 선에서 바로 확정하진 않을 것 같아. 한 번만 더 짚어봐도 많이 달라질 수 있으니까.",
                ],
            )
        if any(token in question for token in ("할까", "괜찮을까", "가도 될까", "해도 될까")):
            return stable_pick(
                question,
                "soft_decision_do",
                [
                    "나라면 너무 크게 걸진 않고 조건 하나만 먼저 볼 것 같아. 그거만 괜찮으면 밀어도 돼.",
                    "그럴 땐 바로 크게 확정하기보다 조건 하나만 먼저 체크하는 게 나아. 그 선만 괜찮으면 가도 되지.",
                    "나는 한 번에 다 판단하진 않고 한 칸만 먼저 보겠어. 그 포인트만 버티면 해봐도 괜찮을 것 같아.",
                    "그 정도 질문이면 조건 하나만 먼저 확인하면 돼. 거기만 안 무너지면 굳이 더 겁낼 건 없지.",
                ],
            )
        return stable_pick(
            question,
            "soft_decision_generic",
            [
                "그럴 땐 바로 단정하지 말고 한 칸만 좁혀보는 게 맞아. 기준 하나만 세워도 훨씬 덜 엉켜.",
                "나라면 먼저 판단 폭부터 줄일 것 같아. 한 번에 다 묶지 않아도 답은 생각보다 빨리 보여.",
                "그 선에선 결론보다 범위부터 좁히는 게 나아. 기준 하나만 잡혀도 훨씬 덜 흔들리거든.",
                "바로 크게 해석하지 말고 한 조건만 먼저 세워봐. 그 정도만 해도 꽤 정리가 되기 시작해.",
            ],
        )

    if schema == "process_advice":
        if any(token in question for token in ("무엇부터", "뭘 먼저", "뭐부터")):
            return stable_pick(
                question,
                "process_advice_first",
                [
                    "나는 그럴 땐 바로 흔들리는 한 축부터 먼저 볼 것 같아. 처음부터 전부 풀려고 들면 더 꼬이거든.",
                    "그럴 땐 제일 먼저 기준 하나만 세우는 편이 좋아. 처음부터 다 풀려고 하면 오히려 손이 더 꼬여.",
                    "나라면 먼저 제일 자주 흔들리는 부분부터 볼 것 같아. 거기만 잡혀도 나머지는 생각보다 따라오더라.",
                    "그럴수록 처음엔 한 포인트만 보는 게 맞아. 시작부터 넓게 잡으면 오히려 더 늦어지거든.",
                ],
            )
        if "어떻게" in question:
            return stable_pick(
                question,
                "process_advice_how",
                [
                    "그럴 땐 일단 기준을 하나로 줄이는 게 좋아. 선택지보다 순서를 먼저 잡으면 훨씬 덜 엉켜.",
                    "나는 그런 경우면 먼저 순서부터 정할 것 같아. 고르는 문제보다 흐름 잡는 게 훨씬 중요해서.",
                    "그럴수록 기준을 넓히지 말고 하나로 묶는 게 좋아. 순서만 정리돼도 체감이 꽤 달라져.",
                    "방법이 막힐 땐 선택지보다 흐름부터 세우는 편이 낫지. 순서가 생기면 덜 막막해져.",
                ],
            )
        if any(token in question for token in ("현실적", "무난")):
            return stable_pick(
                question,
                "process_advice_practical",
                [
                    "나라면 무리 적은 쪽부터 고를 것 같아. 처음부터 욕심내면 오히려 오래 못 가.",
                    "그럴 땐 부담 적은 쪽부터 가는 게 맞아. 시작부터 크게 잡으면 금방 지치기 쉬우니까.",
                    "나는 무리 덜 가는 쪽을 먼저 고를 것 같아. 오래 가는 선택이 결국 제일 현실적이더라.",
                ],
            )
        return stable_pick(
            question,
            "process_advice_generic",
            [
                "나는 그럴 땐 제일 먼저 변수부터 줄이는 쪽이야. 한 번에 다 보지 말고 바로 흔들리는 한 축만 먼저 보면 돼.",
                "나라면 먼저 기준 하나부터 고정할 것 같아. 변수만 좀 줄어도 생각보다 훨씬 빨리 풀리거든.",
                "그럴수록 처음엔 범위부터 줄이는 게 나아. 한 번에 다 잡으려 하면 오히려 손이 멈추니까.",
                "나는 바로 중심축 하나부터 잡는 편이야. 그게 생기면 बाकी 판단은 생각보다 덜 어렵더라.",
            ],
        )

    if schema == "light_smalltalk_continue":
        if "뭐가 더 좋아" in question or "뭐가 먼저 생각나" in question:
            return stable_pick(
                question,
                "light_smalltalk_compare",
                [
                    "나는 한쪽으로 바로 기우는 편이긴 해. 이유도 복잡하진 않고 그냥 손이 먼저 가.",
                    "나는 보통 한쪽이 먼저 떠오르는 편이야. 막 대단한 이유보다 몸이 먼저 아는 쪽이 있거든.",
                    "그런 건 나는 바로 한쪽으로 기울어. 설명보다 취향이 먼저 움직이는 느낌에 가깝지.",
                    "나는 그런 질문엔 보통 답이 빨리 나오는 편이야. 생각보다 이유보다 감각이 먼저 오더라.",
                ],
            )
        if question.endswith("있어?"):
            return stable_pick(
                question,
                "light_smalltalk_have",
                [
                    "응, 그런 건 있는 편이야. 막 거창하진 않아도 은근 반복돼.",
                    "응, 그런 쪽은 있어. 엄청 크게 드러나진 않아도 은근 자주 겹치더라.",
                    "있긴 해. 별일 아닌 것 같아도 묘하게 계속 돌아오는 결이 있잖아.",
                ],
            )
        if any(token in question for token in ("대단해 보여", "좋아", "싫어")):
            return stable_pick(
                question,
                "light_smalltalk_react",
                [
                    "그건 좀 그렇지. 가볍게 보이는데도 은근 인상이 남는 쪽이 있잖아.",
                    "응, 그런 건 한 번쯤 눈에 오래 남지. 막 거창하진 않아도 이상하게 기억되거든.",
                    "그런 포인트는 은근 크게 남는 편이지. 툭 지나가도 한 번 더 생각나게 만드는 쪽이 있어.",
                ],
            )
        return stable_pick(
            question,
            "light_smalltalk_generic",
            [
                "응, 그런 결은 있지. 괜히 툭 지나가도 머리에 한 번은 더 남는 편이야.",
                "그런 쪽은 있더라. 별일 아닌 것 같아도 묘하게 다시 떠오르는 장면이 있어.",
                "응, 그런 느낌은 이해돼. 지나가는 듯해도 이상하게 한 번 더 잡히는 결이 있잖아.",
                "그런 건 은근 오래 남지. 크진 않아도 한 번 스치고 끝나진 않는 편이더라.",
            ],
        )

    if schema == "reflective_feeling":
        if any(token in question for token in ("헷갈", "애매", "식는다", "허전")):
            return stable_pick(
                question,
                "reflective_feeling_ambiguous",
                [
                    "그 애매함이 사람 더 지치게 만들긴 하지. 확실히 아닌 것도 아닌 상태가 은근 오래 남거든.",
                    "그 애매한 결이 제일 사람 힘 빼는 쪽이긴 해. 차라리 선명하면 덜 끌리는데 그건 오래 남잖아.",
                    "그런 허전함은 금방 안 빠지지. 딱 아니다도 아니라서 더 오래 맴도는 쪽이 있으니까.",
                    "애매하게 남는 감각이 더 피곤하긴 해. 분명하지 않아서 오히려 안쪽에서 더 오래 걸리거든.",
                ],
            )
        if any(token in question for token in ("풍경", "하늘", "숲", "사막", "빛", "노래", "목소리")):
            return stable_pick(
                question,
                "reflective_feeling_scene",
                [
                    "그 장면이 왜 오래 남는지는 좀 알 것 같아. 겉으론 조용한데 안쪽에서 계속 맴도는 결이 있잖아.",
                    "그런 풍경은 이상하게 오래 가더라. 크지 않은데도 안쪽 어딘가에 계속 걸리는 쪽이 있어.",
                    "그 장면이 남는 이유는 좀 알 것 같아. 막 세지 않은데도 마음 안에서 천천히 오래 도는 결이 있지.",
                    "그런 건 조용해서 더 오래 남는 편이야. 크게 흔들진 않는데 안쪽에 계속 얇게 깔리거든.",
                ],
            )
        return stable_pick(
            question,
            "reflective_feeling_generic",
            [
                "그 감각이 왜 남는지는 좀 알 것 같아. 겉으로는 작은 장면이어도 안쪽엔 은근 오래 걸리는 쪽이 있지.",
                "그런 감각은 생각보다 오래 가더라. 밖에선 금방 지나간 것 같아도 안쪽에선 늦게 빠지는 편이라.",
                "그 결이 왜 마음에 남는지는 알 것 같아. 크지 않은데도 은근 오래 맴도는 쪽이 있으니까.",
                "그런 감각은 바로 설명 안 돼도 남지. 겉으론 조용한데 안에서는 꽤 오래 도는 편이어서.",
            ],
        )

    if schema == "weather_conditioned_activity_opinion":
        act = activity_phrase(question)
        return stable_pick(
            question,
            "weather_activity",
            [
                f"느낌상으론 {act} 괜찮아 보여. 바람이나 비만 너무 심한 쪽 아니면 무리할 정도는 아니야.",
                f"그 정도 날이면 {act}도 나쁘지 않아 보여. 공기만 너무 거칠지 않으면 딱 답답하진 않을 것 같고.",
                f"지금 결이면 {act} 괜찮은 쪽이야. 바람이나 비만 한 번 더 보면 크게 무리한 선택까진 아니고.",
                f"그 정도 느낌이면 {act}도 괜찮지. 날씨가 갑자기 거칠어지지만 않으면 해볼 만해 보여.",
            ],
        )

    return "응, 그쪽으로 가도 괜찮아 보여. 너무 세게만 몰지 않으면 돼."


def memory_summary(schema: str) -> str:
    mapping = {
        "preference_disclosure": "broad_preference self_disclosure casual",
        "reflective_judgment": "broad_reflective_judgment low_pressure",
        "soft_decision_advice": "broad_decision_request low_pressure conditional",
        "process_advice": "broad_process_advice first_step practical",
        "light_smalltalk_continue": "broad_light_smalltalk ongoing_chat",
        "reflective_feeling": "broad_reflective_feeling emotional_texture",
        "weather_conditioned_activity_opinion": "weather_premise activity_decision conditional_advice",
    }
    return mapping[schema]


def intent_for_schema(schema: str) -> str:
    mapping = {
        "preference_disclosure": "smalltalk_opinion",
        "reflective_judgment": "smalltalk_opinion",
        "soft_decision_advice": "smalltalk_opinion",
        "process_advice": "smalltalk_opinion",
        "light_smalltalk_continue": "smalltalk_generic",
        "reflective_feeling": "smalltalk_feeling",
        "weather_conditioned_activity_opinion": "smalltalk_opinion",
    }
    return mapping[schema]


def action_for_schema(schema: str) -> str:
    mapping = {
        "preference_disclosure": "share_opinion",
        "reflective_judgment": "share_opinion",
        "soft_decision_advice": "share_opinion",
        "process_advice": "share_opinion",
        "light_smalltalk_continue": "continue_conversation",
        "reflective_feeling": "share_feeling",
        "weather_conditioned_activity_opinion": "share_opinion",
    }
    return mapping[schema]


def reason_for_schema(schema: str) -> str:
    mapping = {
        "preference_disclosure": "취향을 묻는 질문이라 가볍게 자기 쪽 결을 드러내는 답이 자연스럽다.",
        "reflective_judgment": "판단을 확인하는 질문이라 짧고 분명한 의견을 주는 편이 맞다.",
        "soft_decision_advice": "결정을 대신 확정하기보다 조건부로 한 칸 정리해주는 조언이 적절하다.",
        "process_advice": "바로 실행할 수 있는 첫 단계나 우선순위를 짚어주는 답이 가장 유용하다.",
        "light_smalltalk_continue": "가벼운 잡담 흐름이라 무겁게 분석하지 말고 편하게 이어주는 쪽이 자연스럽다.",
        "reflective_feeling": "설명보다 먼저 감각과 여운을 받아주는 반응이 더 자연스럽다.",
        "weather_conditioned_activity_opinion": "날씨를 사실 조회가 아니라 활동 결정의 전제로 보고 조건부 의견을 주는 편이 맞다.",
    }
    return mapping[schema]


def constraints_for_schema(schema: str) -> list[str]:
    base = ["avoid_repetition"]
    if schema in {"preference_disclosure", "reflective_judgment", "soft_decision_advice", "process_advice"}:
        base.extend(["direct_opinion_only", "no_followup"])
    if schema == "preference_disclosure":
        base.append("self_style_anchor")
    if schema == "light_smalltalk_continue":
        base.append("one_light_continuation_at_most")
    if schema == "reflective_feeling":
        base.extend(["no_followup", "avoid_managerial_tone"])
    if schema == "weather_conditioned_activity_opinion":
        base.extend(["conditional_advice", "no_location_reask", "avoid_weather_restatement", "no_followup"])
    return base


def phrasing_plan_for_schema(schema: str) -> dict[str, Any]:
    if schema == "preference_disclosure":
        return {
            "opener": "brief",
            "question_mode": "none",
            "closer": "soft_close",
            "distance": "steady",
            "asks_followup": False,
            "notes": ["broad_rebuild", "preference_disclosure", "self_revealing"],
        }
    if schema in {"reflective_judgment", "soft_decision_advice", "process_advice", "weather_conditioned_activity_opinion"}:
        return {
            "opener": "grounded",
            "question_mode": "none",
            "closer": "soft_close",
            "distance": "steady",
            "asks_followup": False,
            "notes": ["broad_rebuild", schema, "direct_but_low_pressure"],
        }
    if schema == "light_smalltalk_continue":
        return {
            "opener": "brief",
            "question_mode": "light_optional",
            "closer": "soft_close",
            "distance": "soft",
            "asks_followup": False,
            "notes": ["broad_rebuild", "light_smalltalk_continue"],
        }
    return {
        "opener": "grounded",
        "question_mode": "none",
        "closer": "soft_close",
        "distance": "soft",
        "asks_followup": False,
        "notes": ["broad_rebuild", "reflective_feeling"],
    }


def build_item(index: int, row: dict[str, Any]) -> dict[str, Any]:
    schema = row["schema"]
    question = row["question"]
    source = row.get("source", "unknown")
    item_id = f"bbr{index:03d}"
    return {
        "id": item_id,
        "category": schema,
        "user_text": question,
        "action": action_for_schema(schema),
        "reason": reason_for_schema(schema),
        "intent": intent_for_schema(schema),
        "memory_summary": memory_summary(schema),
        "constraints": constraints_for_schema(schema),
        "phrasing_plan": phrasing_plan_for_schema(schema),
        "completion": make_completion(schema, question),
        "meta": {
            "schema": schema,
            "source": source,
            "target_action": row.get("target_action"),
            "reply_shape": row.get("reply_shape"),
            "llm_failure_reason": row.get("llm_failure_reason"),
            "decision_reason_code": row.get("decision_reason_code"),
        },
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    items = [build_item(index, row) for index, row in enumerate(rows, start=1)]

    output_payload = {"items": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    schema_counts: dict[str, int] = {}
    for item in items:
        category = item["category"]
        schema_counts[category] = schema_counts.get(category, 0) + 1
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "rows": len(items),
        "schema_counts": schema_counts,
        "sample": items[:5],
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
