from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "kobart_topic_diversity_repair_seed_20260417.json"


def _base_phrasing_plan(action: str, variant_index: int, category: str) -> dict[str, object]:
    if action == "continue_conversation":
        variants = [
            ("bridging", "soft", "keep_open", "steady", True),
            ("warm", "none", "soft_close", "soft", False),
            ("brief", "none", "soft_close", "steady", False),
            ("bridging", "soft", "keep_open", "soft", True),
        ]
    elif action == "share_feeling":
        variants = [
            ("grounded", "none", "soft_close", "steady", False),
            ("warm", "none", "soft_close", "soft", False),
            ("brief", "none", "soft_close", "steady", False),
            ("grounded", "none", "soft_close", "soft", False),
        ]
    elif action == "share_opinion":
        variants = [
            ("grounded", "none", "soft_close", "steady", False),
            ("informative", "none", "soft_close", "steady", False),
            ("brief", "none", "soft_close", "steady", False),
            ("grounded", "none", "soft_close", "soft", False),
        ]
    elif action in {"react_laugh", "react_surprise"}:
        variants = [
            ("reactive", "none", "none", "playful", False),
            ("reactive", "none", "none", "soft", False),
            ("brief", "none", "none", "playful", False),
            ("reactive", "none", "soft_close", "soft", False),
        ]
    else:
        variants = [
            ("grounded", "none", "soft_close", "steady", False),
            ("warm", "none", "soft_close", "soft", False),
            ("brief", "none", "soft_close", "steady", False),
            ("grounded", "none", "soft_close", "soft", False),
        ]

    opener, question_mode, closer, distance, asks_followup = variants[variant_index]
    return {
        "opener": opener,
        "question_mode": question_mode,
        "closer": closer,
        "distance": distance,
        "asks_followup": asks_followup,
        "notes": [category, f"variant_{variant_index + 1}", "topic_diversity"],
    }


def _expand_category(category: str, bases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base_index, base in enumerate(bases, start=1):
        action = str(base["action"])
        completions = list(base["completion_variants"])
        if len(completions) != 4:
            raise ValueError(f"{category} base {base_index} must contain 4 completion variants")
        for variant_index, completion in enumerate(completions):
            row_id = f"{category[:2]}{base_index:02d}v{variant_index + 1}"
            rows.append(
                {
                    "id": row_id,
                    "category": category,
                    "user_text": str(base["user_text"]),
                    "action": action,
                    "reason": str(base["reason"]),
                    "intent": str(base["intent"]),
                    "memory_summary": str(base["memory_summary"]),
                    "constraints": list(base.get("constraints") or []),
                    "phrasing_plan": _base_phrasing_plan(action, variant_index, category),
                    "completion": str(completion).strip(),
                }
            )
    return rows


def build_seed_rows() -> list[dict[str, object]]:
    daily_report_bases = [
        {
            "user_text": "할 일은 다 끝냈는데 마음은 아직 퇴근을 못 한 느낌이야.",
            "action": "continue_conversation",
            "intent": "smalltalk_generic",
            "reason": "일상 보고형 발화라서 상태를 가볍게 이어받는 답이 적절하다.",
            "memory_summary": "daily_report after_tasks_but_mind_still_running",
            "completion_variants": [
                "몸은 끝났는데 마음만 아직 책상 앞에 남아 있는 거네. 오늘은 특히 뭐가 제일 늦게 떨어져?",
                "하루는 끝났는데 머리는 아직 같은 칸에 있나 보다. 그런 날은 쉬는 쪽도 한 박자 늦지.",
                "끝은 났는데 안쪽은 아직 멈추지 않은 거지. 오늘이 좀 길게 붙어 있었나 봐.",
                "정리는 됐는데 기분은 아직 현장에 남아 있는 느낌이네. 그 잔여감이 은근 오래 가더라.",
            ],
        },
        {
            "user_text": "집에는 왔는데 이동한 피곤이 아직 몸에서 안 내려간다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "바깥 피로가 남아 있는 몸감각 공유라서 직접적인 수용이 맞다.",
            "memory_summary": "daily_report commute_fatigue_lingers",
            "completion_variants": [
                "이동이 끝나도 몸은 한참 뒤에 도착할 때가 있지. 지금은 그 늦은 피곤이 붙어 있는 거야.",
                "집에 들어와도 피곤이 바로 안 꺼질 수 있어. 바깥 긴장이 아직 몸쪽에 남은 거지.",
                "도착은 했는데 몸은 아직 덜 내려온 느낌이네. 그런 피곤은 생각보다 천천히 빠져.",
                "움직임은 멈췄는데 피곤은 계속 따라 들어온 거구나. 그게 은근 오래 붙을 때가 있어.",
            ],
        },
        {
            "user_text": "저녁은 조용한데 머릿속은 아직 낮 소음이 남아 있다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "공공장소 자극 뒤 감각 피로라서 소음 잔상을 받아주는 답이 자연스럽다.",
            "memory_summary": "daily_report residual_noise_after_busy_day",
            "completion_variants": [
                "밖은 조용해졌는데 안쪽은 아직 안 조용한 상태네. 그런 잔소음이 유난히 늦게 빠질 때가 있지.",
                "하루 소음이 머릿속에 한 겹 남아 있는 거구나. 조용한데도 쉬는 느낌이 안 나는 날이 있어.",
                "저녁은 가라앉았는데 머리는 아직 낮 시간에 걸려 있네. 그 차이가 더 지치게 만들지.",
                "지금은 소리가 없는 게 아니라 소리 여운만 남은 쪽인가 보다. 그래서 더 멍할 수 있어.",
            ],
        },
        {
            "user_text": "오늘은 별일 없었는데 이상하게 하루가 오래 붙잡힌다.",
            "action": "continue_conversation",
            "intent": "smalltalk_generic",
            "reason": "작은 하루 회고형 발화라서 장면을 더 꺼내게 이어 주는 답이 적절하다.",
            "memory_summary": "daily_report ordinary_day_lingering",
            "completion_variants": [
                "별일이 없었는데도 길게 남는 날이 있지. 오늘은 어느 장면이 제일 오래 남아 있어?",
                "큰 사건이 없어도 하루가 이상하게 길게 남을 때가 있더라. 오늘은 묘하게 안 떨어지는 날이었나 봐.",
                "그럴 때 있지. 평범했는데도 끝이 잘 안 나는 하루 같은 거.",
                "딱 집을 만한 일은 없는데 하루 전체가 오래 붙는 느낌이네. 그런 날은 설명하기도 애매하지.",
            ],
        },
        {
            "user_text": "메시지는 몇 개 안 쌓였는데 처리해야 할 기분이 계속 남아 있다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "일감 자체보다 잔여 압박을 느끼는 상태라서 압박감을 바로 받는 답이 맞다.",
            "memory_summary": "daily_report low_volume_but_persistent_pressure",
            "completion_variants": [
                "양보다 남아 있는 압박이 더 큰 상태네. 수는 적어도 마음은 계속 붙잡힐 수 있지.",
                "할 건 많지 않은데 마음이 계속 대기 상태인 거구나. 그게 은근 더 피곤하지.",
                "쌓인 건 적은데 끝난 느낌이 안 나는 거네. 그래서 자꾸 다시 떠오를 수 있어.",
                "양은 크지 않은데 처리 중인 기분이 안 꺼진 상태 같아. 그런 압박은 숫자랑 따로 놀더라.",
            ],
        },
    ]

    emotional_share_bases = [
        {
            "user_text": "오늘은 이유 없이 마음이 바닥에 가까운 쪽으로 붙어 있다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "이유를 특정하기 어려운 저하감 공유라서 해석보다 감각 수용이 적절하다.",
            "memory_summary": "emotional_share low_baseline_day",
            "completion_variants": [
                "이유를 딱 집지 못해도 마음이 아래로 붙는 날이 있지. 오늘은 그냥 기본값이 낮은 날인가 보다.",
                "설명은 안 되는데 바닥 쪽으로 기우는 감각이 있구나. 그런 날은 이유 찾는 것도 더 힘들지.",
                "왜 그런지 몰라도 가라앉는 날이 있지. 지금은 그 상태를 버티는 것만으로도 진 빠질 수 있어.",
                "특별한 계기 없이도 마음이 낮게 깔릴 때가 있어. 오늘은 그 결이 더 진한 날 같네.",
            ],
        },
        {
            "user_text": "사소한 말 하나가 생각보다 오래 남아서 좀 민망하다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "작은 민망함이 오래 남는 상태라서 과장 없이 받아주는 답이 맞다.",
            "memory_summary": "emotional_share lingering_embarrassment",
            "completion_variants": [
                "작은 말도 오래 남을 때가 있지. 지나간 뒤에 더 민망해지는 쪽이 특히 그래.",
                "그 정도였는데도 마음에는 오래 남았구나. 민망함은 나중에 더 커질 때가 있더라.",
                "크게 보일 일은 아닌데 혼자선 오래 붙잡히는 순간이 있지. 지금이 딱 그런 쪽인가 봐.",
                "사소했는데도 오래 남으면 괜히 더 부끄러워지지. 시간이 지나서 더 선명해지는 말이 있어.",
            ],
        },
        {
            "user_text": "다 괜찮은 척했는데 집에 오니까 갑자기 기운이 확 빠진다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "밖에서 버틴 뒤 집에서 무너지는 피로라서 뒤늦은 풀림을 직접 받는 답이 적절하다.",
            "memory_summary": "emotional_share collapse_after_holding_it_together",
            "completion_variants": [
                "밖에서는 붙들고 있었는데 혼자 되자마자 풀린 거네. 그렇게 한꺼번에 빠질 때가 있지.",
                "버티는 동안엔 모르다가 집에 와서야 기운이 꺼지는 날이 있어. 그게 더 허무하게 느껴질 수도 있어.",
                "괜찮은 척한 몫이 집에서 한 번에 돌아온 거지. 그러면 몸이 먼저 푹 꺼지더라.",
                "밖에서 다 들고 있었으니까 지금 더 비는 거구나. 그런 빠짐은 뒤늦게 크게 오지.",
            ],
        },
        {
            "user_text": "누가 다정하게 굴면 오히려 내가 더 어색해지는 날이 있다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "친절이 바로 위로로 안 닿는 상태라서 방어감과 어색함을 같이 받는 답이 자연스럽다.",
            "memory_summary": "emotional_share kindness_feels_awkward",
            "completion_variants": [
                "다정함이 바로 편안함으로 안 이어지는 날이 있지. 그럴 땐 고마움보다 어색함이 먼저 오기도 해.",
                "좋은 말인데도 몸이 바로 못 받는 날이 있더라. 그런 어색함이 꼭 이상한 건 아니야.",
                "다정한 쪽이 오히려 더 낯설게 느껴질 때가 있지. 마음이 늦게 따라오는 날인가 보다.",
                "친절이 부담처럼 느껴지는 날이 있어. 그게 상대 때문이라기보다 네 상태 때문일 수도 있고.",
            ],
        },
        {
            "user_text": "피곤한 건지 허전한 건지 구분이 잘 안 된다.",
            "action": "share_feeling",
            "intent": "smalltalk_feeling",
            "reason": "몸 피로와 감정 공백이 섞인 상태라서 둘 중 하나로 단정하지 않고 받아줘야 한다.",
            "memory_summary": "emotional_share tired_or_empty_unclear",
            "completion_variants": [
                "그 두 감각이 같이 섞여 있으면 구분이 잘 안 되지. 몸이 지친 건지 마음이 빈 건지 헷갈릴 수 있어.",
                "피곤함이랑 허전함이 겹치면 이름 붙이기부터 어려워져. 지금은 그 둘이 분리 안 되는 상태 같네.",
                "하나는 몸 쪽이고 하나는 마음 쪽인데 같이 오면 더 막막하지. 그래서 더 애매하게 느껴질 수 있어.",
                "그걸 정확히 나누기 어려운 날이 있지. 피곤함이 허전함처럼 느껴질 때도 있고 그 반대도 있으니까.",
            ],
        },
    ]

    topic_opinion_bases = [
        {
            "user_text": "지금 사과를 먼저 보내는 게 나을지 조금 더 두는 게 나을지 애매하다.",
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "reason": "타이밍에 대한 의견 요청이므로 감정 위로보다 판단을 짧게 주는 편이 맞다.",
            "memory_summary": "topic_opinion apology_timing",
            "completion_variants": [
                "지금 말이 정리돼 있으면 먼저 보내는 쪽이 더 낫다고 봐. 오래 끌수록 더 무거워질 수도 있어서.",
                "애매하면 하루 정도만 더 두고 정리하는 것도 괜찮아. 대신 너무 오래 미루진 않는 쪽이 낫지.",
                "내 쪽 감으론 길게 끄는 것보단 짧게 먼저 보내는 쪽이 덜 꼬일 것 같아.",
                "아직 말이 흐리면 조금만 더 두고, 문장이 잡혔으면 먼저 가는 편이 맞아 보여.",
            ],
        },
        {
            "user_text": "이번 주말엔 쉬는 게 맞는지 나가는 게 맞는지 계속 갈린다.",
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "reason": "휴식/외출 선택에 대한 의견 요청이므로 한쪽 판단을 짧게 주는 게 적절하다.",
            "memory_summary": "topic_opinion rest_vs_go_out",
            "completion_variants": [
                "지금 이미 갈리고 있다면 쉬는 쪽이 더 맞아 보여. 나가도 끝내 피곤만 남을 수 있어서.",
                "답답함이 더 큰 상태면 잠깐 나가는 것도 괜찮아. 다만 오래 있는 일정은 말고.",
                "내 기준엔 반나절만 밖에 나가고 나머진 쉬는 절충안이 제일 무난해 보여.",
                "이미 에너지가 낮으면 이번엔 쉬는 쪽이 더 낫다고 봐. 억지로 나가면 회복이 밀릴 수 있거든.",
            ],
        },
        {
            "user_text": "좋았던 영화인데 결말이 너무 깔끔해서 오히려 덜 남는 느낌이다.",
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "reason": "작품 인상에 대한 의견 공유라서 짧은 판단을 돌려주는 답이 자연스럽다.",
            "memory_summary": "topic_opinion movie_ending_too_neat",
            "completion_variants": [
                "그럴 수 있지. 너무 정리된 결말은 여운보다 정답지처럼 남을 때가 있어.",
                "나도 그런 쪽 이해돼. 좋아도 너무 매끈하면 오래 곱씹을 틈이 적지.",
                "깔끔해서 덜 남는 작품이 있더라. 조금 비어 있는 끝이 오히려 더 오래 가기도 하고.",
                "좋은데 오래 안 남는 이유가 딱 그거일 수 있어. 여운이 들어갈 틈까지 닫힌 거니까.",
            ],
        },
        {
            "user_text": "비싼 물건인데 요즘 계속 생각나는 거면 사는 게 맞는지 모르겠다.",
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "reason": "구매 판단 의견 요청이므로 간단한 기준을 주는 답이 적절하다.",
            "memory_summary": "topic_opinion expensive_purchase",
            "completion_variants": [
                "며칠 지나도 계속 생각나고 실제로 자주 쓸 물건이면 사는 쪽이 맞아 보여. 순간 기분만인지 먼저 보는 게 중요하고.",
                "생각나는 것만으로 바로 사기엔 아직 한 번 더 확인할 여지가 있어. 사용 빈도까지 확실하면 그때 가도 늦진 않지.",
                "내 기준엔 '자주 쓰나'가 제일 커. 그게 확실하면 비싸도 납득이 되더라.",
                "계속 떠오른다고 다 사야 하는 건 아니지만, 쓰임새가 선명하면 미루기만 하는 것도 답은 아니지.",
            ],
        },
        {
            "user_text": "오랜만에 만나는 사람은 차라리 짧게 보는 게 나은지 길게 보는 게 나은지 모르겠다.",
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "reason": "만남 방식에 대한 판단 요청이라서 무난한 선택을 직접 말해주는 편이 맞다.",
            "memory_summary": "topic_opinion reunion_length",
            "completion_variants": [
                "오랜만이면 처음엔 짧게 보는 쪽이 더 편해 보여. 분위기 괜찮으면 다음에 길게 가도 되고.",
                "부담이 있다면 짧은 만남이 낫지. 길게 잡을수록 어색함을 버텨야 하는 시간이 늘어나니까.",
                "내 쪽 감으론 짧게 시작하는 게 안전해. 다시 박자 맞는지 보기에도 좋고.",
                "이미 망설여진다면 길게 보기보단 가볍게 보는 쪽이 더 자연스러울 것 같아.",
            ],
        },
    ]

    light_banter_bases = [
        {
            "user_text": "오늘은 알람이랑 내가 서로 졌다 이긴다 하는 날이네 ㅋㅋ",
            "action": "react_laugh",
            "intent": "smalltalk_generic",
            "reason": "가벼운 자기놀림 장난이라 짧은 웃음 반응이 적절하다.",
            "memory_summary": "light_banter alarm_battle",
            "completion_variants": [
                "ㅋㅋ 그건 알람도 꽤 버텼네.",
                "오늘은 둘 다 진 쪽 아닌가 ㅋㅋ",
                "그 싸움은 아침마다 난전이지 ㅋㅋ",
                "알람도 너도 서로 봐준 날 같네 ㅋㅋ",
            ],
        },
        {
            "user_text": "이 정도면 내가 커피를 마시는 건지 커피가 나를 쓰는 건지 모르겠다.",
            "action": "react_laugh",
            "intent": "smalltalk_generic",
            "reason": "가벼운 과장 장난이라 설명 없이 짧게 받아치는 반응이 맞다.",
            "memory_summary": "light_banter coffee_overuse",
            "completion_variants": [
                "ㅋㅋ 오늘은 커피 쪽이 주도권 잡았네.",
                "그 정도면 이미 역할이 바뀐 거지 ㅋㅋ",
                "커피가 스케줄 짜주는 날 있지 ㅋㅋ",
                "그 말은 좀 인정된다 ㅋㅋ 오늘은 커피가 상사네.",
            ],
        },
        {
            "user_text": "메모장 켰다가 왜 켰는지 까먹는 건 진짜 고전이다.",
            "action": "continue_conversation",
            "intent": "smalltalk_generic",
            "reason": "가벼운 공감형 잡담이라 짧게 웃으며 이어받는 답이 적절하다.",
            "memory_summary": "light_banter forgot_why_opened_notes",
            "completion_variants": [
                "그건 진짜 자주 나오는 고전이지. 막상 다시 떠오르는 건 닫고 나서더라.",
                "너무 익숙해서 웃기지도 않은데 또 웃기긴 해. 보통 다시 생각나는 타이밍도 꼭 늦고.",
                "그건 현대인 기본 패시브 같아. 열었을 때보다 닫고 나서 더 생각나지 않냐.",
                "고전 맞지. 켜는 순간보다 닫으려는 순간에 더 급하게 떠오르더라.",
            ],
        },
        {
            "user_text": "방금 보낸 문장에 오타가 있는데 이미 읽혔을 것 같아서 웃기다.",
            "action": "react_surprise",
            "intent": "smalltalk_generic",
            "reason": "작은 사고를 웃으며 공유한 장면이라 짧은 놀람 반응이 자연스럽다.",
            "memory_summary": "light_banter typo_already_seen",
            "completion_variants": [
                "아 그건 이미 봤을 것 같은 타이밍이라 더 웃기네.",
                "헉 그 미묘하게 늦은 발견이 제일 웃기지.",
                "그건 수정해도 마음만 더 급해지는 구간인데 ㅋㅋ",
                "이미 읽혔을 것 같아서 더 손 못 대는 그 순간 있지.",
            ],
        },
        {
            "user_text": "냉장고 열고 한참 서 있다가 그냥 물만 꺼내는 나 자신이 좀 웃기다.",
            "action": "continue_conversation",
            "intent": "smalltalk_generic",
            "reason": "생활형 소소한 장난이라 짧게 받아치며 이어 주는 답이 적절하다.",
            "memory_summary": "light_banter fridge_staring",
            "completion_variants": [
                "그건 냉장고 앞에서 한 번 인생 회의하고 나온 거지. 결국 물로 돌아오는 결말까지 완벽하네.",
                "한참 들여다봤는데 물만 꺼내면 괜히 허무하게 웃기지. 뭔가 있을 줄 알았는데 없는 결말이라서.",
                "그 몇 초가 은근 길지 ㅋㅋ 결국 답은 늘 물인데 매번 다시 확인하게 되더라.",
                "냉장고 앞에서 옵션 고르다 그냥 물로 끝나는 거, 이상하게 자주 있는 장면이지.",
            ],
        },
    ]

    rows: list[dict[str, object]] = []
    rows.extend(_expand_category("daily_report", daily_report_bases))
    rows.extend(_expand_category("emotional_share", emotional_share_bases))
    rows.extend(_expand_category("topic_opinion", topic_opinion_bases))
    rows.extend(_expand_category("light_banter", light_banter_bases))
    return rows


def main() -> None:
    rows = build_seed_rows()
    payload = {
        "name": "kobart_topic_diversity_repair_seed_20260417",
        "version": "2026-04-17",
        "language": "ko",
        "purpose": [
            "Repair black KoBART topic fit and opener diversity.",
            "Increase category coverage across daily report, emotional share, topic opinion, and light banter.",
            "Provide richer phrasing targets for both full and slim input-mode evaluation.",
        ],
        "items": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        counts[category] = counts.get(category, 0) + 1
    print(f"output={OUTPUT_PATH}")
    print(f"rows={len(rows)}")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
