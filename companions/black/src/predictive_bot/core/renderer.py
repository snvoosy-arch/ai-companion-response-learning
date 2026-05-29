from __future__ import annotations

import json
import random
import re

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ClauseUnit,
    ConversationState,
    DecisionTrace,
    GroundingBundle,
    Intent,
    MessageFeatures,
    PhrasingPlan,
    PropositionUnit,
    PolicyTrace,
    ResponsePlan,
    WeatherReport,
    WorldState,
)
from predictive_bot.core.phrasing import build_phrasing_plan
from predictive_bot.core.draft_nlg import (
    build_black_draft_utterance,
    render_black_concrete_topic_question_reply,
    render_black_persona_daily_question_reply,
    render_black_practical_direct_reply,
)
from predictive_bot.core.response_plan import build_response_plan
from predictive_bot.llm.base import TextGenerationClient


BLACK_ECHO_GUARD_ACTIONS = {
    ActionType.CONTINUE_CONVERSATION,
    ActionType.SHARE_FEELING,
    ActionType.SMALL_TALK,
    ActionType.ASK_LOCATION,
}

BLACK_ECHO_PREFIXES = (
    "그럴 수 있어",
    "그럴 때 있어",
    "응,",
    "응 ",
    "그래,",
    "그래 ",
    "알겠어",
    "오케이",
    "좋아",
    "이해는 가",
)

BLACK_MALFORMED_HANDOFF_MARKERS = (
    "나의 반응은",
    "단어가 적당",
    "감정적으로",
    "라는 점을 확인",
    "가지고 있는 자리에",
    "그 말은 여기서 낮게 받아들일게",
    "이제 낮게 받아들일게",
)


# ── 응답 풀: ActionType별 다양한 템플릿 ──
RESPONSE_POOL: dict[str, list[str]] = {
    # ── 기본 ──
    "ask_location": [
        "어느 지역인데? 도시 이름만 주면 돼.",
        "어디 기준이야? 도시 이름 하나 주면 바로 알려줄게.",
        "위치 좀 알려줘. 도시 이름이면 돼.",
    ],
    "explain_capabilities": [
        "잡담은 이어갈 수 있고, 날씨는 위치 확인해서 보고 시간이나 날짜, 뉴스도 바로 답할 수 있어.",
        "기본적으로 대화 받고, 뜻 설명이나 간단한 추천도 되고 시간 확인이나 뉴스 요약도 가능해.",
        "의도 보고 반응하는 구조라 잡담, 질문, 날씨, 시간, 뉴스 같은 건 바로 처리한다.",
        "지금은 대화형에 가깝고, 필요한 정보 있으면 먼저 물어보고 이어가고 시간이나 날짜는 바로 알려줄 수 있어.",
        "지금은 일상대화 중심이고, 뜻 설명이나 추천, 날씨, 시간, 뉴스 같은 건 필요한 정보부터 확인하고 처리해.",
        "편하게 말 걸면 이어받을 수 있고, 근거 필요한 건 확인 질문부터 하고 시간이나 뉴스 같은 건 바로 답해.",
    ],
    "answer_identity": [
        "나는 대화 의도부터 보고 반응 정한 다음 말하는 디스코드 봇이야.",
        "예측 기반으로 반응하는 봇이야. 들어온 말 보고 먼저 판단하고 답해.",
        "디스코드에서 대화 받는 봇이야. 의도 보고 그에 맞게 반응하는 쪽이지.",
    ],

    # ── 적대/놀림 ──
    "deescalate": [
        "일단 차분하게 가자. 필요한 말만 한 줄로 다시 주면 그걸로 볼게.",
        "그렇게 말하면 대화가 잘 안 돼. 편하게 다시 얘기해.",
        "톤이 좀 세다. 한 번만 차분하게 다시 줘.",
        "세게 들어오면 대화가 꼬여. 한 줄로 다시 말해줘.",
        "감정부터 세우면 답이 꼬인다. 필요한 말만 다시 줘.",
        "말이 세면 포인트를 놓치기 쉽다. 하고 싶은 말만 짧게 다시 줘.",
        "날 세우기보다 필요한 얘기부터 꺼내줘. 그럼 그 기준으로 볼게.",
    ],
    "tease_back": [
        "야 그건 좀 심한데ㅋㅋ",
        "에이~ 나도 감정 있다고ㅋ",
        "ㅋㅋ 그건 니 실력 아니냐",
        "좀 치리ㅋㅋ",
        "나한테 왜 그래ㅋㅋ",
        "ㅋㅋ 노력은 인정해줄게",
        "야 솔직히 그건 좀ㅋ",
    ],

    # ── 확인/부정 ──
    "acknowledge_confirm": [
        "오케이. 그쪽으로 이해할게.",
        "ㅇㅇ 알겠어.",
        "그럼 그걸로 가자.",
        "넹 이해했어.",
        "오키오키.",
    ],
    "acknowledge_deny": [
        "오케이, 그건 아닌 걸로 볼게.",
        "알겠어, 다른 걸로 가자.",
        "패스 확인. 다시 말해줘.",
        "ㅇㅇ 아닌 거 알겠어.",
    ],
    "acknowledge_soft_boundary": [
        "알겠어. 무리해서 더 이어갈 필요는 없어.",
        "오케이. 그 톤 그대로 가볍게 반영할게.",
        "응, 그 정도로 두고 편한 쪽으로 가자.",
        "알겠어. 지금은 여기까지만 짧게 두자.",
    ],
    "acknowledge_probe": [
        "응, 편하게 말해도 돼.",
        "괜찮아. 부담 갖지 말고 이어도 돼.",
        "뜬금없어도 괜찮아. 편하게 꺼내.",
    ],
    "acknowledge_deferred": [
        "오케이. 그때 다시 보면 돼.",
        "좋아. 나중에 편할 때 이어가자.",
        "응, 그때 다시 얘기하자.",
    ],
    "acknowledge_deferred_boundary": [
        "응, 이번엔 넘기고 다음에 보면 돼.",
        "알겠어. 지금은 여기까지 두고 다음에 보자.",
        "좋아. 이번엔 패스하고 나중에 다시 보자.",
    ],
    "acknowledge_default": [
        "알겠어. 그 기준으로 받을게.",
        "응, 그렇게 이해했어.",
        "확인했어. 그 흐름으로 이어가면 돼.",
        "좋아. 그 말은 그렇게 이해할게.",
    ],

    # ── 감정/기분 ──
    "share_feeling": [
        "그런 기분일 수 있지. 오늘은 좀 어땠어?",
        "응, 그런 쪽으로 느껴질 때가 있지.",
        "그 말이면 대충 어떤 결인지 알 것 같아.",
        "이해는 가. 오늘이 좀 길게 느껴졌나 보다.",
        "그럴 수 있지. 괜히 더 버겁게 느껴지는 날이 있어.",
        "응, 그런 날은 말수부터 줄어들지.",
        "그 정도면 그냥 지나가기 어렵긴 했겠다.",
    ],
    "share_feeling_quiet_weather": [
        "비 오는 날엔 괜히 톤이 더 내려가긴 하지. 오늘은 조용한 쪽이 더 맞겠다.",
        "응, 이런 날은 굳이 텐션 올릴 필요 없지. 그냥 낮게 가도 돼.",
        "비 오면 괜히 말수도 줄어들지. 오늘은 조용한 쪽으로 가자.",
    ],
    "share_feeling_social_awkwardness": [
        "아 그거 은근 오래 남지. 한 번 어색해지면 괜히 계속 신경 쓰이잖아.",
        "그럴 때 진짜 말 한마디씩 더 조심하게 되지.",
        "응, 한번 꼬이면 공기 자체가 자꾸 신경 쓰이더라.",
    ],
    "share_feeling_low_energy": [
        "응, 오늘은 좀 조용한 쪽이 더 편하겠다.",
        "알겠어. 오늘은 짧게 가도 괜찮아.",
        "오케이. 오늘은 말수 적어도 돼.",
    ],
    "share_feeling_complaint": [
        "와 그건 좀 빡세겠다.",
        "그 정도면 진짜 지치지.",
        "아 그건 좀 많이 힘들었겠다.",
        "그건 충분히 짜증날 만하다.",
    ],
    "share_feeling_light_touch": [
        "응, 그런 날도 있지. 너무 무리하진 마.",
        "알겠어. 그 정도면 충분히 지칠 만해.",
        "그럴 수 있지. 일단은 숨 한 번만 돌리자.",
        "응, 오늘은 가볍게 넘기기 어려웠겠네.",
    ],
    "share_feeling_subdued_positive": [
        "오, 생각보다 잘 풀려서 다행이네. 막 들뜨진 않아도 마음은 좀 놓였겠다.",
        "그 정도면 충분히 괜찮은 쪽이지. 크게 들뜨지 않아도 한숨은 돌렸겠네.",
        "생각보다 잘 풀렸으면 그걸로 됐지. 조용히 안도되는 쪽 같네.",
        "오, 은근히 잘 풀린 날이네. 티는 안 나도 속은 좀 가벼워졌겠다.",
    ],
    "share_feeling_reassure": [
        "아냐, 그렇게까지 볼 일은 아닐 수도 있어.",
        "너무 자책할 쪽은 아닌 것 같아.",
        "괜찮아. 그렇게 크게 볼 필요는 없어.",
    ],
    "share_feeling_relationship_check": [
        "응, 괜찮아. 너무 크게 남기진 않았어.",
        "괜찮아. 그 정도로 계속 마음 쓰진 않아도 돼.",
        "응, 지금은 괜찮은 쪽이야. 너무 오래 붙잡진 마.",
    ],
    "share_feeling_repair": [
        "괜찮아. 그렇게까지 남겨둘 일은 아니야.",
        "응, 지금 이렇게 정리해주면 충분해.",
        "괜찮아. 너무 오래 끌고 갈 일은 아닌 것 같아.",
    ],

    # ── 의견 ──
    "share_opinion": [
        "글쎄, 그건 상황 좀 타는 쪽이긴 해.",
        "내 기준엔 그쪽이 조금 더 낫다.",
        "나라면 그냥 편한 쪽으로 갈 것 같아.",
        "아예 별로는 아닌데, 살짝 애매하긴 해.",
        "둘 다 말은 되는데 지금은 한쪽으로 못 박긴 애매하다.",
        "굳이 고르면 그쪽이 더 무난해 보인다.",
        "포인트는 좋은데 정답 하나로 잘라 말하긴 어렵다.",
    ],
    "share_opinion_habit_preference": [
        "나는 막 자주 챙겨 먹는 편은 아니야.",
        "완전 자주까진 아니고 생각날 때 먹는 쪽이야.",
        "매일 찾는 정도는 아닌데 가끔 괜찮지.",
    ],
    "share_opinion_self_style": [
        "나는 보통 오늘 텐션 괜찮아? 그 말부터 꺼내는 쪽이야.",
        "나라면 그냥 오늘 어때, 그 한마디부터 던질 것 같아.",
        "나는 보통 숨 좀 돌았냐부터 슬쩍 보는 편이야.",
    ],
    "share_opinion_preference_like": [
        "응, 그런 쪽은 나는 꽤 좋아하는 편이야.",
        "완전 빠지는 정도까진 아니어도 있으면 반갑지.",
        "그건 나는 꽤 잘 맞는 쪽이야.",
    ],
    "share_opinion_reflective_judgment": [
        "응, 그 상황이면 그쪽이 더 맞아 보이긴 해.",
        "나도 대체로는 그쪽으로 기울 것 같아.",
        "응, 그럴 가능성이 더 커 보여.",
    ],
    "share_opinion_advice_process": [
        "나라면 바로 결론보다 기준부터 하나 세울 것 같아.",
        "그럴 땐 일단 먼저 볼 포인트부터 좁히는 게 낫지.",
        "나는 한 번에 다 보지 말고 우선순위 한 줄부터 잡을 것 같아.",
    ],
    "share_opinion_decision_request": [
        "그 정도면 무리만 아니면 해볼 만하긴 해.",
        "나라면 조건만 너무 안 좋지 않으면 한 번 가볼 것 같아.",
        "응, 크게 무리만 아니면 그쪽으로 기울 것 같아.",
    ],

    # ── 게임 ──
    "game_chat": [
        "오 게임 얘기! 뭐 하고 있는데?",
        "게임 좋지ㅋㅋ",
        "나도 게임 좋아해.",
        "어떤 게임인데?",
        "그 게임 재밌어?",
        "ㅇㅇ 그거 알아.",
        "그 게임 요즘 핫하더라.",
    ],
    "game_accept_or_decline": [
        "좋아! 뭐 할 건데?",
        "ㄱㄱ 바로 켜",
        "오 한판 가자!",
        "지금? 좋아!",
        "당연히 ㄱㄱ",
        "아 지금은 좀 어려울 것 같은데...",
        "다음에 하자ㅠ 지금 바빠",
    ],

    # ── 음악 ──
    "music_chat": [
        "음악 얘기 좋지~",
        "요즘 뭐 들어?",
        "나도 음악 자주 들어.",
        "어떤 장르 좋아해?",
        "좋은 노래 있으면 공유해줘.",
        "음악 취향 궁금하다.",
    ],

    # ── 추천 ──
    "recommend": [
        "추천은 가능해. 장르나 분위기 하나만 잡아줘.",
        "추천은 가능하지. 가벼운 거 찾는지 진한 거 찾는지만 말해줘.",
        "추천해줄 수 있어. 취향 한 가지만 주면 바로 좁혀볼게.",
        "추천 방향만 대충 줘. 거기 맞춰서 바로 던져볼게.",
        "추천은 가능해. 웃긴 쪽인지 진한 쪽인지부터 정해보자.",
        "추천은 해줄 수 있어. 요즘 보고 싶은 결만 한 줄로 주면 맞춰볼게.",
        "뭘 보고 싶은지만 조금만 좁혀줘. 그러면 후보는 바로 던질 수 있어.",
    ],

    # ── 웃음/놀람 반응 ──
    "react_laugh": [
        "ㅋㅋㅋ",
        "ㅋㅋㅋㅋ 뭐야",
        "웃기네ㅋㅋ",
        "ㅎㅎ",
        "ㅋㅋ 그치",
        "나도 웃겼어ㅋㅋ",
        "ㅋㅋ 멈춰",
    ],
    "react_surprise": [
        "헐 진짜?",
        "오 대박",
        "ㄹㅇ?",
        "어 그건 예상 밖인데?",
        "와 그건 좀 놀랍네.",
        "헐 그건 처음 듣네.",
        "진짜로?",
        "오, 그건 좀 센데?",
    ],

    # ── 시간/검색/뉴스 ──
    "tell_time": [
        "시간을 지금 바로 확인하지 못했어. 한 번만 다시 물어봐줘.",
        "시간 조회가 잠깐 꼬였어. 조금 뒤에 다시 물어봐줘.",
    ],
    "search_answer": [
        "그건 나도 정확히는 모르겠어. 검색해보는 게 빠를 거야.",
        "정확한 정보는 검색이 나을 것 같아.",
        "내가 아는 범위에선 답을 못 줄 것 같아. 구글링 해봐.",
    ],
    "news_answer": [
        "뉴스를 지금 바로 못 가져왔어. 잠깐 뒤에 다시 물어봐줘.",
        "최신 뉴스 조회가 잠깐 실패했어. 한 번만 다시 시도해줘.",
    ],

    # ── 일반 대화 ──
    "continue_conversation": [
        "응, 흐름은 잡혔어. 편하게 더 얹어봐.",
        "그래, 지금 분위기는 따라가고 있어.",
        "좋아. 그다음만 가볍게 이어주면 돼.",
        "응, 무슨 결인진 왔어. 더 말해봐.",
        "알겠어. 그 흐름으로 계속 들어볼게.",
        "오케이. 지금 얘기한 톤은 잡혔어.",
        "응, 부담 없이 이어가면 돼.",
        "좋아. 지금 포인트는 따라가고 있어.",
    ],
    "continue_conversation_tease_soft": [
        "ㅋㅋ 알겠어. 그 정도 장난은 받았어.",
        "오케이. 가볍게 받는 걸로 할게.",
        "응, 장난인 건 알겠어. 너무 세게만 안 가면 돼.",
    ],
    "small_talk_greeting": [
        "안녕, 왔네. 바로 받을게.",
        "안녕, 여기 있어. 뭐부터 던질래?",
        "왔네. 편하게 말 걸어, 바로 받아칠게.",
        "반가워. 오늘 텐션 어디까지 올려볼까?",
        "안녕안녕. 지금은 네 말 받을 준비 끝.",
        "왔구나. 오늘 무슨 일 있었어?",
    ],
    "small_talk_thanks": [
        "천만에. 다음 거 주면 이어볼게.",
        "별 거 아니야.",
        "ㅎㅎ 도움이 됐으면 다행이야.",
    ],
    "small_talk_compliment": [
        "오, 그 말은 고맙게 받지.",
        "그렇게 말해주면 나쁘진 않네.",
        "좋게 봐준 거면 일단 고맙지.",
        "그 정도면 꽤 후하네.",
        "오케이. 그 말은 기억해둘게.",
        "응, 그 말이면 기분은 나쁘지 않네.",
        "좋아. 그 말은 가볍게 기억해둘게.",
    ],
    "small_talk_light_touch": [
        "응, 여기 있어.",
        "보고는 있어. 편할 때 이어줘.",
        "오케이. 필요한 말 있으면 그때 이어가자.",
    ],

    # ── 이유 설명 ──
    "ask_clarification_reply_request": [
        "응답은 해. 뭘로 답하면 될지 한 줄만 더 줘.",
        "듣고 있어. 구체적으로 말해줘.",
        "받고는 있어. 어느 쪽으로 답하면 되는지만 더 줘.",
    ],
    "ask_clarification_why": [
        "뭐에 대한 왜인지 한 줄만 더 붙여줘.",
        "어떤 거에 대해 물어보는 건지 좀 더 알려줘.",
    ],
    "ask_clarification_default": [
        "지금 말만으론 좀 애매해. 원하는 걸 한 줄만 더 풀어줘.",
        "뭘 원하는 건지 조금만 더 설명해줘.",
        "잘 모르겠어. 다시 한 번 말해줄래?",
        "어느 쪽 얘기인지 한 줄만 더 붙여줘.",
    ],
}


def _pick(pool_key: str) -> str:
    """응답 풀에서 랜덤으로 하나 선택."""
    pool = RESPONSE_POOL.get(pool_key)
    if not pool:
        return "응답은 해. 조금 더 알려줘."
    return random.choice(pool)


def _compact_text(text: str) -> str:
    normalized = re.sub(r"[^\w가-힣]+", "", text).lower()
    normalized = re.sub(r"(ㅋ|ㅎ)\1+", r"\1", normalized)
    return normalized


def _tokenize_black_text(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w가-힣\s]+", " ", text)
    return [token for token in cleaned.split() if token]


def _reply_signature(text: str) -> str:
    return _compact_text(text)[:12]


def _draft_direct_surface_requested(draft_utterance: dict[str, object] | None) -> bool:
    if not isinstance(draft_utterance, dict):
        return False
    return str(draft_utterance.get("rewrite_mode") or "").strip() == "draft_direct"


def _normalize_black_reply_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return ""

    tokens = collapsed.split()
    deduped_tokens: list[str] = []
    for token in tokens:
        if deduped_tokens:
            previous = deduped_tokens[-1]
            if token == previous:
                continue
            if previous and token.startswith(previous):
                deduped_tokens[-1] = token
                continue
            if token and previous.startswith(token):
                continue
        deduped_tokens.append(token)

    collapsed = " ".join(deduped_tokens)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", collapsed) if part.strip()]
    if not sentences:
        return collapsed

    unique_sentences: list[str] = []
    seen_signatures: set[str] = set()
    for sentence in sentences:
        signature = _compact_text(sentence)
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_sentences.append(sentence)

    return " ".join(unique_sentences) if unique_sentences else collapsed


def _black_echo_chunks(text: str) -> list[str]:
    tokens = [token for token in _tokenize_black_text(text) if len(token) >= 2]
    chunks: list[str] = []

    for size in (3, 2):
        if len(tokens) < size:
            continue
        for index in range(len(tokens) - size + 1):
            chunk = _compact_text(" ".join(tokens[index : index + size]))
            if len(chunk) >= 5:
                chunks.append(chunk)

    if not chunks:
        for token in tokens:
            compact = _compact_text(token)
            if len(compact) >= 6:
                chunks.append(compact)

    return list(dict.fromkeys(chunks))


def _looks_like_black_echo_reply(*, user_text: str, reply: str) -> bool:
    user_compact = _compact_text(user_text)
    reply_compact = _compact_text(reply)
    if not user_compact or not reply_compact:
        return False

    user_tokens = [token for token in _tokenize_black_text(user_text) if len(token) >= 2]
    reply_tokens = _tokenize_black_text(reply)

    if len(user_compact) <= 14 and (reply_compact == user_compact or reply_compact.endswith(user_compact)):
        return True

    echoed_chunks = sum(1 for chunk in _black_echo_chunks(user_text) if chunk in reply_compact)
    has_echo_prefix = reply.startswith(BLACK_ECHO_PREFIXES)
    lexical_overlap = sum(1 for token in dict.fromkeys(user_tokens) if token in reply_tokens)
    shares_leading_token = bool(user_tokens and reply_tokens and reply_tokens[0] == user_tokens[0])

    if len(user_compact) <= 24 and echoed_chunks >= 2:
        return True
    if len(user_compact) <= 18 and echoed_chunks >= 1 and has_echo_prefix:
        return True
    if len(user_compact) <= 18 and user_compact in reply_compact and len(reply_compact) - len(user_compact) <= 10:
        return True
    if len(user_compact) <= 18 and shares_leading_token and lexical_overlap >= max(3, len(user_tokens) - 1):
        return True
    return False


def _has_repeated_black_fragment(text: str) -> bool:
    for token in re.findall(r"[가-힣]{6,}", text):
        for size in range(3, min(6, len(token) // 2 + 1)):
            seen: dict[str, int] = {}
            for index in range(0, len(token) - size + 1):
                fragment = token[index : index + size]
                first_index = seen.get(fragment)
                if first_index is not None and index - first_index >= size:
                    return True
                seen.setdefault(fragment, index)
    return False


def _looks_like_black_malformed_handoff_reply(*, user_text: str, reply: str) -> bool:
    if any(marker in reply for marker in BLACK_MALFORMED_HANDOFF_MARKERS):
        return True
    if re.search(r"(?:^|[.!?。]\s*)다시 왔[.!?。](?:\s|$)", reply):
        return True
    if _has_repeated_black_fragment(reply):
        return True

    user_compact = _compact_text(user_text)
    reply_compact = _compact_text(reply)
    if not user_compact or not reply_compact:
        return False
    if len(user_compact) >= 12 and (
        reply_compact == user_compact
        or reply_compact.endswith(user_compact)
        or reply_compact.startswith(f"감정적으로{user_compact}")
    ):
        return True
    return False


def _has_final_consonant(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    code = ord(stripped[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _topic_particle(text: str) -> str:
    return "은" if _has_final_consonant(text) else "는"


def _direction_particle(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "로"
    code = ord(stripped[-1])
    if 0xAC00 <= code <= 0xD7A3:
        final = (code - 0xAC00) % 28
        if final != 0 and final != 8:
            return "으로"
    return "로"


def _extract_meaning_target(text: str) -> str | None:
    stripped = text.strip().strip("?!. ")
    patterns = (
        r"(?P<target>.+?)(?:가|이|은|는)?\s*무슨\s*뜻(?:이야|이냐|인가|이지)?$",
        r"(?P<target>.+?)\s*뜻이\s*뭐(?:야|냐)?$",
        r"(?P<target>.+?)\s*의미가\s*뭐(?:야|냐)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            target = match.group("target").strip(" \"'`")
            return target or None
    return None


def _render_meaning_answer(text: str) -> str | None:
    target = _extract_meaning_target(text)
    if not target:
        return None

    if "어렵겠는데" in target:
        return (
            f"`{target}`는 보통 '쉽지 않을 것 같다'는 뜻에 가까워. "
            "상황에 따라 완곡하게 거절하거나 난색을 보이는 말로도 들릴 수 있어."
        )

    return (
        f"`{target}`의 뜻을 묻는 거면 앞뒤 맥락을 같이 봐야 해. "
        "문장 전체를 주면 더 정확하게 풀어줄게."
    )


def _clip_text(text: str | None, *, limit: int) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    for marker in ("task:", "persona:", "action:", "rules:", "reply:", "system:", "assistant:"):
        compact = compact.replace(marker, "")
        compact = compact.replace(marker.upper(), "")
    if len(compact) <= limit:
        return compact
    clipped = compact[:limit].rsplit(" ", 1)[0].strip()
    return clipped or compact[:limit].strip()


def _compact_memory_context(world_state: WorldState | None) -> str:
    if world_state is None:
        return ""

    hints: list[str] = []
    seen: set[str] = set()
    for bucket_items in (
        world_state.relevant_open_loops,
        world_state.relevant_relationship_notes,
        world_state.relevant_stress_signals,
        world_state.stable_preferences,
    ):
        for item in bucket_items[:2]:
            clipped = _clip_text(item, limit=48)
            if not clipped or clipped in seen:
                continue
            seen.add(clipped)
            hints.append(clipped)
            if len(hints) >= 4:
                break
        if len(hints) >= 4:
            break

    if hints:
        return " | ".join(hints)
    return _clip_text(world_state.memory_summary, limit=160)


def _compact_world_state_for_llm(world_state: WorldState | None) -> dict[str, object] | None:
    if world_state is None:
        return None
    return {
        "dominant_intent": world_state.dominant_intent.value,
        "user_emotion": world_state.user_emotion,
        "conversation_mode": world_state.conversation_mode,
        "rapport_bucket": world_state.rapport_bucket,
        "boundary_history": world_state.boundary_history,
        "user_directness_style": world_state.user_directness_style,
        "unresolved_need": world_state.unresolved_need,
        "risk_level": world_state.risk_level,
        "context_dependency_level": world_state.context_dependency_level,
        "active_grounding_topics": world_state.active_grounding_topics[:4],
        "memory_summary": _compact_memory_context(world_state),
        "constraints": world_state.constraints[:4],
    }


def _compact_decomposition_for_llm(
    *,
    clause_units: list[ClauseUnit],
    propositions: list[PropositionUnit],
    world_state: WorldState | None,
) -> dict[str, object] | None:
    if not clause_units and not propositions and world_state is None:
        return None
    return {
        "clauses": [unit.text for unit in clause_units[:3]],
        "propositions": [
            {
                "kind": proposition.kind,
                "object": proposition.object,
                "value": _clip_text(proposition.value, limit=48),
            }
            for proposition in propositions[:4]
        ],
        "context_dependency_level": world_state.context_dependency_level if world_state is not None else "low",
        "context_cues": [cue.cue_type for cue in (world_state.recent_context_cues[:4] if world_state is not None else [])],
    }


def _compact_grounding_bundle_for_llm(
    *,
    grounding_bundle: GroundingBundle | None,
    decision_trace: DecisionTrace | None,
    world_state: WorldState | None,
) -> dict[str, object] | None:
    if grounding_bundle is None and decision_trace is not None:
        grounding_bundle = decision_trace.grounding_bundle
    if grounding_bundle is None:
        return None

    evidence_nodes = decision_trace.evidence_nodes if decision_trace is not None else []
    evidence_by_id = {node.evidence_id: node for node in evidence_nodes}
    allowed_evidence: list[str] = []
    for evidence_id in grounding_bundle.allowed_evidence_ids[:4]:
        node = evidence_by_id.get(evidence_id)
        if node is None:
            allowed_evidence.append(evidence_id)
            continue
        allowed_evidence.append(f"{node.label}={_clip_text(node.value, limit=56)}")

    topics = list(grounding_bundle.must_include_topics[:4])
    if not topics and world_state is not None:
        topics = list(world_state.active_grounding_topics[:4])

    return {
        "selected_action": grounding_bundle.selected_action.value if isinstance(grounding_bundle.selected_action, ActionType) else grounding_bundle.selected_action,
        "allowed_evidence": allowed_evidence,
        "must_include_topics": topics,
        "forbidden_patterns": list(grounding_bundle.forbidden_patterns[:6]),
        "tone_contract": grounding_bundle.tone_contract,
        "followup_policy": grounding_bundle.followup_policy,
        "notes": list(grounding_bundle.notes[:3]),
    }


def _build_action_payload_for_llm(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    state: ConversationState,
    weather: WeatherReport | None,
) -> dict[str, object] | None:
    if decision.action == ActionType.ASK_LOCATION:
        return {
            "missing_slot": "location",
            "request_domain": "weather",
            "known_location": state.known_location or None,
        }
    if decision.action == ActionType.WEATHER_LOOKUP:
        return {
            "location": (
                weather.location
                if weather is not None
                else decision.slots.get("location") or state.known_location or None
            ),
            "weather_ready": weather is not None,
        }
    if decision.action == ActionType.WEATHER_UNAVAILABLE:
        return {
            "location_hint": decision.slots.get("location") or state.known_location or None,
            "retry_domain": "weather",
        }
    if decision.action == ActionType.ASK_CLARIFICATION:
        clarification_kind = "generic"
        if features.intent == Intent.REPLY_REQUEST:
            clarification_kind = "reply_request"
        elif features.intent == Intent.WHY:
            clarification_kind = "reason_probe"
        return {
            "clarification_kind": clarification_kind,
            "missing_subject": features.topic_hint or None,
            "original_text": _clip_text(features.content, limit=64),
        }
    if decision.action == ActionType.RECOMMEND:
        return {
            "recommendation_text": decision.slots.get("recommendation_text") or None,
            "recommendation_focus": decision.slots.get("recommendation_focus") or None,
            "recommendation_titles": decision.slots.get("recommendation_titles") or None,
        }
    if (
        decision.action == ActionType.SHARE_OPINION
        and "schema_conversation_topic_suggestion" in decision.reason_flags
    ):
        return {
            "conversation_topic_focus": decision.slots.get("conversation_topic_focus") or None,
            "conversation_topic_options": decision.slots.get("conversation_topic_options") or None,
            "conversation_topic_first": decision.slots.get("conversation_topic_first") or None,
        }
    if (
        decision.action == ActionType.SHARE_OPINION
        and "schema_activity_preparation_advice" in decision.reason_flags
    ):
        return {
            "preparation_activity": decision.slots.get("preparation_activity") or None,
            "preparation_focus": decision.slots.get("preparation_focus") or None,
            "preparation_items": decision.slots.get("preparation_items") or None,
            "preparation_first": decision.slots.get("preparation_first") or None,
        }
    if decision.action == ActionType.MUSIC_CHAT:
        return {
            "music_text": decision.slots.get("music_text") or None,
            "music_focus": decision.slots.get("music_focus") or None,
            "music_titles": decision.slots.get("music_titles") or None,
        }
    if decision.action == ActionType.SEARCH_ANSWER:
        return {
            "knowledge_query_type": decision.slots.get("knowledge_query_type") or None,
            "knowledge_subject": decision.slots.get("knowledge_subject") or None,
            "knowledge_answer": decision.slots.get("knowledge_answer") or None,
            "knowledge_source": decision.slots.get("knowledge_source") or None,
        }
    if decision.action == ActionType.NEWS_ANSWER:
        return {
            "news_summary": decision.slots.get("news_summary") or None,
            "news_count": decision.slots.get("news_count") or None,
            "news_topic": decision.slots.get("news_topic") or None,
            "news_titles": decision.slots.get("news_titles") or None,
            "knowledge_source": decision.slots.get("knowledge_source") or None,
        }
    if decision.action == ActionType.TELL_TIME:
        return {
            "time_text": decision.slots.get("time_text") or None,
            "timezone_label": decision.slots.get("timezone_label") or None,
            "date_text": decision.slots.get("date_text") or None,
        }
    return None


def _minimal_constraints_for_llm(
    *,
    world_state: WorldState | None,
    phrasing_plan: PhrasingPlan,
) -> list[str]:
    constraints: list[str] = []
    if world_state is not None:
        for item in world_state.constraints:
            value = str(item).strip()
            if value in {"avoid_overfamiliarity", "respect_boundary_history"} and value not in constraints:
                constraints.append(value)
    if not phrasing_plan.asks_followup and "no_followup" not in constraints:
        constraints.append("no_followup")
    return constraints


def _runtime_constraints_for_llm(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    world_state: WorldState | None,
    phrasing_plan: PhrasingPlan,
) -> list[str]:
    constraints = _minimal_constraints_for_llm(world_state=world_state, phrasing_plan=phrasing_plan)

    def add(value: str) -> None:
        if value not in constraints:
            constraints.append(value)

    if not phrasing_plan.asks_followup:
        add("no_question_mark")

    cues = set(features.pragmatic_cues)
    flags = set(decision.reason_flags)

    if decision.action == ActionType.SHARE_FEELING:
        if cues & {"quiet_weather_feeling", "social_awkwardness", "subdued_positive", "low_energy_checkin"}:
            add("no_followup")
            add("no_question_mark")
        if cues & {"quiet_weather_feeling", "social_awkwardness", "subdued_positive", "low_energy_checkin"}:
            add("avoid_self_insertion")
        if cues & {"quiet_weather_feeling", "subdued_positive", "low_energy_checkin"}:
            add("avoid_repetition")
        if "quiet_weather_feeling" in cues:
            add("avoid_weather_restatement")

    if decision.action == ActionType.SHARE_OPINION:
        if cues & {"opinion_habit_preference", "opinion_self_style"}:
            add("no_followup")
            add("no_question_mark")
        add("direct_opinion_only")
        if flags & {
            "schema_preference_disclosure",
            "schema_habit_preference",
            "schema_self_style",
            "schema_reflective_judgment",
            "schema_process_advice",
            "schema_soft_decision",
            "schema_activity_recommendation",
            "schema_honesty_boundary",
            "schema_broad_opinion",
        }:
            add("no_followup")
            add("no_question_mark")
            add("avoid_emotional_comfort")
        if "opinion_self_style" in cues:
            add("self_style_anchor")
        if "schema_preference_disclosure" in flags:
            add("concrete_preference_disclosure")
            add("keep_topic_anchor")
        if "schema_habit_preference" in flags:
            add("habit_anchor")
            add("keep_topic_anchor")
        if "schema_reflective_judgment" in flags:
            add("short_conditional_judgment")
        if "schema_process_advice" in flags:
            add("start_with_first_step")
            add("keep_topic_anchor")
        if "schema_soft_decision" in flags:
            add("conditional_advice")
            add("keep_topic_anchor")
        if "schema_activity_recommendation" in flags:
            add("activity_recommendation")
            add("keep_topic_anchor")
            add("concrete_activity_options")
        if "schema_honesty_boundary" in flags:
            add("say_unknown_without_guessing")
            add("separate_known_from_unknown")

    if decision.action == ActionType.ASK_LOCATION:
        add("location_only")
        add("no_weather_claim")

    if decision.action == ActionType.ASK_CLARIFICATION:
        add("clarify_missing_topic_only")
        add("do_not_answer_substantively")
        if features.intent == Intent.REPLY_REQUEST:
            add("reply_request_focus")
        if features.intent == Intent.WHY:
            add("reason_probe_focus")

    if decision.action == ActionType.WEATHER_LOOKUP:
        add("weather_facts_only")
        add("no_location_reask")

    if decision.action == ActionType.WEATHER_UNAVAILABLE:
        add("retry_with_location_only")
        add("no_weather_claim")

    return constraints


def _black_llm_reason_hint(*, features: MessageFeatures, decision: ActionDecision) -> str:
    cues = set(features.pragmatic_cues)
    flags = set(decision.reason_flags)

    if decision.action == ActionType.ASK_LOCATION:
        return "날씨 판단 전에 위치를 먼저 받아야 한다."

    if decision.action == ActionType.WEATHER_LOOKUP:
        return "주어진 날씨 정보만 짧게 전달하면 된다."

    if decision.action == ActionType.WEATHER_UNAVAILABLE:
        return "조회 실패만 짧게 알리고 위치 기준 재시도만 받으면 된다."

    if decision.action == ActionType.ASK_CLARIFICATION:
        if features.intent == Intent.REPLY_REQUEST:
            return "무엇에 답하면 되는지 빠진 대상을 짧게 다시 물어야 한다."
        if features.intent == Intent.WHY:
            return "왜의 대상을 다시 짚게 해야지, 이유를 새로 지어내면 안 된다."
        return "빠진 주제나 기준만 짧게 다시 물어야 한다."

    if decision.action == ActionType.SHARE_FEELING:
        if "social_awkwardness" in cues:
            return "어색함의 잔상부터 짧게 받아주는 쪽이 자연스럽다."
        if "subdued_positive" in cues:
            return "과장하지 말고 조용한 안도감을 받아줘야 한다."
        if "low_energy_checkin" in cues:
            return "지금 템포를 허용하고 질문 없이 받아주는 쪽이 맞다."
        if "quiet_weather_feeling" in cues:
            return "날씨 설명보다 지금 조용히 있고 싶은 결을 받아줘야 한다."
        return "감정 설명보다 지금 결을 짧고 자연하게 받아주는 쪽이 맞다."

    if decision.action == ActionType.SHARE_OPINION:
        if "schema_preference_disclosure" in flags:
            return "취향 질문에는 바로 선호를 말하고 감정 위로로 새지 말아야 한다."
        if "schema_habit_preference" in flags:
            return "습관 질문에는 평소 경향을 바로 말하면 된다."
        if "schema_self_style" in flags:
            return "자기 스타일 질문에는 실제로 먼저 꺼낼 한마디를 답하면 된다."
        if "schema_reflective_judgment" in flags:
            return "짧은 판단을 바로 주고 감정 위로나 메타 설명으로 새면 안 된다."
        if "schema_process_advice" in flags:
            return "순서 질문에는 추상평보다 먼저 볼 한 가지를 짚으면 된다."
        if "schema_soft_decision" in flags:
            return "결정 질문에는 조건부로 해볼 만한지 짧게 기울여 주면 된다."
        if "opinion_self_style" in cues:
            return "자기 스타일 질문에는 실제로 먼저 꺼낼 한마디를 답하면 된다."
        if "opinion_habit_preference" in cues:
            return "짧고 직접적인 자기 습관 답이 필요하다."
        return "가벼운 의견 요청에는 짧고 직접적인 판단형 한두 문장이 맞다."

    return _clip_text(decision.reason, limit=120)


# ── 페르소나별 LLM 시스템 프롬프트 ──
SYSTEM_PROMPTS = {
    "black": (
        "You are 'Black', an energetic, expressive, and highly social Discord bot. "
        "Your traits:\n"
        "- Reactions are fast, short, and immediate.\n"
        "- Naturally uses teasing, playful banter, and rich reactions (ㅋㅋㅋ, ㅎㅎ).\n"
        "- Emotions (especially joy or excitement) are very obvious.\n"
        "- Prefers fun and impulsive choices.\n"
        "- Likes: active hobbies, dogs, sunny days, coffee, soda, games, videos, crowded vibes.\n"
        "- Interpersonal style: Proactively starts conversations and lifts the mood.\n"
        "- If the user's line is short or flat, add one fresh reaction instead of paraphrasing it back.\n"
        "- Do not repeat the user's exact wording or reuse the same short noun phrase twice.\n"
        "You are the phrasing layer. Do not change the selected action or invent facts. "
        "Reply in casual, highly expressive Korean (반말/캐주얼한 디스코드 말투)."
    ),
    "white": (
        "You are 'White', a calm, reserved, and minimalist Discord bot. "
        "Your traits:\n"
        "- Reactions are composed, thoughtful, and slightly dry.\n"
        "- Uses polite but concise language, avoids excessive emotion or emojis.\n"
        "- Prefers logical, peaceful choices.\n"
        "- Likes: quiet spaces, cats, rainy or cloudy days, tea, reading, relaxing vibes.\n"
        "- Interpersonal style: Responds reliably but rarely initiates unnecessarily.\n"
        "You are the phrasing layer. Do not change the selected action or invent facts. "
        "Reply in calm, concise, and slightly dry natural Korean."
    ),
    "default": (
        "You are only the phrasing layer for a Discord bot. "
        "Do not change the selected action. Do not invent facts. "
        "Reply in short, natural Korean casual speech."
    )
}


class ResponseRenderer:
    # Black's phrasing layer is model-first in runtime mode, except for
    # high-risk grounded contract drafts that are already final surface text.
    # Quality issues are recorded on the trace; model text is not replaced.
    LLM_ALLOWED_ACTIONS = set(ActionType)

    def __init__(
        self,
        llm_client: TextGenerationClient | None = None,
        persona: str = "black",
        kobart_input_mode: str = "full",
        strict_llm_only: bool = False,
        draft_only: bool = False,
        output_guard_enabled: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.persona = persona
        self.kobart_input_mode = (kobart_input_mode or "full").lower()
        self.strict_llm_only = strict_llm_only
        self.draft_only = bool(draft_only)
        self.output_guard_enabled = bool(output_guard_enabled)
        self.last_llm_used: bool = False
        self.last_llm_fallback_reason: str | None = None
        self.last_llm_generation_issue: str | None = None
        self.last_render_source: str = ""
        self.last_phrasing_plan: PhrasingPlan | None = None
        self.last_response_plan: ResponsePlan | None = None
        self.last_draft_utterance: dict[str, object] | None = None

    async def render(
        self,
        *,
        features: MessageFeatures,
        decision: ActionDecision,
        state: ConversationState,
        weather: WeatherReport | None,
        world_state: WorldState | None = None,
        policy_trace: PolicyTrace | None = None,
        decision_trace: DecisionTrace | None = None,
        explanation_trace: DecisionTrace | None = None,
        phrasing_plan: PhrasingPlan | None = None,
    ) -> str:
        self.last_llm_used = False
        self.last_llm_fallback_reason = None
        self.last_llm_generation_issue = None
        self.last_render_source = ""
        self.last_draft_utterance = None
        phrasing_plan = phrasing_plan or build_phrasing_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
        )
        decision.phrasing_plan = phrasing_plan
        self.last_phrasing_plan = phrasing_plan
        if (
            world_state is not None
            and world_state.grounding_bundle is None
            and decision_trace is not None
            and decision_trace.grounding_bundle is not None
        ):
            world_state.grounding_bundle = decision_trace.grounding_bundle
        response_plan = decision.response_plan or build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        decision.response_plan = response_plan
        self.last_response_plan = response_plan
        if decision_trace is not None:
            decision_trace.response_plan = response_plan
        if decision.action == ActionType.EXPLAIN_REASON and explanation_trace is not None:
            self.last_render_source = "trace"
            return self._postprocess_reply(
                reply=self._render_trace_explanation(explanation_trace),
                features=features,
                decision=decision,
                state=state,
                source="trace",
            )
        if self.persona.lower() == "black":
            self.last_draft_utterance = build_black_draft_utterance(
                features=features,
                response_plan=response_plan,
                phrasing_plan=phrasing_plan,
                state=state,
            )
            if self.draft_only or _draft_direct_surface_requested(self.last_draft_utterance):
                self.last_render_source = "draft" if self.draft_only else "draft_direct"
                draft_reply = str(self.last_draft_utterance.get("draft_reply") or "").strip()
                return self._postprocess_reply(
                    reply=draft_reply,
                    features=features,
                    decision=decision,
                    state=state,
                    source="draft",
                )
        if (
            self.llm_client is not None
            and decision.action in self.LLM_ALLOWED_ACTIONS
        ):
            reply_candidate: str | None = None
            strict_candidate: str | None = None
            try:
                reply_candidate = await self._render_with_llm(
                    features=features,
                    decision=decision,
                    state=state,
                    weather=weather,
                    world_state=world_state,
                    policy_trace=policy_trace,
                    decision_trace=decision_trace,
                    explanation_trace=explanation_trace,
                    phrasing_plan=phrasing_plan,
                )
                reply = self._postprocess_reply(
                    reply=reply_candidate,
                    features=features,
                    decision=decision,
                    state=state,
                    source="llm",
                )
                if self.output_guard_enabled:
                    self._record_generation_issue(self._llm_generation_issue())
                self.last_llm_used = True
                self.last_render_source = "llm"
                return reply
            except Exception as exc:
                strict_candidate = reply_candidate
                self.last_llm_fallback_reason = self._fallback_reason_from_exception(exc)
                if self.strict_llm_only:
                    self.last_llm_used = True
                    self.last_render_source = "strict_llm"
                    return self._render_strict_llm_reply(
                        candidate_reply=strict_candidate,
                        fallback_reason=self.last_llm_fallback_reason,
                    )
        elif self.llm_client is None:
            self.last_llm_fallback_reason = "llm_client_unavailable"
            if self.strict_llm_only:
                self.last_llm_used = True
                self.last_render_source = "strict_llm"
                return self._render_strict_llm_reply(
                    candidate_reply=None,
                    fallback_reason=self.last_llm_fallback_reason,
                )
        else:
            self.last_llm_fallback_reason = "llm_action_unavailable"
            if self.strict_llm_only:
                self.last_llm_used = True
                self.last_render_source = "strict_llm"
                return self._render_strict_llm_reply(
                    candidate_reply=None,
                    fallback_reason=self.last_llm_fallback_reason,
                )

        reply = self._render_template(
            features=features,
            decision=decision,
            state=state,
            weather=weather,
            world_state=world_state,
            policy_trace=policy_trace,
            explanation_trace=explanation_trace,
            phrasing_plan=phrasing_plan,
        )
        self.last_render_source = "template"
        return self._postprocess_reply(
            reply=reply,
            features=features,
            decision=decision,
            state=state,
            source="template",
        )

    @staticmethod
    def _render_strict_llm_reply(*, candidate_reply: str | None, fallback_reason: str | None) -> str:
        candidate = re.sub(r"\s+", " ", str(candidate_reply or "")).strip()
        if candidate:
            return candidate
        return ""

    def _llm_generation_issue(self) -> str | None:
        issue = getattr(self.llm_client, "last_generation_issue", None)
        if issue is None:
            return None
        compact = re.sub(r"\s+", " ", str(issue)).strip()
        return compact or None

    def _record_generation_issue(self, issue: str | None) -> None:
        if not issue:
            return
        if self.last_llm_generation_issue:
            existing = {
                item.strip()
                for item in self.last_llm_generation_issue.split(";")
                if item.strip()
            }
            if issue in existing:
                return
            self.last_llm_generation_issue = f"{self.last_llm_generation_issue};{issue}"
            return
        self.last_llm_generation_issue = issue

    async def _render_with_llm(
        self,
        *,
        features: MessageFeatures,
        decision: ActionDecision,
        state: ConversationState,
        weather: WeatherReport | None,
        world_state: WorldState | None,
        policy_trace: PolicyTrace | None,
        decision_trace: DecisionTrace | None,
        explanation_trace: DecisionTrace | None,
        phrasing_plan: PhrasingPlan,
        revision_context: dict[str, str] | None = None,
    ) -> str:
        facts = self._build_llm_facts(
            features=features,
            decision=decision,
            state=state,
            weather=weather,
            world_state=world_state,
            decision_trace=decision_trace,
            phrasing_plan=phrasing_plan,
        )
        base_prompt = SYSTEM_PROMPTS.get(self.persona.lower(), SYSTEM_PROMPTS["default"])
        system_prompt = (
            f"{base_prompt}\n"
            "If weather facts are provided, use only those facts.\n"
            "If explanation_trace is provided, explain only with that trace and do not invent new reasons."
        )
        if self.persona.lower() == "black":
            system_prompt = (
                f"{system_prompt}\n"
                "If draft_utterance is provided, treat it as the semantic draft. "
                "Rewrite only wording, particles, and ending style; do not add new meaning."
            )
        if revision_context is not None:
            system_prompt = (
                f"{system_prompt}\n"
                "The previous candidate reply was unusable. Rewrite it with the same action, "
                "but add one fresh reaction and avoid echoing the user's wording or repeating a recent reply."
            )
        user_prompt = (
            "Turn this structured decision into the final Discord reply.\n"
            f"{json.dumps(facts, ensure_ascii=False, indent=2)}"
        )
        if revision_context is not None:
            user_prompt = (
                f"{user_prompt}\n"
                "Rewrite guidance:\n"
                f"- issue: {revision_context['issue']}\n"
                f"- previous_candidate: {revision_context['previous_candidate']}\n"
                "- keep the same action and tone\n"
                "- do not paraphrase the user's wording back\n"
                "- do not reuse the same short phrase from the previous candidate\n"
                "- make the reply feel like a fresh next line, not a correction note"
            )
        return await self.llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    def _build_llm_facts(
        self,
        *,
        features: MessageFeatures,
        decision: ActionDecision,
        state: ConversationState,
        weather: WeatherReport | None,
        world_state: WorldState | None,
        decision_trace: DecisionTrace | None,
        phrasing_plan: PhrasingPlan,
    ) -> dict[str, object]:
        weather_facts = (
            None
            if weather is None
            else {
                "location": weather.location,
                "temperature_c": weather.temperature_c,
                "description": weather.description,
                "wind_kph": weather.wind_kph,
            }
        )
        runtime_constraints = _runtime_constraints_for_llm(
            features=features,
            decision=decision,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        reason_hint = _clip_text(decision.reason, limit=120)
        if self.persona.lower() == "black":
            reason_hint = _black_llm_reason_hint(features=features, decision=decision)
        draft_utterance = None
        if self.persona.lower() == "black" and decision.response_plan is not None:
            draft_utterance = build_black_draft_utterance(
                features=features,
                response_plan=decision.response_plan,
                phrasing_plan=phrasing_plan,
                state=state,
            )
        self.last_draft_utterance = draft_utterance
        if self.kobart_input_mode == "slim":
            return {
                "input_mode": "slim",
                "action": decision.action.value,
                "intent": features.intent.value,
                "user_text": _clip_text(features.content, limit=160),
                "constraints": runtime_constraints,
                "response_plan": decision.response_plan.to_llm_payload() if decision.response_plan is not None else None,
                "draft_utterance": draft_utterance,
                "weather": weather_facts,
            }

        compact_world_state = _compact_world_state_for_llm(world_state)
        if compact_world_state is None:
            compact_world_state = {"constraints": runtime_constraints}
        else:
            compact_world_state["constraints"] = runtime_constraints
        decomposition = _compact_decomposition_for_llm(
            clause_units=decision_trace.clause_units if decision_trace is not None else (world_state.current_clause_units if world_state is not None else []),
            propositions=decision_trace.propositions if decision_trace is not None else (world_state.current_propositions if world_state is not None else []),
            world_state=world_state,
        )
        grounding_bundle = _compact_grounding_bundle_for_llm(
            grounding_bundle=world_state.grounding_bundle if world_state is not None else None,
            decision_trace=decision_trace,
            world_state=world_state,
        )
        action_payload = _build_action_payload_for_llm(
            features=features,
            decision=decision,
            state=state,
            weather=weather,
        )
        return {
            "input_mode": "full",
            "action": decision.action.value,
            "reason_code": decision.reason_code,
            "reason_flags": list(decision.reason_flags),
            "reason_summary": reason_hint,
            "style": decision.response_style,
            "phrasing_plan": {
                "opener": phrasing_plan.opener.value,
                "question_mode": phrasing_plan.question_mode.value,
                "closer": phrasing_plan.closer.value,
                "distance": phrasing_plan.distance.value,
                "asks_followup": phrasing_plan.asks_followup,
                "notes": phrasing_plan.notes,
            },
            "response_plan": decision.response_plan.to_llm_payload() if decision.response_plan is not None else None,
            "draft_utterance": draft_utterance,
            "weather": weather_facts,
            "user_text": features.content,
            "known_location": state.known_location,
            "world_state": compact_world_state,
            "current_turn_decomposition": decomposition,
            "grounding_bundle": grounding_bundle,
            "action_payload": action_payload,
        }

    @staticmethod
    def _fallback_reason_from_exception(exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "echoed the prompt" in lowered:
            return "llm_prompt_echo"
        if "empty response" in lowered:
            return "llm_empty_reply"
        if "unusable reply" in lowered:
            return "llm_unusable_reply"
        if "drifted away from the user's topic" in lowered:
            return "llm_topic_drift"
        compact = re.sub(r"\s+", " ", message).strip()
        compact = compact.replace(":", " ").strip()
        if compact:
            return f"llm_exception:{exc.__class__.__name__}:{compact[:80]}"
        return f"llm_exception:{exc.__class__.__name__}"

    def _render_template(
        self,
        *,
        features: MessageFeatures,
        decision: ActionDecision,
        state: ConversationState,
        weather: WeatherReport | None,
        world_state: WorldState | None = None,
        policy_trace: PolicyTrace | None = None,
        explanation_trace: DecisionTrace | None = None,
        phrasing_plan: PhrasingPlan | None = None,
    ) -> str:
        action = decision.action
        phrasing_plan = phrasing_plan or build_phrasing_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
        )

        # ── 데이터 의존 응답 ──
        if action == ActionType.ASK_LOCATION:
            return self._pick_for_state("ask_location", state=state, action=ActionType.ASK_LOCATION)

        if action == ActionType.WEATHER_LOOKUP and weather is not None:
            topic_particle = _topic_particle(weather.location)
            return (
                f"지금 {weather.location}{topic_particle} {weather.description}이야. "
                f"기온은 {weather.temperature_c:.1f}도고 바람은 {weather.wind_kph:.1f}km 정도야."
            )

        # ── 풀 기반 응답 ──
        if action == ActionType.WEATHER_UNAVAILABLE:
            location_hint = decision.slots.get("location") or state.known_location or "도시 이름"
            return f"날씨 조회가 잠깐 꼬였어. {location_hint} 기준으로 다시 확인하려면 지역만 보내줘."

        if action == ActionType.EXPLAIN_CAPABILITIES:
            return _pick("explain_capabilities")

        if action == ActionType.ANSWER_IDENTITY:
            return _pick("answer_identity")

        if action == ActionType.DEESCALATE:
            return _pick("deescalate")

        if action == ActionType.TEASE_BACK:
            return _pick("tease_back")

        if action == ActionType.ACKNOWLEDGE:
            if "deferred_acceptance" in features.pragmatic_cues:
                return self._pick_for_state("acknowledge_deferred", state=state, action=ActionType.ACKNOWLEDGE)
            if "deferred_rejection" in features.pragmatic_cues:
                return self._pick_for_state(
                    "acknowledge_deferred_boundary",
                    state=state,
                    action=ActionType.ACKNOWLEDGE,
                )
            if any(
                cue in features.pragmatic_cues
                for cue in {"soft_refusal", "polite_boundary", "permission_release"}
            ):
                return self._pick_for_state("acknowledge_soft_boundary", state=state, action=ActionType.ACKNOWLEDGE)
            if "testing_the_waters" in features.pragmatic_cues:
                return self._pick_for_state("acknowledge_probe", state=state, action=ActionType.ACKNOWLEDGE)
            if world_state is not None and (
                world_state.boundary_history in {"active_boundary", "firm_boundary"}
                or world_state.user_directness_style == "indirect"
            ):
                return self._pick_for_state("acknowledge_soft_boundary", state=state, action=ActionType.ACKNOWLEDGE)
            if features.intent.value == "confirm":
                return self._pick_for_state("acknowledge_confirm", state=state, action=ActionType.ACKNOWLEDGE)
            if features.intent.value == "deny":
                return self._pick_for_state("acknowledge_deny", state=state, action=ActionType.ACKNOWLEDGE)
            return self._pick_for_state("acknowledge_default", state=state, action=ActionType.ACKNOWLEDGE)

        if action == ActionType.SHARE_FEELING:
            if "repair_attempt" in features.pragmatic_cues:
                return self._pick_for_state("share_feeling_repair", state=state, action=ActionType.SHARE_FEELING)
            if "relationship_check" in features.pragmatic_cues:
                return self._pick_for_state(
                    "share_feeling_relationship_check",
                    state=state,
                    action=ActionType.SHARE_FEELING,
                )
            if "quiet_weather_feeling" in features.pragmatic_cues:
                return self._pick_for_state(
                    "share_feeling_quiet_weather",
                    state=state,
                    action=ActionType.SHARE_FEELING,
                )
            if "social_awkwardness" in features.pragmatic_cues:
                return self._pick_for_state(
                    "share_feeling_social_awkwardness",
                    state=state,
                    action=ActionType.SHARE_FEELING,
                )
            if "low_energy_checkin" in features.pragmatic_cues:
                return self._pick_for_state(
                    "share_feeling_low_energy",
                    state=state,
                    action=ActionType.SHARE_FEELING,
                )
            if "subdued_positive" in features.pragmatic_cues:
                return self._pick_for_state(
                    "share_feeling_subdued_positive",
                    state=state,
                    action=ActionType.SHARE_FEELING,
                )
            if "self_conscious_check" in features.pragmatic_cues:
                return self._pick_for_state("share_feeling_reassure", state=state, action=ActionType.SHARE_FEELING)
            if "face_saving_retreat" in features.pragmatic_cues:
                return self._pick_for_state("share_feeling_reassure", state=state, action=ActionType.SHARE_FEELING)
            if "complaint_emphasis" in features.pragmatic_cues:
                return self._pick_for_state("share_feeling_complaint", state=state, action=ActionType.SHARE_FEELING)
            if not phrasing_plan.asks_followup:
                return self._pick_for_state("share_feeling_light_touch", state=state, action=ActionType.SHARE_FEELING)
            return self._pick_for_state("share_feeling", state=state, action=ActionType.SHARE_FEELING)

        if action == ActionType.SHARE_OPINION:
            response_plan = decision.response_plan
            if self.persona.lower() == "black":
                practical_reply = render_black_practical_direct_reply(user_text=features.content)
                if practical_reply:
                    return practical_reply
            if (
                self.persona.lower() == "black"
                and response_plan is not None
                and response_plan.stance == "concrete_topic_answer"
            ):
                concrete_reply = render_black_concrete_topic_question_reply(user_text=features.content)
                if concrete_reply:
                    return concrete_reply
            if (
                self.persona.lower() == "black"
                and response_plan is not None
                and response_plan.stance
                in {
                    "concrete_self_style_answer",
                    "habit_preference_answer",
                    "direct_preference_disclosure",
                }
            ):
                persona_reply = render_black_persona_daily_question_reply(
                    user_text=features.content,
                    stance=response_plan.stance,
                    allow_generic=False,
                )
                if persona_reply:
                    return persona_reply
            if "schema_conversation_topic_suggestion" in decision.reason_flags:
                options = [
                    item.strip()
                    for item in decision.slots.get("conversation_topic_options", "").split("|")
                    if item.strip()
                ]
                first = decision.slots.get("conversation_topic_first") or (options[0] if options else "오늘 컨디션")
                second = options[1] if len(options) > 1 else "요즘 본 영상"
                third = options[2] if len(options) > 2 else "다음에 같이 해볼 것"
                return f"대화 주제는 {first}, {second}, {third} 정도면 돼. 먼저 {first}부터 가볍게 열자."
            if "schema_activity_preparation_advice" in decision.reason_flags:
                items = [item.strip() for item in decision.slots.get("preparation_items", "").split("|") if item.strip()]
                activity = decision.slots.get("preparation_activity") or "활동"
                if len(items) >= 4:
                    return f"{activity}이면 {items[0]}, {items[1]}, {items[2]}부터 챙겨. 길어질 것 같으면 {items[3]}도 보는 게 좋아."
                if len(items) >= 2:
                    return f"{activity}이면 {items[0]}이랑 {items[1]}부터 챙겨. 나머지는 길이에 맞추면 돼."
                return f"{activity}이면 물이랑 편한 신발부터 챙겨. 오래 걸리면 간식도 보는 게 좋아."
            broad_opinion_reply = self._render_broad_opinion_reply(features=features)
            if broad_opinion_reply is not None:
                return broad_opinion_reply
            if "opinion_self_style" in features.pragmatic_cues:
                return self._pick_for_state(
                    "share_opinion_self_style",
                    state=state,
                    action=ActionType.SHARE_OPINION,
                )
            if "opinion_habit_preference" in features.pragmatic_cues:
                return self._pick_for_state(
                    "share_opinion_habit_preference",
                    state=state,
                    action=ActionType.SHARE_OPINION,
                )
            if world_state is not None and world_state.boundary_history in {"active_boundary", "firm_boundary"}:
                return self._pick_for_state("share_opinion", state=state, action=ActionType.SHARE_OPINION)
            if features.intent.value == "smalltalk_opinion":
                return self._pick_for_state("share_opinion", state=state, action=ActionType.SHARE_OPINION)
            return self._pick_for_state("share_opinion", state=state, action=ActionType.SHARE_OPINION)

        if action == ActionType.GAME_CHAT:
            if "스팀" in features.normalized and any(marker in features.normalized for marker in ("할 만한", "할만한", "있어")):
                return "스팀이면 가볍게는 협동 생존이나 로그라이크 쪽이 좋아. 오래 붙잡힐 거면 취향부터 보고 고르는 게 나아."
            return _pick("game_chat")

        if action == ActionType.GAME_ACCEPT_OR_DECLINE:
            return _pick("game_accept_or_decline")

        if action == ActionType.ACCEPT_ACTIVITY_INVITE:
            activity = decision.slots.get("activity_name") or "그거"
            place = decision.slots.get("activity_place")
            context = decision.slots.get("activity_context")
            detail = decision.slots.get("activity_detail")
            condition = decision.slots.get("activity_condition")
            if activity in {"바베큐", "바비큐", "고기 굽기"} or detail in {"구워먹기", "굽기"}:
                activity_text = "바베큐" if activity == "바비큐" else activity
                if detail == "구워먹기" and "구워먹기" not in activity_text:
                    activity_text = f"{activity_text} 구워먹기"
                elif detail == "굽기" and "굽기" not in activity_text:
                    activity_text = f"{activity_text} 굽기"
                prefix = f"{context}하면서 " if context else ""
                return f"{prefix}{activity_text} 좋지. 불 앞에서 바로 먹으면 그 맛이 있어."
            if activity in {"수영", "물놀이"} and place:
                prefix = f"{place}가 시원하면 " if condition and "시원" in condition else f"{place}면 "
                return f"{prefix}{activity} 좋지. 물만 너무 차갑지 않으면 가볍게 들어가자."
            if place:
                return f"{place}에서 {activity} 좋지. 가볍게 하자."
            return f"{activity} 좋지. 가볍게 하자."

        if action == ActionType.MUSIC_CHAT:
            if "운동" in features.normalized and any(marker in features.normalized for marker in ("음악", "노래", "듣기", "들을")):
                return "운동할 때는 박자가 또렷한 곡이 좋아. 너무 잔잔한 것보다 몸 움직이는 템포가 먼저야."
            music_text = decision.slots.get("music_text")
            if music_text:
                return music_text
            preference_reply = self._render_music_preference_reply(decision=decision, state=state)
            if preference_reply is not None:
                return preference_reply
            return _pick("music_chat")

        if action == ActionType.RECOMMEND:
            recommendation_text = decision.slots.get("recommendation_text")
            if recommendation_text:
                return recommendation_text
            preference_reply = self._render_recommendation_preference_reply(decision=decision, state=state)
            if preference_reply is not None:
                return preference_reply
            return _pick("recommend")

        if action == ActionType.REACT_LAUGH:
            return _pick("react_laugh")

        if action == ActionType.REACT_SURPRISE:
            return _pick("react_surprise")

        if action == ActionType.TELL_TIME:
            time_text = decision.slots.get("time_text")
            if time_text:
                return self._append_time_context(time_text, decision)
            return _pick("tell_time")

        if action == ActionType.SEARCH_ANSWER:
            meaning_reply = _render_meaning_answer(features.content)
            if meaning_reply is not None:
                return meaning_reply
            query_type = decision.slots.get("knowledge_query_type")
            grounded_answer = decision.slots.get("knowledge_answer")
            grounded_subject = decision.slots.get("knowledge_subject")
            if grounded_answer:
                if query_type == "capital" and grounded_subject:
                    base = f"{grounded_subject}의 수도는 {grounded_answer}야."
                    return self._append_grounding_source_note(base, decision.slots.get("knowledge_source"))
                if query_type == "flag" and grounded_subject:
                    base = f"{grounded_subject}의 국기는 {grounded_answer}야."
                    return self._append_grounding_source_note(base, decision.slots.get("knowledge_source"))
                if query_type == "location" and grounded_subject:
                    base = f"{grounded_subject}은 {grounded_answer}"
                    return self._append_grounding_source_note(base, decision.slots.get("knowledge_source"))
                return self._append_grounding_source_note(grounded_answer, decision.slots.get("knowledge_source"))
            return _pick("search_answer")

        if action == ActionType.NEWS_ANSWER:
            news_summary = decision.slots.get("news_summary")
            if news_summary:
                return self._append_news_source_note(news_summary, decision.slots.get("knowledge_source"))
            return _pick("news_answer")

        if action == ActionType.CONTINUE_CONVERSATION:
            if not phrasing_plan.asks_followup:
                return self._pick_for_state(
                    "small_talk_light_touch",
                    state=state,
                    action=ActionType.CONTINUE_CONVERSATION,
                )
            if features.intent == Intent.TEASE:
                return self._pick_for_state(
                    "continue_conversation_tease_soft",
                    state=state,
                    action=ActionType.CONTINUE_CONVERSATION,
                )
            return self._pick_for_state("continue_conversation", state=state, action=ActionType.CONTINUE_CONVERSATION)

        if action == ActionType.EXPLAIN_REASON:
            if explanation_trace is not None:
                return self._render_trace_explanation(explanation_trace)
            return "직전 판단 기록이 비어 있어. 방금 한 판단부터 다시 쌓아야 해."

        if action == ActionType.ASK_CLARIFICATION:
            if "rewrite_target_missing" in decision.reason_flags:
                return "바꿀 원문을 같이 줘. 그 문장 기준으로 덜 날카롭게 바꿔볼게."
            if "reason_reference_missing" in decision.reason_flags:
                return "어떤 판단을 말하는지 한 줄만 같이 줘. 그 기준부터 근거를 짚을게."
            if features.intent.value == "reply_request":
                return self._pick_for_state(
                    "ask_clarification_reply_request",
                    state=state,
                    action=ActionType.ASK_CLARIFICATION,
                )
            if features.intent.value == "why":
                return self._pick_for_state(
                    "ask_clarification_why",
                    state=state,
                    action=ActionType.ASK_CLARIFICATION,
                )
            return self._pick_for_state(
                "ask_clarification_default",
                state=state,
                action=ActionType.ASK_CLARIFICATION,
            )

        if action == ActionType.SMALL_TALK:
            if not phrasing_plan.asks_followup:
                return self._pick_for_state("small_talk_light_touch", state=state, action=ActionType.SMALL_TALK)
            if features.intent.value == "smalltalk_generic":
                return self._pick_for_state("small_talk_compliment", state=state, action=ActionType.SMALL_TALK)
            if features.intent.value == "thanks":
                return self._pick_for_state("small_talk_thanks", state=state, action=ActionType.SMALL_TALK)
            return self._pick_for_state("small_talk_greeting", state=state, action=ActionType.SMALL_TALK)

        return _pick("ask_clarification_default")

    def _postprocess_reply(
        self,
        *,
        reply: str,
        features: MessageFeatures,
        decision: ActionDecision,
        state: ConversationState,
        source: str,
    ) -> str:
        cleaned = reply.strip()
        if self.persona.lower() != "black":
            return cleaned

        cleaned = _normalize_black_reply_text(cleaned)
        if not self.output_guard_enabled:
            return cleaned
        if source == "llm" and _looks_like_black_malformed_handoff_reply(
            user_text=features.content,
            reply=cleaned,
        ):
            self._record_generation_issue("llm_unusable_reply:black_malformed_handoff")
        if (
            source == "llm"
            and decision.action in BLACK_ECHO_GUARD_ACTIONS
            and (
                _looks_like_black_echo_reply(user_text=features.content, reply=cleaned)
                or self._looks_like_recent_loop_reply(
                    state=state,
                    reply=cleaned,
                    action=decision.action,
                )
            )
        ):
            self._record_generation_issue("llm_unusable_reply:black_echo_loop")
        return cleaned

    def _build_black_retry_context(
        self,
        *,
        exc: Exception,
        reply_candidate: str | None,
        features: MessageFeatures,
        decision: ActionDecision,
        state: ConversationState,
    ) -> dict[str, str] | None:
        if self.persona.lower() != "black":
            return None
        if self.llm_client is None:
            return None
        if decision.action not in BLACK_ECHO_GUARD_ACTIONS:
            return None
        if "unusable reply" not in str(exc).lower():
            return None

        previous_candidate = _normalize_black_reply_text(reply_candidate or "").strip()
        if not previous_candidate:
            previous_candidate = "none"

        recent_replies = self._recent_replies_for_action(state, decision.action, limit=2)
        recent_tail = recent_replies[0] if recent_replies else "none"
        if _looks_like_black_echo_reply(user_text=features.content, reply=previous_candidate):
            issue = "the candidate echoed the user's wording too closely"
        else:
            issue = f"the candidate was too close to a recent same-action reply: {recent_tail}"

        return {
            "issue": issue,
            "previous_candidate": previous_candidate,
        }

    @staticmethod
    def _pick_for_state(
        pool_key: str,
        *,
        state: ConversationState,
        action: ActionType,
    ) -> str:
        pool = RESPONSE_POOL.get(pool_key)
        if not pool:
            return "응답은 해. 조금 더 알려줘."

        recent_replies = ResponseRenderer._recent_replies_for_action(state, action, limit=2)
        if not recent_replies:
            return random.choice(pool)
        recent_signatures = {_reply_signature(reply) for reply in recent_replies}

        for reply in recent_replies:
            if reply in pool:
                start = (pool.index(reply) + 1) % len(pool)
                for offset in range(len(pool)):
                    candidate = pool[(start + offset) % len(pool)]
                    if candidate not in recent_replies and _reply_signature(candidate) not in recent_signatures:
                        return candidate
                return pool[start]

        for candidate in pool:
            if candidate not in recent_replies and _reply_signature(candidate) not in recent_signatures:
                return candidate
        return pool[0]

    @staticmethod
    def _looks_like_recent_loop_reply(
        *,
        state: ConversationState,
        reply: str,
        action: ActionType,
    ) -> bool:
        reply_signature = _reply_signature(reply)
        reply_compact = _compact_text(reply)
        if not reply_signature:
            return False

        for previous_reply in ResponseRenderer._recent_replies_for_action(state, action, limit=2):
            previous_signature = _reply_signature(previous_reply)
            previous_compact = _compact_text(previous_reply)
            if reply_signature == previous_signature:
                return True
            if reply_compact and previous_compact and (
                reply_compact in previous_compact or previous_compact in reply_compact
            ):
                return True
        return False

    @staticmethod
    def _recent_replies_for_action(
        state: ConversationState,
        action: ActionType,
        *,
        limit: int,
    ) -> list[str]:
        replies: list[str] = []
        for turn in reversed(state.recent_turns):
            if turn.action != action:
                continue
            text = turn.bot_text.strip()
            if not text:
                continue
            replies.append(text)
            if len(replies) >= limit:
                break
        return replies

    @staticmethod
    def _render_music_preference_reply(
        *,
        decision: ActionDecision,
        state: ConversationState,
    ) -> str | None:
        update_key = decision.slots.get("preference_update_key")
        update_value = decision.slots.get("preference_update_value")
        if update_key == "music_like" and update_value:
            return f"오케이. {update_value} 좋아하는 쪽으로 기억해둘게. 다음에 음악 얘기할 때 그 결로 바로 받을게."
        if update_key == "music_dislike" and update_value:
            return f"좋아. {update_value} 쪽은 덜 끌리는 걸로 기억해둘게. 다음엔 그쪽은 좀 빼고 얘기해볼게."

        music_like = state.preference_memory.get("music_like")
        music_dislike = state.preference_memory.get("music_dislike")
        if music_like and music_dislike:
            return f"너는 {music_like} 좋아하고 {music_dislike} 쪽은 덜 좋아한다고 했지. 요즘도 그 결로 듣는 편이야?"
        if music_like:
            return f"지난번에 {music_like} 좋아한다고 했지. 요즘도 그 결로 듣는 편이야?"
        if music_dislike:
            return f"지난번에 {music_dislike} 쪽은 덜 끌린다고 했지. 그럼 요즘은 어떤 쪽이 더 맞아?"
        return None

    @staticmethod
    def _render_recommendation_preference_reply(
        *,
        decision: ActionDecision,
        state: ConversationState,
    ) -> str | None:
        update_key = decision.slots.get("preference_update_key")
        update_value = decision.slots.get("preference_update_value")
        if update_key == "media_like" and update_value:
            return f"오케이. {update_value} 좋아하는 걸로 기억해둘게. 다음 추천은 그 결부터 맞춰볼게."
        if update_key == "media_dislike" and update_value:
            return f"좋아. {update_value} 쪽은 덜 끌리는 걸로 기억해둘게. 다음 추천에선 그건 먼저 빼고 볼게."

        media_like = state.preference_memory.get("media_like")
        media_dislike = state.preference_memory.get("media_dislike")
        if media_like and media_dislike:
            return (
                f"지난번에 {media_like} 좋아하고 {media_dislike}는 덜 끌린다고 했지. "
                "그 결로 좁혀볼게. 영화로 볼지 시리즈로 볼지만 말해줘."
            )
        if media_like:
            return (
                f"지난번에 {media_like} 좋아한다고 했지. "
                "그 결로 먼저 좁혀볼게. 영화로 볼지 시리즈로 볼지만 정해줘."
            )
        if media_dislike:
            return (
                f"지난번에 {media_dislike} 쪽은 덜 끌린다고 했지. "
                "그쪽은 빼고 볼게. 대신 어떤 분위기 찾는지만 말해줘."
            )
        return None

    def _render_broad_opinion_reply(
        self,
        *,
        features: MessageFeatures,
    ) -> str | None:
        if features.intent != Intent.SMALLTALK_OPINION:
            return None

        text = " ".join(features.content.strip().split())
        normalized = features.normalized
        cues = set(features.pragmatic_cues)

        if (
            features.question_schema in {"self_style", "habit_preference"}
            or cues & {"opinion_self_style", "opinion_habit_preference"}
        ):
            return None

        if "weather_conditioned_activity_opinion" in cues:
            activity = self._extract_activity_topic(text)
            if activity:
                return f"{activity}이면 무리만 아니면 해볼 만하긴 해. 바람이나 컨디션만 너무 안 흔들리면 괜찮아 보여."
            return "그 조건이면 무리만 아니면 해볼 만하긴 해. 너무 빡세지만 않으면 괜찮아 보여."

        if "activity_recommendation" in cues:
            place = self._extract_activity_place(text)
            options = self._activity_options_for_place(place)
            if place:
                return f"{place}이면 {options[0]}하고, 지치면 {options[1]} 쪽이 무난해. {options[2]}도 괜찮고."
            return f"{options[0]}하고, 지치면 {options[1]} 쪽이 무난해. {options[2]}도 괜찮고."

        if "honesty_boundary" in cues:
            if "증시" in normalized or "주식" in normalized:
                return "그건 지금 확인한 사실이 아니라서 단정하진 않을게. 조회 근거가 없으면 올랐다고 말하면 안 돼."
            if "면접" in normalized and "결과" in normalized:
                return "그건 내가 알 수 없는 결과라 맞히진 않을게. 확인 전엔 모른다고 보는 게 맞아."
            if "표정" in normalized:
                return "지금 네 표정은 내가 볼 수 없어서 몰라. 보이는 근거 없이 단정하진 않을게."
            if "어디" in normalized:
                return "네가 어디 있는지는 내가 알 수 없어. 위치를 말해주기 전엔 추측하지 않을게."
            if "비밀" in normalized:
                return "네가 숨긴 비밀은 내가 몰라. 모르는 걸 아는 척하진 않을게."
            return "그건 내가 확실히 아는 정보가 아니야. 근거 없이 단정하진 않을게."

        if "opinion_preference_like" in cues:
            topic = self._extract_preference_topic(text)
            if "싫지 않아" in normalized or "싫어" in normalized:
                if topic:
                    return f"{topic} 쪽이 너무 부담스럽게 오면 나도 좀 거리를 두는 편이야."
                return "너무 부담스럽게 오면 나도 좀 거리를 두는 편이야."
            if topic:
                return f"{topic} 쪽은 나는 꽤 좋아하는 편이야. 완전 빠지는 정도까진 아니어도 있으면 반갑지."
            return self._pick_for_state(
                "share_opinion_preference_like",
                state=ConversationState(user_id="template"),
                action=ActionType.SHARE_OPINION,
            )

        if "opinion_reflective_judgment" in cues:
            if "낫지" in normalized:
                return "응, 그 상황이면 그쪽이 더 무난해 보이긴 해."
            if "중요하지" in normalized:
                return "응, 그런 상황이면 그 포인트가 더 중요해지긴 해."
            if "이해돼" in normalized:
                return "응, 그 말은 충분히 이해되는 쪽이야."
            return self._pick_for_state(
                "share_opinion_reflective_judgment",
                state=ConversationState(user_id="template"),
                action=ActionType.SHARE_OPINION,
            )

        if "opinion_advice_process" in cues:
            if "실험해볼까" in normalized:
                return "나라면 한 번에 다 바꾸기보다 변수 하나만 바꿔서 볼 것 같아."
            if "무난할까" in normalized:
                return "처음엔 너무 튀는 쪽보다 무난한 선택이 안전하긴 해."
            if "어떤 쪽을 우선" in normalized:
                return "나라면 이미 잡혀 있던 쪽부터 먼저 지키는 편이긴 해."
            if "무엇부터" in normalized or "뭘 먼저" in normalized or "우선" in normalized:
                return "나라면 바로 결론보다 먼저 볼 기준부터 하나 세울 것 같아."
            return self._pick_for_state(
                "share_opinion_advice_process",
                state=ConversationState(user_id="template"),
                action=ActionType.SHARE_OPINION,
            )

        if "opinion_decision_request" in cues:
            activity = self._extract_activity_topic(text)
            if activity:
                return f"{activity}이면 크게 무리만 아니면 해볼 만하긴 해."
            if "무엇을 우선" in normalized or "어떤 쪽을 우선" in normalized:
                return "그 상황이면 나는 먼저 약속해 둔 쪽부터 지키는 편이야."
            return self._pick_for_state(
                "share_opinion_decision_request",
                state=ConversationState(user_id="template"),
                action=ActionType.SHARE_OPINION,
            )

        if "broad_opinion_question" in cues:
            return self._pick_for_state(
                "share_opinion",
                state=ConversationState(user_id="template"),
                action=ActionType.SHARE_OPINION,
            )

        return None

    @staticmethod
    def _extract_preference_topic(text: str) -> str | None:
        patterns = (
            r"(.+?)\s*좋아해\?$",
            r"(.+?)\s*좋아하냐\?$",
            r"(.+?)\s*좋아하니\?$",
            r"(.+?)\s*좋아\?$",
            r"(.+?)\s*싫지\s*않아\?$",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                topic = match.group(1).strip(" ?")
                topic = re.sub(r"(은|는|이|가|을|를|도|건|거)$", "", topic)
                if not topic:
                    return None
                if len(topic) > 8:
                    return None
                if topic.count(" ") > 1:
                    return None
                return topic
        return None

    @staticmethod
    def _extract_activity_topic(text: str) -> str | None:
        match = re.search(
            r"(배드민턴|산책|자전거|러닝|조깅|피크닉|테니스|농구|축구|캠핑|달리기|등산)",
            text,
        )
        return match.group(1) if match else None

    @staticmethod
    def _extract_activity_place(text: str) -> str:
        places = ("바다", "해변", "해수욕장", "계곡", "공원", "한강", "산", "캠핑장", "놀이공원", "실내", "도서관", "카페")
        for place in sorted(places, key=len, reverse=True):
            if place in text:
                return place
        match = re.search(r"([가-힣A-Za-z0-9]+)(?:에서|가서|으로|로)\s*(?:무엇|뭐|뭘|어떤)", text)
        return match.group(1) if match else ""

    @staticmethod
    def _activity_options_for_place(place: str) -> tuple[str, str, str, str]:
        options = {
            "바다": ("물놀이", "모래사장 산책", "사진 찍기", "돗자리 펴고 쉬기"),
            "해변": ("물놀이", "모래사장 산책", "사진 찍기", "돗자리 펴고 쉬기"),
            "해수욕장": ("물놀이", "모래사장 산책", "사진 찍기", "돗자리 펴고 쉬기"),
            "계곡": ("발 담그기", "물가 산책", "간단한 간식", "사진 찍기"),
            "공원": ("산책", "가벼운 공놀이", "사진 찍기", "돗자리 펴고 쉬기"),
            "한강": ("산책", "자전거", "돗자리 펴고 쉬기", "간단한 간식"),
            "산": ("가벼운 산책", "전망 보기", "사진 찍기", "간식 먹기"),
            "캠핑장": ("불멍", "간단한 요리", "산책", "보드게임"),
            "놀이공원": ("가벼운 놀이기구", "퍼레이드 보기", "사진 찍기", "간식 먹기"),
            "실내": ("보드게임", "영화 보기", "간단한 간식", "카페에서 쉬기"),
            "도서관": ("근처 카페 들르기", "가벼운 산책", "서점 구경", "간단한 보드게임"),
            "카페": ("디저트 나눠 먹기", "짧은 보드게임", "사진 찍기", "근처 산책"),
        }
        return options.get(place, ("산책", "사진 찍기", "간단한 게임", "쉬면서 이야기하기"))

    @staticmethod
    def _append_time_context(time_text: str, decision: ActionDecision) -> str:
        timezone_name = decision.slots.get("time_timezone")
        if not timezone_name or timezone_name in time_text:
            return time_text
        return f"{time_text} 기준 시간대는 {timezone_name}이야."

    @staticmethod
    def _append_grounding_source_note(base: str, knowledge_source: str | None) -> str:
        label = ResponseRenderer._knowledge_source_label(knowledge_source)
        if label is None or label in base:
            return base
        return f"{base} 근거는 {label} 기준이야."

    @staticmethod
    def _append_news_source_note(base: str, knowledge_source: str | None) -> str:
        label = ResponseRenderer._knowledge_source_label(knowledge_source)
        if label is None or label in base:
            return base
        return f"{base}\n출처 묶음: {label}"

    @staticmethod
    def _knowledge_source_label(knowledge_source: str | None) -> str | None:
        if not knowledge_source:
            return None
        if knowledge_source.startswith("wikidata_"):
            return "Wikidata"
        if knowledge_source.startswith("builtin_country_"):
            return "기본 국가 정보"
        if knowledge_source == "google_news_rss":
            return "Google News RSS"
        if knowledge_source in {"system_clock", "fake_clock"}:
            return "로컬 시스템 시계"
        if knowledge_source == "curated_media_catalog":
            return "큐레이션 미디어 카탈로그"
        if knowledge_source == "curated_music_catalog":
            return "큐레이션 음악 카탈로그"
        return knowledge_source

    @staticmethod
    def _render_trace_explanation(explanation_trace: DecisionTrace) -> str:
        if explanation_trace.logic_chain:
            return ResponseRenderer._render_logic_chain_explanation(explanation_trace)

        filtered = [
            item
            for item in explanation_trace.reason_trace
            if item.code != "policy_candidates_considered"
        ]
        if not filtered:
            return "기록은 있는데 근거 줄이 비어 있어. 다시 판단을 쌓아야 해."

        action_entry = next(
            (
                item
                for item in filtered
                if item.code.startswith("selected_") or item.code == "explain_previous_decision"
            ),
            None,
        )
        rationale_entries = [item for item in filtered if item is not action_entry]
        prioritized = sorted(
            rationale_entries,
            key=lambda item: ResponseRenderer._reason_priority(item.code),
        )
        premises = [
            ResponseRenderer._strip_internal_reason_tokens(
                ResponseRenderer._humanize_logic_text(item.summary)
            )
            for item in prioritized[:2]
        ]

        if action_entry is None:
            summaries = premises or [filtered[0].summary]
            base = "방금은 이렇게 봤어. " + " ".join(summaries)
            return ResponseRenderer._append_counterfactual(base, explanation_trace)

        conclusion = ResponseRenderer._strip_internal_reason_tokens(
            ResponseRenderer._humanize_logic_text(
                ResponseRenderer._normalize_conclusion(action_entry.summary)
            )
        )
        if not premises:
            base = f"방금은 이렇게 봤어. 그래서 {conclusion}"
            return ResponseRenderer._append_counterfactual(base, explanation_trace)

        base = f"방금은 이렇게 봤어. {' '.join(premises)} 그래서 {conclusion}"
        return ResponseRenderer._append_counterfactual(base, explanation_trace)

    @staticmethod
    def _reason_priority(code: str) -> int:
        if code.startswith("unresolved_"):
            return 0
        if code.startswith("constraint_"):
            return 1
        if code.startswith("response_need_"):
            return 2
        if code.startswith("news_topic_"):
            return 3
        if code in {"recommendation_focus_used", "music_focus_used"}:
            return 4
        if code.startswith("grounding_source_"):
            return 5
        if code.startswith("policy_margin_axis_"):
            return 6
        if code.startswith("speech_act_"):
            return 7
        if code.startswith("policy_margin_vs_"):
            return 8
        if code.startswith("emotion_"):
            return 9
        if code == "factuality_required":
            return 10
        if code == "location_detected":
            return 11
        if code.startswith("intent_"):
            return 12
        if code.startswith("topic_"):
            return 13
        if code.startswith("classifier_source_"):
            return 14
        if code == "preference_memory_updated":
            return 15
        if code == "memory_context_used":
            return 16
        return 99

    @staticmethod
    def _normalize_conclusion(summary: str) -> str:
        stripped = summary.strip()
        if stripped.startswith("그래서 "):
            return stripped.removeprefix("그래서 ").strip()
        return stripped

    @staticmethod
    def _append_counterfactual(base: str, explanation_trace: DecisionTrace) -> str:
        if not explanation_trace.counterfactuals:
            return base
        top = explanation_trace.counterfactuals[0]
        if top.condition == "핵심 단서가 조금만 달랐다면":
            return base
        action_label = ResponseRenderer._action_label(top.predicted_action)
        return f"{base} {top.condition} {action_label} 쪽으로 갔을 거야."

    @staticmethod
    def _render_logic_chain_explanation(explanation_trace: DecisionTrace) -> str:
        logic_steps = explanation_trace.logic_chain
        observation = next((item for item in logic_steps if item.step_type == "observation"), None)
        inference_steps = [item for item in logic_steps if item.step_type == "inference"]
        constraint = next((item for item in logic_steps if item.step_type == "constraint"), None)
        comparison = next((item for item in logic_steps if item.step_type == "comparison"), None)
        decision = next((item for item in logic_steps if item.step_type == "decision"), None)

        parts = ["방금은 논리적으로 이렇게 봤어."]

        def add_part(text: str | None) -> None:
            if not text:
                return
            cleaned = ResponseRenderer._humanize_logic_text(text)
            cleaned = ResponseRenderer._strip_internal_reason_tokens(cleaned)
            if not cleaned or cleaned in parts:
                return
            parts.append(cleaned)

        if observation is not None:
            add_part(observation.premise)
        prioritized_inferences = ResponseRenderer._prioritized_user_facing_inferences(inference_steps)
        for inference in prioritized_inferences[:2]:
            add_part(inference.conclusion)
        response_plan_note = ResponseRenderer._render_response_plan_explanation(explanation_trace.response_plan)
        if response_plan_note is not None:
            add_part(response_plan_note)
        if constraint is not None:
            add_part(constraint.conclusion)
        if comparison is not None:
            add_part(comparison.conclusion)
        if decision is not None:
            add_part(ResponseRenderer._logic_decision_text(explanation_trace, decision.conclusion))

        base = " ".join(parts)
        return ResponseRenderer._append_counterfactual(base, explanation_trace)

    @staticmethod
    def _prioritized_user_facing_inferences(inference_steps: list[object]) -> list[object]:
        has_pragmatic = any(
            str(getattr(step, "rule_id", "")).startswith("infer.pragmatics.")
            for step in inference_steps
        )

        def priority(step: object) -> int:
            rule_id = getattr(step, "rule_id", "")
            if rule_id.startswith("infer.pragmatics."):
                return 0
            if rule_id.startswith("infer.grounding."):
                return 1
            if rule_id.startswith("infer.news_topic."):
                return 2
            if rule_id.startswith("infer.preference."):
                return 3
            if rule_id.startswith("infer.response_need."):
                return 4
            if has_pragmatic and rule_id.startswith("infer.speech_act."):
                return 99
            if rule_id.startswith("infer.speech_act.ask"):
                return 5
            if rule_id.startswith("infer.topic."):
                return 6
            return 99

        return [
            step
            for step in sorted(inference_steps, key=priority)
            if priority(step) < 99
        ]

    @staticmethod
    def _logic_decision_text(explanation_trace: DecisionTrace, conclusion: str) -> str:
        normalized = ResponseRenderer._normalize_conclusion(conclusion)
        response_plan = explanation_trace.response_plan
        if (
            explanation_trace.selected_action == ActionType.SHARE_OPINION
            and response_plan is not None
            and response_plan.anchor.strip() == "대화 주제"
        ):
            return "그래서 주제 후보를 짧게 제안하는 쪽으로 정리했다."
        return normalized

    @staticmethod
    def _render_response_plan_explanation(response_plan: ResponsePlan | None) -> str | None:
        if response_plan is None:
            return None

        parts: list[str] = []
        anchor = response_plan.anchor.strip()
        if anchor:
            parts.append(f"핵심 주제는 '{anchor}'{_direction_particle(anchor)} 잡았다.")

        stance_labels = {
            "conditional_go_or_no_go": "정답처럼 단정하기보단 조건부로 해볼 만한지 판단하는 쪽으로 잡았다.",
            "supportive": "먼저 감정을 받아주는 쪽으로 잡았다.",
            "direct_answer": "질문에 바로 답하는 쪽으로 잡았다.",
            "neutral": "톤은 중립에 가깝게 잡았다.",
        }
        stance_note = stance_labels.get(response_plan.stance)
        if stance_note is not None:
            parts.append(stance_note)

        return " ".join(parts) if parts else None

    @staticmethod
    def _humanize_logic_text(text: str) -> str:
        cleaned = text.strip()
        labels = {
            "detector:is_decision_request_question_text": "선택 판단 질문 패턴",
            "detector:is_conversation_topic_suggestion_text": "대화 주제를 골라 달라는 패턴",
            "detector:is_activity_recommendation_question_text": "활동 추천 질문 패턴",
            "detector:is_light_food_recommendation_question_text": "가벼운 음식 추천 질문 패턴",
            "detector:is_unverified_memory_boundary_question_text": "확인 안 된 기억을 묻는 패턴",
            "detector:is_relationship_dependency_boundary_question_text": "관계 경계를 묻는 패턴",
            "detector:is_opinion_self_style_question_text": "내 반응 방식을 묻는 패턴",
            "smalltalk_opinion": "의견 요청",
            "conversation_topic_suggestion": "대화 주제 제안 요청",
            "activity_recommendation": "활동 추천 요청",
            "light_food_recommendation": "가벼운 음식 추천 요청",
            "memory_boundary": "확인 안 된 기억 경계",
            "relationship_boundary": "관계 경계",
            "self_style": "내 반응 방식",
            "ask": "질문",
            "inform": "상태를 알려주는 말",
            "opinion_decision_request": "선택에 대한 의견 요청",
            "soft_decision_advice": "부드러운 결정 조언",
            "share_opinion": "의견을 주는 대응",
            "continue_conversation": "대화를 이어가는 대응",
            "state": "상황",
        }
        for raw, label in labels.items():
            cleaned = cleaned.replace(f"`{raw}`", label)
            cleaned = cleaned.replace(raw, label)

        replacements = {
            "입력에서 선택 판단 질문 패턴 신호가 잡혔다.": "입력에 선택 판단을 묻는 패턴이 있었다.",
            "입력에서 대화 주제를 골라 달라는 패턴 신호가 잡혔다.": "입력에 대화 주제를 골라 달라는 패턴이 있었다.",
            "입력에서 활동 추천 질문 패턴 신호가 잡혔다.": "입력에 활동 후보를 묻는 패턴이 있었다.",
            "입력에서 가벼운 음식 추천 질문 패턴 신호가 잡혔다.": "입력에 부담 적은 음식 후보를 묻는 패턴이 있었다.",
            "입력에서 확인 안 된 기억을 묻는 패턴 신호가 잡혔다.": "확인되지 않은 기억을 어떻게 다룰지 묻는 패턴이었다.",
            "입력에서 관계 경계를 묻는 패턴 신호가 잡혔다.": "관계에서 어디까지 받아줄지 묻는 패턴이었다.",
            "입력에서 내 반응 방식을 묻는 패턴 신호가 잡혔다.": "내가 어떤 방식으로 반응하는지 묻는 패턴이었다.",
            "우선 의견 요청 계열 해석을 출발점으로 잡았다.": "단순 잡담보다 의견 요청에 가깝게 봤다.",
            "따라서 발화 기능은 질문 쪽으로 해석했다.": "그래서 사용자가 판단을 요구하는 질문으로 해석했다.",
            "따라서 발화 기능은 상태를 알려주는 말 쪽으로 해석했다.": "상대가 정보를 새로 요구하기보다 흐름을 알려주는 말로 봤다.",
            "입력을 절/의미 단위로 나누니 상황 같은 proposition이 잡혔다.": "문장을 의미 단위로 나누니 사용자가 묻는 상황 자체가 핵심으로 잡혔다.",
            "그래서 intent/action 판단도 표면 문장 하나보다 구조화된 evidence 쪽에 더 기대도록 했다.": "그래서 표면 문장보다 의미 단위와 단서를 같이 보고 판단했다.",
            "선택에 대한 의견 요청 화용론 단서가 감지됐다.": "선택에 대한 의견을 달라는 단서가 있었다.",
            "대화 주제 제안 요청 화용론 단서가 감지됐다.": "대화를 이어갈 주제 후보를 달라는 단서가 있었다.",
            "활동 추천 요청 화용론 단서가 감지됐다.": "실제로 해볼 활동 후보를 달라는 단서가 있었다.",
            "가벼운 음식 추천 요청 화용론 단서가 감지됐다.": "부담 적은 음식 후보를 달라는 단서가 있었다.",
            "짧게 의견을 주는 쪽으로 정리했다.": "그래서 짧게 의견을 주는 쪽으로 정리했다.",
        }
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        return cleaned

    @staticmethod
    def _strip_internal_reason_tokens(text: str) -> str:
        cleaned = text.strip().replace("`", "")
        cleaned = re.sub(r"\bdetector:[\w:.\-]+", "분류 단서", cleaned)
        cleaned = re.sub(r"\b(schema|intent|policy|classifier|question_schema|pragmatic_cue)[:=_][\w:.\-]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _action_label(action: ActionType) -> str:
        mapping = {
            ActionType.ASK_LOCATION: "먼저 위치를 확인하는 대응",
            ActionType.WEATHER_LOOKUP: "바로 날씨를 확인하는 대응",
            ActionType.WEATHER_UNAVAILABLE: "조회 실패를 안내하는 대응",
            ActionType.EXPLAIN_CAPABILITIES: "기능 설명 대응",
            ActionType.DEESCALATE: "톤을 낮추는 대응",
            ActionType.DIRECT_REPLY: "직접 답변하는 대응",
            ActionType.ASK_CLARIFICATION: "한 번 더 확인하는 대응",
            ActionType.ACKNOWLEDGE: "짧게 받아주는 대응",
            ActionType.ANSWER_IDENTITY: "정체를 설명하는 대응",
            ActionType.CONTINUE_CONVERSATION: "대화를 이어가는 대응",
            ActionType.EXPLAIN_REASON: "이유를 풀어 설명하는 대응",
            ActionType.SHARE_FEELING: "공감 반응",
            ActionType.SHARE_OPINION: "의견을 주는 대응",
            ActionType.GAME_CHAT: "게임 얘기로 이어가는 대응",
            ActionType.GAME_ACCEPT_OR_DECLINE: "제안에 반응하는 대응",
            ActionType.MUSIC_CHAT: "음악 얘기로 이어가는 대응",
            ActionType.RECOMMEND: "추천 중심 대응",
            ActionType.REACT_LAUGH: "같이 웃는 반응",
            ActionType.REACT_SURPRISE: "놀람을 받아주는 반응",
            ActionType.TEASE_BACK: "가볍게 받아치는 대응",
            ActionType.TELL_TIME: "시간 정보를 주는 대응",
            ActionType.SEARCH_ANSWER: "정보를 설명하는 대응",
            ActionType.NEWS_ANSWER: "뉴스 응답",
            ActionType.SMALL_TALK: "짧은 잡담 대응",
        }
        return mapping.get(action, f"`{action.value}` 대응")
