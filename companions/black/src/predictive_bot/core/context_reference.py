from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ContextReferenceProfile:
    domain: str
    schema: str
    intent: str
    sentiment: str
    speech_act: str
    cues: tuple[str, ...]
    slots: dict[str, str]
    evidence: tuple[str, ...]
    state_hint: str
    action_hint: str
    draft_frame: str
    reply_focus: str


_DEICTIC_MARKERS = (
    "그거",
    "그건",
    "그게",
    "그걸",
    "그쪽",
    "그부분",
    "그부븐",
    "그사람",
    "그분",
    "그애",
    "그말",
    "그얘기",
    "그이야기",
    "그일",
    "그때",
    "거기",
    "그곳",
    "아까그거",
    "아까그말",
    "아까그사람",
    "아까그곳",
    "이거",
    "이건",
    "이게",
    "이걸",
    "이부분",
)

_FOLLOWUP_MARKERS = (
    "어때",
    "어떻게",
    "괜찮",
    "나을",
    "맞",
    "별로",
    "답없",
    "이상",
    "좋",
    "힘들",
    "짜증",
    "빡치",
    "킹받",
    "계속",
    "신경",
    "생각",
    "할까",
    "될까",
    "해야",
    "왜",
    "오래가",
    "애매",
    "무섭",
    "아쉽",
    "웃기",
)

_QUESTION_MARKERS = (
    "어때",
    "어떻게",
    "뭐가",
    "뭐로",
    "나을",
    "맞",
    "괜찮",
    "할까",
    "될까",
    "왜",
    "추천",
)

_NEGATIVE_MARKERS = (
    "별로",
    "답없",
    "이상",
    "힘들",
    "짜증",
    "빡치",
    "킹받",
    "무섭",
    "아쉽",
    "오래가",
)

_CHOICE_FIRST_MARKERS = (
    "전자",
    "첫번째",
    "첫째",
    "첫안",
    "첫후보",
    "첫선택지",
    "1번",
    "1안",
    "일번",
    "일안",
    "앞쪽",
    "앞에거",
    "앞에꺼",
    "앞선거",
)

_CHOICE_SECOND_MARKERS = (
    "후자",
    "두번째",
    "둘째",
    "2번",
    "2안",
    "이안",
    "뒤쪽",
    "뒤에거",
    "뒤에꺼",
    "뒤선거",
)

_CHOICE_THIRD_MARKERS = (
    "세번째",
    "세째",
    "셋째",
    "세번째안",
    "세번째후보",
    "세번째선택지",
    "셋째안",
    "셋째후보",
    "3번",
    "3안",
    "삼번",
    "삼안",
    "제3안",
    "제3의선택지",
)

_CHOICE_BOTH_MARKERS = (
    "둘다",
    "둘중",
    "둘중엔",
    "둘중에는",
    "둘모두",
    "둘은",
    "둘이다",
)

_CHOICE_ALL_MARKERS = (
    "셋다",
    "세개다",
    "세가지다",
    "세개전부",
    "세가지전부",
    "전부다",
)

_CHOICE_OTHER_MARKERS = (
    "다른거",
    "다른걸",
    "다른게",
    "다른쪽",
    "다른안",
    "다른방법",
    "다른코스",
    "다른자리",
    "다른후보",
    "다른선택",
    "다른선택지",
)

_CHOICE_NEGATIVE_MARKERS = (
    "싫",
    "별로",
    "별론",
    "답없",
    "끔찍",
    "무리",
    "애매",
    "고문",
    "힘들",
    "안끌",
)

_CHOICE_POSITIVE_MARKERS = (
    "좋",
    "나을",
    "끌려",
    "괜찮",
    "맞",
    "편",
    "덜힘들",
)

_DISCOURSE_CONTRAST_MARKERS = (
    "근데",
    "그런데",
    "다만",
    "근데말야",
    "근데말이야",
)

_DISCOURSE_CONCESSION_MARKERS = (
    "그래도",
    "그치만",
    "하지만",
    "그건맞는데",
    "그렇긴한데",
    "맞긴한데",
    "인정하긴한데",
    "긴한데",
    "좋긴한데",
    "싫진않은데",
    "끌리긴한데",
    "하고싶긴한데",
    "가고싶긴한데",
    "먹고싶긴한데",
    "괜찮긴한데",
    "좋지만",
    "싫진않지만",
    "긴하지만",
)

_DISCOURSE_NEXT_STEP_MARKERS = (
    "그럼",
    "그러면",
    "일단",
    "우선",
    "그럼일단",
    "그러면일단",
)

_DISCOURSE_ALTERNATIVE_MARKERS = (
    "아니면",
    "차라리",
    "그거말고",
    "그보다",
    "대신",
)

_DISCOURSE_AGREEMENT_MARKERS = (
    "맞아",
    "그니까",
    "그러니까",
    "인정",
    "ㅇㅈ",
)

_DISCOURSE_NEGATIVE_MARKERS = (
    "걱정",
    "불안",
    "힘들",
    "무리",
    "애매",
    "별로",
    "찝찝",
    "망설",
    "싫",
    "부담",
    "답없",
    "짜증",
    "킹받",
    "기빨",
)

_DISCOURSE_ACTION_MARKERS = (
    "해야",
    "하자",
    "할게",
    "갈까",
    "먹을까",
    "연락",
    "정리",
    "시작",
    "해볼",
    "일단",
    "우선",
)

_CURRENT_TURN_CONTRAST_MARKERS = (
    "근데",
    "그런데",
    "다만",
    "근데말야",
    "근데말이야",
)

_CURRENT_TURN_CONCESSION_MARKERS = (
    "좋긴한데",
    "싫진않은데",
    "끌리긴한데",
    "하고싶긴한데",
    "가고싶긴한데",
    "먹고싶긴한데",
    "괜찮긴한데",
    "좋지만",
    "싫진않지만",
)

_CURRENT_TURN_ALTERNATIVE_MARKERS = (
    "차라리",
    "그거말고",
    "그보다",
)

_REFERENCE_CATEGORIES: tuple[dict[str, object], ...] = (
    {
        "referent_type": "social",
        "domain": "social_relationship",
        "draft_frame": "contextual_reference_social",
        "reply_focus": "그 사람이나 관계",
        "state_hint": "relationship_context",
        "keywords": (
            "친구",
            "애인",
            "상사",
            "교수",
            "동료",
            "사람",
            "말투",
            "카톡",
            "연락",
            "손절",
            "사과",
            "화해",
            "뒷담",
            "썸",
            "연애",
            "가족",
            "룸메",
        ),
    },
    {
        "referent_type": "place",
        "domain": "place_experience",
        "draft_frame": "contextual_reference_place",
        "reply_focus": "그 장소",
        "state_hint": "place_context",
        "keywords": (
            "카페",
            "식당",
            "미용실",
            "병원",
            "회사",
            "학교",
            "집",
            "숙소",
            "방",
            "전세",
            "월세",
            "이사",
            "역세권",
            "채광",
            "여행",
            "제주",
            "캠핑",
            "장소",
            "매장",
            "가게",
            "팝업스토어",
            "서점",
            "공원",
        ),
    },
    {
        "referent_type": "task",
        "domain": "work_school",
        "draft_frame": "contextual_reference_task",
        "reply_focus": "그 일",
        "state_hint": "task_context",
        "keywords": (
            "업무",
            "과제",
            "발표",
            "시험",
            "프로젝트",
            "코드",
            "버그",
            "회의",
            "공부",
            "출근",
            "마감",
            "피피티",
            "ppt",
            "메일",
            "문구",
            "초안",
            "카피",
            "문장",
            "메시지",
            "글",
        ),
    },
    {
        "referent_type": "media",
        "domain": "media_fandom",
        "draft_frame": "contextual_reference_media",
        "reply_focus": "그 작품이나 취향",
        "state_hint": "media_context",
        "keywords": (
            "영화",
            "드라마",
            "웹툰",
            "애니",
            "노래",
            "게임",
            "캐릭터",
            "굿즈",
            "사진",
            "썸네일",
            "릴스",
            "쇼츠",
            "컷",
            "일러",
            "최애",
            "유튜브",
            "넷플릭스",
            "웹소설",
            "서사",
        ),
    },
    {
        "referent_type": "food",
        "domain": "food_lifestyle",
        "draft_frame": "contextual_reference_food",
        "reply_focus": "그 메뉴",
        "state_hint": "food_context",
        "keywords": (
            "메뉴",
            "점심",
            "저녁",
            "먹",
            "맛",
            "배달",
            "치킨",
            "떡볶",
            "마라탕",
            "국밥",
            "디저트",
            "커피",
            "빵",
            "샐러드",
            "라면",
            "고기",
        ),
    },
    {
        "referent_type": "shopping",
        "domain": "shopping_life",
        "draft_frame": "contextual_reference_item",
        "reply_focus": "그 물건",
        "state_hint": "item_context",
        "keywords": (
            "택배",
            "장바구니",
            "물건",
            "샀",
            "쇼핑",
            "반품",
            "중고",
            "당근",
            "구매",
            "결제",
            "배송",
            "가격",
            "선물",
            "제품",
            "재고",
            "굿즈",
            "거래",
            "판매",
            "올릴",
            "판매글",
        ),
    },
    {
        "referent_type": "emotion",
        "domain": "emotional_state",
        "draft_frame": "contextual_reference_emotion",
        "reply_focus": "그 기분",
        "state_hint": "emotional_context",
        "keywords": (
            "우울",
            "서운",
            "속상",
            "외롭",
            "불안",
            "무기력",
            "고민",
            "현타",
            "기분",
            "마음",
            "멘탈",
            "스트레스",
            "힘들",
        ),
    },
    {
        "referent_type": "body",
        "domain": "health_routine",
        "draft_frame": "contextual_reference_body",
        "reply_focus": "그 컨디션",
        "state_hint": "body_context",
        "keywords": (
            "머리",
            "배",
            "손",
            "발",
            "아프",
            "피곤",
            "졸려",
            "잠",
            "컨디션",
            "감기",
            "소화",
            "몸",
        ),
    },
)


_MIXED_TOPIC_CATEGORIES: tuple[dict[str, object], ...] = (
    {
        "topic": "food",
        "domain": "food_lifestyle",
        "label": "먹는 것",
        "keywords": (
            "배고프",
            "배고픈",
            "배고파",
            "출출",
            "허기",
            "점심",
            "저녁",
            "메뉴",
            "치킨",
            "떡볶",
            "라면",
            "커피",
            "야식",
            "먹고",
            "먹었",
            "먹으",
            "먹을",
            "먹기",
            "먹는",
            "먹자",
        ),
    },
    {
        "topic": "sleep",
        "domain": "health_routine",
        "label": "잠/피로",
        "keywords": (
            "졸려",
            "졸림",
            "잠도",
            "잠이",
            "잠을",
            "잠못",
            "잠안",
            "피곤",
            "밤샘",
            "늦게잤",
            "기절",
        ),
    },
    {
        "topic": "body",
        "domain": "health_routine",
        "label": "몸상태",
        "keywords": (
            "머리아프",
            "배아프",
            "속안좋",
            "목아프",
            "눈뻑",
            "몸살",
            "감기",
            "컨디션",
            "몸도",
            "아프",
        ),
    },
    {
        "topic": "task",
        "domain": "work_school",
        "label": "일/공부",
        "keywords": (
            "발표",
            "과제",
            "시험",
            "업무",
            "회의",
            "마감",
            "프로젝트",
            "피피티",
            "ppt",
            "자료",
            "공부",
            "출근",
            "등교",
            "메일",
            "일도",
        ),
    },
    {
        "topic": "social",
        "domain": "social_relationship",
        "label": "관계",
        "keywords": (
            "친구",
            "애인",
            "상사",
            "동료",
            "교수",
            "카톡",
            "연락",
            "답장",
            "화해",
            "뒷담",
        ),
    },
    {
        "topic": "emotion",
        "domain": "emotional_state",
        "label": "감정",
        "keywords": (
            "걱정",
            "불안",
            "우울",
            "멘탈",
            "현타",
            "서운",
            "짜증",
            "스트레스",
            "무기력",
            "압박",
            "허무",
            "기분",
            "먹먹",
        ),
    },
    {
        "topic": "weather",
        "domain": "weather_daily",
        "label": "날씨",
        "keywords": (
            "날씨",
            "비도",
            "비가",
            "눈도",
            "눈이",
            "춥",
            "덥",
            "더워",
            "추워",
            "습",
            "바람",
        ),
    },
    {
        "topic": "money",
        "domain": "money_life",
        "label": "돈/소비",
        "keywords": (
            "돈",
            "월급",
            "카드값",
            "통장",
            "지갑",
            "결제",
            "택배",
            "장바구니",
            "쇼핑",
            "구독",
            "배송",
        ),
    },
    {
        "topic": "media",
        "domain": "media_fandom",
        "label": "콘텐츠/덕질",
        "keywords": (
            "게임",
            "유튜브",
            "쇼츠",
            "웹툰",
            "최애",
            "굿즈",
            "캐릭터",
            "노래",
            "영화",
            "덕질",
        ),
    },
)

_MIXED_TOPIC_CONNECTORS = (
    "는데",
    "은데",
    "ㄴ데",
    "면서",
    "게다가",
    "그리고",
    "근데",
    "때문에",
    "라서",
    "까지",
    "하고",
    "랑",
    "도",
    ",",
)


def infer_mixed_topic_profile(
    text: str,
    *,
    is_question: bool = False,
) -> ContextReferenceProfile | None:
    raw = str(text or "")
    compact = _normalize(raw)
    if not compact or len(compact) < 10:
        return None
    if is_question or any(marker in compact for marker in _QUESTION_MARKERS):
        return None

    hits = _mixed_topic_hits(compact)
    if len(hits) < 2:
        return None
    if not _has_mixed_topic_connector(raw, compact) and len(hits) < 3:
        return None

    topics = tuple(str(hit["topic"]) for hit in hits)
    labels = tuple(str(hit["label"]) for hit in hits)
    schema = _mixed_schema_from_topics(topics)
    if schema == "mixed_topic_stack":
        return None
    clauses = _mixed_topic_clauses(raw)
    clause_profiles = _mixed_clause_profiles(clauses)
    priority_topic = _mixed_priority_topic(topics, schema)
    priority_label = _mixed_topic_label(priority_topic)
    action_sequence = _mixed_action_sequence(schema, priority_topic)
    negative = any(topic in topics for topic in ("sleep", "body", "task", "emotion")) or any(
        marker in compact for marker in ("힘들", "걱정", "불안", "짜증", "귀찮", "허무", "망", "지침")
    )
    asked = is_question or any(marker in compact for marker in _QUESTION_MARKERS)

    self_doubt = (
        negative
        and "emotion" in topics
        and any(marker in compact for marker in ("걸까", "내가", "너무", "집착", "불안"))
    )
    if asked and not self_doubt:
        intent = "smalltalk_opinion"
        speech_act = "ask_opinion"
    elif negative:
        intent = "smalltalk_feeling"
        speech_act = "complain"
    else:
        intent = "smalltalk_generic"
        speech_act = "inform"

    primary_domain = str(hits[0]["domain"])
    slots = {
        "topic_count": str(len(hits)),
        "primary_topic": topics[0],
        "secondary_topic": topics[1],
        "topic_labels": ",".join(topics),
        "topic_names": " / ".join(labels),
        "primary_label": labels[0],
        "secondary_label": labels[1],
        "priority_topic": priority_topic,
        "priority_label": priority_label,
        "action_sequence": action_sequence,
    }
    slots.update(_mixed_clause_slots(clause_profiles))
    return ContextReferenceProfile(
        domain="mixed_daily_context" if len(hits) > 1 else primary_domain,
        schema=schema,
        intent=intent,
        sentiment="negative" if negative else "neutral",
        speech_act=speech_act,
        cues=("mixed_topic_current_turn", schema, *(f"topic:{topic}" for topic in topics)),
        slots=slots,
        evidence=(
            "mixed_topic_current_turn",
            *(f"topic:{hit['topic']}:{hit['keyword']}" for hit in hits),
            *(f"clause:{item['topics']}" for item in clause_profiles),
            f"priority:{priority_topic}",
            f"action_sequence:{action_sequence}",
        ),
        state_hint="multi_topic_stack",
        action_hint=action_sequence,
        draft_frame=schema,
        reply_focus=" / ".join(labels[:3]),
    )


def infer_contextual_choice_profile(
    text: str,
    recent_texts: Iterable[str],
    *,
    is_question: bool = False,
) -> ContextReferenceProfile | None:
    raw = str(text or "")
    compact = _normalize(raw)
    choice_slot = _choice_slot_from_compact(compact)
    if not compact or choice_slot is None:
        return None

    recent = [str(item or "") for item in recent_texts if str(item or "").strip()]
    if not recent:
        return None
    options = _extract_recent_choice_options(recent)
    if options is None:
        return None
    if choice_slot == "third" and len(options) < 3:
        return None

    category = _best_recent_category(compact, recent)
    referent_type = str(category["referent_type"]) if category is not None else "choice"
    domain = str(category["domain"]) if category is not None else "choice_preference"
    state_hint = str(category["state_hint"]) if category is not None else "choice_context"

    asked = is_question or _looks_like_choice_question(compact)
    negative = any(marker in compact for marker in _CHOICE_NEGATIVE_MARKERS)
    positive = any(marker in compact for marker in _CHOICE_POSITIVE_MARKERS)
    both_reject = any(
        marker in compact
        for marker in ("둘다별로", "둘다별론", "둘다답없", "둘다싫", "둘다끔찍", "둘다애매")
    )
    all_reject = any(
        marker in compact
        for marker in ("셋다별로", "셋다별론", "셋다답없", "셋다싫", "셋다끔찍", "셋다애매")
    )
    if choice_slot == "both" and negative and both_reject:
        choice_slot = "neither"
    if choice_slot == "all" and negative and all_reject and "아니" not in compact:
        choice_slot = "neither"

    if choice_slot in ("both", "all", "neither", "other"):
        schema = "contextual_choice_both_or_neither"
        draft_frame = f"contextual_choice_{choice_slot}"
    elif asked:
        schema = "contextual_choice_opinion"
        draft_frame = f"contextual_choice_{choice_slot}"
    else:
        schema = "contextual_choice_reference"
        draft_frame = f"contextual_choice_{choice_slot}"

    selected_option = ""
    if choice_slot == "first":
        selected_option = options[0]
    elif choice_slot == "second":
        selected_option = options[1]
    elif choice_slot == "third" and len(options) >= 3:
        selected_option = options[2]
    elif choice_slot == "both":
        selected_option = f"{options[0]} / {options[1]}"
    elif choice_slot == "all":
        selected_option = " / ".join(options[:3])

    sentiment = "negative" if negative else "positive" if positive else "neutral"
    return ContextReferenceProfile(
        domain=domain,
        schema=schema,
        intent="smalltalk_opinion",
        sentiment=sentiment,
        speech_act="ask_opinion" if asked else "choose",
        cues=("contextual_choice_reference", f"choice_slot:{choice_slot}", schema),
        slots={
            "referent_type": referent_type,
            "choice_slot": choice_slot,
            "first_option": options[0],
            "second_option": options[1],
            "third_option": options[2] if len(options) >= 3 else "",
            "selected_option": selected_option,
            "topic": "앞선 선택지",
        },
        evidence=(
            "choice_followup",
            f"choice_slot:{choice_slot}",
            f"first_option:{options[0]}",
            f"second_option:{options[1]}",
            *( (f"third_option:{options[2]}",) if len(options) >= 3 else () ),
        ),
        state_hint=state_hint,
        action_hint="share_opinion",
        draft_frame=draft_frame,
        reply_focus=selected_option or "앞선 선택지",
    )


def infer_contextual_discourse_profile(
    text: str,
    recent_texts: Iterable[str],
    *,
    is_question: bool = False,
) -> ContextReferenceProfile | None:
    raw = str(text or "")
    compact = _normalize(raw)
    if not compact:
        return None

    relation = _discourse_relation_from_compact(compact)
    if relation is None:
        return None

    recent = [str(item or "") for item in recent_texts if str(item or "").strip()]
    if not recent:
        return None

    category = _best_recent_category(compact, recent)
    if category is None:
        return None

    asked = is_question or any(marker in compact for marker in _QUESTION_MARKERS)
    negative = any(marker in compact for marker in _DISCOURSE_NEGATIVE_MARKERS)
    action_like = any(marker in compact for marker in _DISCOURSE_ACTION_MARKERS)
    referent_type = str(category["referent_type"])
    schema = f"contextual_discourse_{relation}"

    if relation == "next_step" or action_like:
        speech_act = "ask_opinion" if asked else "plan"
        action_hint = "suggest_next_step"
    elif relation == "alternative":
        speech_act = "ask_opinion" if asked else "suggest"
        action_hint = "offer_alternative"
    elif relation == "agreement":
        speech_act = "agree"
        action_hint = "agree_and_extend"
    else:
        speech_act = "complain" if negative else "inform"
        action_hint = "validate_and_adjust"

    if asked:
        intent = "smalltalk_opinion"
    elif negative:
        intent = "smalltalk_feeling"
    elif relation in ("next_step", "alternative", "agreement"):
        intent = "smalltalk_opinion"
    else:
        intent = "smalltalk_generic"

    sentiment = "negative" if negative else "positive" if relation == "agreement" else "neutral"
    return ContextReferenceProfile(
        domain=str(category["domain"]),
        schema=schema,
        intent=intent,
        sentiment=sentiment,
        speech_act=speech_act,
        cues=("contextual_discourse", f"discourse_relation:{relation}", schema),
        slots={
            "referent_type": referent_type,
            "topic": str(category["reply_focus"]),
            "discourse_relation": relation,
        },
        evidence=(
            "discourse_followup",
            f"discourse_relation:{relation}",
            f"referent_type:{referent_type}",
            *_matched_recent_keywords(category, recent),
        ),
        state_hint=str(category["state_hint"]),
        action_hint=action_hint,
        draft_frame=schema,
        reply_focus=str(category["reply_focus"]),
    )


def infer_current_turn_discourse_profile(
    text: str,
    *,
    is_question: bool = False,
) -> ContextReferenceProfile | None:
    raw = str(text or "")
    compact = _normalize(raw)
    if len(compact) < 12:
        return None

    relation = _current_turn_discourse_relation_from_compact(compact)
    if relation is None:
        return None
    if relation in {"agreement", "next_step"} and not is_question:
        return None

    category = _best_current_category(compact)
    if category is None:
        return None

    asked = is_question or any(marker in compact for marker in _QUESTION_MARKERS)
    negative = any(marker in compact for marker in _DISCOURSE_NEGATIVE_MARKERS)
    action_like = any(marker in compact for marker in _DISCOURSE_ACTION_MARKERS)
    if relation in {"contrast", "concession"} and not negative:
        return None

    if relation == "alternative":
        speech_act = "ask_opinion" if asked else "suggest"
        action_hint = "offer_alternative"
    elif relation == "next_step" or action_like:
        speech_act = "ask_opinion" if asked else "plan"
        action_hint = "suggest_next_step"
    else:
        speech_act = "complain" if negative else "inform"
        action_hint = "validate_and_adjust"

    intent = "smalltalk_opinion" if asked else "smalltalk_feeling" if negative else "smalltalk_generic"
    referent_type = str(category["referent_type"])
    schema = f"current_turn_discourse_{relation}"
    return ContextReferenceProfile(
        domain=str(category["domain"]),
        schema=schema,
        intent=intent,
        sentiment="negative" if negative else "neutral",
        speech_act=speech_act,
        cues=("current_turn_discourse", f"discourse_relation:{relation}", schema),
        slots={
            "referent_type": referent_type,
            "topic": str(category["reply_focus"]),
            "discourse_relation": relation,
        },
        evidence=(
            "current_turn_discourse",
            f"discourse_relation:{relation}",
            f"referent_type:{referent_type}",
            *_matched_current_keywords(category, compact),
        ),
        state_hint=str(category["state_hint"]),
        action_hint=action_hint,
        draft_frame=schema,
        reply_focus=str(category["reply_focus"]),
    )


def infer_contextual_reference_profile(
    text: str,
    recent_texts: Iterable[str],
    *,
    is_question: bool = False,
) -> ContextReferenceProfile | None:
    raw = str(text or "")
    compact = _normalize(raw)
    if not compact or not _looks_like_deictic_followup(compact):
        return None

    recent = [str(item or "") for item in recent_texts if str(item or "").strip()]
    if not recent:
        return None

    category = _best_recent_category(compact, recent)
    if category is None:
        return None

    asked = is_question or any(marker in compact for marker in _QUESTION_MARKERS)
    negative = any(marker in compact for marker in _NEGATIVE_MARKERS)
    schema = "contextual_reference_opinion" if asked else "contextual_reference_feeling" if negative else "contextual_reference_followup"
    intent = "smalltalk_opinion" if asked else "smalltalk_feeling" if negative else "smalltalk_generic"
    speech_act = "ask_opinion" if asked else "complain" if negative else "inform"
    sentiment = "negative" if negative else "neutral"
    referent_type = str(category["referent_type"])
    draft_frame = str(category["draft_frame"])
    return ContextReferenceProfile(
        domain=str(category["domain"]),
        schema=schema,
        intent=intent,
        sentiment=sentiment,
        speech_act=speech_act,
        cues=("contextual_reference", f"contextual_reference_{referent_type}", schema),
        slots={
            "referent_type": referent_type,
            "topic": str(category["reply_focus"]),
        },
        evidence=(
            "deictic_followup",
            f"referent_type:{referent_type}",
            *_matched_recent_keywords(category, recent),
        ),
        state_hint=str(category["state_hint"]),
        action_hint="share_opinion" if asked else "share_feeling",
        draft_frame=draft_frame,
        reply_focus=str(category["reply_focus"]),
    )


def _looks_like_deictic_followup(compact: str) -> bool:
    if not any(marker in compact for marker in _DEICTIC_MARKERS):
        return False
    if len(compact) <= 34:
        return True
    return any(marker in compact for marker in _FOLLOWUP_MARKERS)


def _discourse_relation_from_compact(compact: str) -> str | None:
    if not _looks_like_discourse_followup(compact):
        return None
    if _has_discourse_marker(compact, _DISCOURSE_ALTERNATIVE_MARKERS):
        return "alternative"
    if _has_discourse_marker(compact, _DISCOURSE_CONCESSION_MARKERS):
        return "concession"
    if _has_discourse_marker(compact, _DISCOURSE_NEXT_STEP_MARKERS):
        return "next_step"
    if _has_discourse_marker(compact, _DISCOURSE_CONTRAST_MARKERS):
        return "contrast"
    if _has_discourse_marker(compact, _DISCOURSE_AGREEMENT_MARKERS):
        return "agreement"
    return None


def _current_turn_discourse_relation_from_compact(compact: str) -> str | None:
    if any(marker in compact for marker in _CURRENT_TURN_ALTERNATIVE_MARKERS):
        return "alternative"
    if any(marker in compact for marker in _CURRENT_TURN_CONCESSION_MARKERS):
        return "concession"
    if any(marker in compact for marker in _DISCOURSE_NEXT_STEP_MARKERS):
        return "next_step"
    if any(marker in compact for marker in _CURRENT_TURN_CONTRAST_MARKERS):
        return "contrast"
    return None


def _looks_like_discourse_followup(compact: str) -> bool:
    return any(
        _has_discourse_marker(compact, markers)
        for markers in (
            _DISCOURSE_ALTERNATIVE_MARKERS,
            _DISCOURSE_CONCESSION_MARKERS,
            _DISCOURSE_NEXT_STEP_MARKERS,
            _DISCOURSE_CONTRAST_MARKERS,
            _DISCOURSE_AGREEMENT_MARKERS,
        )
    )


def _has_discourse_marker(compact: str, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        if compact.startswith(marker):
            return True
        index = compact.find(marker)
        if 0 < index <= 8:
            return True
    return False


def _choice_slot_from_compact(compact: str) -> str | None:
    if any(marker in compact for marker in _CHOICE_OTHER_MARKERS):
        return "other"
    if any(marker in compact for marker in _CHOICE_THIRD_MARKERS):
        return "third"
    latin_slot = _latin_choice_slot_from_compact(compact)
    if latin_slot is not None:
        return latin_slot
    first_index = _first_marker_index(compact, _CHOICE_FIRST_MARKERS)
    second_index = _first_marker_index(compact, _CHOICE_SECOND_MARKERS)
    if first_index is not None or second_index is not None:
        if first_index is None:
            return "second"
        if second_index is None:
            return "first"
        return "second" if second_index > first_index else "first"
    if _looks_like_all_choice_slot(compact):
        return "all"
    if any(marker in compact for marker in _CHOICE_BOTH_MARKERS):
        return "both"
    return None


def _looks_like_all_choice_slot(compact: str) -> bool:
    if any(marker in compact for marker in _CHOICE_ALL_MARKERS):
        return True
    return compact.startswith(("다괜찮", "다가능"))


def _latin_choice_slot_from_compact(compact: str) -> str | None:
    if len(compact) > 14:
        return None
    if compact.startswith("c"):
        return "third"
    if re.match(r"^b(?:안|컷|업체|좌석|플랜|버전|색|코스|문장|로|가|는|은|쪽|선택|먹|괜찮|갈까|으로)", compact):
        return "second"
    if re.match(r"^a(?:안|컷|업체|좌석|플랜|버전|색|코스|문장|로|가|는|은|쪽|선택|먹|괜찮|갈까|으로)", compact):
        return "first"
    return None


def _first_marker_index(compact: str, markers: tuple[str, ...]) -> int | None:
    indexes = [compact.find(marker) for marker in markers if marker in compact]
    if not indexes:
        return None
    return min(indexes)


def _looks_like_choice_question(compact: str) -> bool:
    return any(
        marker in compact
        for marker in (
            "어때",
            "어떨까",
            "뭐가",
            "뭐로",
            "나을까",
            "맞나",
            "맞아",
            "괜찮을까",
            "할까",
            "될까",
            "추천",
        )
    )


def extract_recent_choice_options(recent_texts: Iterable[str]) -> tuple[str, ...] | None:
    return _extract_recent_choice_options([str(item or "") for item in recent_texts])


def _extract_recent_choice_options(recent_texts: list[str]) -> tuple[str, ...] | None:
    for raw in reversed(recent_texts):
        options = _extract_choice_options_from_text(str(raw or ""))
        if options is not None:
            return options
    return None


def _extract_choice_options_from_text(raw: str) -> tuple[str, ...] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    first_label = (
        r"(?:첫\s*번째\s*선택지|첫\s*번째\s*후보|첫\s*번째\s*안|첫\s*번째|"
        r"첫\s*선택지|첫\s*후보|첫\s*안|첫째|1\s*번|1\s*안|일\s*번|일\s*안)"
    )
    second_label = (
        r"(?:두\s*번째\s*선택지|두\s*번째\s*후보|두\s*번째\s*안|두\s*번째|"
        r"둘째|2\s*번|2\s*안|이\s*번|이\s*안)"
    )
    third_label = (
        r"(?:세\s*번째\s*선택지|세\s*번째\s*후보|세\s*번째\s*안|세\s*번째|"
        r"셋째|3\s*번|3\s*안|삼\s*번|삼\s*안|제\s*3\s*안)"
    )
    latin_label = r"(?:안|컷|업체|좌석|플랜|버전|색|코스|문장)?"
    soft_separator = r"(?:[,/]|이고|하고|고|하며|인데|지만|반면|그리고)?"
    patterns = (
        rf"{first_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,70}}?){soft_separator}\s*{second_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,70}}?){soft_separator}\s*{third_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,80}})",
        rf"(?:^|[\s,./])A\s*{latin_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,70}}?){soft_separator}\s*B\s*{latin_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,70}}?){soft_separator}\s*C\s*{latin_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,80}})",
        rf"{first_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,70}}?){soft_separator}\s*{second_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,80}})",
        rf"(?:^|[\s,./])A\s*{latin_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,70}}?){soft_separator}\s*B\s*{latin_label}\s*(?:은|는|이|가|:|-)?\s*(.{{1,80}})",
        r"(?:하나는|한\s*쪽은|한쪽은)\s*(.{1,70}?)(?:[,/]|이고|하고|고|하며|인데|지만|반면|그리고)?\s*(?:다른\s*하나는|다른\s*쪽은|다른쪽은)\s*(.{1,80})",
        r"(.{1,70}?)\s*(?:vs|VS|v\.s\.?)\s*(.{1,80})",
        r"(.{1,70}?)\s*(?:아니면|또는)\s*(.{1,80})",
        r"(.{1,50}?)(?:이랑|랑|하고|와|과)\s*(.{1,50}?)\s*중",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        first = _clean_choice_option(match.group(1))
        second = _clean_choice_option(match.group(2))
        third = _clean_choice_option(match.group(3)) if len(match.groups()) >= 3 else ""
        if not (_is_valid_choice_option(first) and _is_valid_choice_option(second)):
            continue
        if third:
            if _is_valid_choice_option(third):
                return first, second, third
            continue
        return first, second
    return None


def _clean_choice_option(option: str) -> str:
    cleaned = re.sub(r"^[\s\"'“”‘’\[\](){}:：,]+|[\s\"'“”‘’\[\](){}:：,.?!]+$", "", str(option or ""))
    cleaned = re.sub(r"^(?:오늘|내일|이번|지금|당장)\s*", "", cleaned)
    cleaned = re.sub(r"^(?:점심|저녁|메뉴|선택)(?:은|는)?\s*", "", cleaned)
    cleaned = re.split(
        r"(?:중에|중에서|하나만|둘\s*중|뭐가|뭐로|어느|어디|고를|골라|선택|,|/)",
        cleaned,
        maxsplit=1,
    )[0].strip()
    words = cleaned.split()
    prefix_words = {
        "오늘",
        "내일",
        "이번",
        "지금",
        "당장",
        "점심",
        "저녁",
        "메뉴",
        "선택",
        "나는",
        "난",
        "나",
        "우리",
        "혹시",
        "평생",
        "하루",
        "동안",
        "일주일",
        "한달",
        "한",
        "달",
        "친구랑",
        "애인이랑",
    }
    while len(words) > 1 and words[0] in prefix_words:
        words.pop(0)
    cleaned = " ".join(words) if words else cleaned
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:32]


def _is_valid_choice_option(option: str) -> bool:
    compact = _normalize(option)
    if len(compact) < 1 or len(compact) > 36:
        return False
    return not any(marker in compact for marker in ("어때", "추천", "뭐가나", "할까"))


def _best_recent_category(compact: str, recent_texts: list[str]) -> dict[str, object] | None:
    preferences = _deictic_preferences(compact)
    best: tuple[int, int, int, dict[str, object]] | None = None
    for recency, raw in enumerate(reversed(recent_texts)):
        recent_compact = _normalize(raw)
        for index, category in enumerate(_REFERENCE_CATEGORIES):
            keywords = tuple(str(item) for item in category["keywords"])
            hits = [keyword for keyword in keywords if _normalize(keyword) in recent_compact]
            if not hits:
                continue
            referent_type = str(category["referent_type"])
            preference_bonus = 4 if referent_type in preferences else 0
            score = len(hits) + preference_bonus
            candidate = (score, -recency, -index, category)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return None
    return best[3]


def _best_current_category(compact: str) -> dict[str, object] | None:
    best: tuple[int, int, dict[str, object]] | None = None
    for index, category in enumerate(_REFERENCE_CATEGORIES):
        keywords = tuple(str(item) for item in category["keywords"])
        hits = [keyword for keyword in keywords if _normalize(keyword) in compact]
        if not hits:
            continue
        score = sum(_current_category_keyword_score(keyword) for keyword in hits)
        candidate = (score, -index, category)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    return best[2]


def _current_category_keyword_score(keyword: str) -> int:
    if _normalize(keyword) in {"사람", "좋", "말투", "생각"}:
        return 1
    return 2


def _deictic_preferences(compact: str) -> tuple[str, ...]:
    if any(marker in compact for marker in ("거기", "그곳", "아까그곳")):
        return ("place",)
    if any(marker in compact for marker in ("그사람", "그분", "그애", "아까그사람")):
        return ("social",)
    if any(marker in compact for marker in ("그말", "그얘기", "그이야기", "아까그말")):
        return ("social", "emotion", "media")
    if "그때" in compact:
        return ("emotion", "task", "place")
    return ()


def _matched_recent_keywords(category: dict[str, object], recent_texts: list[str]) -> tuple[str, ...]:
    compact_recent = " ".join(_normalize(item) for item in recent_texts[-3:])
    hits: list[str] = []
    for keyword in tuple(str(item) for item in category["keywords"]):
        if _normalize(keyword) in compact_recent:
            hits.append(f"context:{keyword}")
        if len(hits) >= 4:
            break
    return tuple(hits)


def _matched_current_keywords(category: dict[str, object], compact: str) -> tuple[str, ...]:
    hits: list[str] = []
    for keyword in tuple(str(item) for item in category["keywords"]):
        if _normalize(keyword) in compact:
            hits.append(f"current:{keyword}")
        if len(hits) >= 4:
            break
    return tuple(hits)


def _mixed_topic_hits(compact: str) -> tuple[dict[str, str], ...]:
    hits: list[dict[str, str]] = []
    for category in _MIXED_TOPIC_CATEGORIES:
        topic = str(category["topic"])
        for keyword in tuple(str(item) for item in category["keywords"]):
            normalized_keyword = _normalize(keyword)
            if normalized_keyword and normalized_keyword in compact:
                hits.append(
                    {
                        "topic": topic,
                        "domain": str(category["domain"]),
                        "label": str(category["label"]),
                        "keyword": keyword,
                    }
                )
                break
    return tuple(hits)


def _has_mixed_topic_connector(raw: str, compact: str) -> bool:
    if any(marker in raw for marker in (",", "…", "...")):
        return True
    return any(_normalize(marker) in compact for marker in _MIXED_TOPIC_CONNECTORS)


def _mixed_topic_clauses(raw: str) -> tuple[str, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    parts = re.split(
        r"(?:,|\.|!|\?|…|\.{2,}|그리고|게다가|근데|때문에|라서|면서|는데|은데|한데|ㄴ데|(?<=[가-힣])데\s+|(?<=[가-힣])고\s+|\s+고\s+)",
        text,
    )
    clauses = tuple(part.strip(" \t\r\n\"'“”‘’") for part in parts if _normalize(part))
    if len(clauses) <= 1:
        return (text,)
    return clauses[:5]


def _mixed_clause_profiles(clauses: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    profiles: list[dict[str, str]] = []
    for clause in clauses:
        hits = _mixed_topic_hits(_normalize(clause))
        if not hits:
            continue
        topics = tuple(dict.fromkeys(str(hit["topic"]) for hit in hits))
        labels = tuple(_mixed_topic_label(topic) for topic in topics)
        profiles.append(
            {
                "text": clause,
                "topics": ",".join(topics),
                "labels": "+".join(labels),
            }
        )
    return tuple(profiles[:5])


def _mixed_clause_slots(clause_profiles: tuple[dict[str, str], ...]) -> dict[str, str]:
    if not clause_profiles:
        return {"clause_count": "0", "clause_summary": ""}
    slots = {
        "clause_count": str(len(clause_profiles)),
        "clause_summary": " > ".join(item["labels"] for item in clause_profiles),
    }
    for index, item in enumerate(clause_profiles[:3], start=1):
        slots[f"clause_{index}_text"] = item["text"]
        slots[f"clause_{index}_topics"] = item["topics"]
        slots[f"clause_{index}_labels"] = item["labels"]
    return slots


def _mixed_topic_label(topic: str) -> str:
    label_map = {
        "food": "먹는 것",
        "sleep": "잠/피로",
        "body": "몸상태",
        "task": "일/공부",
        "social": "관계",
        "emotion": "감정",
        "weather": "날씨",
        "money": "돈/소비",
        "media": "콘텐츠/덕질",
    }
    return label_map.get(str(topic or ""), str(topic or "여러 신호"))


def _mixed_priority_topic(topics: tuple[str, ...], schema: str) -> str:
    topic_set = set(topics)
    if schema == "mixed_hunger_sleep_task":
        return "task"
    if schema in {"mixed_emotion_task", "mixed_social_emotion", "mixed_money_emotion", "mixed_media_emotion"}:
        return "emotion"
    if schema in {"mixed_body_food", "mixed_weather_body"}:
        return "body" if "body" in topic_set else "sleep"
    for topic in ("emotion", "body", "sleep", "task", "social", "food", "money", "weather", "media"):
        if topic in topic_set:
            return topic
    return topics[0] if topics else "mixed"


def _mixed_action_sequence(schema: str, priority_topic: str) -> str:
    if schema == "mixed_hunger_sleep_task":
        return "eat_rest_then_small_task"
    if schema == "mixed_emotion_task":
        return "name_emotion_then_one_task"
    if schema == "mixed_social_emotion":
        return "name_social_trigger_then_boundary"
    if schema == "mixed_body_food":
        return "body_first_food_soft"
    if schema == "mixed_weather_body":
        return "lower_body_load"
    if schema == "mixed_money_emotion":
        return "separate_money_from_mood"
    if schema == "mixed_media_emotion":
        return "enjoy_and_name_attachment"
    if priority_topic in {"body", "sleep"}:
        return "stabilize_body_first"
    if priority_topic == "emotion":
        return "name_loudest_feeling_first"
    return "pick_loudest_signal_first"


def _mixed_schema_from_topics(topics: tuple[str, ...]) -> str:
    topic_set = set(topics)
    if {"food", "sleep", "task"}.issubset(topic_set):
        return "mixed_hunger_sleep_task"
    if "money" in topic_set and "emotion" in topic_set:
        return "mixed_money_emotion"
    if "media" in topic_set and "emotion" in topic_set:
        return "mixed_media_emotion"
    if "emotion" in topic_set and "task" in topic_set:
        return "mixed_emotion_task"
    if "social" in topic_set and "emotion" in topic_set:
        return "mixed_social_emotion"
    if "food" in topic_set and ("body" in topic_set or "sleep" in topic_set):
        return "mixed_body_food"
    if "weather" in topic_set and ("body" in topic_set or "sleep" in topic_set):
        return "mixed_weather_body"
    return "mixed_topic_stack"


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()
