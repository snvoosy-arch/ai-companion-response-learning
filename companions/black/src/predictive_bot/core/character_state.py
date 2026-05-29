from __future__ import annotations

import re

from predictive_bot.core.models import (
    ActionType,
    CharacterState,
    EvidencePacket,
    Intent,
    MeaningPacket,
    MessageFeatures,
    StateAction,
    StateDelta,
    WorldState,
)


_NEGATIVE_MARKERS = (
    "힘들",
    "피곤",
    "우울",
    "서운",
    "불안",
    "짜증",
    "화나",
    "스트레스",
    "망했",
)
_PLAYFUL_MARKERS = ("ㅋㅋ", "ㅎㅎ", "웃기", "장난", "농담", "놀자", "하자")
_ANIMAL_TERMS = (
    "동물원",
    "사파리",
    "아쿠아리움",
    "강아지",
    "고양이",
    "호랑이",
    "사자",
    "판다",
    "다람쥐",
    "청설모",
    "독수리",
    "매",
    "거미",
    "햄스터",
    "돌고래",
    "상어",
    "수달",
    "펭귄",
    "물개",
)
_FOOD_TERMS = (
    "점심",
    "저녁",
    "야식",
    "치킨",
    "라면",
    "떡볶이",
    "삼겹살",
    "소고기",
    "카페",
    "커피",
    "디저트",
    "붕어빵",
    "탕수육",
    "민트초코",
)
_SKY_WEATHER_TERMS = (
    "하늘",
    "노을",
    "구름",
    "별",
    "달",
    "무지개",
    "일출",
    "일몰",
    "비",
    "눈",
    "햇살",
    "오로라",
)
_WORK_SCHOOL_TERMS = (
    "출근",
    "퇴근",
    "등교",
    "하교",
    "회사",
    "학교",
    "상사",
    "선생님",
    "회의",
    "조별과제",
    "월급",
    "공부",
)
_RELATIONSHIP_TERMS = (
    "연애",
    "소개팅",
    "썸",
    "애인",
    "남사친",
    "여사친",
    "데이트",
    "질투",
    "결혼",
    "이상형",
)
_ACTIVITY_TERMS = (
    "바다",
    "계곡",
    "캠핑",
    "등산",
    "산책",
    "수영",
    "물놀이",
    "여행",
    "운동",
    "자전거",
    "피크닉",
)
_DOMAIN_LEXICON = {
    "animal_place": _ANIMAL_TERMS,
    "food": _FOOD_TERMS,
    "sky_weather_feeling": _SKY_WEATHER_TERMS,
    "work_school": _WORK_SCHOOL_TERMS,
    "relationship": _RELATIONSHIP_TERMS,
    "activity": _ACTIVITY_TERMS,
}
_CONCRETE_QUESTION_MARKERS = (
    "있던가",
    "있나",
    "있어",
    "어때",
    "좋아",
    "싫어",
    "고를래",
    "뭐가",
    "어느",
    "어떤",
    "이유",
    "생각",
)
_TOPIC_STOPWORDS = {
    "오늘",
    "요즘",
    "진짜",
    "그냥",
    "너는",
    "나는",
    "내가",
    "혹시",
    "갑자기",
    "어떻게",
    "뭐",
    "무슨",
    "어떤",
    "있어",
    "좋아",
}


def build_evidence_packet(
    *,
    features: MessageFeatures,
    world_state: WorldState | None = None,
) -> EvidencePacket:
    meaning = features.meaning_packet
    domain_scores = _axis_scores(meaning, "domain", fallback=meaning.domain if meaning else None)
    schema_scores = _axis_scores(meaning, "schema", fallback=features.question_schema or (meaning.schema if meaning else None))
    speech_act_scores = _axis_scores(meaning, "speech_act", fallback=features.speech_act)
    coarse_scores = _axis_scores(meaning, "coarse_intent", fallback=features.intent.value)
    emotion_scores = _axis_scores(meaning, "emotion", fallback=None)
    state_hint_scores = _axis_scores(meaning, "state_hint", fallback=None)
    action_hint_scores = _axis_scores(meaning, "action_hint", fallback=None)
    draft_frame_family_scores = _axis_scores(meaning, "draft_frame_family", fallback=None)
    draft_frame_scores = _axis_scores(meaning, "draft_frame", fallback=None)
    tone_scores = _axis_scores(meaning, "tone", fallback=None)
    followup_policy_scores = _axis_scores(meaning, "followup_policy", fallback=None)
    lexical = _lexical_evidence(features.normalized, is_question=features.is_question)
    _merge_scores(domain_scores, lexical["domain_scores"])
    if not schema_scores or max(schema_scores.values()) < 0.5:
        _merge_scores(schema_scores, lexical["schema_scores"])
    _merge_scores(speech_act_scores, lexical["speech_act_scores"])
    if features.intent == Intent.UNKNOWN or not coarse_scores or max(coarse_scores.values()) < 0.6:
        _merge_scores(coarse_scores, lexical["coarse_scores"])

    topics = _dedupe_preserve_order(
        [
            *(world_state.active_grounding_topics if world_state else []),
            *(features.meaning_packet.slots.values() if features.meaning_packet else []),
            *lexical["topics"],
            *(features.topic_hint or "", features.news_topic or ""),
            *_lexical_topics(features.normalized),
        ]
    )
    slots = {**lexical["slots"], **dict(meaning.slots if meaning else {})}
    slots.setdefault("_raw_text", features.normalized)
    tone = _infer_tone(features)
    answer_need = _answer_need(features)
    pressure = _pressure(features)
    playfulness = _playfulness(features)
    domain_hint = _top_label(domain_scores) or (meaning.domain if meaning else None)
    schema_hint = _top_label(schema_scores) or features.question_schema or (meaning.schema if meaning else None)
    speech_act_hint = _top_label(speech_act_scores) or features.speech_act or "other"

    sources = ["classifier"]
    if meaning is not None:
        sources.append(meaning.resolver)
        sources.extend(signal.source for signal in meaning.signals if signal.source)
    if lexical["sources"]:
        sources.extend(lexical["sources"])
    if world_state is not None and world_state.active_grounding_topics:
        sources.append("world_state.grounding_topics")

    return EvidencePacket(
        domain_scores=domain_scores,
        schema_scores=schema_scores,
        speech_act_scores=speech_act_scores,
        coarse_scores=coarse_scores,
        emotion_scores=emotion_scores,
        state_hint_scores=state_hint_scores,
        action_hint_scores=action_hint_scores,
        draft_frame_family_scores=draft_frame_family_scores,
        draft_frame_scores=draft_frame_scores,
        tone_scores=tone_scores,
        followup_policy_scores=followup_policy_scores,
        topics=topics[:8],
        slots=slots,
        tone=tone,
        answer_need=answer_need,
        pressure=pressure,
        playfulness=playfulness,
        schema_hint=schema_hint,
        domain_hint=domain_hint,
        speech_act_hint=speech_act_hint,
        sources=list(dict.fromkeys(sources)),
    )


def infer_state_delta(
    *,
    evidence: EvidencePacket,
    state: CharacterState,
) -> StateDelta:
    reasons: list[str] = []
    topic_focus = evidence.topics[0] if evidence.topics else state.topic_focus
    curiosity_delta = 0.0
    engagement_delta = 0.0
    pressure_delta = 0.0
    affinity_delta = 0.0
    energy_delta = 0.0

    if topic_focus and topic_focus != state.topic_focus:
        curiosity_delta += 0.07
        engagement_delta += 0.05
        reasons.append("new_or_refreshed_topic")

    if evidence.answer_need >= 0.7:
        curiosity_delta += 0.05
        engagement_delta += 0.06
        reasons.append("user_is_inviting_an_answer")

    if evidence.pressure >= 0.55:
        pressure_delta += 0.12
        energy_delta -= 0.05
        affinity_delta += 0.02
        reasons.append("user_pressure_or_negative_tone")
    elif evidence.tone in {"casual", "playful"}:
        pressure_delta -= 0.03
        energy_delta += 0.03
        affinity_delta += 0.03
        reasons.append("low_pressure_social_tone")

    if evidence.speech_act_hint in {"invite", "react"} or "activity_invite" in evidence.coarse_scores:
        engagement_delta += 0.08
        affinity_delta += 0.03
        reasons.append("interaction_is_socially_open")

    mood = _next_mood(evidence=evidence, state=state)
    return StateDelta(
        mood=mood,
        energy_delta=energy_delta,
        curiosity_delta=curiosity_delta,
        affinity_delta=affinity_delta,
        pressure_delta=pressure_delta,
        engagement_delta=engagement_delta,
        topic_focus=topic_focus,
        reasons=reasons or ["state_kept_stable"],
    )


def apply_state_delta(state: CharacterState, delta: StateDelta) -> CharacterState:
    topic_focus = delta.topic_focus or state.topic_focus
    recent_topics = list(state.recent_topics)
    if topic_focus:
        recent_topics = [topic for topic in recent_topics if topic != topic_focus]
        recent_topics.append(topic_focus)
        recent_topics = recent_topics[-6:]
    return CharacterState(
        mood=delta.mood or state.mood,
        energy=_clamp(state.energy + delta.energy_delta),
        curiosity=_clamp(state.curiosity + delta.curiosity_delta),
        affinity=_clamp(state.affinity + delta.affinity_delta),
        pressure=_clamp(state.pressure + delta.pressure_delta),
        engagement=_clamp(state.engagement + delta.engagement_delta),
        topic_focus=topic_focus,
        recent_topics=recent_topics,
        recent_actions=list(state.recent_actions[-6:]),
    )


def choose_state_action(
    *,
    evidence: EvidencePacket,
    state: CharacterState,
    delta: StateDelta,
) -> StateAction:
    schema = evidence.schema_hint or ""
    speech_act = evidence.speech_act_hint or "other"
    coarse = _top_label(evidence.coarse_scores) or ""
    action_hint = _top_label(evidence.action_hint_scores) or ""
    state_hint = _top_label(evidence.state_hint_scores) or ""
    draft_frame_family = _top_label(evidence.draft_frame_family_scores) or ""
    draft_frame = _top_label(evidence.draft_frame_scores) or ""

    if _looks_like_explicit_support_request(evidence):
        return StateAction(
            action=ActionType.SHARE_FEELING,
            mode="explicit_support_request",
            score=0.94,
            reason="the user explicitly asks for comfort or encouragement, so emotional support should beat clarification or generic continuation",
            response_hints=["comfort_directly", "avoid_generic_clarification"],
        )

    if _looks_like_direct_roleplay_request(evidence):
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="roleplay_direct_response",
            score=0.93,
            reason="the user asks Black to speak inside a roleplay situation, so answer in that frame directly",
            response_hints=["enter_roleplay_frame", "answer_directly"],
        )

    if _looks_like_direct_output_request(evidence):
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="direct_output_request",
            score=0.94,
            reason="the user asks for a concrete line, joke, phrase, or short text output, so answer directly instead of treating it as a feeling statement",
            response_hints=["produce_requested_line", "avoid_feeling_route"],
        )

    if _looks_like_open_persona_answer_request(evidence):
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="open_persona_answer",
            score=0.88,
            reason="open persona questions ask for Black's own preference or choice, so answer directly before specialist routes",
            response_hints=["answer_directly", "preserve_persona_question"],
        )

    if _looks_like_weather_home_activity_question(evidence):
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="weather_home_activity_answer",
            score=0.93,
            reason="weather is being used as context for a home-activity suggestion, so answer the activity request directly",
            response_hints=["answer_with_activity", "avoid_weather_slot_route"],
        )

    if _looks_like_belief_reason_opinion_question(evidence):
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="belief_reason_opinion",
            score=0.92,
            reason="the user asks for Black's position and reasoning on a belief question, so answer directly instead of asking what 'why' refers to",
            response_hints=["answer_position_with_reason", "avoid_reason_clarification"],
        )

    if (
        action_hint == ActionType.SHARE_OPINION.value
        and evidence.answer_need >= 0.7
        and (
            state_hint in {"playful_affinity", "practical_focus"}
            or draft_frame
            in {
                "direct_choice_with_reason",
                "playful_absurd_answer",
                "playful_secret_complicity",
                "preference_answer_with_reason",
                "practical_direct_advice",
            }
            or draft_frame_family
            in {
                "choice_preference",
                "playful_output",
                "practical_guidance",
                "reflective_position",
                "roleplay_output",
                "situational_tactic",
            }
            or schema
            in {
                "absurd_hypothetical",
                "hypothetical_choice",
                "preference_disclosure",
                "social_mishap",
                "soft_decision_advice",
            }
        )
    ):
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="planner_share_opinion",
            score=0.93,
            reason="planner heads indicate a direct opinion/choice answer should beat incidental specialist topics",
            response_hints=["answer_directly", "preserve_planner_frame"],
        )

    if coarse in {Intent.WEATHER.value, Intent.TIME_DATE.value, Intent.SEARCH_REQUEST.value, Intent.NEWS.value}:
        return StateAction(
            action=ActionType.CONTINUE_CONVERSATION,
            mode="defer_to_grounded_route",
            score=0.58,
            reason="external-fact turns should be handled by the grounded specialist route",
            response_hints=["do_not_override_grounded_policy"],
        )

    if coarse == Intent.WHY.value or schema == "reason_probe":
        return StateAction(
            action=ActionType.CONTINUE_CONVERSATION,
            mode="defer_to_reason_route",
            score=0.50,
            reason="reason probes should stay on the explanation/clarification route",
            response_hints=["do_not_override_reason_policy"],
        )

    if evidence.pressure >= 0.72:
        return StateAction(
            action=ActionType.SHARE_FEELING,
            mode="comfort",
            score=0.88,
            reason="character pressure rose, so comfort beats schema precision",
            response_hints=["acknowledge_pressure", "keep_low_pressure"],
        )

    if evidence.pressure >= 0.55 and evidence.answer_need < 0.7:
        return StateAction(
            action=ActionType.SHARE_FEELING,
            mode="comfort",
            score=0.84,
            reason="user pressure is the main signal, so comfort beats topic chasing",
            response_hints=["acknowledge_pressure", "do_not_chase_topic"],
        )

    if schema in {
        "preference_disclosure",
        "habit_preference",
        "self_style",
        "hypothetical_choice",
    }:
        if schema == "habit_preference" and speech_act == "inform":
            return StateAction(
                action=ActionType.SHARE_FEELING,
                mode="acknowledge_habit_preference",
                score=0.88,
                reason="user stated a daily habit/preference, so acknowledge their rhythm before chasing topics",
                response_hints=["acknowledge_user_preference", "avoid_stock_continue"],
            )
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="answer_lightly_then_ask_back",
            score=0.88,
            reason="daily preference schemas should beat specialist topic routes like music or food",
            response_hints=["answer_briefly", "preserve_daily_topic"],
        )

    if schema in {
        "body_signal_interpretation",
        "low_energy_support",
        "comfort_request",
    } and speech_act in {"inform", "complain", "react", "other"}:
        return StateAction(
            action=ActionType.SHARE_FEELING,
            mode="body_state_support",
            score=0.90,
            reason="body-state and low-energy statements should be acknowledged without asking for an unrelated clarification",
            response_hints=["reflect_body_state", "avoid_generic_clarification"],
        )

    if coarse == Intent.GAME_INVITE.value or _has_topic(evidence, "game", "게임", "겜"):
        action = ActionType.GAME_ACCEPT_OR_DECLINE if coarse == Intent.GAME_INVITE.value else ActionType.GAME_CHAT
        return StateAction(
            action=action,
            mode="game_accept_or_chat",
            score=0.86 if action == ActionType.GAME_ACCEPT_OR_DECLINE else 0.79,
            reason="game topic should stay on the game specialist path",
            response_hints=["stay_on_game_topic"],
        )

    if _looks_like_playful_direct_reaction_question(evidence):
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="playful_direct_reaction",
            score=0.89,
            reason="playful scenario questions ask for a concrete reaction, so answer the situation before specialist topic routing",
            response_hints=["answer_directly", "preserve_playful_scene"],
        )

    if coarse == Intent.MUSIC.value or _has_topic(evidence, "music", "음악", "노래", "곡"):
        return StateAction(
            action=ActionType.MUSIC_CHAT,
            mode="topic_chat",
            score=0.82,
            reason="music topic should stay on the music specialist path",
            response_hints=["stay_on_music_topic"],
        )

    if coarse == Intent.MEDIA_RECOMMEND.value or _has_topic(evidence, "media", "영화", "드라마", "넷플릭스", "유튜브"):
        return StateAction(
            action=ActionType.RECOMMEND,
            mode="topic_recommend",
            score=0.80,
            reason="media topic should stay on the recommendation path",
            response_hints=["give_one_media_option"],
        )

    if (
        coarse == Intent.ACTIVITY_INVITE.value
        or speech_act == "invite"
        or schema == "activity_invite"
    ):
        return StateAction(
            action=ActionType.ACCEPT_ACTIVITY_INVITE,
            mode="accept_activity_invite",
            score=0.90,
            reason="user opened a shared activity frame",
            response_hints=["accept_lightly", "stay_on_activity_topic"],
        )

    if schema == "expressive_request" and _looks_like_missing_rewrite_target(evidence):
        return StateAction(
            action=ActionType.ASK_CLARIFICATION,
            mode="schema_clarify",
            score=0.94,
            reason="rewrite request names a sentence but does not provide the source text",
            response_hints=["ask_for_source_text", "do_not_rewrite_unknown_text"],
        )

    if schema in {
        "expressive_request",
        "aesthetic_reflection",
        "reflective_observation",
    }:
        return StateAction(
            action=ActionType.CONTINUE_CONVERSATION,
            mode="schema_continue",
            score=0.92,
            reason="schema-specific continue turn should not be converted into an opinion answer",
            response_hints=["preserve_schema", "continue_lightly"],
        )

    if schema in {
        "comparative_reflection",
        "relational_interpretation",
    }:
        return StateAction(
            action=ActionType.SHARE_FEELING,
            mode="schema_feeling",
            score=0.92,
            reason="schema-specific feeling turn should keep the emotional reading",
            response_hints=["reflect_feeling", "avoid_advice"],
        )

    if schema in {
        "activity_recommendation",
        "soft_decision_advice",
        "process_advice",
        "budget_reflection",
        "habit_support",
        "activity_preparation_advice",
        "safety_boundary",
        "honesty_boundary",
        "reflective_judgment",
        "broad_opinion",
    }:
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="soft_recommend",
            score=0.94,
            reason="state layer treats advice-like schemas as opinion guidance, not exact labels",
            response_hints=["give_one_concrete_option", "avoid_schema_terms"],
        )

    if schema in {
        "preference_disclosure",
        "habit_preference",
        "self_style",
    }:
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="answer_lightly_then_ask_back",
            score=0.86,
            reason="state layer treats personal preference schemas as light opinion questions",
            response_hints=["answer_briefly", "optionally_ask_back"],
        )

    if schema == "concrete_topic_question":
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="answer_lightly_then_ask_back",
            score=0.93,
            reason="lexical evidence found a concrete topic even though the main classifier was uncertain",
            response_hints=["answer_briefly", "stay_on_detected_topic"],
        )

    if evidence.domain_hint in {
        "animal_place",
        "food",
        "sky_weather_feeling",
        "work_school",
        "relationship",
        "activity",
    } and evidence.answer_need >= 0.7:
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="domain_grounded_answer",
            score=0.94,
            reason="domain evidence plus a direct question is enough to answer lightly instead of asking for clarification",
            response_hints=["answer_on_detected_domain", "avoid_generic_clarification"],
        )

    if evidence.answer_need >= 0.75:
        return StateAction(
            action=ActionType.SHARE_OPINION,
            mode="answer_lightly_then_ask_back",
            score=0.84,
            reason="question needs a light answer even if schema is uncertain",
            response_hints=["answer_briefly", "optionally_ask_back"],
        )

    if state.pressure >= 0.45 or evidence.tone == "strained":
        return StateAction(
            action=ActionType.SHARE_FEELING,
            mode="comfort",
            score=0.82,
            reason="current character state is carrying pressure",
            response_hints=["soften", "do_not_expand_too_much"],
        )

    return StateAction(
        action=ActionType.CONTINUE_CONVERSATION,
        mode="ask_back",
        score=0.64,
        reason="state is stable and the turn can continue socially",
        response_hints=["carry_topic_focus", "ask_back_if_natural"],
    )


def remember_state_action(state: CharacterState, action: StateAction) -> CharacterState:
    recent_actions = [item for item in state.recent_actions if item != action.mode]
    recent_actions.append(action.mode)
    return CharacterState(
        mood=state.mood,
        energy=state.energy,
        curiosity=state.curiosity,
        affinity=state.affinity,
        pressure=state.pressure,
        engagement=state.engagement,
        topic_focus=state.topic_focus,
        recent_topics=list(state.recent_topics[-6:]),
        recent_actions=recent_actions[-6:],
    )


def _axis_scores(meaning: MeaningPacket | None, axis: str, *, fallback: str | None) -> dict[str, float]:
    scores: dict[str, float] = {}
    if meaning is not None:
        for signal in meaning.signals:
            if signal.axis == axis and signal.label:
                scores[signal.label] = max(scores.get(signal.label, 0.0), float(signal.confidence))
    if fallback:
        scores.setdefault(str(fallback), 0.55 if not scores else 0.01)
    return scores


def _merge_scores(base: dict[str, float], extra: dict[str, float]) -> None:
    for label, score in extra.items():
        base[label] = max(base.get(label, 0.0), score)


def _lexical_evidence(text: str, *, is_question: bool) -> dict[str, object]:
    normalized = str(text or "")
    compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
    domain_scores: dict[str, float] = {}
    topics: list[str] = []
    slots: dict[str, str] = {}
    for domain, terms in _DOMAIN_LEXICON.items():
        hits = _lexicon_hits(normalized, terms)
        if not hits:
            continue
        confidence = min(0.92, 0.70 + (0.06 * len(hits)))
        domain_scores[domain] = confidence
        topics.extend(hits[:4])

    schema_scores: dict[str, float] = {}
    coarse_scores: dict[str, float] = {}
    speech_act_scores: dict[str, float] = {}
    if is_question and topics and any(marker in normalized for marker in _CONCRETE_QUESTION_MARKERS):
        schema_scores["concrete_topic_question"] = 0.78
        coarse_scores[Intent.SMALLTALK_OPINION.value] = 0.72
        speech_act_scores["ask"] = 0.76
        slots["topic"] = "|".join(_dedupe_preserve_order(topics[:4]))

    if is_question and _looks_like_general_activity_question(normalized, compact):
        schema_scores["activity_recommendation"] = max(schema_scores.get("activity_recommendation", 0.0), 0.82)
        coarse_scores[Intent.SMALLTALK_OPINION.value] = max(
            coarse_scores.get(Intent.SMALLTALK_OPINION.value, 0.0),
            0.74,
        )
        speech_act_scores["ask"] = max(speech_act_scores.get("ask", 0.0), 0.78)
        if "활동" not in topics:
            topics.append("활동")
        slots.setdefault("topic", "활동")

    return {
        "domain_scores": domain_scores,
        "schema_scores": schema_scores,
        "speech_act_scores": speech_act_scores,
        "coarse_scores": coarse_scores,
        "topics": _dedupe_preserve_order(topics),
        "slots": slots,
        "sources": ["lexical_evidence"] if domain_scores or schema_scores or speech_act_scores else [],
    }


def _looks_like_general_activity_question(normalized: str, compact: str) -> bool:
    return bool(
        re.search(r"(?:무엇|뭐|뭘|어떤)\s*(?:하(?:고|면서)|하면서|하고서)?\s*(?:놀래|놀까|놀자|놀면|할까|하지|하면)", normalized)
        or re.search(r"(?:뭐하면서|뭐하고).*(?:놀래|놀까|놀자|시간보낼까|보낼까)", compact)
    )


def _lexicon_hits(text: str, terms: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for term in terms:
        if not term:
            continue
        if len(term) <= 1:
            pattern = rf"(?<![0-9A-Za-z가-힣]){re.escape(term)}(?:[은는이가을를도와과]|(?=$|[^0-9A-Za-z가-힣]))"
            if re.search(pattern, text):
                hits.append(term)
            continue
        if term in text:
            hits.append(term)
    return hits


def _top_label(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def _has_topic(evidence: EvidencePacket, *needles: str) -> bool:
    labels = [evidence.domain_hint or "", evidence.schema_hint or "", *evidence.topics]
    lowered = [str(label or "").lower() for label in labels]
    for needle in needles:
        item = needle.lower()
        if item == "곡":
            if any(label == item for label in lowered):
                return True
            continue
        if any(item in label for label in lowered):
            return True
    return False


def _looks_like_missing_rewrite_target(evidence: EvidencePacket) -> bool:
    text = re.sub(
        r"\s+",
        " ",
        str(evidence.slots.get("_raw_text") or " ".join([*evidence.topics, *evidence.slots.values()])),
    ).strip()
    if not text:
        return False
    if not re.search(r"(바꿔|고쳐|수정|다듬)", text):
        return False
    if not re.search(r"(문장|말투|표현|대사)", text):
        return False
    if re.search(r"['\"“”‘’`].+['\"“”‘’`]", text):
        return False
    return bool(re.search(r"(이\s*문장|그\s*문장|문장\s*좀|말투\s*좀|표현\s*좀)", text))


def _looks_like_playful_direct_reaction_question(evidence: EvidencePacket) -> bool:
    text = re.sub(
        r"[^0-9A-Za-z가-힣]+",
        "",
        str(evidence.slots.get("_raw_text") or " ".join([*evidence.topics, *evidence.slots.values()])),
    ).lower()
    if not text:
        return False
    return (
        "어떻게리액션" in text
        or "리액션할래" in text
        or "어떻게반응" in text
        or "반응해줄래" in text
        or ("노래방" in text and "음이탈" in text)
    )


def _raw_evidence_text(evidence: EvidencePacket) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(evidence.slots.get("_raw_text") or " ".join([*evidence.topics, *evidence.slots.values()])),
    ).strip()


def _compact_evidence_text(evidence: EvidencePacket) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", _raw_evidence_text(evidence)).lower()


def _looks_like_weather_home_activity_question(evidence: EvidencePacket) -> bool:
    raw = _raw_evidence_text(evidence)
    compact = _compact_evidence_text(evidence)
    if not raw:
        return False
    if "날씨" not in raw:
        return False
    if not ("집에서" in raw or "이런날씨" in compact):
        return False
    return bool(
        re.search(r"(?:뭐|뭘|무엇).{0,12}(?:하면|할까|하지|좋을까|좋을지)", raw)
        or re.search(r"(?:뭐|뭘|무엇).{0,12}(?:하면|할까|하지|좋을까|좋을지)", compact)
    )


def _looks_like_explicit_support_request(evidence: EvidencePacket) -> bool:
    raw = _raw_evidence_text(evidence)
    compact = _compact_evidence_text(evidence)
    if not raw:
        return False
    if any(marker in compact for marker in ("위로해줘", "위로해", "응원한마디", "응원해줘", "응원해")):
        return True
    return any(
        marker in compact
        for marker in (
            "자신감이떨어",
            "자신감떨어",
            "잘하고있는건지",
            "자꾸의심",
            "실패할까봐",
            "너무두려워",
            "시작을못하겠",
            "너무속상",
            "아끼는물건을잃어",
            "잃어버려서너무속상",
            "너무긴장",
            "긴장돼",
            "상처받는말",
            "상처받은말",
            "마인드컨트롤",
            "털어버리는",
            "마음을다잡",
        )
    )


def _looks_like_direct_roleplay_request(evidence: EvidencePacket) -> bool:
    raw = _raw_evidence_text(evidence)
    compact = _compact_evidence_text(evidence)
    if not raw:
        return False
    return (
        ("상황" in raw or "역할극" in raw or raw.startswith("["))
        and (
            "통화하는것처럼" in compact
            or "관제탑역할" in compact
            or "말걸어줘" in compact
            or "달래줄거야" in compact
            or "반응해봐" in compact
            or "댓글달아봐" in compact
            or "변명해봐" in compact
            or "생떼부려봐" in compact
            or "철벽칠래" in compact
            or "역할을해줘" in compact
            or "풀어봐" in compact
        )
    )


def _looks_like_direct_output_request(evidence: EvidencePacket) -> bool:
    compact = _compact_evidence_text(evidence)
    if not compact:
        return False
    if not any(
        noun in compact
        for noun in (
            "문장",
            "멘트",
            "대사",
            "댓글",
            "개그",
            "아재개그",
            "신조어",
            "밈",
            "한마디",
            "말투",
            "목표",
            "특징",
            "취미",
            "변명",
            "변명거리",
        )
    ):
        return False
    return any(
        verb in compact
        for verb in (
            "만들어봐",
            "만들어줘",
            "써봐",
            "써줘",
            "쳐볼래",
            "해봐",
            "던져봐",
            "달아봐",
            "들려줘",
            "지어줘",
            "말해봐",
            "걸어봐",
            "어필해봐",
            "영업해줘",
        )
    )


def _looks_like_belief_reason_opinion_question(evidence: EvidencePacket) -> bool:
    compact = _compact_evidence_text(evidence)
    if not compact:
        return False
    if not any(marker in compact for marker in ("믿어", "믿는다면", "안믿는다면", "생각해")):
        return False
    return any(marker in compact for marker in ("귀신", "영혼", "운명", "완벽함", "도덕", "자유"))


def _looks_like_open_persona_answer_request(evidence: EvidencePacket) -> bool:
    text = re.sub(
        r"[^0-9A-Za-z가-힣]+",
        "",
        str(evidence.slots.get("_raw_text") or " ".join([*evidence.topics, *evidence.slots.values()])),
    ).lower()
    if not text:
        return False
    markers = (
        "오늘하루중",
        "시간여행",
        "좋아하는계절",
        "영향받은책",
        "영향을받은책",
        "영향받은영화",
        "영향을받은영화",
        "어디로든여행",
        "장래희망",
        "새롭게배운",
        "관심이생긴분야",
        "스트레스를받을때",
        "복권에당첨",
        "한가지음식",
        "표현하는단어",
        "초능력",
        "아침형인간",
        "올빼미형",
        "잘했다고생각하는결정",
        "무인도",
        "좋아하는색깔",
        "존경하는실존인물",
        "역사적인물",
        "인생좌우명",
        "이번주말",
        "반려동물",
        "크게웃",
        "혼자있는시간",
        "이상적인하루",
        "익스트림스포츠",
        "최고의선물",
        "삶을책으로쓴다면",
        "커피와차",
        "두려워하는것",
        "어린시절",
        "오늘밤자기전에",
        "동물과대화",
        "모든전기",
        "인생을영화장르",
        "절대실패하지",
        "마음을읽는능력",
        "투명인간",
        "나만의행성",
        "좀비",
        "위대한발명품",
        "스마트폰없이",
        "미래로가서",
        "외계인이지구에와서",
        "지구의대표음식",
        "악기",
        "복제인간",
        "살이찌지않",
        "다른사람으로살",
        "비가오는날듣고싶",
        "비오는날듣고싶",
        "우주여행",
        "달과화성",
        "새로운과목",
        "똑같은옷",
        "10년전으로돌아",
        "동화속세계",
        "내가만든요리",
        "말한마디도하지않고",
        "인터넷전혀안하기",
        "10년뒤의나에게편지",
        "법을바꿀수있다면",
        "세계일주",
        "잠을자지않아도",
        "새로운언어",
        "10만원을줍",
        "십만원을줍",
        "처음으로내돈",
        "내돈을주고샀던물건",
        "엘리베이터에갇혔",
        "일기장을훔쳐보",
        "보여주기싫은페이지",
        "1억원을기부",
        "일억원을기부",
        "나를가장잘아는친구",
        "한문장으로설명",
        "모든동물이멸종",
        "한종만살릴",
        "최근1년동안내삶",
        "긍정적인변화",
        "눈을감고지금당장",
        "세가지단어",
        "내가가진단점",
        "거짓말을100",
        "알아채는능력",
        "100살까지살수있는알약",
        "백살까지살수있는알약",
        "삶을한편의다큐멘터리",
        "다큐멘터리의제목",
        "단한가지향기",
        "인내심스위치",
        "당신은참좋은사람",
        "내가만든물건중",
        "가장자랑스러운것",
        "외계인이나를납치",
        "지구를그리워하게",
        "어른이되었다고",
        "실감했던순간",
        "용서를받을수있는쿠폰",
        "무조건적인용서",
        "전자기기",
        "천재란",
        "역사책에당신의이름",
        "한줄남는다면",
        "가장피하고싶은주제",
        "실력이늘지않아서포기",
        "도저히실력이늘지않",
        "10년뒤의세상",
        "사라졌으면하는것",
        "정반대의성향",
        "단짝친구가될수",
        "운이좋았다",
        "노래가사한줄",
        "마음을울린적",
        "똑같은생각을하는사람",
        "오늘하루를한가지색깔",
        "양치질안하기",
        "샤워안하기",
        "생각을영상",
        "모든언어를완벽",
        "낡은램프",
        "지니",
        "잠을잘수없는",
        "12시간무조건",
        "충성하는용",
        "동물로언제든변신",
        "백과사전한권",
        "통기타한대",
        "100억이든통장",
        "20살로어려진",
        "모든음식이매운맛",
        "모든음식이단맛",
        "시트콤",
        "같은영화",
        "비밀의방",
        "거짓말을들을때마다",
        "귀에서삐",
        "처음으로교신",
        "지구대표",
        "성공하는식당",
        "유령도시",
        "겨울옷입고여름",
        "여름옷입고겨울",
        "모든기억을가진채",
        "다시태어난다면",
        "데스노트",
        "하늘을나는자동차",
        "순간이동장치",
        "눈물이진주",
        "슬픈영화",
        "연예인과평생친구",
        "연예인과한달",
        "과거의위인",
        "저녁을대접",
        "남에게들리지않는방",
        "음악장르",
        "라면을끓일때마다",
        "계란프라이",
        "명장면",
        "인터넷이영원히사라지는",
        "새로운과일",
        "지구가멸망",
        "10억원이생겼는데",
        "10억이생겼는데",
        "백그라운드음악",
        "bgm",
        "영화의조연",
        "외계인친구",
        "지구음식",
        "48시간",
        "마법의물약",
        "텔레파시",
        "구름을타고",
        "가장귀여운동물",
        "원하는나이",
        "성별이바뀌",
        "스마트폰없이살기",
        "에어컨히터없이",
        "감기달고살기",
        "만성소화불량",
        "고기못먹기",
        "밀가루",
        "원할때투명인간",
        "거짓말할때마다",
        "야외수영장",
        "패딩입고등산",
        "절친의전애인",
        "전애인의절친",
        "음악없이살기",
        "영상없이살기",
        "짝사랑",
        "혼자서세계여행",
        "내방에서만놀기",
        "50확률로100억",
        "100확률로100만원",
        "매력포인트",
        "불리고싶",
        "한계를뛰어넘",
        "삶의원칙",
        "잘맞는친구",
        "사과를할때",
        "인생을3막",
        "남은수명이딱1년",
        "고마움의대상",
        "위로하는",
        "샤워할때",
        "단골노래",
        "카카오톡",
        "유튜브를제외",
        "자주켜는앱",
        "피자가장자리",
        "도우를남기는편",
        "비행기탈때",
        "창가자리",
        "통로자리",
        "냉장고에무조건",
        "혼자밥을먹어야",
        "카페에가면",
        "잠이안올때",
        "수면유도",
        "계절이바뀔때",
        "소확행아이템",
        "유치원생시절",
        "등짝을때려",
        "꽉안아",
        "10년뒤의나를상상",
        "학생시절",
        "짓궂은장난",
        "생생하게기억나는꿈",
        "통장잔고",
        "어릴적에진심으로믿",
        "묘비명",
        "어제하루중",
        "무의미하게보냈다고",
        "5년전에했던고민",
        "50개의질문",
        "음료수",
        "앱중하나로태어난다면",
        "10m이내에접근",
        "자동으로들리는배경음악",
        "사소하지만절대적인매너",
        "수첩에내일의로또번호",
        "로또번호가적혀",
        "영혼이바뀐다면",
        "웃음표현을절대쓸수없",
        "방안에있는물건들뿐",
        "쓸모없는초능력",
        "관찰예능프로그램",
        "직접지은집",
        "흙냄새",
        "갓인쇄된책냄새",
        "주유소기름냄새",
        "허락도없이소스를",
        "초봄하나로고정",
        "늦가을하나로고정",
        "퍼레이드카",
        "공룡시대",
        "서기3000년",
        "칵테일",
        "밤하늘의별",
        "우주의끝",
        "게임속npc",
        "가위를낼수없는",
        "바위를낼수없는",
        "유튜브구독자100만",
        "층수버튼을다누르고",
        "뷔페에가서",
        "무협지의기연",
        "0칼로리인치킨",
        "숙취없는최고급와인",
        "직립보행",
        "문방구앞오락기",
        "뽑기기계",
        "넘어졌는데아무도못본줄",
        "아아아이스아메리카노",
        "뜨아따뜻한아메리카노",
        "은행에찾으러갈때",
        "머릿속을가장강렬하게",
        "일일강사",
        "자신있게가르칠",
        "돈주고도못살경험",
        "게이지바",
        "상태를나타내는",
        "학교소풍",
        "수학여행",
        "장기자랑",
        "방에있는물건들이",
        "스포일러를강제로",
        "줄거리요약본",
        "어깨에기대",
        "번호를물어",
        "동물원의동물",
        "과거의나에게딱세글자",
        "뷔페에갔는데접시",
        "딱3가지",
        "젓가락질을못",
        "숟가락을못",
        "완벽하게부합하는이상형",
        "싫어하는음식",
        "조선시대의노비",
        "조선시대의왕",
        "샤워기물온도",
        "친구들과사진",
        "마법의지우개",
        "흑역사",
        "난방안되는방",
        "모기10마리",
        "템플스테이",
        "로봇청소기",
        "노래방에서마지막1분",
        "서프라이즈파티",
        "애착물건",
        "알람소리",
        "투명망토",
        "쪽지나롤링페이퍼",
        "작은돌멩이",
        "양말이물에젖은",
        "콘서트vip",
        "맨뒷자리",
        "질문을계속받는지금",
        "자는모습을몰래동영상",
        "흑역사가찍혔을확률",
        "유리창이나거울에비친",
        "좋아하는라면조리법",
        "우주의통치자",
        "샤워후거울",
        "흥얼거리는cm송",
        "민트초코에밥",
        "재채기가나올것같다가",
        "엘리베이터에나혼자",
        "내가만약자판기",
        "주운usb",
        "외계인의지구침공계획서",
        "튀겨진야채",
        "배신감",
        "바지지퍼",
        "직성이풀리는루틴",
        "선물을줄때",
        "매일일기를써야하는법",
        "영화속악당",
        "작은틈새",
        "중요한물건을빠뜨려",
        "겨울에아이스크림",
        "뜨거운붕어빵",
        "동물원의사육사",
        "왕자님",
        "내가만약공책",
        "나혼자가서플렉스",
        "모든음식에케첩",
        "모든음식에마요네즈",
        "10원짜리동전",
        "비밀아지트",
        "머리를쓰다듬어",
        "스스로칭찬",
        "초특급슈퍼스타",
        "당신의얼굴을알아보",
        "뮤지컬대사처럼노래",
        "시트콤웃음소리",
        "텔레비전을보며",
        "게임속최종보스",
        "미래에살게될집",
        "설탕대신소금",
        "소금대신설탕",
        "일대기를책으로출판",
        "단하나의거짓말",
        "그림자가나를떠나",
        "63빌딩",
        "두려워하는귀신",
        "장롱면허",
        "페라리를운전",
        "비오는날만있는도시",
        "눈오는날만있는도시",
        "로봇과인간이구분되지않는",
        "접시를딱한번",
        "핸드폰케이스라면",
        "중고로산것처럼흠집",
        "포장지가절대안뜯어지는",
        "이건마법이다",
        "모른척하는거지",
        "왼쪽굽이1cm낮",
        "한쪽다리가3cm짧",
        "지구의춤",
        "연예인의집에몰래",
        "코를파는모습",
        "이상한강박증",
        "결말5분을못보는",
        "첫10분을못보는",
        "지구가멸망한다는뉴스",
        "도로위에있는신호등",
        "나자신만을위해쓸것",
        "도플갱어",
        "무지개색줄무늬티셔츠",
        "눈을한번깜빡이는동안",
        "완전히똑같은옷",
        "상하의신발까지",
        "한부위만로봇부품",
        "로봇부품으로교체",
        "모든과일이사람처럼말",
        "수다스러울것같은과일",
        "주차장에서욕을하며화내",
        "이마에자막",
        "조선시대말투",
        "하오체",
        "합쇼체",
        "비욘세",
        "singleladies",
        "무한위장",
        "부수러갈맛집",
        "놀이터의그네",
        "풋풋한시절의부모님",
        "매일조금씩물건을잃어버",
        "내방에모르는물건",
        "정수리냄새",
        "치킨부위",
        "두개다먹어버린다면",
        "절대그주식사지마",
        "주식을하는사람",
        "구해줄배가1년뒤",
        "궁극기",
        "필살기",
        "패딩없이얇은티셔츠",
        "두꺼운패딩입기",
        "비둘기요정",
        "투명한요정친구",
        "컴컴한방에서혼자살",
        "아무하고도연락하지않고",
        "맨앞자리에서만보기",
        "무조건서서보기",
        "커피를쏟았다",
        "세탁비핑계",
        "민트초코맛이나는저주",
        "파인애플피자맛이나는저주",
        "빨간불인데1시간",
        "롤플레잉게임",
        "rpg",
        "샴푸를짜서머리에",
        "물이끊겼다면",
        "건물이갑자기우주선",
        "우주로날아간다면",
        "10원짜리동전100만개",
        "흩뿌려져있는걸발견",
        "아주사소한일",
        "떠오르는색깔세가지",
        "친구와신나게수다",
        "내친구가아니라",
        "똑같은옷을입은모르는사람",
        "험담을길게썼는데",
        "본인에게전송",
        "상사본인에게",
        "물결표",
        "3개씩의무",
        "로맨틱코미디",
        "첫만남장소",
        "100만유튜버",
        "아이템3가지",
        "의미심장하게웃",
        "모기한마리",
        "헛기침소리",
        "한도초과",
        "지갑에현금도없",
        "좋아하는음식의맛",
        "샴푸맛",
        "최고급스테이크맛",
        "내묘비",
        "빈칸에들어갈말",
        "새끼드래곤",
        "알에서갓깨어난",
        "신발끈이매일5번",
        "바지지퍼가매일1번",
        "저신천지",
        "사이비",
        "동요를작곡",
        "그림자가나에게반항",
        "내자리에모르는사람",
        "내음식을먹고있",
        "첫마디를아니그게아니라",
        "10년만에구조",
        "검색해볼단어",
        "지옥철",
        "할머니와동시에눈",
        "탬버린",
        "백댄서를해줄수있는요정",
        "최악의이별통보",
        "머리를감지않아도",
        "찰랑찰랑한머릿결",
        "양치를안해도",
        "너냄새나",
        "동물들과대화",
        "100만원만빌려",
        "당첨사실을말하고",
        "지구인들의행동",
        "이해안가는행동",
        "05배속",
        "2배속으로만",
        "아픈척하며바닥",
        "퀘스트를완료",
        "보상아이템",
        "닫힘버튼",
        "열림버튼",
        "대답하기귀찮",
        "이런걸왜물어봐",
        "싫어하는직장상사",
        "싫어하는교수님",
        "눈이마주친상태에서닫힘버튼",
        "방귀소리",
        "트럼펫소리",
        "바닐라향",
        "뒤에서누군가박수를",
        "쿨하게일어나는순간",
        "핸드폰배터리라면",
        "1가남았는데도",
        "충전기를안꽂고",
        "미지근한물로만샤워",
        "1분마다5초씩얼음물",
        "아끼는물건",
        "살짝부러뜨렸",
        "줄넘기쌩쌩이",
        "내특기가오직",
        "짱구목소리",
        "와맛있다",
        "블루투스연결이끊겨",
        "폰스피커로노래",
        "라면면발이라면",
        "꼬들꼬들하게익히기전에",
        "카카오톡메시지가부모님께",
        "검색어가내이마",
        "사자가나를보고",
        "앞발을모아하트",
        "비대신미스트",
        "눈대신팝콘",
        "오글거리는감성글귀",
        "방벽지가모두",
        "첫접시에디저트",
        "디저트만잔뜩",
        "길고양이라면",
        "츄르를주기위해",
        "세계를구할준비",
        "요원님",
        "잠옷만입고외출",
        "풀정장",
        "집에서쉬기",
        "마법의리모컨",
        "일시정지",
        "되감기5분",
        "닭다리두개를연속",
        "합리적인제지방법",
        "양말이물에살짝젖",
        "작은모래알",
        "지구를정복하기전에",
        "치명적인약점",
        "머리카락굵기를1mm",
        "쓸데없는초능력",
        "내것이훨씬맛있어보일때",
        "한입줄건가",
        "10년전오늘",
        "쓸데없는고민",
        "책상모서리라면",
        "발가락을찧고",
        "드라마의마지막화",
        "영화의반전",
        "외계비행선",
        "나와똑같이생겼다면",
        "하루에10번씩",
        "파이팅을외쳐야",
        "상상력이방전",
        "상상질문에시달리",
        "만원이라겨우탔는데",
        "가장안쪽구석",
        "내릴게요",
        "과일의씨가무조건수박씨",
        "오돌뼈가박혀",
        "모든사람의이름이김철수",
        "연락처에있는",
        "내가만약모기라면",
        "얄미운인간의유형",
        "우산없이걷기",
        "눈오는날반팔",
        "음식이상했다",
        "셰프가나를뚫어지게",
        "트로트메들리",
        "모든알람",
        "10만원을빌려",
        "자명종시계라면",
        "머리를쾅쾅",
        "시험공부안하고놀고",
        "딱한대만때릴수",
        "칫솔대신손가락",
        "주방세제쓰기",
        "텅빈지하철칸",
        "바짝붙어앉",
        "당첨용지를잃어버",
        "그걸삼켰다면",
        "팝콘대신생쌀",
        "커피대신소금물",
        "길거리전봇대라면",
        "꼴보기싫은일",
        "재채기를멈출수없",
        "화장실변기가막혔",
        "뚫을도구가전혀없",
        "아니근데",
        "어쩔티비",
        "벽장문이열리더니",
        "와이파이가안터진다면",
        "다리가3개들어있",
        "모든신발에작은돌멩이",
        "바지지퍼가반쯤",
        "냉장고안의반찬통",
        "곰팡이가피어",
        "방귀뀌고싶어하는생각",
        "주변10m",
        "요리를해줬는데맛이정말끔찍",
        "어때라고물어",
        "자판기가하나",
        "버튼이딱하나",
        "앞머리가일자로만",
        "뒷머리가까치집",
        "귀여운강아지",
        "정색하고나를물려고",
        "싫어하던음식브랜드",
        "모델이되어있다면",
        "겨울에에어컨",
        "여름에보일러",
        "지금이질문을읽고있는",
        "당신의자세",
        "물티슈와휴지가없",
        "종업원은너무바빠",
        "중학교졸업사진",
        "우주최강귀요미",
        "1일마법사",
        "기억속에서나를완전히지울",
        "모기향이라면",
        "10년뒤의내가유튜브채널",
        "매운거먹고울고",
        "신발에물이살짝스며",
        "엉덩이부분이살짝찢",
        "100만원짜리명품백",
        "몰래카메라가설치",
        "눕기만하면정신이말똥",
        "서있으면무조건1분",
        "길거리의쓰레기통",
        "어제그일은비밀",
        "오페라톤",
        "강남스타일말춤",
        "나침반돋보기성냥",
        "성냥중단하나",
        "자기핸드폰만보고",
        "액정보호필름이라면",
        "기포를남긴채",
        "오늘운세",
        "홀로그램",
        "휴지가사포",
        "수건이항상물에젖",
        "옆자리커플",
        "크게싸워",
        "지우개라면",
        "샤프심으로찌를",
        "전국방송",
        "짱구춤",
        "질척거리는진밥",
        "덜익은된밥",
        "이름이똑같",
        "시계바늘이라면",
        "세계구급악당두목",
        "세계급악당두목",
        "히터고장난버스",
        "에어컨고장난지하철",
        "우리치킨먹으러갈래",
        "배가너무부른상태",
        "안경이라면",
        "김서린채",
        "은행을털수있는기회",
        "재수없는저주",
        "제일싫어하는연예인",
        "침대밑에서낯선사람의일기장",
        "내일을예언",
        "상상력의한계테스트",
        "뇌는지금어떤상태",
        "지독한방귀를뀌고내렸",
        "다음층에서다른사람이탈",
        "볼륨이항상최대치",
        "진동모드로만고정",
        "좋아하는음식코너",
        "새로채워질때까지기다려야",
        "버려진껌이라면",
        "밟고지나가는사람",
        "쓰레기통을발로찼",
        "진짜고양이가들어",
        "상대방의말을두번씩반복",
        "앵무새병",
        "앞머리없는삭발머리",
        "게임플레이중계방송",
        "귓가에서",
        "엄마아빠하고달려와안겼",
        "진짜부모님",
        "최악의노래실력",
        "양말을무조건짝짝이",
        "하루에10시간씩투덜",
        "함께떨어진사람",
        "마음속잔고",
        "잔고가보이는초능력",
        "1시간30분뒤에오는저주",
        "식어서오는저주",
        "치과의사라면",
        "사탕을먹겠다는환자",
        "삑삑이신발소리",
        "걸을때마다",
        "축가를부르게",
        "mr이고장",
        "무반주",
        "보일러를켜면에어컨",
        "에어컨을켜면보일러",
        "텐트에서살기",
        "거울속의내가",
        "잘살고있냐",
        "닭의모든뼈가연골",
        "흐물흐물",
        "궁서체로삐뚤빼뚤",
        "거꾸로만써지는",
        "모기장이라면",
        "모기들이나를뚫으려",
        "칭찬하는소리만안들리는",
        "사준옷이정말내취향이아니고",
        "촌스럽다면",
        "마법의램프",
        "빨리하나만말해",
        "바지의주머니가막혀",
        "상의에주머니가5개",
        "넘어진사람을도와주려고",
        "도망간다면",
        "마지막뉴스앵커",
        "지구가멸망하기10분전",
        "겨울에선풍기",
        "여름에전기장판",
        "가장떠오르는영화대사",
        "질문폭탄에시달린",
        "우리다섯명밖에없지",
        "음산하게웃",
        "셀카모드로만고정",
        "화질이144p",
        "싫어하는한가지재료",
        "억지로라도드실",
        "세워진마네킹",
        "옷을갈아입혀주는",
        "가로수를쳤",
        "통째로뽑혀",
        "헬륨가스마신목소리",
        "다스베이더목소리",
        "문워크로돌기",
        "동네한바퀴",
        "asmr먹방소리",
        "길잃은고양이못보셨어요",
        "변기라면",
        "스카치테이프로만신발",
        "자기자랑만하는사람",
        "10시간씩자기자랑",
        "남은수명",
        "택배가무조건옆집",
        "3일씩늦게오는",
        "유아인머리",
        "미용사인데",
        "치킨냄새가진동",
        "축사를맡게",
        "원고를잃어버렸",
        "전기장판없이살기",
        "에어컨선풍기없이",
        "오늘밤파티준비",
        "스프가하나도없고",
        "면만5봉지",
        "유치원생수준",
        "공포스러운분위기",
        "단한가지소원",
        "속으로부르는노래",
        "나사실외계인이야",
        "병원에데려갈",
        "어떤음료수가나오길",
        "나를쳐다보지마세요",
        "흰색쫄쫄이",
        "당신드디어왔군",
        "양말을허물벗듯",
        "겨울에아이스아메리카노",
        "여름에뜨거운국밥",
        "꿈에나올까봐두려운",
        "소름돋는상상",
        "비보잉브레이크댄스",
        "3초간정지",
        "이모티콘을절대쓸수없는",
        "이모티콘만쓸수있고",
        "향수냄새가너무좋",
        "심호흡을하다가",
        "알람을끄고다시자는",
        "흑염룡",
        "이불킥예약",
        "데헷",
        "크큭",
        "내이야기를아주재미있게엿듣",
        "각색해서떠들",
        "코고는소리",
        "24시간동안",
        "쓰레기봉투에코를박",
        "스마트폰배터리1%상태",
        "틱톡을켤",
        "신발좌우를바꿔",
        "포기할신발",
        "춤만추는댄서",
        "오늘배변여부",
        "테이프로100바퀴",
        "상자가열린채",
        "데이트가있는데망쳐주세요",
        "앞이안보인다",
        "축의금을내려고봉투",
        "영수증뭉치",
        "반팔입고덜덜",
        "목도리칭칭",
        "지퍼열렸었어",
        "한강라면",
        "물조절에실패",
        "눈을반쯤감은",
        "심령사진",
        "모기가나를뚫지못하고분노",
        "어떤비웃음",
        "속으로욕하는소리",
        "전생에뽀로로",
        "크롱",
        "나도방금램프에서쫓겨났어",
        "목덜미상표",
        "발가락봉제선",
        "드디어덫에걸렸군",
        "무거운책을올려놓고탈",
        "겨울에선글라스",
        "여름에털장갑",
        "현실로도피",
    )
    return any(marker in text for marker in markers)


def _answer_need(features: MessageFeatures) -> float:
    if features.requests_external_fact:
        return 0.95
    if features.is_question:
        return 0.78
    if "clarification" in features.response_needs:
        return 0.72
    if "explanation" in features.response_needs:
        return 0.70
    if "social_followup" in features.response_needs:
        return 0.45
    return 0.25


def _pressure(features: MessageFeatures) -> float:
    value = 0.15
    if features.intent == Intent.HOSTILE:
        value += 0.65
    if features.sentiment == "negative":
        value += 0.30
    if any(marker in features.normalized for marker in _NEGATIVE_MARKERS):
        value += 0.20
    if "complaint_emphasis" in features.pragmatic_cues:
        value += 0.18
    return _clamp(value)


def _playfulness(features: MessageFeatures) -> float:
    value = 0.0
    if features.intent in {Intent.LAUGH, Intent.TEASE}:
        value += 0.45
    if any(marker in features.normalized for marker in _PLAYFUL_MARKERS):
        value += 0.35
    return _clamp(value)


def _infer_tone(features: MessageFeatures) -> str:
    if features.intent == Intent.HOSTILE:
        return "hostile"
    if _pressure(features) >= 0.55:
        return "strained"
    if _playfulness(features) >= 0.35:
        return "playful"
    if features.is_question or "social_followup" in features.response_needs:
        return "casual"
    return "neutral"


def _next_mood(*, evidence: EvidencePacket, state: CharacterState) -> str:
    if evidence.pressure >= 0.7:
        return "steady"
    if evidence.pressure >= 0.45:
        return "supportive"
    if evidence.playfulness >= 0.35:
        return "playful"
    if evidence.answer_need >= 0.7:
        return "curious"
    return state.mood or "relaxed"


def _lexical_topics(text: str) -> list[str]:
    chunks = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    return [chunk for chunk in chunks if chunk not in _TOPIC_STOPWORDS][:6]


def _dedupe_preserve_order(items) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        for part in str(item or "").split("|"):
            cleaned = part.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, round(float(value), 4)))
