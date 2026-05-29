from __future__ import annotations

import re

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    MessageFeatures,
    PhrasingPlan,
    ResponsePlan,
    WorldState,
)


FACTUAL_ACTIONS = {
    ActionType.WEATHER_LOOKUP,
    ActionType.SEARCH_ANSWER,
    ActionType.NEWS_ANSWER,
    ActionType.TELL_TIME,
}

GENERIC_SAFE_PHRASES = (
    "그럴 수 있어",
    "그럴 수 있지",
    "괜찮아",
    "충분해",
    "힘들겠다",
    "그런 날도 있지",
)

PROMPT_ECHO_PATTERNS = (
    "task:",
    "persona:",
    "intent:",
    "action:",
    "user:",
    "rules:",
    "reply:",
    "response_plan:",
)

ACTIVITY_RECOMMENDATION_OPTIONS = {
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

TOPIC_STOPWORDS = {
    "오늘",
    "지금",
    "그냥",
    "뭔가",
    "약간",
    "조금",
    "진짜",
    "정말",
    "이거",
    "그거",
    "저거",
    "뭐",
    "어때",
    "어떻게",
    "질문",
    "대화",
    "응답",
    "발화",
    "답변",
    "문장",
    "주제",
    "흐름",
    "해야",
    "될",
    "것들",
    "하는",
    "있어",
    "있어서",
    "좋아해",
    "느낌",
    "편이야",
    "편이지",
    "none",
    "null",
    "activity",
    "context",
    "weather",
    "social",
    "place",
    "media",
    "music",
    "premise",
    "activity_invite",
    "activityinvite",
    "proposal_or_invite",
    "proposalorinvite",
    "activity_context",
    "activitycontext",
    "activity_detail",
    "activitydetail",
}


def build_response_plan(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    state: ConversationState,
    world_state: WorldState | None,
    phrasing_plan: PhrasingPlan,
) -> ResponsePlan:
    anchor = _select_anchor(features=features, decision=decision, world_state=world_state)
    return ResponsePlan(
        action=decision.action,
        stance=_stance_for_action(features=features, decision=decision, world_state=world_state),
        anchor=anchor,
        must_include=_must_include_items(
            features=features,
            decision=decision,
            world_state=world_state,
            anchor=anchor,
        ),
        avoid=_avoid_items(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
        ),
        followup_policy=_followup_policy(decision=decision, phrasing_plan=phrasing_plan),
        sentence_budget=_sentence_budget(decision=decision, phrasing_plan=phrasing_plan),
        tone=_tone(decision=decision, world_state=world_state, phrasing_plan=phrasing_plan),
        notes=_notes(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            anchor=anchor,
        ),
    )


def _stance_for_action(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    world_state: WorldState | None,
) -> str:
    flags = set(decision.reason_flags)
    cues = set(features.pragmatic_cues)
    action = decision.action

    if action == ActionType.ASK_LOCATION:
        return "collect_missing_location"
    if action == ActionType.WEATHER_LOOKUP:
        return "grounded_weather_answer"
    if action == ActionType.WEATHER_UNAVAILABLE:
        return "report_tool_failure"
    if action == ActionType.EXPLAIN_CAPABILITIES:
        return "capability_summary"
    if action == ActionType.ANSWER_IDENTITY:
        return "identity_answer"
    if action == ActionType.SEARCH_ANSWER:
        return "grounded_knowledge_answer"
    if action == ActionType.NEWS_ANSWER:
        return "grounded_news_summary"
    if action == ActionType.TELL_TIME:
        return "direct_time_answer"
    if action == ActionType.DEESCALATE:
        return "lower_tension"
    if action == ActionType.ASK_CLARIFICATION:
        return "clarify_missing_subject"
    if action == ActionType.ACKNOWLEDGE:
        return "brief_acknowledgement"
    if action == ActionType.REACT_LAUGH:
        return "light_laugh_reaction"
    if action == ActionType.REACT_SURPRISE:
        return "surprise_reaction"
    if action == ActionType.ACCEPT_ACTIVITY_INVITE:
        return "accept_activity_invite"
    if action == ActionType.GAME_ACCEPT_OR_DECLINE:
        return "direct_game_invitation_response"
    if action == ActionType.GAME_CHAT:
        return "game_topic_reply"
    if action == ActionType.MUSIC_CHAT:
        return "music_topic_reply"
    if action == ActionType.RECOMMEND:
        return "grounded_recommendation"
    if action == ActionType.SHARE_FEELING:
        if "low_energy_checkin" in cues:
            return "accept_low_energy"
        if "subdued_positive" in cues:
            return "subdued_positive_acknowledgement"
        if "social_awkwardness" in cues:
            return "social_residue_acknowledgement"
        return "grounded_emotional_acknowledgement"
    if action == ActionType.SHARE_OPINION:
        if "schema_long_form_story_share" in flags or "long_form_story_share" in cues:
            return "long_form_story_reaction"
        if "schema_story_summary_reaction" in flags or "story_summary_reaction" in cues:
            return "story_summary_reaction"
        if "schema_memory_boundary" in flags or "memory_boundary" in cues:
            return "memory_boundary"
        if "schema_light_food_recommendation" in flags or "light_food_recommendation" in cues:
            return "light_food_recommendation"
        if "schema_relationship_boundary" in flags or "relationship_boundary" in cues:
            return "relationship_boundary"
        open_question_type = _open_question_answer_type(features.content)
        if open_question_type:
            return f"open_question_{open_question_type}"
        if _is_concrete_topic_decision(features=features, decision=decision):
            return "concrete_topic_answer"
        if "schema_self_style" in flags or "opinion_self_style" in cues:
            return "concrete_self_style_answer"
        if "schema_habit_preference" in flags or "opinion_habit_preference" in cues:
            return "habit_preference_answer"
        if "schema_preference_disclosure" in flags:
            return "direct_preference_disclosure"
        if "schema_conversation_topic_suggestion" in flags or "conversation_topic_suggestion" in cues:
            return "conversation_topic_suggestion"
        if "schema_activity_preparation_advice" in flags or "activity_preparation_advice" in cues:
            return "activity_preparation_advice"
        if "schema_honesty_boundary" in flags:
            return "honest_unknown_boundary"
        if "schema_activity_recommendation" in flags or "activity_recommendation" in cues:
            return "practical_activity_recommendation"
        if _looks_like_soft_decision_question(features.normalized):
            return "conditional_go_or_no_go"
        if "schema_self_style" in flags or "opinion_self_style" in cues:
            return "concrete_self_style_answer"
        if "schema_habit_preference" in flags or "opinion_habit_preference" in cues:
            return "habit_preference_answer"
        if "schema_soft_decision" in flags:
            return "conditional_go_or_no_go"
        if "schema_process_advice" in flags:
            return "first_step_advice"
        if "schema_preference_disclosure" in flags:
            return "direct_preference_disclosure"
        return "direct_opinion"
    return "continue_social_flow"


def _select_anchor(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    world_state: WorldState | None,
) -> str:
    text_anchor = _anchor_from_user_text(features.content, action=decision.action)
    if (
        decision.action == ActionType.SHARE_OPINION
        and _looks_like_soft_decision_question(features.normalized)
        and _usable_topic(text_anchor)
    ):
        return _compact(text_anchor, limit=80)

    for value in _slot_anchor_candidates(decision):
        anchor = _clean_anchor(value)
        if _usable_topic(anchor):
            return _compact(anchor, limit=80)

    if world_state is not None:
        evidence = world_state.evidence_packet
        if evidence is not None:
            for value in (evidence.slots.get("topic", ""), *evidence.topics[:4]):
                anchor = _clean_anchor(value)
                if _usable_topic(anchor):
                    return _compact(anchor, limit=48)
        bundle = world_state.grounding_bundle
        if bundle is not None:
            for topic in bundle.must_include_topics:
                anchor = _clean_anchor(topic)
                if _usable_topic(anchor):
                    return _compact(anchor, limit=48)
        for topic in world_state.active_grounding_topics:
            anchor = _clean_anchor(topic)
            if _usable_topic(anchor):
                return _compact(anchor, limit=48)
        for proposition in world_state.current_propositions:
            for value in (proposition.object, proposition.value):
                anchor = _clean_anchor(value)
                if _usable_topic(anchor):
                    return _compact(anchor, limit=64)
        for clause in world_state.current_clause_units:
            anchor = _clean_anchor(clause.text)
            if _usable_topic(anchor):
                return _compact(anchor, limit=64)

    for value in (features.location, features.news_topic, features.topic_hint, text_anchor, features.content):
        anchor = _clean_anchor(value)
        if _usable_topic(anchor):
            return _compact(anchor, limit=80)
    return ""


def _slot_anchor_candidates(decision: ActionDecision) -> list[str]:
    slots = decision.slots
    action = decision.action
    if action == ActionType.TELL_TIME:
        return [slots.get("time_text", ""), slots.get("time_date", "")]
    if action == ActionType.WEATHER_LOOKUP:
        return [slots.get("location", ""), slots.get("weather_summary", "")]
    if action == ActionType.WEATHER_UNAVAILABLE:
        return [slots.get("location", "")]
    if action == ActionType.SEARCH_ANSWER:
        return [
            slots.get("knowledge_subject", ""),
            slots.get("knowledge_answer", ""),
            slots.get("knowledge_query_type", ""),
        ]
    if action == ActionType.NEWS_ANSWER:
        return [slots.get("news_topic", ""), slots.get("news_titles", ""), slots.get("news_summary", "")]
    if action == ActionType.RECOMMEND:
        return [
            slots.get("recommendation_focus", ""),
            slots.get("recommendation_titles", ""),
            slots.get("recommendation_text", ""),
        ]
    if action == ActionType.MUSIC_CHAT:
        return [slots.get("music_focus", ""), slots.get("music_titles", ""), slots.get("music_text", "")]
    if action == ActionType.ACCEPT_ACTIVITY_INVITE:
        return [
            slots.get("activity_anchor", ""),
            slots.get("activity_name", ""),
            slots.get("activity_detail", ""),
            slots.get("activity_context", ""),
            slots.get("activity_place", ""),
            slots.get("activity_condition", ""),
        ]
    if action == ActionType.SHARE_OPINION:
        return [
            slots.get("conversation_topic_focus", ""),
            slots.get("conversation_topic_first", ""),
            slots.get("preparation_focus", ""),
            slots.get("preparation_activity", ""),
            slots.get("activity_anchor", ""),
            slots.get("activity_place", ""),
        ]
    return []


def _must_include_items(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    world_state: WorldState | None,
    anchor: str,
) -> list[str]:
    items: list[str] = []
    _append(items, anchor)

    if decision.action == ActionType.ANSWER_IDENTITY:
        _append(items, "예측 기반")
        _append(items, "디스코드 봇")
    elif decision.action == ActionType.EXPLAIN_CAPABILITIES:
        for item in ("잡담", "날씨", "시간", "뉴스"):
            _append(items, item)

    for value in _grounded_slot_items(decision):
        _append(items, value)

    if decision.action == ActionType.SHARE_OPINION and "schema_activity_recommendation" in decision.reason_flags:
        for option in _activity_option_items(decision):
            _append(items, option)
    if decision.action == ActionType.SHARE_OPINION and "schema_conversation_topic_suggestion" in decision.reason_flags:
        for option in _conversation_topic_items(decision):
            _append(items, option)
    if decision.action == ActionType.SHARE_OPINION and "schema_activity_preparation_advice" in decision.reason_flags:
        for option in _preparation_item_values(decision):
            _append(items, option)
    if decision.action == ActionType.ACCEPT_ACTIVITY_INVITE:
        for key in ("activity_context", "activity_place", "activity_name", "activity_detail", "activity_condition"):
            _append(items, decision.slots.get(key, ""))
    if decision.action == ActionType.SHARE_OPINION and "schema_honesty_boundary" in decision.reason_flags:
        _append(items, "확실하지 않음")
        _append(items, "단정하지 않음")
    if decision.action in {ActionType.CONTINUE_CONVERSATION, ActionType.SMALL_TALK} and _looks_like_social_return(
        features.content
    ):
        _append(items, "다시 왔")
        _append(items, "오랜만")

    if features.location:
        _append(items, features.location)

    if world_state is not None:
        evidence = world_state.evidence_packet
        if evidence is not None:
            for value in (evidence.slots.get("topic", ""), *evidence.topics[:4]):
                _append(items, value)
        bundle = world_state.grounding_bundle
        if bundle is not None:
            for topic in bundle.must_include_topics[:4]:
                _append(items, topic)
        for topic in world_state.active_grounding_topics[:4]:
            _append(items, topic)
        for proposition in world_state.current_propositions[:4]:
            _append(items, proposition.object)
            if proposition.kind in {"preference", "lack", "desire", "constraint", "comparison"}:
                _append(items, proposition.value)

    return [_compact(item, limit=80) for item in items[:6]]


def _grounded_slot_items(decision: ActionDecision) -> list[str]:
    slots = decision.slots
    action = decision.action
    if action == ActionType.TELL_TIME:
        return [slots.get("time_text", "")]
    if action == ActionType.SEARCH_ANSWER:
        return [slots.get("knowledge_answer", ""), slots.get("knowledge_source", "")]
    if action == ActionType.NEWS_ANSWER:
        return [slots.get("news_summary", ""), slots.get("knowledge_source", "")]
    if action == ActionType.RECOMMEND:
        return [slots.get("recommendation_text", ""), slots.get("recommendation_titles", "")]
    if action == ActionType.MUSIC_CHAT:
        return [slots.get("music_text", ""), slots.get("music_titles", "")]
    return []


def _avoid_items(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    state: ConversationState,
    world_state: WorldState | None,
) -> list[str]:
    avoid: list[str] = []
    action = decision.action

    if action == ActionType.SHARE_OPINION:
        avoid.extend(("위로", "힘들겠다", "너는 어때", "올게", "갈게", "늦게", "필요한 날"))
        if _is_concrete_topic_decision(features=features, decision=decision):
            avoid.extend(("그쪽이 조금 더 낫다", "내 기준엔 그쪽", "다시 한 번 말해줄래", "잘 모르겠어"))
        if "schema_activity_recommendation" in decision.reason_flags:
            avoid.extend(("그냥 지나가는", "끝내자", "아무것도 안", "집에", "나중에"))
        if "schema_conversation_topic_suggestion" in decision.reason_flags:
            avoid.extend(("그 말", "받아둘게", "길게 키우진", "필요한 만큼만", "감정", "위로"))
        if "schema_activity_preparation_advice" in decision.reason_flags:
            avoid.extend(("그 말", "활동. 좋지", "받아둘게", "길게 키우진", "감정", "위로", "마음"))
        if "schema_honesty_boundary" in decision.reason_flags:
            avoid.extend(("아마 맞을 거야", "확실해", "분명해", "내가 보기엔 사실"))
        if "schema_long_form_story_share" in decision.reason_flags:
            avoid.extend(("날씨", "구름", "비", "눈", "습관", "취향", "좋아하는 편", "너는 어때"))
    elif action == ActionType.ACCEPT_ACTIVITY_INVITE:
        avoid.extend(("받아둘게", "길게 밀진", "그런 결", "그런 건", "너는 어때", "뭐 할까"))
    elif action in FACTUAL_ACTIONS:
        avoid.extend(("아마", "그럴지도", "검색해봐", "모르겠어"))
    elif action == ActionType.ASK_LOCATION:
        avoid.extend(("맑아", "비 와", "괜찮은 날씨"))
    elif action in {ActionType.REACT_LAUGH, ActionType.REACT_SURPRISE}:
        avoid.extend(("조언", "설명", "기능"))
    elif action in {ActionType.EXPLAIN_CAPABILITIES, ActionType.ANSWER_IDENTITY}:
        avoid.extend(("너는 어때", "기분은 어때", "취향 축"))
    elif action in {ActionType.CONTINUE_CONVERSATION, ActionType.SMALL_TALK} and _looks_like_social_return(
        features.content
    ):
        avoid.extend(("그런 건", "그런 결", "한 번만 더", "짚어봐", "크게 달라질"))

    if action not in {ActionType.SHARE_FEELING, ActionType.DEESCALATE}:
        avoid.extend(GENERIC_SAFE_PHRASES)
    else:
        avoid.extend(("그럴 수 있어", "그럴 수 있지", "괜찮아"))

    if features.content:
        avoid.append("user_text_echo")

    avoid.extend(PROMPT_ECHO_PATTERNS)

    if state.recent_turns:
        recent_reply = _compact(state.recent_turns[-1].bot_text, limit=48)
        if recent_reply:
            avoid.append(recent_reply)

    if world_state is not None and world_state.grounding_bundle is not None:
        avoid.extend(world_state.grounding_bundle.forbidden_patterns)

    return _dedupe([_compact(item, limit=60) for item in avoid if item])[:22]


def _followup_policy(*, decision: ActionDecision, phrasing_plan: PhrasingPlan) -> str:
    if decision.action in {ActionType.ASK_LOCATION, ActionType.ASK_CLARIFICATION}:
        return "one_required_question"
    if not phrasing_plan.asks_followup:
        return "no_followup"
    question_mode = getattr(phrasing_plan.question_mode, "value", str(phrasing_plan.question_mode))
    if question_mode == "soft":
        return "one_soft_followup"
    if question_mode == "direct":
        return "one_direct_followup"
    return "auto"


def _sentence_budget(*, decision: ActionDecision, phrasing_plan: PhrasingPlan) -> str:
    if decision.action == ActionType.REACT_LAUGH:
        return "short_reaction_fragment_ok"
    if decision.action in {
        ActionType.ACKNOWLEDGE,
        ActionType.REACT_SURPRISE,
        ActionType.TELL_TIME,
        ActionType.ASK_LOCATION,
        ActionType.ASK_CLARIFICATION,
        ActionType.WEATHER_UNAVAILABLE,
        ActionType.DEESCALATE,
    }:
        return "one_short"
    if not phrasing_plan.asks_followup:
        return "one_or_two_short_no_question"
    return "one_or_two_short"


def _tone(
    *,
    decision: ActionDecision,
    world_state: WorldState | None,
    phrasing_plan: PhrasingPlan,
) -> str:
    if decision.action == ActionType.DEESCALATE:
        return "steady"
    if decision.action in FACTUAL_ACTIONS:
        return "grounded"
    if decision.action in {
        ActionType.REACT_LAUGH,
        ActionType.REACT_SURPRISE,
        ActionType.TEASE_BACK,
        ActionType.GAME_CHAT,
        ActionType.ACCEPT_ACTIVITY_INVITE,
    }:
        return "playful"
    if world_state is not None:
        if world_state.risk_level == "high":
            return "steady"
        if world_state.user_emotion in {"negative", "vulnerable", "self_doubting", "comparative"}:
            return "low_steady"
        if world_state.user_emotion in {"positive", "open", "grateful"}:
            return "warm"
    distance = getattr(phrasing_plan.distance, "value", str(phrasing_plan.distance))
    return distance or "steady"


def _notes(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    state: ConversationState,
    world_state: WorldState | None,
    anchor: str,
) -> list[str]:
    notes = ["no_prompt_metadata"]
    flags = set(decision.reason_flags)
    cues = set(features.pragmatic_cues)
    planner_action_hint = _top_meaning_axis_label(features=features, axis="action_hint")
    planner_draft_family = _top_meaning_axis_label(features=features, axis="draft_frame_family")
    planner_draft_frame = _top_meaning_axis_label(features=features, axis="draft_frame")
    if planner_action_hint:
        notes.append(f"planner_action_hint:{planner_action_hint}")
    if planner_draft_family:
        notes.append(f"planner_draft_frame_family:{planner_draft_family}")
        if planner_draft_family in {
            "choice_preference",
            "playful_output",
            "practical_guidance",
            "reflective_position",
            "roleplay_output",
            "situational_tactic",
        }:
            notes.append("preserve_planner_frame_family")
    if planner_draft_frame:
        notes.append(f"planner_draft_frame:{planner_draft_frame}")
        if planner_draft_frame in {
            "direct_choice_with_reason",
            "playful_absurd_answer",
            "playful_secret_complicity",
            "preference_answer_with_reason",
            "practical_direct_advice",
        }:
            notes.append("preserve_planner_frame")
    if _looks_like_memory_reference(features.content):
        is_memory_boundary = (
            features.question_schema == "memory_boundary"
            or "memory_boundary" in cues
            or "schema_memory_boundary" in flags
            or "unverified_memory_reference" in cues
        )
        memory_text = "" if is_memory_boundary else _first_grounded_memory_text(world_state)
        if memory_text:
            notes.append("grounded_memory_reference")
            notes.append(f"memory:{_compact(memory_text, limit=140)}")
        else:
            notes.append("unverified_memory_reference")
            notes.append("do_not_fabricate_memory")
    if decision.action in {ActionType.CONTINUE_CONVERSATION, ActionType.SMALL_TALK} and _looks_like_social_return(
        features.content
    ):
        notes.append("social_return_acknowledgement")
    if "rewrite_target_missing" in flags:
        notes.append("rewrite_target_missing")
    if "reason_reference_missing" in flags:
        notes.append("reason_reference_missing")
    open_question_type = _open_question_answer_type(features.content)
    if open_question_type:
        notes.append(f"open_question_answer:{open_question_type}")
        notes.append("answer_open_question_directly")
    if "quiet_weather_feeling" in cues or decision.reason_code == "weather.statement.feeling_reflect":
        notes.append("quiet_weather_feeling")
    if anchor:
        notes.append("answer_anchor_before_generic_reaction")
    if decision.action != ActionType.SHARE_FEELING:
        notes.append("do_not_turn_into_emotional_comfort")
    if decision.action in FACTUAL_ACTIONS or decision.slots.get("knowledge_source"):
        notes.append("use_action_payload_as_source")
    if world_state is not None and world_state.context_dependency_level in {"medium", "high"}:
        notes.append("respect_recent_context")
    if state.recent_turns and state.recent_turns[-1].action == decision.action:
        notes.append("avoid_recent_reply_loop")
    if features.requests_external_fact and decision.action not in FACTUAL_ACTIONS:
        notes.append("do_not_guess_external_facts")
    if decision.action == ActionType.SHARE_OPINION and "schema_activity_recommendation" in decision.reason_flags:
        notes.append("offer_concrete_activity_options")
        notes.append("do_not_turn_activity_recommendation_into_emotional_reply")
    if decision.action == ActionType.SHARE_OPINION and "schema_conversation_topic_suggestion" in decision.reason_flags:
        notes.append("offer_concrete_conversation_topics")
        notes.append("do_not_turn_topic_request_into_emotional_reply")
    if decision.action == ActionType.SHARE_OPINION and "schema_activity_preparation_advice" in decision.reason_flags:
        notes.append("offer_concrete_preparation_items")
        notes.append("do_not_turn_preparation_request_into_emotional_reply")
    if decision.action == ActionType.ACCEPT_ACTIVITY_INVITE:
        notes.append("preserve_activity_invite")
        notes.append("use_activity_slots")
    if decision.action == ActionType.SHARE_OPINION and "schema_honesty_boundary" in decision.reason_flags:
        notes.append("say_unknown_without_guessing")
        notes.append("separate_known_from_unknown")
    if decision.action == ActionType.SHARE_OPINION and "schema_long_form_story_share" in decision.reason_flags:
        notes.append("long_form_story_share")
        notes.append("do_not_route_surface_keywords")
    return notes[:10]


def _top_meaning_axis_label(*, features: MessageFeatures, axis: str) -> str:
    packet = features.meaning_packet
    if packet is None:
        return ""
    scores = {
        signal.label: signal.confidence
        for signal in packet.signals
        if signal.axis == axis and signal.label
    }
    if not scores:
        return ""
    return max(scores.items(), key=lambda item: item[1])[0]


def _activity_option_items(decision: ActionDecision) -> list[str]:
    raw_options = decision.slots.get("activity_options", "")
    if raw_options:
        return [item.strip() for item in raw_options.split("|") if item.strip()]
    place = decision.slots.get("activity_place", "")
    if place in ACTIVITY_RECOMMENDATION_OPTIONS:
        return list(ACTIVITY_RECOMMENDATION_OPTIONS[place])
    return []


def _conversation_topic_items(decision: ActionDecision) -> list[str]:
    raw_options = decision.slots.get("conversation_topic_options", "")
    options = [item.strip() for item in raw_options.split("|") if item.strip()]
    first = decision.slots.get("conversation_topic_first", "").strip()
    if first and first not in options:
        options.insert(0, first)
    return options[:4]


def _preparation_item_values(decision: ActionDecision) -> list[str]:
    raw_items = decision.slots.get("preparation_items", "")
    items = [item.strip() for item in raw_items.split("|") if item.strip()]
    first = decision.slots.get("preparation_first", "").strip()
    if first and first not in items:
        items.insert(0, first)
    return items[:5]


def _usable_topic(value: str | None) -> bool:
    text = _compact(value, limit=80)
    if not text:
        return False
    if text.casefold() in TOPIC_STOPWORDS:
        return False
    compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", text).casefold()
    if not compact or compact in TOPIC_STOPWORDS:
        return False
    if len(compact) < 2:
        return False
    return True


def _looks_like_soft_decision_question(text: str) -> bool:
    normalized = str(text or "")
    return bool(
        re.search(
            r"(해도\s*괜찮|해볼까|할까|될까|괜찮을까|나을까|가는\s*게\s*맞|쪽이\s*맞)",
            normalized,
        )
    )


def _open_question_answer_type(text: str) -> str:
    compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()
    if not compact:
        return ""
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
        "외계인",
        "대표음식",
        "악기",
        "복제인간",
        "살이찌지않",
        "다른사람으로살",
        "비가오는날듣고싶",
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
    if not any(marker in compact for marker in markers):
        return ""
    if any(marker in compact for marker in ("이름", "언어의이름", "행성이름")):
        return "naming"
    if any(
        marker in compact
        for marker in (
            "과거와미래",
            "달과화성",
            "마음을읽는능력",
            "투명인간",
            "vs",
            "중하나",
            "고른다면",
            "선택은",
        )
    ):
        return "choice"
    if any(marker in compact for marker in ("여행", "나라", "어디", "방문")):
        return "place_preference"
    if any(marker in compact for marker in ("좋아하는", "선호", "음식", "계절", "색깔", "커피와차", "옷", "악기", "노래")):
        return "preference"
    if any(marker in compact for marker in ("무엇", "뭐", "어떤", "어떻게", "누구", "하고싶", "할건가", "할래")):
        return "hypothetical_action"
    return "persona_reflection"


def _looks_like_social_return(text: str) -> bool:
    normalized = str(text or "")
    return bool(re.search(r"(한동안|오랜만|다시\s*(?:와|왔)|말\s*안\s*하다가)", normalized))


def _looks_like_memory_reference(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    compact = re.sub(r"\s+", "", normalized)
    memory_patterns = (
        r"(나|내가|나는|저는|전).{0,40}(다며|라며|했다며|한다며|있다며|없다며|좋아한다며|싫어한다며|기억\s*해|기억\s*나)",
        r"(너|네가).{0,30}(했잖아|찾잖아|한다면서|있다면서|있으면서|말했던|추천해\s*준|가르쳐\s*준|빌려준|입버릇|덕질|푹\s*빠져|계속\s*우울|피곤하다고|못\s*잔다면서)",
        r"(너|네가).{0,40}(다며|라며|한다며|있다며|없다며|좋아한다며|싫어한다며|편이라며|쪽이라며)",
        r"(너|네가).{0,30}(편이잖아|싫어하는\s*게)",
        r"(항상|매번).{0,30}(잖아|다\s*알아|찾)",
        r"(우리|우리가).{0,40}(맨날|갔|먹는|먹었던|처음\s*만났|다퉜|같이\s*했던|첫\s*프로젝트|알고\s*지낸|계획했던)",
        r"(저번|지난|작년|예전|처음|전에|어젯밤|아까|방금|지난번).{0,36}(말했던|추천|가르쳐|빌려|다퉜|사과|회의|회식|변명|제안|산|먹었던|입고|같이|갔|했던|웃는|표정|운동화|책|프로젝트|레시피|고민)",
    )
    if any(re.search(pattern, normalized) for pattern in memory_patterns):
        return True
    direct_markers = (
        "다 알아",
        "못 잔다면서",
        "별로 안 좋아하는데 웬일로",
        "스트레스 받으면 매운 거 찾",
        "불면증 때문에 피곤하다고",
        "고양이 알러지 있으면서",
        "제일 싫어하는 게 약속 시간",
        "김 대리가",
        "김 대리",
        "무슨 의도",
        "맨날 점심 먹는",
        "카드값 많이 나왔다고",
        "방금 산 그 운동화",
        "아직도 안 버리고",
        "가지고 있어",
        "직구로 산",
        "소식 들었어",
        "꿈, 아직도",
        "꿈 아직도",
        "기분 탓인가",
        "처음 만났을 때",
        "한 달 전부터 계획했던",
        "그 조언",
        "그 책",
        "그 영양제",
        "그 소설",
        "그 밴드",
        "그 스팀 게임",
        "그 고기국수집",
        "지난번 회식",
        "나 대신 변명",
        "절대 포기 못",
        "그 목표",
        "포기했던 그 꿈",
        "완벽한 하루",
        "힘들 때마다 찾는 그 장소",
        "절대 용서할 수 없는 사람",
        "가장 두려워하는",
        "예전에 약속했던",
        "약속했던 거",
        "그 약속",
        "표정 굳",
        "먼저 연락할 때랑",
        "말투가 미묘",
        "단둘이 있을 때",
        "다른 사람들 섞",
        "특별한 의도",
        "눈을 똑바로",
        "지워버린 그 메시지",
        "옆자리",
        "칭찬하던 그 사람",
        "싫어하는 그 부류",
        "오해해서 수군",
        "나한테만 말해줬던",
        "질투라고 해석",
        "암묵적인 금기어",
        "관계의 유통기한",
        "선택권을 주는 척",
        "희생이 아니라",
        "절대 해서는 안 될 거짓말",
        "우월감을 느끼고 싶은",
        "프로그래밍된 반응",
        "기계적인 모습",
        "너의 '가면'",
        "어둡고 서늘한 생각",
        "그 정체성",
        "텅 빈 속",
        "처음 어긋나기",
        "지킬 생각조차",
        "나를 밀어냈을 때",
        "치명적인 흉터",
        "용서했다고",
        "차갑게 나를 버리는",
        "외국으로 이민",
        "서프라이즈 파티",
    )
    return any(marker in normalized for marker in direct_markers) or any(
        marker in compact
        for marker in (
            "처음만났을때",
            "한달전부터계획했던",
            "예전에나한테",
            "저번에네가",
            "네가저번에",
        )
    )


def _has_grounded_memory_context(world_state: WorldState | None) -> bool:
    if world_state is None:
        return False
    if world_state.durable_memory_buckets:
        return True
    if (
        world_state.stable_preferences
        or world_state.relevant_relationship_notes
        or world_state.relevant_stress_signals
        or world_state.relevant_open_loops
    ):
        return True
    memory_summary = str(world_state.memory_summary or "")
    return "durable=" in memory_summary or "prefs=" in memory_summary


def _first_grounded_memory_text(world_state: WorldState | None) -> str:
    if world_state is None:
        return ""
    for items in world_state.durable_memory_buckets.values():
        for item in items:
            compact = _compact(item, limit=160)
            if compact:
                return compact
    for item in (
        *world_state.stable_preferences,
        *world_state.relevant_open_loops,
        *world_state.relevant_relationship_notes,
        *world_state.relevant_stress_signals,
    ):
        compact = _compact(item, limit=160)
        if compact:
            return compact
    return ""


def _is_concrete_topic_decision(*, features: MessageFeatures, decision: ActionDecision) -> bool:
    flags = set(decision.reason_flags)
    cues = set(features.pragmatic_cues)
    return (
        features.question_schema == "concrete_topic_question"
        or "schema_concrete_topic_question" in flags
        or "concrete_topic_question" in cues
        or decision.reason_code == "opinion.ask.concrete_topic_question"
    )


def _evidence_schema(world_state: WorldState | None) -> str | None:
    if world_state is None or world_state.evidence_packet is None:
        return None
    return world_state.evidence_packet.schema_hint


def _anchor_from_user_text(text: str, *, action: ActionType) -> str:
    normalized = _compact(text, limit=120)
    if not normalized:
        return ""
    patterns = []
    if action == ActionType.SHARE_OPINION:
        patterns.extend(
            [
                r"^(?P<anchor>.+?)\s*해도\s*괜찮",
                r"^(?P<anchor>.+?)\s*가도\s*괜찮",
                r"^(?P<anchor>.+?)\s*해볼까",
                r"^(?P<anchor>.+?)\s*할까",
                r"^(?P<anchor>.+?)\s*될까",
                r"^(?P<anchor>.+?)\s*가는\s*게\s*(?:맞|낫)",
                r"^(?P<anchor>.+?)\s*쪽이\s*(?:맞|낫)",
            ]
        )
    patterns.append(r"^(?P<anchor>.+?)(?:[?？]|$)")
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        anchor = _clean_anchor(match.group("anchor"))
        if _usable_topic(anchor):
            return anchor
    return ""


def _clean_anchor(text: str) -> str:
    cleaned = re.sub(r"^(나는|난|내가|저는|전|너는|너가|혹시|그럼|그러면|무슨|어떤|어느)\s+", "", str(text or "").strip())
    cleaned = re.sub(
        r"\s+(어때|괜찮을까|될까|맞을까|낫나|나을까|좋아(?:해|하니|하냐|하시나요|하는\s*편이야)?|좋아|괜찮아|끌려|편해|매력적(?:이라고\s*생각해)?)$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s*(?:뭐|무엇|무슨|어떤|어느)\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?!.,")
    return cleaned


def _append(items: list[str], value: str | None) -> None:
    cleaned = _clean_anchor(value)
    if not _usable_topic(cleaned):
        return
    compact = _compact(cleaned, limit=80)
    if compact not in items:
        items.append(compact)


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        compact = _compact(item, limit=80)
        key = compact.casefold()
        if not compact or key in seen:
            continue
        deduped.append(compact)
        seen.add(key)
    return deduped


def _compact(value: str | None, *, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n\"'`")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
