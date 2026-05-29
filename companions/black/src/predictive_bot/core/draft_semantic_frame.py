from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SILVER_FRAME_RESOLVER = "draft_reason_silver_frame_v1"


@dataclass(frozen=True, slots=True)
class DraftSemanticFrame:
    source_reason: str
    coarse_intent: str = "smalltalk_opinion"
    domain: str = "daily_life"
    schema: str = "preference_disclosure"
    speech_act: str = "ask"
    emotion: str = "curious"
    state_hint: str = "light_social"
    action_hint: str = "share_opinion"
    draft_frame_family: str = "choice_preference"
    draft_frame: str = "preference_answer"
    tone: str = "warm_playful"
    followup_policy: str = "none"
    priority: str = "choice_judgment"
    advice_request: bool = False
    choice_request: bool = True
    no_fake: bool = False
    pragmatic_cues: tuple[str, ...] = field(default_factory=tuple)
    slots: dict[str, str] = field(default_factory=dict)
    resolver: str = SILVER_FRAME_RESOLVER

    def to_dict(self) -> dict[str, Any]:
        signals = [
            {
                "axis": axis,
                "label": label,
                "confidence": 1.0,
                "source": self.resolver,
                "evidence": [self.source_reason],
            }
            for axis, label in (
                ("coarse_intent", self.coarse_intent),
                ("domain", self.domain),
                ("schema", self.schema),
                ("speech_act", self.speech_act),
                ("emotion", self.emotion),
                ("state_hint", self.state_hint),
                ("action_hint", self.action_hint),
                ("draft_frame_family", self.draft_frame_family),
                ("draft_frame", self.draft_frame),
                ("tone", self.tone),
                ("followup_policy", self.followup_policy),
            )
            if label
        ]
        targets = {
            "coarse_intent": self.coarse_intent,
            "domain": self.domain,
            "schema": self.schema,
            "speech_act": self.speech_act,
            "emotion": self.emotion,
            "state_hint": self.state_hint,
            "action_hint": self.action_hint,
            "draft_frame_family": self.draft_frame_family,
            "draft_frame": self.draft_frame,
            "tone": self.tone,
            "followup_policy": self.followup_policy,
            "slots": dict(self.slots),
        }
        return {
            "resolver": self.resolver,
            "source_reason": self.source_reason,
            "coarse_intent": self.coarse_intent,
            "domain": self.domain,
            "schema": self.schema,
            "speech_act": self.speech_act,
            "emotion": self.emotion,
            "state_hint": self.state_hint,
            "action_hint": self.action_hint,
            "draft_frame_family": self.draft_frame_family,
            "draft_frame": self.draft_frame,
            "tone": self.tone,
            "followup_policy": self.followup_policy,
            "priority": self.priority,
            "advice_request": self.advice_request,
            "choice_request": self.choice_request,
            "no_fake": self.no_fake,
            "pragmatic_cues": list(dict.fromkeys(self.pragmatic_cues)),
            "slots": dict(self.slots),
            "signals": signals,
            "targets": targets,
        }


def infer_draft_semantic_frame(
    *,
    direct_surface_reason: str,
    output_shape: str = "",
    draft_domain: str = "",
    draft_frame_detail: str = "",
    action: str = "",
    tone: str = "",
) -> DraftSemanticFrame | None:
    reason = str(direct_surface_reason or output_shape or "").strip()
    if not reason:
        return None

    frame = _frame_name(reason=reason, output_shape=output_shape, draft_frame_detail=draft_frame_detail)
    family = _family_for(reason, frame)
    domain = _domain_for(reason, draft_domain=draft_domain)
    schema = _schema_for(reason)
    priority = _priority_for(reason)
    no_fake = _is_no_fake_reason(reason)
    emotion = _emotion_for(reason, no_fake=no_fake)
    state_hint = _state_hint_for(reason, priority=priority, no_fake=no_fake)
    action_hint = _action_hint_for(reason, action=action)
    response_tone = _tone_for(reason, tone=tone, no_fake=no_fake)
    advice_request = priority in {"immediate_action", "practical_action", "emotion_stabilization"} or _has_any(
        reason,
        "advice",
        "practical",
        "first_steps",
        "how_to",
        "recovery",
        "conflict",
        "slump",
    )
    choice_request = _is_hypothetical_choice_reason(reason) or _is_basic_preference_reason(reason) or _has_any(
        reason,
        "choice",
        "preference",
        "icebreak",
        "vs",
        "lottery",
        "superpower",
        "time_machine",
        "one_month_abroad",
        "desert_island",
        "immortality",
        "character_design",
    )
    cues = _pragmatic_cues_for(
        reason=reason,
        schema=schema,
        frame=frame,
        family=family,
        priority=priority,
        no_fake=no_fake,
        advice_request=advice_request,
        choice_request=choice_request,
    )
    return DraftSemanticFrame(
        source_reason=reason,
        coarse_intent=_coarse_intent_for(reason, action=action),
        domain=domain,
        schema=schema,
        speech_act=_speech_act_for(reason),
        emotion=emotion,
        state_hint=state_hint,
        action_hint=action_hint,
        draft_frame_family=family,
        draft_frame=frame,
        tone=response_tone,
        followup_policy="none",
        priority=priority,
        advice_request=advice_request,
        choice_request=choice_request,
        no_fake=no_fake,
        pragmatic_cues=cues,
    )


def _frame_name(*, reason: str, output_shape: str, draft_frame_detail: str) -> str:
    cleaned = _strip_prefix(reason)
    if cleaned.startswith("draft_frame_detail_"):
        cleaned = cleaned.removeprefix("draft_frame_detail_")
    if cleaned in {"", "unknown"}:
        cleaned = _strip_prefix(output_shape) or draft_frame_detail or "preference_answer"
    if cleaned.startswith("daily_"):
        cleaned = cleaned.removeprefix("daily_")
    if cleaned == "practical_sleep_noise_environment" and draft_frame_detail:
        return draft_frame_detail
    if cleaned == "practical_gas_stove_ignition_issue" and draft_frame_detail:
        return draft_frame_detail
    if cleaned == "practical_appliance_design_review" and draft_frame_detail:
        return draft_frame_detail
    if cleaned == "practical_heating_bill_anxiety" and draft_frame_detail:
        return draft_frame_detail
    if cleaned == "practical_living_cost_pressure" and draft_frame_detail:
        return draft_frame_detail
    return cleaned


def _strip_prefix(value: str) -> str:
    text = str(value or "").strip()
    for prefix in (
        "korean_daily_",
        "daily_",
        "draft_frame_detail_",
    ):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _is_practical_guidance_reason(reason: str) -> bool:
    if _is_hypothetical_choice_reason(reason):
        return False
    return _has_any(
        reason,
        "emergency",
        "first_steps",
        "first_action",
        "first_step",
        "first_purchase",
        "first_sentence",
        "draft_first",
        "how_to",
        "practical_",
        "productivity_",
        "specialized_",
        "basic_diet",
        "menu_pick",
        "meal_invite",
        "weather_check",
        "weather_chat",
        "dusty_weather_chat",
        "snow_aesthetic_safety",
        "safe_way_home",
        "exercise_habit_routine",
        "forgetfulness_practical",
        "money_",
        "recovery",
        "judgment_quit",
        "body_",
        "playful_short_attention_start",
        "playful_bed_human_reboot",
        "playful_plan_first_box",
        "weekend_hibernation_limit",
        "work_overload_pause_or_push",
        "work_ambiguous_task_boundary_line",
        "work_meeting_slip_repair_sentence",
        "relationship_one_sided_contact_boundary",
        "relationship_group_chat_flow_check",
        "relationship_contact_frequency_anxiety",
        "ai_no_generation_high_context_tradeoff",
        "ai_frame_axis_risk",
        "ai_rules_as_silver_not_engine",
        "ai_draft_frame_shadow_gate",
        "ai_context_signal_priority",
        "logic_rest_anxiety_plan_ratio",
        "choice_exercise_rest_guilt",
        "choice_efficiency_regret_cutoff",
        "preference_coffee_routine_taper",
        "preference_weekend_home_out_compromise",
    )


def _is_situational_tactic_reason(reason: str) -> bool:
    if _is_hypothetical_choice_reason(reason):
        return False
    return _has_any(
        reason,
        "conflict",
        "relationship",
        "group_chat",
        "new_person_charm",
        "read_receipt",
        "work_",
        "foundation_",
        "logic_anger_reason_boundary_explain",
        "logic_content_tone_separation",
        "logic_gut_feeling_no_evidence",
        "preference_rain_home_or_out",
    )


def _is_practical_priority_reason(reason: str) -> bool:
    if _is_hypothetical_choice_reason(reason):
        return False
    return _is_practical_guidance_reason(reason) or _has_any(
        reason,
        "work_",
        "read_receipt",
        "foundation_",
        "conflict",
        "relationship_kakao_tone_anxiety_check",
        "relationship_repeated_apology_boundary",
        "relationship_grievance_low_start",
        "logic_anger_reason_boundary_explain",
        "logic_content_tone_separation",
        "logic_gut_feeling_no_evidence",
        "preference_rain_home_or_out",
    )


def _is_emotion_stabilization_reason(reason: str) -> bool:
    if _has_any(reason, "body_"):
        return False
    if _is_basic_emotional_checkin_reason(reason):
        return True
    return _has_any(
        reason,
        "emotion_",
        "comfort",
        "grief",
        "panic",
        "loneliness",
        "stress",
        "slump",
        "hard_day",
        "mental_",
        "counsel_",
        "burnout",
        "disappointment_support",
        "anxiety_support",
        "irritability_support",
        "reassurance_received",
    )


def _is_basic_social_ritual_reason(reason: str) -> bool:
    return _has_any(
        reason,
        "basic_good_morning",
        "basic_how_are_you",
        "basic_weekend_checkin",
        "basic_long_time_no_see",
        "basic_appearance_compliment",
        "basic_cheer_received",
        "basic_late_apology_reassurance",
        "basic_time_thanks",
        "basic_coffee_invite",
        "basic_cafe_dessert_reaction",
        "basic_cozy_restaurant_reaction",
    )


def _is_basic_emotional_checkin_reason(reason: str) -> bool:
    return _has_any(reason, "basic_tired_state", "basic_concern_checkin")


def _is_reflective_definition_reason(reason: str) -> bool:
    return _has_any(reason, "basic_happiness_definition")


def _is_basic_preference_reason(reason: str) -> bool:
    return _has_any(reason, "basic_after_home_routine", "basic_travel_wish")


def _is_hypothetical_choice_reason(reason: str) -> bool:
    return _has_any(
        reason,
        "desert_island",
        "protagonist",
        "time_machine",
        "superpower",
        "lottery",
        "one_month_abroad",
        "immortality",
        "zombie_escape",
    )


def _is_contextual_life_scene_reason(reason: str) -> bool:
    return _strip_prefix(reason).startswith("more_")


def _is_meta_content_reference_reason(reason: str) -> bool:
    return _has_any(
        reason,
        "meta_content_reference_guard",
        "meta_worry_word_reframed_as_song_earworm",
    )


def _family_for(reason: str, frame: str) -> str:
    if _is_meta_content_reference_reason(reason):
        return "context_disambiguation"
    if _is_basic_social_ritual_reason(reason):
        return "social_ritual"
    if _is_emotion_stabilization_reason(reason):
        return "emotional_support"
    if _is_practical_guidance_reason(reason):
        return "practical_guidance"
    if _is_situational_tactic_reason(reason):
        return "situational_tactic"
    if _is_reflective_definition_reason(reason) or _has_any(reason, "ai_", "logic", "philosophy", "learning", "core_values", "immortality", "historical", "career_", "values_"):
        return "reflective_position"
    if _has_any(reason, "identity", "memory_boundary", "no_fake", "honesty", "persona"):
        return "identity_boundary"
    if _is_basic_preference_reason(reason):
        return "choice_preference"
    if _is_hypothetical_choice_reason(reason):
        return "choice_preference"
    if _has_any(reason, "absurd", "playful", "meme", "lottery", "superpower", "desert_island", "protagonist"):
        return "playful_output" if "choice" not in reason else "choice_preference"
    if _has_any(frame, "choice", "preference", "icebreak", "time_machine", "one_month_abroad", "praise", "character_design"):
        return "choice_preference"
    if _is_contextual_life_scene_reason(reason):
        return "situational_context"
    return "social_acknowledgement"


def _domain_for(reason: str, *, draft_domain: str) -> str:
    if draft_domain:
        return draft_domain
    if _has_any(reason, "meta_worry_word_reframed_as_song_earworm"):
        return "attention_language"
    if _has_any(reason, "meta_content_reference_guard"):
        return "content_authoring"
    if _is_reflective_definition_reason(reason):
        return "life_reflection"
    if _is_basic_emotional_checkin_reason(reason):
        return "emotional_state"
    if _is_basic_social_ritual_reason(reason):
        if _has_any(reason, "coffee", "dessert", "restaurant"):
            return "food_lifestyle"
        return "social_relationship"
    if _has_any(reason, "basic_travel_wish"):
        return "hypothetical_values"
    if _is_hypothetical_choice_reason(reason):
        return "hypothetical_values"
    if _has_any(reason, "weather_chat", "weather_check", "dusty_weather_chat", "snow_aesthetic_safety"):
        return "weather_season"
    if _has_any(reason, "transport_destination"):
        return "transport_commute"
    if _has_any(reason, "more_weather"):
        return "weather_season"
    if _has_any(reason, "more_food"):
        return "food_lifestyle"
    if _has_any(reason, "more_social"):
        return "social_relationship"
    if _has_any(reason, "more_home"):
        return "home_life"
    if _has_any(reason, "more_growth"):
        return "self_growth"
    if _has_any(reason, "character_design"):
        return "character_design"
    if _has_any(reason, "safe_way_home"):
        return "daily_safety"
    if _has_any(reason, "new_hobby_preference", "hobby_preference"):
        return "hobby_lifestyle"
    if _has_any(reason, "body_", "sleep", "exercise", "stiffness", "fatigue"):
        return "body_state"
    if _has_any(
        reason,
        "food",
        "coffee",
        "dessert",
        "spicy",
        "restaurant",
        "cooking",
        "breakfast",
        "soul",
        "lunch",
        "dinner",
        "meal",
        "menu",
        "drink",
    ):
        return "food_lifestyle"
    if _has_any(reason, "music", "movie", "media", "youtube", "culture", "performance", "goods", "book"):
        return "media_culture"
    if _has_any(reason, "money", "lottery", "gas", "cost", "selling", "quote", "budget"):
        return "money_living"
    if _has_any(reason, "relationship", "conflict", "group_chat", "new_person", "hard_day", "conversation_style", "read_receipt"):
        return "social_relationship"
    if _has_any(reason, "ai_", "identity", "memory"):
        return "ai_companion"
    if _has_any(reason, "work", "school", "interview", "presentation", "deadline", "learning", "productivity", "career"):
        return "work_school"
    if _has_any(reason, "travel", "one_month_abroad", "time_machine", "historical", "superpower", "immortality"):
        return "hypothetical_values"
    if _has_any(reason, "stress", "slump", "emotion", "grief", "panic", "loneliness", "mental", "counsel"):
        return "emotional_state"
    return "daily_life"


def _schema_for(reason: str) -> str:
    if _is_meta_content_reference_reason(reason):
        return "context_disambiguation"
    if _is_no_fake_reason(reason):
        return "honesty_boundary"
    if _is_basic_social_ritual_reason(reason):
        return "social_ritual"
    if _is_emotion_stabilization_reason(reason):
        return "emotional_support"
    if _is_practical_guidance_reason(reason):
        return "practical_advice"
    if _is_situational_tactic_reason(reason):
        return "social_tactic"
    if _is_reflective_definition_reason(reason) or _has_any(reason, "core_values", "immortality", "historical", "ai_", "logic", "philosophy", "learning", "career_", "values_"):
        return "reflective_judgment"
    if _is_contextual_life_scene_reason(reason):
        return "contextual_reaction"
    if _is_basic_preference_reason(reason):
        return "preference_disclosure"
    if _has_any(reason, "choice", "vs", "time_machine", "superpower", "lottery", "desert_island", "one_month_abroad", "protagonist"):
        return "hypothetical_choice"
    if _has_any(reason, "preference", "praise", "persona", "character_design", "transport_destination"):
        return "preference_disclosure"
    if "icebreak" in reason:
        return "preference_disclosure"
    return "direct_reply"


def _priority_for(reason: str) -> str:
    if _is_meta_content_reference_reason(reason):
        return "meta_reflection"
    if _has_any(reason, "emergency", "first_steps", "wrong_transfer", "oil_fire", "water_damage", "double_dose"):
        return "immediate_action"
    if _has_any(reason, "first_sentence", "how_to", "first_action", "first_step", "first_purchase", "draft_first"):
        return "practical_action"
    if _is_emotion_stabilization_reason(reason):
        return "emotion_stabilization"
    if _is_practical_priority_reason(reason):
        return "practical_action"
    if _is_reflective_definition_reason(reason) or _has_any(reason, "ai_", "logic", "philosophy", "learning", "core_values", "historical", "immortality", "values_"):
        return "meta_reflection"
    return "choice_judgment"


def _is_no_fake_reason(reason: str) -> bool:
    return _has_any(
        reason,
        "no_fake",
        "recent_media",
        "restaurant_memory",
        "recent_travel",
        "recent_photo",
        "childhood_warm_memory",
        "adventurous_moment",
        "frequent_contact",
    )


def _is_anxious_reason(reason: str) -> bool:
    return _has_any(
        reason,
        "panic",
        "anxiety",
        "worried",
        "fear",
        "scared",
        "unsettled",
        "mental_anxiety",
        "fatigue_signal",
        "work_new_project_first_step",
        "work_ambiguous_task_boundary_line",
    )


def _emotion_for(reason: str, *, no_fake: bool) -> str:
    if no_fake:
        return "curious"
    if _is_anxious_reason(reason):
        return "anxious"
    if _has_any(reason, "grief", "loneliness", "hurt", "ignored", "sad"):
        return "hurt"
    if _has_any(reason, "stress", "slump", "hard_day", "perfectionism", "counsel_"):
        return "stressed"
    if _has_any(reason, "playful", "lottery", "superpower", "desert_island", "protagonist"):
        return "playful"
    return "curious"


def _state_hint_for(reason: str, *, priority: str, no_fake: bool) -> str:
    if no_fake:
        return "honesty_boundary"
    if _has_any(reason, "meta_worry_word_reframed_as_song_earworm"):
        return "word_sense_context"
    if _has_any(reason, "meta_content_reference_guard"):
        return "content_reference_context"
    if _is_basic_social_ritual_reason(reason):
        return "social_ritual"
    if priority in {"immediate_action", "practical_action"}:
        return "practical_focus"
    if priority == "emotion_stabilization":
        return "emotional_context"
    if priority == "meta_reflection":
        return "reflective_context"
    if _has_any(reason, "relationship", "conflict", "group_chat", "new_person"):
        return "relationship_context"
    return "light_social"


def _action_hint_for(reason: str, *, action: str) -> str:
    if _is_meta_content_reference_reason(reason):
        return "reframe_context"
    if _has_any(reason, "identity"):
        return "answer_identity"
    if _is_practical_priority_reason(reason):
        return "share_opinion"
    if action in {"answer_identity", "reply_request", "share_feeling", "share_opinion"}:
        return action
    if _has_any(reason, "comfort", "emotion_", "grief", "panic", "stress", "slump", "hard_day", "mental_", "counsel_"):
        return "share_feeling"
    return "share_opinion"


def _tone_for(reason: str, *, tone: str, no_fake: bool) -> str:
    if no_fake:
        return "grounded"
    if _is_meta_content_reference_reason(reason):
        return "steady"
    if _is_basic_social_ritual_reason(reason):
        return "warm_playful"
    if _is_practical_priority_reason(reason):
        return "steady"
    if _has_any(reason, "emotion_", "comfort", "grief", "loneliness", "hard_day", "slump", "mental_", "counsel_"):
        return "warm_steady"
    if _has_any(reason, "lottery", "superpower", "desert_island", "protagonist", "icebreak"):
        return "warm_playful"
    if tone and tone != "neutral":
        return tone
    return "steady"


def _coarse_intent_for(reason: str, *, action: str) -> str:
    if action == "reply_request":
        return "reply_request"
    if _is_meta_content_reference_reason(reason):
        return "context_disambiguation"
    if _has_any(reason, "emotion_", "comfort", "grief", "stress", "slump", "hard_day", "mental_", "counsel_"):
        return "smalltalk_feeling"
    return "smalltalk_opinion"


def _speech_act_for(reason: str) -> str:
    if _is_meta_content_reference_reason(reason):
        return "inform"
    if _is_basic_social_ritual_reason(reason):
        return "social_ritual"
    if _has_any(reason, "reaction", "acknowledgement"):
        return "inform"
    return "ask"


def _pragmatic_cues_for(
    *,
    reason: str,
    schema: str,
    frame: str,
    family: str,
    priority: str,
    no_fake: bool,
    advice_request: bool,
    choice_request: bool,
) -> tuple[str, ...]:
    cues = [schema, frame, family, priority]
    if no_fake:
        cues.append("no_fake")
    if advice_request:
        cues.append("advice_request")
    if choice_request:
        cues.append("choice_request")
    if reason.startswith("korean_daily_icebreak_"):
        cues.append("icebreak")
    if _is_meta_content_reference_reason(reason):
        cues.extend(("false_positive_guard", "not_life_event", "content_reference"))
        if _has_any(reason, "meta_worry_word_reframed_as_song_earworm"):
            cues.append("earworm_reframe")
        else:
            cues.append("content_authoring_context")
    return tuple(cues)


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles if needle)
