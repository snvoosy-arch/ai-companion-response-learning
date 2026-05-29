from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

from predictive_bot.core.context_reference import (
    infer_contextual_choice_profile,
    infer_contextual_discourse_profile,
    infer_contextual_reference_profile,
    infer_mixed_topic_profile,
)
from predictive_bot.core.draft_word_senses import (
    WordSenseContext,
    build_word_sense_context_from_texts,
    resolve_word_senses,
)
from predictive_bot.core.models import (
    ClassifierEvidence,
    ConversationState,
    Intent,
    MeaningPacket,
    MeaningSignal,
    MessageFeatures,
    ScoredLabel,
)


@dataclass(frozen=True, slots=True)
class _SenseSignals:
    raw: str
    compact: str


class MeaningResolver:
    """Combines coarse model labels and text evidence into Black's meaning packet."""

    _general_activity_request_patterns = (
        r"(?:오늘|지금|이번엔|이따|주말)?(?:은|는)?\s*(?:무엇|뭐|머|뭘|어떤)\s*(?:하(?:고|면서)|하면서|하고서)?\s*(?:놀래|놀까|놀자|놀면|할래|할까|하지|하면|보낼까)",
        r"(?:(?:뭐|머)\s*하(?:고|면서)|뭐하면서|뭐하고|머하면서|머하고).*(?:놀래|놀까|놀자|시간\s*보낼까|보낼까|할래|할까)?",
        r"(?:오늘|지금|이따|주말).*(?:뭐\s*하지|뭐하지|뭐\s*할까|뭐할까|머\s*할까|머할까|뭐\s*할래|뭐할래|머\s*할래|머할래)",
    )
    _concrete_topic_terms = (
        "동물원",
        "사파리",
        "아쿠아리움",
        "호랑이",
        "사자",
        "판다",
        "코끼리",
        "상어",
        "돌고래",
        "수달",
        "펭귄",
        "물개",
        "학교",
        "회사",
        "공원",
        "캠핑장",
    )
    _conversation_topic_request_patterns = (
        r".*(?:대화|얘기|이야기)\s*할?\s*(?:주제|거리).*(?:생각해\s*봐|생각해봐|추천|골라|정해|던져|말해|뽑아|잡아)",
        r".*(?:주제|대화거리|얘깃거리).*(?:아무거나|하나|몇\s*개|몇개).*(?:생각|추천|말해|던져|뽑아|잡아)",
        r".*(?:아무\s*)?주제로\s*(?:대화|얘기|이야기).*(?:해\s*봐|해봐|하자|시작|열어|이어)",
        r".*(?:대화|얘기|이야기).*(?:아무거나|아무\s*주제|주제\s*하나).*(?:해\s*봐|해봐|하자|시작|말해|던져)",
        r".*(?:무슨|어떤).*(?:대화|얘기|이야기).*(?:할까|하지|해볼까)",
        r".*(?:대화|얘기|이야기).*(?:뭐로|무슨\s*주제|어떤\s*주제).*(?:할까|하지|가볼까)",
    )
    _activity_preparation_patterns = (
        r".*(?:할\s*때|갈\s*때|하려면|가려면|가기\s*전|하기\s*전).*(?:필요|준비물|챙|가져갈|필수품).*(?:말해|알려|정리|추천)?",
        r".*(?:필요|준비물|챙|가져갈|필수품).*(?:말해|알려|정리|추천)",
        r".*(?:뭐|무엇|뭘).*(?:필요|준비|챙겨|가져가).*(?:돼|해|좋)",
    )
    _activity_preparation_markers = (
        "등산",
        "산행",
        "캠핑",
        "바다",
        "해변",
        "계곡",
        "여행",
        "운동",
        "러닝",
        "수영",
        "낚시",
        "피크닉",
        "자전거",
    )

    def __init__(self, *, heuristic: Any) -> None:
        self.heuristic = heuristic

    def resolve(
        self,
        *,
        text: str,
        state: ConversationState,
        heuristic_features: MessageFeatures,
        model_features: MessageFeatures | None = None,
    ) -> MessageFeatures:
        base = model_features or heuristic_features
        signals = self._signals(
            heuristic_features=heuristic_features,
            model_features=model_features,
            normalized=heuristic_features.normalized,
        )
        schema = heuristic_features.question_schema or base.question_schema
        slots: dict[str, str] = {}
        domain: str | None = None
        speech_act_override: str | None = None
        if model_features is not None and model_features.meaning_packet is not None:
            domain = model_features.meaning_packet.domain
            if not schema and model_features.meaning_packet.schema:
                schema = model_features.meaning_packet.schema
            slots.update(model_features.meaning_packet.slots)

        contextual_profile = self._mixed_topic_profile(
            heuristic_features.normalized,
            heuristic_features.is_question or base.is_question,
        )
        if contextual_profile is None:
            contextual_profile = self._contextual_statement_profile(
                heuristic_features.normalized,
                heuristic_features.is_question or base.is_question,
            )
        if contextual_profile is None:
            contextual_profile = self._contextual_choice_profile(
                heuristic_features.normalized,
                heuristic_features.is_question or base.is_question,
                self._recent_user_texts_from_state(state),
            )
        if contextual_profile is None:
            contextual_profile = self._contextual_discourse_profile(
                heuristic_features.normalized,
                heuristic_features.is_question or base.is_question,
                self._recent_user_texts_from_state(state),
            )
        if contextual_profile is None:
            contextual_profile = self._contextual_reference_profile(
                heuristic_features.normalized,
                heuristic_features.is_question or base.is_question,
                self._recent_user_texts_from_state(state),
            )
        if contextual_profile is None:
            contextual_profile = self._contextual_word_sense_profile(
                heuristic_features.normalized,
                heuristic_features.is_question or base.is_question,
                self._word_sense_context_from_state(state),
            )
        if contextual_profile is not None:
            domain = str(contextual_profile["domain"])
            schema = str(contextual_profile["schema"])
            speech_act_override = str(contextual_profile.get("speech_act") or "inform")
            slots.update(contextual_profile.get("slots") or {})
            evidence_hits = list(contextual_profile.get("evidence") or [])
            signals.extend(
                [
                    MeaningSignal(
                        axis="domain",
                        label=domain,
                        confidence=0.96,
                        source="lexical_context_bridge",
                        evidence=evidence_hits,
                    ),
                    MeaningSignal(
                        axis="schema",
                        label=schema,
                        confidence=0.96,
                        source="lexical_context_bridge",
                        evidence=evidence_hits,
                    ),
                    MeaningSignal(
                        axis="speech_act",
                        label=speech_act_override,
                        confidence=0.94,
                        source="lexical_context_bridge",
                        evidence=evidence_hits,
                    ),
                    MeaningSignal(
                        axis="coarse_intent",
                        label=str(contextual_profile["intent"]),
                        confidence=0.93,
                        source="lexical_context_bridge",
                        evidence=evidence_hits,
                    ),
                ]
            )
            for axis, confidence in (
                ("emotion", 0.86),
                ("state_hint", 0.88),
                ("action_hint", 0.88),
                ("draft_frame_family", 0.87),
                ("draft_frame", 0.9),
                ("word_sense", 0.9),
            ):
                label = contextual_profile.get(axis)
                if label:
                    signals.append(
                        MeaningSignal(
                            axis=axis,
                            label=str(label),
                            confidence=confidence,
                            source="lexical_context_bridge",
                            evidence=evidence_hits,
                        )
                    )

        if self._is_proactive_checkin_request(heuristic_features.normalized):
            schema = "proactive_checkin"
            slots.update({"request": "proactive_checkin"})
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="proactive_checkin",
                    confidence=0.92,
                    source="schema_bridge",
                    evidence=["proactive_checkin_request"],
                )
            )

        if self._is_conversation_topic_suggestion_request(heuristic_features.normalized):
            schema = "conversation_topic_suggestion"
            slots.update(self._conversation_topic_slots())
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="conversation_topic_suggestion",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["conversation_topic_suggestion_request"],
                )
            )

        if self._is_activity_preparation_advice_request(heuristic_features.normalized):
            schema = "activity_preparation_advice"
            slots.update(self._activity_preparation_slots(heuristic_features.normalized))
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="activity_preparation_advice",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["activity_preparation_advice_request"],
                )
            )

        if self._is_practical_mood_refresh_request(heuristic_features.normalized):
            schema = "activity_recommendation"
            slots.update({"request": "practical_mood_refresh", "activity": "mood_refresh"})
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="activity_recommendation",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["practical_mood_refresh_request"],
                )
            )

        if self._is_transport_destination_preference_question(heuristic_features.normalized):
            domain = "general"
            schema = "preference_disclosure"
            slots.update(
                {
                    "request": "transport_destination_preference",
                    "transport": self._transport_destination_preference_transport(
                        heuristic_features.normalized
                    ),
                }
            )
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="preference_disclosure",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["transport_destination_preference_question"],
                )
            )

        if self._is_food_lifestyle_comparison_request(heuristic_features.normalized):
            domain = "food_lifestyle"
            schema = "preference_disclosure"
            slots.update({"request": "food_lifestyle_comparison", **self._food_lifestyle_comparison_slots(heuristic_features.normalized)})
            signals.append(
                MeaningSignal(
                    axis="domain",
                    label="food_lifestyle",
                    confidence=0.88,
                    source="schema_bridge",
                    evidence=["food_lifestyle_comparison_request"],
                )
            )
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="preference_disclosure",
                    confidence=0.86,
                    source="schema_bridge",
                    evidence=["food_lifestyle_comparison_request"],
                )
            )

        if self._is_light_food_recommendation_request(heuristic_features.normalized):
            domain = "food_lifestyle"
            schema = "light_food_recommendation"
            slots.update({"request": "light_food_recommendation", "food_style": "light"})
            signals.append(
                MeaningSignal(
                    axis="domain",
                    label="food_lifestyle",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["light_food_recommendation_request"],
                )
            )
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="light_food_recommendation",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["light_food_recommendation_request"],
                )
            )

        daily_priority_text = text or heuristic_features.content or heuristic_features.normalized
        daily_priority_schema = (
            "hypothetical_choice"
            if heuristic_features.question_schema == "hypothetical_choice"
            else self._daily_companion_priority_schema(
                daily_priority_text,
                heuristic_features.is_question or base.is_question,
            )
        )
        if daily_priority_schema is not None:
            schema = daily_priority_schema
            domain = "hypothetical" if daily_priority_schema == "hypothetical_choice" else "general"
            if daily_priority_schema == "habit_preference":
                speech_act_override = "inform"
            elif daily_priority_schema == "hypothetical_choice":
                speech_act_override = "ask"
            slots = self._daily_companion_priority_slots(
                daily_priority_text,
                daily_priority_schema,
            )
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label=daily_priority_schema,
                    confidence=0.92,
                    source="schema_bridge",
                    evidence=["daily_companion_priority"],
                )
            )
            signals.append(
                MeaningSignal(
                    axis="domain",
                    label=domain,
                    confidence=0.88,
                    source="schema_bridge",
                    evidence=["daily_companion_priority"],
                )
            )
            if speech_act_override is not None:
                signals.append(
                    MeaningSignal(
                        axis="speech_act",
                        label=speech_act_override,
                        confidence=0.93,
                        source="schema_bridge",
                        evidence=["daily_companion_priority"],
                    )
                )

        daily_practical_route = None
        if daily_priority_schema is None:
            daily_practical_route = self._daily_practical_priority_route(daily_priority_text)
        if daily_practical_route is not None:
            domain = daily_practical_route["domain"]
            schema = daily_practical_route["schema"]
            speech_act_override = "ask"
            slots = {
                "request": schema,
                "daily_practical_topic": daily_practical_route["topic"],
                "schema": schema,
            }
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label=schema,
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["daily_practical_priority"],
                )
            )
            signals.append(
                MeaningSignal(
                    axis="domain",
                    label=domain,
                    confidence=0.88,
                    source="schema_bridge",
                    evidence=["daily_practical_priority"],
                )
            )

        if schema == "memory_boundary" or "memory_boundary" in heuristic_features.pragmatic_cues:
            domain = "memory_context"
            schema = "memory_boundary"
            slots.update({"request": "memory_boundary"})
            signals.append(
                MeaningSignal(
                    axis="domain",
                    label="memory_context",
                    confidence=0.88,
                    source="schema_bridge",
                    evidence=["memory_boundary_request"],
                )
            )
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="memory_boundary",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["memory_boundary_request"],
                )
            )

        if self._is_concrete_topic_question(
            heuristic_features.normalized,
            heuristic_features.is_question or base.is_question,
        ):
            schema = "concrete_topic_question"
            slots.update(self._concrete_topic_slots(heuristic_features.normalized))
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="concrete_topic_question",
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["concrete_topic_question"],
                )
            )

        persona_question_schema = self._daily_persona_question_schema(
            heuristic_features.normalized,
            heuristic_features.is_question or base.is_question,
        )
        persona_bridge_blocked_schemas = {
            "activity_recommendation",
            "activity_preparation_advice",
            "conversation_topic_suggestion",
            "concrete_topic_question",
            "process_advice",
            "soft_decision_advice",
            "budget_reflection",
            "habit_support",
            "safety_boundary",
            "light_food_recommendation",
            "memory_boundary",
            "relationship_boundary",
            "story_summary_reaction",
            "long_form_story_share",
            "proactive_checkin",
        }
        if (
            daily_priority_schema is None
            and daily_practical_route is None
            and domain != "food_lifestyle"
            and schema not in persona_bridge_blocked_schemas
            and persona_question_schema is not None
        ):
            persona_slots = self._daily_persona_question_slots(
                heuristic_features.normalized,
                persona_question_schema,
            )
            schema = persona_question_schema
            if persona_slots:
                slots = {}
            slots.update({"request": persona_question_schema, **persona_slots})
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label=persona_question_schema,
                    confidence=0.9,
                    source="schema_bridge",
                    evidence=["daily_persona_question"],
                )
            )
            if persona_slots:
                slot_signal_label = "open_topic_preference" if "topic" in persona_slots else "season_place_activity_preference"
                signals.append(
                    MeaningSignal(
                        axis="slots",
                        label=slot_signal_label,
                        confidence=0.86,
                        source="slot_bridge",
                        evidence=list(persona_slots.keys()),
                    )
                )

        if daily_practical_route is None and persona_question_schema is None and self._is_general_activity_recommendation_question(
            heuristic_features.normalized,
            heuristic_features.is_question or base.is_question,
        ):
            schema = "activity_recommendation"
            slots.update(self._activity_slots(heuristic_features.normalized))
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label="activity_recommendation",
                    confidence=0.88,
                    source="schema_bridge",
                    evidence=["general_play_activity_question"],
                )
            )

        packet = MeaningPacket(
            coarse_intent=base.intent.value,
            domain=domain,
            schema=schema,
            speech_act=speech_act_override or base.speech_act,
            slots=slots,
            pragmatic_cues=list(dict.fromkeys([*heuristic_features.pragmatic_cues, *base.pragmatic_cues])),
            signals=signals,
        )

        if contextual_profile is not None:
            return self._rebuild_contextual_statement(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
                profile=contextual_profile,
            )

        if schema == "activity_recommendation" and (
            base.question_schema != "activity_recommendation" or base.intent != Intent.SMALLTALK_OPINION
        ):
            return self._rebuild_activity_recommendation(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
            )
        if schema == "concrete_topic_question" and (
            base.question_schema != "concrete_topic_question" or base.intent != Intent.SMALLTALK_OPINION
        ):
            return self._rebuild_concrete_topic_question(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
            )
        if schema == "proactive_checkin" and (
            base.question_schema != "proactive_checkin" or base.intent != Intent.SMALLTALK_GENERIC
        ):
            return self._rebuild_proactive_checkin(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
            )
        if schema == "conversation_topic_suggestion" and (
            base.question_schema != "conversation_topic_suggestion" or base.intent != Intent.SMALLTALK_OPINION
        ):
            return self._rebuild_conversation_topic_suggestion(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
            )
        if schema == "activity_preparation_advice" and (
            base.question_schema != "activity_preparation_advice" or base.intent != Intent.SMALLTALK_OPINION
        ):
            return self._rebuild_activity_preparation_advice(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
            )
        if daily_practical_route is not None:
            return self._rebuild_daily_practical_priority(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
                domain=daily_practical_route["domain"],
                schema=daily_practical_route["schema"],
            )
        if schema == "hypothetical_choice" and (
            base.question_schema != "hypothetical_choice" or base.intent != Intent.SMALLTALK_OPINION
        ):
            return self._rebuild_hypothetical_choice(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
            )
        if schema == "habit_preference" and (speech_act_override == "inform" or packet.speech_act == "inform"):
            return self._rebuild_habit_preference_statement(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
            )
        if schema in {"self_style", "habit_preference", "preference_disclosure"} and (
            base.question_schema != schema or base.intent != Intent.SMALLTALK_OPINION
        ):
            return self._rebuild_persona_question(
                base=base,
                heuristic_features=heuristic_features,
                state=state,
                packet=packet,
                schema=schema,
            )

        return replace(base, meaning_packet=packet)

    def _rebuild_activity_recommendation(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_OPINION
        sentiment = "neutral"
        speech_act = "ask"
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    "activity_recommendation",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit="meaning_bridge:activity_recommendation.general_play_question",
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema="activity_recommendation",
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                schema="activity_recommendation",
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_concrete_topic_question(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_OPINION
        sentiment = "neutral"
        speech_act = "ask"
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    "concrete_topic_question",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit="meaning_bridge:concrete_topic_question",
            schema="concrete_topic_question",
            chosen_reason=(
                "coarse model output was combined with raw-text topic evidence; "
                "the turn asks about a concrete topic rather than Black's personal preference"
            ),
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema="concrete_topic_question",
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                schema="concrete_topic_question",
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_proactive_checkin(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_GENERIC
        sentiment = "neutral"
        speech_act = "ask"
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    "proactive_checkin",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit="meaning_bridge:proactive_checkin",
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema="proactive_checkin",
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                schema="proactive_checkin",
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_conversation_topic_suggestion(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_OPINION
        sentiment = "neutral"
        speech_act = "ask"
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    "conversation_topic_suggestion",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit="meaning_bridge:conversation_topic_suggestion",
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema="conversation_topic_suggestion",
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                schema="conversation_topic_suggestion",
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_activity_preparation_advice(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_OPINION
        sentiment = "neutral"
        speech_act = "ask"
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    "activity_preparation_advice",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit="meaning_bridge:activity_preparation_advice",
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema="activity_preparation_advice",
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                schema="activity_preparation_advice",
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_persona_question(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
        schema: str,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_OPINION
        sentiment = "neutral"
        speech_act = "ask"
        cue_by_schema = {
            "self_style": "opinion_self_style",
            "habit_preference": "opinion_habit_preference",
            "preference_disclosure": "opinion_preference_like",
        }
        cue = cue_by_schema.get(schema, schema)
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    cue,
                    schema,
                ]
            )
        )
        if packet.domain == "food_lifestyle":
            pragmatic_cues = [
                cue
                for cue in pragmatic_cues
                if cue not in {"activity_preparation_advice", "activity_recommendation"}
            ]
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit=f"meaning_bridge:daily_persona_question.{schema}",
            schema=schema,
            chosen_reason=(
                "coarse model output was combined with raw-text schema evidence; "
                "the turn asks Black's style, habit, or preference rather than a new task"
            ),
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema=schema,
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                schema=schema,
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_daily_practical_priority(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
        domain: str,
        schema: str,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_OPINION
        sentiment = "neutral"
        speech_act = "ask"
        cue_by_schema = {
            "process_advice": "opinion_advice_process",
            "soft_decision_advice": "opinion_decision_request",
            "budget_reflection": "budget_reflection",
            "habit_support": "habit_support",
            "activity_preparation_advice": "activity_preparation_advice",
            "safety_boundary": "safety_boundary",
            "activity_recommendation": "activity_recommendation",
        }
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    cue_by_schema.get(schema, schema),
                    schema,
                    "daily_practical_priority",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit=f"meaning_bridge:daily_practical_priority.{schema}",
            schema=schema,
            chosen_reason=(
                "daily practical bridge recognized an everyday action request; "
                "use one concrete next step instead of a generic decision reply"
            ),
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema=schema,
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                domain=domain,
                schema=schema,
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_hypothetical_choice(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_OPINION
        sentiment = "neutral"
        speech_act = "ask"
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    "hypothetical_choice",
                    "opinion_preference_like",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit="meaning_bridge:daily_companion_priority.hypothetical_choice",
            schema="hypothetical_choice",
            chosen_reason=(
                "daily companion bridge recognized a low-stakes hypothetical choice; "
                "answer as a concrete preference rather than routing to a specialist topic"
            ),
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema="hypothetical_choice",
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                domain="hypothetical",
                schema="hypothetical_choice",
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_habit_preference_statement(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
    ) -> MessageFeatures:
        intent = Intent.SMALLTALK_FEELING
        sentiment = "neutral"
        speech_act = "inform"
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    *heuristic_features.pragmatic_cues,
                    *base.pragmatic_cues,
                    "habit_preference",
                    "opinion_habit_preference",
                    "daily_habit_statement",
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = self._bridge_evidence(
            base=base,
            heuristic_features=heuristic_features,
            bridge_hit="meaning_bridge:daily_companion_priority.habit_statement",
            schema="habit_preference",
            chosen_reason=(
                "daily companion bridge recognized a user habit/preference statement; "
                "acknowledge the user's tendency instead of continuing with a generic stock reply"
            ),
        )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema="habit_preference",
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                domain="general",
                schema="habit_preference",
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    def _rebuild_contextual_statement(
        self,
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        state: ConversationState,
        packet: MeaningPacket,
        profile: dict[str, object],
    ) -> MessageFeatures:
        intent = Intent(str(profile["intent"]))
        domain = str(profile["domain"])
        schema = str(profile["schema"])
        sentiment = str(profile.get("sentiment") or "neutral")
        speech_act = str(profile.get("speech_act") or "inform")
        profile_cues = [str(cue) for cue in profile.get("cues") or []]
        blocked_cues = set(profile.get("blocked_cues") or [])
        pragmatic_cues = list(
            dict.fromkeys(
                [
                    cue
                    for cue in [
                        *heuristic_features.pragmatic_cues,
                        *base.pragmatic_cues,
                        *profile_cues,
                        schema,
                    ]
                    if cue and cue not in blocked_cues
                ]
            )
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=heuristic_features.topic_hint,
            sentiment=sentiment,
            location=heuristic_features.location,
            requests_external_fact=False,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        if (
            base.classifier_evidence is not None
            and base.classifier_evidence.source == "bert"
            and base.intent == intent
        ):
            evidence = base.classifier_evidence
        else:
            evidence = self._bridge_evidence(
                base=base,
                heuristic_features=heuristic_features,
                bridge_hit=f"meaning_bridge:lexical_context.{schema}",
                schema=schema,
                chosen_reason=(
                    "raw-text context evidence identified a body-state, transport, or observation statement; "
                    "keep that frame even if the coarse model proposed an invite or clarification route"
                ),
            )
        return replace(
            base,
            intent=intent,
            sentiment=sentiment,
            requests_external_fact=False,
            speech_act=speech_act,
            question_schema=schema,
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=replace(
                packet,
                coarse_intent=intent.value,
                domain=domain,
                schema=schema,
                speech_act=speech_act,
                pragmatic_cues=pragmatic_cues,
            ),
        )

    @staticmethod
    def _recent_user_texts_from_state(state: ConversationState) -> tuple[str, ...]:
        return tuple(
            turn.user_text
            for turn in state.recent_turns[-6:]
            if str(turn.user_text or "").strip()
        )

    @staticmethod
    def _word_sense_context_from_state(state: ConversationState) -> WordSenseContext | None:
        recent_texts = list(MeaningResolver._recent_user_texts_from_state(state))
        if not recent_texts:
            return None
        context = build_word_sense_context_from_texts(recent_texts)
        if not context.compact and not context.tags:
            return None
        return context

    @staticmethod
    def _mixed_topic_profile(
        normalized: str,
        is_question: bool,
    ) -> dict[str, object] | None:
        profile = infer_mixed_topic_profile(
            normalized,
            is_question=is_question,
        )
        if profile is None:
            return None
        if profile.schema == "mixed_social_emotion" and is_question and any(
            marker in normalized for marker in ("불안", "집착", "걸까")
        ):
            return None
        return {
            "domain": profile.domain,
            "schema": profile.schema,
            "intent": profile.intent,
            "sentiment": profile.sentiment,
            "speech_act": profile.speech_act,
            "cues": list(profile.cues),
            "slots": profile.slots,
            "evidence": list(profile.evidence),
            "state_hint": profile.state_hint,
            "action_hint": profile.action_hint,
            "draft_frame_family": "mixed_topic",
            "draft_frame": profile.draft_frame,
            "blocked_cues": {"activity_invite", "proposal_or_invite", "clarification"},
        }

    @staticmethod
    def _contextual_reference_profile(
        normalized: str,
        is_question: bool,
        recent_texts: tuple[str, ...],
    ) -> dict[str, object] | None:
        profile = infer_contextual_reference_profile(
            normalized,
            recent_texts,
            is_question=is_question,
        )
        if profile is None:
            return None
        return {
            "domain": profile.domain,
            "schema": profile.schema,
            "intent": profile.intent,
            "sentiment": profile.sentiment,
            "speech_act": profile.speech_act,
            "cues": list(profile.cues),
            "slots": profile.slots,
            "evidence": list(profile.evidence),
            "state_hint": profile.state_hint,
            "action_hint": profile.action_hint,
            "draft_frame_family": "contextual_reference",
            "draft_frame": profile.draft_frame,
            "blocked_cues": {"activity_invite", "proposal_or_invite", "clarification"},
        }

    @staticmethod
    def _contextual_choice_profile(
        normalized: str,
        is_question: bool,
        recent_texts: tuple[str, ...],
    ) -> dict[str, object] | None:
        profile = infer_contextual_choice_profile(
            normalized,
            recent_texts,
            is_question=is_question,
        )
        if profile is None:
            return None
        return {
            "domain": profile.domain,
            "schema": profile.schema,
            "intent": profile.intent,
            "sentiment": profile.sentiment,
            "speech_act": profile.speech_act,
            "cues": list(profile.cues),
            "slots": profile.slots,
            "evidence": list(profile.evidence),
            "state_hint": profile.state_hint,
            "action_hint": profile.action_hint,
            "draft_frame_family": "contextual_choice_reference",
            "draft_frame": profile.draft_frame,
            "blocked_cues": {"activity_invite", "proposal_or_invite", "clarification"},
        }

    @staticmethod
    def _contextual_discourse_profile(
        normalized: str,
        is_question: bool,
        recent_texts: tuple[str, ...],
    ) -> dict[str, object] | None:
        profile = infer_contextual_discourse_profile(
            normalized,
            recent_texts,
            is_question=is_question,
        )
        if profile is None:
            return None
        return {
            "domain": profile.domain,
            "schema": profile.schema,
            "intent": profile.intent,
            "sentiment": profile.sentiment,
            "speech_act": profile.speech_act,
            "cues": list(profile.cues),
            "slots": profile.slots,
            "evidence": list(profile.evidence),
            "state_hint": profile.state_hint,
            "action_hint": profile.action_hint,
            "draft_frame_family": "contextual_discourse",
            "draft_frame": profile.draft_frame,
            "blocked_cues": {"activity_invite", "proposal_or_invite", "clarification"},
        }

    @staticmethod
    def _is_context_only_word_sense(match: object) -> bool:
        matched_cues = tuple(getattr(match, "matched_cues", ()) or ())
        return bool(matched_cues) and all(
            str(cue).startswith(("context:", "context_phrase:")) for cue in matched_cues
        )

    @staticmethod
    def _looks_like_ambiguous_word_followup(compact: str) -> bool:
        return any(
            marker in compact
            for marker in (
                "답없",
                "왜이",
                "이상",
                "난리",
                "별로",
                "망",
                "애매",
                "힘들",
                "불편",
                "짜증",
                "미치",
                "장난아니",
                "감당안",
                "모르겠",
                "꼬였",
                "터질",
                "하얘",
                "멍",
                "무겁",
                "아프",
                "상태",
            )
        )

    @staticmethod
    def _contextual_word_sense_profile(
        normalized: str,
        is_question: bool,
        context: WordSenseContext | None,
    ) -> dict[str, object] | None:
        if is_question or context is None:
            return None
        raw = str(normalized or "")
        compact = re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+", "", raw).lower()
        if not compact:
            return None
        if not any(word in compact for word in ("머리", "배", "눈", "속", "말", "손", "발", "일", "차", "밤", "사과", "풀")):
            return None
        if not MeaningResolver._looks_like_ambiguous_word_followup(compact):
            return None

        def profile(
            *,
            domain: str,
            schema: str,
            intent: Intent,
            sentiment: str = "negative",
            speech_act: str = "complain",
            cues: list[str] | None = None,
            slots: dict[str, str] | None = None,
            evidence: list[str] | None = None,
            state_hint: str | None = None,
            action_hint: str | None = None,
            draft_frame: str | None = None,
        ) -> dict[str, object]:
            slot_values = slots or {}
            return {
                "domain": domain,
                "schema": schema,
                "intent": intent.value,
                "sentiment": sentiment,
                "speech_act": speech_act,
                "cues": cues or [],
                "slots": slot_values,
                "evidence": evidence or [],
                "state_hint": state_hint,
                "action_hint": action_hint,
                "draft_frame_family": "contextual_word_sense",
                "draft_frame": draft_frame,
                "word_sense": slot_values.get("word_sense"),
                "blocked_cues": {"activity_invite", "proposal_or_invite"},
            }

        signals = _SenseSignals(raw=raw, compact=compact)
        resolved = resolve_word_senses(signals, context=context)
        for match in resolved:
            if not MeaningResolver._is_context_only_word_sense(match):
                continue
            evidence = [f"word_sense:{match.word}.{match.sense}", *match.matched_cues]
            if match.sense == "body_head":
                return profile(
                    domain="health_routine",
                    schema="body_signal_interpretation",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["body_signal_interpretation", "physical_discomfort", "contextual_word_sense"],
                    slots={"body_signal": "headache", "word_sense": "body_head", "topic": "머리"},
                    evidence=evidence,
                    state_hint="physical_discomfort",
                    action_hint="share_feeling",
                    draft_frame="contextual_headache_state",
                )
            if match.sense == "hair_style":
                return profile(
                    domain="beauty_style",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["style_frustration", "personal_observation", "contextual_word_sense"],
                    slots={"topic": "머리 스타일", "word_sense": "hair_style"},
                    evidence=evidence,
                    state_hint="style_frustration",
                    action_hint="share_feeling",
                    draft_frame="contextual_hair_style_state",
                )
            if match.sense == "thinking_brain":
                return profile(
                    domain="thinking_state",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["mental_overload", "process_blocked", "contextual_word_sense"],
                    slots={"topic": "생각 정리", "word_sense": "thinking_brain"},
                    evidence=evidence,
                    state_hint="mental_overload",
                    action_hint="share_feeling",
                    draft_frame="contextual_thinking_state",
                )
            if match.sense == "body_stomach":
                body_signal = "hunger" if "hunger" in context.tags and "pain" not in context.tags else "stomach_discomfort"
                return profile(
                    domain="health_routine",
                    schema="body_signal_interpretation",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["body_signal_interpretation", "stomach_context", "contextual_word_sense"],
                    slots={"body_signal": body_signal, "word_sense": "body_stomach", "topic": "배"},
                    evidence=evidence,
                    state_hint="physical_discomfort" if body_signal != "hunger" else "hunger",
                    action_hint="share_feeling",
                    draft_frame="contextual_stomach_state",
                )
            if match.sense == "ship":
                return profile(
                    domain="travel_transport",
                    schema="personal_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["transport_experience", "personal_observation", "contextual_word_sense"],
                    slots={"topic": "배", "word_sense": "ship", "activity": "transport_experience"},
                    evidence=evidence,
                    state_hint="travel_context",
                    action_hint="share_feeling",
                    draft_frame="contextual_ship_travel_state",
                )
            if match.sense == "pear":
                return profile(
                    domain="food_lifestyle",
                    schema="personal_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["food_observation", "fruit_context", "contextual_word_sense"],
                    slots={"topic": "과일 배", "word_sense": "pear"},
                    evidence=evidence,
                    state_hint="food_context",
                    action_hint="share_feeling",
                    draft_frame="contextual_fruit_pear_state",
                )
            if match.sense == "eye":
                return profile(
                    domain="health_routine",
                    schema="body_signal_interpretation",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["body_signal_interpretation", "eye_strain", "contextual_word_sense"],
                    slots={"body_signal": "eye_strain", "word_sense": "eye", "topic": "눈"},
                    evidence=evidence,
                    state_hint="physical_discomfort",
                    action_hint="share_feeling",
                    draft_frame="contextual_eye_state",
                )
            if match.sense == "snow":
                return profile(
                    domain="weather_season",
                    schema="reflective_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["weather_observation", "snow_context", "contextual_word_sense"],
                    slots={"topic": "눈 날씨", "word_sense": "snow"},
                    evidence=evidence,
                    state_hint="weather_observation",
                    action_hint="share_feeling",
                    draft_frame="contextual_snow_weather_state",
                )
            if match.sense == "gaze_attention":
                return profile(
                    domain="social_relationship",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["social_awareness", "gaze_context", "contextual_word_sense"],
                    slots={"topic": "시선", "word_sense": "gaze_attention"},
                    evidence=evidence,
                    state_hint="social_self_consciousness",
                    action_hint="share_feeling",
                    draft_frame="contextual_gaze_social_state",
                )
            if match.sense == "stomach":
                return profile(
                    domain="health_routine",
                    schema="body_signal_interpretation",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["body_signal_interpretation", "stomach_context", "contextual_word_sense"],
                    slots={"body_signal": "stomach_discomfort", "word_sense": "stomach", "topic": "속"},
                    evidence=evidence,
                    state_hint="physical_discomfort",
                    action_hint="share_feeling",
                    draft_frame="contextual_stomach_state",
                )
            if match.sense == "inner_emotion":
                return profile(
                    domain="emotional_state",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["inner_emotion", "emotional_pressure", "contextual_word_sense"],
                    slots={"topic": "속마음", "word_sense": "inner_emotion"},
                    evidence=evidence,
                    state_hint="inner_emotional_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_inner_emotion_state",
                )
            if match.sense == "speed":
                return profile(
                    domain="performance_digital",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["performance_frustration", "speed_context", "contextual_word_sense"],
                    slots={"topic": "속도", "word_sense": "speed"},
                    evidence=evidence,
                    state_hint="performance_frustration",
                    action_hint="share_feeling",
                    draft_frame="contextual_speed_state",
                )
            if match.sense == "speech":
                return profile(
                    domain="communication_style",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["communication_friction", "speech_context", "contextual_word_sense"],
                    slots={"topic": "말/말투", "word_sense": "speech"},
                    evidence=evidence,
                    state_hint="communication_friction",
                    action_hint="share_feeling",
                    draft_frame="contextual_speech_state",
                )
            if match.sense == "time_end":
                return profile(
                    domain="daily_routine",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["schedule_pressure", "period_end_context", "contextual_word_sense"],
                    slots={"topic": "기간 말", "word_sense": "time_end"},
                    evidence=evidence,
                    state_hint="schedule_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_time_end_state",
                )
            if match.sense == "horse":
                return profile(
                    domain="animal_activity",
                    schema="personal_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["animal_context", "horse_context", "contextual_word_sense"],
                    slots={"topic": "말/승마", "word_sense": "horse"},
                    evidence=evidence,
                    state_hint="animal_activity_context",
                    action_hint="share_feeling",
                    draft_frame="contextual_horse_state",
                )
            if match.sense == "car_vehicle":
                return profile(
                    domain="car_life",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["car_context", "daily_frustration", "contextual_word_sense"],
                    slots={"topic": "차/운전", "word_sense": "car_vehicle"},
                    evidence=evidence,
                    state_hint="traffic_or_car_stress",
                    action_hint="share_feeling",
                    draft_frame="contextual_car_state",
                )
            if match.sense == "tea_drink":
                return profile(
                    domain="food_lifestyle",
                    schema="personal_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["tea_context", "drink_context", "contextual_word_sense"],
                    slots={"topic": "차/티", "word_sense": "tea_drink"},
                    evidence=evidence,
                    state_hint="drink_context",
                    action_hint="share_feeling",
                    draft_frame="contextual_tea_state",
                )
            if match.sense == "difference_gap":
                return profile(
                    domain="comparison_gap",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["comparison_gap", "contrast_context", "contextual_word_sense"],
                    slots={"topic": "차이/격차", "word_sense": "difference_gap"},
                    evidence=evidence,
                    state_hint="comparison_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_gap_state",
                )
            if match.sense == "body_hand":
                return profile(
                    domain="health_routine",
                    schema="body_signal_interpretation",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["body_signal_interpretation", "hand_context", "contextual_word_sense"],
                    slots={"body_signal": "hand_discomfort", "word_sense": "body_hand", "topic": "손"},
                    evidence=evidence,
                    state_hint="physical_discomfort",
                    action_hint="share_feeling",
                    draft_frame="contextual_hand_state",
                )
            if match.sense == "customer_service":
                return profile(
                    domain="service_work",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["service_stress", "customer_context", "contextual_word_sense"],
                    slots={"topic": "손님 응대", "word_sense": "customer_service"},
                    evidence=evidence,
                    state_hint="service_stress",
                    action_hint="share_feeling",
                    draft_frame="contextual_customer_service_state",
                )
            if match.sense == "loss":
                return profile(
                    domain="money_spending",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["loss_frustration", "money_context", "contextual_word_sense"],
                    slots={"topic": "손해", "word_sense": "loss"},
                    evidence=evidence,
                    state_hint="loss_frustration",
                    action_hint="share_feeling",
                    draft_frame="contextual_loss_state",
                )
            if match.sense == "body_foot":
                return profile(
                    domain="health_routine",
                    schema="body_signal_interpretation",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["body_signal_interpretation", "foot_context", "contextual_word_sense"],
                    slots={"body_signal": "foot_discomfort", "word_sense": "body_foot", "topic": "발"},
                    evidence=evidence,
                    state_hint="physical_discomfort",
                    action_hint="share_feeling",
                    draft_frame="contextual_foot_state",
                )
            if match.sense == "presentation":
                return profile(
                    domain="work_school",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["presentation_pressure", "performance_context", "contextual_word_sense"],
                    slots={"topic": "발표", "word_sense": "presentation"},
                    evidence=evidence,
                    state_hint="performance_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_presentation_state",
                )
            if match.sense == "work_task":
                return profile(
                    domain="work_school",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["work_pressure", "task_context", "contextual_word_sense"],
                    slots={"topic": "업무/일", "word_sense": "work_task"},
                    evidence=evidence,
                    state_hint="work_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_work_task_state",
                )
            if match.sense == "event_happening":
                return profile(
                    domain="life_event",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["event_confusion", "situation_context", "contextual_word_sense"],
                    slots={"topic": "상황/사건", "word_sense": "event_happening"},
                    evidence=evidence,
                    state_hint="event_confusion",
                    action_hint="share_feeling",
                    draft_frame="contextual_event_state",
                )
            if match.sense == "night_time":
                return profile(
                    domain="sleep_routine",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["night_context", "sleep_pressure", "contextual_word_sense"],
                    slots={"topic": "밤/새벽", "word_sense": "night_time"},
                    evidence=evidence,
                    state_hint="late_night_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_night_state",
                )
            if match.sense == "chestnut_food":
                return profile(
                    domain="food_lifestyle",
                    schema="personal_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["food_observation", "chestnut_context", "contextual_word_sense"],
                    slots={"topic": "밤/군밤", "word_sense": "chestnut_food"},
                    evidence=evidence,
                    state_hint="food_context",
                    action_hint="share_feeling",
                    draft_frame="contextual_chestnut_state",
                )
            if match.sense == "apology":
                return profile(
                    domain="social_relationship",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["apology_context", "relationship_repair", "contextual_word_sense"],
                    slots={"topic": "사과/화해", "word_sense": "apology"},
                    evidence=evidence,
                    state_hint="relationship_repair_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_apology_state",
                )
            if match.sense == "apple":
                return profile(
                    domain="food_lifestyle",
                    schema="personal_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["food_observation", "apple_context", "contextual_word_sense"],
                    slots={"topic": "사과/과일", "word_sense": "apple"},
                    evidence=evidence,
                    state_hint="food_context",
                    action_hint="share_feeling",
                    draft_frame="contextual_apple_state",
                )
            if match.sense == "solve":
                return profile(
                    domain="problem_solving",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["problem_solving", "study_context", "contextual_word_sense"],
                    slots={"topic": "풀이/해결", "word_sense": "solve"},
                    evidence=evidence,
                    state_hint="problem_solving_pressure",
                    action_hint="share_feeling",
                    draft_frame="contextual_solve_state",
                )
            if match.sense == "relax":
                return profile(
                    domain="emotional_state",
                    schema="comfort_request",
                    intent=Intent.SMALLTALK_FEELING,
                    cues=["emotional_release", "relax_context", "contextual_word_sense"],
                    slots={"topic": "기분 풀기", "word_sense": "relax"},
                    evidence=evidence,
                    state_hint="emotional_release_needed",
                    action_hint="share_feeling",
                    draft_frame="contextual_relax_state",
                )
            if match.sense == "glue":
                return profile(
                    domain="craft_life",
                    schema="personal_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["craft_context", "glue_context", "contextual_word_sense"],
                    slots={"topic": "풀/접착제", "word_sense": "glue"},
                    evidence=evidence,
                    state_hint="craft_context",
                    action_hint="share_feeling",
                    draft_frame="contextual_glue_state",
                )
            if match.sense == "grass":
                return profile(
                    domain="nature_observation",
                    schema="reflective_observation",
                    intent=Intent.SMALLTALK_GENERIC,
                    sentiment="neutral",
                    speech_act="inform",
                    cues=["nature_observation", "grass_context", "contextual_word_sense"],
                    slots={"topic": "풀/잔디", "word_sense": "grass"},
                    evidence=evidence,
                    state_hint="nature_observation",
                    action_hint="share_feeling",
                    draft_frame="contextual_grass_state",
                )
        return None

    @staticmethod
    def _contextual_statement_profile(normalized: str, is_question: bool) -> dict[str, object] | None:
        raw = str(normalized or "")
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", raw).lower()
        if not compact:
            return None
        if is_question and MeaningResolver._is_light_food_recommendation_request(raw):
            return None

        def profile(
            *,
            domain: str,
            schema: str,
            intent: Intent,
            sentiment: str = "neutral",
            speech_act: str = "inform",
            cues: list[str] | None = None,
            slots: dict[str, str] | None = None,
            evidence: list[str] | None = None,
            emotion: str | None = None,
            state_hint: str | None = None,
            action_hint: str | None = None,
            draft_frame_family: str | None = None,
            draft_frame: str | None = None,
        ) -> dict[str, object]:
            return {
                "domain": domain,
                "schema": schema,
                "intent": intent.value,
                "sentiment": sentiment,
                "speech_act": speech_act,
                "cues": cues or [],
                "slots": slots or {},
                "evidence": evidence or [],
                "emotion": emotion,
                "state_hint": state_hint,
                "action_hint": action_hint,
                "draft_frame_family": draft_frame_family,
                "draft_frame": draft_frame,
                "blocked_cues": {"activity_invite", "proposal_or_invite"},
            }

        if any(
            marker in compact
            for marker in (
                "배가고프",
                "배고프",
                "배고파",
                "배가고파",
                "배고픔",
                "속이빈",
                "허기",
                "출출",
                "아침을안먹",
                "점심을안먹",
                "끼니를걸",
            )
        ):
            return profile(
                domain="health_routine",
                schema="body_signal_interpretation",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                cues=["body_signal_interpretation", "hunger_body_signal"],
                slots={"body_signal": "hunger"},
                evidence=["hunger_or_empty_stomach_statement"],
            )

        if any(marker in compact for marker in ("배가아프", "복통", "속이아프", "속이안좋", "속이불편", "메스꺼", "체한")):
            return profile(
                domain="health_routine",
                schema="body_signal_interpretation",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["body_signal_interpretation", "physical_discomfort"],
                slots={"body_signal": "stomach_discomfort"},
                evidence=["stomach_or_abdominal_discomfort_statement"],
            )

        if any(marker in compact for marker in ("목이좀칼칼", "목이칼칼", "목이따끔", "목이아프")):
            return profile(
                domain="health_routine",
                schema="body_signal_interpretation",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                cues=["body_signal_interpretation", "physical_discomfort"],
                slots={"body_signal": "throat_discomfort"},
                evidence=["throat_discomfort_statement"],
            )

        if any(
            marker in compact
            for marker in (
                "목말라",
                "목마르",
                "목이말라",
                "목이마르",
                "갈증",
                "물이마시고싶",
                "물마시고싶",
            )
        ):
            return profile(
                domain="health_routine",
                schema="body_signal_interpretation",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                cues=["body_signal_interpretation", "thirst_body_signal"],
                slots={"body_signal": "thirst"},
                evidence=["thirst_statement"],
            )

        if any(marker in compact for marker in ("머리가살짝아프", "머리가살짝아파", "머리가아프", "머리아파", "두통")):
            return profile(
                domain="health_routine",
                schema="body_signal_interpretation",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["body_signal_interpretation", "physical_discomfort"],
                slots={"body_signal": "headache"},
                evidence=["headache_statement"],
            )

        if any(marker in compact for marker in ("배가좀나왔", "배가나왔", "뱃살", "배가나온것")):
            return profile(
                domain="health_routine",
                schema="body_signal_interpretation",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                cues=["body_signal_interpretation", "body_shape_observation"],
                slots={"body_signal": "body_shape_observation"},
                evidence=["body_shape_observation_statement"],
            )

        if any(
            marker in compact
            for marker in (
                "기운이별로없",
                "기운이없",
                "기운없",
                "몸이축처진",
                "몸이축처지",
                "몸이무겁",
                "몸이무거워",
                "피곤",
                "졸려",
                "졸리",
                "졸림",
                "컨디션별로",
                "컨디션이별로",
            )
        ):
            return profile(
                domain="health_routine",
                schema="low_energy_support",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["low_energy_support", "body_signal_interpretation"],
                slots={"body_signal": "low_energy"},
                evidence=["low_energy_statement"],
            )

        if any(marker in compact for marker in ("몸이으슬으슬", "으슬으슬하다", "감기기운")):
            return profile(
                domain="health_routine",
                schema="body_signal_interpretation",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                cues=["body_signal_interpretation", "physical_discomfort"],
                slots={"body_signal": "cold_chills"},
                evidence=["cold_chills_statement"],
            )

        sleep_context = any(
            marker in compact
            for marker in (
                "잠잘때",
                "잘때",
                "자려고",
                "자려는데",
                "자려고누웠",
                "잠들려고",
                "잠자려고",
                "자는데",
                "자는중",
                "잠을못자",
                "잠못자",
                "잠을설치",
                "잠설치",
                "자기전",
                "누웠는데",
                "밤마다",
                "밤에",
                "새벽마다",
                "새벽에",
            )
        )
        noise_complaint = any(
            marker in compact
            for marker in (
                "시끄럽",
                "시끄러",
                "소음",
                "쿵쾅",
                "쿵쿵",
                "떠들",
                "떠드",
                "고성방가",
                "발망치",
                "층간소음",
                "벽간소음",
                "오토바이소리",
                "배달소리",
                "차소리",
                "문쾅",
                "키보드소리",
                "알람소리",
            )
        )
        benign_sleep_sound = any(
            marker in compact
            for marker in (
                "백색소음",
                "수면음악",
                "asmr",
                "빗소리",
                "장작타는소리",
            )
        )
        if sleep_context and noise_complaint and not benign_sleep_sound:
            return profile(
                domain="sleep_routine",
                schema="practical_advice",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["sleep_noise_environment", "sleep_disruption", "practical_advice"],
                slots={"topic": "sleep_noise", "environment_issue": "noise"},
                evidence=["sleep_noise_environment_statement"],
                emotion="stressed",
                state_hint="practical_focus",
                action_hint="share_feeling",
                draft_frame_family="practical_guidance",
                draft_frame="sleep_noise_environment",
            )

        gas_stove_ignition_context = any(
            marker in compact
            for marker in (
                "가스레인지",
                "가스렌지",
                "가스버너",
                "버너",
                "화구",
            )
        ) and any(
            marker in compact
            for marker in (
                "점화장치",
                "점화",
                "불이안붙",
                "불안붙",
                "불이안켜",
                "불안켜",
                "불꽃이안",
                "딸깍",
            )
        ) and any(
            marker in compact
            for marker in (
                "문제",
                "고장",
                "한쪽만",
                "안붙",
                "안켜",
                "안올라",
                "막혔",
            )
        )
        gas_stove_ignition_blocked = any(
            marker in compact
            for marker in (
                "가스냄새",
                "가스냄세",
                "누출",
                "환기",
                "119",
                "불꽃차단",
                "기름때",
                "청소",
                "닦여",
                "안닦",
                "디자인",
                "사진",
                "후기",
                "예뻐",
                "예쁜",
            )
        )
        if gas_stove_ignition_context and not gas_stove_ignition_blocked:
            return profile(
                domain="home_maintenance",
                schema="practical_advice",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["gas_stove_ignition_issue", "home_maintenance", "practical_advice"],
                slots={"topic": "gas_stove_ignition", "home_issue": "ignition"},
                evidence=["gas_stove_ignition_issue_statement"],
                emotion="stressed",
                state_hint="practical_focus",
                action_hint="share_feeling",
                draft_frame_family="practical_guidance",
                draft_frame="gas_stove_ignition_issue",
            )

        appliance_design_review_context = any(
            marker in compact
            for marker in (
                "가스레인지",
                "가스렌지",
                "가스버너",
                "버너",
                "화구",
                "가전",
                "제품",
                "주방가전",
            )
        ) and any(
            marker in compact
            for marker in (
                "디자인",
                "예뻐",
                "예쁜",
                "사진저장",
                "저장",
                "끌리",
                "취향",
            )
        ) and any(
            marker in compact
            for marker in (
                "후기",
                "리뷰",
                "평",
                "별로",
                "안좋",
                "고장",
                "점화장치",
                "점화",
                "성능",
                "내구성",
            )
        )
        appliance_design_review_blocked = any(
            marker in compact
            for marker in (
                "캐릭터",
                "웹툰",
                "일러스트",
                "굿즈",
                "피규어",
                "가스냄새",
                "가스냄세",
                "누출",
                "환기",
                "119",
                "불꽃차단",
                "불이안붙",
                "불안붙",
                "불이안켜",
                "불안켜",
                "불꽃이안",
                "딸깍거리기만",
                "기름때",
                "청소",
                "닦여",
                "안닦",
            )
        )
        if appliance_design_review_context and not appliance_design_review_blocked:
            return profile(
                domain="home_appliance",
                schema="practical_advice",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="negative",
                speech_act="inform",
                cues=["appliance_design_review_judgment", "purchase_judgment", "practical_advice"],
                slots={"topic": "appliance_purchase", "risk": "function_review"},
                evidence=["appliance_design_review_judgment_statement"],
                emotion="curious",
                state_hint="practical_focus",
                action_hint="share_opinion",
                draft_frame_family="practical_guidance",
                draft_frame="appliance_design_review_judgment",
            )

        heating_bill_context = any(
            marker in compact
            for marker in (
                "가스비",
                "도시가스비",
                "난방비",
                "공과금",
                "관리비",
                "전기요금",
                "전기세",
                "생활비",
            )
        ) and any(
            marker in compact
            for marker in (
                "보일러",
                "난방",
                "온수",
                "히터",
                "고지서",
                "이번달생활비",
            )
        )
        heating_cost_pressure = any(
            marker in compact
            for marker in (
                "무서",
                "불안",
                "부담",
                "겁나",
                "아끼",
                "꺼야하나",
                "켜기",
                "켜는게",
                "틀기",
                "버텨야",
                "고민",
                "올라",
                "신경쓰",
                "줄여야",
                "계산",
                "손이멈",
            )
        )
        gas_safety_context = any(marker in compact for marker in ("가스냄새", "가스냄세", "누출", "환기", "119"))
        if heating_bill_context and heating_cost_pressure and not gas_safety_context:
            return profile(
                domain="money_living",
                schema="practical_advice",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["heating_bill_anxiety", "utility_bill_pressure", "practical_advice"],
                slots={"topic": "heating_bill", "pressure": "utility_cost"},
                evidence=["heating_bill_anxiety_statement"],
                emotion="anxious",
                state_hint="practical_focus",
                action_hint="share_feeling",
                draft_frame_family="practical_guidance",
                draft_frame="heating_bill_anxiety",
            )

        living_cost_context = any(
            marker in compact
            for marker in (
                "기름값",
                "주유비",
                "휘발유값",
                "유가",
                "물가",
                "식비",
                "식료품값",
                "장보기",
                "마트",
                "주유소",
            )
        ) and any(
            marker in compact
            for marker in (
                "주유소",
                "주유",
                "기름",
                "마트",
                "장보",
                "장보기",
                "장바구니",
                "지갑",
                "예산",
                "식비",
            )
        )
        living_cost_pressure = any(
            marker in compact
            for marker in (
                "올라",
                "비싸",
                "무서",
                "불안",
                "아파",
                "흔들",
                "부담",
                "겁나",
                "줄여야",
                "아껴",
                "예산",
                "미루",
                "커져",
                "커지",
                "빼야",
                "뺄",
                "문제",
            )
        )
        living_cost_blocked = any(
            marker in compact
            for marker in (
                "가스냄새",
                "누출",
                "기름불",
                "불붙",
                "화재",
                "물가산책",
                "강가물가",
                "계곡물가",
            )
        )
        if living_cost_context and living_cost_pressure and not living_cost_blocked:
            return profile(
                domain="money_living",
                schema="practical_advice",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["living_cost_pressure", "budget_pressure", "practical_advice"],
                slots={"topic": "living_cost", "pressure": "budget"},
                evidence=["living_cost_pressure_statement"],
                emotion="anxious",
                state_hint="practical_focus",
                action_hint="share_feeling",
                draft_frame_family="practical_guidance",
                draft_frame="living_cost_pressure",
            )

        if any(marker in compact for marker in ("잠을거의못잤", "잠을못잤", "잠못잤", "잠을설쳤")):
            return profile(
                domain="health_routine",
                schema="low_energy_support",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["low_energy_support", "sleep_disruption"],
                slots={"body_signal": "poor_sleep"},
                evidence=["poor_sleep_statement"],
            )

        if not is_question and any(marker in compact for marker in ("기차를놓쳤", "길을잘못들었", "길을잘못들었다")):
            return profile(
                domain="general",
                schema="comfort_request",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="negative",
                speech_act="complain",
                cues=["travel_mishap", "personal_observation"],
                slots={"topic": "transport_mishap"},
                evidence=["transport_mishap_statement"],
            )

        if not is_question and (
            compact in {"배를탔다", "버스를탔다"}
            or re.search(r"(배|버스|지하철|기차|택시|비행기)(를|을)?탔", compact)
            or re.search(r"배타고.+(?:들어왔|들어왔다|도착했|왔다)", compact)
        ):
            vehicle = "배" if "배" in compact else "버스" if "버스" in compact else "이동"
            return profile(
                domain="general",
                schema="personal_observation",
                intent=Intent.SMALLTALK_GENERIC,
                cues=["personal_observation", "transport_experience"],
                slots={"topic": vehicle, "activity": "transport_experience"},
                evidence=["transport_experience_statement"],
            )

        if not is_question and any(
            marker in compact
            for marker in (
                "한국엔사계절이뚜렷",
                "한국에는사계절이뚜렷",
                "공기가좀차갑",
                "해가빨리지",
                "방이갑자기조용",
                "비온뒤냄새가좋",
                "도시불빛이생각보다예쁘",
                "한강에배한척이지나가",
            )
        ):
            domain = "aesthetic" if any(marker in compact for marker in ("예쁘", "냄새가좋", "공기가", "해가", "조용")) else "general"
            schema = "aesthetic_reflection" if domain == "aesthetic" else "reflective_observation"
            return profile(
                domain=domain,
                schema=schema,
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="positive" if any(marker in compact for marker in ("예쁘", "좋")) else "neutral",
                cues=[schema, "reflective_observation", "personal_observation"],
                slots={"topic": "simple_observation"},
                evidence=["simple_observation_statement"],
            )

        return None

    @classmethod
    def _is_conversation_topic_suggestion_request(cls, normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        if not any(marker in text for marker in ("대화", "얘기", "이야기", "주제", "대화거리", "얘깃거리")):
            return False
        return any(re.search(pattern, text) for pattern in cls._conversation_topic_request_patterns)

    @staticmethod
    def _is_proactive_checkin_request(normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        return (
            any(marker in text for marker in ("안부", "컨디션", "상태", "괜찮은지", "괜찮냐"))
            and any(marker in text for marker in ("확인", "물어", "봐줘", "한 줄", "한줄", "가볍게", "조용한"))
            and not any(marker in text for marker in ("사용자 발화", "직전", "금지", "규칙", "프롬프트"))
        )

    @staticmethod
    def _is_practical_mood_refresh_request(normalized: str) -> bool:
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(normalized or "")).lower()
        if not text:
            return False
        if not any(marker in text for marker in ("기분전환", "환기")):
            return False
        return any(
            marker in text
            for marker in (
                "하나만",
                "말해줘",
                "추천",
                "할만한거",
                "시간이안가",
                "시간이너무안가",
                "퇴근까지",
                "하교까지",
            )
        )

    @staticmethod
    def _is_transport_destination_preference_question(normalized: str) -> bool:
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(normalized or "")).lower()
        if not text:
            return False
        if not any(marker in text for marker in ("어디로", "어디가", "어디를")):
            return False
        if not any(marker in text for marker in ("가고싶", "떠나고싶", "갈래")):
            return False
        return any(marker in text for marker in ("지하철", "전철", "버스", "기차", "ktx", "비행기"))

    @staticmethod
    def _transport_destination_preference_transport(normalized: str) -> str:
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(normalized or "")).lower()
        for marker, value in (
            ("지하철", "지하철"),
            ("전철", "지하철"),
            ("ktx", "기차"),
            ("기차", "기차"),
            ("비행기", "비행기"),
            ("버스", "버스"),
        ):
            if marker in text:
                return value
        return "이동"

    @staticmethod
    def _is_light_food_recommendation_request(normalized: str) -> bool:
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(normalized or "")).lower()
        if not text:
            return False
        hunger_markers = ("배고프", "배고픈", "배고파", "배가고파", "배는고픈", "고픈데", "출출", "허기")
        light_markers = (
            "무거운건싫",
            "무겁지않",
            "기름진건피",
            "기름진건싫",
            "가볍게",
            "가벼운",
            "속편",
            "속이답답",
            "부담없는",
            "간단하게",
        )
        request_markers = ("뭐먹", "먹기좋", "먹는게좋", "먹을만한", "먹을만", "추천", "뭐있", "뭐가좋")
        return (
            any(marker in text for marker in hunger_markers)
            and any(marker in text for marker in light_markers)
            and any(marker in text for marker in request_markers)
        )

    @staticmethod
    def _is_food_lifestyle_comparison_request(normalized: str) -> bool:
        raw = str(normalized or "")
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", raw).lower()
        if not text:
            return False
        food_markers = (
            "김밥",
            "라면",
            "샌드위치",
            "우동",
            "죽",
            "계란밥",
            "국밥",
            "샐러드",
            "치킨",
            "떡볶이",
            "삼겹살",
            "피자",
            "햄버거",
            "점심",
            "저녁",
        )
        comparison_markers = ("중에", "이랑", "랑", "아니면", "vs", "덜무거", "가벼운", "부담")
        ask_markers = ("뭐가", "어느쪽", "뭘", "덜무거울까", "나을까", "좋을까", "고를까")
        return (
            sum(1 for marker in food_markers if marker in text) >= 2
            and any(marker in text for marker in comparison_markers)
            and any(marker in text for marker in ask_markers)
        )

    @staticmethod
    def _food_lifestyle_comparison_slots(normalized: str) -> dict[str, str]:
        text = str(normalized or "")
        foods = []
        for marker in ("김밥", "라면", "샌드위치", "우동", "죽", "계란밥", "국밥", "샐러드", "치킨", "떡볶이", "삼겹살", "피자", "햄버거"):
            if marker in text and marker not in foods:
                foods.append(marker)
        slots: dict[str, str] = {}
        if foods:
            slots["food_options"] = ",".join(foods[:4])
        if any(marker in text for marker in ("덜 무거", "덜무거", "가볍", "부담")):
            slots["food_criterion"] = "light"
        return slots

    @staticmethod
    def _daily_persona_question_schema(normalized: str, is_question: bool) -> str | None:
        raw_text = str(normalized or "")
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", raw_text).lower()
        if not text or not is_question:
            return None
        if MeaningResolver._is_practical_mood_refresh_request(text):
            return None
        if "날씨" in text and not any(marker in text for marker in ("좋아", "편이", "좋아하")):
            return None
        seasonal_slots = MeaningResolver._daily_persona_question_slots(raw_text)
        if seasonal_slots and seasonal_slots.get("preference_type"):
            if seasonal_slots["preference_type"] in {"feeling", "familiarity", "courage", "fear_or_comfort"}:
                return "habit_preference"
            return "preference_disclosure"
        open_slots = MeaningResolver._open_topic_preference_slots(raw_text)
        if open_slots and open_slots.get("preference_type"):
            if open_slots["preference_type"] in {"feeling", "familiarity"}:
                return "habit_preference"
            return "preference_disclosure"
        if any(
            marker in text
            for marker in (
                "커피",
                "차좋아",
                "음료",
                "간식",
                "빠진노래",
                "추천해주고싶은곡",
                "자주듣는노래",
                "음악장르",
                "자주듣는음악",
                "좋아하는계절",
                "좋아하는과일",
                "좋아하는동물",
                "색깔",
                "부먹",
                "찍먹",
                "비가오는날씨를좋아",
                "비오는날씨를좋아",
                "대화할때가장편안",
                "혼자만의시간",
                "절대타협할수없는",
                "중요한가치",
                "깊은관계",
                "핵심적인덕목",
                "변하지않을것",
                "본성",
                "작은행복",
            )
        ):
            return "preference_disclosure"
        if any(
            marker in text
            for marker in (
                "잠은잘자",
                "몇시에잤",
                "운동하고",
                "요리자주",
                "요리하는걸좋아",
                "자신있는메뉴",
                "시간 보내",
                "시간보내",
                "시간을보내",
                "읽고있는책",
                "읽고있는책",
                "게임하고",
                "게임하고",
                "늦잠",
                "배우고싶은",
                "배우고싶은",
                "일어나기힘들",
                "일어나기힘들",
                "가장먼저하는행동",
                "잠들기전",
                "아침형인간",
                "저녁형인간",
                "취미",
                "치유하고다시일어나는",
                "치유하고",
                "오해를받았을때",
                "해명하고",
                "시간이증명",
                "내면의평화",
                "평화를유지",
                "번아웃",
                "에너지를다시채워",
                "중대한결정",
                "이성과직관",
                "갈등이생겼을때",
                "풀어나가",
            )
        ):
            return "habit_preference"
        if any(
            marker in text
            for marker in (
                "점심",
                "메뉴로",
                "저녁",
                "기분",
                "스트레스",
                "스트레스를받을때",
                "하루 어땠",
                "하루어땠",
                "본 영화",
                "본영화",
                "드라마나영화",
                "보는 드라마",
                "보는드라마",
                "크게웃었던",
                "장래희망",
                "여행지",
                "좋은 일",
                "좋은일",
                "밖에 나갔다",
                "밖에나갔다",
                "새로 산",
                "새로산",
                "만족스러운소비",
                "가장맛있었던",
                "잘했다",
                "로또1등",
                "사고싶",
                "휴가",
                "하고싶",
                "한단어",
                "한문장",
                "신념",
                "삶의철학",
                "완벽한하루",
                "마지막순간",
                "단하나의목표",
                "가면을벗고",
                "솔직하고편안하게",
                "결핍",
                "어른이된다는것",
                "성공적인삶",
                "과거의나",
                "현재의나",
                "미래의나",
                "단점",
                "콤플렉스",
                "제약이전혀없다면",
                "모든걸멈추고",
                "인생을한권의소설책",
                "소설책",
                "기억되고싶",
                "일정",
                "누구 만났",
                "누구만났",
                "피곤",
                "연휴",
                "어디 가고 싶",
                "어디가고싶",
                "뭐 할 거야",
                "뭐할거야",
            )
        ):
            return "self_style"
        return None

    @staticmethod
    def _daily_companion_priority_schema(normalized: str, is_question: bool) -> str | None:
        raw_text = str(normalized or "")
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", raw_text).lower()
        if not text:
            return None
        if MeaningResolver._is_daily_hypothetical_choice_text(raw_text, is_question):
            return "hypothetical_choice"
        if MeaningResolver._is_daily_habit_preference_statement(raw_text):
            return "habit_preference"
        if is_question and MeaningResolver._is_daily_preference_disclosure_question(raw_text):
            return "preference_disclosure"
        return None

    @staticmethod
    def _daily_companion_priority_slots(normalized: str, schema: str) -> dict[str, str]:
        source = " ".join(str(normalized or "").strip().split())
        slots = MeaningResolver._open_topic_preference_slots(source)
        for key in ("request", "schema"):
            slots.pop(key, None)
        if schema == "habit_preference":
            preferred = MeaningResolver._extract_daily_preferred_side(source)
            if preferred:
                slots["topic"] = preferred
                slots["preference_type"] = "habit"
        elif schema == "hypothetical_choice":
            left, right = MeaningResolver._extract_daily_choice_options(source)
            if right:
                slots["topic"] = right
                slots["preference_type"] = "comparison_choice"
            if left and right:
                slots["options"] = f"{left}|{right}"
        elif schema == "preference_disclosure":
            preferred = MeaningResolver._extract_daily_preferred_side(source)
            if preferred:
                slots["topic"] = preferred
                slots["preference_type"] = "comparison_choice"
        slots["schema"] = schema
        return slots

    @staticmethod
    def _is_daily_preference_disclosure_question(raw_text: str) -> bool:
        source = " ".join(str(raw_text or "").strip().split())
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", source).lower()
        if not text:
            return False
        return (
            "보다" in text
            and "쪽이더편" in text
            and any(marker in text for marker in ("특이한가", "이상한가", "괜찮은가", "괜찮아"))
        )

    @staticmethod
    def _is_daily_habit_preference_statement(raw_text: str) -> bool:
        source = " ".join(str(raw_text or "").strip().split())
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", source).lower()
        if not text:
            return False
        return (
            ("나는" in text or "난" in text)
            and "늘" in text
            and "쪽으로가" in text
            and any(marker in text for marker in ("얘기만나오면", "말나오면", "나오면"))
        )

    @staticmethod
    def _is_daily_hypothetical_choice_text(raw_text: str, is_question: bool) -> bool:
        source = " ".join(str(raw_text or "").strip().split())
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", source).lower()
        if not text or not is_question:
            return False
        return (
            "만약" in text
            and "하나만골라야한다면" in text
            and "중" in text
            and any(marker in text for marker in ("뭐가덜질릴까", "뭐가나을까", "어느쪽이덜"))
        )

    @staticmethod
    def _extract_daily_preferred_side(raw_text: str) -> str:
        source = " ".join(str(raw_text or "").strip().split())
        patterns = (
            r"보다\s*(.+?)\s*쪽이\s*더\s*편",
            r"늘\s*(.+?)\s*쪽으로\s*가",
        )
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return match.group(1).strip(" .?!,")
        return ""

    @staticmethod
    def _extract_daily_choice_options(raw_text: str) -> tuple[str, str]:
        source = " ".join(str(raw_text or "").strip().split())
        vs_match = re.search(r"(.+?)\s*(?:vs|VS)\s*(.+?)(?:[?!.。]|$)", source)
        if vs_match:
            left = vs_match.group(1).strip(" .?!,")
            right = vs_match.group(2).strip(" .?!,")
            return left, right
        match = re.search(r"한다면\s*(.+?)(?:이랑|랑|와|과|하고)\s*(.+?)\s*중", source)
        if not match:
            return "", ""
        left = match.group(1).strip(" .?!,")
        right = match.group(2).strip(" .?!,")
        return left, right

    @staticmethod
    def _daily_practical_priority_route(raw_text: str) -> dict[str, str] | None:
        source = " ".join(str(raw_text or "").strip().split())
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", source).lower()
        if not text:
            return None
        if not any(marker in text for marker in ("지금뭐부터할까", "오늘은어디까지하면충분해")):
            return None

        def route(domain: str, schema: str, topic: str) -> dict[str, str]:
            return {"domain": domain, "schema": schema, "topic": topic}

        if any(marker in text for marker in ("자료정리", "발표준비", "회의참여", "메일답장", "마감직전", "공부시간")):
            return route("work_school", "process_advice", "work_school")
        if any(marker in text for marker in ("저녁메뉴", "아침식사", "야식고민", "카페선택")):
            return route("food_lifestyle", "soft_decision_advice", "food_lifestyle")
        if any(marker in text for marker in ("배달비", "충동구매", "가계부")):
            return route("money_lifestyle", "budget_reflection", "money_lifestyle")
        if any(marker in text for marker in ("수면리듬", "운동시작", "컨디션")):
            return route("health_routine", "habit_support", "health_routine")
        if any(marker in text for marker in ("밤산책", "등산계획", "비오는외출", "처음가는모임")):
            return route("general", "activity_preparation_advice", "activity_preparation")
        if any(marker in text for marker in ("택시귀가", "차량번호", "모르는링크")):
            return route("general", "safety_boundary", "safety_boundary")
        if "쉬는날" in text and any(marker in text for marker in ("돈은적게", "기분은바꾸")):
            return route("general", "activity_recommendation", "low_cost_refresh")
        return None

    @staticmethod
    def _daily_persona_question_slots(normalized: str, schema: str | None = None) -> dict[str, str]:
        raw_text = str(normalized or "")
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", raw_text).lower()
        if not text:
            return {}

        slots: dict[str, str] = {}

        def pick_all(pairs: tuple[tuple[str, str], ...]) -> str | None:
            matches: list[tuple[int, int, str]] = []
            occupied: list[tuple[int, int]] = []
            values: set[str] = set()
            for marker, value in pairs:
                start = text.find(marker)
                while start >= 0:
                    end = start + len(marker)
                    overlaps = any(not (end <= left or start >= right) for left, right in occupied)
                    if not overlaps and value not in values:
                        matches.append((start, end, value))
                        occupied.append((start, end))
                        values.add(value)
                        break
                    start = text.find(marker, start + 1)
            matches.sort(key=lambda item: (item[0], item[1]))
            return "|".join(value for _, _, value in matches) if matches else None

        season = pick_all(
            (
                ("봄", "봄"),
                ("벚꽃", "봄"),
                ("꽃냄새", "봄"),
                ("여름", "여름"),
                ("가을", "가을"),
                ("단풍", "가을"),
                ("겨울", "겨울"),
                ("눈내리", "겨울"),
                ("얼어붙", "겨울"),
            )
        )
        place = pick_all(
            (
                ("밤바다", "밤바다"),
                ("해안도로", "해안도로"),
                ("아쿠아리움", "아쿠아리움"),
                ("수족관", "수족관"),
                ("수영장", "수영장"),
                ("온수풀", "온수풀"),
                ("온천", "온천"),
                ("백사장", "해변"),
                ("해수욕장", "해변"),
                ("해변", "해변"),
                ("바닷속", "바다"),
                ("바다", "바다"),
                ("강변", "강변"),
                ("강물", "강"),
                ("강가", "강"),
                ("계곡", "계곡"),
                ("공원", "공원"),
                ("강", "강"),
            )
        )
        activity = pick_all(
            (
                ("스쿠버다이빙", "스쿠버 다이빙"),
                ("스쿠버", "스쿠버 다이빙"),
                ("다이빙", "스쿠버 다이빙"),
                ("드라이브", "드라이브"),
                ("자전거", "자전거"),
                ("물멍", "물멍"),
                ("물놀이", "물놀이"),
                ("잠수", "잠수"),
                ("수영", "수영"),
                ("낚시", "낚시"),
                ("산책", "산책"),
                ("걷", "걷기"),
                ("구경", "구경"),
            )
        )
        sensory = pick_all(
            (
                ("파도소리", "파도 소리"),
                ("소독약냄새", "소독약 냄새"),
                ("강물냄새", "강물 냄새"),
                ("꽃냄새", "꽃냄새"),
                ("고요함", "고요함"),
                ("칼바람", "칼바람"),
                ("바람", "바람"),
                ("햇빛", "햇빛"),
                ("장마", "장마"),
                ("습기", "습기"),
                ("단풍", "단풍"),
                ("벚꽃", "벚꽃"),
                ("정취", "정취"),
                ("비오는", "비"),
                ("비가", "비"),
                ("빗소리", "비"),
                ("눈", "눈"),
            )
        )
        obj = pick_all(
            (
                ("해산물", "해산물"),
                ("열대물고기", "열대 물고기"),
                ("관상용물고기", "관상용 물고기"),
                ("강물고기", "강 물고기"),
                ("바다물고기", "바다 물고기"),
                ("물고기요리", "물고기 요리"),
                ("물고기", "물고기"),
                ("빙어", "빙어"),
                ("갈매기", "갈매기"),
                ("철새", "철새"),
            )
        )

        if season:
            slots["season"] = season
        if place:
            slots["place"] = place
        if activity:
            slots["activity"] = activity
        if sensory:
            slots["sensory"] = sensory
        if obj:
            slots["object"] = obj
        if not slots:
            slots.update(MeaningResolver._open_topic_preference_slots(raw_text))

        if not slots:
            return {}

        if any(marker in text for marker in ("감정", "기분", "느낌이드", "어떤느낌", "분위기")):
            slots["preference_type"] = "feeling"
        elif any(marker in text for marker in ("익숙", "냄새중어느")):
            slots["preference_type"] = "familiarity"
        elif any(marker in text for marker in ("용기", "버킷리스트")):
            slots["preference_type"] = "courage"
        elif any(marker in text for marker in ("무서워", "꺼리는편", "즐기는편")):
            slots["preference_type"] = "fear_or_comfort"
        elif any(
            marker in text
            for marker in (
                "선호",
                "더끌리",
                "더매력",
                "어떤것이더",
                "중어느",
                "어느쪽",
                "어느곳",
                "어느계절",
                "고르",
                "선택",
                "vs",
                "편하",
                "견딜만",
                "어울린",
            )
        ):
            slots["preference_type"] = "comparison_choice"
        elif any(marker in text for marker in ("해보고싶", "하고싶", "마음이있", "상상", "보고싶")):
            slots["preference_type"] = "wish_or_imagination"
        elif any(marker in text for marker in ("좋아", "좋으", "좋은", "낭만", "정취")):
            slots["preference_type"] = "like"

        if schema:
            slots["schema"] = schema
        return slots

    @staticmethod
    def _open_topic_preference_slots(raw_text: str) -> dict[str, str]:
        source = " ".join(str(raw_text or "").strip().split())
        if not source:
            return {}
        text = re.sub(r"[^0-9A-Za-z가-힣]+", "", source).lower()
        if not text:
            return {}

        def clean_topic(value: str) -> str:
            topic = re.sub(r"^[\s,.'\"!?]+|[\s,.'\"!?]+$", "", value or "")
            topic = re.sub(r"^(나는|혹시|그럼|그러면|요즘|최근에|나중에|평소에|조용한|작은|새로운|무슨|어떤|어느)\s+", "", topic)
            topic = re.sub(r"\s*(을|를)?\s*(가까이서\s*)?보면$", "", topic)
            topic = re.sub(r"\s*하루$", "", topic)
            topic = re.sub(r"\s*(?:뭐|무엇|무슨|어떤|어느)\s*$", "", topic)
            topic = MeaningResolver._strip_open_topic_particle(topic)
            topic = topic.strip()
            if not topic:
                return ""
            blocked = {
                "뭐",
                "무엇",
                "어느",
                "어떤",
                "하나",
                "하나만",
                "둘",
                "둘중",
                "요즘",
                "최근",
                "평소",
                "굳이",
                "진짜",
                "혹시",
                "그거",
                "그것",
                "이거",
                "이것",
            }
            compact = re.sub(r"\s+", "", topic)
            if compact in blocked or (len(compact) < 2 and compact not in {"섬", "차", "향"}) or len(compact) > 18:
                return ""
            return topic

        def classify_topic(value: str) -> str:
            compact = re.sub(r"\s+", "", value)
            if compact.endswith(("카페", "공원", "도서관", "미술관", "박물관", "영화관", "해변", "시장", "골목", "섬")):
                return "place"
            return "topic"

        def pack(values: list[str]) -> str:
            packed: list[str] = []
            for value in values:
                cleaned = clean_topic(value)
                if cleaned and cleaned not in packed:
                    packed.append(cleaned)
            return "|".join(packed)

        slots: dict[str, str] = {}
        comparison_patterns = (
            r"(.+?)(이랑|랑|와|과|하고)\s*(.+?)\s*(?:중|중에|중에서)\s*(?:뭐|무엇|어느|어디|어떤|하나)",
            r"(.+?)\s*(vs|VS)\s*(.+?)\s*(?:중|중에|중에서|라면|이면)?",
        )
        for pattern in comparison_patterns:
            match = re.search(pattern, source)
            if not match:
                continue
            left = match.group(1)
            delimiter = match.group(2)
            right = match.group(3)
            if delimiter == "이랑" and not MeaningResolver._has_final_consonant(left):
                left = f"{left}이"
            values = pack([left, right])
            if values:
                label = "place" if all(classify_topic(value) == "place" for value in values.split("|")) else "topic"
                slots[label] = values
                slots["preference_type"] = "comparison_choice"
                return slots

        wish_match = re.search(r"(.+?)\s*(?:키워보고|먹어보고|마셔보고|써보고|해보고|가보고|사보고|배워보고|입어보고|들어보고|머물러보고|맡아보고|봐보고)\s*싶", source)
        if wish_match:
            topic = clean_topic(wish_match.group(1))
            if topic:
                slots[classify_topic(topic)] = topic
                slots["preference_type"] = "wish_or_imagination"
                return slots

        feeling_match = re.search(r"(.+?)\s*(?:어떤\s*느낌|어떤\s*기분|무슨\s*느낌)", source)
        if feeling_match:
            topic = clean_topic(feeling_match.group(1))
            if topic:
                slots[classify_topic(topic)] = topic
                slots["preference_type"] = "feeling"
                return slots

        like_match = re.search(r"(.+?)\s*(?:좋아(?:해|하|하는|하시|하니|하냐)?|끌려|괜찮아|매력적|편안해|편해|편안할\s*것\s*같아|편할\s*것\s*같아)", source)
        if like_match:
            topic = clean_topic(like_match.group(1))
            if topic:
                slots[classify_topic(topic)] = topic
                slots["preference_type"] = "like"
                return slots

        return {}

    @staticmethod
    def _strip_open_topic_particle(topic: str) -> str:
        cleaned = str(topic or "").strip()
        for particle in ("에서", "부터", "까지", "으로", "로", "에게", "한테", "은", "는", "가", "을", "를", "도", "만", "에"):
            if cleaned.endswith(particle) and len(re.sub(r"[^0-9A-Za-z가-힣]+", "", cleaned[: -len(particle)])) >= 2:
                return cleaned[: -len(particle)].strip()
        return cleaned

    @staticmethod
    def _has_final_consonant(text: str) -> bool:
        cleaned = re.sub(r"[^가-힣]+", "", str(text or ""))
        if not cleaned:
            return False
        code = ord(cleaned[-1])
        if not 0xAC00 <= code <= 0xD7A3:
            return False
        return (code - 0xAC00) % 28 != 0

    @staticmethod
    def _conversation_topic_slots() -> dict[str, str]:
        return {
            "request": "conversation_topic",
            "conversation_topic_focus": "대화 주제",
            "conversation_topic_options": "오늘 컨디션|요즘 본 영상|다음에 같이 해볼 것",
            "conversation_topic_first": "오늘 컨디션",
        }

    @classmethod
    def _is_activity_preparation_advice_request(cls, normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        if not any(marker in text for marker in cls._activity_preparation_markers):
            return False
        if not any(marker in text for marker in ("필요", "준비물", "챙", "가져갈", "가져가야", "필수품")):
            return False
        return any(re.search(pattern, text) for pattern in cls._activity_preparation_patterns)

    @classmethod
    def _activity_preparation_slots(cls, normalized: str) -> dict[str, str]:
        activity = "활동"
        for marker in cls._activity_preparation_markers:
            if marker in normalized:
                activity = marker
                break
        return {
            "request": "activity_preparation",
            "activity_name": activity,
        }

    @classmethod
    def _is_general_activity_recommendation_question(cls, normalized: str, is_question: bool) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        if not is_question and not any(
            marker in text
            for marker in (
                "뭐하지",
                "뭐 할까",
                "뭐할까",
                "머할까",
                "뭐 할래",
                "뭐할래",
                "머 할래",
                "머할래",
                "놀까",
                "놀래",
            )
        ):
            return False
        if re.search(r"(살지|살아야|인생|죽|망했|막막|모르겠다|모르겠어)", text):
            return False
        return any(re.search(pattern, text) for pattern in cls._general_activity_request_patterns)

    @classmethod
    def _is_concrete_topic_question(cls, normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        text = str(normalized or "")
        if not text:
            return False
        if not any(term in text for term in cls._concrete_topic_terms):
            return False
        if re.search(r"(?:있던가|있든가|있나|있어|있을까|볼\s*수\s*있|만날\s*수\s*있)", text):
            return True
        return bool(re.search(r"(?:에는|에|에서).*(?:있|보|만나).*(?:가|나|어|을까|던가|든가)(?:\?|$)", text))

    @classmethod
    def _concrete_topic_slots(cls, normalized: str) -> dict[str, str]:
        hits = [term for term in cls._concrete_topic_terms if term in normalized]
        return {
            "request": "concrete_topic_question",
            "topic": "|".join(dict.fromkeys(hits[:4])),
        }

    @staticmethod
    def _activity_slots(normalized: str) -> dict[str, str]:
        slots: dict[str, str] = {"request": "play_activity"}
        if "오늘" in normalized:
            slots["time"] = "오늘"
        elif "지금" in normalized:
            slots["time"] = "지금"
        elif "주말" in normalized:
            slots["time"] = "주말"
        return slots

    @staticmethod
    def _signals(
        *,
        heuristic_features: MessageFeatures,
        model_features: MessageFeatures | None,
        normalized: str,
    ) -> list[MeaningSignal]:
        signals: list[MeaningSignal] = [
            MeaningSignal(
                axis="coarse_intent",
                label=heuristic_features.intent.value,
                confidence=1.0,
                source="heuristic",
                evidence=list(heuristic_features.classifier_evidence.rule_hits)
                if heuristic_features.classifier_evidence
                else [],
            )
        ]
        if heuristic_features.question_schema:
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label=heuristic_features.question_schema,
                    confidence=0.92,
                    source="heuristic",
                    evidence=list(heuristic_features.pragmatic_cues),
                )
            )
        model_packet_axes: set[str] = set()
        if model_features is not None and model_features.meaning_packet is not None:
            model_packet_axes = {signal.axis for signal in model_features.meaning_packet.signals}
        model_coarse_signal_allowed = (
            model_features is not None
            and (
                model_features.meaning_packet is None
                or model_features.meaning_packet.resolver != "multihead_meaning_model_v1"
                or "coarse_intent" in model_packet_axes
            )
        )
        if (
            model_features is not None
            and model_features.classifier_evidence is not None
            and model_coarse_signal_allowed
        ):
            top_score = model_features.classifier_evidence.top_scores[0].score if model_features.classifier_evidence.top_scores else 0.0
            signals.append(
                MeaningSignal(
                    axis="coarse_intent",
                    label=model_features.intent.value,
                    confidence=float(top_score),
                    source=model_features.classifier_evidence.source,
                    evidence=[f"normalized={normalized}"],
                )
            )
        if model_features is not None and model_features.meaning_packet is not None:
            signals.extend(model_features.meaning_packet.signals)
        return signals

    @staticmethod
    def _bridge_evidence(
        *,
        base: MessageFeatures,
        heuristic_features: MessageFeatures,
        bridge_hit: str,
        schema: str = "activity_recommendation",
        chosen_reason: str | None = None,
    ) -> ClassifierEvidence:
        rule_hits: list[str] = []
        if heuristic_features.classifier_evidence is not None:
            rule_hits.extend(heuristic_features.classifier_evidence.rule_hits)
        if base.classifier_evidence is not None:
            rule_hits.extend(base.classifier_evidence.rule_hits)
        rule_hits.append(bridge_hit)
        top_scores: list[ScoredLabel] = []
        if base.classifier_evidence is not None:
            top_scores.extend(base.classifier_evidence.top_scores)
        top_scores.insert(0, ScoredLabel(label=f"schema:{schema}", score=0.88))
        return ClassifierEvidence(
            source="meaning_resolver",
            chosen_reason=chosen_reason or (
                "coarse model output was combined with raw-text schema evidence; "
                "the turn is a general play/activity recommendation question"
            ),
            rule_hits=list(dict.fromkeys(rule_hits)),
            top_scores=top_scores[:6],
            override_applied=True,
            fallback_source=base.classifier_evidence.source if base.classifier_evidence else None,
            fallback_intent=base.intent.value,
        )
