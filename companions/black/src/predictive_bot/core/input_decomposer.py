from __future__ import annotations

from dataclasses import dataclass, field
import re

from predictive_bot.core.models import (
    ActionType,
    ClauseUnit,
    ContextCue,
    ConversationState,
    EvidenceNode,
    MessageFeatures,
    PropositionUnit,
)


@dataclass(slots=True)
class InputDecomposition:
    clause_units: list[ClauseUnit] = field(default_factory=list)
    propositions: list[PropositionUnit] = field(default_factory=list)
    context_cues: list[ContextCue] = field(default_factory=list)
    evidence_nodes: list[EvidenceNode] = field(default_factory=list)
    active_grounding_topics: list[str] = field(default_factory=list)
    context_dependency_level: str = "low"
    duo_role_state: str | None = None


class InputDecomposer:
    """Extracts clause-level and proposition-level evidence for the current turn."""

    _negative_markers = (
        "없",
        "못",
        "안 ",
        "안.",
        "안?",
        "지치",
        "불편",
        "어색",
        "버겁",
        "힘들",
        "부담",
        "덜 ",
        "모자라",
    )
    _positive_markers = (
        "좋",
        "괜찮",
        "편하",
        "반갑",
        "안정",
        "재밌",
        "재미있",
    )
    _hedge_markers = ("같아", "같네", "듯", "수도", "아마", "왠지", "조금", "좀")
    _future_markers = ("내일", "나중", "곧", "될 것", "하려고")
    _past_markers = ("어제", "아까", "방금", "전엔", "끝나고", "다녀와서")
    _ongoing_markers = ("아직", "계속", "요즘", "계속해서", "남아", "안 빠져")
    _quiet_markers = ("조용", "말수", "짧게", "짧아도", "짧은", "톤이 내려", "저에너지")
    _aftereffect_markers = ("계속", "남아", "떠올라", "복기", "안 빠져", "잔상", "여운", "공기")
    _weather_condition_markers = ("날씨", "햇빛", "바람", "선선", "맑", "좋", "괜찮")
    _activity_decision_terms = (
        "배드민턴",
        "산책",
        "자전거",
        "러닝",
        "조깅",
        "피크닉",
        "테니스",
        "농구",
        "축구",
        "캠핑",
        "달리기",
        "수영",
        "물놀이",
        "커피",
        "밥",
        "라면",
        "스파게티",
        "파스타",
        "피자",
        "치킨",
        "떡볶이",
        "볶음밥",
        "영화",
        "보드게임",
        "사진",
        "운동",
        "바베큐",
        "바비큐",
        "고기",
        "구워먹",
        "불멍",
        "요리",
    )
    _activity_recommendation_places = (
        "바다",
        "해변",
        "해수욕장",
        "계곡",
        "공원",
        "한강",
        "산",
        "캠핑장",
        "놀이공원",
        "실내",
    )
    _stop_topics = {
        "나는",
        "내가",
        "오늘은",
        "요즘",
        "그냥",
        "정도",
        "편",
        "쪽",
        "집",
        "말수",
        "user",
        "activity",
        "context",
        "place",
        "premise",
        "activity_invite",
    }

    def decompose(
        self,
        *,
        features: MessageFeatures,
        state: ConversationState,
    ) -> InputDecomposition:
        text = self._normalize_text(features.content)
        clause_texts = self._split_clauses(text)
        clause_units: list[ClauseUnit] = []
        propositions: list[PropositionUnit] = []
        evidence_nodes: list[EvidenceNode] = []

        for index, clause_text in enumerate(clause_texts, start=1):
            clause_id = f"clause_{index}"
            clause = ClauseUnit(
                clause_id=clause_id,
                text=clause_text,
                clause_type=self._clause_type(clause_text),
                polarity=self._polarity(clause_text),
                certainty=self._certainty(clause_text),
                time_scope=self._time_scope(clause_text),
                connective=self._connective(clause_text),
            )
            clause_units.append(clause)
            evidence_nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_clause_{index}",
                    source="input_decomposer",
                    label="clause",
                    value=f"{clause.clause_type}:{clause.text}",
                    confidence=0.82,
                    derived_from=[clause_id],
                )
            )
            clause_props = self._extract_propositions(clause)
            propositions.extend(clause_props)
            for prop in clause_props:
                evidence_nodes.append(
                    EvidenceNode(
                        evidence_id=f"ev_{prop.proposition_id}",
                        source="input_decomposer",
                        label=f"proposition:{prop.kind}",
                        value=prop.value or prop.object or clause.text,
                        confidence=min(1.0, max(0.55, prop.weight)),
                        derived_from=[prop.source_clause_id, prop.proposition_id],
                    )
                )

        global_props = self._extract_global_activity_invite_props(text, clause_units)
        propositions.extend(global_props)
        for prop in global_props:
            evidence_nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_{prop.proposition_id}",
                    source="input_decomposer",
                    label=f"proposition:{prop.kind}",
                    value=prop.value or prop.object or text,
                    confidence=min(1.0, max(0.55, prop.weight)),
                    derived_from=[prop.source_clause_id, prop.proposition_id],
                )
            )

        context_cues = self._extract_context_cues(
            text=text,
            features=features,
            state=state,
            clause_units=clause_units,
            propositions=propositions,
        )
        for cue in context_cues:
            evidence_nodes.append(
                EvidenceNode(
                    evidence_id=f"ev_{cue.cue_id}",
                    source="context_cue",
                    label=f"cue:{cue.cue_type}",
                    value=cue.value,
                    confidence=cue.confidence,
                    derived_from=[
                        item
                        for item in [cue.source_clause_id, cue.source_proposition_id]
                        if item
                    ],
                )
            )

        return InputDecomposition(
            clause_units=clause_units,
            propositions=propositions,
            context_cues=context_cues,
            evidence_nodes=evidence_nodes,
            active_grounding_topics=self._active_grounding_topics(features, propositions, context_cues),
            context_dependency_level=self._context_dependency_level(
                text=text,
                state=state,
                context_cues=context_cues,
                clause_units=clause_units,
            ),
            duo_role_state=self._duo_role_state(state, context_cues),
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _split_clauses(self, text: str) -> list[str]:
        working = text
        for token in (" 근데 ", " 그런데 ", " 하지만 ", " 다만 ", " 그리고 "):
            working = working.replace(token, token.strip() + "|||")
        working = re.sub(r"(은데|는데|한데|인데|던데|지만)(\s+)", r"\1|||", working)
        working = re.sub(r"([.!?])\s+", r"\1|||", working)
        working = working.replace(",", "|||")
        parts = [part.strip(" ,") for part in working.split("|||") if part.strip(" ,")]
        return parts or [text]

    def _clause_type(self, clause_text: str) -> str:
        if clause_text.endswith("?") or re.search(r"(표현해줘|묘사해줘|설명해줘|그려줘|풀어줘)\.?$", clause_text):
            return "question"
        if any(marker in clause_text for marker in self._aftereffect_markers):
            return "aftereffect"
        if clause_text.startswith(("근데", "그런데", "하지만", "다만")) or re.search(r"(는데|지만)$", clause_text):
            return "contrast"
        return "statement"

    def _polarity(self, clause_text: str) -> str:
        lowered = clause_text.casefold()
        has_negative = any(marker in lowered for marker in self._negative_markers)
        has_positive = any(marker in lowered for marker in self._positive_markers)
        if has_negative and has_positive:
            return "mixed"
        if has_negative:
            return "negative"
        if has_positive:
            return "positive"
        return "neutral"

    def _certainty(self, clause_text: str) -> str:
        return "hedged" if any(marker in clause_text for marker in self._hedge_markers) else "certain"

    def _time_scope(self, clause_text: str) -> str:
        if any(marker in clause_text for marker in self._future_markers):
            return "future"
        if any(marker in clause_text for marker in self._past_markers):
            return "past"
        if any(marker in clause_text for marker in self._ongoing_markers):
            return "ongoing"
        return "present"

    @staticmethod
    def _connective(clause_text: str) -> str | None:
        if clause_text.startswith(("근데", "그런데")):
            return "turn"
        if clause_text.startswith("하지만") or clause_text.endswith("지만"):
            return "contrast"
        if clause_text.endswith("는데"):
            return "lead"
        if clause_text.startswith("그리고"):
            return "additive"
        return None

    def _extract_propositions(self, clause: ClauseUnit) -> list[PropositionUnit]:
        propositions: list[PropositionUnit] = []
        prop_index = 1

        def append_prop(kind: str, *, subject: str | None = None, obj: str | None = None, value: str | None = None, weight: float = 0.8) -> None:
            nonlocal prop_index
            propositions.append(
                PropositionUnit(
                    proposition_id=f"{clause.clause_id}_prop_{prop_index}",
                    kind=kind,
                    source_clause_id=clause.clause_id,
                    subject=subject,
                    object=obj,
                    value=value,
                    weight=weight,
                )
            )
            prop_index += 1

        def has_kind(kind: str) -> bool:
            return any(item.kind == kind for item in propositions)

        text = clause.text
        is_question = clause.clause_type == "question"
        if re.search(r"(좋아|좋은|좋다|좋아해)", text):
            append_prop(
                "preference",
                subject="user",
                obj=self._topic_near(text, r"(좋아|좋은|좋다|좋아해)"),
                value=text,
                weight=0.92,
            )
        if re.search(r"(없어|없다|없네|부족|모자라|비어)", text):
            append_prop(
                "lack",
                subject="user",
                obj=self._topic_near(text, r"(없어|없다|없네|부족|모자라|비어)"),
                value=text,
                weight=0.94,
            )
        if re.search(r"(싶어|싶다|원해|원한다|바라)", text):
            append_prop("desire", subject="user", value=text, weight=0.85)
        if re.search(r"(못|안 |어렵|힘들|부담|줄이면|낮추면)", text):
            append_prop("constraint", subject="user", value=text, weight=0.80)
        if re.search(r"(비교|보다|처럼|잘되는 거|잘 되는 거|씁쓸)", text):
            append_prop("comparison", subject="user", value=text, weight=0.84)
        if self._is_weather_premise_clause(text):
            append_prop(
                "weather_premise",
                subject="weather",
                obj="weather",
                value=text,
                weight=0.86,
            )
        activity_object = self._extract_activity_decision_object(text)
        if activity_object is not None:
            append_prop(
                "activity_candidate",
                subject="user",
                obj=activity_object,
                value=text,
                weight=0.92,
            )
        activity_place = self._extract_activity_recommendation_place(text)
        if self._is_activity_recommendation_question(text, is_question):
            append_prop(
                "activity_recommendation_question",
                subject="user",
                obj=f"{activity_place} 놀이" if activity_place else "놀거리",
                value=text,
                weight=0.92,
            )
            if activity_place is not None:
                append_prop(
                    "activity_place",
                    subject="place",
                    obj=activity_place,
                    value=activity_place,
                    weight=0.86,
                )
        if self._is_decision_request_clause(text) and not has_kind("decision_request"):
            append_prop(
                "decision_request",
                subject="user",
                value=text,
                weight=0.82,
            )
        if self._is_preference_like_question(text, is_question):
            append_prop(
                "preference_like_question",
                subject="user",
                obj=self._extract_question_topic(text),
                value=text,
                weight=0.90,
            )
        if self._is_habit_preference_question(text, is_question):
            append_prop(
                "habit_preference_question",
                subject="user",
                obj=self._extract_question_topic(text),
                value=text,
                weight=0.88,
            )
        if self._is_self_style_question(text, is_question):
            append_prop(
                "self_style_question",
                subject="user",
                value=text,
                weight=0.89,
            )
        if self._is_reflective_judgment_question(text, is_question):
            append_prop(
                "reflective_judgment_question",
                subject="user",
                value=text,
                weight=0.88,
            )
        if self._is_process_advice_question(text, is_question):
            append_prop(
                "process_advice_question",
                subject="user",
                value=text,
                weight=0.90,
            )
        if self._is_expressive_request(text, is_question):
            append_prop(
                "expressive_request",
                subject="user",
                obj=self._extract_expressive_request_topic(text),
                value=text,
                weight=0.86,
            )
        if self._is_reflective_observation_statement(text, is_question):
            append_prop(
                "reflective_observation_statement",
                subject="user",
                obj=self._extract_reflective_topic(text),
                value=text,
                weight=0.84,
            )
        if self._is_relational_interpretation_statement(text, is_question):
            append_prop(
                "relational_interpretation_statement",
                subject="user",
                obj=self._extract_reflective_topic(text),
                value=text,
                weight=0.88,
            )
        if self._is_comparative_reflection_statement(text, is_question):
            append_prop(
                "comparative_reflection_statement",
                subject="user",
                obj=self._extract_reflective_topic(text),
                value=text,
                weight=0.87,
            )
        if self._is_aesthetic_reflection_statement(text, is_question):
            append_prop(
                "aesthetic_reflection_statement",
                subject="user",
                obj=self._extract_reflective_topic(text),
                value=text,
                weight=0.86,
            )
        if self._is_reason_probe_question(text, is_question):
            append_prop(
                "reason_probe_question",
                subject="user",
                value=text,
                weight=0.88,
            )
        if self._is_soft_decision_question(text, is_question) and not has_kind("decision_request"):
            append_prop(
                "decision_request",
                subject="user",
                value=text,
                weight=0.82,
            )
        if self._is_soft_decision_question(text, is_question):
            append_prop(
                "soft_decision_question",
                subject="user",
                obj=activity_object or self._extract_question_topic(text),
                value=text,
                weight=0.86,
            )
        if any(marker in text for marker in self._aftereffect_markers):
            append_prop("aftereffect", subject="user", value=text, weight=0.88)
        if self._certainty(text) == "hedged":
            append_prop("uncertainty", subject="user", value=text, weight=0.72)
        if re.search(r"(나는|난 |내가|오늘은|요즘|집에|그냥)", text):
            append_prop("self_disclosure", subject="user", value=text, weight=0.68)
        if not propositions:
            append_prop("state", subject="user", value=text, weight=0.60)
        return propositions

    def _extract_context_cues(
        self,
        *,
        text: str,
        features: MessageFeatures,
        state: ConversationState,
        clause_units: list[ClauseUnit],
        propositions: list[PropositionUnit],
    ) -> list[ContextCue]:
        cues: list[ContextCue] = []
        kinds = {prop.kind for prop in propositions}

        def append_cue(
            cue_type: str,
            value: str,
            *,
            confidence: float = 0.8,
            clause_id: str | None = None,
            proposition_id: str | None = None,
        ) -> None:
            cue_id = f"cue_{len(cues) + 1}"
            cues.append(
                ContextCue(
                    cue_id=cue_id,
                    cue_type=cue_type,
                    value=value,
                    confidence=confidence,
                    source_clause_id=clause_id,
                    source_proposition_id=proposition_id,
                )
            )

        if "preference" in kinds and "lack" in kinds:
            append_cue("contrast_gap", "preferred thing is absent right now", confidence=0.93)
        if any(marker in text for marker in self._quiet_markers):
            first_quiet_prop = next((prop for prop in propositions if prop.kind in {"constraint", "self_disclosure"}), None)
            append_cue(
                "quiet_mode",
                "user is signaling lower-energy or shorter-tempo conversation",
                confidence=0.87,
                proposition_id=first_quiet_prop.proposition_id if first_quiet_prop else None,
            )
        if "aftereffect" in kinds:
            aftereffect_prop = next((prop for prop in propositions if prop.kind == "aftereffect"), None)
            append_cue(
                "aftereffect_hold",
                "current feeling likely depends on a previous social or emotional event",
                confidence=0.90,
                proposition_id=aftereffect_prop.proposition_id if aftereffect_prop else None,
            )
        if "uncertainty" in kinds:
            uncertain_prop = next((prop for prop in propositions if prop.kind == "uncertainty"), None)
            append_cue(
                "hedged_disclosure",
                "user is disclosing softly rather than asking directly",
                confidence=0.74,
                proposition_id=uncertain_prop.proposition_id if uncertain_prop else None,
            )
        if "relationship_check" in features.pragmatic_cues:
            clause_id = clause_units[-1].clause_id if clause_units else None
            append_cue(
                "relationship_check",
                "current turn explicitly checks the emotional or relational state",
                confidence=0.92,
                clause_id=clause_id,
            )
        question_schema_map = {
            "preference_like_question": "opinion_preference_like",
            "habit_preference_question": "opinion_habit_preference",
            "self_style_question": "opinion_self_style",
            "reflective_judgment_question": "opinion_reflective_judgment",
            "process_advice_question": "opinion_advice_process",
            "expressive_request": "expressive_request",
            "reflective_observation_statement": "reflective_observation",
            "relational_interpretation_statement": "relational_interpretation",
            "comparative_reflection_statement": "comparative_reflection",
            "aesthetic_reflection_statement": "aesthetic_reflection",
            "reason_probe_question": "reason_probe",
            "soft_decision_question": "opinion_decision_request",
            "activity_recommendation_question": "activity_recommendation",
        }
        for proposition_kind, cue_type in question_schema_map.items():
            prop = next((item for item in propositions if item.kind == proposition_kind), None)
            if prop is None:
                continue
            append_cue(
                cue_type,
                f"question schema detected: {proposition_kind}",
                confidence=0.89,
                proposition_id=prop.proposition_id,
            )
        if state.turn_count > 0 and state.last_action in {
            ActionType.SHARE_FEELING,
            ActionType.CONTINUE_CONVERSATION,
        }:
            append_cue(
                "recent_handoff",
                "current turn follows a conversational handoff from the previous bot action",
                confidence=0.70,
            )
        if self._looks_like_weather_conditioned_activity_decision(text, propositions):
            activity_prop = next((prop for prop in propositions if prop.kind == "activity_candidate"), None)
            append_cue(
                "weather_conditioned_activity_decision",
                "user is asking for a conditional activity opinion grounded on their own weather impression",
                confidence=0.94,
                proposition_id=activity_prop.proposition_id if activity_prop else None,
            )
        if self._looks_like_activity_invite(text):
            activity_prop = next((prop for prop in propositions if prop.kind == "activity_invite"), None)
            append_cue(
                "activity_invite",
                "user is proposing or inviting a concrete shared activity",
                confidence=0.94,
                proposition_id=activity_prop.proposition_id if activity_prop else None,
            )
        return cues

    def _active_grounding_topics(
        self,
        features: MessageFeatures,
        propositions: list[PropositionUnit],
        context_cues: list[ContextCue],
    ) -> list[str]:
        topics: list[str] = []
        topic_hint = self._clean_topic(features.topic_hint)
        if topic_hint:
            topics.append(topic_hint)
        for proposition in propositions:
            for candidate in (proposition.object, proposition.subject):
                topic = self._clean_topic(candidate)
                if topic:
                    topics.append(topic)
        for cue in context_cues:
            if cue.cue_type in {
                "quiet_mode",
                "aftereffect_hold",
                "contrast_gap",
                "weather_conditioned_activity_decision",
                "opinion_preference_like",
                "opinion_habit_preference",
                "opinion_self_style",
                "opinion_reflective_judgment",
                "opinion_advice_process",
                "activity_recommendation",
                "expressive_request",
                "reflective_observation",
                "aesthetic_reflection",
                "reason_probe",
                "opinion_decision_request",
            }:
                topics.append(cue.cue_type)
        return self._dedupe_topics(topics)

    def _context_dependency_level(
        self,
        *,
        text: str,
        state: ConversationState,
        context_cues: list[ContextCue],
        clause_units: list[ClauseUnit],
    ) -> str:
        cue_types = {cue.cue_type for cue in context_cues}
        if "relationship_check" in cue_types:
            return "high"
        if "aftereffect_hold" in cue_types and state.turn_count > 0:
            return "high"
        if "recent_handoff" in cue_types and ("quiet_mode" in cue_types or len(text) <= 18):
            return "medium"
        if len(clause_units) >= 2 or "contrast_gap" in cue_types:
            return "medium"
        return "low"

    @staticmethod
    def _duo_role_state(state: ConversationState, context_cues: list[ContextCue]) -> str | None:
        cue_types = {cue.cue_type for cue in context_cues}
        if state.last_action in {ActionType.SHARE_FEELING, ActionType.CONTINUE_CONVERSATION}:
            if "recent_handoff" in cue_types:
                return "handoff_followup"
        return "fresh_turn" if state.turn_count == 0 else None

    @staticmethod
    def _topic_near(text: str, verb_pattern: str) -> str | None:
        match = re.search(rf"([0-9A-Za-z가-힣]{{1,12}})(?:이|가|은|는|을|를|도)?\s*{verb_pattern}", text)
        if not match:
            return None
        return match.group(1)

    def _clean_topic(self, value: str | None) -> str | None:
        if not value:
            return None
        value = value.strip()
        if not value.endswith("놀이"):
            value = re.sub(r"(이|가|은|는|을|를|도|만)$", "", value)
        if not value or value in self._stop_topics:
            return None
        if len(value) <= 1:
            return None
        return value

    @staticmethod
    def _dedupe_topics(values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value and value not in deduped:
                deduped.append(value)
        return deduped

    def _is_weather_premise_clause(self, text: str) -> bool:
        if "날씨" not in text and not any(marker in text for marker in ("햇빛", "바람")):
            return False
        if any(marker in text for marker in ("몇 도", "몇도", "어때", "알려", "비 오", "눈 오")):
            return False
        return any(marker in text for marker in ("좋", "괜찮", "선선", "맑", "좋아"))

    def _extract_activity_decision_object(self, text: str) -> str | None:
        match = re.search(
            rf"({'|'.join(self._activity_decision_terms)})(?:을|를|이|가)?\s*(?:칠까|할까|갈까|탈까|뛸까|해볼까|가볼까|해도\s*될까)",
            text,
        )
        if match:
            return match.group(1)
        for term in self._activity_decision_terms:
            if term in text and self._is_decision_request_clause(text):
                return term
        return None

    def _extract_global_activity_invite_props(
        self,
        text: str,
        clause_units: list[ClauseUnit],
    ) -> list[PropositionUnit]:
        if not self._looks_like_activity_invite(text):
            return []
        source_clause_id = clause_units[-1].clause_id if clause_units else "clause_1"
        props: list[PropositionUnit] = []
        index = 1

        def append(kind: str, *, subject: str, obj: str | None = None, value: str | None = None, weight: float) -> None:
            nonlocal index
            props.append(
                PropositionUnit(
                    proposition_id=f"global_activity_invite_prop_{index}",
                    kind=kind,
                    source_clause_id=source_clause_id,
                    subject=subject,
                    object=obj,
                    value=value,
                    weight=weight,
                )
            )
            index += 1

        activity = self._extract_activity_invite_object(text)
        if activity:
            append("activity_invite", subject="user", obj=activity, value=text, weight=0.94)
        context = self._extract_activity_context(text)
        if context:
            append("activity_context", subject="context", obj=context, value=context, weight=0.86)
        detail = self._extract_activity_detail(text)
        if detail:
            append("activity_detail", subject="activity", obj=detail, value=detail, weight=0.88)
        place = self._extract_activity_recommendation_place(text)
        if place:
            append("activity_place", subject="place", obj=place, value=place, weight=0.86)
        condition = self._extract_activity_condition(text, place=place or "")
        if condition:
            append("activity_condition", subject="premise", obj=condition, value=condition, weight=0.82)
        return props

    def _extract_activity_invite_object(self, text: str) -> str | None:
        food_match = re.search(r"([0-9A-Za-z가-힣 ]{1,20}?)(?:이나|라도|좀|나)?\s*(?:해\s*먹자|해먹자)", text)
        if food_match:
            candidate = re.sub(r".*(?:은데|는데|한데|인데)\s*", "", food_match.group(1)).strip(" ?!.,")
            if candidate:
                return candidate
        if "바베큐" in text or "바비큐" in text:
            return "바베큐"
        if "고기" in text and re.search(r"(굽|구워\s*먹|구워먹)", text):
            return "고기 굽기"
        if "불멍" in text:
            return "불멍"
        if "요리" in text:
            return "요리"
        if re.search(r"(굽|구워\s*먹|구워먹)", text):
            return "구워먹기"
        for term in sorted(self._activity_decision_terms, key=len, reverse=True):
            if term in text and self._looks_like_activity_invite(text):
                if term == "밥":
                    return "밥 먹기"
                if term == "커피":
                    return "커피 마시기"
                if term == "영화":
                    return "영화 보기"
                if term == "사진":
                    return "사진 찍기"
                return term
        match = re.search(r"([0-9A-Za-z가-힣 ]{1,20}?)(?:이나|라도|좀|나)?\s*(?:하자|가자|먹자|보자)", text)
        if match:
            candidate = re.sub(r".*(?:은데|는데|한데|인데)\s*", "", match.group(1)).strip(" ?!.,")
            return candidate or None
        return None

    @staticmethod
    def _extract_activity_context(text: str) -> str | None:
        if "캠핑" in text:
            return "캠핑"
        if "바다" in text:
            return "바다"
        if "계곡" in text:
            return "계곡"
        return None

    @staticmethod
    def _extract_activity_detail(text: str) -> str | None:
        if "고기" in text and "준비" in text:
            return "고기 준비"
        if re.search(r"(해\s*먹|해먹)", text):
            return "해먹기"
        if re.search(r"(구워\s*먹|구워먹)", text):
            return "구워먹기"
        if "굽" in text:
            return "굽기"
        if "바베큐" in text or "바비큐" in text:
            return "바베큐"
        return None

    def _extract_activity_recommendation_place(self, text: str) -> str | None:
        for place in sorted(self._activity_recommendation_places, key=len, reverse=True):
            if place in text:
                return place
        match = re.search(r"([가-힣A-Za-z0-9]+)(?:에서|가서|으로|로)\s*(?:무엇|뭐|뭘|어떤)", text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_activity_condition(text: str, *, place: str) -> str | None:
        if place:
            match = re.search(rf"({re.escape(place)})(?:이|가|은|는)?\s*(시원|선선|좋|맑|따뜻|더워|추워)", text)
            if match:
                return f"{place}가 {match.group(2)}함"
        match = re.search(r"(날씨|바람|공기)(?:이|가|은|는)?\s*(시원|선선|좋|맑|따뜻|더워|추워)", text)
        if match:
            return f"{match.group(1)}가 {match.group(2)}함"
        return None

    def _is_activity_recommendation_question(self, text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        if "추천" in text:
            return False
        place = self._extract_activity_recommendation_place(text)
        place_prefix = rf"(?:{re.escape(place)}).*" if place else ""
        patterns = (
            rf"{place_prefix}(?:무엇|뭐|뭘|어떤).*(?:하고\s*)?놀면\s*(?:좋|재밌|재미있|무난|괜찮)",
            rf"{place_prefix}(?:무엇|뭐|뭘|어떤).*(?:하고\s*)?(?:놀까|쉴까|쉬면|보내면|타는\s*게)",
            rf"{place_prefix}(?:무엇|뭐|뭘|어떤).*(?:하(?:고|면서)|하면서|하고서)?\s*(?:놀래|놀자|할까|하지|하면|보낼까)",
            rf"{place_prefix}(?:뭐\s*하(?:고|면서)|뭐하면서|뭐하고).*(?:놀래|놀까|놀자|시간\s*보낼까|보낼까)",
            rf"{place_prefix}(?:놀거리|할\s*만한|할만한|뭐\s*하지|뭐하지)",
            rf"{place_prefix}(?:뭐부터).*(?:타|할|볼)",
            rf"{place_prefix}(?:가장\s*먼저|먼저|처음).*(?:해야\s*할|하면\s*좋을|할)\s*(?:건|것|거).*(?:무엇|뭐|뭘)",
            rf"{place_prefix}(?:무엇|뭐|뭘).*(?:먼저|처음|우선).*(?:해야|하면|할)",
            r"(?:비\s*오는\s*날|비오는\s*날).*(?:실내|안에서).*(?:무엇|뭐|뭘|어떤).*(?:하고\s*)?(?:놀|하면|할까|놀까)",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_decision_request_clause(text: str) -> bool:
        return bool(re.search(r"(칠까|할까|갈까|탈까|뛸까|해볼까|가볼까|해도\s*될까)(\?|$)", text))

    @staticmethod
    def _looks_like_activity_invite(text: str) -> bool:
        if any(marker in text for marker in ("말하자", "얘기하자", "대화하자", "정리하자")):
            return False
        role_request_pattern = (
            r"(?:넌|너는|너가|네가)\s*[0-9A-Za-z가-힣 ]{0,18}?"
            r"(?:준비해\s*줘|챙겨\s*줘|맡아\s*줘|가져와\s*줘|구워\s*줘)"
        )
        has_invite_surface = bool(re.search(r"(하자|가자|먹자|보자)([.!?…]*)$", text))
        has_role_request = bool(re.search(role_request_pattern, text))
        if not has_invite_surface and not has_role_request:
            return False
        concrete_terms = (
            "수영",
            "물놀이",
            "산책",
            "러닝",
            "조깅",
            "자전거",
            "피크닉",
            "테니스",
            "농구",
            "축구",
            "캠핑",
            "커피",
            "밥",
            "영화",
            "보드게임",
            "사진",
            "운동",
            "바베큐",
            "바비큐",
            "고기",
            "구워먹",
            "불멍",
            "요리",
            "바다",
            "해변",
            "해수욕장",
            "계곡",
            "공원",
            "한강",
            "카페",
        )
        return any(term in text for term in concrete_terms)

    @staticmethod
    def _is_preference_like_question(text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".+좋아해\?$",
            r".+좋아하냐\?$",
            r".+좋아하니\?$",
            r".+좋아\?$",
            r".+싫어해\?$",
            r".+싫지\s*않아\?$",
            r".+로망\s*하나\s*있어\?$",
            r".+가장\s*좋을\s*거\s*같아\?$",
            r".+가장\s*좋을\s*것\s*같아\?$",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_reflective_judgment_question(text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".+같지\?$",
            r".+낫지\?$",
            r".+하지\?$",
            r".+되지\?$",
            r".+중요하지\?$",
            r".+이해돼\?$",
            r".+않아\?$",
            r".+않나\?$",
            r".+겠지\?$",
            r".+달라지지\?$",
            r".+차분해\?$",
            r".+뿌듯하지\?$",
            r".+실감날\s*것\s*같지\?$",
            r".+대단해\s*보여\?$",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_process_advice_question(text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".+무엇부터\s*해야\s*할까\?$",
            r".+(?:가장\s*먼저|먼저|처음)\s*해야\s*할\s*(?:건|것|거)\s*(?:무엇|뭐|뭘)(?:일까)?\?$",
            r".+무엇부터\s*점검해야\s*할까\?$",
            r".+무엇부터\s*분명히\s*하면\s*좋을까\?$",
            r".+무엇부터\s*관찰할까\?$",
            r".+무엇부터\s*해보는\s*게\s*좋을까\?$",
            r".+뭘\s*먼저\s*(봐야|해야)\s*할까\?$",
            r".+우선\s*(확인|봐야)\s*할까\?$",
            r".+우선\s*(확인|점검)해야\s*할까\?$",
            r".+어떻게\s*읽어야\s*할까\?$",
            r".+어떤\s*순서로\s*보면\s*좋을까\?$",
            r".+어떻게\s*시작해야\s*할까\?$",
            r".+어떻게\s*시작할까\?$",
            r".+어떻게\s*가볍게\s*확인할\s*수\s*있을까\?$",
            r".+어떻게\s*입는\s*게\s*좋을까\?$",
            r".+뭐라고\s*하는\s*게\s*좋을까\?$",
            r".+뭘\s*실험해볼까\?$",
            r".+뭐가\s*무난할까\?$",
            r".+어떤\s*쪽을\s*우선해야\s*할까\?$",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_expressive_request(text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".+표현해줘\.?$",
            r".+묘사해줘\.?$",
            r".+설명해줘\.?$",
            r".+그려줘\.?$",
            r".+풀어줘\.?$",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_reflective_observation_statement(text: str, is_question: bool) -> bool:
        if is_question:
            question_patterns = (
                r".+실감할\s*때\s*있어\?$",
                r".+잊히지\?$",
                r".+공기지\?$",
                r".+재밌지\?$",
                r".+중\s*하나\s*같아\?$",
                r".+느껴지지\?$",
            )
            return any(re.search(pattern, text) for pattern in question_patterns)
        if len(text) < 20:
            return False
        end_patterns = (
            r".+같아[.!]?$",
            r".+같아서[.!]?$",
            r".+생각이\s*들어[.!]?$",
            r".+생각이\s*들더라[.!]?$",
            r".+느껴져[.!]?$",
            r".+보여[.!]?$",
            r".+좋더라[.!]?$",
            r".+같거든[.!]?$",
            r".+종류의.+[.!]?$",
            r".+처럼.+느껴져[.!]?$",
            r".+느끼게\s*돼[.!]?$",
            r".+느끼게\s*되더라[.!]?$",
        )
        clause_markers = (
            "생각이 들어",
            "생각이 들더라",
            "느껴져",
            "보여서",
            "좋더라",
            "같거든",
            "이해가 가더라",
            "선명해지는 식으로",
            "느끼게 돼",
            "느끼게 되더라",
            "들어 있다는 거",
            "변하지 않았다는",
        )
        return any(re.search(pattern, text) for pattern in end_patterns) or any(
            marker in text for marker in clause_markers
        )

    @staticmethod
    def _is_relational_interpretation_statement(text: str, is_question: bool) -> bool:
        if is_question or len(text) < 18:
            return False
        markers = (
            "하트만 남기고 끝났",
            "화제 바뀌",
            "허락을 받았다는 느낌보다",
            "죄책감을",
            "이해받기 위한 시도라기보다",
            "닫힌 문을 다시 두드리",
            "보류나 거절 쪽으로",
            "반찬 챙겨두는 말이 없어졌",
            "안쪽은 아니구나",
            "계속 밀리는 사람은 나구나",
            "말이 없어졌",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_comparative_reflection_statement(text: str, is_question: bool) -> bool:
        if is_question or len(text) < 16:
            return False
        patterns = (
            r".+보다\s*덜\s*상처받는\s*게.+중요해\s*보인다\.?$",
            r".+보다\s*덜\s*힘들게\s*보내는\s*게\s*목표가\s*된다\.?$",
            r".+보다\s*덜\s*[가-힣]+\s*게.+\.?$",
        )
        markers = (
            "잘 넘기는 것보다 덜 상처받는",
            "잘 보내는 것보다 덜 힘들게",
            "덜 상처받는 게 더 중요해",
            "덜 힘들게 보내는 게 목표",
        )
        return any(re.search(pattern, text) for pattern in patterns) or any(
            marker in text for marker in markers
        )

    @classmethod
    def _is_aesthetic_reflection_statement(cls, text: str, is_question: bool) -> bool:
        aesthetic_markers = (
            "빛",
            "어둠",
            "파랑",
            "빨강",
            "바다",
            "물고기",
            "냄새",
            "풍경",
            "분위기",
            "밤하늘",
            "스탠드",
            "강은",
            "색감",
            "침묵",
            "여백",
            "장면",
            "차갑",
            "예쁘",
        )
        if not any(marker in text for marker in aesthetic_markers):
            return False
        if is_question:
            return bool(re.search(r".+(다르지|달라지지|예쁘기도\s*하지|차분해)\?$", text))
        strong_non_question_markers = (
            "파랑",
            "빨강",
            "냄새",
            "풍경",
            "분위기",
            "색감",
            "침묵",
            "여백",
            "장면",
            "차갑",
            "예쁘",
            "스탠드",
        )
        if not any(marker in text for marker in strong_non_question_markers):
            return False
        return cls._is_reflective_observation_statement(text, is_question=False)

    @staticmethod
    def _is_reason_probe_question(text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".*왜\s*저러는\s*거야\?$",
            r".*왜\s*그런\s*거야\?$",
            r".*이유가\s*뭘까\?$",
            r".*왜\s*그렇게\s*되는\s*걸까\?$",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @classmethod
    def _is_soft_decision_question(cls, text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".+할까\?$",
            r".+될까\?$",
            r".+해볼까\?$",
            r".+가볼까\?$",
            r".+할지\s*말지\s*애매하",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_self_style_question(text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r"너는.*무슨\s*말부터.*편이야",
            r"너는.*어떤\s*말부터.*편이야",
            r"너는.*먼저.*무슨\s*말",
            r"너는.*뭐부터.*꺼내는\s*편이야",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_habit_preference_question(text: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r"(자주|보통|원래|대체로).*(편이야|편이냐|편이니)(\?|$)",
            r"같은\s*건.*(편이야|편이냐|편이니)(\?|$)",
            r"[가-힣a-z0-9\s]+는\s*편이야(\?|$)",
            r"[가-힣a-z0-9\s]+는\s*편이냐(\?|$)",
            r"[가-힣a-z0-9\s]+는\s*편이니(\?|$)",
            r"좋아하는\s*편이야(\?|$)",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _extract_question_topic(text: str) -> str | None:
        patterns = (
            r"(.+?)\s*좋아해\?$",
            r"(.+?)\s*좋아하냐\?$",
            r"(.+?)\s*좋아하니\?$",
            r"(.+?)\s*좋아\?$",
            r"(.+?)\s*싫지\s*않아\?$",
            r"(.+?)\s*편이야\?$",
            r"(.+?)\s*편이냐\?$",
            r"(.+?)\s*편이니\?$",
            r"(.+?)\s*할까\?$",
            r"(.+?)\s*될까\?$",
            r"(.+?)\s*해볼까\?$",
            r"(.+?)\s*가볼까\?$",
            r"(.+?)\s*주어지면\s*가장\s*좋을\s*거\s*같아\?$",
            r"(.+?)\s*주어지면\s*가장\s*좋을\s*것\s*같아\?$",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            topic = match.group(1).strip(" ?")
            topic = re.sub(r"(은|는|이|가|을|를|도|건|거)$", "", topic)
            if topic and len(topic) <= 24:
                return topic
        return None

    @staticmethod
    def _extract_expressive_request_topic(text: str) -> str | None:
        match = re.search(
            r"(.+?)(?:를|을)\s+(?:문장\s*리듬으로\s*|말로\s*|짧게\s*|한\s*문장으로\s*)?(표현해줘|묘사해줘|설명해줘|그려줘|풀어줘)\.?$",
            text,
        )
        if not match:
            match = re.search(r"(.+?)\s*(표현해줘|묘사해줘|설명해줘|그려줘|풀어줘)\.?$", text)
        if not match:
            return None
        topic = match.group(1).strip(" ?.")
        topic = re.sub(r"(을|를|이|가|은|는|도|만)$", "", topic)
        if topic and len(topic) <= 24:
            return topic
        return None

    @staticmethod
    def _extract_reflective_topic(text: str) -> str | None:
        for marker in ("은", "는", "이", "가"):
            match = re.search(rf"([0-9A-Za-z가-힣 ]{{1,20}}){marker}\s", text)
            if match:
                topic = match.group(1).strip()
                if 1 < len(topic) <= 20:
                    return topic
        return None

    def _looks_like_weather_conditioned_activity_decision(
        self,
        text: str,
        propositions: list[PropositionUnit],
    ) -> bool:
        proposition_kinds = {prop.kind for prop in propositions}
        if "activity_candidate" not in proposition_kinds or "decision_request" not in proposition_kinds:
            return False
        if "weather_premise" in proposition_kinds:
            return True
        return self._is_weather_premise_clause(text)
