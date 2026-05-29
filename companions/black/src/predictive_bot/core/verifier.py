from __future__ import annotations

import random
import re
from dataclasses import dataclass

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    VerificationReport,
    WeatherReport,
    WorldState,
)


@dataclass(slots=True, frozen=True)
class ClaimCheck:
    claim: str
    supported: bool
    evidence_label: str | None = None


@dataclass(slots=True, frozen=True)
class EvidenceText:
    label: str
    text: str


class ClaimVerifier:
    """Checks high-risk factual sentences against the evidence already attached to the decision."""

    FACTUAL_ACTIONS = {
        ActionType.WEATHER_LOOKUP,
        ActionType.SEARCH_ANSWER,
        ActionType.NEWS_ANSWER,
        ActionType.TELL_TIME,
    }
    _FACTUAL_MARKERS = (
        "수도",
        "국기",
        "위치",
        "어디",
        "기온",
        "날씨",
        "바람",
        "뉴스",
        "헤드라인",
        "출처",
        "기준",
        "현재",
        "최신",
        "오늘",
        "내일",
        "요일",
        "시간",
    )
    _NUMERIC_FACT_RE = re.compile(r"\d+(?:[.,:]\d+)?\s*(?:도|km|%|원|달러|월|일|시|분|명|개)?")

    def verify(
        self,
        *,
        reply: str,
        decision: ActionDecision,
        world_state: WorldState,
        weather: WeatherReport | None,
    ) -> list[ClaimCheck]:
        if decision.action not in self.FACTUAL_ACTIONS:
            return []

        claims = self._extract_factual_claims(reply)
        if not claims:
            return []

        evidence = self._build_evidence_texts(decision=decision, weather=weather)
        checks: list[ClaimCheck] = []
        for claim in claims:
            evidence_label = self._supporting_evidence_label(claim, evidence)
            checks.append(
                ClaimCheck(
                    claim=claim,
                    supported=evidence_label is not None,
                    evidence_label=evidence_label,
                )
            )
        return checks

    @classmethod
    def _extract_factual_claims(cls, reply: str) -> list[str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", reply)
            if sentence.strip()
        ]
        claims: list[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if cls._NUMERIC_FACT_RE.search(sentence) or any(marker in lowered for marker in cls._FACTUAL_MARKERS):
                claims.append(sentence)
        return claims

    @classmethod
    def _build_evidence_texts(
        cls,
        *,
        decision: ActionDecision,
        weather: WeatherReport | None,
    ) -> list[EvidenceText]:
        evidence: list[EvidenceText] = []
        slots = decision.slots

        for key in (
            "knowledge_answer",
            "knowledge_subject",
            "knowledge_source",
            "news_summary",
            "news_titles",
            "time_text",
            "time_date",
            "time_weekday",
            "time_timezone",
            "recommendation_text",
            "music_text",
        ):
            value = slots.get(key)
            if value:
                text = _grounding_source_label(value) if key == "knowledge_source" else value
                evidence.append(EvidenceText(label=f"slot:{key}", text=text or value))

        if weather is not None:
            evidence.append(
                EvidenceText(
                    label="tool:weather",
                    text=(
                        f"{weather.location} {weather.description} "
                        f"{weather.temperature_c:.1f} {weather.wind_kph:.1f}"
                    ),
                )
            )

        return evidence

    @classmethod
    def _supporting_evidence_label(cls, claim: str, evidence: list[EvidenceText]) -> str | None:
        if not evidence:
            return None
        claim_norm = cls._normalize_claim_text(claim)
        claim_tokens = cls._claim_tokens(claim)
        for item in evidence:
            evidence_norm = cls._normalize_claim_text(item.text)
            if claim_norm and (claim_norm in evidence_norm or evidence_norm in claim_norm):
                return item.label
            evidence_tokens = cls._claim_tokens(item.text)
            if not claim_tokens or not evidence_tokens:
                continue
            overlap = claim_tokens & evidence_tokens
            if item.label in {"tool:weather", "slot:knowledge_answer", "slot:time_text", "slot:news_summary"} and overlap:
                return item.label
            required_overlap = 1 if len(claim_tokens) <= 2 else 2
            if len(overlap) >= required_overlap:
                return item.label
        return None

    @staticmethod
    def _normalize_claim_text(text: str) -> str:
        return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).casefold()

    @staticmethod
    def _claim_tokens(text: str) -> set[str]:
        raw_tokens = re.findall(r"[0-9A-Za-z가-힣.]+", text.casefold())
        tokens: set[str] = set()
        for token in raw_tokens:
            cleaned = token.strip(".")
            if len(cleaned) < 2:
                continue
            number_match = re.match(r"^\d+(?:[.,:]\d+)?", cleaned)
            if number_match:
                tokens.add(number_match.group(0))
            tokens.add(_strip_korean_topic_particle(cleaned))
        return tokens


class ResponseVerifier:
    """Final guardrail layer that can rewrite unsafe or unsupported replies."""

    def __init__(self, *, claim_verifier: ClaimVerifier | None = None) -> None:
        self.claim_verifier = claim_verifier or ClaimVerifier()

    @staticmethod
    def _pick_rewrite(seed: str, options: tuple[str, ...]) -> str:
        return random.Random(seed).choice(options)

    def verify(
        self,
        *,
        reply: str,
        decision: ActionDecision,
        state: ConversationState,
        world_state: WorldState,
        weather: WeatherReport | None,
        draft_utterance: dict[str, object] | None = None,
    ) -> VerificationReport:
        normalized_reply = reply.strip()
        if not normalized_reply:
            return VerificationReport(
                ok=False,
                severity="high",
                issues=["empty_reply"],
                revised_reply=None,
            )

        draft_direct_reason = ""
        if isinstance(draft_utterance, dict):
            draft_direct_reason = str(draft_utterance.get("direct_surface_reason") or "").strip()
        draft_direct_persona_reply = draft_direct_reason.startswith("korean_daily_") or draft_direct_reason in {
            "companion_presence_style",
            "food_lifestyle_direct_reply",
            "work_school_direct_reply",
            "relationship_boundary_direct_reply",
            "relationship_deep_context_direct_reply",
            "relationship_extreme_boundary_direct_reply",
        }

        if (
            decision.action == ActionType.WEATHER_LOOKUP
            and weather is None
            and not draft_direct_persona_reply
        ):
            location_hint = state.known_location or "도시 이름"
            return VerificationReport(
                ok=False,
                severity="high",
                issues=["missing_weather_grounding"],
                revised_reply=f"날씨는 근거가 있어야 해서 바로 못 말해. {location_hint}만 주면 다시 볼게.",
            )

        knowledge_grounded = decision.slots.get("knowledge_grounded") == "true"

        if (
            "do_not_guess_facts" in world_state.constraints
            and decision.action in {ActionType.SEARCH_ANSWER, ActionType.NEWS_ANSWER}
            and not knowledge_grounded
            and not draft_direct_persona_reply
        ):
            if _is_ungrounded_fact_boundary(normalized_reply):
                return VerificationReport(ok=True)
            return VerificationReport(
                ok=False,
                severity="medium",
                issues=["external_fact_not_grounded"],
                revised_reply="이건 지금 바로 단정하면 위험해. 검색 기능부터 붙이면 더 정확하게 답할 수 있어.",
            )

        if draft_direct_persona_reply:
            return VerificationReport(ok=True)

        claim_checks = self.claim_verifier.verify(
            reply=normalized_reply,
            decision=decision,
            world_state=world_state,
            weather=weather,
        )
        unsupported_claims = [check.claim for check in claim_checks if not check.supported]
        if unsupported_claims:
            grounded_reply = _grounded_reply_from_decision(decision=decision, state=state, weather=weather)
            return VerificationReport(
                ok=False,
                severity="medium",
                issues=[
                    "unsupported_factual_claim",
                    *[
                        f"unsupported_claim:{_compact_issue_text(claim)}"
                        for claim in unsupported_claims[:3]
                    ],
                ],
                revised_reply=grounded_reply,
            )

        if "avoid_escalation" in world_state.constraints and any(
            bad_token in normalized_reply for bad_token in ("바보", "멍청", "꺼져")
        ):
            return VerificationReport(
                ok=False,
                severity="medium",
                issues=["reply_escalates_conflict"],
                revised_reply=self._pick_rewrite(
                    normalized_reply,
                    (
                        "조금만 차분하게 다시 말해주면 그 기준으로 볼게.",
                        "말을 조금만 낮춰서 다시 주면 더 정확히 볼 수 있어.",
                        "차분하게 다시 말해주면 그 기준으로 이어갈게.",
                    ),
                ),
            )

        if "respect_boundary_history" in world_state.constraints and decision.action in {
            ActionType.ACKNOWLEDGE,
            ActionType.CONTINUE_CONVERSATION,
            ActionType.SMALL_TALK,
        }:
            pushy_tokens = ("더 말해봐", "이어봐", "계속해", "다시 말해줘", "한 줄만 더 줘", "다음은?")
            if any(token in normalized_reply for token in pushy_tokens):
                revised_reply = self._pick_rewrite(
                    normalized_reply,
                    (
                        "알겠어. 무리해서 더 이어갈 필요는 없어.",
                        "응, 편할 때 이어줘.",
                        "좋아. 지금은 여기까지만 받아둘게.",
                    ),
                )
                return VerificationReport(
                    ok=False,
                    severity="medium",
                    issues=["boundary_tone_mismatch"],
                    revised_reply=revised_reply,
                )

        if "avoid_overfamiliarity" in world_state.constraints and decision.action in {
            ActionType.CONTINUE_CONVERSATION,
            ActionType.SMALL_TALK,
            ActionType.TEASE_BACK,
        }:
            overfamiliar_tokens = ("ㅋㅋ", "ㄱㄱ", "왔구나", "반가워~", "야 ")
            if any(token in normalized_reply for token in overfamiliar_tokens):
                return VerificationReport(
                    ok=False,
                    severity="medium",
                    issues=["overfamiliar_tone_mismatch"],
                    revised_reply=self._pick_rewrite(
                        normalized_reply,
                        (
                            "응, 여기 있어. 편한 쪽으로 이어가자.",
                            "응, 편한 톤으로만 이어가자.",
                            "좋아. 너무 가까이 붙지 않고 가볍게 이어갈게.",
                        ),
                    ),
                )

        surface_issues = _surface_quality_issues(reply=normalized_reply, decision=decision)
        if _has_malformed_surface_text(normalized_reply):
            surface_issues.append("malformed_surface_text")
        if surface_issues:
            return VerificationReport(
                ok=False,
                severity="high",
                issues=list(dict.fromkeys(surface_issues)),
                revised_reply=None,
            )

        draft_issues = _draft_preservation_issues(
            reply=normalized_reply,
            draft_utterance=draft_utterance,
        )
        if draft_issues:
            return VerificationReport(
                ok=False,
                severity="medium",
                issues=draft_issues,
                revised_reply=None,
            )

        return VerificationReport(ok=True)


def _grounded_reply_from_decision(
    *,
    decision: ActionDecision,
    state: ConversationState,
    weather: WeatherReport | None,
) -> str:
    slots = decision.slots
    if decision.action == ActionType.WEATHER_LOOKUP and weather is not None:
        topic_particle = "은" if _has_final_consonant(weather.location) else "는"
        return (
            f"지금 {weather.location}{topic_particle} {weather.description}이야. "
            f"기온은 {weather.temperature_c:.1f}도고 바람은 {weather.wind_kph:.1f}km 정도야."
        )

    if decision.action == ActionType.TELL_TIME and slots.get("time_text"):
        return str(slots["time_text"])

    if decision.action == ActionType.NEWS_ANSWER and slots.get("news_summary"):
        return str(slots["news_summary"])

    if decision.action == ActionType.SEARCH_ANSWER and slots.get("knowledge_answer"):
        answer = str(slots["knowledge_answer"])
        subject = str(slots.get("knowledge_subject") or "").strip()
        query_type = str(slots.get("knowledge_query_type") or "").strip()
        if query_type == "capital" and subject:
            base = f"{subject}의 수도는 {answer}야."
            return _append_grounding_source_note(base, slots.get("knowledge_source"))
        if query_type == "flag" and subject:
            base = f"{subject}의 국기는 {answer}야."
            return _append_grounding_source_note(base, slots.get("knowledge_source"))
        if query_type == "location" and subject:
            base = f"{subject}은 {answer}"
            return _append_grounding_source_note(base, slots.get("knowledge_source"))
        return _append_grounding_source_note(answer, slots.get("knowledge_source"))

    location_hint = state.known_location or "확인된 근거"
    return f"지금은 {location_hint} 기준으로 확인된 내용만 말할게."


def _compact_issue_text(text: str, *, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _is_ungrounded_fact_boundary(reply: str) -> bool:
    normalized = _normalize_draft_text(reply)
    if not normalized:
        return False
    has_boundary = any(marker in normalized for marker in ("모르", "확인된근거", "근거없이", "단정하지", "못해"))
    if not has_boundary:
        return False
    return not any(marker in normalized for marker in ("올랐어", "오를거", "내릴거", "상승", "하락", "확실"))


def _draft_preservation_issues(
    *,
    reply: str,
    draft_utterance: dict[str, object] | None,
) -> list[str]:
    if not draft_utterance:
        return []

    draft_reply = str(draft_utterance.get("draft_reply") or "").strip()
    if not draft_reply:
        return []

    issues: list[str] = []
    for phrase in _string_list(draft_utterance.get("avoid")):
        if _draft_avoid_phrase_found(phrase=phrase, reply=reply):
            issues.append(f"draft_avoid_phrase_used:{_compact_issue_text(phrase, limit=36)}")
            if len(issues) >= 3:
                break

    draft_normalized = _normalize_draft_text(draft_reply)
    draft_reply_tokens = _draft_content_tokens(draft_reply)
    anchor = str(draft_utterance.get("anchor") or "").strip()
    must_include = _string_list(draft_utterance.get("must_include"))
    required_terms: list[str] = []
    if _draft_term_is_specific(anchor) and _draft_required_term_found(
        term=anchor,
        normalized_reply=draft_normalized,
        reply_tokens=draft_reply_tokens,
    ):
        required_terms.append(anchor)
    required_terms.extend(term for term in must_include if _draft_term_is_specific(term))
    normalized_reply = _normalize_draft_text(reply)
    reply_tokens = _draft_content_tokens(reply)
    found_required = [
        term
        for term in required_terms
        if _draft_required_term_found(term=term, normalized_reply=normalized_reply, reply_tokens=reply_tokens)
    ]
    if required_terms and not found_required:
        issues.append("draft_anchor_missing")

    if str(draft_utterance.get("action") or "") == "accept_activity_invite":
        for term in must_include:
            if " " in term or not _draft_term_is_specific(term) or term.endswith("함"):
                continue
            if not _draft_required_term_found(term=term, normalized_reply=normalized_reply, reply_tokens=reply_tokens):
                issues.append(f"draft_required_term_missing:{_compact_issue_text(term, limit=36)}")
                if len(issues) >= 3:
                    break

    if _draft_has_negation(draft_reply) and not _draft_has_negation(reply):
        issues.append("draft_negation_missing")

    if (
        "오랜만" in draft_normalized
        and "다시" in draft_normalized
        and "오랜만" not in normalized_reply
        and "다시" not in normalized_reply
    ):
        issues.append("draft_required_cue_missing:social_return")

    draft_tokens = _draft_content_tokens(" ".join([draft_reply, anchor, *must_include]))
    if (
        len(draft_tokens) >= 4
        and len(reply_tokens) >= 3
        and not (draft_tokens & reply_tokens)
        and not found_required
    ):
        issues.append("draft_semantic_drift")

    return list(dict.fromkeys(issues))


def _has_malformed_surface_text(text: str) -> bool:
    surface = str(text or "")
    if "\ufffd" in surface:
        return True
    if re.search(r"[\u4e00-\u9fff]{2,}", surface) and not re.search(r"[가-힣]", surface):
        return True
    for token in re.findall(r"[A-Za-z가-힣]+", surface):
        if re.fullmatch(r"(?:Black|White)[은는이가을를와과도]?", token):
            continue
        if re.search(r"[a-z]", token) and re.search(r"[가-힣]", token):
            return True
    return False


def _surface_quality_issues(*, reply: str, decision: ActionDecision) -> list[str]:
    issues: list[str] = []
    if _has_instruction_leak(reply):
        issues.append("instruction_leak")
    if _has_internal_label_leak(reply):
        issues.append("internal_label_leak")
    if _has_phrase_fragment(reply):
        issues.append("phrase_fragment")
    if _has_rewrite_artifact_phrase(reply):
        issues.append("rewrite_artifact_phrase")
    if _has_dangling_terminal_fragment(reply):
        issues.append("dangling_terminal_fragment")
    if _is_generic_stock_reply(reply) and not _allows_reused_structural_surface(decision):
        issues.append("generic_stock_reply")
    if _is_wrong_location_request(reply=reply, decision=decision):
        issues.append("wrong_location_request")
    return issues


def _has_instruction_leak(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    leak_markers = (
        "주어진 초안",
        "문장 다듬기",
        "다듬기를 수행",
        "변환해 드리겠습니다",
        "변환해 드리겠",
        "자연스러운 표현으로 변경",
        "자연스러운 대답을 제공",
        "다시 한 번 변환",
        "한국어 문장 다듬기",
        "answer_blueprint",
        "draft_utterance",
        "rewrite this",
    )
    lowered = normalized.lower()
    return any(marker.lower() in lowered for marker in leak_markers)


def _has_internal_label_leak(text: str) -> bool:
    normalized = str(text or "")
    if not normalized.strip():
        return False
    return bool(
        re.search(
            r"(?:타인의\s*행동|입장|톤|필수\s*단어|행동|response_plan|reason_code|reason_flags|must_include|avoid|draft_reply|action)\s*[:：=]",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _has_phrase_fragment(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    fragment_markers = (
        "그런 결, 그런 건",
        "한 번만 더",
        "지금 결이 너무",
        "여운이 남는 쪽, 위로",
        "여운이 남는 쪽으로.",
    )
    if any(marker in normalized for marker in fragment_markers):
        return True
    hangul_fragments = [
        part.strip()
        for part in re.split(r"[,，/]+", normalized)
        if re.search(r"[가-힣]", part)
    ]
    return len(hangul_fragments) >= 5 and sum(1 for part in hangul_fragments if len(part) <= 10) >= 4


def _has_rewrite_artifact_phrase(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return False
    artifact_markers = (
        "기가 빨리 가까워",
        "축제를 지치는",
        "나무의 흔적을 찾아",
        "카드를 받으면",
        "확실하지 않음 단정하지 않음",
    )
    return any(marker in compact for marker in artifact_markers)


def _has_dangling_terminal_fragment(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return False
    if re.search(r"(?:기준으로|쪽으로)\s*(?:짧게|가볍게|차분히|천천히)[.!?。]?$", compact):
        return True
    if re.search(r"(?:은|는|이|가|을|를|으로|처럼|부터|까지|하고|하며|하면서|있는|없는|있을|없을)$", compact):
        return True
    return bool(re.search(r"(?:습관|생각|기분|마음|말|쪽|편|일|것)(?:은|는)?\s+있을[.!?。]?$", compact))


def _is_generic_stock_reply(text: str) -> bool:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return False
    stock_replies = (
        "그 말은 길게 키우진 않을게. 필요한 만큼만 이어가자.",
        "그 마음은 그냥 넘기긴 어렵지. 지금은 숨 돌릴 틈부터 두자.",
        "그 마음은 그냥 넘기긴 어렵지. 지금은 숨 돌릴 틈이 너무나도 있다.",
        "그쪽은 나는 꽤 맞는 편이야. 강도만 맞으면 무난하게 봐.",
        "그 생각은 이해돼. 다만 무리하게 밀 필요는 없어.",
        "오늘은 차분한 쪽으로 둘게. 억지로 텐션 올리는 편이야.",
        "그 선택은 부담이 너무 크지 않으면 해볼 만해. 무리가 아니면 선택지로 둘 수 있어.",
    )
    if any(compact == stock or compact.startswith(stock) for stock in stock_replies):
        return True
    embedded_stock_markers = (
        "나는 꽤 맞는 편이야. 강도만 맞으면 무난하게 봐.",
        "이해돼. 다만 무리하게 밀 필요는 없어.",
        "가벼운 게임과 산책이 무난해.",
        "여유 있으면 간단한 간식도 좋아.",
        "활동 준비는 여벌옷",
        "간단한 간식부터 챙겨.",
        "나머지는 코스 길이에 맞추면 돼.",
        "부담이 너무 크지 않으면 해볼 만해.",
        "무리만 아니면 선택지로 둘 수 있어.",
        "무리가 아니면 선택지로 둘 수 있어.",
        "실제로 밖에 나갔다거나 뭘 샀다고 꾸미진 않을게.",
        "현재 읽는 책이 아니야.",
        "이 책을 읽는 것 같아서요.",
        "강은 물이 흐르는 속도가",
        "강은 꽤 맞는 쪽이야.",
    )
    return any(marker in compact for marker in embedded_stock_markers)


def _allows_reused_structural_surface(decision: ActionDecision) -> bool:
    flags = set(decision.reason_flags)
    return bool(
        flags
        & {
            "schema_activity_recommendation",
            "schema_activity_preparation_advice",
            "schema_concrete_topic_question",
            "schema_conversation_topic_suggestion",
        }
    )


def _is_wrong_location_request(*, reply: str, decision: ActionDecision) -> bool:
    normalized = _normalize_draft_text(reply)
    if "그지역에서어떤위치일까요" not in normalized and "어떤위치일까요" not in normalized:
        return False
    return decision.action not in {
        ActionType.ASK_LOCATION,
        ActionType.WEATHER_UNAVAILABLE,
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _draft_has_negation(text: str) -> bool:
    normalized = _normalize_draft_text(text)
    return any(marker in normalized for marker in ("않", "아니", "없", "말진", "말지", "말고", "못"))


def _draft_avoid_phrase_found(*, phrase: str, reply: str) -> bool:
    phrase = str(phrase or "").strip()
    if not phrase:
        return False
    if not re.fullmatch(r"[가-힣]+", phrase):
        return phrase in reply
    return bool(re.search(rf"(?<![가-힣]){re.escape(phrase)}(?![가-힣])", reply))


def _draft_term_is_specific(term: str) -> bool:
    normalized = _normalize_draft_text(term)
    if len(normalized) < 2:
        return False
    if len(normalized) > 10:
        return False
    return normalized not in {
        "그말",
        "그쪽",
        "그마음",
        "그문제",
        "그선택",
        "지금",
        "오늘",
        "맞아",
        "응",
        "그래",
    }


def _draft_required_term_found(
    *,
    term: str,
    normalized_reply: str,
    reply_tokens: set[str],
) -> bool:
    normalized = _normalize_draft_text(term)
    if normalized and normalized in normalized_reply:
        return True
    term_tokens = _draft_content_tokens(term)
    if term_tokens & reply_tokens:
        return True
    return any(
        left.startswith(right) or right.startswith(left)
        for left in term_tokens
        for right in reply_tokens
        if len(left) >= 3 and len(right) >= 3
    )


def _draft_content_tokens(text: str) -> set[str]:
    stopwords = {
        "오늘",
        "지금",
        "그냥",
        "조금",
        "너무",
        "정도",
        "같아",
        "있어",
        "없어",
        "그건",
        "그쪽",
        "그말",
        "무난해",
        "괜찮아",
        "보여",
        "해야",
        "하면",
    }
    tokens = set()
    for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", str(text or "").casefold()):
        cleaned = _strip_korean_topic_particle(token)
        if len(cleaned) >= 2 and cleaned not in stopwords:
            tokens.add(cleaned)
    return tokens


def _normalize_draft_text(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()


def _has_final_consonant(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    code = ord(stripped[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def _strip_korean_topic_particle(token: str) -> str:
    for suffix in ("으로는", "로는", "하지만", "이야", "야", "입니다", "다", "은", "는", "이", "가", "을", "를", "도", "에", "의"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 2:
            return token[: -len(suffix)]
    return token


def _append_grounding_source_note(reply: str, source: str | None) -> str:
    label = _grounding_source_label(source)
    if not label:
        return reply
    return f"{reply} (기준: {label})"


def _grounding_source_label(source: str | None) -> str | None:
    if not source:
        return None
    labels = {
        "builtin_country_facts": "기본 국가 정보",
        "builtin_country_capitals": "기본 국가 정보",
        "builtin_country_flags": "기본 국가 정보",
        "builtin_country_locations": "기본 국가 정보",
        "wikidata": "Wikidata",
        "google_news_rss": "Google News RSS",
        "system_clock": "로컬 시스템 시계",
    }
    return labels.get(source, source)
