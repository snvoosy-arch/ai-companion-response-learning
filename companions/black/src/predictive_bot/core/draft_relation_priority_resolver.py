from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


NONE_LABEL = "__none__"
RESOLVER_SOURCE = "black_relation_priority_resolver_v1_from_frame_axes"


@dataclass(frozen=True, slots=True)
class RelationPriorityResolution:
    relation_priority: str
    relation_type: str
    confidence: float
    source: str = RESOLVER_SOURCE
    evidence: tuple[str, ...] = ()
    blocked_evidence: tuple[str, ...] = ()

    def to_signal(self) -> dict[str, Any]:
        return {
            "axis": "relation_priority",
            "label": self.relation_priority,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": list(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_priority": self.relation_priority,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": list(self.evidence),
            "blocked_evidence": list(self.blocked_evidence),
        }


PRACTICAL_RELATION_TYPES = {
    "appliance_design_review_judgment",
    "car_accident_first_steps_practical",
    "deadline_file_loss_practical_first",
    "delivery_tired_compromise_practical",
    "device_water_damage_practical_first",
    "fever_body_check_practical_first",
    "gas_smell_emergency_practical_first",
    "gas_stove_ignition_issue_practical",
    "heating_bill_anxiety_practical",
    "impulse_spending_payment_friction",
    "interview_missed_bus_practical_first",
    "liver_seasoning_health_check",
    "living_cost_pressure_practical",
    "lottery_practical_first",
    "medicine_double_dose_practical_first",
    "neighbor_noise_record_first_practical",
    "new_project_first_step_practical",
    "oil_fire_water_misuse_practical_first",
    "online_scam_evidence_first",
    "perfectionism_sixty_point_start",
    "relationship_kakao_tone_anxiety_check",
    "wrong_transfer_practical_first",
}

EMOTION_RELATION_TYPES = {
    "ally_loneliness_emotion_first",
    "breakup_long_message_emotion_first",
    "group_chat_silence_emotion_first",
    "late_night_long_message_save",
    "parent_value_conflict_boundary",
    "pet_talk_care_first",
    "read_receipt_hurt_emotion_first",
}

JUDGMENT_RELATION_TYPES = {
    "choice_regret_judgment",
    "friend_partner_complaint_boundary",
    "grievance_logic_rebuttal_judgment",
    "quit_after_feedback_impulse",
    "read_receipt_uncertainty_hold_judgment",
    "relationship_boundary_polite_firm",
    "stock_fomo_judgment_brake",
    "success_standard_values",
    "white_lie_truth_tradeoff_judgment",
}

META_RELATION_TYPES = {
    "human_emotion_alarm_system",
    "semantic_relation_map_meta",
}

RELATION_PRIORITY_BY_TYPE: dict[str, str] = {
    **{name: "practical_first" for name in PRACTICAL_RELATION_TYPES},
    **{name: "emotion_stabilize" for name in EMOTION_RELATION_TYPES},
    **{name: "judgment" for name in JUDGMENT_RELATION_TYPES},
    **{name: "meta" for name in META_RELATION_TYPES},
}

PRIORITY_RANK = {
    "practical_first": 0,
    "emotion_stabilize": 1,
    "judgment": 2,
    "meta": 3,
    NONE_LABEL: 99,
}


def resolve_relation_priority(
    frame: dict[str, Any],
    *,
    raw_text: str = "",
    pragmatic_cues: list[str] | tuple[str, ...] | None = None,
) -> RelationPriorityResolution:
    text = _compact(raw_text or str(frame.get("text") or ""))
    cues = tuple(str(cue) for cue in (pragmatic_cues or frame.get("pragmatic_cues") or ()) if str(cue))
    relation_type = _axis_label(frame, "relation_type")

    if relation_type in META_RELATION_TYPES or _has_meta_text(text):
        return RelationPriorityResolution(
            relation_priority="meta",
            relation_type=relation_type,
            confidence=0.86,
            evidence=("meta_relation_context", f"relation_type:{relation_type}"),
        )

    hard_none = _hard_none_evidence(frame=frame, compact_text=text)
    text_priority = _text_priority_evidence(text)
    if hard_none and not _has_immediate_practical_text(text):
        return RelationPriorityResolution(
            relation_priority=NONE_LABEL,
            relation_type=relation_type,
            confidence=0.93,
            evidence=("hard_none_context", *hard_none),
            blocked_evidence=_blocked_from_relation_type(relation_type),
        )

    if text_priority is not None:
        priority, evidence = text_priority
        return RelationPriorityResolution(
            relation_priority=priority,
            relation_type=relation_type,
            confidence=0.92,
            evidence=("raw_text_priority", *evidence),
        )

    mapped_priority = RELATION_PRIORITY_BY_TYPE.get(relation_type)
    if mapped_priority is not None:
        if _relation_type_is_frame_incompatible(relation_type, frame=frame, compact_text=text):
            frame_priority = _frame_priority_if_specific(frame)
            if frame_priority is not None:
                priority, evidence = frame_priority
                return RelationPriorityResolution(
                    relation_priority=priority,
                    relation_type=relation_type,
                    confidence=0.81,
                    evidence=("relation_type_rejected_frame_fallback", *evidence),
                    blocked_evidence=(f"relation_type:{relation_type}",),
                )
            return RelationPriorityResolution(
                relation_priority=NONE_LABEL,
                relation_type=relation_type,
                confidence=0.82,
                evidence=("relation_type_rejected_by_frame", *_frame_evidence(frame)),
                blocked_evidence=(f"relation_type:{relation_type}",),
            )
        return RelationPriorityResolution(
            relation_priority=mapped_priority,
            relation_type=relation_type,
            confidence=_confidence_from_relation_type(relation_type, frame=frame, cues=cues),
            evidence=(f"relation_type:{relation_type}", f"mapped_priority:{mapped_priority}", *_frame_evidence(frame)),
        )

    frame_priority = _frame_priority_if_specific(frame)
    if frame_priority is not None:
        priority, evidence = frame_priority
        return RelationPriorityResolution(
            relation_priority=priority,
            relation_type=relation_type,
            confidence=0.78,
            evidence=("frame_specific_priority", *evidence),
        )

    model_priority = _axis_label(frame, "relation_priority")
    if model_priority in {"meta"}:
        return RelationPriorityResolution(
            relation_priority=model_priority,
            relation_type=relation_type,
            confidence=0.62,
            evidence=(f"model_relation_priority:{model_priority}",),
        )

    return RelationPriorityResolution(
        relation_priority=NONE_LABEL,
        relation_type=relation_type,
        confidence=0.72,
        evidence=("default_no_relation_priority", *_frame_evidence(frame)),
    )


def apply_relation_priority_resolution(
    semantic_frame: dict[str, Any],
    *,
    raw_text: str = "",
) -> dict[str, Any]:
    resolution = resolve_relation_priority(semantic_frame, raw_text=raw_text)
    payload = dict(semantic_frame)
    targets = dict(payload.get("targets") if isinstance(payload.get("targets"), dict) else {})
    payload_slots = dict(payload.get("slots") if isinstance(payload.get("slots"), dict) else {})
    target_slots = dict(targets.get("slots") if isinstance(targets.get("slots"), dict) else {})

    payload["relation_priority"] = resolution.relation_priority
    targets["relation_priority"] = resolution.relation_priority
    payload_slots["relation_priority"] = resolution.relation_priority
    target_slots["relation_priority"] = resolution.relation_priority
    targets["slots"] = target_slots
    payload["slots"] = payload_slots
    payload["targets"] = targets
    payload["relation_priority_resolution"] = resolution.to_dict()

    cues = list(payload.get("pragmatic_cues") if isinstance(payload.get("pragmatic_cues"), list) else [])
    cues.append("relation_priority_resolver_v1")
    cues.append(f"resolved_relation_priority:{resolution.relation_priority}")
    payload["pragmatic_cues"] = list(dict.fromkeys(cues))

    signals = list(payload.get("signals") if isinstance(payload.get("signals"), list) else [])
    signals.append(resolution.to_signal())
    payload["signals"] = signals
    return payload


def _axis_label(frame: dict[str, Any], axis: str) -> str:
    value = frame.get(axis)
    if value is None and isinstance(frame.get("targets"), dict):
        value = frame["targets"].get(axis)
    if value is None and isinstance(frame.get("slots"), dict):
        value = frame["slots"].get(axis)
    text = str(value or "").strip()
    if not text or text in {"None", "null"}:
        return NONE_LABEL
    return text


def _compact(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+", "", str(text or "")).lower()


def _has_all(text: str, *needles: str) -> bool:
    return all(_compact(needle) in text for needle in needles)


def _has_any(text: str, *needles: str) -> bool:
    return any(_compact(needle) in text for needle in needles)


def _text_priority_evidence(text: str) -> tuple[str, tuple[str, ...]] | None:
    if _has_immediate_practical_text(text):
        return "practical_first", ("immediate_practical_text",)
    if _has_judgment_text(text):
        return "judgment", ("judgment_text",)
    if _has_emotion_stabilize_text(text):
        return "emotion_stabilize", ("emotion_stabilize_text",)
    if _has_meta_text(text):
        return "meta", ("meta_text",)
    return None


def _has_immediate_practical_text(text: str) -> bool:
    return (
        (_has_any(text, "가스레인지", "가스렌지", "가스버너", "화구", "버너") and _has_any(text, "점화", "딸깍", "불이안", "불안붙", "불안켜", "불꽃"))
        or (_has_any(text, "가스비", "난방비", "보일러", "난방", "공과금", "관리비") and _has_any(text, "무서", "부담", "아끼", "절약", "온도", "외출모드", "켜기", "끄"))
        or (_has_any(text, "마감", "제출", "과제", "보고서", "회의자료", "발표자료") and _has_any(text, "파일", "문서", "자료") and _has_any(text, "날아", "복구", "저장", "꺼졌", "안열", "깨졌", "삭제", "덮어쓴"))
        or (_has_any(text, "카톡", "답장", "말투", "톤") and _has_any(text, "차가워", "딱딱", "건조", "달라져") and _has_any(text, "확인", "물어", "추궁", "체크"))
        or (_has_any(text, "가스냄새", "가스누출") and _has_any(text, "창문", "환기", "밸브", "119", "관리사무소", "나가"))
        or (_has_any(text, "면접", "버스놓", "늦") and _has_any(text, "택시", "담당자", "연락"))
        or (_has_any(text, "피싱", "문자", "링크") and _has_any(text, "계좌", "비밀번호", "차단", "카드"))
        or (_has_any(text, "옆집", "층간소음", "소음") and _has_any(text, "관리사무소", "기록", "쪽지"))
        or (_has_any(text, "충동구매", "스트레스받으면사", "돈모으") and _has_any(text, "장치", "막는", "차단", "걸까"))
        or (_has_any(text, "새프로젝트", "새프로젝트맡", "프로젝트맡") and _has_any(text, "첫단추", "뭐부터", "막막"))
        or (_has_any(text, "배달") and _has_any(text, "지쳐", "아무것도못", "돈아끼", "합리적"))
    )


def _has_judgment_text(text: str) -> bool:
    return (
        (_has_any(text, "읽씹", "안읽씹", "답장없", "답이없") and _has_any(text, "바쁜", "단정", "보류", "폰만", "결론"))
        or (_has_any(text, "퇴사", "사표", "그만두") and _has_any(text, "상사", "피드백", "충동", "자존심"))
        or (_has_any(text, "서운") and _has_any(text, "팩트", "논리", "반박", "감정부터"))
        or (_has_any(text, "주식", "투자", "코인") and _has_any(text, "조급", "뒤처", "손실", "한도"))
        or (_has_any(text, "성공기준", "인생기준") and _has_any(text, "보여주기", "버틸수", "조건"))
        or (_has_any(text, "애인흉", "감정쓰레기통") and _has_any(text, "선그어", "의리", "헷갈"))
    )


def _has_emotion_stabilize_text(text: str) -> bool:
    return (
        (_has_any(text, "내편", "기댈", "받아줄사람", "혼자", "외로", "고독", "서러") and _has_any(text, "사람", "편", "기대", "마음", "버텨"))
        or (_has_any(text, "단톡", "카톡방", "단체톡") and _has_any(text, "무반응", "반응이없", "묻히", "투명인간", "소외", "상처"))
        or (_has_any(text, "장문의카톡", "장문카톡", "새벽카톡") and _has_any(text, "후회", "저장", "보내면"))
        or (_has_any(text, "헤어지", "이별", "붙잡") and _has_any(text, "장문", "보내", "후회"))
    )


def _has_meta_text(text: str) -> bool:
    return _has_any(text, "인간감정", "감정비효율", "이성을잃", "비효율") and _has_any(
        text,
        "경보",
        "어떻게봐",
        "궁금",
        "힘들",
    )


def _hard_none_evidence(*, frame: dict[str, Any], compact_text: str) -> tuple[str, ...]:
    evidence: list[str] = []
    draft_frame = _axis_label(frame, "draft_frame")
    domain = _axis_label(frame, "domain")
    schema = _axis_label(frame, "schema")

    hard_none_frames = {
        "ai_comfort_before_emotion_proof",
        "career_passion_job_tradeoff",
        "counsel_rest_as_productivity",
        "counsel_rest_day_guilt",
        "foundation_refusal_bad_person_guilt",
        "relationship_late_message_short",
    }
    if draft_frame in hard_none_frames:
        evidence.append(f"draft_frame:{draft_frame}")
    if domain == "ai_companion" and schema == "emotional_support":
        evidence.append("ai_companion_emotional_support")
    if _has_any(compact_text, "ai", "인공지능") and _has_any(compact_text, "감정", "위로", "공감"):
        evidence.append("ai_comfort_text")
    if _has_any(compact_text, "약속", "지각", "늦") and _has_any(compact_text, "연락", "문자", "메시지", "뭐라고"):
        evidence.append("late_message_text")
    if _has_any(compact_text, "거절", "부탁") and _has_any(compact_text, "나쁜사람", "죄책감", "미움", "싫어도"):
        evidence.append("refusal_guilt_text")
    if (
        draft_frame != "money_delivery_tired_compromise"
        and not _has_any(compact_text, "배달")
        and _has_any(compact_text, "아무것도못", "쉬", "회복", "방전", "도태", "자괴감")
        and _has_any(compact_text, "몸", "지쳐", "불안", "죄책감")
    ):
        evidence.append("rest_guilt_text")
    if _has_any(compact_text, "좋아하는일", "직업") and _has_any(compact_text, "돈문제", "생계", "기준"):
        evidence.append("career_tradeoff_text")
    return tuple(dict.fromkeys(evidence))


def _relation_type_is_frame_incompatible(
    relation_type: str,
    *,
    frame: dict[str, Any],
    compact_text: str,
) -> bool:
    draft_frame = _axis_label(frame, "draft_frame")
    schema = _axis_label(frame, "schema")
    state_hint = _axis_label(frame, "state_hint")
    domain = _axis_label(frame, "domain")

    if relation_type in EMOTION_RELATION_TYPES:
        if _hard_none_evidence(frame=frame, compact_text=compact_text):
            return True
        if relation_type == "group_chat_silence_emotion_first":
            return not (_has_any(compact_text, "단톡", "카톡방", "단체톡") or draft_frame == "emotion_group_chat_ignored_stabilize")
        if relation_type == "ally_loneliness_emotion_first":
            return not (
                _has_any(compact_text, "내편", "외로", "고독", "기댈", "혼자")
                or draft_frame == "grief_loneliness_no_safe_person"
            )
        return schema not in {"emotional_support", "social_tactic"} and state_hint != "emotional_context"

    if relation_type in PRACTICAL_RELATION_TYPES:
        if relation_type == "relationship_kakao_tone_anxiety_check":
            return not (_has_any(compact_text, "카톡", "말투", "답장", "톤") or draft_frame == "relationship_kakao_tone_anxiety_check")
        if relation_type == "gas_stove_ignition_issue_practical":
            return not (_has_any(compact_text, "가스레인지", "가스렌지", "화구", "버너") or draft_frame == "gas_stove_ignition_issue")
        if relation_type == "deadline_file_loss_practical_first":
            return not (_has_any(compact_text, "마감", "파일", "문서", "자료", "제출") or draft_frame == "practical_deadline_file_recovery")
        return schema not in {"practical_advice", "social_tactic"} and state_hint != "practical_focus"

    if relation_type in JUDGMENT_RELATION_TYPES:
        if relation_type == "read_receipt_uncertainty_hold_judgment":
            return not (_has_any(compact_text, "읽씹", "답장", "단정", "보류") or draft_frame == "read_receipt_uncertainty")
        if relation_type == "quit_after_feedback_impulse":
            return not (_has_any(compact_text, "퇴사", "사표", "상사", "피드백") or draft_frame == "judgment_quit_impulse_after_feedback")
        if relation_type == "grievance_logic_rebuttal_judgment":
            return not (_has_any(compact_text, "서운", "팩트", "논리", "반박") or draft_frame == "relationship_grievance_logic_before_rebuttal")
        return domain not in {"social_relationship", "daily_life", "money_living"} and state_hint not in {"reflective_context", "practical_focus"}

    return False


def _frame_priority_if_specific(frame: dict[str, Any]) -> tuple[str, tuple[str, ...]] | None:
    draft_frame = _axis_label(frame, "draft_frame")
    mapping = {
        "emotion_group_chat_ignored_stabilize": "emotion_stabilize",
        "grief_loneliness_no_safe_person": "emotion_stabilize",
        "relationship_kakao_tone_anxiety_check": "practical_first",
        "practical_deadline_file_recovery": "practical_first",
        "gas_stove_ignition_issue": "practical_first",
        "heating_bill_anxiety": "practical_first",
        "living_cost_pressure": "practical_first",
        "money_delivery_tired_compromise": "practical_first",
        "money_stress_impulse_buying": "practical_first",
        "work_new_project_first_step": "practical_first",
        "read_receipt_uncertainty": "judgment",
        "judgment_quit_impulse_after_feedback": "judgment",
        "relationship_grievance_logic_before_rebuttal": "judgment",
        "relationship_friend_partner_complaint_fatigue": "judgment",
        "ai_human_emotion_efficiency": "meta",
    }
    priority = mapping.get(draft_frame)
    if priority is None:
        return None
    relation_type = _axis_label(frame, "relation_type")
    if relation_type == NONE_LABEL and priority in {"emotion_stabilize", "judgment"}:
        return None
    return priority, (f"draft_frame:{draft_frame}",)


def _confidence_from_relation_type(
    relation_type: str,
    *,
    frame: dict[str, Any],
    cues: tuple[str, ...],
) -> float:
    confidence = 0.8
    draft_frame = _axis_label(frame, "draft_frame")
    if relation_type in str(draft_frame) or draft_frame in relation_type:
        confidence += 0.06
    if f"relation_type:{relation_type}" in cues:
        confidence += 0.04
    if _axis_label(frame, "relation_priority") == RELATION_PRIORITY_BY_TYPE.get(relation_type):
        confidence += 0.03
    return round(min(confidence, 0.95), 4)


def _frame_evidence(frame: dict[str, Any]) -> tuple[str, ...]:
    evidence = []
    for axis in ("domain", "schema", "emotion", "state_hint", "action_hint", "draft_frame_family", "draft_frame"):
        value = _axis_label(frame, axis)
        if value != NONE_LABEL:
            evidence.append(f"{axis}:{value}")
    return tuple(evidence)


def _blocked_from_relation_type(relation_type: str) -> tuple[str, ...]:
    if relation_type == NONE_LABEL:
        return ()
    return (f"blocked_relation_type:{relation_type}",)
