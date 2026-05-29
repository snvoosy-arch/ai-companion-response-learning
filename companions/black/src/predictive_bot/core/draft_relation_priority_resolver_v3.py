from __future__ import annotations

from typing import Any

from predictive_bot.core import draft_relation_priority_resolver as v1
from predictive_bot.core import draft_relation_priority_resolver_v2 as v2


NONE_LABEL = v2.NONE_LABEL
RESOLVER_SOURCE = "black_relation_priority_resolver_v3_judgment_emotion_recall"

RelationPriorityResolution = v2.RelationPriorityResolution


def resolve_relation_priority(
    frame: dict[str, Any],
    *,
    raw_text: str = "",
    pragmatic_cues: list[str] | tuple[str, ...] | None = None,
) -> RelationPriorityResolution:
    base = v2.resolve_relation_priority(
        frame,
        raw_text=raw_text,
        pragmatic_cues=pragmatic_cues,
    )
    text = v1._compact(raw_text or str(frame.get("text") or ""))

    if base.relation_priority == "practical_first":
        return _wrap_base(base)

    if _has_animal_loneliness_boundary_text(text):
        return _resolution(
            NONE_LABEL,
            base.relation_type,
            0.9,
            ("boundary_text_v3", "animal_talk_loneliness_boundary"),
            blocked=tuple(base.evidence),
        )

    if _has_emotion_over_judgment_text_v3(text):
        return _resolution(
            "emotion_stabilize",
            base.relation_type,
            0.91,
            ("raw_text_priority_v3", "emotion_over_judgment_text_v3"),
            blocked=tuple(base.evidence) if base.relation_priority != NONE_LABEL else (),
        )

    if base.relation_priority == NONE_LABEL:
        judgment_evidence = _judgment_text_evidence_v3(text)
        if judgment_evidence:
            return _resolution(
                "judgment",
                base.relation_type,
                0.9,
                ("raw_text_priority_v3", *judgment_evidence),
                blocked=tuple(base.evidence),
            )
        emotion_evidence = _emotion_text_evidence_v3(text)
        if emotion_evidence:
            return _resolution(
                "emotion_stabilize",
                base.relation_type,
                0.9,
                ("raw_text_priority_v3", *emotion_evidence),
                blocked=tuple(base.evidence),
            )

    return _wrap_base(base)


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
    cues.append("relation_priority_resolver_v3")
    cues.append(f"resolved_relation_priority:{resolution.relation_priority}")
    payload["pragmatic_cues"] = list(dict.fromkeys(cues))

    signals = list(payload.get("signals") if isinstance(payload.get("signals"), list) else [])
    signals.append(resolution.to_signal())
    payload["signals"] = signals
    return payload


def _wrap_base(base: RelationPriorityResolution) -> RelationPriorityResolution:
    return _resolution(
        base.relation_priority,
        base.relation_type,
        base.confidence,
        ("resolver_v2_base", *tuple(base.evidence)),
        blocked=tuple(base.blocked_evidence),
    )


def _resolution(
    priority: str,
    relation_type: str,
    confidence: float,
    evidence: tuple[str, ...],
    *,
    blocked: tuple[str, ...] = (),
) -> RelationPriorityResolution:
    return RelationPriorityResolution(
        relation_priority=priority,
        relation_type=relation_type,
        confidence=confidence,
        source=RESOLVER_SOURCE,
        evidence=tuple(dict.fromkeys(evidence)),
        blocked_evidence=tuple(dict.fromkeys(blocked)),
    )


def _judgment_text_evidence_v3(text: str) -> tuple[str, ...]:
    if v1._has_any(text, "계속확인", "계속확인하게") and v1._has_any(text, "판단은아직이른", "아직이른"):
        return ("judgment_text_v3", "hold_judgment_check_loop")
    if v1._has_any(text, "상대가상처받", "상처받았다면") and v1._has_any(text, "먼저확인", "확인해야"):
        return ("judgment_text_v3", "hurt_acknowledgement_check")
    if v1._has_any(text, "논리") and v1._has_any(text, "감정") and v1._has_any(text, "순서", "어렵"):
        return ("judgment_text_v3", "logic_emotion_order")
    if v1._has_any(text, "답장없", "답장없음", "답장을기다리", "친구답장") and v1._has_any(
        text,
        "관계를망",
        "망했다고",
        "혼자결론",
        "결론내릴",
        "보면안",
    ):
        return ("judgment_text_v3", "reply_absence_overconclusion")
    if v1._has_any(text, "반박") and v1._has_any(text, "싸움", "첫문장", "문장"):
        return ("judgment_text_v3", "rebuttal_first_sentence")
    if v1._has_any(text, "사표", "그만두", "나가고싶") and v1._has_any(
        text,
        "감정인지",
        "누적문제",
        "판단기준",
        "기준이필요",
        "나눠",
    ):
        return ("judgment_text_v3", "quit_impulse_judgment_basis")
    if v1._has_any(text, "서운함", "서운") and v1._has_any(text, "인정") and v1._has_any(text, "내잘못", "잘못인정"):
        return ("judgment_text_v3", "hurt_acknowledgement_boundary")
    if v1._has_any(text, "피드백") and v1._has_any(text, "자존감") and v1._has_any(text, "그만두", "그만두고싶"):
        return ("judgment_text_v3", "feedback_quit_after_selfworth_drop")
    return ()


def _emotion_text_evidence_v3(text: str) -> tuple[str, ...]:
    if v1._has_any(text, "내말만", "내메시지", "내메시지는") and v1._has_any(
        text,
        "반응이약",
        "묻히",
        "못본척",
        "넘긴",
        "사라지는",
    ) and v1._has_any(text, "소외감", "자존감", "서럽", "서운함", "마음부터"):
        return ("emotion_text_v3", "message_ignored_selfworth")
    if v1._has_any(text, "다들바빠") and v1._has_any(text, "힘든얘기", "꺼내기가어렵"):
        return ("emotion_text_v3", "hard_to_disclose_distress")
    if v1._has_any(text, "단톡") and v1._has_any(
        text,
        "분위기",
        "인간관계가다가짜",
        "나한테만조용",
        "웃고떠드는데",
    ):
        return ("emotion_text_v3", "group_chat_social_exclusion")
    if v1._has_any(text, "마음둘데가없", "마음둘데없"):
        return ("emotion_text_v3", "no_safe_place_for_feelings")
    if v1._has_any(text, "반응없는걸", "반응없") and v1._has_any(text, "마음이안따라", "마음이안"):
        return ("emotion_text_v3", "reaction_hurt_not_settled")
    if v1._has_any(text, "사소한말") and v1._has_any(text, "혼자남겨진", "남겨진느낌"):
        return ("emotion_text_v3", "small_comment_abandonment_feeling")
    if v1._has_any(text, "안전한편") and v1._has_any(text, "먼저필요", "먼저필요한"):
        return ("emotion_text_v3", "safe_ally_needed_first")
    if v1._has_any(text, "친구들이") and v1._has_any(text, "마음은이미크게다쳤", "서운함이커", "내얘기만넘긴"):
        return ("emotion_text_v3", "friends_hurt_after_relay")
    if v1._has_any(text, "카톡방") and v1._has_any(text, "내말만사라지는", "마음부터잡"):
        return ("emotion_text_v3", "chatroom_message_disappears")
    return ()


def _has_emotion_over_judgment_text_v3(text: str) -> bool:
    return (
        v1._has_any(text, "읽씹인지바쁜건지", "바쁜건지읽씹인지")
        and v1._has_any(text, "마음이불편", "일단마음")
        and not v1._has_any(text, "판단보류", "단정보류", "확정", "결론")
    )


def _has_animal_loneliness_boundary_text(text: str) -> bool:
    return (
        v1._has_any(text, "동물한테만", "동물에게만")
        and v1._has_any(text, "말걸")
        and v1._has_any(text, "외로", "마음이계속커")
    )
