from __future__ import annotations

from typing import Any

from predictive_bot.core import draft_relation_priority_resolver as v1
from predictive_bot.core import draft_relation_priority_resolver_v3 as v3


NONE_LABEL = v3.NONE_LABEL
RESOLVER_SOURCE = "black_relation_priority_resolver_v4_practical_residual_repair"

RelationPriorityResolution = v3.RelationPriorityResolution


def resolve_relation_priority(
    frame: dict[str, Any],
    *,
    raw_text: str = "",
    pragmatic_cues: list[str] | tuple[str, ...] | None = None,
) -> RelationPriorityResolution:
    base = v3.resolve_relation_priority(
        frame,
        raw_text=raw_text,
        pragmatic_cues=pragmatic_cues,
    )
    text = v1._compact(raw_text or str(frame.get("text") or ""))

    boundary_evidence = _practical_boundary_evidence_v4(text)
    if boundary_evidence:
        return _resolution(
            NONE_LABEL,
            base.relation_type,
            0.92,
            ("boundary_text_v4", *boundary_evidence),
            blocked=tuple(base.evidence),
        )

    if base.relation_priority == "practical_first":
        return _wrap_base(base)

    practical_evidence = _practical_text_evidence_v4(text, frame)
    if practical_evidence:
        return _resolution(
            "practical_first",
            base.relation_type,
            0.9,
            ("raw_text_priority_v4", *practical_evidence),
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
    cues.append("relation_priority_resolver_v4")
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
        ("resolver_v3_base", *tuple(base.evidence)),
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


def _practical_boundary_evidence_v4(text: str) -> tuple[str, ...]:
    if _has_gas_stove_status_only_boundary_text_v4(text):
        return ("gas_stove_status_only_boundary",)
    return ()


def _practical_text_evidence_v4(text: str, frame: dict[str, Any]) -> tuple[str, ...]:
    if _has_gas_stove_practical_text_v4(text):
        return ("practical_text_v4", "gas_stove_action_or_safety")
    if _has_cost_practical_text_v4(text):
        return ("practical_text_v4", "cost_action_or_budget_decision")
    if _has_file_recovery_practical_text_v4(text):
        return ("practical_text_v4", "file_recovery_or_deadline")
    if _has_relationship_check_practical_text_v4(text):
        return ("practical_text_v4", "relationship_short_check_action")
    if _has_product_measurement_review_practical_text_v4(text):
        return ("practical_text_v4", "product_measurement_review_confidence")
    return ()


def _has_gas_stove_status_only_boundary_text_v4(text: str) -> bool:
    if not v1._has_any(text, "가스레인지", "가스렌지"):
        return False
    if not v1._has_any(text, "밥"):
        return False
    if not v1._has_any(text, "못해먹", "못해먹고", "할때마다긴장", "밥할때마다긴장"):
        return False
    return not _has_gas_stove_direct_action_marker_v4(text)


def _has_gas_stove_practical_text_v4(text: str) -> bool:
    has_ignition_surface = v1._has_any(
        text,
        "가스레인지",
        "가스렌지",
        "점화",
        "불이안",
        "불안붙",
        "불안켜",
        "안켜져",
        "불꽃",
    )
    return has_ignition_surface and _has_gas_stove_direct_action_marker_v4(text)


def _has_gas_stove_direct_action_marker_v4(text: str) -> bool:
    return v1._has_any(
        text,
        "첫확인순서",
        "확인순서",
        "확인할게",
        "기사부르기전",
        "고장인지",
        "단순막힘인지",
        "막힘인지",
        "라이터",
        "붙여도되는지",
        "사용해도되는지",
        "가스냄새",
        "냄새는없",
        "애매해",
        "불꽃이약",
        "약하게만",
    )


def _has_cost_practical_text_v4(text: str) -> bool:
    if v1._has_any(text, "식비쪽", "식비쪾") and v1._has_any(text, "예산경보"):
        return True
    if v1._has_any(text, "고정비") and v1._has_any(text, "취미지출", "지출") and v1._has_any(text, "멈춰야할지", "줄여야할지", "고민"):
        return True
    if v1._has_any(text, "식비아끼려고", "장을봤는데") and v1._has_any(text, "다음엔어떻게", "오히려더쓴"):
        return True
    if v1._has_any(text, "보일러", "보일러비", "난방비") and v1._has_any(
        text,
        "예약을줄이면",
        "기준이필요",
        "돈이새는느낌",
        "자꾸참게",
        "아끼려다감기",
        "감기걸릴까",
    ):
        return True
    return False


def _has_file_recovery_practical_text_v4(text: str) -> bool:
    return (
        v1._has_any(text, "USB", "파일", "자동저장", "작업물", "구글드라이브", "드라이브")
        and v1._has_any(
            text,
            "안보",
            "안보여",
            "제출",
            "어디서확인",
            "확인하는지",
            "기록부터",
            "날아간",
            "날아간것",
            "복구",
            "저장",
        )
    )


def _has_relationship_check_practical_text_v4(text: str) -> bool:
    has_relation_surface = v1._has_any(text, "카톡", "답장", "말투", "장문", "상대", "바쁜건지", "식은건지")
    has_check_action = v1._has_any(
        text,
        "확인한줄",
        "확인하고싶",
        "먼저확인",
        "물어보는게낫",
        "짧게물어",
        "체크만",
        "체크만할까",
        "추궁말고",
    )
    has_overread_guard = v1._has_any(text, "단정", "상상만커", "불안", "차가운답장", "바쁜건지", "식은건지", "장문보내기전")
    return has_relation_surface and has_check_action and has_overread_guard


def _has_product_measurement_review_practical_text_v4(text: str) -> bool:
    return (
        v1._has_any(text, "체온계")
        and v1._has_any(text, "리뷰")
        and v1._has_any(text, "정상측정", "측정")
        and v1._has_any(text, "반복")
    )
