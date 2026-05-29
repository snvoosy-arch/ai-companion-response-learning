from __future__ import annotations

from typing import Any

from predictive_bot.core import draft_relation_priority_resolver as v1


NONE_LABEL = v1.NONE_LABEL
RESOLVER_SOURCE = "black_relation_priority_resolver_v2_false_positive_emotion_recall"

RelationPriorityResolution = v1.RelationPriorityResolution
PRACTICAL_RELATION_TYPES = v1.PRACTICAL_RELATION_TYPES
EMOTION_RELATION_TYPES = v1.EMOTION_RELATION_TYPES
JUDGMENT_RELATION_TYPES = v1.JUDGMENT_RELATION_TYPES
META_RELATION_TYPES = v1.META_RELATION_TYPES
RELATION_PRIORITY_BY_TYPE = v1.RELATION_PRIORITY_BY_TYPE


def resolve_relation_priority(
    frame: dict[str, Any],
    *,
    raw_text: str = "",
    pragmatic_cues: list[str] | tuple[str, ...] | None = None,
) -> RelationPriorityResolution:
    text = v1._compact(raw_text or str(frame.get("text") or ""))
    cues = tuple(str(cue) for cue in (pragmatic_cues or frame.get("pragmatic_cues") or ()) if str(cue))
    relation_type = v1._axis_label(frame, "relation_type")

    if relation_type in META_RELATION_TYPES or v1._has_meta_text(text):
        return _resolution(
            "meta",
            relation_type,
            0.86,
            ("meta_relation_context", f"relation_type:{relation_type}"),
        )

    if _is_content_reference_context(frame, text):
        return _resolution(
            NONE_LABEL,
            relation_type,
            0.91,
            ("content_reference_context", *v1._frame_evidence(frame)),
            blocked=v1._blocked_from_relation_type(relation_type),
        )

    hard_none = v1._hard_none_evidence(frame=frame, compact_text=text)
    if _has_absolute_hard_none(hard_none):
        return _resolution(
            NONE_LABEL,
            relation_type,
            0.94,
            ("hard_none_context", *hard_none),
            blocked=v1._blocked_from_relation_type(relation_type),
        )

    text_priority = _text_priority_evidence_v2(text, frame)
    if hard_none and not _can_text_override_hard_none(text_priority, text, frame):
        return _resolution(
            NONE_LABEL,
            relation_type,
            0.93,
            ("hard_none_context", *hard_none),
            blocked=v1._blocked_from_relation_type(relation_type),
        )

    if text_priority is not None:
        priority, evidence = text_priority
        return _resolution(
            priority,
            relation_type,
            0.92,
            ("raw_text_priority", *evidence),
        )

    mapped_priority = RELATION_PRIORITY_BY_TYPE.get(relation_type)
    if mapped_priority is not None:
        if _relation_type_is_frame_incompatible_v2(relation_type, frame=frame, compact_text=text):
            frame_priority = _frame_priority_if_specific_v2(frame, compact_text=text)
            if frame_priority is not None:
                priority, evidence = frame_priority
                return _resolution(
                    priority,
                    relation_type,
                    0.81,
                    ("relation_type_rejected_frame_fallback", *evidence),
                    blocked=(f"relation_type:{relation_type}",),
                )
            return _resolution(
                NONE_LABEL,
                relation_type,
                0.83,
                ("relation_type_rejected_by_frame", *v1._frame_evidence(frame)),
                blocked=(f"relation_type:{relation_type}",),
            )
        return _resolution(
            mapped_priority,
            relation_type,
            _confidence_from_relation_type_v2(relation_type, frame=frame, cues=cues),
            (f"relation_type:{relation_type}", f"mapped_priority:{mapped_priority}", *v1._frame_evidence(frame)),
        )

    frame_priority = _frame_priority_if_specific_v2(frame, compact_text=text)
    if frame_priority is not None:
        priority, evidence = frame_priority
        return _resolution(
            priority,
            relation_type,
            0.79,
            ("frame_specific_priority", *evidence),
        )

    model_priority = v1._axis_label(frame, "relation_priority")
    if model_priority in {"meta"}:
        return _resolution(
            model_priority,
            relation_type,
            0.62,
            (f"model_relation_priority:{model_priority}",),
        )

    return _resolution(
        NONE_LABEL,
        relation_type,
        0.73,
        ("default_no_relation_priority", *v1._frame_evidence(frame)),
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
    cues.append("relation_priority_resolver_v2")
    cues.append(f"resolved_relation_priority:{resolution.relation_priority}")
    payload["pragmatic_cues"] = list(dict.fromkeys(cues))

    signals = list(payload.get("signals") if isinstance(payload.get("signals"), list) else [])
    signals.append(resolution.to_signal())
    payload["signals"] = signals
    return payload


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


def _text_priority_evidence_v2(
    text: str,
    frame: dict[str, Any],
) -> tuple[str, tuple[str, ...]] | None:
    if _has_immediate_practical_text_v2(text, frame):
        return "practical_first", ("immediate_practical_text_v2",)
    if _has_judgment_text_v2(text):
        return "judgment", ("judgment_text_v2",)
    if _has_emotion_stabilize_text_v2(text):
        return "emotion_stabilize", ("emotion_stabilize_text_v2",)
    if v1._has_meta_text(text):
        return "meta", ("meta_text",)
    return None


def _has_immediate_practical_text_v2(text: str, frame: dict[str, Any]) -> bool:
    if _is_content_reference_context(frame, text) or _is_late_message_text(text):
        return False
    return (
        _has_gas_stove_practical_text(text)
        or _has_heating_practical_text(text)
        or _has_living_cost_practical_text(text)
        or _has_billing_cost_practical_text(text)
        or _has_deadline_file_practical_text(text)
        or _has_kakao_tone_check_text(text)
        or _has_gas_smell_emergency_text(text)
        or _has_interview_late_practical_text(text)
        or _has_phishing_practical_text(text)
        or _has_neighbor_noise_practical_text(text)
        or _has_impulse_spending_practical_text(text)
        or _has_new_project_practical_text(text)
        or _has_delivery_compromise_text(text)
        or _has_online_scam_practical_text(text)
        or _has_appliance_review_practical_text(text)
    )


def _has_judgment_text_v2(text: str) -> bool:
    return (
        (v1._has_any(text, "읽씹", "안읽씹", "답장없", "답이없") and v1._has_any(text, "바쁜", "단정", "보류", "폰만", "결론"))
        or (
            v1._has_any(text, "퇴사", "사표", "그만두")
            and v1._has_any(text, "상사", "피드백", "충동", "자존심")
            and v1._has_any(text, "결정", "안되", "말아", "보류", "참아", "충동")
        )
        or (v1._has_any(text, "서운") and v1._has_any(text, "팩트", "논리", "반박", "감정부터"))
        or (v1._has_any(text, "주식", "투자", "코인") and v1._has_any(text, "조급", "뒤처", "손실", "한도"))
        or (v1._has_any(text, "성공기준", "인생기준") and v1._has_any(text, "보여주기", "버틸수", "조건"))
        or (v1._has_any(text, "애인흉", "감정쓰레기통", "파트너불평") and v1._has_any(text, "선그어", "의리", "헷갈", "피곤"))
    )


def _has_emotion_stabilize_text_v2(text: str) -> bool:
    return (
        v1._has_emotion_stabilize_text(text)
        or (
            v1._has_any(text, "내편", "내쪽", "기댈", "받아줄사람", "외로", "고독", "서러")
            and v1._has_any(text, "없", "혼자", "마음", "버텨", "버티", "무너", "힘들", "춥")
        )
        or (
            v1._has_any(text, "단톡", "카톡방", "단체톡")
            and v1._has_any(text, "무시", "묻히", "무반응", "반응없", "투명인간", "소외", "상처", "아무도안웃", "곱씹")
        )
        or (
            v1._has_any(text, "혼자")
            and v1._has_any(text, "버티", "버텨")
            and v1._has_any(text, "무겁", "버겁", "오늘은")
        )
        or (
            v1._has_any(text, "카톡")
            and v1._has_any(text, "무반응", "반응없")
            and v1._has_any(text, "자존감", "버겁", "상처", "내려")
        )
    )


def _is_content_reference_context(frame: dict[str, Any], text: str) -> bool:
    draft_frame = v1._axis_label(frame, "draft_frame")
    domain = v1._axis_label(frame, "domain")
    schema = v1._axis_label(frame, "schema")
    state_hint = v1._axis_label(frame, "state_hint")

    if _has_social_emotion_context_text(text):
        return False
    if draft_frame in {
        "appliance_design_review_judgment",
        "practical_online_purchase_scam",
        "money_delivery_tired_compromise",
        "money_stress_impulse_buying",
        "work_new_project_first_step",
    }:
        return False

    if state_hint in {
        "content_authoring_context",
        "content_reference_context",
        "media_reference_context",
        "word_sense_context",
        "social_relay_context",
    }:
        return True
    if draft_frame.startswith("meta_"):
        return True
    if domain in {"content_authoring", "content_reference", "language_meta", "attention_language", "content_operations"}:
        return True
    if _has_content_authoring_text(text):
        return True
    if _has_media_reference_context_text(text) and not _has_direct_action_request_text(text):
        return True
    if schema == "context_disambiguation" and _has_media_or_reference_text(text):
        return True
    if domain == "media_culture" and _has_media_or_reference_text(text) and not _has_direct_action_request_text(text):
        return True
    return False


def _has_absolute_hard_none(hard_none: tuple[str, ...]) -> bool:
    absolute = {
        "draft_frame:ai_comfort_before_emotion_proof",
        "ai_comfort_text",
        "draft_frame:career_passion_job_tradeoff",
        "career_tradeoff_text",
        "draft_frame:foundation_refusal_bad_person_guilt",
        "refusal_guilt_text",
        "draft_frame:relationship_late_message_short",
        "late_message_text",
    }
    return any(item in absolute for item in hard_none)


def _can_text_override_hard_none(
    text_priority: tuple[str, tuple[str, ...]] | None,
    text: str,
    frame: dict[str, Any],
) -> bool:
    if text_priority is None:
        return False
    priority, _evidence = text_priority
    if priority != "practical_first":
        return False
    draft_frame = v1._axis_label(frame, "draft_frame")
    return draft_frame == "money_delivery_tired_compromise" or v1._has_any(text, "배달")


def _relation_type_is_frame_incompatible_v2(
    relation_type: str,
    *,
    frame: dict[str, Any],
    compact_text: str,
) -> bool:
    draft_frame = v1._axis_label(frame, "draft_frame")
    if _is_content_reference_context(frame, compact_text):
        return True
    if relation_type == "relationship_kakao_tone_anxiety_check":
        return not _has_kakao_tone_check_text(compact_text)
    if relation_type in {"heating_bill_anxiety_practical", "living_cost_pressure_practical"}:
        return not (
            _has_heating_practical_text(compact_text)
            or _has_living_cost_practical_text(compact_text)
            or _has_billing_cost_practical_text(compact_text)
        )
    if relation_type == "gas_stove_ignition_issue_practical":
        return not _has_gas_stove_practical_text(compact_text)
    if relation_type == "online_scam_evidence_first":
        return not (_has_online_scam_practical_text(compact_text) or draft_frame == "practical_online_purchase_scam")
    if relation_type == "appliance_design_review_judgment":
        return not (_has_appliance_review_practical_text(compact_text) or draft_frame == "appliance_design_review_judgment")
    if relation_type == "neighbor_noise_record_first_practical":
        return not _has_neighbor_noise_practical_text(compact_text)
    if relation_type == "quit_after_feedback_impulse":
        return not _has_judgment_text_v2(compact_text)
    if relation_type == "friend_partner_complaint_boundary":
        return not (
            _has_judgment_text_v2(compact_text)
            or draft_frame == "relationship_friend_partner_complaint_fatigue"
        )
    return v1._relation_type_is_frame_incompatible(relation_type, frame=frame, compact_text=compact_text)


def _frame_priority_if_specific_v2(
    frame: dict[str, Any],
    *,
    compact_text: str,
) -> tuple[str, tuple[str, ...]] | None:
    draft_frame = v1._axis_label(frame, "draft_frame")
    relation_type = v1._axis_label(frame, "relation_type")

    if draft_frame == "relationship_kakao_tone_anxiety_check":
        if _has_kakao_tone_check_text(compact_text):
            return "practical_first", (f"draft_frame:{draft_frame}", "kakao_tone_check_text")
        return None
    if draft_frame in {"heating_bill_anxiety", "living_cost_pressure"}:
        if (
            _has_heating_practical_text(compact_text)
            or _has_living_cost_practical_text(compact_text)
            or _has_billing_cost_practical_text(compact_text)
        ):
            return "practical_first", (f"draft_frame:{draft_frame}", "cost_practical_text")
        return None
    if draft_frame == "gas_stove_ignition_issue":
        if _has_gas_stove_practical_text(compact_text):
            return "practical_first", (f"draft_frame:{draft_frame}", "gas_stove_practical_text")
        return None
    if draft_frame == "practical_deadline_file_recovery" and _has_deadline_file_practical_text(compact_text):
        return "practical_first", (f"draft_frame:{draft_frame}",)
    if draft_frame == "practical_online_purchase_scam" and (
        _has_online_scam_practical_text(compact_text) or relation_type == "online_scam_evidence_first"
    ):
        return "practical_first", (f"draft_frame:{draft_frame}",)
    if draft_frame == "appliance_design_review_judgment" and (
        _has_appliance_review_practical_text(compact_text) or relation_type == "appliance_design_review_judgment"
    ):
        return "practical_first", (f"draft_frame:{draft_frame}",)
    if draft_frame in {
        "money_delivery_tired_compromise",
        "money_stress_impulse_buying",
        "work_new_project_first_step",
    }:
        if _has_immediate_practical_text_v2(compact_text, frame):
            return "practical_first", (f"draft_frame:{draft_frame}",)
        return None
    if draft_frame in {"emotion_group_chat_ignored_stabilize", "grief_loneliness_no_safe_person"}:
        if relation_type != NONE_LABEL or _has_emotion_stabilize_text_v2(compact_text):
            return "emotion_stabilize", (f"draft_frame:{draft_frame}",)
        return None
    if draft_frame == "judgment_quit_impulse_after_feedback":
        if _has_judgment_text_v2(compact_text):
            return "judgment", (f"draft_frame:{draft_frame}",)
        return None
    if draft_frame in {
        "read_receipt_uncertainty",
        "relationship_grievance_logic_before_rebuttal",
        "relationship_friend_partner_complaint_fatigue",
    }:
        if relation_type != NONE_LABEL or _has_judgment_text_v2(compact_text):
            return "judgment", (f"draft_frame:{draft_frame}",)
        return None
    if draft_frame == "ai_human_emotion_efficiency":
        return "meta", (f"draft_frame:{draft_frame}",)
    return None


def _confidence_from_relation_type_v2(
    relation_type: str,
    *,
    frame: dict[str, Any],
    cues: tuple[str, ...],
) -> float:
    confidence = v1._confidence_from_relation_type(relation_type, frame=frame, cues=cues)
    if relation_type in {
        "relationship_kakao_tone_anxiety_check",
        "heating_bill_anxiety_practical",
        "living_cost_pressure_practical",
        "gas_stove_ignition_issue_practical",
    }:
        return round(min(confidence + 0.01, 0.95), 4)
    return confidence


def _has_gas_stove_practical_text(text: str) -> bool:
    return (
        v1._has_any(text, "가스레인지", "가스렌지", "가스버너", "화구", "버너")
        and v1._has_any(text, "점화", "딸깍", "불이안", "불안붙", "불안켜", "불꽃")
    )


def _has_heating_practical_text(text: str) -> bool:
    if not v1._has_any(text, "가스비", "도시가스비", "난방비", "보일러", "난방", "히터", "공과금", "관리비", "전기요금"):
        return False
    if v1._has_any(text, "경보") and v1._has_any(text, "느낌") and not _has_direct_action_request_text(text):
        return False
    if v1._has_any(text, "감기", "아프") and not v1._has_any(
        text,
        "어떻게",
        "뭐부터",
        "온도",
        "외출모드",
        "켜",
        "꺼",
        "해야",
        "될까",
        "방법",
        "줄여",
    ):
        return False
    return (
        v1._has_any(
            text,
            "보일러켜기",
            "난방켜기",
            "히터켜기",
            "켜기무서",
            "켜기고민",
            "켜는게불안",
            "켜려니",
            "켜자니",
            "안켜자니",
            "켤때",
            "예약난방",
            "난방예약",
            "온수",
            "온도",
            "외출모드",
            "절약방법",
            "뭐부터",
            "어떻게",
        )
        or v1._has_any(text, "줄여야", "줄일", "켜야", "꺼야", "계획", "순서", "계산하게", "터졌", "신경쓰")
        or (
            v1._has_any(text, "무서", "부담", "걱정", "겁나", "불안", "신경", "손이떨", "눈치")
            and v1._has_any(text, "보일러", "난방", "히터", "온수")
        )
    )


def _has_living_cost_practical_text(text: str) -> bool:
    return (
        v1._has_any(text, "식료품값", "식비", "물가", "마트", "장보기", "장바구니", "주유비", "휘발유값", "기름값", "기름")
        and v1._has_any(text, "예산", "생활비", "지갑", "계산대", "돈", "값", "지출", "영수증", "주유소", "마트", "장바구니")
        and v1._has_any(
            text,
            "무너",
            "불안",
            "겁나",
            "무서",
            "힘들",
            "비싸",
            "올라",
            "고민",
            "줄여",
            "커져",
            "흔들",
            "눈치",
            "신경쓰",
            "터질",
            "터졌",
            "문제",
            "부담",
        )
    )


def _has_billing_cost_practical_text(text: str) -> bool:
    if v1._has_any(text, "경보") and v1._has_any(text, "느낌") and not _has_direct_action_request_text(text):
        return False
    return (
        v1._has_any(text, "공과금", "관리비", "고지서", "전기요금", "도시가스비", "난방비")
        and v1._has_any(
            text,
            "생활비",
            "예산",
            "불안",
            "부담",
            "무서",
            "고민",
            "줄여",
            "이번달",
            "흔들",
            "신경",
            "겁나",
            "손이떨",
            "눈치",
            "온수",
            "예약난방",
            "난방예약",
        )
    )


def _has_social_emotion_context_text(text: str) -> bool:
    return (
        (
            v1._has_any(text, "단톡", "카톡방", "단체톡")
            and v1._has_any(text, "무반응", "반응없", "아무도안웃", "곱씹", "묻히", "상처")
        )
        or (
            v1._has_any(text, "카톡")
            and v1._has_any(text, "무반응", "반응없", "자존감", "버겁")
        )
        or (
            v1._has_any(text, "혼자")
            and v1._has_any(text, "버티", "무겁", "버겁")
        )
        or (
            v1._has_any(text, "내쪽", "내편")
            and v1._has_any(text, "아무도", "마음", "춥")
        )
    )


def _has_deadline_file_practical_text(text: str) -> bool:
    return (
        v1._has_any(text, "마감", "제출", "과제", "보고서", "회의자료", "발표자료")
        and v1._has_any(text, "파일", "문서", "자료")
        and v1._has_any(text, "날아", "복구", "저장", "꺼졌", "안열", "깨졌", "삭제", "덮어쓴")
    )


def _has_kakao_tone_check_text(text: str) -> bool:
    return (
        v1._has_any(text, "카톡", "답장", "말투", "톤")
        and v1._has_any(text, "차가워", "딱딱", "건조", "달라져")
        and v1._has_any(text, "확인", "물어", "추궁", "체크", "봐야")
    )


def _has_gas_smell_emergency_text(text: str) -> bool:
    return v1._has_any(text, "가스냄새", "가스누출") and v1._has_any(
        text,
        "창문",
        "환기",
        "밸브",
        "119",
        "관리사무소",
        "나가",
    )


def _has_interview_late_practical_text(text: str) -> bool:
    return v1._has_any(text, "면접", "버스놓", "늦") and v1._has_any(text, "택시", "담당자", "연락")


def _has_phishing_practical_text(text: str) -> bool:
    return v1._has_any(text, "피싱", "문자", "링크") and v1._has_any(text, "계좌", "비밀번호", "차단", "카드")


def _has_neighbor_noise_practical_text(text: str) -> bool:
    return v1._has_any(text, "옆집", "층간소음", "소음", "쿵쾅") and v1._has_any(
        text,
        "관리사무소",
        "기록",
        "쪽지",
        "녹음",
    )


def _has_impulse_spending_practical_text(text: str) -> bool:
    return v1._has_any(text, "충동구매", "스트레스받으면사", "돈모으") and v1._has_any(
        text,
        "장치",
        "막는",
        "차단",
        "걸까",
    )


def _has_new_project_practical_text(text: str) -> bool:
    return v1._has_any(text, "새프로젝트", "새프로젝트맡", "프로젝트맡") and v1._has_any(
        text,
        "첫단추",
        "뭐부터",
        "막막",
    )


def _has_delivery_compromise_text(text: str) -> bool:
    return v1._has_any(text, "배달") and v1._has_any(text, "지쳐", "아무것도못", "돈아끼", "합리적")


def _has_online_scam_practical_text(text: str) -> bool:
    return (
        v1._has_any(text, "온라인물건", "중고거래", "거래", "반품거부", "사기", "판매자")
        and v1._has_any(text, "증거", "캡처", "신고", "환불", "반품", "계좌", "기록")
    )


def _has_appliance_review_practical_text(text: str) -> bool:
    return (
        v1._has_any(text, "음식물처리기", "식기세척기", "공기청정기", "체온계", "가전", "제품")
        and v1._has_any(text, "리뷰", "디자인", "성능", "구매", "사도", "살까")
        and v1._has_any(text, "사도될", "판단", "고민", "선택", "살까", "구매해도")
    )


def _has_content_authoring_text(text: str) -> bool:
    return (
        v1._has_any(text, "블로그", "제목", "콘텐츠", "쇼츠", "대본", "광고문구", "소개글", "카피")
        and v1._has_any(text, "써", "추천", "만들", "소개", "문장", "표현", "제목")
    )


def _has_media_or_reference_text(text: str) -> bool:
    return v1._has_any(
        text,
        "영상",
        "드라마",
        "영화",
        "리뷰",
        "댓글",
        "기사",
        "설명",
        "표현",
        "단어",
        "뜻",
        "짤",
    )


def _has_media_reference_context_text(text: str) -> bool:
    return (
        _has_media_or_reference_text(text)
        and v1._has_any(text, "봤", "보니까", "보는데", "영상", "쇼츠", "설명", "댓글", "기사", "리뷰")
        and not v1._has_any(text, "못해먹", "못해먹고", "고장인지", "지금안", "안켜져서")
    )


def _has_direct_action_request_text(text: str) -> bool:
    return v1._has_any(
        text,
        "어떻게",
        "뭐부터",
        "해야",
        "될까",
        "잡아줘",
        "판단해줘",
        "먼저야",
        "확인해",
        "해결",
        "줄여",
        "사도될",
    )


def _is_late_message_text(text: str) -> bool:
    return v1._has_any(text, "약속", "지각", "늦") and v1._has_any(
        text,
        "연락",
        "문자",
        "메시지",
        "뭐라고",
    )
