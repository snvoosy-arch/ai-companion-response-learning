from __future__ import annotations

from bot_shared.vtuber import VTuberActionCue, VTuberTurnPacket

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    Intent,
    MessageFeatures,
    PolicyTrace,
    VerificationReport,
    WeatherReport,
    WorldState,
)


FACTUAL_ACTIONS = {
    ActionType.WEATHER_LOOKUP,
    ActionType.SEARCH_ANSWER,
    ActionType.NEWS_ANSWER,
    ActionType.TELL_TIME,
}


def build_black_turn_packet(
    *,
    reply: str,
    features: MessageFeatures,
    decision: ActionDecision,
    world_state: WorldState | None,
    policy_trace: PolicyTrace | None,
    verification: VerificationReport | None,
    weather: WeatherReport | None,
    llm_used: bool,
    llm_fallback_reason: str | None,
) -> VTuberTurnPacket:
    emotion_state = _emotion_state(features=features, decision=decision, world_state=world_state)
    facial_expression = _facial_expression(
        emotion_state=emotion_state,
        features=features,
        decision=decision,
        world_state=world_state,
    )
    confidence = _confidence(features=features, policy_trace=policy_trace)
    priority = _priority(decision=decision, world_state=world_state)
    return VTuberTurnPacket(
        speaker="black",
        text=reply,
        brain="black.predictive_policy",
        emotion_state=emotion_state,
        action_intent=decision.action.value,
        facial_expression=facial_expression,
        voice_style=_voice_style(emotion_state),
        priority=priority,
        can_interrupt=priority != "high",
        confidence=confidence,
        evidence_used=_evidence_used(
            decision=decision,
            world_state=world_state,
            weather=weather,
        ),
        policy_trace_summary=_policy_trace_summary(
            decision=decision,
            policy_trace=policy_trace,
            world_state=world_state,
        ),
        safety_notes=_safety_notes(verification=verification, llm_fallback_reason=llm_fallback_reason),
        action_cues=_action_cues(
            emotion_state=emotion_state,
            facial_expression=facial_expression,
            decision=decision,
            world_state=world_state,
        ),
        metadata={
            "intent": features.intent.value,
            "sentiment": features.sentiment,
            "reason_code": decision.reason_code,
            "response_plan_stance": decision.response_plan.stance if decision.response_plan else "",
            "response_plan_anchor": decision.response_plan.anchor if decision.response_plan else "",
            "llm_used": "true" if llm_used else "false",
            "conversation_mode": world_state.conversation_mode if world_state else "",
            "risk_level": world_state.risk_level if world_state else "",
            "verification_ok": "true" if verification is None or verification.ok else "false",
        },
    )


def _emotion_state(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    world_state: WorldState | None,
) -> str:
    cues = set(features.pragmatic_cues)
    risk_level = world_state.risk_level if world_state else "low"
    user_emotion = world_state.user_emotion if world_state else features.sentiment
    if decision.action == ActionType.DEESCALATE or risk_level == "high":
        return "steady"
    if decision.action in FACTUAL_ACTIONS:
        return "grounded"
    if decision.action in {ActionType.ASK_LOCATION, ActionType.ASK_CLARIFICATION}:
        return "attentive"
    if "subdued_positive" in cues:
        return "subdued_positive"
    if decision.action in {ActionType.REACT_LAUGH, ActionType.TEASE_BACK} or features.intent == Intent.LAUGH:
        return "amused"
    if decision.action == ActionType.REACT_SURPRISE or features.intent == Intent.SURPRISE:
        return "surprised"
    if features.sentiment == "negative" or user_emotion == "negative":
        return "low_steady"
    if features.sentiment == "positive":
        return "warm"
    if decision.action in {ActionType.SHARE_FEELING, ActionType.ACKNOWLEDGE}:
        return "soft"
    return "neutral"


def _facial_expression(
    *,
    emotion_state: str,
    features: MessageFeatures,
    decision: ActionDecision,
    world_state: WorldState | None,
) -> str:
    if decision.action in FACTUAL_ACTIONS:
        return "focused"
    if decision.action in {ActionType.ASK_LOCATION, ActionType.ASK_CLARIFICATION}:
        return "attentive"
    if emotion_state in {"amused", "warm"}:
        return "smile"
    if emotion_state == "surprised":
        return "surprised"
    if emotion_state in {"steady", "low_steady"}:
        return "steady"
    if emotion_state in {"soft", "subdued_positive"}:
        return "soft"
    if world_state is not None and world_state.tension_bucket in {"tense", "hot"}:
        return "steady"
    if features.is_question:
        return "attentive"
    return "neutral"


def _voice_style(emotion_state: str) -> str:
    if emotion_state == "grounded":
        return "black_grounded"
    if emotion_state in {"steady", "low_steady"}:
        return "black_steady"
    if emotion_state in {"soft", "subdued_positive", "warm"}:
        return "black_soft"
    if emotion_state in {"amused", "surprised"}:
        return "black_light"
    if emotion_state == "attentive":
        return "black_clear"
    return "black_neutral"


def _priority(*, decision: ActionDecision, world_state: WorldState | None) -> str:
    if decision.action == ActionType.DEESCALATE:
        return "high"
    if world_state is not None and world_state.risk_level == "high":
        return "high"
    if decision.action in {ActionType.ASK_LOCATION, ActionType.ASK_CLARIFICATION}:
        return "normal"
    return "normal"


def _confidence(*, features: MessageFeatures, policy_trace: PolicyTrace | None) -> float:
    scores: list[float] = []
    evidence = features.classifier_evidence
    if evidence is not None:
        scores.extend(float(item.score) for item in evidence.top_scores[:3])
    if policy_trace is not None:
        scores.extend(float(item.score) for item in policy_trace.candidates[:3])
    if not scores:
        return 0.65
    return round(max(0.0, min(max(scores), 1.0)), 3)


def _evidence_used(
    *,
    decision: ActionDecision,
    world_state: WorldState | None,
    weather: WeatherReport | None,
) -> list[str]:
    evidence: list[str] = []
    if weather is not None:
        evidence.append(
            f"weather:{weather.location}:{weather.description}:{weather.temperature_c:.1f}C"
        )
    source = decision.slots.get("knowledge_source")
    if source:
        evidence.append(f"source:{source}")
    for key in ("knowledge_answer", "time_text", "news_summary", "recommendation_titles", "music_titles"):
        value = decision.slots.get(key)
        if value:
            evidence.append(f"{key}:{_compact(value)}")
    if world_state is not None:
        evidence.extend(world_state.evidence[:4])
        for node in world_state.evidence_nodes[:4]:
            evidence.append(f"{node.label}={node.value}")
    return _dedupe(evidence, limit=8)


def _policy_trace_summary(
    *,
    decision: ActionDecision,
    policy_trace: PolicyTrace | None,
    world_state: WorldState | None,
) -> list[str]:
    summary = [
        f"action={decision.action.value}",
        f"reason_code={decision.reason_code}",
    ]
    if decision.reason_flags:
        summary.append("reason_flags=" + ",".join(decision.reason_flags[:5]))
    if world_state is not None:
        summary.append(f"mode={world_state.conversation_mode}")
        summary.append(f"risk={world_state.risk_level}")
        if world_state.unresolved_need:
            summary.append(f"unresolved={world_state.unresolved_need}")
    if policy_trace is not None:
        for candidate in policy_trace.candidates[:3]:
            summary.append(f"candidate:{candidate.action.value}:{candidate.score:.3f}")
    return summary[:8]


def _safety_notes(
    *,
    verification: VerificationReport | None,
    llm_fallback_reason: str | None,
) -> list[str]:
    notes: list[str] = []
    if verification is not None and verification.issues:
        notes.extend(f"verification:{issue}" for issue in verification.issues[:4])
    if llm_fallback_reason:
        notes.append(f"generation:{llm_fallback_reason}")
    return notes


def _expression_intensity(world_state: WorldState | None) -> float:
    if world_state is None:
        return 0.55
    if world_state.risk_level == "high":
        return 0.85
    if world_state.tension_bucket in {"tense", "hot"}:
        return 0.75
    if world_state.rapport_bucket in {"warm", "close"}:
        return 0.65
    return 0.55


def _action_cues(
    *,
    emotion_state: str,
    facial_expression: str,
    decision: ActionDecision,
    world_state: WorldState | None,
) -> list[VTuberActionCue]:
    intensity = _expression_intensity(world_state)
    cues = [
        VTuberActionCue(kind="speak", intensity=1.0),
        VTuberActionCue(kind=f"expression:{facial_expression}", intensity=intensity),
        VTuberActionCue(kind="gaze:viewer", intensity=0.65),
    ]
    if decision.action in FACTUAL_ACTIONS or emotion_state == "grounded":
        cues.extend(
            [
                VTuberActionCue(kind="posture:focused", intensity=0.72),
                VTuberActionCue(kind="gesture:small_nod", intensity=0.42),
            ]
        )
    elif decision.action in {ActionType.ASK_LOCATION, ActionType.ASK_CLARIFICATION}:
        cues.extend(
            [
                VTuberActionCue(kind="gesture:lean_in", intensity=0.58),
                VTuberActionCue(kind="gaze:viewer", intensity=0.85),
            ]
        )
    elif decision.action == ActionType.DEESCALATE or emotion_state in {"steady", "low_steady"}:
        cues.extend(
            [
                VTuberActionCue(kind="posture:steady", intensity=0.82),
                VTuberActionCue(kind="gesture:slow_nod", intensity=0.34),
            ]
        )
    elif emotion_state in {"amused", "surprised"}:
        cues.append(VTuberActionCue(kind="gesture:small_bounce", intensity=0.54))
    elif emotion_state in {"soft", "subdued_positive", "warm"}:
        cues.append(VTuberActionCue(kind="posture:soft", intensity=0.48))
    return cues


def _compact(value: str, *, limit: int = 80) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "..."


def _dedupe(items: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        compact = _compact(item)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        result.append(compact)
        if len(result) >= limit:
            break
    return result
