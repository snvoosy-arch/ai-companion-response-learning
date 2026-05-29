from __future__ import annotations

from dataclasses import asdict, dataclass, field

from bot_shared.vtuber import VTuberActionCue, VTuberTurnPacket

from .web_search import SearchContext


@dataclass(slots=True)
class WhiteRuntimeEvent:
    kind: str
    prompt: str
    user_name: str
    reply_mode: str = "reply"
    has_images: bool = False
    search_used: bool = False
    duo: bool = False


@dataclass(slots=True)
class WhiteMoodState:
    mood: str = "neutral"
    energy: float = 0.72
    attention_target: str = "chat"
    last_event_kind: str = "none"
    event_count: int = 0

    def to_metadata(self) -> dict[str, str]:
        return {
            "mood": self.mood,
            "energy": f"{self.energy:.3f}",
            "attention_target": self.attention_target,
            "last_event_kind": self.last_event_kind,
            "event_count": str(self.event_count),
        }


@dataclass(slots=True)
class WhiteOutputPacket:
    speaker: str
    text: str
    emotion: str
    avatar_action: str
    mouth_mode: str = "speech"
    reply_priority: str = "normal"
    can_interrupt: bool = True
    action_intent: str = "chat_reply"
    facial_expression: str = "neutral"
    voice_style: str = "white_neutral"
    evidence_used: list[str] = field(default_factory=list)
    action_cues: list[VTuberActionCue] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    schema_version: str = "white.output.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class WhitePerformanceBrain:
    def __init__(self, *, max_recent_events: int = 16) -> None:
        self.max_recent_events = max(1, max_recent_events)
        self.mood_state = WhiteMoodState()
        self._recent_events: list[WhiteRuntimeEvent] = []

    @property
    def recent_events(self) -> tuple[WhiteRuntimeEvent, ...]:
        return tuple(self._recent_events)

    def observe_event(self, event: WhiteRuntimeEvent) -> WhiteMoodState:
        self._recent_events.append(event)
        if len(self._recent_events) > self.max_recent_events:
            self._recent_events = self._recent_events[-self.max_recent_events :]
        self.mood_state = self._next_mood_state(event)
        return self.mood_state

    def build_turn_packet(
        self,
        *,
        event: WhiteRuntimeEvent,
        reply: str,
        search_context: SearchContext | None = None,
    ) -> VTuberTurnPacket:
        mood = self.observe_event(event)
        action_intent = _action_intent(event=event, mood=mood)
        expression = _facial_expression(mood=mood, event=event)
        output_packet = self._build_output_packet_from_mood(
            event=event,
            reply=reply,
            mood=mood,
            action_intent=action_intent,
            expression=expression,
            search_context=search_context,
        )
        return VTuberTurnPacket(
            speaker="white",
            text=output_packet.text,
            brain="white.llm_event_loop",
            emotion_state=output_packet.emotion,
            action_intent=output_packet.action_intent,
            facial_expression=output_packet.facial_expression,
            voice_style=output_packet.voice_style,
            priority=output_packet.reply_priority,
            can_interrupt=output_packet.can_interrupt,
            confidence=_confidence(reply=output_packet.text, search_context=search_context),
            evidence_used=output_packet.evidence_used,
            policy_trace_summary=[
                f"event={event.kind}",
                f"reply_mode={event.reply_mode}",
                f"search_used={'true' if event.search_used else 'false'}",
                f"duo={'true' if event.duo else 'false'}",
                f"attention={mood.attention_target}",
            ],
            action_cues=output_packet.action_cues,
            metadata={
                **output_packet.metadata,
                "white_output_schema": output_packet.schema_version,
                "avatar_action": output_packet.avatar_action,
                "mouth_mode": output_packet.mouth_mode,
            },
        )

    def build_output_packet(
        self,
        *,
        event: WhiteRuntimeEvent,
        reply: str,
        search_context: SearchContext | None = None,
    ) -> WhiteOutputPacket:
        mood = self.observe_event(event)
        action_intent = _action_intent(event=event, mood=mood)
        expression = _facial_expression(mood=mood, event=event)
        return self._build_output_packet_from_mood(
            event=event,
            reply=reply,
            mood=mood,
            action_intent=action_intent,
            expression=expression,
            search_context=search_context,
        )

    def _build_output_packet_from_mood(
        self,
        *,
        event: WhiteRuntimeEvent,
        reply: str,
        mood: WhiteMoodState,
        action_intent: str,
        expression: str,
        search_context: SearchContext | None,
    ) -> WhiteOutputPacket:
        cues = _action_cues(event=event, mood=mood, expression=expression)
        return WhiteOutputPacket(
            speaker="white",
            text=reply,
            emotion=mood.mood,
            avatar_action=_avatar_action(event=event, mood=mood, expression=expression),
            mouth_mode="speech" if reply.strip() else "idle",
            reply_priority=_priority(event),
            can_interrupt=not event.duo,
            action_intent=action_intent,
            facial_expression=expression,
            voice_style=_voice_style(mood.mood),
            evidence_used=_evidence_used(search_context),
            action_cues=cues,
            metadata={
                **mood.to_metadata(),
                "user_name": event.user_name,
                "has_images": "true" if event.has_images else "false",
            },
        )

    def _next_mood_state(self, event: WhiteRuntimeEvent) -> WhiteMoodState:
        prompt = event.prompt.strip().lower()
        mood = "neutral"
        attention = "chat"
        energy_delta = 0.0
        if event.search_used:
            mood = "focused"
            attention = "search"
            energy_delta = -0.02
        if event.has_images:
            mood = "curious"
            attention = "image"
            energy_delta = 0.04
        if event.duo:
            mood = "playful"
            attention = "partner"
            energy_delta = 0.08
        if not event.duo and _contains_any(
            prompt,
            (
                "힘들",
                "힘내",
                "위로",
                "우울",
                "불안",
                "지침",
                "지쳤",
                "지친",
                "짜증",
                "슬퍼",
                "서운",
                "울적",
                "외롭",
                "무거워",
                "망했",
                "실수",
                "아무것도 못",
                "버틴",
                "울컥",
                "곁에",
                "안아",
                "괜찮아",
                "피곤",
                "무너지",
                "혼자 있는",
                "잠이 안 와",
                "보고 싶다",
                "해결책보다",
                "해결책 말고",
            ),
        ):
            mood = "soft"
            attention = "viewer"
            energy_delta = -0.08
        elif not event.duo and _contains_any(prompt, ("ㅋㅋ", "ㅎㅎ", "웃", "재밌", "웃기")):
            mood = "playful"
            attention = "chat"
            energy_delta = 0.06
        elif not event.duo and _contains_any(
            prompt,
            (
                "고마워",
                "반가",
                "안녕",
                "하이",
                "인사",
                "안부",
                "좋은 아침",
                "퇴근",
                "오랜만",
                "처음 보는",
                "하루 어땠",
                "가볍게 물어",
                "잘 자",
            ),
        ):
            mood = "warm"
            attention = "viewer"
            energy_delta = 0.03
        next_energy = (self.mood_state.energy * 0.68) + ((0.72 + energy_delta) * 0.32)
        return WhiteMoodState(
            mood=mood,
            energy=round(max(0.2, min(next_energy, 1.15)), 3),
            attention_target=attention,
            last_event_kind=event.kind,
            event_count=self.mood_state.event_count + 1,
        )


def _action_intent(*, event: WhiteRuntimeEvent, mood: WhiteMoodState) -> str:
    if event.duo:
        return "duo_reply"
    if event.search_used:
        return "grounded_search_reply"
    if event.has_images:
        return "visual_reply"
    if mood.mood == "soft":
        return "support_reply"
    return "chat_reply"


def _facial_expression(*, mood: WhiteMoodState, event: WhiteRuntimeEvent) -> str:
    if event.search_used:
        return "focused"
    if mood.mood in {"warm", "playful"}:
        return "smile"
    if mood.mood == "soft":
        return "soft"
    if mood.mood == "curious":
        return "attentive"
    return "neutral"


def _voice_style(mood: str) -> str:
    if mood == "focused":
        return "white_focused"
    if mood == "soft":
        return "white_soft"
    if mood == "playful":
        return "white_playful"
    if mood == "warm":
        return "white_warm"
    if mood == "curious":
        return "white_curious"
    return "white_neutral"


def _avatar_action(*, event: WhiteRuntimeEvent, mood: WhiteMoodState, expression: str) -> str:
    if event.duo:
        return "gaze_partner_bounce"
    if event.search_used:
        return "focused_read"
    if event.has_images:
        return "lean_in_look"
    if mood.mood == "soft":
        return "slow_nod"
    if mood.mood in {"warm", "playful"}:
        return "small_bounce"
    if expression == "attentive":
        return "attentive_gaze"
    return "viewer_gaze"


def _priority(event: WhiteRuntimeEvent) -> str:
    if event.duo:
        return "normal"
    if event.search_used:
        return "normal"
    return "normal"


def _action_cues(
    *,
    event: WhiteRuntimeEvent,
    mood: WhiteMoodState,
    expression: str,
) -> list[VTuberActionCue]:
    base_intensity = max(0.2, min(mood.energy, 1.2))
    cues = [
        VTuberActionCue(kind="speak", intensity=base_intensity),
        VTuberActionCue(kind=f"expression:{expression}", intensity=max(0.2, min(mood.energy, 1.0))),
    ]
    if event.duo:
        cues.extend(
            [
                VTuberActionCue(kind="gaze:partner", intensity=0.78),
                VTuberActionCue(kind="gesture:small_bounce", intensity=0.52),
            ]
        )
    elif event.search_used:
        cues.extend(
            [
                VTuberActionCue(kind="posture:focused", intensity=0.7),
                VTuberActionCue(kind="gaze:search", intensity=0.68),
            ]
        )
    elif event.has_images:
        cues.extend(
            [
                VTuberActionCue(kind="gesture:lean_in", intensity=0.5),
                VTuberActionCue(kind="gaze:viewer", intensity=0.75),
            ]
        )
    elif mood.mood == "soft":
        cues.extend(
            [
                VTuberActionCue(kind="posture:soft", intensity=0.58),
                VTuberActionCue(kind="gesture:slow_nod", intensity=0.36),
            ]
        )
    elif mood.mood in {"warm", "playful"}:
        cues.append(VTuberActionCue(kind="gesture:small_bounce", intensity=0.44))
    else:
        cues.append(VTuberActionCue(kind="gaze:viewer", intensity=0.58))
    return cues


def _confidence(*, reply: str, search_context: SearchContext | None) -> float:
    if search_context is not None and search_context.results:
        return 0.82
    compact = " ".join(reply.split())
    if len(compact) < 8:
        return 0.55
    return 0.72


def _evidence_used(search_context: SearchContext | None) -> list[str]:
    if search_context is None:
        return []
    evidence = [f"search_query:{search_context.query}"]
    for result in search_context.results[:3]:
        evidence.append(f"source:{result.title}")
    return evidence


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)
