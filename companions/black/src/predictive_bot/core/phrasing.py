from __future__ import annotations

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    MessageFeatures,
    PhrasingCloser,
    PhrasingDistance,
    PhrasingOpener,
    PhrasingPlan,
    PhrasingQuestionMode,
    WorldState,
)


_BOUNDARY_HISTORIES = {"active_boundary", "firm_boundary"}


def build_phrasing_plan(
    *,
    features: MessageFeatures,
    decision: ActionDecision,
    state: ConversationState,
    world_state: WorldState | None,
) -> PhrasingPlan:
    boundary_active = bool(world_state is not None and world_state.boundary_history in _BOUNDARY_HISTORIES)
    indirect_user = bool(world_state is not None and world_state.user_directness_style == "indirect")
    notes: list[str] = []

    if boundary_active:
        notes.append("boundary_active")
    if indirect_user:
        notes.append("indirect_user")
    if state.turn_count >= 4:
        notes.append("mid_conversation")

    if decision.action == ActionType.SHARE_FEELING:
        asks_followup = not boundary_active
        if asks_followup:
            notes.append("emotion_followup")
        return PhrasingPlan(
            opener=PhrasingOpener.WARM,
            question_mode=PhrasingQuestionMode.SOFT if asks_followup else PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.KEEP_OPEN if asks_followup else PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.SOFT,
            asks_followup=asks_followup,
            notes=notes,
        )

    if decision.action == ActionType.CONTINUE_CONVERSATION:
        asks_followup = not boundary_active
        return PhrasingPlan(
            opener=PhrasingOpener.BRIDGING,
            question_mode=PhrasingQuestionMode.SOFT if indirect_user and asks_followup else (
                PhrasingQuestionMode.DIRECT if asks_followup else PhrasingQuestionMode.NONE
            ),
            closer=PhrasingCloser.KEEP_OPEN if asks_followup else PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.SOFT if boundary_active else PhrasingDistance.NEUTRAL,
            asks_followup=asks_followup,
            notes=notes,
        )

    if decision.action == ActionType.ASK_CLARIFICATION:
        return PhrasingPlan(
            opener=PhrasingOpener.CLARIFYING,
            question_mode=PhrasingQuestionMode.SOFT if (boundary_active or indirect_user) else PhrasingQuestionMode.DIRECT,
            closer=PhrasingCloser.KEEP_OPEN,
            distance=PhrasingDistance.SOFT if boundary_active else PhrasingDistance.STEADY,
            asks_followup=True,
            notes=notes,
        )

    if decision.action == ActionType.ASK_LOCATION:
        return PhrasingPlan(
            opener=PhrasingOpener.CLARIFYING,
            question_mode=PhrasingQuestionMode.DIRECT,
            closer=PhrasingCloser.KEEP_OPEN,
            distance=PhrasingDistance.STEADY,
            asks_followup=True,
            notes=notes,
        )

    if decision.action == ActionType.WEATHER_LOOKUP:
        return PhrasingPlan(
            opener=PhrasingOpener.INFORMATIVE,
            question_mode=PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.STEADY,
            asks_followup=False,
            notes=notes,
        )

    if decision.action == ActionType.WEATHER_UNAVAILABLE:
        return PhrasingPlan(
            opener=PhrasingOpener.INFORMATIVE,
            question_mode=PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.STEADY,
            asks_followup=False,
            notes=notes,
        )

    if decision.action == ActionType.SHARE_OPINION:
        return PhrasingPlan(
            opener=PhrasingOpener.GROUNDED,
            question_mode=PhrasingQuestionMode.SOFT if indirect_user else PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.KEEP_OPEN if indirect_user else PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.SOFT if boundary_active else PhrasingDistance.STEADY,
            asks_followup=indirect_user,
            notes=notes,
        )

    if decision.action == ActionType.ACCEPT_ACTIVITY_INVITE:
        return PhrasingPlan(
            opener=PhrasingOpener.REACTIVE,
            question_mode=PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.LIGHT,
            asks_followup=False,
            notes=notes,
        )

    if decision.action in {ActionType.REACT_LAUGH, ActionType.REACT_SURPRISE}:
        return PhrasingPlan(
            opener=PhrasingOpener.REACTIVE,
            question_mode=PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.NONE,
            distance=PhrasingDistance.PLAYFUL,
            asks_followup=False,
            notes=notes,
        )

    if decision.action in {ActionType.EXPLAIN_CAPABILITIES, ActionType.ANSWER_IDENTITY}:
        return PhrasingPlan(
            opener=PhrasingOpener.INFORMATIVE,
            question_mode=PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.STEADY,
            asks_followup=False,
            notes=notes,
        )

    if decision.action == ActionType.ACKNOWLEDGE:
        return PhrasingPlan(
            opener=PhrasingOpener.BRIEF,
            question_mode=PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.SOFT if boundary_active else PhrasingDistance.LIGHT,
            asks_followup=False,
            notes=notes,
        )

    if decision.action == ActionType.SMALL_TALK:
        asks_followup = not boundary_active
        return PhrasingPlan(
            opener=PhrasingOpener.LIGHT,
            question_mode=PhrasingQuestionMode.SOFT if asks_followup else PhrasingQuestionMode.NONE,
            closer=PhrasingCloser.KEEP_OPEN if asks_followup else PhrasingCloser.SOFT_CLOSE,
            distance=PhrasingDistance.SOFT if boundary_active else PhrasingDistance.LIGHT,
            asks_followup=asks_followup,
            notes=notes,
        )

    return PhrasingPlan(notes=notes)
