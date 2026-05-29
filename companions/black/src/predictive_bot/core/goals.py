from __future__ import annotations

from predictive_bot.core.models import ConversationState, Goal, Intent, MessageFeatures


class GoalManager:
    def __init__(self, default_location: str | None = None) -> None:
        self.default_location = default_location

    def build_goals(self, features: MessageFeatures, state: ConversationState) -> list[Goal]:
        goals: list[Goal] = []

        if features.intent == Intent.HOSTILE:
            goals.append(Goal("deescalate", 1.00, "사용자 메시지가 공격적이므로 분위기를 낮추는 것이 우선입니다."))
            goals.append(Goal("stay_safe", 0.95, "갈등을 더 키우지 않는 쪽이 안전합니다."))
        elif "polite_boundary" in features.pragmatic_cues or "soft_refusal" in features.pragmatic_cues:
            goals.append(Goal("respect_boundary", 0.90, "완곡하게 선을 긋는 입력이라 경계를 부드럽게 존중하는 것이 우선입니다."))
            goals.append(Goal("keep_face_safe", 0.76, "상대가 직설을 피하고 있어 체면을 살리는 반응이 더 자연스럽습니다."))
        elif "tentative_request" in features.pragmatic_cues:
            goals.append(Goal("respond_gently", 0.82, "조심스럽게 부탁하거나 떠보는 톤이라 부드럽게 받는 편이 좋습니다."))
            goals.append(Goal("preserve_social_flow", 0.73, "부담을 줄이는 톤을 유지하는 것이 자연스럽습니다."))
        elif "empathy" in features.response_needs:
            goals.append(Goal("acknowledge_feeling", 0.93, "지금은 정보 응답보다 감정 반응이나 공감이 먼저입니다."))
            goals.append(Goal("keep_conversation_warm", 0.78, "공감 이후 부담 없이 이어지는 반응이 자연스럽습니다."))
        elif features.intent == Intent.WEATHER:
            goals.append(Goal("answer_weather", 1.00, "사용자가 날씨 정보를 요청했습니다."))
            goals.append(Goal("avoid_hallucination", 0.97, "사실을 추측하지 말고 도구로 조회해야 합니다."))
            if not (features.location or state.known_location or self.default_location):
                goals.append(Goal("collect_location", 0.98, "날씨 조회에는 지역 정보가 필요합니다."))
        elif features.intent == Intent.PROVIDE_LOCATION:
            goals.append(Goal("resolve_missing_slot", 0.96, "사용자가 이전에 요청한 지역 정보를 제공했습니다."))
        elif features.intent == Intent.HELP:
            goals.append(Goal("explain_capabilities", 0.92, "사용자가 봇이 할 수 있는 일을 묻고 있습니다."))
        elif features.intent == Intent.WHO_ARE_YOU:
            goals.append(Goal("state_identity", 0.92, "사용자가 봇의 정체와 역할을 묻고 있습니다."))
        elif features.intent == Intent.REPLY_REQUEST:
            goals.append(Goal("ack_request", 0.90, "사용자는 봇이 반응하고 있는지 확인하려고 합니다."))
            goals.append(Goal("collect_missing_context", 0.88, "주제가 부족하면 짧게 추가 설명을 요청해야 합니다."))
        elif features.intent == Intent.WHY:
            goals.append(Goal("explain_previous_behavior", 0.91, "바로 전 응답의 이유를 설명하는 것이 자연스럽습니다."))
        elif features.intent == Intent.CONFIRM:
            goals.append(Goal("acknowledge_confirmation", 0.80, "짧게 수긍하고 대화를 이어갑니다."))
        elif features.intent == Intent.DENY:
            goals.append(Goal("acknowledge_correction", 0.80, "부정을 반영하고 다시 방향을 확인합니다."))
        elif features.intent == Intent.SMALLTALK_FEELING:
            goals.append(Goal("respond_to_feeling", 0.86, "감정이나 불편함 표현에는 공감 반응이 우선입니다."))
        elif features.intent == Intent.SMALLTALK_OPINION:
            if features.is_question or features.speech_act == "ask":
                goals.append(Goal("share_opinion", 0.84, "의견을 묻는 쪽이라 짧게 관점을 주는 것이 자연스럽습니다."))
            else:
                goals.append(Goal("keep_conversation_warm", 0.76, "질문보다 여운을 남기는 코멘트라 바로 판단하기보다 가볍게 이어받는 편이 자연스럽습니다."))
        elif features.intent == Intent.ACTIVITY_INVITE:
            goals.append(Goal("respond_to_activity_invite", 0.86, "같이 하자는 활동 제안이라 제안 자체에 가볍게 호응하는 것이 자연스럽습니다."))
            goals.append(Goal("preserve_activity_context", 0.78, "장소와 활동 조건을 잃지 않는 것이 중요합니다."))
        elif features.intent == Intent.GREETING:
            goals.append(Goal("connect", 0.72, "짧게 인사하는 것이 자연스럽습니다."))
        elif features.intent == Intent.THANKS:
            goals.append(Goal("acknowledge", 0.74, "감사 표현에는 짧게 받아주는 것이 좋습니다."))
        elif features.intent == Intent.SMALLTALK_GENERIC:
            goals.append(Goal("keep_conversation_warm", 0.70, "부담 없이 이어지는 짧은 잡담이 적절합니다."))
        else:
            goals.append(Goal("clarify_intent", 0.70, "의도가 불분명하므로 짧게 되묻는 편이 안전합니다."))

        if state.boundary_pressure >= 0.45:
            goals.append(Goal("respect_distance", 0.84, "이전 몇 턴에서 선을 긋는 흐름이 보여 과하게 밀어붙이지 않는 것이 좋습니다."))

        if state.rapport <= 0.35 and features.intent in {Intent.TEASE, Intent.SMALLTALK_GENERIC, Intent.GAME_INVITE}:
            goals.append(Goal("avoid_overfamiliarity", 0.80, "아직 친밀도가 높지 않아 너무 가까운 톤은 피하는 편이 자연스럽습니다."))

        if state.rapport >= 0.72 and features.intent in {Intent.GREETING, Intent.THANKS, Intent.SMALLTALK_GENERIC}:
            goals.append(Goal("maintain_rapport", 0.68, "이미 쌓인 친밀감을 무리 없이 유지하는 응답이 좋습니다."))

        return sorted(goals, key=lambda goal: goal.priority, reverse=True)
