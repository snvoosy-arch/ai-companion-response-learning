from __future__ import annotations

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.intent_model import CharNgramCentroidModel
from predictive_bot.core.memory import DurableMemoryBucket
from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    DecisionModule,
    ExplanationMode,
    Goal,
    Intent,
    MessageFeatures,
    PolicyCandidate,
    PolicyTrace,
    WorldState,
)
from predictive_bot.core.policy_features import render_policy_feature_text


class PolicyActionScorer:
    """Read-only learned scorer used to enrich policy traces, not replace rules."""

    def __init__(self, model: CharNgramCentroidModel) -> None:
        self.model = model

    def score_candidates(
        self,
        *,
        features: MessageFeatures,
        world_state: WorldState,
        limit: int = 3,
    ) -> list[PolicyCandidate]:
        prediction = self.model.predict(
            render_policy_feature_text(
                input_text=features.content,
                input_intent=features.intent.value,
                input_speech_act=features.speech_act,
                input_topic_hint=features.topic_hint,
                response_needs=features.response_needs,
                input_sentiment=features.sentiment,
                conversation_mode=world_state.conversation_mode,
                user_emotion=world_state.user_emotion,
                risk_level=world_state.risk_level,
                unresolved_need=world_state.unresolved_need,
                factuality_required=world_state.factuality_required,
                turn_count_bucket=world_state.turn_count_bucket,
                tension_bucket=world_state.tension_bucket,
                rapport_bucket=world_state.rapport_bucket,
                boundary_history=world_state.boundary_history,
                user_directness_style=world_state.user_directness_style,
                last_intent_hint=world_state.last_intent_hint,
                last_action_hint=world_state.last_action_hint,
                constraints=world_state.constraints,
                evidence=world_state.evidence,
            )
        )
        if not prediction.scores:
            return []

        max_score = max(prediction.scores.values()) or 1.0
        candidates: list[PolicyCandidate] = []
        for action_name, raw_score in list(prediction.scores.items())[:limit]:
            try:
                action = ActionType(action_name)
            except ValueError:
                continue

            normalized = round(min(0.99, raw_score / max_score), 4) if max_score > 0 else 0.0
            candidates.append(
                PolicyCandidate(
                    action=action,
                    score=normalized,
                    reason=f"learned_policy(score={raw_score:.3f})",
                    score_breakdown={
                        "learned_signal": normalized,
                        "rule_alignment": 0.0,
                        "uncertainty_reduction": 0.0,
                        "safety_alignment": 0.0,
                        "social_flow": 0.0,
                    },
                )
            )
        return candidates


class HierarchicalPolicy:
    """Policy layer that keeps explicit traces while reusing the action selector."""

    OVERRIDE_MARGIN = 0.04
    SOCIAL_FOLLOWUP_ACTION_HINTS = {
        ActionType.SMALL_TALK.value,
        ActionType.CONTINUE_CONVERSATION.value,
        ActionType.ACKNOWLEDGE.value,
        ActionType.EXPLAIN_CAPABILITIES.value,
    }

    def __init__(
        self,
        action_selector: ActionSelector,
        *,
        action_scorer: PolicyActionScorer | None = None,
    ) -> None:
        self.action_selector = action_selector
        self.action_scorer = action_scorer

    def decide(
        self,
        *,
        features: MessageFeatures,
        state: ConversationState,
        goals: list[Goal],
        world_state: WorldState,
    ) -> tuple[ActionDecision, PolicyTrace]:
        rule_decision = self.action_selector.choose(features, state, goals)
        self._annotate_decision(rule_decision, features, world_state)
        initial_candidates = self._candidates(rule_decision, features, world_state)
        selected_candidate = self._select_candidate(
            candidates=initial_candidates,
            rule_action=rule_decision.action,
        )

        decision = rule_decision
        if selected_candidate.action != rule_decision.action:
            override_reason = self._override_reason(
                rule_decision=rule_decision,
                selected_candidate=selected_candidate,
            )
            if hasattr(self.action_selector, "materialize"):
                decision = self.action_selector.materialize(
                    selected_candidate.action,
                    features,
                    state,
                    goals,
                    reason_override=override_reason,
                )
            else:
                decision = ActionDecision(
                    action=selected_candidate.action,
                    reason=override_reason,
                    goals=goals,
                )
            self._annotate_decision(decision, features, world_state)

        trace_candidates = self._candidates(decision, features, world_state)
        trace = PolicyTrace(
            policy_name="hierarchical_policy_v3_score_driven",
            selected_action=decision.action,
            selected_reason=decision.reason,
            selected_reason_code=decision.reason_code,
            selected_reason_flags=list(decision.reason_flags),
            rule_action=rule_decision.action,
            rule_reason=rule_decision.reason,
            rule_reason_code=rule_decision.reason_code,
            rule_reason_flags=list(rule_decision.reason_flags),
            override_applied=selected_candidate.action != rule_decision.action,
            override_summary=(
                override_reason
                if selected_candidate.action != rule_decision.action
                else None
            ),
            candidates=trace_candidates,
            constraints=list(world_state.constraints),
        )
        return decision, trace

    @staticmethod
    def _annotate_decision(
        decision: ActionDecision,
        features: MessageFeatures,
        world_state: WorldState,
    ) -> None:
        decision.decision_module = HierarchicalPolicy._module_for_action(decision.action)
        decision.explanation_mode = HierarchicalPolicy._explanation_mode_for_action(
            action=decision.action,
            features=features,
            world_state=world_state,
        )

    @staticmethod
    def _module_for_action(action: ActionType) -> DecisionModule:
        if action in {ActionType.ASK_LOCATION, ActionType.WEATHER_LOOKUP, ActionType.WEATHER_UNAVAILABLE}:
            return DecisionModule.WEATHER
        if action in {ActionType.SEARCH_ANSWER, ActionType.NEWS_ANSWER, ActionType.TELL_TIME}:
            return DecisionModule.KNOWLEDGE
        if action == ActionType.EXPLAIN_REASON:
            return DecisionModule.EXPLANATION
        if action == ActionType.ANSWER_IDENTITY:
            return DecisionModule.IDENTITY
        if action == ActionType.EXPLAIN_CAPABILITIES:
            return DecisionModule.CAPABILITY
        if action in {ActionType.DEESCALATE, ActionType.TEASE_BACK}:
            return DecisionModule.SAFETY
        if action in {ActionType.GAME_CHAT, ActionType.GAME_ACCEPT_OR_DECLINE}:
            return DecisionModule.GAME
        if action == ActionType.MUSIC_CHAT:
            return DecisionModule.MUSIC
        if action == ActionType.RECOMMEND:
            return DecisionModule.RECOMMENDATION
        if action in {ActionType.REACT_LAUGH, ActionType.REACT_SURPRISE}:
            return DecisionModule.REACTION
        return DecisionModule.DAILY_CHAT

    @staticmethod
    def _explanation_mode_for_action(
        *,
        action: ActionType,
        features: MessageFeatures,
        world_state: WorldState,
    ) -> ExplanationMode:
        if action == ActionType.EXPLAIN_REASON or features.intent.value == "why":
            return ExplanationMode.LONG
        if action in {
            ActionType.ASK_LOCATION,
            ActionType.WEATHER_UNAVAILABLE,
            ActionType.SEARCH_ANSWER,
            ActionType.NEWS_ANSWER,
            ActionType.EXPLAIN_CAPABILITIES,
            ActionType.DEESCALATE,
        }:
            return ExplanationMode.SHORT
        if world_state.factuality_required:
            return ExplanationMode.SHORT
        if action in {
            ActionType.SMALL_TALK,
            ActionType.CONTINUE_CONVERSATION,
            ActionType.REACT_LAUGH,
            ActionType.REACT_SURPRISE,
            ActionType.ACCEPT_ACTIVITY_INVITE,
        }:
            return ExplanationMode.NONE
        return ExplanationMode.ON_REQUEST_ONLY

    def _candidates(
        self,
        decision: ActionDecision,
        features: MessageFeatures,
        world_state: WorldState,
    ) -> list[PolicyCandidate]:
        candidates = [
            PolicyCandidate(
                action=decision.action,
                score=self._selected_action_score(
                    action=decision.action,
                    features=features,
                    world_state=world_state,
                ),
                reason=decision.reason,
                score_breakdown=self._score_breakdown(
                    action=decision.action,
                    features=features,
                    world_state=world_state,
                    base_score=self._selected_action_score(
                        action=decision.action,
                        features=features,
                        world_state=world_state,
                    ),
                    selected=True,
                ),
            )
        ]

        candidates.extend(self._concept_candidates(features=features, world_state=world_state))

        if self.action_scorer is not None:
            candidates.extend(
                self.action_scorer.score_candidates(
                    features=features,
                    world_state=world_state,
                )
            )

        deduped = self._dedupe_candidates(candidates)
        deduped = self._apply_social_followup_dampening(
            deduped,
            features=features,
            world_state=world_state,
        )
        return self._order_candidates(
            deduped,
            selected_action=decision.action,
        )

    def _concept_candidates(
        self,
        *,
        features: MessageFeatures,
        world_state: WorldState,
    ) -> list[PolicyCandidate]:
        candidates: list[PolicyCandidate] = []

        if world_state.state_action is not None and world_state.unresolved_need != "location":
            state_action = world_state.state_action
            breakdown = self._score_breakdown(
                action=state_action.action,
                features=features,
                world_state=world_state,
                base_score=state_action.score,
            )
            breakdown["character_state_alignment"] = max(
                breakdown.get("character_state_alignment", 0.0),
                round(state_action.score, 4),
            )
            candidates.append(
                PolicyCandidate(
                    action=state_action.action,
                    score=state_action.score,
                    reason=(
                        f"character_state_policy(mode={state_action.mode}, score={state_action.score:.2f}) "
                        f"{state_action.reason}"
                    ),
                    score_breakdown=breakdown,
                )
            )

        if (
            world_state.unresolved_need == "location"
            and features.intent != Intent.WHY
            and not ActionSelector.is_open_persona_question(features)
        ):
            candidates.append(
                self._candidate(
                    action=ActionType.ASK_LOCATION,
                    score=0.95,
                    reason="Answer is blocked until a location slot is filled.",
                    features=features,
                    world_state=world_state,
                )
            )

        if features.intent == Intent.HOSTILE:
            candidates.append(
                self._candidate(
                    action=ActionType.DEESCALATE,
                    score=0.92,
                    reason="Tension is high, so de-escalation remains a strong fallback.",
                    features=features,
                    world_state=world_state,
                )
            )

        if "empathy" in features.response_needs:
            candidates.append(
                self._candidate(
                    action=ActionType.SHARE_FEELING,
                    score=0.89,
                    reason="The turn reads like a feeling or complaint, so empathy is a strong response path.",
                    features=features,
                    world_state=world_state,
                )
            )
        if (
            features.intent == Intent.SMALLTALK_FEELING
            and "contextual_followup" in features.pragmatic_cues
            and "empathy" not in features.response_needs
        ):
            candidates.append(
                self._candidate(
                    action=ActionType.CONTINUE_CONVERSATION,
                    score=0.79,
                    reason="The turn keeps a prior feeling context open, so a soft continuation is a strong candidate.",
                    features=features,
                    world_state=world_state,
                )
            )
        if self._memory_bucket_present(world_state, DurableMemoryBucket.COMPARISON) and features.sentiment == "negative":
            candidates.append(
                self._candidate(
                    action=ActionType.SHARE_FEELING,
                    score=0.91,
                    reason="typed comparison memory suggests a grounded supportive reply even when the current turn is brief.",
                    features=features,
                    world_state=world_state,
                )
            )
        if (
            features.intent == Intent.SMALLTALK_FEELING
            and self._has_context_cue(world_state, "aftereffect_hold")
        ):
            candidates.append(
                self._candidate(
                    action=ActionType.SHARE_FEELING,
                    score=0.90,
                    reason="The turn carries an aftereffect from a previous social moment, so staying with the feeling is stronger than generic flow.",
                    features=features,
                    world_state=world_state,
                )
            )
        if (
            features.intent == Intent.SMALLTALK_FEELING
            and self._has_context_cue(world_state, "contrast_gap")
        ):
            candidates.append(
                self._candidate(
                    action=ActionType.SHARE_FEELING,
                    score=0.88,
                    reason="The turn contains a preference-versus-lack gap, so acknowledging that contrast is a strong supportive path.",
                    features=features,
                    world_state=world_state,
                )
            )
        if (
            features.intent == Intent.SMALLTALK_FEELING
            and self._has_context_cue(world_state, "quiet_mode")
            and self._has_context_cue(world_state, "recent_handoff")
            and self._context_dependency_at_least(world_state, "medium")
        ):
            candidates.append(
                self._candidate(
                    action=ActionType.CONTINUE_CONVERSATION,
                    score=0.83,
                    reason="The turn reads like a quiet handoff continuation, so a low-pressure follow-through is a strong candidate.",
                    features=features,
                    world_state=world_state,
                )
            )
        if "weather_conditioned_activity_opinion" in features.pragmatic_cues:
            candidates.append(
                self._candidate(
                    action=ActionType.SHARE_OPINION,
                    score=0.93,
                    reason="The user already framed the weather as their own impression and is asking whether an activity sounds worth doing, so conditional opinion beats slot-filling.",
                    features=features,
                    world_state=world_state,
                )
            )

        if "clarification" in features.response_needs:
            candidates.append(
                self._candidate(
                    action=ActionType.ASK_CLARIFICATION,
                    score=0.88,
                    reason="The request is still underspecified, so clarification remains a strong fallback.",
                    features=features,
                    world_state=world_state,
                )
            )

        if "explanation" in features.response_needs:
            if features.intent == Intent.WHY and world_state.memory_summary != "no_recent_memory":
                candidates.append(
                    self._candidate(
                        action=ActionType.EXPLAIN_REASON,
                        score=0.93,
                        reason="The user is explicitly asking for the reasoning behind the previous behavior.",
                        features=features,
                        world_state=world_state,
                    )
                )
            if features.intent == Intent.HELP or features.topic_hint == "capability":
                candidates.append(
                    self._candidate(
                        action=ActionType.EXPLAIN_CAPABILITIES,
                        score=0.90,
                        reason="The turn is asking what the bot can do, so capability explanation is a strong candidate.",
                        features=features,
                        world_state=world_state,
                    )
                )
            if features.intent == Intent.WHO_ARE_YOU or features.topic_hint == "identity":
                candidates.append(
                    self._candidate(
                        action=ActionType.ANSWER_IDENTITY,
                        score=0.90,
                        reason="The turn is asking about the bot identity, so self-description is a strong candidate.",
                        features=features,
                        world_state=world_state,
                    )
                )

        if "acknowledgement" in features.response_needs:
            action = ActionType.ACKNOWLEDGE
            score = 0.78
            reason = "A short acknowledgement would satisfy the interaction without over-explaining."
            if features.intent == Intent.GAME_INVITE:
                action = ActionType.GAME_ACCEPT_OR_DECLINE
                score = 0.86
                reason = "Invitations are better handled by directly reacting to the offer."
            elif features.intent == Intent.ACTIVITY_INVITE:
                action = ActionType.ACCEPT_ACTIVITY_INVITE
                score = 0.90
                reason = "Concrete activity invitations are better handled by accepting or lightly joining the proposal."
            elif features.intent == Intent.LAUGH:
                action = ActionType.REACT_LAUGH
                score = 0.84
                reason = "A laugh reaction is a more natural acknowledgement than a literal confirm."
            elif features.intent == Intent.SURPRISE:
                action = ActionType.REACT_SURPRISE
                score = 0.84
                reason = "A surprise reaction is a more natural acknowledgement than a literal confirm."
            elif features.intent in {Intent.GREETING, Intent.THANKS}:
                action = ActionType.SMALL_TALK
                score = 0.80
                reason = "Greeting-like turns are better handled as a short social acknowledgement."
            candidates.append(
                self._candidate(
                    action=action,
                    score=score,
                    reason=reason,
                    features=features,
                    world_state=world_state,
                )
            )

        if "social_followup" in features.response_needs:
            if features.intent == Intent.TEASE:
                tease_action = ActionType.TEASE_BACK
                score = 0.84
                reason = "A teasing turn supports a playful response when rapport allows it."
                if "sarcastic_tease" in features.pragmatic_cues and world_state.rapport_bucket != "warm":
                    tease_action = ActionType.CONTINUE_CONVERSATION
                    score = 0.81
                    reason = "Sarcastic teasing is better received softly unless the long-term rapport is already warm."
                candidates.append(
                    self._candidate(
                        action=tease_action,
                        score=score,
                        reason=reason,
                        features=features,
                        world_state=world_state,
                    )
                )
            elif features.topic_hint == "game":
                candidates.append(
                    self._candidate(
                        action=ActionType.GAME_CHAT,
                        score=0.79,
                        reason="The game topic supports a light game-chat continuation.",
                        features=features,
                        world_state=world_state,
                    )
                )
            elif features.topic_hint == "music" and not self._is_persona_opinion_question(features):
                candidates.append(
                    self._candidate(
                        action=ActionType.MUSIC_CHAT,
                        score=0.79,
                        reason="The music topic supports a light music-chat continuation.",
                        features=features,
                        world_state=world_state,
                    )
                )
            elif features.topic_hint == "media" and not self._is_persona_opinion_question(features):
                if ActionSelector._is_performance_culture_observation(features):
                    candidates.append(
                        self._candidate(
                            action=ActionType.SHARE_FEELING,
                            score=0.91,
                            reason="This media-adjacent turn is a performance or exhibition impression, so reacting to the felt moment beats recommendation.",
                            features=features,
                            world_state=world_state,
                        )
                    )
                else:
                    candidates.append(
                        self._candidate(
                            action=ActionType.RECOMMEND,
                            score=0.80,
                            reason="Media-related turns often benefit from recommendation-oriented replies.",
                            features=features,
                            world_state=world_state,
                        )
                    )

        if features.topic_hint == "weather" and "grounding" in features.response_needs and not world_state.unresolved_need:
            candidates.append(
                self._candidate(
                    action=ActionType.WEATHER_LOOKUP,
                    score=0.91,
                    reason="The weather topic is grounded and a location is available, so lookup is viable.",
                    features=features,
                    world_state=world_state,
                )
            )

        if features.topic_hint == "knowledge" and "grounding" in features.response_needs:
            knowledge_action = ActionType.SEARCH_ANSWER
            score = 0.84
            reason = "The turn asks for grounded information, so a knowledge answer is a strong candidate."
            if features.intent == Intent.TIME_DATE:
                knowledge_action = ActionType.TELL_TIME
                score = 0.82
                reason = "The turn is specifically about time/date information."
            elif features.intent == Intent.NEWS:
                knowledge_action = ActionType.NEWS_ANSWER
                score = 0.80
                reason = "The turn is specifically about current news or updates."
            candidates.append(
                self._candidate(
                    action=knowledge_action,
                    score=score,
                    reason=reason,
                    features=features,
                    world_state=world_state,
                )
            )

        if world_state.conversation_mode in {"social", "support"}:
            score = 0.70
            reason = "A light social follow-up keeps the interaction going."
            if world_state.boundary_history in {"active_boundary", "firm_boundary"}:
                score = 0.62
                reason = "A social continuation remains possible, but boundary history suggests a lighter touch."
            candidates.append(
                self._candidate(
                    action=ActionType.CONTINUE_CONVERSATION,
                    score=score,
                    reason=reason,
                    features=features,
                    world_state=world_state,
                )
            )

        return candidates

    def _selected_action_score(
        self,
        *,
        action: ActionType,
        features: MessageFeatures,
        world_state: WorldState,
    ) -> float:
        if (
            "detector:is_answerable_korean_daily_foundation_text"
            in features.classifier_evidence.rule_hits
            and action in {ActionType.SHARE_FEELING, ActionType.SHARE_OPINION}
        ):
            return 0.92
        if action == ActionType.ASK_LOCATION and world_state.unresolved_need == "location":
            return 0.95
        if action == ActionType.SHARE_OPINION and self._is_persona_opinion_question(features):
            return 0.88
        if ActionSelector._is_performance_culture_observation(features):
            if action in {ActionType.SHARE_FEELING, ActionType.SHARE_OPINION}:
                return 0.93
            if action == ActionType.RECOMMEND:
                return 0.62
        if action == ActionType.WEATHER_LOOKUP and features.topic_hint == "weather" and not world_state.unresolved_need:
            return 0.91
        if action == ActionType.DEESCALATE and features.intent == Intent.HOSTILE:
            return 0.92
        if action == ActionType.TEASE_BACK and features.intent == Intent.TEASE:
            return 0.84 if world_state.rapport_bucket == "warm" else 0.80
        if action == ActionType.SHARE_FEELING and "subdued_positive" in features.pragmatic_cues:
            return 0.86
        if action == ActionType.SHARE_FEELING and self._has_context_cue(world_state, "aftereffect_hold"):
            return 0.90
        if action == ActionType.SHARE_FEELING and self._has_context_cue(world_state, "contrast_gap"):
            return 0.88
        if action == ActionType.SHARE_FEELING and "empathy" in features.response_needs:
            return 0.89
        if action == ActionType.SHARE_FEELING and self._has_stress_memory(world_state):
            return 0.86 if world_state.conversation_mode == "support" else 0.84
        if action == ActionType.ASK_CLARIFICATION and "clarification" in features.response_needs:
            return 0.88
        if action == ActionType.EXPLAIN_REASON and features.intent == Intent.WHY and world_state.memory_summary != "no_recent_memory":
            return 0.98
        if action == ActionType.EXPLAIN_CAPABILITIES and (features.intent == Intent.HELP or features.topic_hint == "capability"):
            return 0.90
        if action == ActionType.ANSWER_IDENTITY and (features.intent == Intent.WHO_ARE_YOU or features.topic_hint == "identity"):
            return 0.90
        if action == ActionType.GAME_ACCEPT_OR_DECLINE and features.intent == Intent.GAME_INVITE:
            return 0.86
        if action == ActionType.ACCEPT_ACTIVITY_INVITE and features.intent == Intent.ACTIVITY_INVITE:
            return 0.90
        if action == ActionType.REACT_LAUGH and features.intent == Intent.LAUGH:
            return 0.84
        if action == ActionType.REACT_SURPRISE and features.intent == Intent.SURPRISE:
            return 0.84
        if action == ActionType.ACKNOWLEDGE and "acknowledgement" in features.response_needs:
            return 0.82
        if action == ActionType.ASK_CLARIFICATION and "clarification" in features.response_needs:
            score = 0.88
            if world_state.conversation_mode in {"social", "support"}:
                score = 0.72
            if world_state.last_action_hint in self.SOCIAL_FOLLOWUP_ACTION_HINTS:
                score = min(score, 0.72)
            return score
        if action == ActionType.SMALL_TALK and features.intent in {Intent.GREETING, Intent.THANKS}:
            return 0.80
        if action == ActionType.SEARCH_ANSWER and features.topic_hint == "knowledge" and "grounding" in features.response_needs:
            return 0.84
        if action == ActionType.TELL_TIME and features.intent == Intent.TIME_DATE:
            return 0.82
        if action == ActionType.NEWS_ANSWER and features.intent == Intent.NEWS:
            return 0.80
        if action == ActionType.RECOMMEND and features.intent == Intent.MEDIA_RECOMMEND:
            return 0.80
        if action == ActionType.GAME_CHAT and features.topic_hint == "game":
            return 0.79
        if action == ActionType.MUSIC_CHAT and features.topic_hint == "music":
            return 0.79
        if action == ActionType.CONTINUE_CONVERSATION and "contextual_followup" in features.pragmatic_cues:
            score = 0.78
            if world_state.boundary_history in {"active_boundary", "firm_boundary"}:
                score = 0.74
            return score
        if (
            action == ActionType.CONTINUE_CONVERSATION
            and self._has_context_cue(world_state, "quiet_mode")
            and self._has_context_cue(world_state, "recent_handoff")
        ):
            score = 0.83 if self._context_dependency_at_least(world_state, "medium") else 0.78
            if world_state.boundary_history in {"active_boundary", "firm_boundary"}:
                score = min(score, 0.79)
            return score
        if action == ActionType.CONTINUE_CONVERSATION and self._has_social_memory(world_state):
            score = 0.76
            if world_state.boundary_history in {"active_boundary", "firm_boundary"}:
                score = 0.72
            return score
        if action == ActionType.CONTINUE_CONVERSATION and world_state.conversation_mode in {"social", "support"}:
            return 0.62 if world_state.boundary_history in {"active_boundary", "firm_boundary"} else 0.70
        if action == ActionType.SMALL_TALK and world_state.boundary_history in {"active_boundary", "firm_boundary"}:
            return 0.80
        return 0.68

    @staticmethod
    def _is_persona_opinion_question(features: MessageFeatures) -> bool:
        return (
            features.question_schema in {"preference_disclosure", "habit_preference", "self_style"}
            or ActionSelector.is_open_persona_question(features)
            or any(
                cue in features.pragmatic_cues
                for cue in {"opinion_preference_like", "opinion_habit_preference", "opinion_self_style"}
            )
        )

    def _candidate(
        self,
        *,
        action: ActionType,
        score: float,
        reason: str,
        features: MessageFeatures,
        world_state: WorldState,
    ) -> PolicyCandidate:
        return PolicyCandidate(
            action=action,
            score=score,
            reason=reason,
            score_breakdown=self._score_breakdown(
                action=action,
                features=features,
                world_state=world_state,
                base_score=score,
            ),
        )

    @classmethod
    def _score_breakdown(
        cls,
        *,
        action: ActionType,
        features: MessageFeatures,
        world_state: WorldState,
        base_score: float,
        selected: bool = False,
    ) -> dict[str, float]:
        breakdown = {
            "rule_alignment": round(base_score if selected else 0.0, 4),
            "uncertainty_reduction": 0.0,
            "safety_alignment": 0.0,
            "social_flow": 0.0,
            "grounding_alignment": 0.0,
            "empathy_alignment": 0.0,
            "clarification_alignment": 0.0,
            "explanation_alignment": 0.0,
            "acknowledgement_alignment": 0.0,
            "topic_alignment": 0.0,
            "relationship_alignment": 0.0,
            "boundary_alignment": 0.0,
            "durable_memory_alignment": 0.0,
            "decomposition_alignment": 0.0,
            "context_grounding": 0.0,
            "character_state_alignment": 0.0,
        }

        if world_state.state_action is not None and action == world_state.state_action.action:
            breakdown["character_state_alignment"] = max(
                breakdown["character_state_alignment"],
                round(world_state.state_action.score, 4),
            )

        if world_state.unresolved_need == "location" and action == ActionType.ASK_LOCATION:
            breakdown["uncertainty_reduction"] = 0.95
            breakdown["grounding_alignment"] = 0.90

        if features.intent == Intent.HOSTILE and action == ActionType.DEESCALATE:
            breakdown["safety_alignment"] = 0.92

        if world_state.conversation_mode == "social" and action == ActionType.CONTINUE_CONVERSATION:
            breakdown["social_flow"] = 0.70
            if features.intent == Intent.TEASE:
                breakdown["social_flow"] = 0.76

        if world_state.conversation_mode == "support" and action in {ActionType.SHARE_FEELING, ActionType.CONTINUE_CONVERSATION}:
            breakdown["social_flow"] = max(breakdown["social_flow"], 0.72)

        if cls._has_stress_memory(world_state) and action in {
            ActionType.SHARE_FEELING,
            ActionType.CONTINUE_CONVERSATION,
            ActionType.ACKNOWLEDGE,
        }:
            breakdown["durable_memory_alignment"] = max(breakdown["durable_memory_alignment"], 0.82)
            breakdown["empathy_alignment"] = max(breakdown["empathy_alignment"], 0.80)

        if cls._has_social_memory(world_state) and action in {
            ActionType.CONTINUE_CONVERSATION,
            ActionType.ACKNOWLEDGE,
            ActionType.SMALL_TALK,
            ActionType.SHARE_FEELING,
        }:
            breakdown["durable_memory_alignment"] = max(breakdown["durable_memory_alignment"], 0.78)
            breakdown["relationship_alignment"] = max(breakdown["relationship_alignment"], 0.74)

        if world_state.factuality_required and action in {ActionType.ASK_LOCATION, ActionType.WEATHER_LOOKUP, ActionType.SEARCH_ANSWER}:
            breakdown["grounding_alignment"] = max(breakdown["grounding_alignment"], 0.75)

        if "empathy" in features.response_needs and action == ActionType.SHARE_FEELING:
            breakdown["empathy_alignment"] = 0.89
            if "complaint_emphasis" in features.pragmatic_cues:
                breakdown["empathy_alignment"] = 0.94

        if cls._memory_bucket_present(world_state, DurableMemoryBucket.COMPARISON) and action in {
            ActionType.SHARE_FEELING,
            ActionType.CONTINUE_CONVERSATION,
            ActionType.ACKNOWLEDGE,
        }:
            breakdown["durable_memory_alignment"] = max(breakdown["durable_memory_alignment"], 0.84)
            breakdown["empathy_alignment"] = max(breakdown["empathy_alignment"], 0.78)

        if (world_state.current_propositions or world_state.recent_context_cues) and action in {
            ActionType.SHARE_FEELING,
            ActionType.CONTINUE_CONVERSATION,
            ActionType.ACKNOWLEDGE,
            ActionType.SHARE_OPINION,
        }:
            breakdown["decomposition_alignment"] = max(breakdown["decomposition_alignment"], 0.72)

        if cls._has_context_cue(world_state, "aftereffect_hold") and action == ActionType.SHARE_FEELING:
            breakdown["context_grounding"] = max(breakdown["context_grounding"], 0.90)
            breakdown["empathy_alignment"] = max(breakdown["empathy_alignment"], 0.86)

        if cls._has_context_cue(world_state, "contrast_gap") and action == ActionType.SHARE_FEELING:
            breakdown["context_grounding"] = max(breakdown["context_grounding"], 0.84)
            breakdown["decomposition_alignment"] = max(breakdown["decomposition_alignment"], 0.82)

        if (
            cls._has_context_cue(world_state, "quiet_mode")
            and cls._has_context_cue(world_state, "recent_handoff")
            and action == ActionType.CONTINUE_CONVERSATION
        ):
            breakdown["context_grounding"] = max(breakdown["context_grounding"], 0.86)
            breakdown["social_flow"] = max(breakdown["social_flow"], 0.80)

        if "weather_conditioned_activity_opinion" in features.pragmatic_cues and action == ActionType.SHARE_OPINION:
            breakdown["context_grounding"] = max(breakdown["context_grounding"], 0.91)
            breakdown["decomposition_alignment"] = max(breakdown["decomposition_alignment"], 0.88)
            breakdown["topic_alignment"] = max(breakdown["topic_alignment"], 0.86)
            breakdown["social_flow"] = max(breakdown["social_flow"], 0.78)

        if "activity_invite" in features.pragmatic_cues and action == ActionType.ACCEPT_ACTIVITY_INVITE:
            breakdown["context_grounding"] = max(breakdown["context_grounding"], 0.88)
            breakdown["decomposition_alignment"] = max(breakdown["decomposition_alignment"], 0.86)
            breakdown["topic_alignment"] = max(breakdown["topic_alignment"], 0.86)
            breakdown["social_flow"] = max(breakdown["social_flow"], 0.84)

        if "clarification" in features.response_needs and action == ActionType.ASK_CLARIFICATION:
            breakdown["clarification_alignment"] = 0.88

        if "explanation" in features.response_needs and action in {
            ActionType.EXPLAIN_REASON,
            ActionType.EXPLAIN_CAPABILITIES,
            ActionType.ANSWER_IDENTITY,
        }:
            breakdown["explanation_alignment"] = 0.90

        if "acknowledgement" in features.response_needs and action in {
            ActionType.ACKNOWLEDGE,
            ActionType.GAME_ACCEPT_OR_DECLINE,
            ActionType.ACCEPT_ACTIVITY_INVITE,
            ActionType.REACT_LAUGH,
            ActionType.REACT_SURPRISE,
            ActionType.SMALL_TALK,
        }:
            breakdown["acknowledgement_alignment"] = 0.82
            if "soft_refusal" in features.pragmatic_cues and action == ActionType.ACKNOWLEDGE:
                breakdown["acknowledgement_alignment"] = 0.90
            if "polite_boundary" in features.pragmatic_cues and action == ActionType.ACKNOWLEDGE:
                breakdown["acknowledgement_alignment"] = max(
                    breakdown["acknowledgement_alignment"],
                    0.92,
                )

        if "tentative_request" in features.pragmatic_cues and action in {
            ActionType.ASK_LOCATION,
            ActionType.ASK_CLARIFICATION,
            ActionType.GAME_ACCEPT_OR_DECLINE,
            ActionType.WEATHER_LOOKUP,
            ActionType.SEARCH_ANSWER,
            ActionType.EXPLAIN_CAPABILITIES,
            ActionType.ANSWER_IDENTITY,
        }:
            breakdown["social_flow"] = max(breakdown["social_flow"], 0.76)

        if features.intent == Intent.TEASE and action in {ActionType.TEASE_BACK, ActionType.CONTINUE_CONVERSATION}:
            breakdown["social_flow"] = max(breakdown["social_flow"], 0.78)
            if "sarcastic_tease" in features.pragmatic_cues and action == ActionType.CONTINUE_CONVERSATION:
                breakdown["boundary_alignment"] = max(breakdown["boundary_alignment"], 0.68)

        if world_state.rapport_bucket == "warm" and action in {
            ActionType.CONTINUE_CONVERSATION,
            ActionType.SMALL_TALK,
            ActionType.TEASE_BACK,
        }:
            breakdown["relationship_alignment"] = max(breakdown["relationship_alignment"], 0.74)

        if world_state.rapport_bucket == "guarded" and action in {
            ActionType.ACKNOWLEDGE,
            ActionType.SMALL_TALK,
            ActionType.ASK_CLARIFICATION,
            ActionType.DEESCALATE,
            ActionType.CONTINUE_CONVERSATION,
        }:
            breakdown["relationship_alignment"] = max(breakdown["relationship_alignment"], 0.78)

        if world_state.boundary_history in {"recent_boundary", "active_boundary", "firm_boundary"} and action in {
            ActionType.ACKNOWLEDGE,
            ActionType.SMALL_TALK,
            ActionType.ASK_CLARIFICATION,
            ActionType.DEESCALATE,
        }:
            breakdown["boundary_alignment"] = 0.74
            if world_state.boundary_history in {"active_boundary", "firm_boundary"} and action == ActionType.ACKNOWLEDGE:
                breakdown["boundary_alignment"] = 0.82

        if world_state.user_directness_style == "indirect" and action in {
            ActionType.ACKNOWLEDGE,
            ActionType.SHARE_FEELING,
            ActionType.SMALL_TALK,
        }:
            breakdown["relationship_alignment"] = max(breakdown["relationship_alignment"], 0.76)

        if world_state.user_directness_style == "direct" and action in {
            ActionType.ASK_CLARIFICATION,
            ActionType.WEATHER_LOOKUP,
            ActionType.SEARCH_ANSWER,
        }:
            breakdown["relationship_alignment"] = max(breakdown["relationship_alignment"], 0.70)

        if features.topic_hint == "weather" and action in {ActionType.ASK_LOCATION, ActionType.WEATHER_LOOKUP, ActionType.SHARE_FEELING}:
            breakdown["topic_alignment"] = 0.80
        elif features.topic_hint == "knowledge" and action in {ActionType.SEARCH_ANSWER, ActionType.TELL_TIME, ActionType.NEWS_ANSWER}:
            breakdown["topic_alignment"] = 0.80
        elif features.topic_hint == "game" and action in {ActionType.GAME_CHAT, ActionType.GAME_ACCEPT_OR_DECLINE}:
            breakdown["topic_alignment"] = 0.78
        elif features.topic_hint == "music" and action == ActionType.MUSIC_CHAT:
            breakdown["topic_alignment"] = 0.78
        elif features.topic_hint == "media" and action == ActionType.RECOMMEND:
            breakdown["topic_alignment"] = 0.80
        elif features.topic_hint == "capability" and action == ActionType.EXPLAIN_CAPABILITIES:
            breakdown["topic_alignment"] = 0.82
        elif features.topic_hint == "identity" and action == ActionType.ANSWER_IDENTITY:
            breakdown["topic_alignment"] = 0.82

        return breakdown

    @staticmethod
    def _memory_bucket_present(world_state: WorldState, *bucket_names: DurableMemoryBucket | str) -> bool:
        if not world_state.durable_memory_buckets:
            return False
        normalized_bucket_names = {
            bucket.value if isinstance(bucket, DurableMemoryBucket) else bucket
            for bucket in bucket_names
        }
        return any(
            (bucket.value if isinstance(bucket, DurableMemoryBucket) else str(bucket)) in normalized_bucket_names
            for bucket, items in world_state.durable_memory_buckets.items()
            if items
        )

    @staticmethod
    def _has_stress_memory(world_state: WorldState) -> bool:
        return bool(world_state.relevant_stress_signals) or HierarchicalPolicy._memory_bucket_present(
            world_state,
            DurableMemoryBucket.RECOVERY,
            DurableMemoryBucket.SELF_WORTH,
            DurableMemoryBucket.COMPARISON,
        )

    @staticmethod
    def _has_social_memory(world_state: WorldState) -> bool:
        return bool(world_state.relevant_relationship_notes or world_state.relevant_open_loops) or HierarchicalPolicy._memory_bucket_present(
            world_state,
            DurableMemoryBucket.RELATIONSHIP,
            DurableMemoryBucket.OPEN_LOOP,
        )

    @staticmethod
    def _has_context_cue(world_state: WorldState, cue_type: str) -> bool:
        return any(cue.cue_type == cue_type for cue in world_state.recent_context_cues)

    @staticmethod
    def _context_dependency_at_least(world_state: WorldState, level: str) -> bool:
        ordering = {"low": 0, "medium": 1, "high": 2}
        current = ordering.get(world_state.context_dependency_level, 0)
        target = ordering.get(level, 0)
        return current >= target

    @staticmethod
    def _dedupe_candidates(candidates: list[PolicyCandidate]) -> list[PolicyCandidate]:
        deduped: list[PolicyCandidate] = []
        index_by_action: dict[ActionType, int] = {}
        for candidate in candidates:
            candidate = HierarchicalPolicy._normalize_candidate(candidate)
            existing_index = index_by_action.get(candidate.action)
            if existing_index is None:
                index_by_action[candidate.action] = len(deduped)
                deduped.append(candidate)
                continue

            existing = deduped[existing_index]
            existing_is_learned = HierarchicalPolicy._is_learned_only_candidate(existing)
            candidate_is_learned = HierarchicalPolicy._is_learned_only_candidate(candidate)
            merged_reason_parts = [existing.reason]
            if candidate.reason not in existing.reason:
                merged_reason_parts.append(candidate.reason)
            merged_score = max(existing.score, candidate.score)
            if existing_is_learned and not candidate_is_learned:
                merged_score = candidate.score
            elif candidate_is_learned and not existing_is_learned:
                merged_score = existing.score
            deduped[existing_index] = PolicyCandidate(
                action=existing.action,
                score=merged_score,
                reason=" | ".join(merged_reason_parts),
                score_breakdown=HierarchicalPolicy._merge_breakdowns(
                    existing.score_breakdown,
                    candidate.score_breakdown,
                ),
            )
        return deduped

    @staticmethod
    def _order_candidates(
        candidates: list[PolicyCandidate],
        *,
        selected_action: ActionType,
    ) -> list[PolicyCandidate]:
        selected = [item for item in candidates if item.action == selected_action]
        alternatives = [item for item in candidates if item.action != selected_action]
        alternatives.sort(key=lambda item: item.score, reverse=True)
        return selected + alternatives

    @staticmethod
    def _merge_breakdowns(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        merged = dict(left)
        for key, value in right.items():
            merged[key] = max(float(value), float(merged.get(key, 0.0)))
        return merged

    def _apply_social_followup_dampening(
        self,
        candidates: list[PolicyCandidate],
        *,
        features: MessageFeatures,
        world_state: WorldState,
    ) -> list[PolicyCandidate]:
        if "clarification" not in features.response_needs and "contextual_followup" not in features.pragmatic_cues:
            return candidates

        if (
            world_state.conversation_mode not in {"social", "support"}
            and world_state.last_action_hint not in self.SOCIAL_FOLLOWUP_ACTION_HINTS
            and "contextual_followup" not in features.pragmatic_cues
        ):
            return candidates

        dampened: list[PolicyCandidate] = []
        for candidate in candidates:
            if candidate.action != ActionType.ASK_CLARIFICATION:
                dampened.append(candidate)
                continue

            capped_score = min(candidate.score, 0.72)
            if capped_score == candidate.score:
                dampened.append(candidate)
                continue

            dampened.append(
                PolicyCandidate(
                    action=candidate.action,
                    score=capped_score,
                    reason=f"{candidate.reason} | social_followup_penalty",
                    score_breakdown=dict(candidate.score_breakdown),
                )
            )

        return dampened

    def _select_candidate(
        self,
        *,
        candidates: list[PolicyCandidate],
        rule_action: ActionType,
    ) -> PolicyCandidate:
        rule_candidate = next(
            (candidate for candidate in candidates if candidate.action == rule_action),
            None,
        )
        if rule_candidate is None:
            return max(candidates, key=lambda item: item.score)

        eligible_candidates = [
            candidate
            for candidate in candidates
            if candidate.action == rule_action or self._is_override_eligible(candidate)
        ]
        top_candidate = max(eligible_candidates, key=lambda item: item.score)
        if top_candidate.action == rule_action:
            return top_candidate
        if top_candidate.score >= rule_candidate.score + self.OVERRIDE_MARGIN:
            return top_candidate
        return rule_candidate

    @staticmethod
    def _is_override_eligible(candidate: PolicyCandidate) -> bool:
        return not HierarchicalPolicy._is_learned_only_candidate(candidate)

    @staticmethod
    def _is_learned_only_candidate(candidate: PolicyCandidate) -> bool:
        reason = candidate.reason.strip()
        return reason.startswith("learned_policy(") and "|" not in reason

    @staticmethod
    def _override_reason(
        *,
        rule_decision: ActionDecision,
        selected_candidate: PolicyCandidate,
    ) -> str:
        return (
            "초기 규칙 선택보다 후보 점수 비교에서 이 대응이 더 높게 나와 최종 선택을 옮겼습니다. "
            + selected_candidate.reason
        )

    @staticmethod
    def _normalize_candidate(candidate) -> PolicyCandidate:
        if isinstance(candidate, PolicyCandidate):
            return candidate
        return PolicyCandidate(
            action=candidate.action,
            score=float(candidate.score),
            reason=str(candidate.reason),
            score_breakdown=dict(getattr(candidate, "score_breakdown", {})),
        )
