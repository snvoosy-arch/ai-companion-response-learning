from __future__ import annotations

import asyncio
from datetime import date
import re

from bot_shared.rejected_generations import try_record_rejected_generation

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.character_state import (
    apply_state_delta,
    build_evidence_packet,
    choose_state_action,
    infer_state_delta,
    remember_state_action,
)
from predictive_bot.core.classifier import IntentClassifier
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    EngineResult,
    Intent,
    MessageFeatures,
    StateInferenceEntry,
    TurnRecord,
)
from predictive_bot.core.memory import (
    DurableMemoryBucket,
    DurableMemoryEntry,
    classify_durable_memory_bucket,
    durable_memory_entry_within_retention,
    prepare_durable_memory_capture_text,
)
from predictive_bot.core.performance import build_black_turn_packet
from predictive_bot.core.phrasing import build_phrasing_plan
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.response_plan import build_response_plan
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import StateStore
from predictive_bot.core.trace_builder import DecisionTraceBuilder
from predictive_bot.core.tools import (
    BasicKnowledgeService,
    CuratedRecommendationService,
    GoogleNewsRssService,
    KnowledgeLookupError,
    NewsLookupError,
    OpenMeteoWeatherService,
    RecommendationLookupError,
    RecommendationService,
    RecommendationAnswer,
    SystemTimeService,
    WeatherLookupError,
)
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder


class PredictiveEngine:
    _NEGATIVE_PREFERENCE_MARKERS = (
        "안 좋아해",
        "안좋아해",
        "싫어해",
        "싫어함",
        "별로야",
        "별로 안 좋아해",
        "안 끌려",
        "안땡겨",
    )
    _POSITIVE_PREFERENCE_MARKERS = (
        "좋아해",
        "좋아함",
        "좋아하는 편",
        "좋아하는데",
        "선호해",
        "취향이야",
        "취향이더라",
    )
    _DURABLE_MEMORY_MARKERS = (
        "요즘",
        "최근",
        "자꾸",
        "계속",
        "아직",
        "오랜만",
        "준비",
        "면접",
        "시험",
        "프로젝트",
        "취준",
        "취업",
        "이직",
        "퇴사",
        "회사",
        "학교",
        "이사",
        "가족",
        "친구",
        "연락",
        "비교",
        "씁쓸",
        "서운",
        "허탈",
        "허무",
        "부럽",
        "질투",
        "병원",
        "약",
        "불안",
        "우울",
        "공황",
        "불면",
        "잠",
        "아프",
        "수술",
        "치료",
    )
    _NEWS_TOPIC_KEYWORDS = {
        "ai": ("ai", "인공지능", "챗gpt", "chatgpt", "오픈ai", "openai", "llm"),
        "economy": ("경제", "주식", "증시", "코스피", "코스닥", "물가", "금리", "환율"),
        "game": ("게임", "겜", "닌텐도", "플스", "playstation", "xbox", "스팀", "lck", "e스포츠", "esports"),
        "sports": ("스포츠", "야구", "축구", "농구", "배구", "mlb", "epl", "nba", "kbo"),
        "politics": ("정치", "대통령", "국회", "선거", "정부", "여당", "야당"),
        "entertainment": ("연예", "아이돌", "영화", "드라마", "배우", "케이팝", "kpop"),
        "tech": ("테크", "기술", "반도체", "스마트폰", "애플", "삼성", "구글", "메타"),
    }
    _NEWS_TOPIC_LABELS = {
        "ai": "AI",
        "economy": "경제",
        "game": "게임",
        "sports": "스포츠",
        "politics": "정치",
        "entertainment": "연예",
        "tech": "테크",
    }

    def __init__(
        self,
        *,
        classifier: IntentClassifier,
        goal_manager: GoalManager,
        action_selector: ActionSelector,
        world_state_builder: WorldStateBuilder | None = None,
        policy: HierarchicalPolicy | None = None,
        trace_builder: DecisionTraceBuilder | None = None,
        renderer: ResponseRenderer,
        verifier: ResponseVerifier | None = None,
        weather_service: OpenMeteoWeatherService,
        knowledge_service: BasicKnowledgeService | None = None,
        time_service: SystemTimeService | None = None,
        news_service: GoogleNewsRssService | None = None,
        recommendation_service: RecommendationService | None = None,
        state_store: StateStore,
    ) -> None:
        self.classifier = classifier
        self.goal_manager = goal_manager
        self.action_selector = action_selector
        self.world_state_builder = world_state_builder or WorldStateBuilder()
        self.policy = policy or HierarchicalPolicy(action_selector=action_selector)
        self.trace_builder = trace_builder or DecisionTraceBuilder()
        self.renderer = renderer
        self.verifier = verifier or ResponseVerifier()
        self.weather_service = weather_service
        self.knowledge_service = knowledge_service or BasicKnowledgeService()
        self.time_service = time_service or SystemTimeService()
        self.news_service = news_service or GoogleNewsRssService()
        self.recommendation_service = recommendation_service or CuratedRecommendationService()
        self.state_store = state_store

    async def respond(self, user_id: str, text: str) -> EngineResult:
        state = self.state_store.get_or_create(user_id)
        features = self.classifier.classify(text, state)
        preference_update = self._extract_preference_update(features)

        if features.location:
            state.known_location = features.location
        if preference_update is not None:
            preference_key, preference_value = preference_update
            state.preference_memory[preference_key] = preference_value

        world_state = self.world_state_builder.build(user_id=user_id, features=features, state=state)
        evidence_packet = build_evidence_packet(features=features, world_state=world_state)
        state_delta = infer_state_delta(evidence=evidence_packet, state=state.character_state)
        character_state = apply_state_delta(state.character_state, state_delta)
        state_action = choose_state_action(
            evidence=evidence_packet,
            state=character_state,
            delta=state_delta,
        )
        world_state.evidence_packet = evidence_packet
        world_state.state_delta = state_delta
        world_state.character_state = character_state
        world_state.state_action = state_action
        world_state.evidence.extend(
            [
                f"character_mood={character_state.mood}",
                f"character_topic_focus={character_state.topic_focus or 'none'}",
                f"state_action={state_action.action.value}:{state_action.mode}",
            ]
        )
        world_state.inference_trace.extend(
            [
                StateInferenceEntry(
                    field="character_state",
                    value=character_state.mood,
                    reasons=[
                        f"energy={character_state.energy:.2f}",
                        f"curiosity={character_state.curiosity:.2f}",
                        f"affinity={character_state.affinity:.2f}",
                        f"pressure={character_state.pressure:.2f}",
                        f"engagement={character_state.engagement:.2f}",
                    ],
                ),
                StateInferenceEntry(
                    field="state_delta",
                    value=state_delta.topic_focus,
                    reasons=list(state_delta.reasons),
                ),
                StateInferenceEntry(
                    field="state_action",
                    value=state_action.action.value,
                    reasons=[state_action.reason, f"mode={state_action.mode}", f"score={state_action.score:.2f}"],
                ),
            ]
        )
        state.character_state = character_state
        goals = self.goal_manager.build_goals(features, state)
        decision, policy_trace = self.policy.decide(
            features=features,
            state=state,
            goals=goals,
            world_state=world_state,
        )
        explanation_trace = None
        if decision.action == ActionType.EXPLAIN_REASON:
            explanation_trace = self.state_store.get_latest_decision_trace(user_id)

        weather = None
        if decision.action == ActionType.WEATHER_LOOKUP:
            try:
                weather = await self.weather_service.get_current_weather(decision.slots["location"])
            except WeatherLookupError:
                failed_location = decision.slots.get("location", state.known_location or "지역")
                decision = ActionDecision(
                    action=ActionType.WEATHER_UNAVAILABLE,
                    reason="날씨 조회가 필요했지만 외부 조회가 실패해 재시도를 안내하는 쪽이 맞습니다.",
                    goals=goals,
                    slots={"location": failed_location},
                    response_style="짧고 사실 기반으로 재시도를 안내하는 말투",
                )

        if decision.action == ActionType.SEARCH_ANSWER:
            try:
                knowledge_answer = self.knowledge_service.answer(text)
                decision.slots.update(
                    {
                        "knowledge_query_type": knowledge_answer.query_type,
                        "knowledge_subject": knowledge_answer.subject,
                        "knowledge_answer": knowledge_answer.answer,
                        "knowledge_source": knowledge_answer.source,
                        "knowledge_grounded": "true",
                    }
                )
            except KnowledgeLookupError:
                pass

        if decision.action == ActionType.TELL_TIME:
            time_answer = self.time_service.get_current_time()
            decision.slots.update(self._build_time_slots(features.normalized, time_answer))

        if decision.action == ActionType.NEWS_ANSWER:
            try:
                fetch_limit = 8 if features.news_topic else 3
                headlines = await asyncio.to_thread(self.news_service.top_headlines, limit=fetch_limit)
                selected_headlines, topic_matched = self._select_news_headlines(
                    headlines=headlines,
                    news_topic=features.news_topic,
                    limit=3,
                )
                decision.slots.update(
                    {
                        "news_summary": self._render_news_summary(
                            headlines=selected_headlines,
                            news_topic=features.news_topic,
                            topic_matched=topic_matched,
                        ),
                        "news_count": str(len(selected_headlines)),
                        "news_topic": features.news_topic or "",
                        "news_filter_applied": "true" if features.news_topic else "false",
                        "news_topic_match": "true" if topic_matched else "false",
                        "news_titles": "|".join(item.title for item in selected_headlines),
                        "knowledge_source": "google_news_rss",
                        "knowledge_grounded": "true",
                    }
                )
            except NewsLookupError:
                pass

        if decision.action == ActionType.RECOMMEND:
            try:
                recommendation_answer = self.recommendation_service.recommend_media(
                    query=text,
                    preferences=state.preference_memory,
                    limit=3,
                )
                decision.slots.update(
                    {
                        "recommendation_text": self._render_media_recommendation_text(
                            features=features,
                            state=state,
                            answer=recommendation_answer,
                            preference_update=preference_update,
                        ),
                        "recommendation_focus": recommendation_answer.focus_label or "",
                        "recommendation_titles": "|".join(item.title for item in recommendation_answer.items),
                        "knowledge_source": recommendation_answer.source,
                    }
                )
            except RecommendationLookupError:
                pass

        if decision.action == ActionType.MUSIC_CHAT:
            try:
                recommendation_answer = self.recommendation_service.recommend_music(
                    query=text,
                    preferences=state.preference_memory,
                    limit=3,
                )
                decision.slots.update(
                    {
                        "music_text": self._render_music_recommendation_text(
                            features=features,
                            state=state,
                            answer=recommendation_answer,
                            preference_update=preference_update,
                        ),
                        "music_focus": recommendation_answer.focus_label or "",
                        "music_titles": "|".join(item.title for item in recommendation_answer.items),
                        "knowledge_source": recommendation_answer.source,
                    }
                )
            except RecommendationLookupError:
                pass

        if preference_update is not None:
            preference_key, preference_value = preference_update
            decision.slots.update(
                {
                    "preference_update_key": preference_key,
                    "preference_update_value": preference_value,
                }
            )

        phrasing_plan = build_phrasing_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
        )
        decision.phrasing_plan = phrasing_plan
        decision_trace = self.trace_builder.build(
            user_id=user_id,
            features=features,
            world_state=world_state,
            decision=decision,
            policy_trace=policy_trace,
        )
        if world_state.grounding_bundle is None:
            world_state.grounding_bundle = decision_trace.grounding_bundle
        response_plan = build_response_plan(
            features=features,
            decision=decision,
            state=state,
            world_state=world_state,
            phrasing_plan=phrasing_plan,
        )
        decision.response_plan = response_plan
        decision_trace.response_plan = response_plan
        reply = await self.renderer.render(
            features=features,
            decision=decision,
            state=state,
            weather=weather,
            world_state=world_state,
            policy_trace=policy_trace,
            decision_trace=decision_trace,
            explanation_trace=explanation_trace,
            phrasing_plan=phrasing_plan,
        )
        llm_used = bool(getattr(self.renderer, "last_llm_used", False))
        llm_fallback_reason = getattr(self.renderer, "last_llm_fallback_reason", None)
        llm_generation_issue = getattr(self.renderer, "last_llm_generation_issue", None)
        render_source = str(getattr(self.renderer, "last_render_source", "") or "")
        draft_utterance = getattr(self.renderer, "last_draft_utterance", None)
        self._coerce_daily_draft_action(decision=decision, draft_utterance=draft_utterance)
        rendered_reply = reply
        verification = self.verifier.verify(
            reply=reply,
            decision=decision,
            state=state,
            world_state=world_state,
            weather=weather,
            draft_utterance=draft_utterance,
        )
        if verification.revised_reply:
            reply = verification.revised_reply

        decision_trace.output_text = reply
        decision_trace.verification_issues = list(verification.issues)
        decision_trace.llm_used = llm_used
        decision_trace.llm_fallback_reason = llm_fallback_reason
        performance_packet = build_black_turn_packet(
            reply=reply,
            features=features,
            decision=decision,
            world_state=world_state,
            policy_trace=policy_trace,
            verification=verification,
            weather=weather,
            llm_used=llm_used,
            llm_fallback_reason=llm_fallback_reason,
        )

        self._update_conversation_state(
            state=state,
            features=features,
            decision=decision,
        )
        state.character_state = remember_state_action(state.character_state, state_action)
        durable_memory_note = self._extract_durable_memory_note(text=text, features=features)
        if durable_memory_note is not None:
            self._remember_durable_memory(state, durable_memory_note)

        self.state_store.append_turn(
            user_id,
            TurnRecord(
                user_text=text,
                bot_text=reply,
                action=decision.action,
                decision_reason=decision.reason,
            ),
        )
        self.state_store.save_decision_trace(decision_trace)
        self._record_rejected_generation_if_needed(
            user_text=text,
            rendered_reply=rendered_reply,
            final_reply=reply,
            decision=decision,
            verification_issues=list(verification.issues),
            draft_utterance=draft_utterance,
            llm_generation_issue=llm_generation_issue,
            llm_fallback_reason=llm_fallback_reason,
            llm_used=llm_used,
        )

        return EngineResult(
            reply=reply,
            decision=decision,
            features=features,
            weather=weather,
            decision_trace=decision_trace,
            explanation_trace=explanation_trace,
            world_state=world_state,
            policy_trace=policy_trace,
            verification=verification,
            phrasing_plan=phrasing_plan,
            response_plan=response_plan,
            draft_utterance=draft_utterance,
            evidence_packet=evidence_packet,
            state_delta=state_delta,
            character_state=state.character_state,
            state_action=state_action,
            llm_used=llm_used,
            llm_fallback_reason=llm_fallback_reason,
            llm_generation_issue=llm_generation_issue,
            render_source=render_source,
            performance_packet=performance_packet,
        )

    @staticmethod
    def _coerce_daily_draft_action(
        *,
        decision: ActionDecision,
        draft_utterance: dict[str, object] | None,
    ) -> None:
        if not isinstance(draft_utterance, dict):
            return
        direct_reason = str(draft_utterance.get("direct_surface_reason") or "").strip()
        if not direct_reason.startswith("korean_daily_"):
            return
        if decision.action == ActionType.GAME_CHAT and direct_reason.startswith(
            (
                "korean_daily_expansion_game_",
                "korean_daily_question_game_",
                "korean_daily_basic_game_",
            )
        ):
            return
        if decision.action == ActionType.CONTINUE_CONVERSATION and direct_reason in {
            "korean_daily_basic_presence_check",
        }:
            draft_utterance["action"] = ActionType.CONTINUE_CONVERSATION.value
            return
        if decision.action not in {
            ActionType.ASK_CLARIFICATION,
            ActionType.ASK_LOCATION,
            ActionType.SEARCH_ANSWER,
            ActionType.DEESCALATE,
            ActionType.CONTINUE_CONVERSATION,
            ActionType.GAME_CHAT,
            ActionType.ANSWER_IDENTITY,
            ActionType.EXPLAIN_CAPABILITIES,
        }:
            return

        feeling_markers = (
            "emotion",
            "feeling",
            "mood",
            "hurt",
            "lonely",
            "worry",
            "self_praise",
            "low_energy",
            "quiet_rest",
        )
        next_action = (
            ActionType.SHARE_FEELING
            if any(marker in direct_reason for marker in feeling_markers)
            else ActionType.SHARE_OPINION
        )
        decision.action = next_action
        decision.awaiting_slot = None
        decision.reason_code = "daily_draft.action.coerced"
        decision.reason = "Black draft가 일상 대화 의미를 확정해 확인 질문 대신 바로 반응합니다."
        if "daily_draft_action_coerced" not in decision.reason_flags:
            decision.reason_flags.append("daily_draft_action_coerced")
        if decision.response_plan is not None:
            decision.response_plan.action = next_action
        draft_utterance["action"] = next_action.value

    def _record_rejected_generation_if_needed(
        self,
        *,
        user_text: str,
        rendered_reply: str,
        final_reply: str,
        decision: ActionDecision,
        verification_issues: list[str],
        draft_utterance: dict[str, object] | None,
        llm_generation_issue: str | None,
        llm_fallback_reason: str | None,
        llm_used: bool,
    ) -> None:
        issues = _black_rejection_issues(
            verification_issues=verification_issues,
            llm_generation_issue=llm_generation_issue,
            llm_fallback_reason=llm_fallback_reason,
            reply=rendered_reply,
        )
        if not issues:
            return
        llm_client = getattr(self.renderer, "llm_client", None)
        model = str(getattr(llm_client, "model_name_or_path", "") or getattr(llm_client, "model", "") or "")
        raw_generation = str(getattr(llm_client, "last_raw_generation", "") or rendered_reply)
        try_record_rejected_generation(
            speaker="black",
            source="black.engine",
            model=model,
            input_text=user_text,
            raw_reply=raw_generation,
            final_reply=final_reply,
            issues=issues,
            action=decision.action.value,
            decision=decision.reason_code or decision.reason,
            draft_utterance=draft_utterance,
            metadata={
                "llm_used": llm_used,
                "llm_generation_issue": llm_generation_issue or "",
                "llm_fallback_reason": llm_fallback_reason or "",
                "rendered_reply": rendered_reply,
            },
        )

    @classmethod
    def _extract_durable_memory_note(cls, *, text: str, features: MessageFeatures) -> str | None:
        normalized = text.strip()
        if not normalized:
            return None
        if features.is_question:
            return None
        if len(normalized) < 10:
            return None
        if features.requests_external_fact:
            return None
        if features.intent in {Intent.HELP, Intent.WHO_ARE_YOU, Intent.TIME_DATE, Intent.WEATHER, Intent.NEWS}:
            return None
        if not any(marker in normalized for marker in cls._DURABLE_MEMORY_MARKERS):
            return None

        return prepare_durable_memory_capture_text(normalized, max_length=120)

    @staticmethod
    def _remember_durable_memory(state: ConversationState, memory_note: str) -> None:
        normalized = memory_note.strip()
        if not normalized:
            return
        bucket = classify_durable_memory_bucket(normalized)
        entries = [
            item
            for item in state.durable_memory
            if item.text != normalized and durable_memory_entry_within_retention(item, current_turn=state.turn_count)
        ]
        entries.append(
            DurableMemoryEntry(
                bucket=bucket,
                text=normalized,
                source="turn",
                captured_turn=state.turn_count,
            )
        )

        retained: list[DurableMemoryEntry] = []
        per_bucket: dict[DurableMemoryBucket, int] = {}
        for entry in reversed(entries):
            if len(retained) >= 12:
                break
            bucket_count = per_bucket.get(entry.bucket, 0)
            if bucket_count >= 4:
                continue
            retained.append(entry)
            per_bucket[entry.bucket] = bucket_count + 1
        state.durable_memory = list(reversed(retained))

    @classmethod
    def _extract_preference_update(cls, features: MessageFeatures) -> tuple[str, str] | None:
        if features.is_question:
            return None
        if features.topic_hint not in {"media", "music"}:
            return None

        preference_subject = cls._extract_preference_subject(
            features.normalized,
            markers=cls._NEGATIVE_PREFERENCE_MARKERS,
            topic_hint=features.topic_hint,
        )
        if preference_subject is not None:
            return (f"{features.topic_hint}_dislike", preference_subject)

        preference_subject = cls._extract_preference_subject(
            features.normalized,
            markers=cls._POSITIVE_PREFERENCE_MARKERS,
            topic_hint=features.topic_hint,
        )
        if preference_subject is not None:
            return (f"{features.topic_hint}_like", preference_subject)
        return None

    @classmethod
    def _extract_preference_subject(
        cls,
        normalized: str,
        *,
        markers: tuple[str, ...],
        topic_hint: str,
    ) -> str | None:
        marker_pattern = "|".join(re.escape(marker) for marker in markers)
        patterns = (
            rf"^(?:난|나는|내가|전|저는|요즘|원래|보통|개인적으로)?\s*(?P<subject>.+?)\s*(?:은|는|이|가|를|을)?\s*(?:{marker_pattern})(?:$|[.!?])",
            rf"^(?:난|나는|내가|전|저는|요즘|원래|보통|개인적으로)?\s*(?:{marker_pattern})\s*하는\s*(?P<subject>.+?)(?:$|[.!?])",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            cleaned = cls._clean_preference_subject(match.group("subject"), topic_hint=topic_hint)
            if cleaned:
                return cleaned
        return None

    @staticmethod
    def _clean_preference_subject(subject: str, *, topic_hint: str) -> str | None:
        cleaned = subject.strip(" \"'`.,!?")
        cleaned = re.sub(r"^(난|나는|내가|전|저는|요즘|원래|보통|개인적으로)\s+", "", cleaned)
        cleaned = re.sub(r"^(영화|드라마|음악|노래|플리)\s*는\s+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if topic_hint == "media":
            cleaned = cleaned.removesuffix(" 쪽")
        if topic_hint == "music":
            cleaned = cleaned.removesuffix(" 스타일")

        blocked = {
            "이거",
            "그거",
            "저거",
            "이런 거",
            "그런 거",
            "요즘",
            "그냥",
        }
        if cleaned in blocked or len(cleaned) < 2 or len(cleaned) > 24:
            return None
        return cleaned

    @staticmethod
    def _detect_time_query_type(normalized: str) -> str:
        asks_weekday = "요일" in normalized
        asks_date = any(token in normalized for token in ("날짜", "며칠", "몇일", "몇 일")) or bool(
            re.search(r"몇\s*월", normalized)
        )

        if asks_date and asks_weekday:
            return "date_and_weekday"
        if asks_weekday:
            return "weekday"
        if asks_date:
            return "date"
        return "time"

    @staticmethod
    def _format_korean_weekday(formatted_date: str) -> str | None:
        try:
            weekday_index = date.fromisoformat(formatted_date).weekday()
        except ValueError:
            return None

        weekdays = (
            "월요일",
            "화요일",
            "수요일",
            "목요일",
            "금요일",
            "토요일",
            "일요일",
        )
        return weekdays[weekday_index]

    @classmethod
    def _build_time_slots(cls, normalized: str, time_answer) -> dict[str, str]:
        query_type = cls._detect_time_query_type(normalized)
        weekday_name = cls._format_korean_weekday(time_answer.formatted_date)

        if query_type == "date_and_weekday":
            if weekday_name:
                time_text = f"오늘은 {time_answer.formatted_date}, {weekday_name}이야."
            else:
                time_text = f"오늘 날짜는 {time_answer.formatted_date}이야."
        elif query_type == "weekday":
            if weekday_name:
                time_text = f"오늘은 {weekday_name}이야."
            else:
                time_text = f"오늘 날짜는 {time_answer.formatted_date}이야."
        elif query_type == "date":
            time_text = f"오늘 날짜는 {time_answer.formatted_date}이야."
        else:
            time_text = f"지금 시간은 {time_answer.formatted_time}이야."

        slots = {
            "time_text": time_text,
            "time_query_type": query_type,
            "time_date": time_answer.formatted_date,
            "time_timezone": time_answer.timezone_name,
            "knowledge_source": time_answer.source,
            "knowledge_grounded": "true",
        }
        if weekday_name:
            slots["time_weekday"] = weekday_name
        return slots

    @staticmethod
    def _format_recommendation_lines(items) -> str:
        return "\n".join(
            f"{index}. {item.title}: {item.detail}"
            for index, item in enumerate(items, start=1)
        )

    @classmethod
    def _select_news_headlines(
        cls,
        *,
        headlines,
        news_topic: str | None,
        limit: int,
    ):
        if not news_topic:
            return list(headlines[:limit]), False

        keywords = cls._NEWS_TOPIC_KEYWORDS.get(news_topic, ())
        matched = [
            item
            for item in headlines
            if any(keyword in f"{item.title} {item.source}".lower() for keyword in keywords)
        ]
        if matched:
            return matched[:limit], True
        return list(headlines[:limit]), False

    @classmethod
    def _render_news_summary(cls, *, headlines, news_topic: str | None, topic_matched: bool) -> str:
        if news_topic:
            topic_label = cls._NEWS_TOPIC_LABELS.get(news_topic, news_topic)
            if topic_matched:
                lead = f"지금 `{topic_label}` 쪽으로 보이는 뉴스는 이 정도야."
            else:
                lead = f"지금 `{topic_label}` 쪽으로 딱 맞는 헤드라인이 적어서, 전체에서 눈에 띄는 것부터 묶었어."
        else:
            lead = "지금 눈에 띄는 뉴스는 이 정도야."
        return lead + "\n" + "\n".join(
            f"{index}. {item.title} ({item.source})"
            for index, item in enumerate(headlines, start=1)
        )

    @classmethod
    def _render_media_recommendation_text(
        cls,
        *,
        features: MessageFeatures,
        state: ConversationState,
        answer: RecommendationAnswer,
        preference_update: tuple[str, str] | None,
    ) -> str:
        if preference_update is not None and preference_update[0] == "media_like":
            lead = f"오케이. {preference_update[1]} 좋아하는 걸로 기억해둘게. 바로 떠오르는 건 이런 쪽이야."
        elif preference_update is not None and preference_update[0] == "media_dislike":
            lead = f"좋아. {preference_update[1]} 쪽은 덜 끌리는 걸로 기억해둘게. 그걸 빼고 보면 이런 쪽이 맞아."
        elif state.preference_memory.get("media_like"):
            lead = f"지난번에 {state.preference_memory['media_like']} 좋아한다고 했지. 그 결로 바로 가면 이런 쪽이야."
        elif state.preference_memory.get("media_dislike"):
            lead = f"지난번에 {state.preference_memory['media_dislike']} 쪽은 덜 끌린다고 했지. 그걸 빼고 고르면 이런 쪽이야."
        elif answer.focus_label:
            lead = f"{answer.focus_label} 쪽으로 바로 던지면 이런 후보가 무난해."
        elif features.is_question:
            lead = "가볍게 바로 던지면 이런 쪽이 무난해."
        else:
            lead = "이 결이면 바로 떠오르는 건 이런 쪽이야."
        return f"{lead}\n{cls._format_recommendation_lines(answer.items)}"

    @classmethod
    def _render_music_recommendation_text(
        cls,
        *,
        features: MessageFeatures,
        state: ConversationState,
        answer: RecommendationAnswer,
        preference_update: tuple[str, str] | None,
    ) -> str:
        if preference_update is not None and preference_update[0] == "music_like":
            lead = f"오케이. {preference_update[1]} 좋아하는 쪽으로 기억해둘게. 바로 떠오르는 건 이런 곡들이야."
        elif preference_update is not None and preference_update[0] == "music_dislike":
            lead = f"좋아. {preference_update[1]} 쪽은 덜 끌리는 걸로 기억해둘게. 그걸 빼고 보면 이런 곡이 맞아."
        elif state.preference_memory.get("music_like"):
            lead = f"지난번에 {state.preference_memory['music_like']} 좋아한다고 했지. 그 결로 바로 가면 이런 곡들이야."
        elif state.preference_memory.get("music_dislike"):
            lead = f"지난번에 {state.preference_memory['music_dislike']} 쪽은 덜 끌린다고 했지. 그걸 빼고 보면 이런 쪽이야."
        elif answer.focus_label:
            lead = f"{answer.focus_label} 쪽으로 바로 던지면 이런 곡이 무난해."
        elif features.is_question:
            lead = "가볍게 바로 틀기엔 이런 곡들이 무난해."
        else:
            lead = "이 결이면 바로 떠오르는 건 이런 곡들이야."
        return f"{lead}\n{cls._format_recommendation_lines(answer.items)}"

    @staticmethod
    def _clamp(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _move_toward(value: float, *, target: float, amount: float) -> float:
        if value < target:
            return min(target, value + amount)
        if value > target:
            return max(target, value - amount)
        return value

    @staticmethod
    def _has_recent_repair_context(state: ConversationState) -> bool:
        if state.last_action == ActionType.DEESCALATE or state.last_intent == Intent.HOSTILE:
            return True
        if state.tension >= 0.2:
            return True
        recent_turns = state.recent_turns[-3:]
        return any(turn.action == ActionType.DEESCALATE for turn in recent_turns)

    @staticmethod
    def _is_social_recovery_turn(features: MessageFeatures, decision: ActionDecision) -> bool:
        if decision.action not in {
            ActionType.SMALL_TALK,
            ActionType.CONTINUE_CONVERSATION,
            ActionType.SHARE_FEELING,
            ActionType.ACKNOWLEDGE,
        }:
            return False
        if features.pragmatic_cues:
            return False
        if features.intent in {
            Intent.GREETING,
            Intent.THANKS,
            Intent.SMALLTALK_GENERIC,
            Intent.SMALLTALK_OPINION,
        }:
            return True
        return features.intent == Intent.SMALLTALK_FEELING and features.sentiment != "negative"

    def _update_conversation_state(
        self,
        *,
        state: ConversationState,
        features: MessageFeatures,
        decision: ActionDecision,
    ) -> None:
        repair_context_active = self._has_recent_repair_context(state)
        state.turn_count += 1
        state.last_intent = features.intent
        state.last_action = decision.action
        state.awaiting_slot = decision.awaiting_slot

        if features.intent == Intent.HOSTILE:
            state.tension = self._clamp(state.tension + 0.25)
        else:
            state.tension = self._clamp(state.tension - 0.10)
        if "repair_attempt" in features.pragmatic_cues:
            state.tension = self._clamp(state.tension - 0.08)

        rapport_delta = 0.0
        if features.intent == Intent.HOSTILE:
            rapport_delta -= 0.20
        elif features.intent in {Intent.GREETING, Intent.THANKS}:
            rapport_delta += 0.08
        elif features.intent in {Intent.SMALLTALK_GENERIC, Intent.SMALLTALK_FEELING, Intent.SMALLTALK_OPINION}:
            rapport_delta += 0.03
        elif features.intent == Intent.DENY and "soft_refusal" not in features.pragmatic_cues:
            rapport_delta -= 0.04
        if "repair_attempt" in features.pragmatic_cues:
            rapport_delta += 0.04
        if "relationship_check" in features.pragmatic_cues and repair_context_active:
            rapport_delta += 0.03

        if decision.action in {ActionType.SHARE_FEELING, ActionType.ACKNOWLEDGE, ActionType.SMALL_TALK}:
            rapport_delta += 0.03
        if decision.action == ActionType.DEESCALATE:
            rapport_delta -= 0.02
        state.rapport = self._clamp(state.rapport + rapport_delta)

        boundary_delta = -0.10
        if "soft_refusal" in features.pragmatic_cues:
            boundary_delta += 0.32
        if "polite_boundary" in features.pragmatic_cues:
            boundary_delta += 0.26
        if any(
            cue in features.pragmatic_cues
            for cue in {"testing_the_waters", "self_conscious_check", "face_saving_retreat"}
        ):
            boundary_delta += 0.22
        if "relationship_check" in features.pragmatic_cues:
            boundary_delta += 0.08 if repair_context_active else 0.22
        if "repair_attempt" in features.pragmatic_cues:
            boundary_delta -= 0.08
        if features.intent == Intent.HOSTILE:
            boundary_delta += 0.18
        if decision.action in {ActionType.SHARE_FEELING, ActionType.ACKNOWLEDGE, ActionType.SMALL_TALK}:
            boundary_delta -= 0.04
        state.boundary_pressure = self._clamp(state.boundary_pressure + boundary_delta)

        directness_delta = 0.0
        if any(
            cue in features.pragmatic_cues
            for cue in {
                "soft_refusal",
                "hedging",
                "tentative_request",
                "polite_boundary",
                "indirect_negation",
                "testing_the_waters",
                "self_conscious_check",
                "face_saving_retreat",
                "deferred_acceptance",
                "repair_attempt",
            }
        ):
            directness_delta += 0.18
        if "relationship_check" in features.pragmatic_cues:
            directness_delta += 0.10 if repair_context_active else 0.18
        elif features.intent == Intent.HOSTILE:
            directness_delta -= 0.22
        elif features.is_question and features.speech_act == "ask":
            directness_delta -= 0.06
        state.directness_score = self._clamp(state.directness_score + directness_delta)

        if self._is_social_recovery_turn(features, decision):
            state.boundary_pressure = self._clamp(state.boundary_pressure - 0.08)
            state.directness_score = self._move_toward(state.directness_score, target=0.5, amount=0.08)
            state.rapport = self._clamp(state.rapport + 0.03)


def _black_rejection_issues(
    *,
    verification_issues: list[str],
    llm_generation_issue: str | None,
    llm_fallback_reason: str | None,
    reply: str,
) -> list[str]:
    issues: list[str] = [str(item) for item in verification_issues if str(item).strip()]
    if llm_generation_issue:
        for item in (part.strip() for part in llm_generation_issue.split(";")):
            if item and not item.endswith(":recovered"):
                issues.append(f"llm_generation_issue:{item}")
    if llm_fallback_reason:
        issues.append(f"llm_fallback_reason:{llm_fallback_reason}")
    if not reply.strip() and "empty_reply" not in issues:
        issues.append("empty_reply")
    return list(dict.fromkeys(issues))
