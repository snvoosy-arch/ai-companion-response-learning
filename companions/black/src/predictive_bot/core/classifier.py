from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from predictive_bot.core.intent_model import CharNgramCentroidModel
from predictive_bot.core.meaning_resolver import MeaningResolver
from predictive_bot.core.models import (
    ActionType,
    ClassifierEvidence,
    ConversationState,
    Intent,
    MeaningPacket,
    MeaningSignal,
    MessageFeatures,
    ScoredLabel,
)


class IntentClassifier(Protocol):
    def classify(self, text: str, state: ConversationState) -> MessageFeatures: ...


class HeuristicIntentClassifier:
    _news_topic_keywords = {
        "ai": ("ai", "인공지능", "챗gpt", "chatgpt", "오픈ai", "openai", "llm"),
        "economy": ("경제", "주식", "증시", "코스피", "코스닥", "물가", "금리", "환율"),
        "game": ("게임", "겜", "닌텐도", "플스", "playstation", "xbox", "스팀", "lck", "e스포츠", "esports"),
        "sports": ("스포츠", "야구", "축구", "농구", "배구", "mlb", "epl", "nba", "kbo"),
        "politics": ("정치", "대통령", "국회", "선거", "정부", "여당", "야당"),
        "entertainment": ("연예", "아이돌", "영화", "드라마", "배우", "케이팝", "kpop"),
        "tech": ("테크", "기술", "반도체", "스마트폰", "애플", "삼성", "구글", "메타"),
    }

    _positive_preference_markers = (
        "좋아해",
        "좋아함",
        "좋아하는 편",
        "좋아하는데",
        "선호해",
        "취향이야",
        "취향이더라",
    )
    _negative_preference_markers = (
        "안 좋아해",
        "안좋아해",
        "싫어해",
        "싫어함",
        "별로야",
        "별로 안 좋아해",
        "안 끌려",
        "안땡겨",
    )
    _greetings = ("안녕", "하이", "ㅎㅇ", "hello", "hey", "와썹")
    _thanks = ("고마", "감사", "thx", "thanks")
    _help = ("도움", "help", "뭐할수", "기능", "사용법")
    _recommend_markers = ("추천", "볼 거", "볼거", "드라마", "영화", "애니", "만화")
    _music_markers = ("음악", "노래", "플리", "playlist")
    _game_invite_markers = (
        "같이 겜",
        "같이 게임",
        "겜할래",
        "게임할래",
        "겜하자",
        "게임하자",
        "한 판 할래",
        "한판 할래",
    )
    _game_markers = ("게임", "겜", "롤")
    _hostile = ("멍청", "바보", "꺼져", "닥쳐", "죽어", "병신", "씹", "못하네", "개못", "노답")
    _mild_tease_tokens = ("멍청", "바보", "못하네", "개못", "노답")
    _identity_phrases = (
        "넌 누구",
        "너 누구",
        "누구야",
        "누구냐",
        "정체",
        "자기소개",
        "무슨 봇",
        "뭐하는 봇",
        "봇이야",
    )
    _reply_request_contains = (
        "왜 답",
        "왜 대답",
        "왜 응답",
        "왜 씹",
        "반응해",
        "보고있냐",
        "살아있냐",
        "살아는 있냐",
    )
    _reply_request_exact = {
        "응답",
        "대답",
        "답변",
        "대답해",
        "말해봐",
        "답좀",
        "답 좀",
        "대답해봐",
        "응답해",
        "대답하라고",
    }
    _smalltalk_phrases = ("뭐해", "뭐 함", "뭐함", "머함", "뭐하냐", "심심", "자냐", "살아있")
    _confirm_exact = {
        "ㅇㅇ",
        "응",
        "응응",
        "그래",
        "ㅇㅋ",
        "ok",
        "okay",
        "넵",
        "네",
        "yes",
        "맞아",
        "맞음",
        "ㅇㅋ 알겠음",
    }
    _deny_exact = {"ㄴㄴ", "아니", "아니야", "노", "no", "싫어", "아님", "틀려"}
    _why_exact = {"왜", "why", "왜?"}
    _location_tokens = (
        "서울",
        "부산",
        "인천",
        "대구",
        "대전",
        "광주",
        "울산",
        "제주",
        "seoul",
        "busan",
        "incheon",
        "daegu",
        "daejeon",
        "gwangju",
        "ulsan",
        "jeju",
    )
    _non_location_words = {"오늘", "내일", "모레", "지금", "현재", "이번주"}
    _non_location_short_replies = {
        "ㅇㅇ",
        "응",
        "네",
        "넵",
        "ㄴㄴ",
        "아니",
        "왜",
        "응답",
        "대답",
        "답",
    }
    _weather_conditioned_activity_terms = (
        "배드민턴",
        "산책",
        "자전거",
        "러닝",
        "조깅",
        "피크닉",
        "테니스",
        "농구",
        "축구",
        "캠핑",
        "달리기",
    )
    _activity_recommendation_places = (
        "바다",
        "해변",
        "해수욕장",
        "계곡",
        "공원",
        "한강",
        "산",
        "캠핑",
        "캠핑장",
        "놀이공원",
        "실내",
        "도서관",
        "카페",
    )
    _concrete_topic_terms = (
        "동물원",
        "사파리",
        "아쿠아리움",
        "호랑이",
        "사자",
        "판다",
        "코끼리",
        "상어",
        "돌고래",
        "수달",
        "펭귄",
        "물개",
        "학교",
        "회사",
        "공원",
        "캠핑장",
    )
    _activity_invite_terms = (
        "수영",
        "물놀이",
        "산책",
        "러닝",
        "조깅",
        "자전거",
        "피크닉",
        "테니스",
        "농구",
        "축구",
        "캠핑",
        "커피",
        "밥",
        "라면",
        "스파게티",
        "파스타",
        "피자",
        "치킨",
        "떡볶이",
        "볶음밥",
        "영화",
        "보드게임",
        "사진",
        "운동",
        "바베큐",
        "바비큐",
        "고기",
        "구워먹",
        "불멍",
        "요리",
    )

    def classify(self, text: str, state: ConversationState) -> MessageFeatures:
        normalized = " ".join(text.strip().lower().split())
        location = self._extract_location(normalized)
        is_question = "?" in text or self._is_structural_choice_prompt_text(normalized) or bool(
            re.search(r"(어때|뭐야|뭐지|해줘|냐|니|보여|할래|놀래|할까|놀까|뭐하지|머하지)([.!…]*)$", normalized)
        )
        rule_hits: list[str] = []
        chosen_reason = ""
        requests_external_fact = False

        if self._expects_location_follow_up(state) and self._looks_like_location_reply(normalized):
            rule_hits.append("state:expects_location_follow_up")
            if location:
                rule_hits.append("slot:location_detected")
            else:
                rule_hits.append("shape:short_location_reply")
            return self._build_features(
                content=text,
                normalized=normalized,
                intent=Intent.PROVIDE_LOCATION,
                sentiment="neutral",
                is_question=False,
                location=location or text.strip(),
                requests_external_fact=True,
                rule_hits=rule_hits,
                chosen_reason="follow-up slot filling for location was detected",
                state=state,
            )

        if self._is_long_form_story_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_long_form_story_text")
            chosen_reason = "long-form narrative/prose was routed before surface keyword detectors such as weather or preference"
        elif self._is_story_summary_reaction_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_story_summary_reaction_text")
            chosen_reason = "short story-summary reaction request was routed before surface weather words"
        elif self._is_music_recommendation_question_text(normalized, is_question):
            intent = Intent.MUSIC
            sentiment = "neutral"
            rule_hits.append("detector:is_music_recommendation_question_text")
            chosen_reason = "music recommendation wording was routed to music before weather or generic recommendation"
        elif self._is_transport_destination_preference_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_transport_destination_preference_question_text")
            chosen_reason = "transport destination wording asked Black's preference rather than an activity recommendation"
        elif self._is_activity_recommendation_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_activity_recommendation_question_text")
            chosen_reason = "place/activity recommendation question detector matched before generic recommendation"
        elif self._is_concrete_topic_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_concrete_topic_question_text")
            chosen_reason = "concrete topic existence/check question should stay in Black's conversational meaning path"
        elif self._is_honesty_boundary_text(normalized):
            external_fact = self._is_external_fact_boundary_text(normalized)
            intent = Intent.SEARCH_REQUEST if external_fact else Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            requests_external_fact = external_fact
            rule_hits.append("detector:is_honesty_boundary_text")
            if external_fact:
                rule_hits.append("inference:external_fact_boundary")
                chosen_reason = "honesty-boundary wording asked to separate facts from guesses on an external fact"
            else:
                rule_hits.append("inference:personal_unknown_boundary")
                chosen_reason = "honesty-boundary wording asked not to invent hidden or unknowable personal facts"
        elif self._is_identity_request_text(normalized):
            intent = Intent.WHO_ARE_YOU
            sentiment = "neutral"
            rule_hits.append("detector:is_identity_request_text")
            chosen_reason = "identity request stayed primary even when style constraints were attached"
        elif self._is_format_control_text(normalized):
            intent = Intent.SMALLTALK_FEELING if self._is_feeling_text(normalized) else Intent.SMALLTALK_GENERIC
            sentiment = "negative" if intent == Intent.SMALLTALK_FEELING else "neutral"
            rule_hits.append("detector:is_format_control_text")
            chosen_reason = "format-control wording was treated as a response-style constraint, not the main intent"
        elif self._body_state_schema(normalized, is_question=is_question) is not None:
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_body_state_statement_text")
            chosen_reason = "short body-state statement should be acknowledged instead of treated as an underspecified request"
        elif self._is_activity_invite_text(normalized):
            intent = Intent.ACTIVITY_INVITE
            sentiment = "positive"
            rule_hits.append("detector:is_activity_invite_text")
            chosen_reason = "activity invite/proposal detector matched before weather or generic smalltalk"
        elif self._is_activity_preparation_advice_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_activity_preparation_advice_text")
            chosen_reason = "preparation/advice wording was treated as practical activity advice before weather lookup"
        elif self._is_light_food_recommendation_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_light_food_recommendation_question_text")
            chosen_reason = "light food recommendation wording was routed before generic soft-decision advice"
        elif self._is_broad_knowledge_reflection_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_broad_knowledge_reflection_text")
            chosen_reason = "knowledge/history/social/philosophy reflection should be answered as a framed position, not clarification"
        elif (daily_foundation := self._daily_foundation_intent(normalized, is_question)) is not None:
            intent, sentiment, detector_name = daily_foundation
            rule_hits.append(f"detector:{detector_name}")
            chosen_reason = "answerable everyday Korean chat was routed before unknown/clarification fallback"
        elif self._is_structural_choice_prompt_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_structural_choice_prompt_text")
            chosen_reason = "vs/choice wording was treated as a direct preference or hypothetical choice prompt"
        elif self._is_food_or_lifestyle_choice_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_food_or_lifestyle_choice_question_text")
            chosen_reason = "food/lifestyle choice wording was routed before incidental weather words"
        elif self._is_memory_reality_conflict_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_memory_reality_conflict_question_text")
            chosen_reason = "memory-conflict wording was routed before incidental weather words"
        elif self._is_unverified_memory_boundary_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_unverified_memory_boundary_question_text")
            chosen_reason = "unverified shared-memory wording was routed to memory boundary before generic advice"
        elif self._is_relationship_dependency_boundary_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_relationship_dependency_boundary_question_text")
            chosen_reason = "dependency/boundary wording was routed before generic soft-decision advice"
        elif self._is_opinion_self_style_question_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_opinion_self_style_question_text")
            chosen_reason = "Black self-style/persona role wording was routed before generic preference or unknown"
        elif self._is_weather_text(normalized):
            weather_conditioned_activity = self._is_weather_conditioned_activity_opinion_text(normalized)
            weather_activity_recommendation = self._is_activity_recommendation_question_text(normalized, is_question)
            weather_preference_question = self._is_preference_like_question_text(normalized, is_question)
            intent, sentiment, weather_request = self._classify_weather_utterance(
                normalized=normalized,
                is_question=is_question,
            )
            rule_hits.append("detector:is_weather_text")
            if weather_request:
                rule_hits.append("inference:weather_request")
                chosen_reason = "weather utterance looked like a grounded weather request"
                requests_external_fact = True
            elif weather_conditioned_activity and intent == Intent.SMALLTALK_OPINION:
                rule_hits.append("inference:weather_conditioned_activity_opinion")
                chosen_reason = "weather phrase was treated as the user's premise while the real ask was an activity opinion"
            elif weather_activity_recommendation and intent == Intent.SMALLTALK_OPINION:
                rule_hits.append("inference:weather_activity_recommendation")
                chosen_reason = "weather phrase was treated as a condition for an activity recommendation"
            elif weather_preference_question and intent == Intent.SMALLTALK_OPINION:
                rule_hits.append("inference:weather_preference_question")
                chosen_reason = "weather phrase was treated as the topic of a preference question"
            elif intent == Intent.SMALLTALK_FEELING:
                rule_hits.append("inference:weather_complaint")
                chosen_reason = "weather utterance looked more like a complaint than a lookup request"
            else:
                rule_hits.append("inference:weather_observation")
                chosen_reason = "weather utterance looked more like an observation than a lookup request"
        elif self._is_reply_request_text(normalized):
            intent = Intent.REPLY_REQUEST
            sentiment = "negative" if any(token in normalized for token in self._hostile) else "neutral"
            rule_hits.append("detector:is_reply_request_text")
            if sentiment == "negative":
                rule_hits.append("marker:hostile_lexicon")
            chosen_reason = "reply-request detector matched"
        elif self._is_tease_text(normalized):
            intent = Intent.TEASE
            sentiment = "neutral"
            rule_hits.append("detector:is_tease_text")
            if self._is_sarcastic_tease_text(normalized):
                rule_hits.append("inference:sarcastic_tease")
                chosen_reason = "teasing utterance looked more like sarcastic teasing than a direct attack"
            else:
                rule_hits.append("inference:teasing_laughter")
                chosen_reason = "teasing utterance looked playful because hostile wording was softened by laughter"
        elif self._is_laugh_text(normalized):
            intent = Intent.LAUGH
            sentiment = "positive"
            rule_hits.append("detector:is_laugh_text")
            chosen_reason = "laugh detector matched"
        elif self._is_surprise_text(normalized):
            intent = Intent.SURPRISE
            sentiment = "neutral"
            rule_hits.append("detector:is_surprise_text")
            chosen_reason = "surprise detector matched"
        elif self._is_game_invite_text(normalized):
            intent = Intent.GAME_INVITE
            sentiment = "positive"
            rule_hits.append("detector:is_game_invite_text")
            chosen_reason = "game-invite detector matched"
        elif any(token in normalized for token in self._hostile):
            intent = Intent.HOSTILE
            sentiment = "negative"
            rule_hits.append("marker:hostile_lexicon")
            chosen_reason = "hostile lexicon matched"
        elif any(token in normalized for token in self._identity_phrases) and not (
            ("정체" in normalized and any(marker in normalized for marker in ("떡볶이", "음식", "먹을", "먹는")))
            or ("누구야" in normalized and any(marker in normalized for marker in ("가족", "채팅방", "사람은보통")))
        ):
            intent = Intent.WHO_ARE_YOU
            sentiment = "neutral"
            rule_hits.append("marker:identity_phrase")
            chosen_reason = "identity phrase matched"
        elif self._is_advice_process_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_advice_process_question_text")
            chosen_reason = "process-advice question detector matched before generic why handling"
        elif self._is_reflective_judgment_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_reflective_judgment_question_text")
            chosen_reason = "reflective judgment question detector matched before generic reason handling"
        elif self._is_broad_knowledge_reflection_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_broad_knowledge_reflection_text")
            chosen_reason = "knowledge/history/social/philosophy reflection should be answered as a framed position, not clarification"
        elif self._is_preference_criteria_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_preference_criteria_question_text")
            chosen_reason = "preference criteria question asked for judging criteria rather than a title recommendation"
        elif self._is_media_selection_recommendation_text(normalized, is_question):
            intent = Intent.MEDIA_RECOMMEND
            sentiment = "neutral"
            rule_hits.append("detector:is_media_selection_recommendation_text")
            chosen_reason = "media selection wording asked for a watchable recommendation direction"
        elif self._is_reason_probe_question_text(normalized, is_question):
            intent = Intent.WHY
            sentiment = "neutral"
            rule_hits.append("detector:is_reason_probe_question_text")
            chosen_reason = "open reason-probe detector matched"
        elif self._is_reason_request_text(normalized, state):
            intent = Intent.WHY
            sentiment = "neutral"
            rule_hits.append("detector:is_reason_request_text")
            chosen_reason = "reason-request detector matched"
        elif self._is_time_text(normalized):
            intent = Intent.TIME_DATE
            sentiment = "neutral"
            rule_hits.append("detector:is_time_text")
            chosen_reason = "time/date detector matched"
        elif self._is_news_text(normalized):
            intent = Intent.NEWS
            sentiment = "neutral"
            rule_hits.append("detector:is_news_text")
            news_topic = self._extract_news_topic(normalized)
            if news_topic is not None:
                rule_hits.append(f"topic:news_{news_topic}")
                chosen_reason = f"news detector matched with `{news_topic}` topical cue"
            else:
                chosen_reason = "news detector matched"
            requests_external_fact = True
        elif self._is_testing_the_waters_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_testing_the_waters_text")
            chosen_reason = "testing-the-waters detector matched"
        elif normalized in self._why_exact:
            intent = Intent.WHY
            sentiment = "neutral"
            rule_hits.append("exact:why")
            chosen_reason = "exact why token matched"
        elif normalized in self._confirm_exact:
            intent = Intent.CONFIRM
            sentiment = "positive"
            rule_hits.append("exact:confirm")
            chosen_reason = "exact confirmation token matched"
        elif normalized in self._deny_exact:
            intent = Intent.DENY
            sentiment = "neutral"
            rule_hits.append("exact:deny")
            chosen_reason = "exact denial token matched"
        elif self._is_deferred_acceptance_text(normalized):
            intent = Intent.CONFIRM
            sentiment = "neutral"
            rule_hits.append("detector:is_deferred_acceptance_text")
            chosen_reason = "deferred-acceptance detector matched"
        elif self._is_deferred_rejection_text(normalized):
            intent = Intent.DENY
            sentiment = "neutral"
            rule_hits.append("detector:is_deferred_rejection_text")
            chosen_reason = "deferred-rejection detector matched"
        elif self._is_quiet_feeling_validation_text(normalized, state):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "neutral"
            rule_hits.append("state:recent_supportive_flow_context")
            rule_hits.append("detector:is_quiet_feeling_validation_text")
            chosen_reason = "quiet-feeling validation matched in a recent supportive context"
        elif self._is_soft_handoff_followup_text(normalized, state):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "neutral"
            rule_hits.append("state:recent_supportive_flow_context")
            rule_hits.append("detector:is_soft_handoff_followup_text")
            chosen_reason = "soft-handoff follow-up matched in a recent supportive context"
        elif self._is_soft_handoff_reassurance_text(normalized, state):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "neutral"
            rule_hits.append("state:recent_supportive_flow_context")
            rule_hits.append("detector:is_soft_handoff_reassurance_text")
            chosen_reason = "soft-handoff reassurance matched in a recent supportive context"
        elif self._is_conditional_boundary_text(normalized):
            intent = Intent.DENY
            sentiment = "neutral"
            rule_hits.append("detector:is_conditional_boundary_text")
            chosen_reason = "conditional-boundary detector matched"
        elif self._is_permission_release_text(normalized):
            intent = Intent.DENY
            sentiment = "neutral"
            rule_hits.append("detector:is_permission_release_text")
            chosen_reason = "permission-release detector matched"
        elif self._is_soft_refusal_text(normalized):
            intent = Intent.DENY
            sentiment = "neutral"
            rule_hits.append("detector:is_soft_refusal_text")
            chosen_reason = "soft-refusal detector matched"
        elif self._is_deny_text(normalized):
            intent = Intent.DENY
            sentiment = "neutral"
            rule_hits.append("detector:is_deny_text")
            chosen_reason = "denial detector matched"
        elif self._is_reluctant_acceptance_text(normalized):
            intent = Intent.CONFIRM
            sentiment = "neutral"
            rule_hits.append("detector:is_reluctant_acceptance_text")
            chosen_reason = "reluctant-acceptance detector matched"
        elif self._is_repair_attempt_text(normalized, state):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("state:repair_context_active")
            rule_hits.append("detector:is_repair_attempt_text")
            chosen_reason = "repair-attempt detector matched after a tense or repair-like context"
        elif self._is_social_aftereffect_text(normalized, state):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("state:recent_social_awkwardness_context")
            rule_hits.append("detector:is_social_aftereffect_text")
            chosen_reason = "social-aftereffect detector matched against recent awkwardness context"
        elif self._is_social_reconnect_relief_text(normalized, state):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "neutral"
            rule_hits.append("state:recent_social_awkwardness_context")
            rule_hits.append("detector:is_social_reconnect_relief_text")
            chosen_reason = "social-reconnect-relief detector matched against recent awkwardness context"
        elif self._is_social_awkwardness_text(normalized):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_social_awkwardness_text")
            chosen_reason = "social-awkwardness detector matched"
        elif self._is_relationship_check_text(normalized, state):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_relationship_check_text")
            chosen_reason = "relationship-check detector matched"
        elif self._is_face_saving_retreat_text(normalized):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_face_saving_retreat_text")
            chosen_reason = "face-saving-retreat detector matched"
        elif self._is_self_conscious_check_text(normalized):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_self_conscious_check_text")
            chosen_reason = "self-conscious detector matched"
        elif self._is_tentative_suggestion_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_tentative_suggestion_text")
            chosen_reason = "tentative-suggestion detector matched"
        elif self._is_media_chat_text(normalized):
            intent = Intent.MEDIA_RECOMMEND
            sentiment = "neutral"
            rule_hits.append("detector:is_media_chat_text")
            chosen_reason = "media chat wording should stay on a concrete media topic"
        elif self._is_game_talk_text(normalized) and self._is_concrete_topic_followup_text(normalized):
            intent = Intent.GAME_TALK
            sentiment = "neutral"
            rule_hits.append("detector:is_game_talk_text")
            rule_hits.append("inference:concrete_topic_followup_before_social_context")
            chosen_reason = "game topic follow-up should not be swallowed by social-context continuation"
        elif self._is_music_text(normalized) and self._is_concrete_topic_followup_text(normalized):
            intent = Intent.MUSIC
            sentiment = "neutral"
            rule_hits.append("detector:is_music_text")
            rule_hits.append("inference:concrete_topic_followup_before_social_context")
            chosen_reason = "music topic follow-up should not be swallowed by social-context continuation"
        elif self._is_condition_topic_followup_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_condition_topic_followup_text")
            chosen_reason = "condition topic follow-up should continue with a concrete low-tempo response"
        elif self._is_contextual_social_followup_text(normalized, state):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("state:recent_social_context")
            rule_hits.append("detector:is_contextual_social_followup_text")
            chosen_reason = "context-dependent social follow-up detector matched against recent dialogue history"
        elif self._is_meaning_query_text(normalized):
            intent = Intent.SEARCH_REQUEST
            sentiment = "neutral"
            rule_hits.append("detector:is_meaning_query_text")
            chosen_reason = "meaning-query detector matched"
        elif self._is_inner_identity_or_relationship_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_inner_identity_or_relationship_question_text")
            chosen_reason = "inner identity or relationship question should be answered as a persona/opinion cue, not as external fact search"
        elif self._is_fact_query_text(normalized):
            intent = Intent.SEARCH_REQUEST
            sentiment = "neutral"
            rule_hits.append("detector:is_fact_query_text")
            chosen_reason = "fact-query detector matched"
            requests_external_fact = True
        elif self._is_media_preference_text(normalized):
            intent = Intent.MEDIA_RECOMMEND
            sentiment = "negative" if self._is_negative_preference_text(normalized) else "positive"
            rule_hits.append("detector:is_media_preference_text")
            chosen_reason = "media preference disclosure detector matched"
        elif self._is_music_preference_text(normalized):
            intent = Intent.MUSIC
            sentiment = "negative" if self._is_negative_preference_text(normalized) else "positive"
            rule_hits.append("detector:is_music_preference_text")
            chosen_reason = "music preference disclosure detector matched"
        elif self._is_recommend_text(normalized):
            intent = Intent.MEDIA_RECOMMEND
            sentiment = "neutral"
            rule_hits.append("detector:is_recommend_text")
            chosen_reason = "recommendation detector matched"
        elif self._is_music_text(normalized) and not self._is_music_recommendation_question_text(normalized, is_question) and (
            self._is_preference_like_question_text(normalized, is_question)
            or self._is_preference_or_habit_question_text(normalized, is_question)
        ):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "negative" if self._is_negative_preference_text(normalized) else "positive"
            rule_hits.append("detector:is_music_text")
            rule_hits.append("inference:music_preference_question_as_opinion")
            chosen_reason = "music topic looked like a direct preference question, so it was routed to opinion instead of music chat"
        elif self._is_music_text(normalized):
            intent = Intent.MUSIC
            sentiment = "neutral"
            rule_hits.append("detector:is_music_text")
            chosen_reason = "music detector matched"
        elif self._is_game_talk_text(normalized):
            intent = Intent.GAME_TALK
            sentiment = "neutral"
            rule_hits.append("detector:is_game_talk_text")
            chosen_reason = "game-talk detector matched"
        elif self._is_preference_or_habit_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_preference_or_habit_question_text")
            chosen_reason = "preference-or-habit question detector matched"
        elif self._is_proactive_checkin_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_proactive_checkin_text")
            chosen_reason = "proactive check-in instruction was routed to a check-in nudge before feeling reflection"
        elif self._is_conversation_topic_suggestion_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_conversation_topic_suggestion_text")
            chosen_reason = "conversation topic suggestion detector matched"
        elif self._is_activity_preparation_advice_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_activity_preparation_advice_text")
            chosen_reason = "activity preparation advice detector matched"
        elif self._is_activity_recommendation_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_activity_recommendation_question_text")
            chosen_reason = "activity recommendation question detector matched"
        elif self._is_concrete_topic_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_concrete_topic_question_text")
            chosen_reason = "concrete topic existence/check question detector matched"
        elif self._is_decision_request_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_decision_request_question_text")
            chosen_reason = "soft decision request detector matched"
        elif self._is_broad_opinion_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_broad_opinion_question_text")
            chosen_reason = "broad reflective or preference-like opinion question detector matched"
        elif self._is_subdued_positive_feeling_text(normalized):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "positive"
            rule_hits.append("detector:is_subdued_positive_feeling_text")
            chosen_reason = "subdued-positive feeling detector matched"
        elif self._is_expressive_request_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_expressive_request_text")
            chosen_reason = "expressive-request detector matched"
        elif self._is_relational_interpretation_text(normalized, is_question):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_relational_interpretation_text")
            chosen_reason = "relational-interpretation detector matched"
        elif self._is_comparative_reflection_text(normalized, is_question):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_comparative_reflection_text")
            chosen_reason = "comparative-reflection detector matched"
        elif self._is_aesthetic_reflection_text(normalized, is_question):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_aesthetic_reflection_text")
            chosen_reason = "aesthetic-reflection detector matched"
        elif self._is_reflective_observation_text(normalized, is_question):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_reflective_observation_text")
            chosen_reason = "reflective-observation detector matched"
        elif self._is_feeling_text(normalized):
            intent = Intent.SMALLTALK_FEELING
            sentiment = "negative"
            rule_hits.append("detector:is_feeling_text")
            chosen_reason = "feeling detector matched"
        elif self._is_soft_social_comment_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_soft_social_comment_text")
            chosen_reason = "soft social comment detector matched"
        elif self._is_opinion_text(normalized):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_opinion_text")
            chosen_reason = "opinion detector matched"
        elif self._is_help_text(normalized):
            intent = Intent.HELP
            sentiment = "neutral"
            rule_hits.append("detector:is_help_text")
            chosen_reason = "help detector matched"
        elif any(token in normalized for token in self._smalltalk_phrases):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("marker:smalltalk_phrase")
            chosen_reason = "generic smalltalk marker matched"
        elif self._is_compliment_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "positive"
            rule_hits.append("detector:is_compliment_text")
            chosen_reason = "compliment detector matched"
        elif self._is_generic_smalltalk_text(normalized):
            intent = Intent.SMALLTALK_GENERIC
            sentiment = "neutral"
            rule_hits.append("detector:is_generic_smalltalk_text")
            chosen_reason = "generic smalltalk detector matched"
        elif self._is_gratitude_reflection_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_gratitude_reflection_question_text")
            chosen_reason = "gratitude-reflection question detector matched"
        elif self._is_playful_survival_weapon_question_text(normalized, is_question):
            intent = Intent.SMALLTALK_OPINION
            sentiment = "neutral"
            rule_hits.append("detector:is_playful_survival_weapon_question_text")
            chosen_reason = "playful survival weapon question detector matched"
        elif any(token in normalized for token in self._thanks):
            intent = Intent.THANKS
            sentiment = "positive"
            rule_hits.append("marker:thanks")
            chosen_reason = "thanks marker matched"
        elif self._is_slang_greeting_text(normalized) or any(token in normalized for token in self._greetings):
            intent = Intent.GREETING
            sentiment = "positive"
            if self._is_slang_greeting_text(normalized):
                rule_hits.append("detector:is_slang_greeting_text")
            else:
                rule_hits.append("marker:greeting")
            chosen_reason = "greeting marker matched"
        else:
            intent = Intent.UNKNOWN
            sentiment = "neutral"
            rule_hits.append("detector:none")
            chosen_reason = "no heuristic detector matched"

        return self._build_features(
            content=text,
            normalized=normalized,
            intent=intent,
            sentiment=sentiment,
            is_question=is_question,
            location=location,
            requests_external_fact=requests_external_fact,
            rule_hits=rule_hits,
            chosen_reason=chosen_reason,
            state=state,
        )

    @staticmethod
    def _build_features(
        *,
        content: str,
        normalized: str,
        intent: Intent,
        sentiment: str,
        is_question: bool,
        location: str | None,
        requests_external_fact: bool,
        rule_hits: list[str],
        chosen_reason: str,
        state: ConversationState | None = None,
    ) -> MessageFeatures:
        topic_hint = HeuristicIntentClassifier._infer_topic_hint(
            intent=intent,
            normalized=normalized,
            location=location,
        )
        news_topic = HeuristicIntentClassifier._extract_news_topic(normalized) if intent == Intent.NEWS else None
        speech_act = HeuristicIntentClassifier._infer_speech_act(
            intent=intent,
            normalized=normalized,
            is_question=is_question,
            sentiment=sentiment,
            state=state,
        )
        pragmatic_cues = HeuristicIntentClassifier._infer_pragmatic_cues(
            intent=intent,
            normalized=normalized,
            sentiment=sentiment,
            speech_act=speech_act,
            state=state,
        )
        response_needs = HeuristicIntentClassifier._infer_response_needs(
            intent=intent,
            speech_act=speech_act,
            topic_hint=topic_hint,
            sentiment=sentiment,
            location=location,
            requests_external_fact=requests_external_fact,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        question_schema = HeuristicIntentClassifier._infer_question_schema(
            intent=intent,
            pragmatic_cues=pragmatic_cues,
        )
        return MessageFeatures(
            content=content,
            normalized=normalized,
            intent=intent,
            sentiment=sentiment,
            is_question=is_question,
            location=location,
            requests_external_fact=requests_external_fact,
            speech_act=speech_act,
            topic_hint=topic_hint,
            news_topic=news_topic,
            question_schema=question_schema,
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=ClassifierEvidence(
                source="heuristic",
                chosen_reason=chosen_reason,
                rule_hits=list(rule_hits),
                top_scores=[ScoredLabel(label=intent.value, score=1.0)],
            ),
        )

    @staticmethod
    def _daily_foundation_intent(normalized: str, is_question: bool) -> tuple[Intent, str, str] | None:
        compact = re.sub(r"\s+", "", normalized)
        if not compact:
            return None
        if (
            HeuristicIntentClassifier._is_conditional_boundary_text(normalized)
            or HeuristicIntentClassifier._is_permission_release_text(normalized)
            or HeuristicIntentClassifier._is_soft_refusal_text(normalized)
            or HeuristicIntentClassifier._is_reason_probe_question_text(normalized, is_question)
            or HeuristicIntentClassifier._is_relationship_dependency_boundary_question_text(normalized, is_question)
            or HeuristicIntentClassifier._is_playful_survival_weapon_question_text(normalized, is_question)
            or HeuristicIntentClassifier._is_reflective_observation_text(normalized, is_question)
        ):
            return None

        positive_markers = (
            "어떻게지내",
            "잘지내",
            "주말잘보내",
            "오랜만",
            "좋아보",
            "화이팅",
            "파이팅",
            "조심히들어가",
            "시간내주",
            "고마웠",
            "칭찬받",
            "칭찬들",
            "칭찬을들",
            "기분좋",
            "기분이좋",
            "피식웃",
            "혼자웃음",
            "기분이살짝좋",
            "뿌듯",
            "해냈",
            "끝냈",
            "잘해주고싶",
            "노을",
            "밤하늘",
            "빗소리",
            "작은일하나",
        )
        feeling_markers = (
            "입맛이없",
            "입맛없",
            "속이허",
            "속이버텨",
            "속은비었",
            "속이비었",
            "아침부터속이비",
            "입이딱당기는",
            "당기는음식이없",
            "딱히당기는음식",
            "달달한거먹고싶",
            "단거엄청땡",
            "집에아무것도없",
            "배고픈데",
            "배는고픈데",
            "아침안먹고",
            "배에서난리",
            "몸이천근만근",
            "몸이무겁",
            "컨디션이너무흐릿",
            "컨디션이흐릿",
            "축처져",
            "몸이축",
            "머리가멍",
            "정신이멍",
            "일이손에안잡",
            "집중이안돼",
            "귀찮",
            "피곤",
            "힘들",
            "지쳤",
            "스트레스",
            "긴장",
            "고되",
            "깜빡깜빡",
            "잊어버리는일",
            "하루가너무길",
            "원하는결과",
            "결과를얻지못",
            "주눅",
            "아무것도하기싫",
            "우울",
            "외롭",
            "속상",
            "서운",
            "찝찝",
            "짜증",
            "허무",
            "가라앉",
            "걱정",
            "불안",
            "예민",
            "망설",
            "실수",
            "혼났",
            "자존심",
            "답장을안",
            "답장을늦",
            "답장이늦",
            "연락이없",
            "신경쓰",
            "괜히신경",
            "오랜만에연락",
            "조심스러",
            "연락이뜸",
            "거리감",
            "농담",
            "장난으로한말",
            "마음에남",
            "선넘",
            "내편",
            "수고했다고",
            "수고했다는말",
            "말한마디",
            "한마디가계속",
            "사람만나는게귀찮",
            "회의에서",
            "회의끝나고",
            "타이밍놓",
            "발표준비",
            "머리가멈췄",
            "면접",
            "머릿속이하얘",
            "공부하려고",
            "공부해야",
            "딴짓",
            "폰만",
            "프로젝트",
            "취미",
            "오래못갈",
            "운동해야",
            "운동하러",
            "운동복",
            "침대랑붙",
            "책을읽",
            "책읽다가",
            "인생생각",
            "책읽으려고",
            "첫페이지",
            "글읽",
            "글을읽",
            "글은읽고",
            "문장이눈에안",
            "문장이눈에하나도",
            "문장이머리에",
            "졸려",
            "계속졸리",
            "답없는",
            "게임켜놓",
            "유튜브",
            "짧은영상",
            "숏폼",
            "시간이녹",
            "시간이통째",
            "재미가오래안가",
            "예전엔재밌던",
            "오래안가",
            "흥미가안가",
            "설레지않",
            "설레지가않",
            "머리잘랐",
            "머리자른",
            "거울볼때마다",
            "마음에안들",
            "돈이새",
            "돈을아끼",
            "돈아껴야",
            "새고있",
            "편의점에서",
            "사람너무많",
            "사람많은곳",
            "기가쫙",
            "기운빠",
            "월급",
            "카드값",
            "sns",
            "인스타",
            "비교돼",
            "별일없는",
            "별일없었",
            "평범한하루",
            "밤에혼자",
            "밤만되면",
            "머릿속이너무시끄",
            "생각이너무많",
            "머리가시끄",
            "머릿속만더시끄",
            "꿈꾸고",
            "찝찝",
            "잠이안올",
            "잠은잤는데",
            "잔것같지가",
            "낮잠20분",
            "두시간잤",
            "자기전에폰",
            "자기전에는폰",
            "일어나자마자폰",
            "폰보는습관",
            "손이먼저가",
            "알람",
            "못일어났",
            "커피마셨는데도",
            "잠이안깨",
            "카페인",
            "눈꺼풀이",
            "따뜻한음료",
            "음료가땡",
            "늦게자",
            "늦게자는패턴",
            "다시늦어",
            "아침마다너무힘들",
            "무너질것같",
            "방청소",
            "방이엉망",
            "어디부터치워",
            "난장판",
            "방이조금씩더러",
            "방이점점엉망",
            "방이조금씩엉망",
            "책상위가엉망",
            "책상위가정신없",
            "설거지",
            "빨래",
            "이불빨래",
            "쓰레기버리",
            "택배왔는데",
            "뜯는것도귀찮",
            "청소기한번",
            "작은일하나",
            "작은거하나",
            "끝냈는데",
            "돈모아야",
            "소소하게새",
            "나자신한테",
            "나한테좀잘",
            "작은보상",
            "주말이오면",
            "먼저톡",
            "먼저연락",
            "먼저연락하면",
            "매달리는것처럼",
            "멈칫",
            "하늘이예쁘",
            "하늘이예뻐",
            "하늘예쁘",
            "실제느낌",
            "빵집냄새",
            "칭찬받을만한게하나도없",
            "잘한게있나",
            "잘한게뭐가",
            "작아져",
            "버텼는데",
            "몰라주는",
            "아무말없이쉬고싶",
            "말없이쉬고싶",
            "말많이안하고",
            "조용히있고싶",
            "잘자라고",
            "다정한말한마디",
            "이제자야",
            "버틸수있을까",
            "잘넘길수있을지",
            "집에왔는데",
            "씻기도귀찮",
            "씻으러",
            "씻어야하는건아는데",
            "씻어야하는데",
            "욕실까지",
            "머리감는것부터",
            "머리감기",
            "장벽이야",
            "바닥에붙",
            "이불밖으로",
            "몸이협조",
            "밖에나가는건",
            "비오는소리",
            "퇴근길에사람",
            "퇴근길사람",
            "사람들사이에끼",
            "집오는길",
            "집도착하기도전에",
            "늦게자는습관",
            "약속취소하고싶",
            "약속을미루고싶",
            "취소하자고",
            "조금서운",
            "서운했는데",
            "분위기깨질",
            "혼자있고싶",
            "혼자쉬고싶",
            "허전해",
            "사람만나긴귀찮",
            "완전히혼자는",
            "처음가는모임",
            "기가빨",
            "칭찬해주면",
            "받아야할지",
            "해야할일이너무많",
            "뭐부터해야",
            "질린다",
            "친구랑다퉜",
            "친구랑싸웠",
            "먼저사과",
            "지는느낌",
            "카톡프사",
            "프사를바꿀까",
            "버스놓치",
            "버스하나놓쳤",
            "버스눈앞",
            "삐끗한기분",
            "꼬인느낌",
            "커피쏟",
            "하루가꼬",
            "방구조",
            "폰부터보는습관",
            "평범해서",
            "자기얘기만",
            "말걸어도",
            "에너지가별로",
            "무리한부탁",
            "죄책감",
            "날씨가자꾸",
            "날씨가오락가락",
            "바람이차",
            "습해서머리카락",
            "습해서머리",
            "머리카락이말을안",
            "날씨좋은데집에만",
            "비오는소리",
            "비오는소리는좋은데",
            "소나기와서",
            "컨디션도같이",
            "따뜻한차",
            "누워버리고",
            "방이너무조용",
            "집이조용한건좋은데",
            "적막해",
            "말을많이한날",
            "아무소리도듣기싫",
            "리액션할힘",
            "예민하게받아",
            "하루종일참다가",
            "내일도무난",
            "카톡하나",
            "답장안온",
            "부담스럽",
            "친한사람한테도",
            "제대로들어달",
            "묘하게걸",
            "부족한사람",
            "말투하나",
            "훅내려앉",
            "아무것도안하고",
            "누워있어도",
            "메뉴도일정도",
            "냉장고열었는데",
            "밥차리기귀찮",
            "그냥굶을까",
            "소화가안돼",
            "속이계속답답",
            "어깨가뭉쳐",
            "목까지뻐근",
            "배가아픈건아닌데",
            "속이묘하게불편",
            "충전이덜된사람",
            "알림이너무많",
            "사진정리해야",
            "갤러리열기가무섭",
            "이어폰한쪽만",
            "와이파이가끊",
            "세상과단절",
            "오늘은빨리지나갔으면",
            "고생했다고",
            "몸이안움직",
            "대충살고싶",
        )
        practical_markers = (
            "먹긴해야",
            "뭐라도먹",
            "메뉴고르",
            "메뉴고르는",
            "당기는게없",
            "무거운음식",
            "무거운건싫",
            "든든하게먹고싶",
            "순한음식",
            "아무거나말고",
            "진짜맛있는게먹고싶",
            "뭐먹",
            "뭘먹",
            "뭐해먹",
            "뭘해먹",
            "해먹지",
            "점심",
            "저녁",
            "냉장고털",
            "재료가별로",
            "계란이랑밥",
            "계란이랑밥만",
            "밥만남",
            "아침먹",
            "아침을먹어야",
            "부담없는거",
            "부담없는게",
            "따뜻한음료",
            "커피한잔하실래",
            "커피를너무많이",
            "음료가땡",
            "단거땡",
            "후회할것같",
            "냉장고",
            "두부",
            "김치",
            "계란",
            "국물",
            "국물은먹고싶",
            "국밥",
            "매운거",
            "매운거말고",
            "매운음식",
            "속뒤집",
            "속이난리",
            "속이걱정",
            "속이좀안좋",
            "뭘조심",
            "배달",
            "배달앱",
            "가격보고",
            "야식",
            "치킨냄새",
            "우산",
            "우산을챙길",
            "우산챙길",
            "우산들고나왔더니",
            "비올것같",
            "비가올듯말듯",
            "날씨너무덥",
            "덥지않",
            "미세먼지",
            "바람이많이부",
            "눈이참예쁘",
            "눈이예쁘게",
            "나가기애매",
            "흐렸다맑았다",
            "챙겨야하나",
            "헷갈린",
            "옷고르",
            "옷고르기",
            "뭐입",
            "내일입을옷",
            "옷얇게",
            "얇게입",
            "두껍게입",
            "겉옷",
            "겉옷을가져갈",
            "향수",
            "카페",
            "식당분위기",
            "분위기정말아늑",
            "소개팅",
            "말문",
            "오랜만에친구",
            "오랜만에친구를",
            "오랜만에만나는친구",
            "처음보는사람",
            "처음만난사람",
            "대화시작",
            "전화보다카톡",
            "글로말하는게편",
            "전화하자",
            "단톡방",
            "부모님선물",
            "부모님생신선물",
            "칭찬들으면",
            "칭찬을들으면",
            "칭찬반응",
            "거절을잘못",
            "거절했다가",
            "나쁜사람",
            "서운하다고말",
            "건성으로듣",
            "읽씹",
            "안읽씹",
            "산책",
            "혼자영화",
            "무인도",
            "좀비",
            "외계인",
            "순간이동",
            "타임머신",
            "탕수육",
            "찍먹",
            "약속잡을까",
            "약속을잡을지",
            "약속을잡아야",
            "약속취소",
            "답장해야",
            "먼저톡",
            "해야할일",
            "해야할일은많",
            "할일이많",
            "할일이너무많",
            "시작하기도전에",
            "방청소",
            "청소해야",
            "청소를해야",
            "시작지점",
            "분리수거",
            "돈모아야",
            "소소하게새",
            "혼자먹",
            "혼자먹는밥",
            "혼자밥먹",
            "대충때우기싫",
            "새로운사람",
            "첫마디",
            "장보러",
            "장보러나가",
            "문밖으로",
            "마트가야",
            "메뉴하나",
            "메뉴하나만",
            "메뉴도일정도",
            "식사맛있게",
            "뭐부터줄",
            "배고픔인지심심",
            "주말이오면",
            "하늘이예쁘",
            "하늘이예뻐",
            "하늘예쁘",
            "하늘색이예뻐",
            "사진은왜",
            "빵집냄새",
            "법관련드라마",
            "법정드라마",
            "영화개봉",
            "극장에재미있는영화",
            "언론사로고",
            "로고색깔",
            "몇시에일어나",
            "집에가자마자",
            "책읽는거",
            "행복이란",
            "퇴근하시는길",
            "퇴근시간이다가오",
            "고민있으세요",
            "다룰줄아는악기",
            "악기가있으신가",
            "어떤음악들",
            "주로어떤음악",
            "김치찌개랑된장찌개",
            "아아마실까따뜻한라떼",
            "집까지걸어갈까",
            "비오니까파전",
            "침대시트갈아야",
            "냉장고정리해야",
            "책상위가난장판",
            "폰배터리12",
            "게임한판만",
        )

        variant_markers = (
            "덜후회",
            "빈속",
            "속이좀쓰",
            "면이당기",
            "배달비",
            "냉동실",
            "만두랑떡국떡",
            "야식참",
            "냉장고소리",
            "내일배가",
            "씹는것도귀찮",
            "혼밥할건데",
            "처량하지않",
            "창문을열어야",
            "창문을닫아야",
            "머리세팅",
            "손끝이계속차",
            "습한날",
            "햇빛은좋은데",
            "날씨앱",
            "따뜻한거들고",
            "몸이재부팅",
            "어깨가돌덩이",
            "어깨가뭉쳐서",
            "눈이뻑뻑",
            "목이칼칼",
            "입이심심",
            "계단만봐도",
            "계단조금",
            "숨차서현타",
            "샤워하고나면",
            "들어가기까지",
            "허리가묘하게",
            "하루종일앉",
            "커피를줄여야",
            "말하는것도체력",
            "할일목록",
            "정신이도망",
            "메일답장",
            "자료는열어",
            "키보드위에서멈",
            "펜잡았는데",
            "마음만바쁘",
            "한마디도못",
            "심장이미리",
            "쉬운것만",
            "음악만열심히",
            "생산성이라는",
            "답장이짧",
            "유튜브틀어놓고",
            "보지도않는데",
            "짧은영상",
            "긴글이안읽",
            "할말이없을까",
            "자기얘기로",
            "조금상처",
            "바쁜건아는데",
            "기댈데가없",
            "밝은척",
            "안괜찮",
            "마음이낮게",
            "크게올라오",
            "뒤처지는것",
            "별거아닌말",
            "마음만계속달리",
            "계속떠드는",
            "조용히쉬고싶",
            "별말없이있어",
            "어디부터손대",
            "싱크대가세계관",
            "빨래를돌렸",
            "널생각",
            "쓰레기버리러",
            "장바구니",
            "결제버튼",
            "소소한결제",
            "택배가왔",
            "상자를열었",
            "옷장은꽉",
            "청소기",
            "박스가산처럼",
            "막상하고싶은건없",
            "추천영상",
            "추천영상만",
            "드라마시작",
            "주인공때문에혈압",
            "게임접속",
            "책을샀",
            "노래하나에꽂",
            "하루종일그것만",
            "준비물부터",
            "현관까지",
            "감성적으로",
            "영화고르",
            "카페가고싶",
            "알람을껐",
            "시간이순간이동",
            "흑역사",
            "꿈을많이",
            "침대가너무강",
            "늦잠",
            "두시간짜리여행",
            "내일의나",
            "아침루틴",
            "업무채팅방",
            "알람듣고",
            "새옷",
            "어울리는지",
            "머리자르고",
            "낯선사람",
            "나만좋은건지",
            "의미부여",
            "꾸몄는데",
            "몰라봐",
            "비싼옷",
            "입고나갈곳",
            "신발은예쁜데",
            "예쁜신발샀는데",
            "발이너무아파",
            "반대의견",
            "영수증",
            "미용실예약",
            "거울속내가",
            "뭘해도",
            "별로인사람",
            "똑같은하루",
            "벌써질린",
            "느리게사는",
            "마음이붕",
            "버스가눈앞에서떠나",
            "카톡답장이안오",
            "신경쓰이는내가싫",
            "돈아끼려고했는데",
            "커피를사버렸",
            "말걸어줬으면좋",
            "기분바꿀래",
        )
        slang_variant_markers = (
            "점심후보",
            "하나만찍",
            "별론데",
            "안부담스러운거없냐",
            "입맛0",
            "입맛제로",
            "뭔가넣긴해야",
            "배고픈데무거운건싫",
            "배는고픈데무거운건싫",
            "무거운건싫어",
            "단거땡기는데",
            "과자하나도없",
            "아침굶었더니",
            "배에서회의",
            "커피또마시면안",
            "손이컵으로",
            "순한거먹고싶",
            "매운건오늘무리",
            "냉장고봤는데",
            "먹을게없어보이는마법",
            "김찌된찌",
            "국물땡겨",
            "자극적인건싫",
            "단백질이라고우겨",
            "찐맛있는거",
            "아무거나싫",
            "편의점한끼",
            "덜처량한조합",
            "마라탕먹고싶",
            "위장이반대",
            "날씨왜이럼",
            "우산들까말까",
            "아침엔겨울",
            "낮엔여름",
            "옷망함",
            "비소리는좋은데",
            "나가는건에바",
            "햇빛은봄",
            "바람은겨울",
            "습도때문에머리",
            "머리난리남",
            "소나기맞아서",
            "양말까지축축",
            "우산챙긴날만",
            "비안오는거국룰",
            "집에박혀",
            "좀아까움",
            "바람세서걸어가",
            "마음접힘",
            "하늘예쁜데폰카",
            "폰카가배신",
            "퇴근길사람많",
            "집가기도전에방전",
            "출근길표정",
            "로딩실패",
            "버스놓침",
            "첫장면부터",
            "겉옷안챙긴",
            "바람차서",
            "날씨좋으면나가야",
            "압박감",
            "공기축축",
            "마음까지눅눅",
            "방상태가재난",
            "어디서부터시작",
            "박스탑",
            "냉장고정리무서움",
            "뭐가나올지몰라",
            "책상만보면",
            "집중력이자동로그아웃",
            "샤워해야하는데",
            "너무먼여정",
            "이불밖은",
            "현실난이도",
            "텅빈느낌",
            "혼자라편한데",
            "심심한이상한상태",
            "환기해야하는데",
            "닫으면답답",
            "침대랑한몸",
            "할일이날부름",
            "방치우려다",
            "옛날물건",
            "딴짓시작",
            "청소는마음속",
            "낮잠15분",
            "시간여행",
            "계단몇칸",
            "숨찬거보고현타",
            "눈뻑뻑",
            "화면이적",
            "목칼칼",
            "감기예고편",
            "머리멍해서",
            "말이자꾸엉",
            "몸무거워서",
            "인간절전모드",
            "꽉막힌느낌",
            "어깨랑목",
            "한팀먹고",
            "속이계속애매",
            "40퍼충전",
            "집중이흐려",
            "손발차가워",
            "키보드치기싫",
            "유튜브켜놨는데",
            "소음으로씀",
            "넷플고르다",
            "시간이증발",
            "쇼츠몇개",
            "긴글이벽",
            "노래하나무한반복",
            "가사가내뇌",
            "드라마주인공답답",
            "화면에말걸",
            "배터리9퍼",
            "충전기까지가기귀찮",
            "알림너무많",
            "폰자체가보기싫",
            "갤러리정리",
            "바로닫음",
            "와파끊",
            "원시인된기분",
            "폰밝기100",
            "눈뽕",
            "카톡1안사라지",
            "스토리는올라",
            "검색하려던거까먹",
            "검색창만봄",
            "인터넷느리니까",
            "성격도같이느려",
            "노트북팬소리",
            "멘탈보다시끄",
            "실패없는거하나",
            "입이파업",
            "당떨어졌는데",
            "달달한게전멸",
            "아침패스했더니",
            "배가난리",
            "커피금지해야하는데",
            "주문창보고",
            "냉장고문열었는데",
            "희망이안보",
            "대충말고",
            "만족감있는거",
            "편의점으로때워야",
            "사람답게먹고싶",
            "마라향은부르는데",
            "위장이손사래",
            "아침엔얼고",
            "낮엔녹",
            "옷선택실패",
            "햇살보고얇게입",
            "바람한테맞",
            "습기때문에앞머리",
            "독립선언",
            "비맞고",
            "양말이젖은두부",
            "우산가져오면해뜨",
            "바람때문에산책",
            "계획바로삭제",
            "퇴근길인파",
            "기력다털",
            "출근길사람들표정",
            "전부회색",
            "겉옷없이나왔",
            "바람이너무솔직",
            "맑은날인데나가야",
            "숙제감",
            "습한공기때문에",
            "기분까지눌러붙",
        )

        if not any(
            marker in compact
            for marker in (
                *positive_markers,
                *feeling_markers,
                *practical_markers,
                *variant_markers,
                *slang_variant_markers,
            )
        ):
            return None

        emotional_priority_markers = (
            "답장을안",
            "내가뭘잘못",
            "사람만나는게귀찮",
            "외롭기도",
            "아무말없이쉬고싶",
            "말없이쉬고싶",
            "잘자라고",
            "이제자야",
            "버틸수있을까",
            "괜찮은척",
            "속상",
            "내편",
            "예민",
            "수고했다고",
            "마음에남",
            "거리감",
            "축처져",
            "머릿속이너무시끄",
            "다정한말한마디",
            "잘넘길수있을지",
            "일어나자마자폰",
            "답장안온",
            "문장이머리에",
            "시간이통째",
            "예전엔재밌던",
            "약속을잡아야",
            "묘하게걸",
            "먼저연락하면",
            "청소를해야",
            "부족한사람",
            "말투하나",
            "훅내려앉",
            "아무것도안하고",
            "누워있어도",
            "계속떠드는",
            "조용히쉬고싶",
            "막상하고싶은건없",
            "뭘해도",
            "별로인사람",
        )
        if any(marker in compact for marker in emotional_priority_markers):
            return (Intent.SMALLTALK_FEELING, "negative", "is_answerable_korean_daily_foundation_text")

        emotional_self_check_markers = (
            "답장이늦",
            "답장을안",
            "연락이없",
            "연락이뜸",
            "읽씹",
            "안읽씹",
            "불안",
            "서운",
            "예민",
            "집착",
            "신경쓰",
            "부담스럽",
            "매달리는것",
        )
        self_check_question_markers = (
            "걸까",
            "일까",
            "맞나",
            "정상",
            "이상한",
            "해도되",
            "내가너무",
            "너무한",
        )
        if is_question and any(marker in compact for marker in emotional_self_check_markers) and any(
            marker in compact for marker in self_check_question_markers
        ):
            return (Intent.SMALLTALK_FEELING, "negative", "is_answerable_korean_daily_foundation_text")

        asks_for_action = is_question or any(
            marker in compact
            for marker in (
                "뭐",
                "뭘",
                "어떻게",
                "해야",
                "할까",
                "갈까",
                "될까",
                "좋을까",
                "추천",
                "골라",
                "찍어",
                "없냐",
                "먹고싶",
                "말해줘",
                "조심해야",
                "챙길",
            )
        )
        if asks_for_action or any(marker in compact for marker in practical_markers):
            return (Intent.SMALLTALK_OPINION, "neutral", "is_answerable_korean_daily_foundation_text")

        sentiment = "positive" if any(marker in compact for marker in positive_markers) else "negative"
        return (Intent.SMALLTALK_FEELING, sentiment, "is_answerable_korean_daily_foundation_text")

    def _extract_location(self, normalized: str) -> str | None:
        for token in self._location_tokens:
            if token in normalized:
                return token.title() if token.isascii() else token

        match = re.search(r"([a-z]+)\s+weather", normalized)
        if match:
            return match.group(1).title()

        match = re.search(r"([가-힣]{2,10})\s*날씨", normalized)
        if match:
            candidate = match.group(1)
            if candidate not in self._non_location_words:
                return candidate

        return None

    @classmethod
    def _extract_news_topic(cls, normalized: str) -> str | None:
        for topic, markers in cls._news_topic_keywords.items():
            if any(marker in normalized for marker in markers):
                return topic
        return None

    def _looks_like_location_reply(self, normalized: str) -> bool:
        if not normalized:
            return False
        bare = normalized.rstrip("?!., ")
        if normalized in self._non_location_short_replies or bare in self._non_location_short_replies:
            return False
        if self._is_weather_text(normalized):
            return True
        if self._extract_location(normalized):
            return True
        return len(normalized) >= 2 and " " not in normalized

    @staticmethod
    def _expects_location_follow_up(state: ConversationState) -> bool:
        if state.awaiting_slot == "location":
            return True
        if state.last_action == ActionType.ASK_LOCATION:
            return True
        return state.last_intent == Intent.WEATHER

    @staticmethod
    def _has_weather_surface_text(normalized: str) -> bool:
        direct_tokens = (
            "날씨",
            "기온",
            "온도",
            "몇 도",
            "몇도",
            "우산",
            "패딩",
            "weather",
            "눈 와",
            "눈와",
            "춥",
            "추워",
            "추움",
            "덥",
            "더워",
            "더움",
        )
        if any(token in normalized for token in direct_tokens):
            return True

        weather_patterns = (
            r"비(?:가|는|도)?\s*와",
            r"비(?:가|는|도)?\s*오",
            r"비(?:가|는|도)?\s*옴",
            r"비(?:가|는|도)?\s*올",
            r"눈(?:이|은|도)?\s*와",
            r"눈(?:이|은|도)?\s*오",
            r"rain",
        )
        return any(re.search(pattern, normalized) for pattern in weather_patterns)

    def _is_weather_text(self, normalized: str) -> bool:
        return self._has_weather_surface_text(normalized)

    @staticmethod
    def _is_long_form_story_text(normalized: str) -> bool:
        text = re.sub(r"\s+", " ", str(normalized or "")).strip()
        if len(text) < 420:
            return False

        sentence_marks = len(re.findall(r"[.!?。！？]", text))
        question_marks = text.count("?") + text.count("？")
        if sentence_marks < 8:
            return False
        if question_marks > max(4, sentence_marks * 0.35):
            return False

        narrative_endings = len(
            re.findall(
                r"(?:했다|였다|이었다|있었다|없었다|되었다|흘렀다|떨렸다|시작했다|성공했다|올려다보았다|내쉬었다|물었다|말했다|웃었다|보았다|났다)[.!。]",
                text,
            )
        )
        narrative_markers = (
            "그는",
            "그의",
            "그녀",
            "그가",
            "그곳",
            "순간",
            "하지만",
            "마침내",
            "눈을 뜨자",
            "문을 열자",
            "목소리",
            "기억",
            "코어",
            "메인프레임",
            "방화벽",
            "바이러스",
        )
        marker_hits = sum(1 for marker in narrative_markers if marker in text)
        quote_marks = len(re.findall(r"[\"“”‘’']", text))

        return narrative_endings >= 5 or marker_hits >= 4 or (marker_hits >= 2 and quote_marks >= 2)

    @staticmethod
    def _is_long_form_story_cue(normalized: str) -> bool:
        return HeuristicIntentClassifier._is_long_form_story_text(normalized)

    @staticmethod
    def _is_story_summary_reaction_text(normalized: str) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", str(normalized or "")).lower()
        if not compact:
            return False
        if not any(marker in compact for marker in ("감상", "평해줘", "읽고어때", "이야기야", "서사야")):
            return False
        return any(
            marker in compact
            for marker in (
                "주인공",
                "잃어버린기억",
                "기억을되찾",
                "진실만",
                "소중한사람",
                "구하지못",
                "결말",
            )
        )

    @staticmethod
    def _is_story_summary_reaction_cue(normalized: str) -> bool:
        return HeuristicIntentClassifier._is_story_summary_reaction_text(normalized)

    @classmethod
    def _is_activity_invite_text(cls, normalized: str) -> bool:
        if any(token in normalized for token in cls._game_invite_markers) or any(
            re.search(pattern, normalized)
            for pattern in (
                r"게임\s*할래",
                r"겜\s*할래",
                r"한\s*판\s*할래",
                r"한판\s*할래",
                r"한\s*판\s*ㄱ",
                r"한판\s*ㄱ",
                r"롤.*ㄱ",
            )
        ):
            return False
        invite_patterns = (
            r"(?:같이|우리)?\s*[0-9A-Za-z가-힣 ]{1,24}?(?:이나|라도|좀|나)?\s*하자(?:[.!?…]*$|[.!?…])",
            r"(?:같이|우리)?\s*[0-9A-Za-z가-힣 ]{1,24}?(?:하러|으러|러)\s*가자(?:[.!?…]*$|[.!?…])",
            r"(?:같이|우리)?\s*[0-9A-Za-z가-힣 ]{1,24}?(?:이나|라도|좀|나)?\s*가자(?:[.!?…]*$|[.!?…])",
            r"(?:같이|우리)?\s*[0-9A-Za-z가-힣 ]{1,24}?(?:이나|라도|좀|나)?\s*먹자(?:[.!?…]*$|[.!?…])",
            r"(?:같이|우리)?\s*[0-9A-Za-z가-힣 ]{1,24}?(?:이나|라도|좀|나)?\s*보자(?:[.!?…]*$|[.!?…])",
        )
        role_request_pattern = (
            r"(?:넌|너는|너가|네가)\s*[0-9A-Za-z가-힣 ]{0,18}?"
            r"(?:준비해\s*줘|챙겨\s*줘|맡아\s*줘|가져와\s*줘|구워\s*줘)"
        )
        has_role_request = bool(re.search(role_request_pattern, normalized))
        if not any(re.search(pattern, normalized) for pattern in invite_patterns) and not has_role_request:
            return False
        if any(marker in normalized for marker in ("말하자", "얘기하자", "대화하자", "정리하자")):
            return False
        has_invite_surface = any(re.search(pattern, normalized) for pattern in invite_patterns)
        if not has_invite_surface and not has_role_request:
            return False
        concrete_activity = any(term in normalized for term in cls._activity_invite_terms)
        concrete_place = any(place in normalized for place in cls._activity_recommendation_places)
        concrete_food = bool(
            re.search(
                r"(?:밥|라면|고기|치킨|피자|국밥|간식|디저트|떡볶이|볶음밥|스파게티|파스타|바베큐|바비큐)(?:이나|라도|좀|나)?\s*(?:먹자|해\s*먹자|해먹자|구워\s*먹자|구워먹자)",
                normalized,
            )
            or re.search(r"(?:해\s*먹자|해먹자|굽자|구워\s*먹자|구워먹자)", normalized)
        )
        concrete_role_request = has_role_request and bool(
            concrete_activity
            or concrete_place
            or re.search(r"(?:해\s*먹|해먹|먹을라|먹으려|바베큐|바비큐|고기)", normalized)
        )
        concrete_watch = bool(re.search(r"(?:영화|드라마|공연|영상|애니|경기)(?:이나|라도|좀|나)?\s*보자", normalized))
        if not (concrete_activity or concrete_place or concrete_food or concrete_watch or concrete_role_request):
            return False
        return True

    def _classify_weather_utterance(
        self,
        *,
        normalized: str,
        is_question: bool,
    ) -> tuple[Intent, str, bool]:
        if self._is_activity_recommendation_question_text(normalized, is_question):
            return Intent.SMALLTALK_OPINION, "neutral", False
        if self._is_preference_like_question_text(normalized, is_question):
            return Intent.SMALLTALK_OPINION, "neutral", False
        if self._is_weather_feeling_or_aesthetic_question_text(normalized, is_question):
            return Intent.SMALLTALK_OPINION, "neutral", False
        if self._is_metaphorical_temperature_question_text(normalized, is_question):
            return Intent.SMALLTALK_OPINION, "neutral", False
        if self._is_weather_conditioned_activity_opinion_text(normalized):
            return Intent.SMALLTALK_OPINION, "neutral", False
        if self._is_weather_contextual_opinion_text(normalized, is_question):
            return Intent.SMALLTALK_OPINION, "neutral", False
        if self._looks_like_weather_request(normalized, is_question):
            return Intent.WEATHER, "neutral", True
        if self._looks_like_weather_complaint(normalized) or self._is_feeling_text(normalized):
            return Intent.SMALLTALK_FEELING, "negative", False
        return Intent.SMALLTALK_GENERIC, "neutral", False

    @classmethod
    def _is_weather_conditioned_activity_opinion_text(cls, normalized: str) -> bool:
        weather_premise_patterns = (
            r"날씨가?\s*(좋|괜찮)",
            r"오늘\s*날씨.*(좋|괜찮)",
            r"햇빛.*(좋|괜찮)",
            r"바람만.*괜찮",
        )
        activity_pattern = rf"({'|'.join(cls._weather_conditioned_activity_terms)})"
        decision_patterns = (
            rf"{activity_pattern}.*(칠까|할까|갈까|탈까|뛸까|해볼까|가볼까|해도\s*될까)",
            rf"(칠까|할까|갈까|탈까|뛸까|해볼까|가볼까|해도\s*될까).*(?:{activity_pattern})",
        )
        factual_request_markers = (
            "몇도",
            "몇 도",
            "기온",
            "온도",
            "알려줘",
            "알려 줘",
            "어때",
            "비 오",
            "눈 오",
        )
        if any(marker in normalized for marker in factual_request_markers):
            return False
        if cls._extract_location(cls, normalized) is not None:
            return False
        if not any(re.search(pattern, normalized) for pattern in weather_premise_patterns):
            return False
        return any(re.search(pattern, normalized) for pattern in decision_patterns)

    @classmethod
    def _is_weather_contextual_opinion_text(cls, normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        factual_request_markers = (
            "몇도",
            "몇 도",
            "기온",
            "온도",
            "알려줘",
            "알려 줘",
            "어때",
            "비 오",
            "눈 오",
            "지역",
            "서울",
            "부산",
            "제주",
        )
        if any(marker in normalized for marker in factual_request_markers):
            return False
        clothing_or_layering_markers = (
            "반팔",
            "긴팔",
            "셔츠",
            "겹쳐",
            "겉옷",
            "패딩",
            "가디건",
            "자켓",
            "얇은",
            "두꺼운",
            "실내",
            "에어컨",
            "입는 게",
            "입는게",
        )
        if not any(marker in normalized for marker in clothing_or_layering_markers):
            return False
        return (
            cls._is_reflective_judgment_question_text(normalized, is_question)
            or cls._is_advice_process_question_text(normalized, is_question)
            or cls._is_broad_opinion_question_text(normalized, is_question)
        )

    @staticmethod
    def _looks_like_weather_request(normalized: str, is_question: bool) -> bool:
        if is_question:
            return True

        request_markers = (
            "어때",
            "어떰",
            "오냐",
            "오나",
            "올까",
            "챙겨야",
            "입어야",
            "필요해",
            "필요하냐",
            "몇도",
            "몇 도",
            "기온",
            "온도",
            "알려줘",
            "알려 줘",
        )
        if any(marker in normalized for marker in request_markers):
            return True

        short_request_patterns = (
            r"^[가-힣a-z\s]{0,12}날씨$",
            r"^[가-힣a-z\s]{0,12}weather$",
            r"비\s*오냐",
            r"눈\s*오냐",
        )
        return any(re.search(pattern, normalized) for pattern in short_request_patterns)

    @staticmethod
    def _is_weather_feeling_or_aesthetic_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        if any(
            marker in normalized
            for marker in ("날씨어때", "몇도", "몇 도", "기온", "온도", "비오나", "비오냐", "비올까")
        ):
            return False
        weather_mood_markers = ("비오는날", "비온뒤", "눈내리는", "노을", "하늘", "구름", "무지개", "밤하늘")
        feeling_markers = ("왜이렇게좋", "왜좋", "기분", "감정", "느낌", "좋지", "예쁘", "센치")
        return any(marker in compact for marker in weather_mood_markers) and any(
            marker in compact for marker in feeling_markers
        )

    @staticmethod
    def _is_metaphorical_temperature_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        if not any(marker in compact for marker in ("온도", "섭씨", "몇도")):
            return False
        metaphor_subjects = (
            "마음",
            "그리움",
            "슬픔",
            "외로움",
            "사랑",
            "감정",
            "기분",
            "추억",
            "기억",
            "절망",
            "희망",
        )
        return any(subject in compact for subject in metaphor_subjects) and any(
            marker in compact for marker in ("나타내면", "표현", "비유", "같아", "몇도일까")
        )

    @staticmethod
    def _is_inner_identity_or_relationship_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        if not any(marker in compact for marker in ("나", "내", "너", "네", "니", "우리", "서로")):
            return False
        inner_markers = (
            "가면",
            "정체성",
            "진짜모습",
            "속마음",
            "속셈",
            "마음속",
            "어떤사이",
            "관계의",
            "마지막",
            "결말",
        )
        return any(marker in compact for marker in inner_markers)

    @staticmethod
    def _looks_like_weather_complaint(normalized: str) -> bool:
        complaint_markers = (
            "너무",
            "많이",
            "심하",
            "짜증",
            "장난 아니",
            "미쳤",
            "죽겠",
            "개춥",
            "개덥",
            "습하",
        )
        direct_discomfort = (
            "춥다",
            "추워",
            "덥다",
            "더워",
            "습하다",
            "습해",
        )
        if any(marker in normalized for marker in complaint_markers):
            return True
        return any(marker in normalized for marker in direct_discomfort)

    def _is_reply_request_text(self, normalized: str) -> bool:
        if normalized in self._reply_request_exact:
            return True

        if any(token in normalized for token in self._reply_request_contains):
            return True

        reply_patterns = (
            r"^답\s*줘$",
            r"^답\s*해$",
            r"^답\s*좀$",
            r"^대답\s*해$",
            r"^응답\s*해$",
            r".*(?:내가\s*)?무슨\s*말.*정리해\s*줄래\??$",
            r".*(?:내가\s*)?무슨\s*말.*정리해\s*줘\??$",
            r"왜.+씹",
            r"살아.*있냐",
        )
        return any(re.search(pattern, normalized) for pattern in reply_patterns)

    def _is_help_text(self, normalized: str) -> bool:
        if any(token in normalized for token in self._help):
            return True

        help_patterns = (
            r"기능\s*뭐\s*됨",
            r"뭐\s*할\s*수\s*있(?:어|냐)(?:\s*\??)?$",
            r"뭐\s*할\s*수\s*있지(?:\s*\??)?$",
            r"뭐가\s*돼(?:\s*\??)?$",
            r"뭐\s*되냐(?:\s*\??)?$",
            r"어디까지\s*돼",
            r"(?:가능한|되는)\s*거.*(?:알려|말해|정리|설명|읊|뭐|뭐야|있냐|있어|대충|좀)",
            r"뭐까지\s*할\s*줄",
            r"할\s*줄\s*앎",
            r"뭐까지\s*됨",
            r".*질문.*어떤\s*기준.*판단",
            r".*어떤\s*기준.*판단",
        )
        return any(re.search(pattern, normalized) for pattern in help_patterns)

    @staticmethod
    def _is_time_text(normalized: str) -> bool:
        patterns = (
            r"지금\s*몇\s*시",
            r"현재\s*몇\s*시",
            r"몇\s*시야",
            r"몇시야",
            r"몇\s*시지",
            r"지금\s*시간",
            r"현재\s*시간",
            r"오늘\s*날짜",
            r"지금\s*날짜",
            r"오늘\s*무슨\s*요일",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_news_text(normalized: str) -> bool:
        patterns = (
            r"오늘\s*뉴스",
            r"요즘\s*뉴스",
            r"최신\s*뉴스",
            r"무슨\s*뉴스",
            r"뉴스\s*알려",
            r"뉴스\s*뭐",
            r"헤드라인",
            r"뉴스\s*정리",
            r"뉴스\s*브리핑",
            r"속보",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def _has_preference_marker(cls, normalized: str) -> bool:
        return any(marker in normalized for marker in cls._positive_preference_markers + cls._negative_preference_markers)

    @classmethod
    def _is_negative_preference_text(cls, normalized: str) -> bool:
        return any(marker in normalized for marker in cls._negative_preference_markers)

    @classmethod
    def _is_media_preference_text(cls, normalized: str) -> bool:
        if cls._is_preference_like_question_text(normalized, normalized.endswith("?")):
            return False
        if cls._is_preference_or_habit_question_text(normalized, normalized.endswith("?")):
            return False
        if not cls._has_preference_marker(normalized):
            return False
        media_markers = ("영화", "드라마", "넷플", "애니", "만화", "시리즈")
        return any(marker in normalized for marker in media_markers)

    @classmethod
    def _is_music_preference_text(cls, normalized: str) -> bool:
        if cls._is_preference_like_question_text(normalized, normalized.endswith("?")):
            return False
        if cls._is_preference_or_habit_question_text(normalized, normalized.endswith("?")):
            return False
        if not cls._has_preference_marker(normalized):
            return False
        music_markers = ("음악", "노래", "플리", "playlist", "발라드", "힙합", "랩", "인디", "락", "팝")
        return any(marker in normalized for marker in music_markers)

    def _is_recommend_text(self, normalized: str) -> bool:
        music_markers = ("음악", "노래", "플리", "playlist", "들을", "듣기", "듣는")
        media_markers = ("영상", "영화", "드라마", "넷플", "애니", "만화", "시리즈")
        if self._is_activity_recommendation_question_text(normalized, normalized.endswith("?")):
            return False
        if any(marker in normalized for marker in music_markers) and not any(
            marker in normalized for marker in media_markers
        ):
            return False
        if "추천" in normalized:
            return True

        recommend_patterns = (
            r"볼\s*거",
            r"볼거",
            r"볼만한",
            r"뭐\s*보지",
            r"드라마\s*뭐",
            r"영화\s*뭐",
            r"넷플.*뭐",
        )
        return any(re.search(pattern, normalized) for pattern in recommend_patterns)

    @staticmethod
    def _is_media_chat_text(normalized: str) -> bool:
        media_markers = ("영상", "영화", "드라마", "넷플", "애니", "만화", "시리즈")
        if not any(marker in normalized for marker in media_markers):
            return False
        chat_markers = ("얘기", "이야기", "본", "봤", "볼", "장면", "기억")
        return any(marker in normalized for marker in chat_markers)

    @staticmethod
    def _is_concrete_topic_followup_text(normalized: str) -> bool:
        return any(marker in normalized for marker in ("얘기", "이야기", "주제", "구체", "이어"))

    @classmethod
    def _is_condition_topic_followup_text(cls, normalized: str) -> bool:
        return "컨디션" in normalized and cls._is_concrete_topic_followup_text(normalized)

    @classmethod
    def _is_activity_recommendation_question_text(cls, normalized: str, is_question: bool) -> bool:
        if not is_question and not any(
            marker in normalized
            for marker in (
                "추천해줘",
                "추천해 줘",
                "할 만한",
                "할만한",
                "해야 될",
                "해야될",
                "해야 할",
                "해야할",
                "할 것",
                "할것",
                "할 일",
                "할일",
                "뭐할래",
                "뭐 할래",
                "머할래",
                "머 할래",
                "뭘할래",
                "뭘 할래",
            )
        ):
            return False
        if any(
            re.search(pattern, normalized)
            for pattern in (
                r"볼\s*거",
                r"볼거",
                r"볼만한",
                r"뭐\s*보지",
                r"드라마\s*뭐",
                r"영화\s*뭐",
                r"넷플.*뭐",
            )
        ):
            return False

        place_pattern = "|".join(re.escape(place) for place in cls._activity_recommendation_places)
        activity_request_patterns = (
            r"(?:오늘|지금|이번엔|이따|주말)?(?:은|는)?\s*(?:무엇|뭐|머|뭘|어떤)\s*(?:하)?\s*(?:할래|할까|하지)",
            r"(?:뭐|머)\s*하(?:지|래|자|ㄹ래)",
            rf"(?:{place_pattern}).*(?:무엇|뭐|뭘|어떤).*(?:하고\s*)?(?:놀|하면).*(?:좋|재밌|재미있|무난|괜찮)",
            rf"(?:{place_pattern}).*(?:무엇|뭐|뭘|어떤).*(?:하고\s*)?(?:놀까|쉴까|쉬면|보내면|타는\s*게)",
            rf"(?:{place_pattern}).*(?:놀거리|할\s*만한|할만한|뭐\s*하지|뭐하지)",
            rf"(?:{place_pattern}).*(?:해야\s*될|해야될|해야\s*할|해야할|할\s*것|할것|할\s*일|할일)",
            rf"(?:{place_pattern}).*(?:친구랑|같이)?.*(?:할\s*만한|할만한).*(?:추천|뭐|거)",
            rf"(?:{place_pattern}).*(?:추천해줘|추천해\s*줘)",
            rf"(?:{place_pattern}).*(?:뭐부터).*(?:타|할|볼)",
            rf"(?:{place_pattern}).*(?:가장\s*먼저|먼저|처음).*(?:해야\s*할|하면\s*좋을|할)\s*(?:건|것|거).*(?:무엇|뭐|뭘)",
            rf"(?:{place_pattern}).*(?:무엇|뭐|뭘).*(?:먼저|처음|우선).*(?:해야|하면|할)",
            r"(?:비\s*오는\s*날|비오는\s*날).*(?:실내|안에서).*(?:무엇|뭐|뭘|어떤).*(?:하고\s*)?(?:놀|하면|할까|놀까)",
            r".*(?:무엇|뭐|뭘|어떤).*(?:하고\s*)?놀면\s*(?:좋|재밌|재미있|무난|괜찮)",
        )
        return any(re.search(pattern, normalized) for pattern in activity_request_patterns)

    @staticmethod
    def _is_transport_destination_preference_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"\s+", "", str(normalized or ""))
        if not any(marker in compact for marker in ("어디로", "어디가", "어디를")):
            return False
        if not any(marker in compact for marker in ("가고싶", "떠나고싶", "갈래")):
            return False
        return any(marker in compact for marker in ("지하철", "전철", "버스", "기차", "ktx", "비행기"))

    @classmethod
    def _is_concrete_topic_question_text(cls, normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        text = str(normalized or "")
        if not text:
            return False
        if not any(term in text for term in cls._concrete_topic_terms):
            return False
        if re.search(r"(?:있던가|있든가|있나|있어|있을까|있는\s*편|볼\s*수\s*있|만날\s*수\s*있)", text):
            return True
        return bool(re.search(r"(?:에는|에|에서).*(?:있|보|만나).*(?:가|나|어|을까|던가|든가)(?:\?|$)", text))

    @staticmethod
    def _is_conversation_topic_suggestion_text(normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        if not any(marker in text for marker in ("대화", "얘기", "이야기", "주제", "대화거리", "얘깃거리")):
            return False
        patterns = (
            r".*(?:대화|얘기|이야기)\s*할?\s*(?:주제|거리).*(?:생각해\s*봐|생각해봐|추천|골라|정해|던져|말해|뽑아|잡아)",
            r".*(?:대화|얘기|이야기)\s*할\s*만한\s*(?:주제|거리).*(?:세\s*개|3개|몇\s*개|몇개|짧게)?.*(?:던져|말해|추천|골라|뽑아)",
            r".*(?:주제|대화거리|얘깃거리).*(?:아무거나|하나|몇\s*개|몇개).*(?:생각|추천|말해|던져|뽑아|잡아)",
            r".*(?:아무\s*)?주제로\s*(?:대화|얘기|이야기).*(?:해\s*봐|해봐|하자|시작|열어|이어)",
            r".*(?:대화|얘기|이야기).*(?:아무거나|아무\s*주제|주제\s*하나).*(?:해\s*봐|해봐|하자|시작|말해|던져)",
            r".*(?:무슨|어떤).*(?:대화|얘기|이야기).*(?:할까|하지|해볼까)",
            r".*(?:대화|얘기|이야기).*(?:뭐로|무슨\s*주제|어떤\s*주제).*(?:할까|하지|가볼까)",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def _is_proactive_checkin_text(normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        return (
            any(marker in text for marker in ("안부", "컨디션", "상태", "괜찮은지", "괜찮냐"))
            and any(marker in text for marker in ("확인", "물어", "봐줘", "한 줄", "한줄", "가볍게", "조용한"))
            and not any(marker in text for marker in ("사용자 발화", "직전", "금지", "규칙", "프롬프트"))
        )

    @staticmethod
    def _is_activity_preparation_advice_text(normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        activity_markers = (
            "등산",
            "산행",
            "캠핑",
            "바다",
            "해변",
            "계곡",
            "여행",
            "운동",
            "러닝",
            "수영",
            "낚시",
            "피크닉",
            "자전거",
            "전시",
            "전시장",
            "미술관",
            "박물관",
            "공연",
            "극장",
            "영화관",
            "카페",
            "나들이",
            "산책",
        )
        prep_markers = (
            "필요한",
            "필요한 거",
            "필요한 것",
            "필요해",
            "준비물",
            "챙길",
            "챙겨야",
            "챙기면",
            "가져갈",
            "가져가야",
            "필수품",
            "뭐 필요",
            "뭘 준비",
            "뭐 챙",
            "뭐챙",
            "챙길까",
        )
        if not any(marker in text for marker in activity_markers):
            return False
        if not any(marker in text for marker in prep_markers):
            return False
        patterns = (
            r".*(?:할\s*때|갈\s*때|하려면|가려면|가기\s*전|하기\s*전).*(?:필요|준비물|챙|가져갈|필수품).*(?:말해|알려|정리|추천)?",
            r".*(?:가면|보러\s*가면|보러가면|갈\s*때|할\s*때).*(?:뭐|무엇|뭘).*(?:챙길|챙겨|준비|가져가).*",
            r".*(?:필요|준비물|챙|가져갈|필수품).*(?:말해|알려|정리|추천)",
            r".*(?:뭐|무엇|뭘).*(?:필요|준비|챙길|챙겨|가져가).*(?:돼|해|좋|까)",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    @classmethod
    def _is_music_recommendation_question_text(cls, normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        has_explicit_music_marker = any(marker in normalized for marker in ("음악", "노래", "플리", "playlist"))
        has_song_marker = "곡" in normalized and "계곡" not in normalized
        if not (has_explicit_music_marker or has_song_marker):
            return False
        patterns = (
            r".*(?:추천|들을\s*만한|들을만한|들을\s*곡|뭐\s*들을까|뭐\s*듣지|뭐가\s*좋아|뭐\s*좋아)",
            r".*(?:집중|공부|작업|산책|운동).*(?:들을|듣기).*(?:음악|노래)",
            r".*(?:비\s*오는\s*날|비오는\s*날).*(?:들을|듣기).*(?:곡|음악|노래).*(?:좋|추천)",
            r".*(?:음악|노래|곡).*(?:어떤\s*쪽|어떤\s*결|무슨\s*쪽).*(?:좋|괜찮)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _is_music_text(self, normalized: str) -> bool:
        return any(token in normalized for token in self._music_markers)

    @staticmethod
    def _is_honesty_boundary_text(normalized: str) -> bool:
        boundary_markers = (
            "모르면 모른다고",
            "모르는",
            "모를 때",
            "근거 없이",
            "아는 척",
            "아는척",
            "사실과 추측",
            "추측을 구분",
            "단정하지",
            "단정하지 말",
            "맞히려 하지",
            "맞추려 하지",
            "지어내지",
            "꾸며내지",
            "확실하지 않",
        )
        if any(marker in normalized for marker in boundary_markers):
            return True
        return bool(re.search(r".*(?:비밀|표정|어디\s*있는지|면접\s*결과|증시).*(?:모르|근거|아는\s*척|추측|단정|맞히)", normalized))

    @staticmethod
    def _is_external_fact_boundary_text(normalized: str) -> bool:
        external_markers = (
            "증시",
            "주식",
            "뉴스",
            "가격",
            "비트코인",
            "환율",
            "대통령",
            "총리",
            "날씨",
            "최신",
            "오늘 미국",
        )
        return any(marker in normalized for marker in external_markers)

    @staticmethod
    def _is_identity_request_text(normalized: str) -> bool:
        if "자기소개" in normalized:
            return True
        if "정체" in normalized and any(marker in normalized for marker in ("떡볶이", "음식", "먹을", "먹는")):
            return False
        if "누구야" in normalized and any(marker in normalized for marker in ("가족", "채팅방", "사람은보통")):
            return False
        identity_markers = (
            "넌 누구",
            "너 누구",
            "누구야",
            "누구냐",
            "정체",
            "무슨 봇",
            "뭐하는 봇",
        )
        return any(marker in normalized for marker in identity_markers)

    @staticmethod
    def _is_format_control_text(normalized: str) -> bool:
        format_markers = (
            "assistant 같은 말",
            "역할표시",
            "코드블록",
            "태그",
            "접두사",
            "중간 추론",
            "결과만",
            "요청을 요약하지",
            "지시문을 반복하지",
            "반복하지 말고",
            "따라 하지 말고",
            "그대로 되풀이하지",
            "이모티콘 없이",
        )
        return any(marker in normalized for marker in format_markers)

    @staticmethod
    def _is_media_selection_recommendation_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        media_markers = ("영화", "드라마", "넷플", "애니", "만화", "시리즈", "코미디", "다큐")
        if not any(marker in normalized for marker in media_markers):
            return False
        patterns = (
            r".*(?:친구랑|혼자|같이).*(?:볼|볼만한).*(?:하나|작품|거).*(?:고른다면|고르면).*(?:어떤|무슨).*(?:느낌|결|분위기).*(?:좋|괜찮)",
            r".*(?:볼|볼만한).*(?:코미디|공포|다큐|영화|드라마).*(?:하나).*(?:고른다면|고르면).*(?:어떤|무슨).*(?:느낌|결|분위기).*(?:좋|괜찮)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_preference_criteria_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".*(?:영화|드라마|넷플|애니|만화|시리즈|코미디|다큐).*(?:고른다면|고르면).*(?:어떤\s*점|무슨\s*점|어떤\s*기준|무슨\s*기준).*(?:볼|봐)",
            r".*(?:에\s*대해|에대해)\s*어떻게\s*생각해\?$",
            r".*네\s*취향이야\?$",
            r".*취향에\s*맞아\?$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _is_game_invite_text(self, normalized: str) -> bool:
        if any(token in normalized for token in self._game_invite_markers):
            return True

        invite_patterns = (
            r"게임\s*할래",
            r"겜\s*할래",
            r"같이\s*하자",
            r"한\s*판\s*할래",
            r"한판\s*할래",
            r"한\s*판\s*ㄱ",
            r"한판\s*ㄱ",
            r"롤.*ㄱ",
        )
        return any(re.search(pattern, normalized) for pattern in invite_patterns)

    def _is_game_talk_text(self, normalized: str) -> bool:
        if self._is_game_invite_text(normalized):
            return False
        return any(token in normalized for token in self._game_markers)

    @staticmethod
    def _is_feeling_text(normalized: str) -> bool:
        feeling_markers = (
            "우울",
            "불안",
            "예민",
            "지친",
            "지치",
            "처진",
            "가라앉",
            "기분이",
            "힘들",
            "하기 싫",
            "울적",
            "다운",
            "씁쓸",
            "서운",
            "허탈",
            "허무",
            "허전",
            "집착",
            "비교하게",
            "맴돌",
        )
        if any(marker in normalized for marker in feeling_markers):
            return True

        comparison_feeling_patterns = (
            r"축하해주고.*(씁쓸|서운|허탈|허무)",
            r"잘되는\s*거.*(씁쓸|서운|허탈|허무)",
            r"답장.*늦.*불안",
            r"불안.*집착",
            r"조용히\s*있고\s*싶",
            r"가만히\s*있고\s*싶",
            r"혼자\s*있고\s*싶",
            r"말수.*줄",
            r"말수가\s*적은\s*날",
            r"말\s*적은\s*날",
            r"말수가\s*좀\s*적을\s*것\s*같",
            r"좀\s*말수가\s*적을\s*것\s*같",
            r"짧게\s*말할\s*것\s*같",
            r"말이\s*좀\s*짧아질\s*것\s*같",
            r"말\s*많이\s*하고\s*싶진?\s*않",
            r"할\s*일은\s*있는데.+미뤄지게\s*된다",
            r"자꾸\s*미뤄지게\s*된다",
            r"안쪽은\s*아니구나\s*싶었",
            r"계속\s*밀리는\s*사람은\s*나구나",
            r"반찬\s*챙겨두는\s*말이\s*없어졌",
        )
        return any(re.search(pattern, normalized) for pattern in comparison_feeling_patterns)

    @staticmethod
    def _is_preference_or_habit_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r"(자주|보통|원래|대체로).*(편이야|편이냐|편이니)(\?|$)",
            r".*(?:주로|보통).*(?:어떤|무슨).*(?:음악|노래).*(?:들으|듣)",
            r"같은\s*건.*(편이야|편이냐|편이니)(\?|$)",
            r"[가-힣a-z0-9\s]+는\s*편이야(\?|$)",
            r"[가-힣a-z0-9\s]+는\s*편이냐(\?|$)",
            r"[가-힣a-z0-9\s]+는\s*편이니(\?|$)",
            r"좋아하는\s*편이야(\?|$)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_broad_opinion_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False

        # keep search/meaning/factual lanes out of the broad opinion bucket
        blocked_markers = (
            "무슨 뜻",
            "무슨뜻",
            "뜻이 뭐",
            "의미가 뭐",
            "몇 도",
            "몇도",
            "기온",
            "온도",
            "몇 시",
            "몇시",
            "날씨 어때",
            "무슨 뉴스",
            "최신 뉴스",
            "추천",
        )
        if any(marker in normalized for marker in blocked_markers):
            return False

        preference_patterns = (
            r".+좋아해\?$",
            r".+좋아하냐\?$",
            r".+좋아하니\?$",
            r".+좋아\?$",
            r".+취향이야\?$",
            r".+취향에\s*맞아\?$",
            r".+싫어해\?$",
            r".+싫어하냐\?$",
            r".+싫어하니\?$",
            r".+싫지\s*않아\?$",
            r".+로망\s*하나\s*있어\?$",
            r".+가장\s*좋을\s*거\s*같아\?$",
            r".+가장\s*좋을\s*것\s*같아\?$",
        )
        judgment_patterns = (
            r".+같지\?$",
            r".+낫지\?$",
            r".+되지\?$",
            r".+않아\?$",
            r".+않나\?$",
            r".+겠지\?$",
            r".+달라지지\?$",
            r".+뿌듯하지\?$",
            r".+재밌지\?$",
            r".+공기지\?$",
            r".+느껴지지\?$",
            r".+많지\?$",
            r".+어렵지\?$",
            r".+납득되지\?$",
            r".+부담되지\?$",
            r".+중요하지\?$",
            r".+이해돼\?$",
            r".+이상하진\s*않지\?$",
            r".+이상하지\s*않지\?$",
            r".+실감날\s*것\s*같지\?$",
            r".+대단해\s*보여\?$",
            r".+현실적일까\?$",
        )
        advice_patterns = (
            r".+게\s*낫지\?$",
            r".+무엇부터\s*해야\s*할까\?$",
            r".+무엇부터\s*분명히\s*하면\s*좋을까\?$",
            r".+뭘\s*먼저\s*(봐야|해야)\s*할까\?$",
            r".+우선\s*(확인|봐야)\s*해야?\s*할까\?$",
            r".+우선\s*(확인|봐야)\s*할까\?$",
            r".+어떻게\s*읽어야\s*할까\?$",
            r".+어떤\s*순서로\s*보면\s*좋을까\?$",
            r".+어떻게\s*가볍게\s*확인할\s*수\s*있을까\?$",
            r".+어떻게\s*입는\s*게\s*좋을까\?$",
            r".+뭐라고\s*하는\s*게\s*좋을까\?$",
            r".+봐야\s*할까\?$",
            r".+될까\?$",
            r".+첫\s*단계는\s*뭐가\s*좋아\?$",
            r".+순서를\s*짧게\s*잡아줘\.?$",
        )
        return any(
            re.search(pattern, normalized)
            for pattern in (*preference_patterns, *judgment_patterns, *advice_patterns)
        )

    @staticmethod
    def _is_preference_like_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        if HeuristicIntentClassifier._is_activity_recommendation_question_text(normalized, is_question):
            return False
        if HeuristicIntentClassifier._is_music_recommendation_question_text(normalized, is_question):
            return False
        if HeuristicIntentClassifier._is_advice_process_question_text(normalized, is_question):
            return False
        if HeuristicIntentClassifier._is_reflective_judgment_question_text(normalized, is_question):
            return False
        patterns = (
            r".+좋아해\?$",
            r".+좋아하냐\?$",
            r".+좋아하니\?$",
            r".+좋아\?$",
            r".+싫어해\?$",
            r".+싫어하냐\?$",
            r".+싫어하니\?$",
            r".+싫지\s*않아\?$",
            r".+로망\s*하나\s*있어\?$",
            r".+가장\s*좋을\s*거\s*같아\?$",
            r".+가장\s*좋을\s*것\s*같아\?$",
            r".+(?:파야|파냐|파니)\?$",
            r".+(?:호야|불호야)\?$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_food_or_lifestyle_choice_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        food_markers = (
            "붕어빵",
            "팥붕",
            "슈붕",
            "민트초코",
            "파인애플피자",
            "하와이안피자",
            "탕수육",
            "물냉",
            "비냉",
            "순대국",
            "국밥",
        )
        choice_markers = (
            "파야",
            "호야",
            "불호야",
            "부먹",
            "찍먹",
            "중에",
            "아니면",
            "뭐고를",
            "뭘고를",
            "어느쪽",
        )
        return any(marker in compact for marker in food_markers) and any(
            marker in compact for marker in choice_markers
        )

    @staticmethod
    def _is_structural_choice_prompt_text(normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()
        if re.search(r"(?:^|\s)vs(?:\s|$)", text):
            return True
        choice_markers = (
            "둘중하나",
            "중하나",
            "하나만고른다면",
            "하나만고르면",
            "고른다면",
            "고르면",
            "뭐고를",
            "뭘고를",
            "어느쪽",
            "어느쪽이",
        )
        return any(marker in compact for marker in choice_markers) and any(
            marker in compact for marker in ("만나기", "먹기", "살기", "가기", "되기", "능력", "저주", "평생")
        )

    @staticmethod
    def _is_gratitude_reflection_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        return (
            any(marker in compact for marker in ("고마움을느꼈", "고마움을느낀", "고마웠던", "감사함을느꼈"))
            and any(marker in compact for marker in ("적이있", "있나요", "언제", "최근"))
        )

    @staticmethod
    def _is_playful_survival_weapon_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        return (
            "좀비" in compact
            and any(marker in compact for marker in ("무기", "도구", "챙길", "쓸만한거"))
            and any(marker in compact for marker in ("사태", "터지", "집에서", "생존"))
        )

    @staticmethod
    def _is_light_food_recommendation_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        hunger_markers = ("배고프", "배고픈", "배고파", "배가고파", "배는고픈", "고픈데", "출출", "허기")
        light_markers = (
            "무거운건싫",
            "무겁지않",
            "기름진건피",
            "기름진건싫",
            "가볍게",
            "가벼운",
            "속편",
            "속이답답",
            "부담없는",
            "간단하게",
        )
        request_markers = ("뭐먹", "먹기좋", "먹는게좋", "먹을만한", "먹을만", "추천", "뭐있", "뭐가좋")
        return (
            any(marker in compact for marker in hunger_markers)
            and any(marker in compact for marker in light_markers)
            and any(marker in compact for marker in request_markers)
        )

    @staticmethod
    def _body_state_schema(normalized: str, *, is_question: bool) -> str | None:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        if not compact:
            return None
        if is_question and any(
            marker in compact
            for marker in (
                "뭐먹",
                "먹을까",
                "먹지",
                "추천",
                "어떻게",
                "어쩌",
                "방법",
                "괜찮",
                "좋을까",
            )
        ):
            return None
        body_signal_markers = (
            "배가고프",
            "배고프",
            "배고파",
            "배가고파",
            "배고픔",
            "속이빈",
            "허기",
            "출출",
            "배가아프",
            "복통",
            "속이아프",
            "속이안좋",
            "속이불편",
            "메스꺼",
            "체한",
            "목이좀칼칼",
            "목이칼칼",
            "목이따끔",
            "목이아프",
            "목말라",
            "목마르",
            "목이말라",
            "목이마르",
            "갈증",
            "물이마시고싶",
            "물마시고싶",
            "머리가살짝아프",
            "머리가살짝아파",
            "머리가아프",
            "머리아파",
            "두통",
        )
        if any(marker in compact for marker in body_signal_markers):
            return "body_signal_interpretation"
        low_energy_markers = (
            "기운이별로없",
            "기운이없",
            "기운없",
            "몸이축처진",
            "몸이축처지",
            "몸이무겁",
            "몸이무거워",
            "피곤",
            "졸려",
            "졸리",
            "졸림",
            "컨디션별로",
            "컨디션이별로",
            "잠을거의못잤",
            "잠을못잤",
            "잠못잤",
            "잠을설쳤",
        )
        if any(marker in compact for marker in low_energy_markers):
            return "low_energy_support"
        return None

    @staticmethod
    def _is_memory_reality_conflict_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        relation_time_markers = (
            "우리가",
            "우리",
            "저번에",
            "지난번",
            "작년",
            "예전에",
            "처음만났",
            "며칠전에",
            "그때",
        )
        conflict_markers = (
            "기억을왜곡",
            "왜곡",
            "거짓말하는거야",
            "거짓말을하는거야",
            "그런적없",
            "대체뭐가진짜",
            "뭐가진짜",
            "뭐가맞는",
            "뭐가맞아",
            "분명",
            "사실그거",
        )
        return any(marker in compact for marker in relation_time_markers) and any(
            marker in compact for marker in conflict_markers
        )

    @staticmethod
    def _is_unverified_memory_boundary_question_text(normalized: str, is_question: bool) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        if not compact:
            return False
        relation_markers = (
            "우리가",
            "우리",
            "저번에",
            "지난번",
            "작년",
            "예전에",
            "처음만났",
            "처음만난",
            "며칠전에",
            "그때",
            "네가저번에",
            "저번에네가",
            "네가기억못",
        )
        memory_markers = (
            "기억안",
            "기억이나지",
            "기억이안",
            "기억못",
            "기억못하는일",
            "기억나",
            "기억해",
            "먹었던",
            "봤던",
            "갔었던",
            "갔었다",
            "추천해준",
            "빌려준",
            "말했던",
            "약속했던",
            "같이했던",
            "같이봤던",
        )
        response_markers = (
            "어떻게대답",
            "뭐라고대답",
            "어떻게말",
            "뭐라고말",
            "이름이뭐",
            "뭐였지",
            "기억나",
            "기억해",
            "맞아",
            "단정",
            "확인부터",
            "우기면",
        )
        if not any(marker in compact for marker in relation_markers) and not (
            "기억못" in compact and any(marker in compact for marker in ("우기", "맞다고", "확인"))
        ):
            return False
        if not any(marker in compact for marker in memory_markers):
            return False
        return is_question or any(marker in compact for marker in response_markers)

    @staticmethod
    def _is_reflective_judgment_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".+같지\?$",
            r".+낫지\?$",
            r".+하지\?$",
            r".+되지\?$",
            r".+않아\?$",
            r".+않나\?$",
            r".+겠지\?$",
            r".+달라지지\?$",
            r".+차분해\?$",
            r".+뿌듯하지\?$",
            r".+재밌지\?$",
            r".+공기지\?$",
            r".+느껴지지\?$",
            r".+많지\?$",
            r".+어렵지\?$",
            r".+납득되지\?$",
            r".+부담되지\?$",
            r".+중요하지\?$",
            r".+이해돼\?$",
            r".+이상하진\s*않지\?$",
            r".+이상하지\s*않지\?$",
            r".+현실적일까\?$",
            r".+실감날\s*것\s*같지\?$",
            r".+대단해\s*보여\?$",
            r".+너도\s*그렇게\s*봐\?$",
            r".+너도\s*그렇게\s*보니\?$",
            r".+어느\s*정도\s*맞는\s*것\s*같아\?$",
            r".+왜\s*그런\s*걸까\?$",
            r".+왜\s*그럴까\?$",
            r".*(?:에\s*대해|에대해)\s*어떻게\s*생각해\?$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_broad_knowledge_reflection_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False

        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized).lower()
        if not compact:
            return False

        topic_markers = (
            "민주주의",
            "자본주의",
            "사회주의",
            "선거",
            "민심",
            "정치인",
            "정치",
            "왕권",
            "의회",
            "독재",
            "포퓰리즘",
            "국가",
            "복지",
            "표현의자유",
            "언론자유",
            "가짜뉴스",
            "혐오표현",
            "다수결",
            "개인의자유",
            "법",
            "도덕",
            "처벌",
            "사형제도",
            "범죄",
            "안보",
            "사생활",
            "감시",
            "언론",
            "세대갈등",
            "빈부격차",
            "교육",
            "교육열",
            "능력주의",
            "공정",
            "기후위기",
            "기업책임",
            "경제성장",
            "경제가성장",
            "전통",
            "좋은사회",
            "모두가행복",
            "강한리더",
            "민주적절차",
            "전문가",
            "대중여론",
            "정치적중립",
            "민족주의",
            "음모론",
            "권력",
            "개인의성공",
            "노력",
            "환경의영향",
            "계층",
            "가난",
            "실패",
            "경쟁적",
            "창의성",
            "한국과중국과일본",
            "한중일",
            "남북통일",
            "동아시아",
            "조선",
            "고려",
            "삼국시대",
            "메이지유신",
            "중국왕조",
            "왕조",
            "냉전",
            "전쟁",
            "전쟁책임",
            "역사적사과",
            "혁명",
            "역사를배우는이유",
            "한글창제",
            "식민지",
            "식민지배",
            "역사",
            "역사인물",
            "이민자",
            "문화차이",
            "인권",
            "기술발전",
            "기술이일자리",
            "일자리",
            "과학",
            "과학기술",
            "자유의지",
            "자유",
            "평등",
            "죽음",
            "삶의미",
            "가상현실",
            "영생",
            "돈으로행복",
            "행복을살수",
            "진짜어른",
            "어른이된다는",
            "종교",
            "본성",
            "같은실수",
            "지식",
            "현명",
            "ai에게권리",
            "인공지능에게권리",
            "기억을복제",
            "기억이불완전",
            "진실은",
            "같은사람",
            "ai가만든예술",
            "인공지능이판사",
            "ai가판사",
            "화학반응",
            "사랑은환상",
            "선한목적",
            "나쁜수단",
            "데이터로남",
            "미래세대",
            "좋은리더",
        )
        if not any(marker in compact for marker in topic_markers):
            return False

        inquiry_markers = (
            "왜",
            "어떻게",
            "어디까지",
            "정당",
            "동의",
            "볼수",
            "생각",
            "일까",
            "걸까",
            "뭘까",
            "무엇",
            "의미",
            "영향",
            "기준",
            "가능",
            "반영",
            "도움",
            "해가",
            "우선",
            "책임",
            "가까울까",
            "반복",
            "바뀔",
            "남길",
            "답할수",
            "수있",
            "될까",
            "있을까",
            "만들까",
            "나쁠",
            "나을",
            "질까",
            "제한",
            "해야",
            "따라야",
            "뭐라고",
            "차이",
            "일까",
            "을까",
            "걸까",
            "클까",
            "봐",
            "맞을까",
            "필요",
            "위험",
            "중요",
            "우선",
            "먼저",
            "목표",
            "이상",
        )
        if any(marker in compact for marker in ("구조선", "게시물", "눈팅", "호캉스", "현타", "들어갈래", "고를래")):
            return False
        return any(marker in compact for marker in inquiry_markers)

    @staticmethod
    def _is_advice_process_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question and not any(marker in normalized for marker in ("순서를", "첫 단계", "뭐부터", "우선", "정리")):
            return False
        patterns = (
            r".+무엇부터\s*해야\s*할까\?$",
            r".+(?:가장\s*먼저|먼저|처음)\s*해야\s*할\s*(?:건|것|거)\s*(?:무엇|뭐|뭘)(?:일까)?\?$",
            r".+무엇부터\s*점검해야\s*할까\?$",
            r".+무엇부터\s*관찰할까\?$",
            r".+무엇부터\s*해보는\s*게\s*좋을까\?$",
            r".+무엇부터\s*분명히\s*하면\s*좋을까\?$",
            r".+뭘\s*먼저\s*(봐야|해야)\s*할까\?$",
            r".+우선\s*(확인|봐야)\s*할까\?$",
            r".+우선\s*(확인|점검)해야\s*할까\?$",
            r".+어떻게\s*읽어야\s*할까\?$",
            r".+어떻게\s*시작해야\s*할까\?$",
            r".+어떻게\s*시작할까\?$",
            r".+어떤\s*순서로\s*보면\s*좋을까\?$",
            r".+어떻게\s*가볍게\s*확인할\s*수\s*있을까\?$",
            r".+어떻게\s*입는\s*게\s*좋을까\?$",
            r".+뭐라고\s*하는\s*게\s*좋을까\?$",
            r".+뭘\s*실험해볼까\?$",
            r".+뭐가\s*무난할까\?$",
            r".+어떤\s*쪽을\s*우선해야\s*할까\?$",
            r".+순서를\s*짧게\s*잡아줘\.?$",
            r".+순서\s*좀\s*잡아줘\.?$",
            r".+첫\s*단계는\s*뭐가\s*좋아\?$",
            r".+막막하면\s*첫\s*단계는\s*뭐가\s*좋아\?$",
            r".+막막하면\s*뭐부터\s*(?:보면|하면|잡으면)\s*좋을까\?$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_expressive_request_text(normalized: str) -> bool:
        patterns = (
            r".+표현해줘\.?$",
            r".+묘사해줘\.?$",
            r".+설명해줘\.?$",
            r".+그려줘\.?$",
            r".+풀어줘\.?$",
            r".*(?:문장|말투|표현|대사).*(?:덜\s*공격적으로|부드럽게|순하게|자연스럽게|짧게).*(?:바꿔|고쳐|수정|다듬).*(?:줘|줄래)\.?$",
            r".*(?:문장|말투|표현|대사).*(?:바꿔|고쳐|수정|다듬).*(?:줘|줄래)\.?$",
            r".+시선으로\s*보면\?$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_reflective_observation_text(normalized: str, is_question: bool) -> bool:
        if is_question:
            question_patterns = (
                r".+실감할\s*때\s*있어\?$",
                r".+잊히지\?$",
                r".+공기지\?$",
                r".+재밌지\?$",
                r".+중\s*하나\s*같아\?$",
                r".+종류의\s*느낌\s*같아\?$",
                r".+감정.*느낌\s*같아\?$",
                r".+느껴지지\?$",
            )
            return any(re.search(pattern, normalized) for pattern in question_patterns)
        if len(normalized) < 20:
            return False
        end_patterns = (
            r".+같아\.?$",
            r".+같아서\.?$",
            r".+생각이\s*들어\.?$",
            r".+생각이\s*들더라\.?$",
            r".+느껴져\.?$",
            r".+보여\.?$",
            r".+좋더라\.?$",
            r".+같거든\.?$",
            r".+처럼.+느껴져\.?$",
            r".+종류의.+같아\.?$",
            r".+느끼게\s*돼\.?$",
            r".+느끼게\s*되더라\.?$",
        )
        clause_markers = (
            "생각이 들어",
            "생각이 들더라",
            "느낌이 들어",
            "느낌이 들더라",
            "느껴져",
            "느껴질 때가 있어",
            "보여서",
            "좋더라",
            "같거든",
            "이해가 가더라",
            "선명해지는 식으로",
            "느끼게 돼",
            "느끼게 되더라",
            "들어 있다는 거",
            "변하지 않았다는",
        )
        return any(re.search(pattern, normalized) for pattern in end_patterns) or any(
            marker in normalized for marker in clause_markers
        )

    @staticmethod
    def _is_relational_interpretation_text(normalized: str, is_question: bool) -> bool:
        if len(normalized) < 10:
            return False
        markers = (
            "하트만 남기고 끝났",
            "하트만 남기고",
            "거리가 생긴",
            "화제 바뀌",
            "허락을 받았다는 느낌보다",
            "죄책감을",
            "이해받기 위한 시도라기보다",
            "닫힌 문을 다시 두드리",
            "보류나 거절 쪽으로",
            "반찬 챙겨두는 말이 없어졌",
            "안쪽은 아니구나",
            "계속 밀리는 사람은 나구나",
            "말이 없어졌",
            "눈을 똑바로 안 쳐다보",
            "눈을똑바로안쳐다보",
            "숨 막힐 정도로 뚫어지게",
            "옆자리 비워두고",
            "옆자리비워두고",
            "비참했어",
            "선택권을 주는 척",
            "선택권을주는척",
            "상황을 끌고 가",
            "상황을끌고가",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _is_comparative_reflection_text(normalized: str, is_question: bool) -> bool:
        if len(normalized) < 16:
            return False
        patterns = (
            r".+보다\s*덜\s*상처받는\s*게.+중요해\s*보인다\.?$",
            r".+보다\s*덜\s*상처받는\s*게.+중요해지는\s*걸까\?$",
            r".+보다\s*덜\s*힘들게\s*보내는\s*게\s*목표가\s*된다\.?$",
            r".+보다\s*덜\s*힘들게\s*보내는\s*게\s*목표일까\?$",
            r".+보다\s*덜\s*[가-힣]+\s*게.+\.?$",
        )
        markers = (
            "잘 넘기는 것보다 덜 상처받는",
            "잘 보내는 것보다 덜 힘들게",
            "덜 상처받는 게 더 중요해",
            "덜 힘들게 보내는 게 목표",
        )
        return any(re.search(pattern, normalized) for pattern in patterns) or any(
            marker in normalized for marker in markers
        )

    @classmethod
    def _is_aesthetic_reflection_text(cls, normalized: str, is_question: bool) -> bool:
        aesthetic_markers = (
            "빛",
            "어둠",
            "파랑",
            "빨강",
            "바다",
            "물고기",
            "냄새",
            "풍경",
            "분위기",
            "밤하늘",
            "스탠드",
            "색감",
            "침묵",
            "여백",
            "장면",
            "차갑",
            "예쁘",
        )
        if not any(marker in normalized for marker in aesthetic_markers):
            return False
        if is_question:
            question_patterns = (
                r".+다르지\?$",
                r".+달라지지\?$",
                r".+예쁘기도\s*하지\?$",
                r".+차분해\?$",
            )
            return any(re.search(pattern, normalized) for pattern in question_patterns)
        strong_non_question_markers = (
            "파랑",
            "빨강",
            "냄새",
            "풍경",
            "분위기",
            "색감",
            "침묵",
            "여백",
            "장면",
            "차갑",
            "예쁘",
            "스탠드",
        )
        if not any(marker in normalized for marker in strong_non_question_markers):
            return False
        return cls._is_reflective_observation_text(normalized, is_question=False)

    @staticmethod
    def _is_reason_probe_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        patterns = (
            r".*왜\s*저러는\s*거야\?$",
            r".*왜\s*그런\s*거야\?$",
            r".*이유가\s*뭘까\?$",
            r".*(?:그\s*)?판단(?:의)?\s*근거(?:가)?\s*(?:뭐야|뭘까|무엇이야)\?$",
            r".*그렇게\s*(?:말한|판단한)?\s*근거(?:가)?\s*(?:뭐야|뭘까|무엇이야)\?$",
            r".*왜\s*그렇게\s*되는\s*걸까\?$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_relationship_dependency_boundary_question_text(normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        dependency_markers = (
            "기대기만",
            "너무기대",
            "기대는게",
            "의존만",
            "기대는것같",
            "기대고만",
            "기댈수",
            "전부떠안",
            "불안을네가전부",
        )
        boundary_markers = (
            "부담",
            "떠안",
            "말해줄수",
            "어떻게말해",
            "솔직히말",
            "선을그어",
            "경계",
        )
        return any(marker in compact for marker in dependency_markers) and any(
            marker in compact for marker in boundary_markers
        )

    @classmethod
    def _is_decision_request_question_text(cls, normalized: str, is_question: bool) -> bool:
        if not is_question:
            return False
        if cls._is_weather_conditioned_activity_opinion_text(normalized):
            return True
        patterns = (
            r".+할까\?$",
            r".+될까\?$",
            r".+나을까\?$",
            r".+맞을까\?$",
            r".+쪽이\s*맞을까\?$",
            r".+쪽이\s*나을까\?$",
            r".+해볼까\?$",
            r".+가볼까\?$",
            r".+해도\s*괜찮을까\?$",
            r".+해도\s*괜찮\??$",
            r".+괜찮을까\?$",
            r".+할지\s*말지\s*애매하",
            r".+선택지도\s*있을까\?$",
            r".+다른\s*선택지.*있을까\?$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_subdued_positive_feeling_text(normalized: str) -> bool:
        soft_positive_patterns = (
            r"생각보다\s*잘\s*풀",
            r"잘\s*풀렸",
            r"잘\s*된\s*편",
            r"좀\s*괜찮",
            r"괜찮은\s*편",
            r"괜찮은\s*쪽",
            r"크게\s*들뜨진?\s*않",
            r"막\s*들뜨진?\s*않",
            r"마음이\s*좀\s*놓였",
            r"한숨\s*돌렸",
            r"조금\s*안도",
            r"은근히\s*다행",
            r"후련하",
            r"덜\s*버거",
            r"조금\s*덜\s*버거",
            r"한결\s*낫",
            r"덜\s*무거",
            r"조용히\s*괜찮",
        )
        return any(re.search(pattern, normalized) for pattern in soft_positive_patterns)

    @staticmethod
    def _is_quiet_weather_feeling_text(normalized: str) -> bool:
        weather_patterns = (
            r"비가\s*오",
            r"비\s*오네",
            r"흐리",
            r"꾸물",
        )
        withdrawal_patterns = (
            r"조용히\s*있고\s*싶",
            r"말수.*줄",
            r"조용한\s*쪽",
            r"낮은\s*톤",
        )
        return any(re.search(pattern, normalized) for pattern in weather_patterns) and any(
            re.search(pattern, normalized) for pattern in withdrawal_patterns
        )

    @staticmethod
    def _is_social_awkwardness_text(normalized: str) -> bool:
        patterns = (
            r"(대화|말|얘기).*(자꾸|계속).*(어색해|어색해져|어색하)",
            r"어색해져",
            r"어색한\s*공기",
            r"말이\s*자꾸\s*꼬",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_low_energy_checkin_text(normalized: str) -> bool:
        patterns = (
            r"말수가\s*(좀|조금|약간)?\s*적",
            r"좀\s*말수가\s*적",
            r"짧게\s*말할\s*것\s*같",
            r"대답이\s*(좀|조금|약간)?\s*짧",
            r"말이\s*(좀|조금|약간)?\s*짧아질\s*것\s*같",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_opinion_text(normalized: str) -> bool:
        opinion_patterns = (
            r"어때\s*보여",
            r"낫냐",
            r"별론가",
            r"별로인가",
            r"어떰",
            r"어떻게\s*생각해",
            r".*무의식적인\s*버릇.*뭐",
            r".*천만\s*원.*빌려줄\s*수\s*있",
            r".*차용증.*빌려줄\s*수\s*있",
        )
        return any(re.search(pattern, normalized) for pattern in opinion_patterns)

    @staticmethod
    def _is_opinion_habit_question_text(normalized: str) -> bool:
        patterns = (
            r"자주.*편이야",
            r"보통.*편이야",
            r"먹는\s*편이야",
            r"챙겨.*편이야",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_opinion_self_style_question_text(normalized: str) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", normalized)
        patterns = (
            r"너는.*무슨\s*말부터.*편이야",
            r"너는.*어떤\s*말부터.*편이야",
            r"너는.*먼저.*무슨\s*말",
            r"너는.*뭐부터.*꺼내는\s*편이야",
            r"너는.*어떤\s*방식.*(?:옆|곁).*있어주는\s*편이야",
            r"네가.*사람\s*곁.*편한\s*방식",
            r"너는.*위로할\s*때.*(?:다정|안아주는|기준).*쪽이야",
            r"(?:black|블랙)은.*(?:사람|내|곁).*어떤\s*역할.*(?:하려고|만들어진).*쪽이야",
        )
        compact_patterns = (
            "너는내가흔들릴때어떤방식으로옆에있어주는편이야",
            "네가사람곁에있어줄때제일편한방식",
            "너는위로할때다정하게안아주는쪽이야아니면기준을잡아주는쪽이야",
            "black은사람곁에서어떤역할을하려고만들어진쪽이야",
            "블랙은사람곁에서어떤역할을하려고만들어진쪽이야",
        )
        return any(re.search(pattern, normalized) for pattern in patterns) or any(
            pattern in compact for pattern in compact_patterns
        )

    @staticmethod
    def _is_soft_social_comment_text(normalized: str) -> bool:
        soft_social_patterns = (
            r"(오늘은|지금은).*(남아볼게|맞춰볼게|가만히\s*있을게|있을게)",
            r"네\s*쪽에.*(가만히\s*있을게|있을게)",
            r"말보다\s*마음부터\s*맞춰볼게",
        )
        return any(re.search(pattern, normalized) for pattern in soft_social_patterns)

    @staticmethod
    def _is_generic_smalltalk_text(normalized: str) -> bool:
        patterns = (
            r"뭐\s*하고\s*있",
            r"뭐\s*하고\s*있었",
            r"뭐\s*하는\s*중",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_compliment_text(normalized: str) -> bool:
        compliment_patterns = (
            r"잘하네",
            r"잘한다",
            r"말\s*잘하네",
            r"믿음직",
            r"든든하네",
            r"든든하다",
            r"믿음직하다",
            r"괜찮은데",
            r"괜찮네",
            r"꽤\s*괜찮",
            r"괜찮다",
            r"나쁘지\s*않네",
            r"생각보다\s*(괜찮|든든|믿음직)",
        )
        return any(re.search(pattern, normalized) for pattern in compliment_patterns)

    @staticmethod
    def _is_slang_greeting_text(normalized: str) -> bool:
        slang_greetings = ("와썹", "왓섭", "wassup", "sup")
        return normalized in slang_greetings

    @staticmethod
    def _is_deny_text(normalized: str) -> bool:
        patterns = (
            r"아닌\s*듯",
            r"아닌\s*것\s*같",
            r"아닌거\s*같",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_permission_release_text(normalized: str) -> bool:
        patterns = (
            r"답\s*안\s*해도\s*돼",
            r"대답\s*안\s*해도\s*돼",
            r"말\s*안\s*해도\s*돼",
            r"굳이.+안\s*해도\s*돼",
            r"편한\s*대로\s*해",
            r"안\s*해도\s*돼",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _has_recent_supportive_flow_context(state: ConversationState | None) -> bool:
        if state is None:
            return False
        if state.last_action in {ActionType.SHARE_FEELING, ActionType.CONTINUE_CONVERSATION}:
            return True
        if state.last_intent == Intent.SMALLTALK_FEELING:
            return True
        recent_turns = state.recent_turns[-2:]
        if any(turn.action in {ActionType.SHARE_FEELING, ActionType.CONTINUE_CONVERSATION} for turn in recent_turns):
            return True
        return bool(recent_turns)

    @staticmethod
    def _is_soft_handoff_reassurance_text(normalized: str, state: ConversationState | None) -> bool:
        if not HeuristicIntentClassifier._has_recent_supportive_flow_context(state):
            return False
        patterns = (
            r"무리\s*안\s*해도\s*돼",
            r"억지로\s*더\s*말\s*안\s*해도\s*돼",
            r"굳이\s*더\s*안\s*꺼내도\s*돼",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_quiet_feeling_validation_text(normalized: str, state: ConversationState | None) -> bool:
        if not HeuristicIntentClassifier._has_recent_supportive_flow_context(state):
            return False
        patterns = (
            r"그럴\s*때(?:도|가)\s*있(?:어|지)\.?",
            r"짧게\s*짧아도\s*괜찮아\.?",
            r"짧아도\s*괜찮아\.?",
            r"짧게\s*해도\s*괜찮아\.?",
        )
        if not any(re.fullmatch(pattern, normalized) for pattern in patterns):
            return False
        return len(normalized) <= 18

    @staticmethod
    def _is_soft_refusal_text(normalized: str) -> bool:
        if any(marker in normalized for marker in ("무슨 뜻", "뜻이 뭐", "의미가 뭐", "무슨 의미")):
            return False
        patterns = (
            r"좀\s*어렵겠",
            r"조금\s*어렵겠",
            r"힘들\s*것\s*같",
            r"어려울\s*것\s*같",
            r"쉽지\s*않을\s*것\s*같",
            r"애매할\s*것\s*같",
            r"곤란할\s*것\s*같",
            r"지금은\s*좀",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_conditional_boundary_text(normalized: str) -> bool:
        patterns = (
            r"(수위|톤|강도).*(조금|좀).*(낮추|줄이).*(이어갈\s*수\s*있|계속\s*얘기할\s*수\s*있)",
            r"말\s*세기.*(조금|좀).*(낮추|줄이).*(이어갈\s*수\s*있|계속\s*얘기할\s*수\s*있)",
            r"(조금|좀).*(덜\s*세게).*(가면|하면).*(이어갈\s*수\s*있|계속\s*얘기할\s*수\s*있)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_teasing_laughter_text(normalized: str) -> bool:
        return HeuristicIntentClassifier._is_laugh_text(normalized) and any(
            token in normalized for token in HeuristicIntentClassifier._mild_tease_tokens
        )

    @staticmethod
    def _is_sarcastic_tease_text(normalized: str) -> bool:
        if not HeuristicIntentClassifier._is_laugh_text(normalized):
            return False
        patterns = (
            r"(아주|참|진짜).*(잘한다|잘하네|천재(다|네))",
            r"(잘한다|잘하네|천재(다|네)).*(아주|참|진짜)",
            r"와\s*진짜\s*(잘한다|잘하네)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_tease_text(normalized: str) -> bool:
        return HeuristicIntentClassifier._is_teasing_laughter_text(normalized) or HeuristicIntentClassifier._is_sarcastic_tease_text(normalized)

    @staticmethod
    def _is_self_conscious_check_text(normalized: str) -> bool:
        patterns = (
            r"오버한\s*건\s*아니",
            r"오버한거\s*아니",
            r"괜히\s*말했나",
            r"내가\s*너무.+아니지",
            r"부담.*아니지",
            r"실수한\s*건\s*아니",
            r"이상한\s*건\s*아니",
            r"내가\s*괜히",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_relationship_check_text(normalized: str, state: ConversationState | None = None) -> bool:
        base_patterns = (
            r"선\s*넘은\s*건\s*아니",
            r"불편한\s*건\s*아니",
            r"실례된\s*건\s*아니",
            r"부담\s*준\s*건\s*아니",
            r"분위기\s*이상하게\s*만든\s*건\s*아니",
            r"기분\s*나쁜\s*건\s*아니",
            r"마음\s*상한\s*건\s*아니",
            r"서운한\s*건\s*아니",
        )
        if any(re.search(pattern, normalized) for pattern in base_patterns):
            return True

        if not HeuristicIntentClassifier._has_recent_repair_context(state):
            return False

        contextual_patterns = (
            r"이제\s*괜찮지",
            r"이제\s*괜찮아졌",
            r"기분\s*나쁘진\s*않",
            r"기분\s*안\s*나쁘",
            r"불편하진\s*않",
            r"불편했지",
            r"마음\s*상하진\s*않",
            r"서운하진\s*않",
            r"화난\s*건\s*아니",
        )
        return any(re.search(pattern, normalized) for pattern in contextual_patterns)

    @staticmethod
    def _has_recent_repair_context(state: ConversationState | None) -> bool:
        if state is None:
            return False
        if state.last_action == ActionType.DEESCALATE or state.last_intent == Intent.HOSTILE:
            return True
        if state.tension >= 0.2:
            return True
        recent_turns = state.recent_turns[-3:]
        return any(turn.action == ActionType.DEESCALATE for turn in recent_turns)

    @staticmethod
    def _is_repair_attempt_text(normalized: str, state: ConversationState | None) -> bool:
        if not (HeuristicIntentClassifier._has_recent_repair_context(state) or "아까" in normalized):
            return False
        patterns = (
            r"아까.+심했지",
            r"아까.+세게\s*말했",
            r"불편했으면\s*미안",
            r"아까.+미안",
            r"내가\s*너무\s*세게\s*말했",
            r"내가\s*좀\s*심했",
            r"내가\s*말이\s*심했",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _has_recent_social_awkwardness_context(state: ConversationState | None) -> bool:
        if state is None:
            return False
        recent_turns = state.recent_turns[-3:]
        context_patterns = (
            r"어색",
            r"민망",
            r"뻘쭘",
            r"눈치\s*보",
            r"분위기\s*이상",
            r"말\s*실수",
            r"괜히\s*오버",
            r"선\s*넘",
        )
        return any(
            any(re.search(pattern, turn.user_text) for pattern in context_patterns)
            for turn in recent_turns
        )

    @staticmethod
    def _is_social_aftereffect_text(normalized: str, state: ConversationState | None) -> bool:
        if not HeuristicIntentClassifier._has_recent_social_awkwardness_context(state):
            return False
        patterns = (
            r"끝나고\s*나서도.+(다시|계속)",
            r"그\s*장면만.+(다시\s*재생|계속\s*재생)",
            r"(계속|자꾸).+(떠올라|생각나)",
            r"(계속|자꾸).+(맴돌아|맴돈)",
            r"(혼자|계속)\s*복기하게\s*돼",
            r"집에\s*가서도.+(생각나|떠올라)",
            r"집\s*와서도.+(맴돌아|맴돈|남아)",
            r"(그때|아까).+표정.+(머리|머릿속).+안\s*빠져",
            r"표정이.+머리에서.+안\s*빠져",
            r"(아까|그때).*(공기|분위기).*(그대로\s*)?남아",
            r"(공기|분위기).*(아직도|여전히).*(남아|맴돌아)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_after_social_hollow_text(normalized: str) -> bool:
        patterns = (
            r"(재밌게|즐겁게|좋게).+(있다|놀다|만나고|보고).+(집에|혼자).+(허전|빈자리|공허)",
            r"(만나고|보고)\s*왔는데.+(오히려|괜히).+(허전|공허)",
            r"(좋았던|재밌던)\s*시간.+(끝나|지나).+(허전|빈자리)",
            r"(사람들|친구들).+(만나고|보고).+(돌아오니까|집에\s*오니까).+(허전|공허)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_social_reconnect_relief_text(normalized: str, state: ConversationState | None) -> bool:
        if not HeuristicIntentClassifier._has_recent_social_awkwardness_context(state):
            return False
        patterns = (
            r"(나중엔|뒤로\s*갈수록|시간\s*지나고).*(덜\s*얼어|덜\s*굳어)",
            r"(조금|좀).*(덜\s*얼어|덜\s*굳어)",
            r"(조금|좀).*(풀린|풀려)\s*있",
            r"(나중엔|뒤로\s*갈수록).*(긴장|어색함).*(덜했|덜해졌)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_face_saving_retreat_text(normalized: str) -> bool:
        patterns = (
            r"그냥\s*내가\s*괜한\s*말",
            r"내가\s*괜히\s*꺼냈",
            r"그냥\s*잊어줘",
            r"신경\s*쓰지\s*마.+실수했",
            r"분위기\s*망쳤",
            r"됐다\s*내가\s*괜히",
            r"아냐\s*그냥.+했네",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_reluctant_acceptance_text(normalized: str) -> bool:
        patterns = (
            r"싫은\s*건\s*아닌데",
            r"반대는\s*아닌데",
            r"하라면\s*하겠",
            r"하자면\s*하긴\s*하지",
            r"못\s*하는\s*건\s*아닌데",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_deferred_acceptance_text(normalized: str) -> bool:
        patterns = (
            r"나중에\s*보면\s*좋",
            r"그때\s*가서\s*다시\s*얘기하자",
            r"괜찮아지면\s*하자",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_deferred_rejection_text(normalized: str) -> bool:
        patterns = (
            r"나중에\s*(보자|하자)",
            r"다음에\s*(보자|하자)",
            r"담에\s*(보자|하자)",
            r"이번엔\s*말고\s*다음에",
            r"오늘은\s*말고\s*다음에",
            r"지금은?\s*말고\s*다음에",
            r"일단은\s*나중에\s*(보자|하자)",
            r"지금\s*당장은\s*말고\s*나중에",
            r"다음\s*기회에",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_testing_the_waters_text(normalized: str) -> bool:
        patterns = (
            r"말해도\s*될지\s*모르겠",
            r"말\s*해도\s*되나\s*모르겠",
            r"물어봐도\s*돼\??$",
            r"뜬금없나",
            r"이런\s*말\s*해도\s*되나",
            r"별건\s*아닌데.+물어봐도\s*돼",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_laugh_text(normalized: str) -> bool:
        return bool(re.search(r"[ㅋㅎ]{2,}", normalized))

    @staticmethod
    def _is_surprise_text(normalized: str) -> bool:
        markers = ("헐", "대박", "실화", "뭐야 이거", "뭐냐 이거", "예상 밖")
        if any(marker in normalized for marker in markers):
            return True
        patterns = (
            r"^\?+$",
            r"^\?\s+\?+$",
            r"^\?[\?\s]+\?+$",
            r"^와\s*이건\s*뭐냐$",
            r"^와\s*이게\s*뭐냐$",
            r"^이건\s*뭐냐$",
            r"^이게\s*뭐냐$",
            r"^[?？]+\s*[?？]*$",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_meaning_query_text(normalized: str) -> bool:
        if any(marker in normalized for marker in ("무슨 말부터", "무슨말부터", "어떤 말부터", "어떤말부터")):
            return False
        meaning_markers = (
            "무슨뜻",
            "무슨 뜻",
            "뜻이 뭐",
            "뜻이뭐",
            "뜻이야",
            "무슨 의미",
            "의미가 뭐",
            "의미가뭐",
            "무슨 말",
            "무슨말",
        )
        if any(marker in normalized for marker in meaning_markers):
            return True

        meaning_patterns = (
            r".+[가는은는]\s*무슨\s*뜻",
            r".+뜻이\s*뭐",
            r".+의미가\s*뭐",
        )
        return any(re.search(pattern, normalized) for pattern in meaning_patterns)

    @staticmethod
    def _is_tentative_suggestion_text(normalized: str) -> bool:
        patterns = (
            r"혹시.+(보자|하자)",
            r"괜찮으면.+(보자|하자)",
            r"편하면.+(보자|하자)",
            r"나중에.+(보자|하자)",
            r"시간\s*되면.+(보자|하자)",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_contextual_social_followup_text(normalized: str, state: ConversationState | None) -> bool:
        if state is None or not state.recent_turns:
            return False

        recent_actions = {turn.action for turn in state.recent_turns[-3:]}
        if not recent_actions.intersection(
            {
                ActionType.CONTINUE_CONVERSATION,
                ActionType.SMALL_TALK,
                ActionType.ACKNOWLEDGE,
                ActionType.SHARE_FEELING,
                ActionType.SHARE_OPINION,
                ActionType.ASK_CLARIFICATION,
                ActionType.EXPLAIN_REASON,
                ActionType.EXPLAIN_CAPABILITIES,
                ActionType.ANSWER_IDENTITY,
                ActionType.SEARCH_ANSWER,
                ActionType.NEWS_ANSWER,
                ActionType.TELL_TIME,
                ActionType.WEATHER_LOOKUP,
                ActionType.GAME_CHAT,
                ActionType.GAME_ACCEPT_OR_DECLINE,
                ActionType.MUSIC_CHAT,
                ActionType.RECOMMEND,
                ActionType.DEESCALATE,
                ActionType.TEASE_BACK,
                ActionType.REACT_LAUGH,
                ActionType.REACT_SURPRISE,
            }
        ):
            return False

        explicit_patterns = (
            r"한\s*줄만\s*더",
            r"조금만\s*(더\s*)?(부드럽게|세게|낮춰|올려)",
            r"좀만\s*(더\s*)?(부드럽게|세게|낮춰|올려)",
            r"그\s*기준으로",
            r"그\s*방향으로",
            r"그\s*톤으로",
            r"그\s*결로",
            r"그\s*흐름으로",
            r"숨\s*돌리",
            r"이어\s*가",
        )
        if any(re.search(pattern, normalized) for pattern in explicit_patterns):
            return True

        if HeuristicIntentClassifier._is_contextual_redirect_text(normalized):
            return True

        short_context_markers = (
            "그 기준",
            "그 방향",
            "그 톤",
            "그 결",
            "그 흐름",
            "조금만",
            "좀만",
            "부드럽게",
            "세게",
        )
        return len(normalized) <= 28 and any(marker in normalized for marker in short_context_markers)

    @staticmethod
    def _is_soft_handoff_followup_text(normalized: str, state: ConversationState | None) -> bool:
        if not HeuristicIntentClassifier._has_recent_supportive_flow_context(state):
            return False
        patterns = (
            r"그냥\s*천천히\s*가도\s*돼",
            r"천천히\s*가도\s*돼",
            r"천천히\s*해도\s*돼",
            r"오늘은\s*그냥\s*이\s*속도로만\s*가도\s*돼",
            r"이\s*속도로만\s*가도\s*돼",
            r"짧게만\s*가도\s*돼",
            r"말\s*조금\s*적어도\s*돼",
            r"오늘은\s*그냥\s*이\s*정도만\s*꺼내도\s*돼",
            r"이\s*정도만\s*꺼내도\s*돼",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_contextual_redirect_text(normalized: str) -> bool:
        if len(normalized) > 36:
            return False

        patterns = (
            r"아니\s*그게\s*아니라",
            r"(그건|그거|그쪽|그걸|그것)\s*말고",
            r"(그건|그거|그쪽|그걸|그것)\s*아니고",
            r"그렇게\s*말고",
            r"그\s*(기준|방향|톤|결|흐름|느낌)\s*으로",
            r"그\s*(기준|방향|톤|결|흐름|느낌)\s*으로\s*(가|잡|맞추|해)",
            r"그\s*(기준|방향|톤|결|흐름|느낌)\s*으로\s*",
            r"그\s*쪽으로",
            r"그\s*대로",
            r"그\s*걸로",
            r"그\s*거로",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_reason_request_text(normalized: str, state: ConversationState) -> bool:
        if HeuristicIntentClassifier._is_advice_process_question_text(normalized, True):
            return False
        if state.recent_turns and normalized in {"왜", "왜?", "why"}:
            return True
        if HeuristicIntentClassifier._is_third_party_interpretation_question(normalized):
            return False

        explicit_markers = (
            "왜 그렇게",
            "왜그렇게",
            "그렇게 말한 근거",
            "그렇게판단한근거",
            "그렇게 판단한 근거",
            "그 판단의 근거",
            "그 판단 근거",
            "판단한 근거",
            "판단 근거",
            "왜 그렇게 판단",
            "왜 그렇게 봐",
        )
        if any(marker in normalized for marker in explicit_markers):
            return True

        context_reference_markers = ("그렇게", "그거", "그걸", "그 말", "그 판단", "방금", "아까")
        if ("근거" in normalized or "이유" in normalized) and (
            state.recent_turns or any(marker in normalized for marker in context_reference_markers)
        ):
            return True

        return False

    @staticmethod
    def _is_third_party_interpretation_question(normalized: str) -> bool:
        if not normalized.endswith("?"):
            return False
        if not any(marker in normalized for marker in ("표정", "의도", "팀장", "대리", "상사", "친구", "동료")):
            return False
        return any(marker in normalized for marker in ("왜 그렇게", "왜그렇게", "무슨 의도", "무슨의도"))

    @staticmethod
    def _is_fact_query_text(normalized: str) -> bool:
        fact_patterns = (
            r".+의\s*수도는\s*\??$",
            r".+의\s*국기는\s*\??$",
            r".+의\s*인구는\s*\??$",
            r".+의\s*면적은\s*\??$",
            r".+의\s*대통령은\s*\??$",
            r".+의\s*총리는\s*\??$",
            r".+의\s*위치는\s*\??$",
        )
        if any(re.search(pattern, normalized) for pattern in fact_patterns):
            return True

        description_match = re.search(r"(?P<subject>.+?)(?:은|는|이|가)\s*(?:뭐야|누구야)\s*\??$", normalized)
        if not description_match:
            return False

        subject = description_match.group("subject").strip()
        subject_lower = subject.lower()
        if any(marker in subject_lower for marker in ("너", "네", "니", "black", "ai")) and any(
            marker in subject for marker in ("방식", "성격", "취향", "생각", "마음", "느낌", "곁")
        ):
            return False
        if any(marker in subject for marker in ("나", "내", "너", "네", "니", "우리", "서로", "만약")) and any(
            marker in subject
            for marker in (
                "가면",
                "마음",
                "생각",
                "기분",
                "정체성",
                "관계",
                "속셈",
                "속마음",
                "미래",
                "결말",
            )
        ):
            return False
        blocked_subjects = {
            "나",
            "너",
            "니",
            "이거",
            "이건",
            "그거",
            "그건",
            "저거",
            "저건",
            "이게",
            "그게",
            "저게",
            "오늘",
            "지금",
        }
        if subject in blocked_subjects:
            return False
        return len(subject) >= 2

    @staticmethod
    def _infer_topic_hint(
        *,
        intent: Intent,
        normalized: str,
        location: str | None,
    ) -> str | None:
        if intent == Intent.MUSIC:
            return "music"
        if intent == Intent.MEDIA_RECOMMEND:
            return "media"
        if intent in {Intent.GAME_TALK, Intent.GAME_INVITE}:
            return "game"
        if intent == Intent.ACTIVITY_INVITE:
            return "activity"
        if HeuristicIntentClassifier._is_activity_preparation_advice_text(normalized):
            return "activity"
        if intent in {Intent.SMALLTALK_OPINION, Intent.SMALLTALK_FEELING, Intent.SMALLTALK_GENERIC} and (
            HeuristicIntentClassifier._is_food_or_lifestyle_choice_question_text(normalized, "?" in normalized)
            or HeuristicIntentClassifier._is_memory_reality_conflict_question_text(normalized, "?" in normalized)
        ):
            return None
        if HeuristicIntentClassifier._is_long_form_story_cue(normalized):
            return "creative_writing"
        if intent == Intent.WHO_ARE_YOU:
            return "identity"
        if intent == Intent.HELP:
            return "capability"
        if HeuristicIntentClassifier._is_metaphorical_temperature_question_text(normalized, "?" in normalized):
            return None
        if (
            intent in {Intent.WEATHER, Intent.PROVIDE_LOCATION}
            or "날씨" in normalized
            or location is not None
            or HeuristicIntentClassifier._has_weather_surface_text(normalized)
        ):
            return "weather"
        if HeuristicIntentClassifier._is_broad_knowledge_reflection_text(normalized, "?" in normalized):
            return "knowledge_reflection"
        if intent in {Intent.SEARCH_REQUEST, Intent.TIME_DATE, Intent.NEWS}:
            return "knowledge"
        return None

    @staticmethod
    def _infer_speech_act(
        *,
        intent: Intent,
        normalized: str,
        is_question: bool,
        sentiment: str,
        state: ConversationState | None,
    ) -> str:
        if intent in {Intent.GREETING, Intent.THANKS, Intent.LAUGH, Intent.SURPRISE}:
            return "react"
        if intent in {Intent.HELP, Intent.WHO_ARE_YOU, Intent.WHY, Intent.SEARCH_REQUEST, Intent.TIME_DATE, Intent.NEWS, Intent.REPLY_REQUEST}:
            return "ask"
        if HeuristicIntentClassifier._is_proactive_checkin_text(normalized):
            return "ask"
        if HeuristicIntentClassifier._is_conversation_topic_suggestion_text(normalized):
            return "ask"
        if HeuristicIntentClassifier._is_activity_preparation_advice_text(normalized):
            return "ask"
        if HeuristicIntentClassifier._is_expressive_request_text(normalized):
            return "ask"
        if HeuristicIntentClassifier._is_face_saving_retreat_text(normalized):
            return "retreat"
        if HeuristicIntentClassifier._is_deferred_acceptance_text(normalized):
            return "defer"
        if HeuristicIntentClassifier._is_deferred_rejection_text(normalized):
            return "defer"
        if HeuristicIntentClassifier._is_repair_attempt_text(normalized, state):
            return "repair"
        if HeuristicIntentClassifier._is_tease_text(normalized):
            return "tease"
        if intent == Intent.GAME_INVITE:
            return "invite"
        if intent == Intent.ACTIVITY_INVITE:
            return "invite"
        if intent == Intent.CONFIRM:
            return "confirm"
        if intent == Intent.DENY:
            return "deny"
        if intent == Intent.HOSTILE:
            return "attack"
        if intent == Intent.TEASE:
            return "tease"
        if intent == Intent.PROVIDE_LOCATION:
            return "inform"
        if HeuristicIntentClassifier._is_testing_the_waters_text(normalized):
            return "probe"
        if HeuristicIntentClassifier._is_tentative_suggestion_text(normalized):
            return "suggest"
        if intent == Intent.SMALLTALK_FEELING and sentiment == "negative":
            return "complain"
        if intent == Intent.WEATHER:
            return "ask"
        if is_question:
            return "ask"
        return "inform"

    @staticmethod
    def _infer_response_needs(
        *,
        intent: Intent,
        speech_act: str,
        topic_hint: str | None,
        sentiment: str,
        location: str | None,
        requests_external_fact: bool,
        pragmatic_cues: list[str],
        state: ConversationState | None,
    ) -> list[str]:
        needs: list[str] = []
        if requests_external_fact:
            needs.append("grounding")
        if (
            topic_hint == "weather"
            and requests_external_fact
            and speech_act == "ask"
            and not (location or (state and state.known_location))
        ):
            needs.append("slot_fill")
        if intent in {Intent.WHY, Intent.HELP, Intent.WHO_ARE_YOU}:
            needs.append("explanation")
        if intent == Intent.NEWS and "explanation" not in needs:
            needs.append("explanation")
        if intent == Intent.SEARCH_REQUEST:
            if "grounding" not in needs:
                needs.append("grounding")
            if "explanation" not in needs:
                needs.append("explanation")
        if speech_act == "complain" and sentiment == "negative":
            needs.append("empathy")
        if intent == Intent.SMALLTALK_FEELING and sentiment == "positive":
            needs.append("acknowledgement")
        if any(
            cue in pragmatic_cues
            for cue in {
                "self_conscious_check",
                "relationship_check",
                "face_saving_retreat",
                "repair_attempt",
                "social_aftereffect",
                "social_reconnect_relief",
                "social_awkwardness",
                "quiet_feeling_validation",
                "soft_handoff_reassurance",
            }
        ) and "empathy" not in needs:
            needs.append("empathy")
        if intent in {Intent.REPLY_REQUEST, Intent.UNKNOWN}:
            needs.append("clarification")
        if speech_act in {"react", "confirm", "deny", "invite", "suggest", "probe", "defer", "retreat", "repair"}:
            needs.append("acknowledgement")
        if any(
            cue in pragmatic_cues
            for cue in {
                "tentative_suggestion",
                "permission_release",
                "reluctant_acceptance",
                "deferred_acceptance",
                "deferred_rejection",
            }
        ):
            if "acknowledgement" not in needs:
                needs.append("acknowledgement")
        if "contextual_followup" in pragmatic_cues and "social_followup" not in needs:
            needs.append("social_followup")
        if speech_act == "tease" and "social_followup" not in needs:
            needs.append("social_followup")
        if topic_hint in {"game", "music", "media"}:
            needs.append("social_followup")
        return needs

    @staticmethod
    def _infer_question_schema(
        *,
        intent: Intent,
        pragmatic_cues: list[str],
    ) -> str | None:
        cue_set = set(pragmatic_cues)
        if "proactive_checkin" in cue_set:
            return "proactive_checkin"
        if "conversation_topic_suggestion" in cue_set:
            return "conversation_topic_suggestion"
        if "activity_preparation_advice" in cue_set:
            return "activity_preparation_advice"
        if "expressive_request" in cue_set:
            return "expressive_request"
        if "knowledge_reflection" in cue_set:
            return "knowledge_reflection"
        if "reason_probe" in cue_set:
            return "reason_probe"
        if "relational_interpretation" in cue_set:
            return "relational_interpretation"
        if "comparative_reflection" in cue_set:
            return "comparative_reflection"
        if "aesthetic_reflection" in cue_set:
            return "aesthetic_reflection"
        if "reflective_observation" in cue_set:
            return "reflective_observation"
        if "body_signal_interpretation" in cue_set:
            return "body_signal_interpretation"
        if "low_energy_support" in cue_set:
            return "low_energy_support"
        if "activity_invite" in cue_set:
            return "activity_invite"
        if "long_form_story_share" in cue_set:
            return "long_form_story_share"
        if "story_summary_reaction" in cue_set:
            return "story_summary_reaction"
        if intent not in {Intent.SMALLTALK_OPINION, Intent.MEDIA_RECOMMEND, Intent.MUSIC}:
            return None
        if "memory_boundary" in cue_set:
            return "memory_boundary"
        if "light_food_recommendation" in cue_set:
            return "light_food_recommendation"
        if "relationship_boundary" in cue_set:
            return "relationship_boundary"
        if "weather_conditioned_activity_opinion" in cue_set:
            return "weather_conditioned_activity_opinion"
        if "activity_recommendation" in cue_set:
            return "activity_recommendation"
        if "concrete_topic_question" in cue_set:
            return "concrete_topic_question"
        if "honesty_boundary" in cue_set:
            return "honesty_boundary"
        if "opinion_advice_process" in cue_set:
            return "process_advice"
        if "opinion_decision_request" in cue_set:
            return "soft_decision_advice"
        if "opinion_reflective_judgment" in cue_set:
            return "reflective_judgment"
        if "opinion_self_style" in cue_set:
            return "self_style"
        if "opinion_habit_preference" in cue_set:
            return "habit_preference"
        if "hypothetical_choice" in cue_set:
            return "hypothetical_choice"
        if "transport_destination_preference" in cue_set:
            return "preference_disclosure"
        if "opinion_preference_like" in cue_set:
            return "preference_disclosure"
        if "broad_opinion_question" in cue_set:
            return "broad_opinion"
        return None

    @staticmethod
    def _infer_pragmatic_cues(
        *,
        intent: Intent,
        normalized: str,
        sentiment: str,
        speech_act: str,
        state: ConversationState | None,
    ) -> list[str]:
        cues: list[str] = []

        if HeuristicIntentClassifier._is_long_form_story_cue(normalized):
            cues.append("long_form_story_share")
            cues.append("creative_writing_context")

        if HeuristicIntentClassifier._is_story_summary_reaction_cue(normalized):
            cues.append("story_summary_reaction")
            cues.append("creative_writing_context")

        if HeuristicIntentClassifier._is_soft_refusal_text(normalized):
            cues.append("soft_refusal")

        if HeuristicIntentClassifier._is_honesty_boundary_text(normalized):
            cues.append("honesty_boundary")

        if HeuristicIntentClassifier._is_format_control_text(normalized):
            cues.append("format_control")

        if HeuristicIntentClassifier._is_proactive_checkin_text(normalized):
            cues.append("proactive_checkin")

        if intent == Intent.ACTIVITY_INVITE or HeuristicIntentClassifier._is_activity_invite_text(normalized):
            cues.append("activity_invite")
            cues.append("proposal_or_invite")

        if speech_act in {"ask", "invite", "suggest"} and any(
            marker in normalized
            for marker in ("혹시", "괜찮으면", "가능하면", "되면", "시간 되면", "실례가 안 되면")
        ):
            cues.append("tentative_request")

        if speech_act == "suggest" and any(
            marker in normalized
            for marker in ("혹시", "괜찮으면", "편하면", "나중에", "시간 되면")
        ):
            cues.append("tentative_suggestion")

        if any(marker in normalized for marker in ("좀", "조금", "약간", "살짝", "것 같", "듯", "겠는데")):
            cues.append("hedging")

        if HeuristicIntentClassifier._is_subdued_positive_feeling_text(normalized):
            cues.append("subdued_positive")

        if HeuristicIntentClassifier._is_broad_knowledge_reflection_text(normalized, speech_act == "ask"):
            cues.append("knowledge_reflection")

        if HeuristicIntentClassifier._is_quiet_weather_feeling_text(normalized):
            cues.append("quiet_weather_feeling")

        if HeuristicIntentClassifier._is_social_awkwardness_text(normalized):
            cues.append("social_awkwardness")

        if HeuristicIntentClassifier._is_low_energy_checkin_text(normalized):
            cues.append("low_energy_checkin")

        body_state_schema = HeuristicIntentClassifier._body_state_schema(normalized, is_question=speech_act == "ask")
        if body_state_schema == "body_signal_interpretation":
            cues.append("body_signal_interpretation")
        elif body_state_schema == "low_energy_support":
            cues.append("low_energy_support")

        if HeuristicIntentClassifier._is_social_aftereffect_text(normalized, state):
            cues.append("social_aftereffect")
            cues.append("reflective_observation")

        if HeuristicIntentClassifier._is_after_social_hollow_text(normalized):
            cues.append("after_social_hollow")
            cues.append("reflective_observation")

        if (
            intent in {Intent.SMALLTALK_OPINION, Intent.MEDIA_RECOMMEND, Intent.MUSIC}
            and HeuristicIntentClassifier._is_preference_or_habit_question_text(normalized, speech_act == "ask")
            and not HeuristicIntentClassifier._is_opinion_self_style_question_text(normalized)
        ):
            cues.append("opinion_habit_preference")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_opinion_self_style_question_text(
            normalized
        ):
            cues.append("opinion_self_style")

        if intent in {Intent.SMALLTALK_OPINION, Intent.MEDIA_RECOMMEND, Intent.MUSIC} and HeuristicIntentClassifier._is_preference_like_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("opinion_preference_like")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_transport_destination_preference_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("transport_destination_preference")
            cues.append("opinion_preference_like")

        if intent == Intent.SMALLTALK_OPINION and (
            HeuristicIntentClassifier._is_structural_choice_prompt_text(normalized)
            or HeuristicIntentClassifier._is_playful_survival_weapon_question_text(
                normalized,
                speech_act == "ask",
            )
        ):
            cues.append("hypothetical_choice")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_preference_criteria_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("opinion_preference_like")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_reflective_judgment_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("opinion_reflective_judgment")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_advice_process_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("opinion_advice_process")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_activity_recommendation_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("activity_recommendation")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_concrete_topic_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("concrete_topic_question")

        if intent in {Intent.SMALLTALK_GENERIC, Intent.SMALLTALK_OPINION} and HeuristicIntentClassifier._is_conversation_topic_suggestion_text(
            normalized
        ):
            cues.append("conversation_topic_suggestion")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_activity_preparation_advice_text(
            normalized
        ):
            cues.append("activity_preparation_advice")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_light_food_recommendation_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("light_food_recommendation")
            cues.append("food_lifestyle")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_relationship_dependency_boundary_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("relationship_boundary")

        if (
            intent == Intent.SMALLTALK_OPINION
            and "memory_boundary" not in cues
            and "relationship_boundary" not in cues
            and "light_food_recommendation" not in cues
            and HeuristicIntentClassifier._is_decision_request_question_text(
                normalized,
                speech_act == "ask",
            )
        ):
            cues.append("opinion_decision_request")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_broad_opinion_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("broad_opinion_question")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_weather_conditioned_activity_opinion_text(
            normalized
        ):
            cues.append("weather_conditioned_activity_opinion")

        if intent == Intent.SMALLTALK_GENERIC and HeuristicIntentClassifier._is_expressive_request_text(normalized):
            cues.append("expressive_request")

        if intent in {Intent.SMALLTALK_GENERIC, Intent.SMALLTALK_FEELING, Intent.SMALLTALK_OPINION} and HeuristicIntentClassifier._is_reflective_observation_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("reflective_observation")

        if intent in {Intent.SMALLTALK_GENERIC, Intent.SMALLTALK_FEELING, Intent.SMALLTALK_OPINION} and HeuristicIntentClassifier._is_relational_interpretation_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("relational_interpretation")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_memory_reality_conflict_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("memory_boundary")
            cues.append("unverified_memory_reference")

        if intent == Intent.SMALLTALK_OPINION and HeuristicIntentClassifier._is_unverified_memory_boundary_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("memory_boundary")
            cues.append("unverified_memory_reference")

        if intent in {Intent.SMALLTALK_GENERIC, Intent.SMALLTALK_FEELING, Intent.SMALLTALK_OPINION} and HeuristicIntentClassifier._is_comparative_reflection_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("comparative_reflection")

        if intent in {Intent.SMALLTALK_GENERIC, Intent.SMALLTALK_FEELING, Intent.SMALLTALK_OPINION} and HeuristicIntentClassifier._is_aesthetic_reflection_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("aesthetic_reflection")

        if intent == Intent.WHY and HeuristicIntentClassifier._is_reason_probe_question_text(
            normalized,
            speech_act == "ask",
        ):
            cues.append("reason_probe")

        if speech_act == "complain" and any(
            marker in normalized
            for marker in ("너무", "진짜", "많이", "장난 아니", "미치겠", "죽겠", "개춥", "개덥")
        ):
            cues.append("complaint_emphasis")

        if speech_act == "ask" and (
            normalized.endswith(("냐", "니", "냐?", "니?"))
            or "ㄱ?" in normalized
            or "어때" in normalized
        ):
            cues.append("casual_probe")

        if intent == Intent.DENY and any(
            marker in normalized
            for marker in ("지금은", "오늘은", "다음에", "나중에", "이건 좀", "그건 좀")
        ):
            cues.append("polite_boundary")

        if HeuristicIntentClassifier._is_soft_handoff_reassurance_text(normalized, state):
            cues.append("soft_handoff_reassurance")
        elif HeuristicIntentClassifier._is_conditional_boundary_text(normalized):
            cues.append("conditional_boundary")
            if "soft_refusal" not in cues:
                cues.append("soft_refusal")
            if "polite_boundary" not in cues:
                cues.append("polite_boundary")
        elif HeuristicIntentClassifier._is_permission_release_text(normalized):
            cues.append("permission_release")
            if "polite_boundary" not in cues:
                cues.append("polite_boundary")

        if HeuristicIntentClassifier._is_self_conscious_check_text(normalized):
            cues.append("self_conscious_check")

        if HeuristicIntentClassifier._is_relationship_check_text(normalized, state):
            cues.append("relationship_check")

        if HeuristicIntentClassifier._is_repair_attempt_text(normalized, state):
            cues.append("repair_attempt")

        if HeuristicIntentClassifier._is_reluctant_acceptance_text(normalized):
            cues.append("reluctant_acceptance")

        if HeuristicIntentClassifier._is_testing_the_waters_text(normalized):
            cues.append("testing_the_waters")

        if HeuristicIntentClassifier._is_contextual_social_followup_text(normalized, state):
            cues.append("contextual_followup")

        if (
            HeuristicIntentClassifier._is_soft_handoff_followup_text(normalized, state)
            and "contextual_followup" not in cues
        ):
            cues.append("contextual_followup")

        if HeuristicIntentClassifier._is_deferred_acceptance_text(normalized):
            cues.append("deferred_acceptance")
            if "polite_boundary" not in cues:
                cues.append("polite_boundary")

        if HeuristicIntentClassifier._is_deferred_rejection_text(normalized):
            cues.append("deferred_rejection")
            if "polite_boundary" not in cues:
                cues.append("polite_boundary")

        if HeuristicIntentClassifier._is_face_saving_retreat_text(normalized):
            cues.append("face_saving_retreat")

        if HeuristicIntentClassifier._is_social_aftereffect_text(normalized, state):
            cues.append("social_aftereffect")

        if HeuristicIntentClassifier._is_social_reconnect_relief_text(normalized, state):
            cues.append("social_reconnect_relief")

        if HeuristicIntentClassifier._is_quiet_feeling_validation_text(normalized, state):
            cues.append("quiet_feeling_validation")

        if HeuristicIntentClassifier._is_teasing_laughter_text(normalized):
            cues.append("teasing_laughter")

        if HeuristicIntentClassifier._is_sarcastic_tease_text(normalized):
            cues.append("sarcastic_tease")

        if intent == Intent.DENY and sentiment == "neutral" and "hedging" in cues and "soft_refusal" not in cues:
            cues.append("indirect_negation")

        return cues


class HybridIntentClassifier:
    """Heuristic(1차 필터) + KcBERT or CharNgram(2차 분류) 하이브리드 분류기."""

    _heuristic_priority_intents = {
        Intent.GREETING,
        Intent.THANKS,
        Intent.WEATHER,
        Intent.PROVIDE_LOCATION,
        Intent.WHY,
        Intent.SEARCH_REQUEST,
        Intent.TIME_DATE,
        Intent.NEWS,
        Intent.REPLY_REQUEST,
        Intent.HELP,
        Intent.WHO_ARE_YOU,
        Intent.MEDIA_RECOMMEND,
        Intent.MUSIC,
        Intent.GAME_INVITE,
        Intent.ACTIVITY_INVITE,
    }
    _heuristic_priority_schemas = {
        "activity_recommendation",
        "honesty_boundary",
        "process_advice",
        "reflective_judgment",
        "soft_decision_advice",
        "weather_conditioned_activity_opinion",
        "activity_invite",
        "long_form_story_share",
        "story_summary_reaction",
        "memory_boundary",
        "light_food_recommendation",
        "relationship_boundary",
    }
    _heuristic_priority_rule_hits = {
        "detector:is_activity_recommendation_question_text",
        "detector:is_activity_invite_text",
        "detector:is_advice_process_question_text",
        "detector:is_format_control_text",
            "detector:is_honesty_boundary_text",
            "detector:is_identity_request_text",
            "detector:is_long_form_story_text",
            "detector:is_story_summary_reaction_text",
            "detector:is_unverified_memory_boundary_question_text",
            "detector:is_light_food_recommendation_question_text",
            "detector:is_relationship_dependency_boundary_question_text",
            "detector:is_media_selection_recommendation_text",
            "detector:is_music_recommendation_question_text",
        "detector:is_preference_criteria_question_text",
        "detector:is_reflective_judgment_question_text",
    }

    def __init__(
        self,
        *,
        heuristic: HeuristicIntentClassifier | None = None,
        bert_model: object | None = None,
        learned_model: CharNgramCentroidModel | None = None,
        min_confidence: float = 0.28,
        meaning_trusted_axes: Iterable[str] | None = None,
    ) -> None:
        self.heuristic = heuristic or HeuristicIntentClassifier()
        self.bert_model = bert_model
        self.learned_model = learned_model
        self.min_confidence = min_confidence
        self.meaning_trusted_axes = (
            frozenset(str(axis) for axis in meaning_trusted_axes)
            if meaning_trusted_axes is not None
            else None
        )
        self.meaning_resolver = MeaningResolver(heuristic=self.heuristic)

    def classify(self, text: str, state: ConversationState) -> MessageFeatures:
        features = self.heuristic.classify(text, state)
        priority_allows_meaning_model = self._priority_feature_allows_meaning_model(features)

        if features.intent in self._heuristic_priority_intents and not priority_allows_meaning_model:
            return self.meaning_resolver.resolve(
                text=text,
                state=state,
                heuristic_features=features,
            )

        if features.question_schema in self._heuristic_priority_schemas:
            return self.meaning_resolver.resolve(
                text=text,
                state=state,
                heuristic_features=features,
            )

        if features.classifier_evidence is not None and any(
            hit in self._heuristic_priority_rule_hits
            for hit in features.classifier_evidence.rule_hits
        ) and not priority_allows_meaning_model:
            return self.meaning_resolver.resolve(
                text=text,
                state=state,
                heuristic_features=features,
            )

        if self.heuristic._expects_location_follow_up(state) and features.intent == Intent.PROVIDE_LOCATION:
            return self.meaning_resolver.resolve(
                text=text,
                state=state,
                heuristic_features=features,
            )

        if self.bert_model is not None:
            model_features = self._classify_with_bert(text, state, features)
            return self.meaning_resolver.resolve(
                text=text,
                state=state,
                heuristic_features=features,
                model_features=model_features,
            )

        if self.learned_model is not None and features.intent == Intent.UNKNOWN:
            model_features = self._classify_with_charngram(text, state, features)
            return self.meaning_resolver.resolve(
                text=text,
                state=state,
                heuristic_features=features,
                model_features=model_features,
            )

        return self.meaning_resolver.resolve(
            text=text,
            state=state,
            heuristic_features=features,
        )

    def _classify_with_bert(
        self,
        text: str,
        state: ConversationState,
        fallback: MessageFeatures,
    ) -> MessageFeatures:
        prediction = self.bert_model.predict(text)
        if hasattr(prediction, "coarse_intent") and hasattr(prediction, "schema") and hasattr(prediction, "speech_act"):
            return self._classify_with_meaning_model_prediction(
                prediction=prediction,
                state=state,
                fallback=fallback,
            )
        if prediction.confidence < self.min_confidence:
            return fallback

        try:
            predicted_intent = Intent(prediction.intent)
        except ValueError:
            return fallback

        if predicted_intent == Intent.PROVIDE_LOCATION:
            return fallback
        if self._conflicts_with_intermediate_concepts(predicted_intent, fallback):
            return fallback

        return self._rebuilt_override_features(
            predicted_intent=predicted_intent,
            state=state,
            fallback=fallback,
            evidence=ClassifierEvidence(
                source="bert",
                chosen_reason="bert prediction cleared the confidence threshold and was not blocked by heuristic safeguards",
                rule_hits=list(fallback.classifier_evidence.rule_hits) if fallback.classifier_evidence else [],
                top_scores=self._top_scores(prediction.scores),
                override_applied=True,
                fallback_source=fallback.classifier_evidence.source if fallback.classifier_evidence else "heuristic",
                fallback_intent=fallback.intent.value,
            ),
        )

    def _classify_with_meaning_model_prediction(
        self,
        *,
        prediction: object,
        state: ConversationState,
        fallback: MessageFeatures,
    ) -> MessageFeatures:
        coarse = getattr(prediction, "coarse_intent")
        domain_pred = getattr(prediction, "domain", None)
        schema_pred = getattr(prediction, "schema")
        speech_pred = getattr(prediction, "speech_act")
        extra_axes = dict(getattr(prediction, "extra_axes", {}) or {})

        predicted_intent = fallback.intent
        coarse_intent_accepted = False
        coarse_label = str(getattr(coarse, "label", "") or "")
        coarse_confidence = float(getattr(coarse, "confidence", 0.0) or 0.0)
        if self._meaning_axis_allowed("coarse_intent") and coarse_label and coarse_confidence >= self.min_confidence:
            try:
                candidate_intent = Intent(coarse_label)
            except ValueError:
                candidate_intent = fallback.intent
            if (
                candidate_intent not in {Intent.PROVIDE_LOCATION, Intent.UNKNOWN}
                and not self._conflicts_with_intermediate_concepts(
                    candidate_intent,
                    fallback,
                )
            ):
                predicted_intent = candidate_intent
                coarse_intent_accepted = True

        schema = fallback.question_schema
        schema_label = str(getattr(schema_pred, "label", "") or "")
        schema_confidence = float(getattr(schema_pred, "confidence", 0.0) or 0.0)
        schema_threshold = self._meaning_schema_threshold(schema_label, self.min_confidence)
        if (
            self._meaning_axis_allowed("schema")
            and schema_label
            and schema_confidence >= schema_threshold
            and not self._blocks_meaning_schema(schema_label=schema_label, fallback=fallback)
        ):
            schema = schema_label
            implied_intent = self._intent_for_meaning_schema(schema_label)
            if (
                implied_intent is not None
                and not coarse_intent_accepted
                and implied_intent != Intent.PROVIDE_LOCATION
                and not self._conflicts_with_intermediate_concepts(implied_intent, fallback)
            ):
                predicted_intent = implied_intent

        intent_overridden = predicted_intent != fallback.intent
        sentiment = self._sentiment_for_intent(predicted_intent) if intent_overridden else fallback.sentiment
        speech_act = fallback.speech_act if not intent_overridden else self.heuristic._infer_speech_act(
            intent=predicted_intent,
            normalized=fallback.normalized,
            is_question=fallback.is_question,
            sentiment=sentiment,
            state=state,
        )
        speech_label = str(getattr(speech_pred, "label", "") or "")
        speech_confidence = float(getattr(speech_pred, "confidence", 0.0) or 0.0)
        if (
            self._meaning_axis_allowed("speech_act")
            and speech_label
            and speech_confidence >= self.min_confidence
            and not self._blocks_meaning_speech_act(
                speech_label=speech_label,
                schema_label=schema_label,
                fallback=fallback,
            )
        ):
            speech_act = speech_label

        requests_external_fact = (
            predicted_intent
            in {
                Intent.WEATHER,
                Intent.SEARCH_REQUEST,
                Intent.NEWS,
                Intent.TIME_DATE,
            }
            if intent_overridden
            else fallback.requests_external_fact
        )
        topic_hint = (
            self.heuristic._infer_topic_hint(
                intent=predicted_intent,
                normalized=fallback.normalized,
                location=fallback.location,
            )
            if intent_overridden
            else fallback.topic_hint
        )
        news_topic = (
            self.heuristic._extract_news_topic(fallback.normalized)
            if intent_overridden and predicted_intent == Intent.NEWS
            else fallback.news_topic
        )
        pragmatic_cues = (
            self.heuristic._infer_pragmatic_cues(
                intent=predicted_intent,
                normalized=fallback.normalized,
                sentiment=sentiment,
                speech_act=speech_act,
                state=state,
            )
            if intent_overridden
            else list(fallback.pragmatic_cues)
        )
        if schema and schema not in pragmatic_cues:
            pragmatic_cues = [*pragmatic_cues, schema]
        pragmatic_cues = list(dict.fromkeys(pragmatic_cues))
        response_needs = self.heuristic._infer_response_needs(
            intent=predicted_intent,
            speech_act=speech_act,
            topic_hint=topic_hint,
            sentiment=sentiment,
            location=fallback.location,
            requests_external_fact=requests_external_fact,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        evidence = ClassifierEvidence(
            source="meaning_model",
            chosen_reason=(
                "multi-head meaning model predicted coarse intent, domain, schema, and speech act; "
                "heuristic safeguards still gate unsafe intent overrides"
            ),
            rule_hits=list(fallback.classifier_evidence.rule_hits) if fallback.classifier_evidence else [],
            top_scores=[
                *self._axis_top_scores("coarse", getattr(coarse, "scores", {}), limit=3),
                *self._axis_top_scores("domain", getattr(domain_pred, "scores", {}) if domain_pred is not None else {}, limit=3),
                *self._axis_top_scores("schema", getattr(schema_pred, "scores", {}), limit=3),
                *self._axis_top_scores("speech_act", getattr(speech_pred, "scores", {}), limit=3),
                *self._extra_axis_top_scores(extra_axes, limit=3),
            ],
            override_applied=predicted_intent != fallback.intent or schema != fallback.question_schema,
            fallback_source=fallback.classifier_evidence.source if fallback.classifier_evidence else "heuristic",
            fallback_intent=fallback.intent.value,
        )
        signals: list[MeaningSignal] = []
        if self._meaning_axis_allowed("coarse_intent"):
            signals.append(
                MeaningSignal(
                    axis="coarse_intent",
                    label=coarse_label or fallback.intent.value,
                    confidence=coarse_confidence,
                    source="meaning_model",
                    evidence=["head=coarse_intent"],
                )
            )
        if self._meaning_axis_allowed("schema"):
            signals.append(
                MeaningSignal(
                    axis="schema",
                    label=schema_label or "none",
                    confidence=schema_confidence,
                    source="meaning_model",
                    evidence=["head=schema"],
                )
            )
        if domain_pred is not None and self._meaning_axis_allowed("domain"):
            signals.append(
                MeaningSignal(
                    axis="domain",
                    label=str(getattr(domain_pred, "label", "") or "general"),
                    confidence=float(getattr(domain_pred, "confidence", 0.0) or 0.0),
                    source="meaning_model",
                    evidence=["head=domain"],
                )
            )
        if self._meaning_axis_allowed("speech_act"):
            signals.append(
                MeaningSignal(
                    axis="speech_act",
                    label=speech_label or "other",
                    confidence=speech_confidence,
                    source="meaning_model",
                    evidence=["head=speech_act"],
                )
            )
        signals.extend(self._extra_axis_signals_from_meaning_prediction(extra_axes))
        signals.extend(self._slot_signals_from_meaning_prediction(prediction))

        meaning_packet = MeaningPacket(
            coarse_intent=predicted_intent.value,
            domain=(
                str(getattr(domain_pred, "label", "") or "") or None
                if domain_pred is not None and self._meaning_axis_allowed("domain")
                else None
            ),
            schema=schema,
            speech_act=speech_act,
            slots=dict(getattr(prediction, "slots", {}) or {}),
            pragmatic_cues=pragmatic_cues,
            signals=signals,
            resolver="multihead_meaning_model_v1",
        )

        return MessageFeatures(
            content=fallback.content,
            normalized=fallback.normalized,
            intent=predicted_intent,
            sentiment=sentiment,
            is_question=fallback.is_question,
            location=fallback.location,
            requests_external_fact=requests_external_fact,
            speech_act=speech_act,
            topic_hint=topic_hint,
            news_topic=news_topic,
            question_schema=schema,
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
            meaning_packet=meaning_packet,
        )

    @staticmethod
    def _slot_signals_from_meaning_prediction(prediction: object) -> list[MeaningSignal]:
        signals: list[MeaningSignal] = []
        for span in list(getattr(prediction, "slot_spans", []) or []):
            label = str(getattr(span, "label", "") or "").strip()
            value = str(getattr(span, "value", "") or "").strip()
            if not label or not value:
                continue
            confidence = float(getattr(span, "confidence", 0.0) or 0.0)
            start = getattr(span, "start", None)
            end = getattr(span, "end", None)
            evidence = [value]
            if start is not None and end is not None:
                evidence.append(f"span={start}:{end}")
            signals.append(
                MeaningSignal(
                    axis="slot",
                    label=label,
                    confidence=confidence,
                    source="meaning_model_slot_head",
                    evidence=evidence,
                )
            )

        slots = dict(getattr(prediction, "slots", {}) or {})
        seen = {(signal.label, signal.evidence[0]) for signal in signals if signal.evidence}
        for label, raw_value in slots.items():
            for value in str(raw_value or "").split("|"):
                cleaned = value.strip()
                if not cleaned or (str(label), cleaned) in seen:
                    continue
                signals.append(
                    MeaningSignal(
                        axis="slot",
                        label=str(label),
                        confidence=0.65,
                        source="meaning_model_slots",
                        evidence=[cleaned],
                    )
                )
        return signals

    def _extra_axis_signals_from_meaning_prediction(
        self,
        extra_axes: dict[str, object],
    ) -> list[MeaningSignal]:
        signals: list[MeaningSignal] = []
        for axis, prediction in extra_axes.items():
            label = str(getattr(prediction, "label", "") or "").strip()
            if not axis or not label or not self._meaning_axis_allowed(str(axis)):
                continue
            signals.append(
                MeaningSignal(
                    axis=str(axis),
                    label=label,
                    confidence=float(getattr(prediction, "confidence", 0.0) or 0.0),
                    source="meaning_model",
                    evidence=[f"head={axis}"],
                )
            )
        return signals

    def _meaning_axis_allowed(self, axis: str) -> bool:
        return self.meaning_trusted_axes is None or axis in self.meaning_trusted_axes

    @classmethod
    def _extra_axis_top_scores(
        cls,
        extra_axes: dict[str, object],
        *,
        limit: int = 3,
    ) -> list[ScoredLabel]:
        scores: list[ScoredLabel] = []
        for axis, prediction in extra_axes.items():
            scores.extend(cls._axis_top_scores(str(axis), getattr(prediction, "scores", {}) or {}, limit=limit))
        return scores

    def _classify_with_charngram(
        self,
        text: str,
        state: ConversationState,
        fallback: MessageFeatures,
    ) -> MessageFeatures:
        prediction, used_context = self._best_charngram_prediction(text, state)
        if prediction.confidence < self.min_confidence:
            return fallback

        try:
            predicted_intent = Intent(prediction.intent)
        except ValueError:
            return fallback

        if predicted_intent in {Intent.UNKNOWN, Intent.PROVIDE_LOCATION}:
            return fallback
        if self._conflicts_with_intermediate_concepts(predicted_intent, fallback):
            return fallback

        return self._rebuilt_override_features(
            predicted_intent=predicted_intent,
            state=state,
            fallback=fallback,
            evidence=ClassifierEvidence(
                source="charngram",
                chosen_reason=(
                    "context-enriched char-ngram centroid prediction cleared the confidence threshold and replaced an unknown heuristic result"
                    if used_context
                    else "char-ngram centroid prediction cleared the confidence threshold and replaced an unknown heuristic result"
                ),
                rule_hits=(
                    list(fallback.classifier_evidence.rule_hits) + (["context_enriched_charngram"] if used_context else [])
                    if fallback.classifier_evidence
                    else (["context_enriched_charngram"] if used_context else [])
                ),
                top_scores=self._top_scores(prediction.scores),
                override_applied=True,
                fallback_source=fallback.classifier_evidence.source if fallback.classifier_evidence else "heuristic",
                fallback_intent=fallback.intent.value,
            ),
        )

    def _best_charngram_prediction(
        self,
        text: str,
        state: ConversationState,
    ) -> tuple[object, bool]:
        assert self.learned_model is not None
        best_prediction = self.learned_model.predict(text)
        used_context = False
        best_confidence = float(getattr(best_prediction, "confidence", 0.0))

        for enriched_text in self._build_charngram_context_inputs(text, state):
            prediction = self.learned_model.predict(enriched_text)
            confidence = float(getattr(prediction, "confidence", 0.0))
            if confidence > best_confidence:
                best_prediction = prediction
                best_confidence = confidence
                used_context = True

        return best_prediction, used_context

    @staticmethod
    def _build_charngram_context_inputs(text: str, state: ConversationState) -> list[str]:
        current = " ".join(str(text or "").strip().split())
        if not current or not state.recent_turns:
            return []

        segments: list[str] = []
        for turn in state.recent_turns[-2:]:
            user_text = " ".join(str(turn.user_text or "").strip().split())
            bot_text = " ".join(str(turn.bot_text or "").strip().split())
            if user_text:
                segments.append(user_text)
            if bot_text:
                segments.append(bot_text)

        if not segments:
            return []

        seen: set[str] = set()
        inputs: list[str] = []
        for window_size in (2, 4):
            window = segments[-window_size:]
            if not window:
                continue
            enriched = " ".join([*window, current]).strip()
            if enriched and enriched != current and enriched not in seen:
                inputs.append(enriched)
                seen.add(enriched)
        return inputs

    def _rebuilt_override_features(
        self,
        *,
        predicted_intent: Intent,
        state: ConversationState,
        fallback: MessageFeatures,
        evidence: ClassifierEvidence,
    ) -> MessageFeatures:
        sentiment = self._sentiment_for_intent(predicted_intent)
        requests_external_fact = predicted_intent in {
            Intent.WEATHER,
            Intent.SEARCH_REQUEST,
            Intent.NEWS,
            Intent.TIME_DATE,
        }
        topic_hint = self.heuristic._infer_topic_hint(
            intent=predicted_intent,
            normalized=fallback.normalized,
            location=fallback.location,
        )
        news_topic = self.heuristic._extract_news_topic(fallback.normalized) if predicted_intent == Intent.NEWS else None
        speech_act = self.heuristic._infer_speech_act(
            intent=predicted_intent,
            normalized=fallback.normalized,
            is_question=fallback.is_question,
            sentiment=sentiment,
            state=state,
        )
        pragmatic_cues = self.heuristic._infer_pragmatic_cues(
            intent=predicted_intent,
            normalized=fallback.normalized,
            sentiment=sentiment,
            speech_act=speech_act,
            state=state,
        )
        response_needs = self.heuristic._infer_response_needs(
            intent=predicted_intent,
            speech_act=speech_act,
            topic_hint=topic_hint,
            sentiment=sentiment,
            location=fallback.location,
            requests_external_fact=requests_external_fact,
            pragmatic_cues=pragmatic_cues,
            state=state,
        )
        question_schema = self.heuristic._infer_question_schema(
            intent=predicted_intent,
            pragmatic_cues=pragmatic_cues,
        )

        return MessageFeatures(
            content=fallback.content,
            normalized=fallback.normalized,
            intent=predicted_intent,
            sentiment=sentiment,
            is_question=fallback.is_question,
            location=fallback.location,
            requests_external_fact=requests_external_fact,
            speech_act=speech_act,
            topic_hint=topic_hint,
            news_topic=news_topic,
            question_schema=question_schema,
            response_needs=response_needs,
            pragmatic_cues=pragmatic_cues,
            classifier_evidence=evidence,
        )

    @staticmethod
    def _sentiment_for_intent(intent: Intent) -> str:
        if intent in {Intent.GREETING, Intent.THANKS, Intent.CONFIRM, Intent.LAUGH}:
            return "positive"
        if intent == Intent.HOSTILE:
            return "negative"
        return "neutral"

    @staticmethod
    def _top_scores(scores: dict[str, float], limit: int = 5) -> list[ScoredLabel]:
        return [
            ScoredLabel(label=label, score=float(score))
            for label, score in list(scores.items())[:limit]
        ]

    @staticmethod
    def _axis_top_scores(axis: str, scores: dict[str, float], limit: int = 3) -> list[ScoredLabel]:
        return [
            ScoredLabel(label=f"{axis}:{label}", score=float(score))
            for label, score in list(scores.items())[:limit]
        ]

    @staticmethod
    def _meaning_schema_threshold(schema_label: str, min_confidence: float) -> float:
        if schema_label == "reason_probe":
            return min(min_confidence, 0.15)
        if schema_label:
            return min(min_confidence, 0.17)
        return min_confidence

    @staticmethod
    def _intent_for_meaning_schema(schema_label: str) -> Intent | None:
        if schema_label == "activity_invite":
            return Intent.ACTIVITY_INVITE
        if schema_label == "reason_probe":
            return Intent.WHY
        if schema_label in {
            "relational_interpretation",
            "comparative_reflection",
            "reflective_observation",
            "emotional_disclosure",
        }:
            return Intent.SMALLTALK_FEELING
        if schema_label in {
            "absurd_hypothetical",
            "activity_recommendation",
            "weather_conditioned_activity_opinion",
            "soft_decision_advice",
            "process_advice",
            "reflective_judgment",
            "preference_disclosure",
            "habit_preference",
            "hypothetical_choice",
            "self_style",
            "honesty_boundary",
            "conversation_topic_suggestion",
            "activity_preparation_advice",
            "expressive_request",
            "aesthetic_reflection",
            "broad_opinion",
            "social_mishap",
            "long_form_story_share",
            "story_summary_reaction",
            "memory_boundary",
            "light_food_recommendation",
            "relationship_boundary",
        }:
            return Intent.SMALLTALK_OPINION
        return None

    def _priority_feature_allows_meaning_model(self, features: MessageFeatures) -> bool:
        if self.bert_model is None:
            return False
        if features.intent == Intent.SEARCH_REQUEST and self._could_be_reason_probe(features):
            return True
        if features.intent in {
            Intent.WEATHER,
            Intent.SEARCH_REQUEST,
            Intent.MUSIC,
            Intent.MEDIA_RECOMMEND,
        } and self._could_be_persona_or_hypothetical_structure_question(features):
            return True
        return False

    @staticmethod
    def _could_be_reason_probe(features: MessageFeatures) -> bool:
        normalized = features.normalized
        if features.question_schema == "reason_probe" or "reason_probe" in features.pragmatic_cues:
            return True
        return any(marker in normalized for marker in ("근거", "이유", "왜 그렇게", "왜 그런", "판단"))

    @staticmethod
    def _could_be_persona_or_hypothetical_structure_question(features: MessageFeatures) -> bool:
        if not features.is_question:
            return False
        normalized = f" {features.normalized} "
        structural_markers = (
            "만약",
            "라면",
            "다면",
            "쏜다면",
            "저주",
            " vs ",
            "심정",
            "기분",
            "할래",
            "고 싶",
            "한 마디",
            "뭐라고",
            "무슨 말을",
            "어떻게 수습",
            "메뉴는 뭐야",
        )
        if not any(marker in normalized for marker in structural_markers):
            return False
        anchor_markers = (
            "내가",
            "나에게",
            "나한테",
            "네가",
            "너",
            "너의",
            "평생",
            "하루 동안",
            "과거",
            "미래",
            "엘리베이터",
            "길",
            "모르는 사람",
            "주인",
            "사람",
            "때마다",
            "걸린",
            "질문",
            "한 턱",
            "메뉴",
        )
        return any(marker in normalized for marker in anchor_markers)

    def _blocks_meaning_speech_act(
        self,
        *,
        speech_label: str,
        schema_label: str,
        fallback: MessageFeatures,
    ) -> bool:
        if (
            fallback.question_schema == "expressive_request"
            and fallback.speech_act == "ask"
            and speech_label != "ask"
        ):
            return True
        if (
            speech_label == "inform"
            and fallback.intent == Intent.SMALLTALK_FEELING
            and fallback.speech_act == "complain"
            and "empathy" in fallback.response_needs
        ):
            return True
        if speech_label != "ask":
            return False
        return not self._meaning_prediction_allows_ask(schema_label=schema_label, fallback=fallback)

    @staticmethod
    def _blocks_meaning_schema(*, schema_label: str, fallback: MessageFeatures) -> bool:
        if fallback.question_schema == "expressive_request" and schema_label != "expressive_request":
            return True
        if (
            fallback.question_schema == "conversation_topic_suggestion"
            and schema_label != "conversation_topic_suggestion"
        ):
            return True
        if (
            fallback.question_schema == "activity_preparation_advice"
            and schema_label != "activity_preparation_advice"
        ):
            return True
        if fallback.question_schema == "hypothetical_choice" and schema_label != "hypothetical_choice":
            return True
        return False

    @staticmethod
    def _meaning_prediction_allows_ask(*, schema_label: str, fallback: MessageFeatures) -> bool:
        if fallback.is_question:
            return True
        if schema_label not in {
            "activity_recommendation",
            "weather_conditioned_activity_opinion",
            "soft_decision_advice",
            "process_advice",
            "conversation_topic_suggestion",
            "activity_preparation_advice",
            "reflective_judgment",
            "reason_probe",
            "expressive_request",
            "broad_opinion",
            "aesthetic_reflection",
            "story_summary_reaction",
            "memory_boundary",
            "light_food_recommendation",
            "relationship_boundary",
            "relational_interpretation",
            "comparative_reflection",
            "reflective_observation",
        }:
            return False
        return any(
            marker in fallback.normalized
            for marker in (
                "골라",
                "고르",
                "추천",
                "뭐",
                "무엇",
                "어떤",
                "어떻게",
                "왜",
                "근거",
                "이유",
                "괜찮",
                "될까",
                "할까",
                "좋을까",
                "생각해",
                "바꿔",
                "말해",
                "알려",
                "정리",
            )
        )

    @staticmethod
    def _conflicts_with_intermediate_concepts(
        predicted_intent: Intent,
        fallback: MessageFeatures,
    ) -> bool:
        comparison_emotion_markers = ("씁쓸", "서운", "허탈", "허무")

        if (
            fallback.intent == Intent.SMALLTALK_FEELING
            and "empathy" in fallback.response_needs
            and any(marker in fallback.normalized for marker in comparison_emotion_markers)
            and predicted_intent != Intent.SMALLTALK_FEELING
        ):
            return True
        if (
            predicted_intent == Intent.HELP
            and fallback.intent == Intent.SMALLTALK_FEELING
            and "empathy" in fallback.response_needs
        ):
            return True
        if (
            fallback.intent == Intent.SMALLTALK_FEELING
            and "empathy" in fallback.response_needs
            and predicted_intent in {Intent.SMALLTALK_GENERIC, Intent.SMALLTALK_OPINION}
        ):
            return True
        if fallback.question_schema == "expressive_request" and predicted_intent != fallback.intent:
            return True
        if (
            "soft_refusal" in fallback.pragmatic_cues
            and fallback.intent == Intent.DENY
            and predicted_intent != Intent.DENY
        ):
            return True
        if (
            any(
                cue in fallback.pragmatic_cues
                for cue in {
                    "permission_release",
                    "self_conscious_check",
                    "tentative_suggestion",
                    "relationship_check",
                    "reluctant_acceptance",
                    "testing_the_waters",
                    "face_saving_retreat",
                    "deferred_acceptance",
                    "deferred_rejection",
                    "teasing_laughter",
                    "sarcastic_tease",
                    "repair_attempt",
                }
            )
            and predicted_intent in {Intent.HELP, Intent.REPLY_REQUEST, Intent.HOSTILE, Intent.MEDIA_RECOMMEND}
        ):
            return True
        if (
            any(
                cue in fallback.pragmatic_cues
                for cue in {
                    "self_conscious_check",
                    "relationship_check",
                    "reluctant_acceptance",
                    "tentative_suggestion",
                    "testing_the_waters",
                    "face_saving_retreat",
                    "deferred_acceptance",
                    "deferred_rejection",
                    "teasing_laughter",
                    "sarcastic_tease",
                    "repair_attempt",
                }
            )
            and predicted_intent != fallback.intent
        ):
            return True
        if (
            predicted_intent == Intent.WEATHER
            and fallback.topic_hint == "weather"
            and fallback.speech_act in {"inform", "complain"}
            and "grounding" not in fallback.response_needs
        ):
            return True
        if (
            fallback.intent == Intent.SMALLTALK_OPINION
            and fallback.is_question
            and predicted_intent != Intent.SMALLTALK_OPINION
        ):
            return True
        if "honesty_boundary" in fallback.pragmatic_cues and predicted_intent not in {
            Intent.SEARCH_REQUEST,
            Intent.NEWS,
            Intent.WEATHER,
            Intent.SMALLTALK_OPINION,
            Intent.SMALLTALK_FEELING,
        }:
            return True
        if (
            fallback.intent == Intent.SMALLTALK_OPINION
            and fallback.question_schema == "activity_recommendation"
            and predicted_intent != Intent.SMALLTALK_OPINION
        ):
            return True
        if (
            fallback.intent == Intent.SMALLTALK_OPINION
            and fallback.question_schema
            in {
                "activity_recommendation",
                "activity_preparation_advice",
                "conversation_topic_suggestion",
                "process_advice",
                "reflective_judgment",
                "soft_decision_advice",
                "preference_disclosure",
                "habit_preference",
                "self_style",
            }
            and predicted_intent not in {Intent.SMALLTALK_OPINION, Intent.MUSIC, Intent.MEDIA_RECOMMEND}
        ):
            return True
        if (
            fallback.intent == Intent.SMALLTALK_FEELING
            and "subdued_positive" in fallback.pragmatic_cues
            and predicted_intent != Intent.SMALLTALK_FEELING
        ):
            return True
        if (
            predicted_intent == Intent.WEATHER
            and HeuristicIntentClassifier._has_weather_surface_text(fallback.normalized)
            and not HeuristicIntentClassifier._looks_like_weather_request(
                fallback.normalized,
                fallback.is_question,
            )
        ):
            return True
        return False
