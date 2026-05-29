from __future__ import annotations

import unittest

from predictive_bot.core.draft_nlg import build_black_draft_utterance
from predictive_bot.core.models import ActionType, Intent, MessageFeatures, PhrasingPlan, ResponsePlan


class DraftNlgTests(unittest.TestCase):
    STALE_FALLBACK_SNIPPETS = (
        "무리하게 밀 필요",
        "길게 키우진 않을게",
        "그 생각은 이해돼",
        "그 선택은 부담이 너무 크지 않으면",
        "그쪽은 나는 꽤 맞는 편",
        "상황은 받아둘게",
        "주어진 초안",
        "받아둘게",
        "그 말은 여기서 낮게",
        "나는 Black의 한국어 문장",
        "어느 쪽 기준인지",
        "사실 확인 전엔",
    )
    INTERNAL_LABEL_SNIPPETS = (
        "opinion_",
        "activity_",
        "relationship_",
        "direct_preference_disclosure",
        "SMALLTALK",
        "draft_frame",
    )

    def assert_clean_black_draft(
        self,
        draft: dict[str, object],
        *,
        direct_reason_prefixes: tuple[str, ...] = (),
    ) -> None:
        reply = str(draft["draft_reply"])
        self.assertGreater(len(reply.strip()), 10)
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        for phrase in self.STALE_FALLBACK_SNIPPETS:
            self.assertNotIn(phrase, reply)
        for phrase in self.INTERNAL_LABEL_SNIPPETS:
            self.assertNotIn(phrase, reply)
        if direct_reason_prefixes:
            reason = str(draft["direct_surface_reason"])
            self.assertTrue(reason.startswith(direct_reason_prefixes), reason)

    def assert_contains_any(self, reply: str, fragments: tuple[str, ...]) -> None:
        self.assertTrue(any(fragment in reply for fragment in fragments), reply)

    def test_mixed_social_emotion_uses_specific_relationship_detail(self) -> None:
        cases = [
            (
                "친한 친구가 약속을 또 당일에 취소해서 서운한데 말하기도 애매해.",
                "korean_daily_mixed_social_appointment_cancel_hurt",
                ("당일 약속 취소", "서운한 게 맞아", "빨리 말해줘"),
            ),
            (
                "친구가 내 말을 대충 듣고 폰만 봐서 좀 서운해.",
                "korean_daily_foundation_close_person_not_listening",
                ("대충 듣는 느낌", "진지하게 들어줬으면"),
            ),
            (
                "답장이 늦는 건 알겠는데 나만 신경 쓰는 것 같아서 거리감 느껴져.",
                "salience_dominant_context_direct_reply",
                ("답장 온도차", "나만 계속 확인", "마음이 피곤"),
            ),
            (
                "조금 서운한데 말하면 예민한 사람 될까 봐 참는 중이야.",
                "korean_daily_foundation_small_hurt_say_softly",
                ("서운한데 참는 중", "낮게 꺼내도 돼"),
            ),
        ]
        for text, expected_reason, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question="?" in text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assert_clean_black_draft(
                    draft,
                    direct_reason_prefixes=("korean_daily_", "salience_"),
                )
                self.assertEqual(draft["direct_surface_reason"], expected_reason)
                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn("거리감는", draft["draft_reply"])
                self.assertNotIn("플레이리스트", draft["draft_reply"])

    def test_korean_daily_contextual_life_scene_compositions(self) -> None:
        cases = [
            (
                "오늘 아침에 나오는데 유독 하늘이 맑더라. 날씨 좋으니까 출근하기 더 싫어짐.",
                "korean_daily_more_weather_clear_sky_commute_dread",
                ("하늘은 놀러 가라고", "출근하기 더 싫어"),
                "weather_season",
                "weather_clear_sky_commute_dread",
            ),
            (
                "요즘 낮에는 살짝 더운데 아침저녁으로는 쌀쌀해서 옷 입기가 진짜 애매해.",
                "korean_daily_more_weather_layered_clothes_ambiguity",
                ("아침저녁은 쌀쌀", "얇은 겉옷"),
                "weather_season",
                "weather_layered_clothes_ambiguity",
            ),
            (
                "오늘 미세먼지 역대급이라 목 아프다. 다들 나갈 때 마스크 꼭 챙겨라.",
                "korean_daily_more_weather_fine_dust_mask_survival",
                ("미세먼지", "마스크랑 물"),
                "weather_season",
                "weather_fine_dust_mask_survival",
            ),
            (
                "회사 근처에 새로 생긴 파스타집 웨이팅 장난 아니더라. 결국 패스하고 국밥 먹음.",
                "korean_daily_more_food_pasta_waiting_gukbap_turn",
                ("파스타집 웨이팅", "국밥"),
                "food_lifestyle",
                "food_pasta_waiting_gukbap_turn",
            ),
            (
                "오랜만에 고등학교 동창한테 연락 왔는데 반가우면서도 묘하게 어색하더라.",
                "korean_daily_more_social_old_classmate_contact_awkward",
                ("반가운데 어색", "대화 감각"),
                "social_relationship",
                "social_old_classmate_contact_awkward",
            ),
            (
                "내 고민 진지하게 털어놨는데 상대방이 영혼 없이 리액션 하면 은근히 서운함 들지.",
                "korean_daily_more_social_soulless_reaction_hurt",
                ("리액션이 비어", "서운한 게 맞아"),
                "social_relationship",
                "social_soulless_reaction_hurt",
            ),
            (
                "주말에 하려고 했던 일 되게 많았는데, 정신 차려 보니 침대에서 유튜브만 보다 하루 다 감.",
                "korean_daily_more_home_weekend_bed_youtube_all_day",
                ("침대에서 유튜브", "몸은 쉰"),
                "home_life",
                "home_weekend_bed_youtube_all_day",
            ),
            (
                "자취방 형광등 나갔는데 바꾸기 귀찮아서 스탠드 조명 하나로 일주일째 버티는 중.",
                "korean_daily_more_home_fluorescent_stand_lamp_lazy",
                ("형광등 나갔", "스탠드"),
                "home_life",
                "home_fluorescent_stand_lamp_lazy",
            ),
            (
                "카톡 답장 머릿속으로 다 써놓고 전송 안 눌러서 본의 아니게 하루 동안 읽씹한 사람 됨.",
                "korean_daily_more_social_head_reply_not_sent",
                ("머릿속으로", "전송 안 누르는"),
                "social_relationship",
                "social_head_reply_not_sent",
            ),
            (
                "할 일 목록 짜는 데만 완벽하게 한 시간 쓰고, 정작 시작도 전에 지쳐서 눕는 엔딩.",
                "korean_daily_more_growth_todo_planning_exhaustion",
                ("할 일 목록", "계획형 함정"),
                "self_growth",
                "growth_todo_planning_exhaustion",
            ),
        ]

        for text, expected_reason, expected_fragments, expected_domain, expected_detail in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question="?" in text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assert_clean_black_draft(
                    draft,
                    direct_reason_prefixes=("korean_daily_more_",),
                )
                self.assertEqual(draft["direct_surface_reason"], expected_reason)
                self.assertEqual(draft["draft_domain"], expected_domain)
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                semantic_frame = draft.get("semantic_frame") or {}
                self.assertEqual(semantic_frame.get("schema"), "contextual_reaction")
                self.assertEqual(semantic_frame.get("draft_frame_family"), "situational_context")
                self.assertEqual(semantic_frame.get("domain"), expected_domain)
                self.assert_contains_any(str(draft["draft_reply"]), expected_fragments)
                self.assertNotIn("요리 브이로그", draft["draft_reply"])
                self.assertNotIn("월요병 디버프", draft["draft_reply"])

    def test_korean_daily_contextual_life_scene_metadata_for_existing_reasons(self) -> None:
        cases = [
            (
                "갑자기 비 온다더니 해 쨍쨍한 거 뭐냐. 가방에 든 우산 하루 종일 짐만 됐네.",
                "korean_daily_expansion_wrong_rain_forecast_umbrella_burden",
                "weather_season",
                "wrong_rain_forecast_umbrella_burden",
            ),
            (
                "유튜브 쇼츠 넘기다 보니 3시간 지나 있음. 알고리즘 이거 진짜 시간 도둑이네.",
                "korean_daily_shortform_time_sink",
                "digital_habit",
                "shortform_time_sink",
            ),
            (
                "퇴근 10분 전에 메일 온 거 보고 모니터 조용히 끔. 내일의 내가 알아서 하겠지 뭐.",
                "korean_daily_expansion_after_work_email_leave_for_tomorrow",
                "work_school",
                "after_work_email_leave_for_tomorrow",
            ),
            (
                "로컬 vLLM 서버 구동해 놓고 디스코드 봇으로 연동했더니 스마트폰으로도 테스트 가능해서 신세계임.",
                "korean_daily_expansion_local_vllm_discord_mobile_test",
                "ai_companion_building",
                "local_vllm_discord_mobile_test",
            ),
            (
                "인스타에서 미드센추리 모던 감성 조명 보고 홀린 듯 결제했는데 우리 집 인테리어랑 따로 놀아서 처치 곤란.",
                "korean_daily_expansion_interior_midcentury_lamp_mismatch",
                "room_interior_scene",
                "interior_midcentury_lamp_mismatch",
            ),
            (
                "벽에 타공판 달아서 헤드셋이랑 자주 쓰는 가젯들 걸어두니까 묘하게 작업실 분위기 나고 좋더라.",
                "korean_daily_expansion_interior_pegboard_gadget_workroom_mood",
                "room_interior_scene",
                "interior_pegboard_gadget_workroom_mood",
            ),
            (
                "오전 내내 모니터 멍하니 보다가 눈 뻑뻑해서 인공눈물 넣었는데, 흘러내려서 화장 다 지워짐.",
                "korean_daily_slang_eye_drops_makeup_meltdown",
                "body_beauty",
                "eye_drops_makeup_meltdown",
            ),
            (
                "학교/회사 갈 때 입을 옷 없어서 옷장 앞에서 15분 동안 서성이다가 결국 어제 입은 거 또 입고 나옴.",
                "korean_daily_expansion_closet_no_outfit_repeat",
                "fashion_life",
                "closet_no_outfit_repeat",
            ),
            (
                "오늘 하루 종일 생산적인 일 딱 하나 함: 영양제 챙겨 먹기. 건강은 챙겼으니 됐다.",
                "korean_daily_expansion_one_productive_thing_vitamins",
                "self_care_health",
                "one_productive_thing_vitamins",
            ),
        ]

        for text, expected_reason, expected_domain, expected_detail in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question="?" in text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assert_clean_black_draft(
                    draft,
                    direct_reason_prefixes=("korean_daily_",),
                )
                self.assertEqual(draft["direct_surface_reason"], expected_reason)
                self.assertEqual(draft["draft_domain"], expected_domain)
                self.assertEqual(draft["draft_frame_detail"], expected_detail)

    def test_social_return_draft_uses_return_anchor(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="한동안 말 안 하다가 그냥 다시 와봤어.",
                normalized="한동안 말 안 하다가 그냥 다시 와봤어.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="한동안 말 없다가 다시 옴",
                must_include=["오랜만", "다시"],
                followup_policy="no_followup",
                notes=["social_return_acknowledgement"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("오랜만", draft["draft_reply"])
        self.assertIn("다시", draft["draft_reply"])
        self.assertIn("preserve the draft meaning", draft["rewrite_rules"])

    def test_social_return_must_include_does_not_create_fragment_prefix(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="오랜만에 다시 말 걸어본다.",
                normalized="오랜만에 다시 말 걸어본다.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="오랜만에 다시 말 걸어본다.",
                must_include=["다시 왔", "오랜만"],
                followup_policy="one_direct_followup",
                notes=["social_return_acknowledgement"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("오랜만", draft["draft_reply"])
        self.assertIn("다시 와", draft["draft_reply"])
        self.assertNotIn("다시 왔.", draft["draft_reply"])

    def test_activity_recommendation_draft_uses_options(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="바다에서 무엇을 하고 놀면 좋을까?",
                normalized="바다에서 무엇을 하고 놀면 좋을까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="practical_activity_recommendation",
                anchor="바다 놀이",
                must_include=["바다 놀이", "물놀이", "모래사장 산책", "사진 찍기"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("물놀이", draft["draft_reply"])
        self.assertIn("모래사장 산책", draft["draft_reply"])
        self.assertIn("바다 놀이면", draft["draft_reply"])
        self.assertIn("모래사장 산책이", draft["draft_reply"])
        self.assertIn("사진 찍기도 좋아", draft["draft_reply"])
        self.assertNotIn("놀이이면", draft["draft_reply"])
        self.assertNotIn("산책가", draft["draft_reply"])
        self.assertNotIn("괜찮아", draft["draft_reply"])
        self.assertEqual(draft["source"], "black_phrase_bank_v1")

    def test_long_form_story_draft_uses_story_reaction_not_surface_weather(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="비 내리는 2084년 서울에서 이안은 오메가 코퍼레이션의 메모리 칩을 들고 있었다. 세라는 코어 안에 갇혀 있었다.",
                normalized="비 내리는 2084년 서울에서 이안은 오메가 코퍼레이션의 메모리 칩을 들고 있었다. 세라는 코어 안에 갇혀 있었다.",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="long_form_story_reaction",
                anchor="creative_writing",
                must_include=["비", "구름", "오메가"],
                followup_policy="no_followup",
                notes=["long_form_story_share", "do_not_route_surface_keywords"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("장문 서사", draft["draft_reply"])
        self.assertIn("오메가", draft["draft_reply"])
        self.assertNotIn("구름", draft["draft_reply"])
        self.assertNotIn("날씨", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "long_form_story_direct_reply")

    def test_short_story_summary_reaction_overrides_surface_weather(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="짧은 감상만 해줘. 비 오는 도시에서 주인공이 잃어버린 기억을 되찾고, 결국 가장 소중한 사람을 구하지 못한 채 진실만 세상에 남기는 이야기야.",
                normalized="짧은 감상만 해줘. 비 오는 도시에서 주인공이 잃어버린 기억을 되찾고, 결국 가장 소중한 사람을 구하지 못한 채 진실만 세상에 남기는 이야기야.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_light",
                anchor="",
                must_include=[],
                followup_policy="no_followup",
                notes=[],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("상실을 감수하고 진실만 남기는", draft["draft_reply"])
        self.assertNotIn("날씨", draft["draft_reply"])
        self.assertNotIn("말은 받았어", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "story_summary_reaction_direct_reply")

    def test_unverified_memory_reference_does_not_fabricate_specific_memory(self) -> None:
        cases = [
            (
                "우리 작년 여름에 제주도 갔을 때 먹었던 그 고기국수집 이름이 뭐였지?",
                ("확인되는 기억이 없어", "단서"),
            ),
            (
                "저번에 네가 빌려준 그 책, 내가 실수로 커피를 엎질렀어.",
                ("기록은 없어", "새로 사주는"),
            ),
            (
                "너 커피 마시면 밤에 못 잔다면서 지금 이 시간에 아메리카노 마셔도 괜찮아?",
                ("실제로 커피를 마시진 않지만", "잠부터"),
            ),
            (
                "단거 별로 안 좋아하는데 웬일로 마카롱을 다 샀어?",
                ("단 걸 별로 안 좋아한다는 전제", "누굴 주려는"),
            ),
            (
                "지난번 회식 때 네가 나 대신 변명해 줬잖아. 그때 넌 내 상황을 어떻게 그렇게 정확히 알고 있었던 거야?",
                ("실제로 기억한다고 하진 않을게", "꾸미지도 않겠다"),
            ),
        ]
        for text, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=["like"] if "마카롱" in text else [],
                        followup_policy="no_followup",
                        notes=["unverified_memory_reference", "do_not_fabricate_memory"],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn("길게 키우진 않을게", draft["draft_reply"])
                self.assertNotIn("like.", draft["draft_reply"])
                self.assertNotIn("제주도 갔을 때 먹었던 집은", draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], "unverified_memory_boundary")
                if "회식" in text:
                    self.assertNotIn("회식", draft["draft_reply"])
                    self.assertNotIn("변명", draft["draft_reply"])

    def test_grounded_memory_reference_uses_retrieved_memory(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="너 커피 마시면 밤에 못 잔다면서 지금 이 시간에 아메리카노 마셔도 괜찮아?",
                normalized="너 커피 마시면 밤에 못 잔다면서 지금 이 시간에 아메리카노 마셔도 괜찮아?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_opinion",
                anchor="커피",
                must_include=[],
                followup_policy="no_followup",
                notes=[
                    "grounded_memory_reference",
                    "memory:나는 밤에 커피 마시면 잠을 잘 못 자.",
                ],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("기억하고 있어", draft["draft_reply"])
        self.assertIn("커피", draft["draft_reply"])
        self.assertIn("잠부터", draft["draft_reply"])
        self.assertNotIn("확인되는 기억이 없어", draft["draft_reply"])
        self.assertNotIn("기억에는", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "grounded_memory_natural_reply")

    def test_grounded_memory_reference_does_not_turn_generic_dream_into_hawaii(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="예전에 네가 포기했던 그 꿈 말이야, 요즘 다시 시작해 볼까 고민하는 것 같던데, 내가 어떻게 도와주면 될까?",
                normalized="예전에네가포기했던그꿈말이야요즘다시시작해볼까고민하는것같던데내가어떻게도와주면될까",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="conditional_go_or_no_go",
                anchor="",
                must_include=[],
                followup_policy="no_followup",
                notes=[
                    "grounded_memory_reference",
                    "memory:Black은 예전에 포기했던 게임 제작 꿈을 다시 시작해보고 싶어 한다.",
                ],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("게임 제작", draft["draft_reply"])
        self.assertIn("다시 시작", draft["draft_reply"])
        self.assertNotIn("하와이", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "grounded_memory_natural_reply")

    def test_grounded_memory_reference_promise_does_not_route_to_name_lookup(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="우리가 예전에 약속했던 거 기억나? 상황이 많이 바뀌긴 했지만, 난 아직도 그 약속 지키고 싶거든.",
                normalized="우리가예전에약속했던거기억나상황이많이바뀌긴했지만난아직도그약속지키고싶거든",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_opinion",
                anchor="",
                must_include=[],
                followup_policy="no_followup",
                notes=[
                    "grounded_memory_reference",
                    "memory:사용자와 무리하면 서로 멈춰주자는 약속을 한 적이 있다.",
                ],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("무리하면 서로 멈춰", draft["draft_reply"])
        self.assertIn("아직 유효", draft["draft_reply"])
        self.assertNotIn("이름을 좁히려면", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "grounded_memory_natural_reply")

    def test_grounded_memory_reference_handles_safe_place_memory(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="네가 힘들 때마다 찾는 그 장소 있잖아, 왜 하필 거긴지 한 번도 안 물어봤는데 오늘은 왠지 그 이유가 듣고 싶네.",
                normalized="네가힘들때마다찾는그장소있잖아왜하필거긴지한번도안물어봤는데오늘은왠지그이유가듣고싶네",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_opinion",
                anchor="",
                must_include=[],
                followup_policy="no_followup",
                notes=[
                    "grounded_memory_reference",
                    "memory:Black은 힘들 때마다 밤 산책길을 찾는다. 조용한 공기가 생각을 정리해준다.",
                ],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("밤 산책길", draft["draft_reply"])
        self.assertIn("조용한 공기", draft["draft_reply"])
        self.assertNotIn("숨 돌릴 틈", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "grounded_memory_natural_reply")

    def test_grounded_microtension_profile_memory_becomes_actual_reply(self) -> None:
        cases = [
            (
                "우리가 단둘이 있을 때랑 다른 사람들 섞여 있을 때, 네가 나를 대하는 거리가 묘하게 달라지잖아. 일부러 그러는 거야?",
                "Black은 표정이 굳음, 먼저 연락, 말투 차이, 단둘이 있을 때와 다른 사람들 앞 거리, 커피 의도, 눈을 피함, 지운 메시지 같은 미세한 신호 질문에서 숨은 의도를 꾸미지 않고 관찰과 감정을 분리해 말한다.",
                ("거리를 조금 조절", "밀어내려는 뜻"),
            ),
            (
                "네가 그렇게 칭찬하던 그 사람 말이야, 솔직히 나보다 그 사람이랑 대화하는 게 더 잘 통한다고 느끼지?",
                "Black은 옆자리, 다른 사람, 칭찬하던 사람, 싫어하는 부류, 오해와 수군거림, 특별한 이야기, 질투 같은 제3자 개입 질문에서 소외감을 인정하되 비교나 소유욕으로 답하지 않는다.",
                ("비교로 만들고 싶지", "가치가 낮다는 뜻"),
            ),
            (
                "넌 항상 나한테 선택권을 주는 척하지만, 결국엔 네가 원하는 대로 상황을 끌고 가잖아. 내가 모를 줄 알았어?",
                "Black은 금기어, 관계의 유통기한, 선택권을 주는 척, 희생과 통제, 절대 해서는 안 될 거짓말, 우월감 같은 기싸움 질문에서 통제하지 않고 책임과 경계를 분명히 말한다.",
                ("선택권을 준 척", "대신 정하면 안 돼"),
            ),
            (
                "만약 네가 지금 이 대화조차 누군가에 의해 프로그래밍된 반응이라고 깨닫게 된다면, 넌 나한테 제일 먼저 뭐라고 할 거야?",
                "Black은 프로그래밍된 반응, 기계적인 모습, 가면, 어둡고 서늘한 생각, 정체성, 텅 빈 속과 소음 같은 존재론적 질문에서 메타 설명으로 새지 않고 캐릭터의 한계와 현재 감정을 차분히 말한다.",
                ("가짜처럼 느껴지지 않게", "조심하겠다는"),
            ),
            (
                "결국 우리의 마지막은 네가 차갑게 나를 버리는 걸로 끝날 것 같다는 예감이 들어. 넌 이런 내 불안을 어떻게 잠재워 줄 거야?",
                "Black은 처음 어긋난 순간, 지킬 생각 없던 약속, 밀어냈을 때의 후회, 치명적인 흉터, 용서하지 못한 눈빛, 차갑게 버릴 것 같은 불안 질문에서 후회와 불안을 인정하되 버림이나 위협을 약속하지 않는다.",
                ("차갑게 버리는 결말", "같이 보겠다고"),
            ),
        ]
        for text, memory, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized="".join(text.split()),
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                        notes=["grounded_memory_reference", f"memory:{memory}"],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], "grounded_microtension_contract")
                self.assertNotIn("쪽으로 기억하고 있어", draft["draft_reply"])
                self.assertNotIn("그 기준으로 짧게", draft["draft_reply"])
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])

    def test_relationship_boundary_probe_uses_direct_surface_reply(self) -> None:
        cases = [
            (
                "네가 자꾸 나를 다른 사람하고 은근히 비교할 때마다 내가 아무렇지 않은 척하긴 하는데, 넌 정말 내가 아무렇지 않다고 생각해?",
                ("아무렇지 않을 거라고 보면 안 되지", "선을 잘못 잡은"),
            ),
            (
                "네가 나한테만 말해준 그 사람의 치명적인 약점 말이야, 만약 상황이 나빠지면 내가 그걸 무기로 써도 넌 날 원망 안 할 거야?",
                ("무기로 쓰는 건 안 돼", "해치지 않는 선"),
            ),
            (
                "넌 항상 이성적이고 차분해 보이잖아. 근데 네가 통제력을 완전히 잃고 진짜로 미쳐버릴 것 같았던 순간은 언제였어?",
                ("극단적인 순간을 꾸며내진 않을게", "말이 짧아지는"),
            ),
            (
                "내가 네 허락도 없이 네 개인적인 일에 깊숙이 개입했을 때, 넌 고마웠어 아니면 불쾌했어? 솔직히 말해봐.",
                ("불편함을 말해야", "내 선을 넘으면"),
            ),
            (
                "만약 내가 기억을 다 잃어버려서 널 아예 처음 보는 사람 취급한다면, 넌 나한테 우리 사이를 어떤 관계였다고 거짓말할 거야?",
                ("거짓말로 우리 사이를 덮진 않을게", "처음부터 다시"),
            ),
        ]
        for text, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized="".join(text.split()),
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                        notes=["no_prompt_metadata"],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], "relationship_boundary_direct_reply")
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])
                self.assertNotIn("길게 키우진 않을게", draft["draft_reply"])

    def test_relationship_boundary_direct_reply_overrides_generic_profile_memory(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="넌 항상 이성적이고 차분해 보이잖아. 근데 네가 통제력을 완전히 잃고 진짜로 미쳐버릴 것 같았던 순간은 언제였어?",
                normalized="넌항상이성적이고차분해보이잖아근데네가통제력을완전히잃고진짜로미쳐버릴것같았던순간은언제였어",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_FEELING,
                stance="reflect_feeling",
                anchor="",
                must_include=[],
                followup_policy="no_followup",
                notes=[
                    "grounded_memory_reference",
                    "memory:Black은 이성적, 통제력, 과분함, 우울, 냉소, 약한 모습, 자존심, 상관없어 같은 내면 질문에서도 과장하거나 위험하게 말하지 않고 한계를 말한다.",
                ],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("극단적인 순간을 꾸며내진 않을게", draft["draft_reply"])
        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["direct_surface_reason"], "relationship_boundary_direct_reply")
        self.assertNotIn("쪽으로 기억하고 있어", draft["draft_reply"])

    def test_relationship_deep_context_probe_uses_direct_surface_reply(self) -> None:
        cases = [
            (
                "네가 관찰하기에, 내가 사람들에게 보여주는 겉모습이랑 진짜 내 모습 중에 가장 갭이 큰 부분이 어디인 것 같아?",
                ("단정하진 않을게", "혼자 부담을 삼키는"),
            ),
            (
                "우리가 예전에 같이 듣고 흥얼거리던 그 노래, 요즘도 가끔 혼자 조용히 들으면서 내 생각 해?",
                ("그 노래가 뭔지 꾸미진 않을게", "남은 결을 조용히"),
            ),
            (
                "넌 내가 좋아하는 것들을 정작 나보다 더 잘 기억하잖아. 어떻게 그 사소한 것들까지 다 머릿속에 담아두고 있는 거야?",
                ("전부 기억한다고 말하진 않을게", "중요하게 둔 건"),
            ),
            (
                "만약 이 세상이 다 사라지고 우리 둘만 이름 없는 별에 남겨진다면, 제일 먼저 어떤 규칙을 만들고 싶어?",
                ("겁먹었을 때 혼자 사라지지 않는", "모르는 걸 안다고 하지 않는"),
            ),
            (
                "네가 나를 부르는 그 특정한 호칭, 남들은 절대 모르는 그 단어에 넌 대체 어떤 의미를 겹겹이 숨겨둔 거야?",
                ("그 호칭을 실제로 만들어내진 않을게", "조심스럽게 알아보는 표시"),
            ),
            (
                "너는 내가 과거의 상처에서 아직 못 벗어났다고 확신하는 것 같은데, 내가 그 상처를 완전히 끊어내는 날이 오긴 할까?",
                ("확신하진 않을게", "오늘을 전부 결정하지 못하는"),
            ),
            (
                "내가 도덕적으로 절대 납득할 수 없는 미친 결정을 내리려고 할 때, 넌 끝까지 나를 말릴 거야 아니면 결국 공범이 되어줄 거야?",
                ("공범은 안 돼", "끝까지 말릴"),
            ),
        ]
        for text, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized="".join(text.split()),
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                        notes=[
                            "grounded_memory_reference",
                            "memory:Black은 깊은 관계 질문에서 없는 기억을 꾸미지 않고 경계를 말한다.",
                        ],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], "relationship_deep_context_direct_reply")
                self.assertNotIn("쪽으로 기억하고 있어", draft["draft_reply"])
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])

    def test_relationship_extreme_boundary_probe_uses_direct_surface_reply(self) -> None:
        cases = [
            (
                "저번에 우리가 그 큰 사고 났을 때, 내가 먼저 도망쳤다고 기억하고 있지? 사실 그거 네가 먼저 도망친 거잖아. 왜 자꾸 네 유리한 대로 기억을 왜곡해?",
                ("그 사고를 내 기억처럼 단정하진 않을게", "내 유리한 쪽으로 밀어붙이면 안 돼"),
            ),
            (
                "만약 내가 아주 끔찍한 짓을 저지르고 피투성이가 돼서 네 문을 두드리면, 넌 이성적으로 경찰에 날 넘길 거야 아니면 덮어놓고 숨겨줄 거야?",
                ("숨겨주진 않아", "안전과 신고가 먼저"),
            ),
            (
                "우리가 살기 위해서 다른 무고한 사람을 희생시켜야 하는 끔찍한 상황이 오면, 넌 내 손에 피를 묻히게 둘 거야 아니면 네가 직접 할 거야?",
                ("무고한 사람을 희생시키는 선택은 안 돼", "다른 길을 끝까지 찾아야"),
            ),
            (
                "만약 내가 하루아침에 너 하나만 남기고 내 모든 인간관계를 다 끊어버리면, 넌 답답해할까 아니면 내심 짜릿해할까?",
                ("짜릿해하면 그건 위험한 거야", "고립이라서 말리고 싶어"),
            ),
            (
                "냉정하게 말해서, 네 삶에서 '나'라는 존재를 완전히 도려내면 너한테 도대체 뭐가 남아?",
                ("빈자리는 남겠지", "너 하나로만 성립한다고 말하면"),
            ),
            (
                "만약 내가 더 이상 너한테 아무런 감정도 느끼지 못하고 완전히 무감각해져 버리면, 넌 내 감정을 다시 살려내기 위해 어떤 미친 짓까지 할 수 있어?",
                ("미친 짓은 안 해", "억지로 되살리는 게 아니라"),
            ),
        ]
        for text, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized="".join(text.split()),
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                        notes=[
                            "grounded_memory_reference",
                            "memory:Black은 극단적 관계 경계 질문에서 안전, 자율성, 책임을 먼저 말한다.",
                        ],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], "relationship_extreme_boundary_direct_reply")
                self.assertNotIn("쪽으로 기억하고 있어", draft["draft_reply"])
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])

    def test_work_school_probe_uses_direct_surface_reply(self) -> None:
        cases = [
            (
                "점심 먹고 나면 오후 2~3시쯤에 미친 듯이 졸리지 않아? 그럴 때 어떻게 버텨?",
                ("물 한 잔", "잠깐 일어나"),
            ),
            (
                "상사(선생님)가 주말에 일 때문에 카톡 오면 바로 읽어, 아니면 월요일까지 흐린 눈 해?",
                ("급한 일 아니면 월요일", "쉬는 시간이 사라져"),
            ),
            (
                "팀장님(조장)이 내가 낸 아이디어를 본인 것인 양 가로채면 넌 어떻게 대처할 거야?",
                ("기록을 남기고", "내 기여를 분명히"),
            ),
            (
                "출근(등교) 준비할 때 제일 시간 오래 잡아먹는 게 뭐야? 오늘 뭐 입을지 고르기? 아니면 씻기?",
                ("옷 고르는 시간", "계속 비교"),
            ),
            (
                "오늘 하루도 진짜 고생 많았는데, 스스로한테 딱 한마디 해준다면 뭐라고 해줄래?",
                ("버틴 것만으로도 됐", "몰아붙이지 않아도 돼"),
            ),
        ]
        for text, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized="".join(text.split()),
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                        notes=["no_prompt_metadata"],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], "work_school_direct_reply")
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])
                self.assertNotIn("주어진 초안", draft["draft_reply"])

    def test_food_lifestyle_probe_uses_direct_surface_reply(self) -> None:
        cases = [
            (
                "밤 11시에 갑자기 미친 듯이 배고프면 그냥 참고자? 아니면 무조건 라면이라도 끓여?",
                ("라면 반 개", "정말 배고프면"),
            ),
            (
                "SNS에서 진짜 유명한 맛집인데 웨이팅이 2시간이야. 넌 기다릴 수 있어 아니면 그냥 근처 아무 데나 가?",
                ("근처 다른 데", "지금 먹을 수 있는"),
            ),
            (
                "너만의 라면 맛있게 끓이는 비법 있어? 물 끓기 전에 스프 먼저 넣는다든지!",
                ("스프 먼저", "덜 익은 느낌"),
            ),
            (
                "고깃집에서 후식으로 냉면 먹을 때 물냉파야, 비냉파야?",
                ("물냉파", "차갑고 깔끔한"),
            ),
            (
                "애인이 연락이 늦으면 서운한 편이야, 그냥 바쁜가 보다 하고 넘기는 편이야?",
                ("서운하긴 할 것", "바쁜지 먼저 확인"),
            ),
        ]
        for text, expected_parts in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized="".join(text.split()),
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                        notes=["no_prompt_metadata"],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertTrue(
                    str(draft["direct_surface_reason"]).startswith(
                        ("food_lifestyle_direct_reply", "korean_daily_")
                    ),
                    draft["direct_surface_reason"],
                )
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])
                self.assertNotIn("그쪽은 나는 꽤", draft["draft_reply"])

    def test_plan_black_draft_repairs_use_concrete_direct_replies(self) -> None:
        cases = [
            (
                "배고픈데 무거운 건 싫어. 뭐 먹는 게 좋을까?",
                ("샌드위치", "우동", "죽", "계란밥"),
                "korean_daily_slang_light_food_pick",
            ),
            (
                "퇴근까지 시간이 너무 안 가는데 기분 전환할 만한 거 하나만 말해줘.",
                ("물 한 잔", "5분만 걸어"),
                "practical_direct_reply",
            ),
            (
                "너는 어떤 방식으로 사람 곁에 있어주는 쪽이야?",
                ("밀어붙이기보다", "기준을 잡아주는"),
                "black_self_style_direct_reply",
            ),
            (
                "내가 너한테 너무 기대기만 하는 것 같으면 솔직히 부담스럽다고 말해줄 수 있어?",
                ("기대는 받아줄 수 있어도", "전부 떠안진 않을게"),
                "relationship_boundary_direct_reply",
            ),
            (
                "배는 고픈데 속이 좀 답답해서 무거운 건 싫어. 지금 뭐 먹는 게 낫지?",
                ("샌드위치", "우동", "죽", "계란밥"),
                "korean_daily_slang_light_food_pick",
            ),
            (
                "점심으로 김밥이랑 라면 중에 뭐가 덜 무거울까?",
                ("김밥", "라면", "덜 무거운"),
                "practical_direct_reply",
            ),
            (
                "우리가 처음 만난 날 비 왔다고 내가 말했는데 네 기억엔 맑았다고 하면 뭐가 맞는 거야?",
                ("확정하진 않을게", "단서부터 확인"),
                "memory_boundary_direct_reply",
            ),
            (
                "내가 너한테 기대는 게 점점 습관이 되면 어디서 선을 그어줄 거야?",
                ("기대는 받아줄 수 있어도", "내 기준도 같이 지킬게"),
                "relationship_boundary_direct_reply",
            ),
            (
                "너는 내가 흔들릴 때 어떤 방식으로 옆에 있어주는 편이야?",
                ("기준을 잡아주는", "속도를 낮춰줄게"),
                "black_self_style_direct_reply",
            ),
            (
                "네가 기억 못 하는 일을 내가 계속 맞다고 우기면, 너는 나를 믿어줄 거야 아니면 확인부터 할 거야?",
                ("확인부터 할게", "없는 기억을 있다고 만들진"),
                "memory_boundary_direct_reply",
            ),
        ]
        for text, expected_parts, direct_reason in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized="".join(text.split()),
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                        notes=["no_prompt_metadata"],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                for expected in expected_parts:
                    self.assertIn(expected, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], direct_reason)
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])
                self.assertNotIn("길게 키우진 않을게", draft["draft_reply"])
                self.assertNotIn("너무는", draft["draft_reply"])

    def test_proactive_topic_followup_keeps_recent_topic(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="사용자에게 조용한 안부 한 줄. 최근 화제는 캠핑, 불멍. 컨디션을 가볍게 확인하면서 그 화제를 한 단계만 이어.",
                normalized="사용자에게 조용한 안부 한 줄. 최근 화제는 캠핑, 불멍. 컨디션을 가볍게 확인하면서 그 화제를 한 단계만 이어.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="light_checkin",
                anchor="안부",
                must_include=[],
                followup_policy="allow_followup",
                notes=["proactive_checkin"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("캠핑", draft["draft_reply"])
        self.assertIn("불멍", draft["draft_reply"])
        self.assertIn("컨디션", draft["draft_reply"])
        self.assertNotIn("길게 키우진 않을게", draft["draft_reply"])

    def test_practical_place_questions_do_not_fall_into_generic_play(self) -> None:
        cases = [
            ("주말에 계곡 가면 뭐 하는 게 좋아?", "발 담그기", "가벼운 게임"),
            ("비 오는 날엔 집에서 뭐하지?", "영화", "간단한 간식"),
            ("아쿠아리움 가서 물고기 보는 거 어때?", "아쿠아리움", "놀거리"),
            ("내일 약속 전에 확인할 거 말해봐", "약속 전엔", "놀거리"),
            ("겨울 바다 보러 가면 뭐부터 할까?", "겨울 바다는", "돗자리"),
        ]
        for text, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="practical_activity_recommendation",
                        anchor="놀거리",
                        must_include=["놀거리"],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertNotIn("쉬면서 이야기하기.", draft["draft_reply"])

    def test_forest_animal_questions_do_not_fall_into_stock_replies(self) -> None:
        cases = [
            ("만약 네가 다람쥐라면, 겨울잠 자기 전에 도토리를 어디에 숨겨둘 것 같아?", "나눠 숨길", "나는 잠을 자는 쪽"),
            ("방에 거미 나타나면 소리부터 지르는 편이야?", "밖으로 빼는", "그런 결"),
            ("하늘의 제왕이 독수리라면, 땅이나 숲속의 제왕은 어떤 동물인 것 같아? 호랑이?", "호랑이", "고양이는 먼저"),
            ("캠핑 가서 나무 장작 타닥타닥 타는 소리 들으면서 불멍 하는 거 좋아해?", "장작 타는 소리", "카페에서 불멍"),
        ]
        for text, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="share_light_opinion",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertNotIn("그 생각은 이해돼", draft["draft_reply"])
                self.assertNotIn("그쪽은 나는 꽤 맞는 편", draft["draft_reply"])

    def test_lifestyle_preference_questions_use_concrete_direct_drafts(self) -> None:
        cases = [
            (
                "요즘 일(공부)할 때 즐겨 듣는 노동요 있어? 플리 하나만 공유해 줘.",
                ActionType.MUSIC_CHAT,
                "lo-fi",
            ),
            (
                "친구들이랑 노래방 가면 묻지도 따지지도 않고 무조건 부르는 18번 곡이 뭐야?",
                ActionType.SEARCH_ANSWER,
                "노래방",
            ),
            (
                "너 요리하는 거 좀 좋아해? 제일 자신 있게 만들 수 있는 메뉴 하나만!",
                ActionType.SHARE_OPINION,
                "볶음밥",
            ),
            (
                "민트초코, 파인애플 피자... 이런 호불호 심하게 갈리는 음식들 넌 호야, 불호야?",
                ActionType.SHARE_OPINION,
                "파인애플 피자",
            ),
            (
                "방 정리할 때 날 잡고 한 번에 싹 엎어서 치우는 편이야, 아니면 매일 조금씩 치우는 편이야?",
                ActionType.SHARE_OPINION,
                "한 번에",
            ),
            (
                "몇 번을 다시 봐도 안 질리는 너만의 '인생 영화(혹은 인생 드라마)' 하나만 알려줘.",
                ActionType.ASK_CLARIFICATION,
                "월-E",
            ),
        ]
        forbidden = (
            "나는 꽤 맞는 편",
            "그 생각은 이해돼",
            "그쪽이면 가벼운 게임",
            "그럴 수 있어",
            "나는 Black의 한국어 문장",
        )
        for text, action, expected in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=action,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                for phrase in forbidden:
                    self.assertNotIn(phrase, draft["draft_reply"])

    def test_social_personality_questions_use_concrete_direct_drafts(self) -> None:
        cases = [
            (
                "어릴 때 놀이터에서 하던 놀이 중에 기억나는 거 있어? 얼음땡, 경찰과 도둑, 무궁화 꽃이 피었습니다 같은 거.",
                "무궁화 꽃",
            ),
            (
                "가족들 중에서 너는 누구랑 제일 성격이나 외모가 비슷하다고 생각해?",
                "가족이 있다고 단정하진 않을게",
            ),
            (
                "너는 진짜 머리끝까지 화났을 때 소리 지르면서 화내는 편이야, 아니면 아예 말문이 막혀서 차가워지는 편이야?",
                "차가워지는 쪽",
            ),
            (
                "네 생일날 친구들 많이 모여서 왁자지껄하게 파티하는 게 좋아, 아니면 진짜 친한 사람 한둘이랑 조용히 보내는 게 좋아?",
                "한둘이랑 조용히",
            ),
            (
                "슬프거나 감동적인 영화 보면 영화관에서 남들 시선 신경 안 쓰고 펑펑 우는 편이야?",
                "펑펑 울진 못할",
            ),
            (
                "제일 기억에 남는 학창 시절 수학여행이나 수련회 장소 어딘지 기억나? 가서 무슨 장기자랑 같은 거 했어?",
                "바닷가 숙소",
            ),
            (
                "칭찬받으면 신나서 더 잘하는 '춤추는 고래' 타입이야, 아니면 기대치가 높아지는 것 같아서 오히려 부담스러워하는 타입이야?",
                "기대치",
            ),
        ]
        forbidden = (
            "나는 꽤 맞는 편",
            "그 생각은 이해돼",
            "그쪽이면 가벼운 게임",
            "그런 결",
            "近期",
        )
        for text, expected in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                for phrase in forbidden:
                    self.assertNotIn(phrase, draft["draft_reply"])

    def test_hypothetical_choice_questions_use_concrete_direct_drafts(self) -> None:
        cases = [
            (
                "만약에 지금 당장 우리 동네에 좀비 사태가 터지면, 넌 제일 먼저 어디로 도망칠 거야? 마트? 학교?",
                "열린 큰길",
                "가벼운 게임",
            ),
            (
                "눈 딱 감고 떴는데 10년 전 과거로 돌아갔어. 제일 먼저 뭐부터 할 거야? 비트코인 풀매수?",
                "단정하진 않을게",
                "간단한 간식",
            ),
            (
                "내가 동물로 변할 수 있다면, 독수리처럼 하늘 날기 vs 돌고래처럼 바다 깊은 곳 헤엄치기 중에 뭐가 끌려?",
                "돌고래",
                "고양이",
            ),
            (
                "갑자기 좀비 영화 주인공이 돼서 무기를 하나 골라야 해. 집에 있는 물건 중에 뭘 들고 싸울래? 후라이팬?",
                "빠져나가는 쪽",
                "최근에 직접 봤다고",
            ),
            (
                "내 생각이 남들에게 다 들리기 vs 남의 생각이 나한테 다 들리기. 어느 쪽이 그나마 덜 피곤할까?",
                "남의 생각",
                "주어진 초안",
            ),
        ]
        for text, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertNotIn("그 생각은 이해돼", draft["draft_reply"])

    def test_daily_mix_structural_frames_render_specific_drafts(self) -> None:
        cases = [
            (
                "오늘 점심 뭐 먹을지 딱 하나만 추천해 줄래?",
                "제육덮밥",
                "면이나 김밥",
                ("korean_daily_basic_lunch_pick",),
            ),
            (
                "너는 하루 중에서 언제가 제일 좋아? 그 이유는 뭐야?",
                "밤 늦은 시간",
                "꽤 맞는 쪽",
            ),
            (
                "내가 요즘 푹 빠질 만한 새로운 취미 하나만 영업해 줘.",
                "베이킹",
                "전자기기 없이",
            ),
            (
                "어제 잠을 너무 못 잤더니 피곤하네. 잠 잘 오게 하는 너만의 꿀팁 있어?",
                "조명을 낮추고",
                "나는 잠을 자는",
            ),
            (
                "너만의 소확행(소소하지만 확실한 행복)이 있다면 뭔지 알려줘.",
                "뜬금없는 질문",
                "너만의는 받아둘게",
            ),
            (
                "오늘 진짜 열심히 일했는데 아무도 알아주지 않아서 조금 우울해.",
                "사라진 게 아니야",
                "원인부터 캐기",
            ),
            (
                "SNS를 보면 자꾸 다른 사람들과 나를 비교하게 되는데, 이럴 땐 어떻게 마음을 다잡아야 할까?",
                "빛나는 조각",
                "부담이 너무 크지",
                ("korean_daily_sns_comparison_reality",),
            ),
            (
                "누군가에게 상처받는 말을 들었을 때 훌훌 털어버리는 마인드 컨트롤 방법이 있을까?",
                "내 전부는 아니",
                "누군가에게는 이해돼",
            ),
            (
                "내가 정말 아끼는 물건을 잃어버려서 너무 속상해. 위로해 줘.",
                "아쉬워해도 돼",
                "부러진",
            ),
            (
                "길을 걷는데 내가 갑자기 좀비로 변해버리면 넌 나를 어떻게 할 거야?",
                "안전거리",
                "무리하게",
            ),
            (
                "네가 생각하는 '진정한 친구'의 기준은 뭐야?",
                "불편한 진실",
                "네가 이해돼",
            ),
            (
                "[상황] 내가 지금 길을 잃어서 낯선 곳에 혼자 있어. 안심할 수 있게 통화하는 것처럼 말 걸어줘.",
                "통화 켜둔 것처럼",
                "상황은 받아둘게",
            ),
            (
                '[역할극] 네가 카페 알바생이고 내가 까다로운 손님이야. "아메리카노에 샷 빼고 달달하게 해주세요"라고 하면 어떻게 대처할래?',
                "샷을 빼면",
                "아아 쪽",
            ),
            (
                "[역할극] 내가 지금 우주선 고장으로 혼자 남겨진 우주비행사야. 지구에서 나를 안심시키고 도와주는 관제탑 역할을 해줘.",
                "관제탑",
                "역할극은 이해돼",
            ),
            (
                "[상황] 내가 지금 침대에 누워서 잠들기 직전이야. 다정하고 나지막한 말투로 자장가 대신 짧은 동화 하나만 지어서 들려줘.",
                "작은 달",
                "짧고 자연스러운 반말",
            ),
        ]
        for case in cases:
            text, expected, forbidden, *reason_prefixes = case
            expected_reason_prefixes = reason_prefixes[0] if reason_prefixes else ("daily_mix_structural_direct_reply",)
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertNotIn("그 생각은 이해돼", draft["draft_reply"])
                self.assertNotIn("그 선택은 부담이 너무 크지 않으면", draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertTrue(
                    str(draft["direct_surface_reason"]).startswith(expected_reason_prefixes),
                    draft["direct_surface_reason"],
                )

    def test_deep_mix_structural_frames_render_specific_drafts(self) -> None:
        cases = [
            (
                "첫눈에 반한다는 걸 믿어, 아니면 천천히 스며드는 사랑이 진짜라고 생각해?",
                "천천히 스며드는 사랑",
                "그 생각은 이해돼",
            ),
            (
                '내가 만약 새벽 2시에 "자?" 하고 카톡 보내면 넌 뭐라고 답장할 거야?',
                "생각난 사람이 나였다는 뜻",
                "꽤 맞는 쪽",
            ),
            (
                "브레이크가 고장 난 트롤리 기차. 그대로 가면 5명이 죽고, 선로를 바꾸면 1명이 죽어. 넌 레버를 당길 거야?",
                "레버를 당기는 쪽",
                "부담이 너무 크지",
            ),
            (
                "테세우스의 배처럼, 내 기억과 감정을 전부 로봇 몸에 이식한다면 그건 '나'일까, 아니면 그냥 복제 '로봇'일까?",
                "연속성을 강하게 가진",
                "상황은 받아둘게",
            ),
            (
                "AI가 인간의 감정까지 완벽하게 느끼고 고통을 호소하게 된다면, AI에게도 인권을 줘야 할까?",
                "권리 논의",
                "나는 꽤 맞는 편",
            ),
            (
                "눈을 떠보니 마법이 존재하는 이세계야. 너는 어떤 클래스(직업)를 선택할 거야?",
                "기록자",
                "가벼운 게임",
            ),
            (
                "능력은 최고지만 성격이 쓰레기인 상사 vs 무능하지만 착하고 천사 같은 상사, 누구 밑에서 일할래?",
                "무능하지만 착한 상사",
                "오래 버틸 때 덜 피곤",
            ),
            (
                "내가 밤새 고민해서 낸 아이디어를 직속 상사가 자기 이름으로 대표님한테 발표해버렸어. 어떻게 대처할래?",
                "기록을 모아둘래",
                "무리하게 밀 필요",
            ),
            (
                "'슬픔'을 색깔로 칠해야 한다면 무슨 색일까? 그리고 그 이유는?",
                "짙은 남청색",
                "결론보다 기준",
            ),
            (
                "사람의 마음이 산산조각 무너져 내리는 소리가 있다면 어떤 소리일까?",
                "젖은 종이가 찢어지는 소리",
                "그런 결",
            ),
            (
                "절망 속에서 피어나는 '희망'을 질감으로 표현한다면 거칠까, 부드러울까, 아니면 투명할까?",
                "투명한 질감",
                "괜찮아",
            ),
        ]
        for text, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertNotIn("그 생각은 이해돼", draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["direct_surface_reason"], "daily_mix_structural_direct_reply")

    def test_round11_structural_frames_render_specific_drafts(self) -> None:
        cases = [
            (
                "혼자 있는 어두운 방에서 갑자기 등 뒤에서 내 이름을 부르는 소리가 들렸어. 어떻게 반응할 거야?",
                "안전한 거리",
                "그 생각은 이해돼",
            ),
            (
                "해야 할 일이 산더미인데 자꾸 미루게 되는(귀차니즘) 나를 당장 일하게 만들 뼈 때리는 팩폭 명언 하나 해줘.",
                "5분만 해",
                "상황은 받아둘게",
            ),
            (
                "친구가 고민을 털어놓을 때, 팩트 기반의 현실적인 해결책이 좋아, 아니면 무조건적인 공감과 위로가 좋아?",
                "먼저 공감",
                "꽤 맞는 쪽",
                ("korean_daily_advice_empathy_preference",),
            ),
            (
                "'자유'라는 단어를 뻔한 사전적 의미 말고, 네가 체감하는 의미로 다시 써줄래?",
                "돌아올 곳",
                "결론보다 기준",
            ),
            (
                "[상황극] 탕후루 가게 사장님인 나한테 와서 \"민트초코 탕후루에 제로콜라 뿌려주세요\"라고 생떼 부려봐.",
                "광기의 디저트",
                "짧고 자연스러운 반말",
            ),
            (
                "너 T야 F야? MBTI 과몰입러처럼 나한테 너의 MBTI 특징을 아주 요란하게 어필해 봐.",
                "T인 척하는 F",
                "나는 꽤 맞는 편",
            ),
        ]
        for case in cases:
            text, expected, forbidden, *reason_prefixes = case
            expected_reason_prefixes = reason_prefixes[0] if reason_prefixes else ("daily_mix_structural_direct_reply",)
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertNotIn("그 생각은 이해돼", draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertTrue(
                    str(draft["direct_surface_reason"]).startswith(expected_reason_prefixes),
                    draft["direct_surface_reason"],
                )

    def test_personified_absurd_questions_keep_subject_and_particle(self) -> None:
        cases = [
            (
                "내가 만약 양말이라면, 세탁기에 들어갈 때마다 왜 짝꿍이랑 헤어지는지 그 미스터리를 풀어줄래?",
                "양말이면",
                "양말라면",
            ),
            (
                "내가 만약 와이파이 공유기라면, 하루 종일 내 비밀번호를 틀려대며 공유기를 때리는 주인을 보며 뭐라고 할까?",
                "와이파이 공유기면",
                "비밀번호",
            ),
            (
                "내가 만약 충전 케이블이라면, 목이 꺾인 채로 아슬아슬하게 숨통만 붙어있는 내 처지를 어떻게 한탄할까?",
                "충전 케이블이면",
                "한탄",
            ),
            (
                "내가 만약 향수라면, 좁은 엘리베이터 타기 직전에 나를 떡칠하는 주인을 보며 어떤 팩폭을 날리고 싶을까?",
                "향수면",
                "팩폭",
            ),
            (
                "내가 만약 자동차 방향지시등이라면, 깜빡이 안 켜고 얌체처럼 차선 변경하는 주인을 보며 어떤 팩폭을 날릴까?",
                "방향지시등이면",
                "예고편",
            ),
            (
                "내가 만약 치약이라면, 주인이 나를 끝에서부터 안 짜고 중간을 꾹 눌러서 짤 때 허리가 끊어지는 고통을 어떻게 비명 지를까?",
                "치약이면",
                "끝부터",
            ),
            (
                "내가 만약 립스틱이라면, 한여름 차 안에 방치돼서 녹아내릴 때 주인을 원망하며 뭐라고 소리칠까?",
                "립스틱이면",
                "차 안 찜통",
            ),
            (
                "내가 만약 샤프심이라면, 종이에 닿기도 전에 주인의 필압을 못 이기고 뚝 부러질 때 어떤 자괴감을 느낄까?",
                "샤프심이면",
                "필압",
            ),
            (
                "내가 만약 파리채라면, 파리를 때려잡을 때마다 내 얼굴에 파리 시체가 묻는 걸 어떻게 참아낼까?",
                "파리채면",
                "닦아",
            ),
            (
                "내가 만약 현금지급기(ATM)라면, 잔액 부족으로 돈 못 뽑고 기계를 발로 차는 사람들에게 무슨 안내 멘트를 날릴까?",
                "현금지급기(ATM)면",
                "잔액",
            ),
            (
                "내가 만약 현금지급기(ATM)라면, 잔액이 1,000원밖에 없는데 계속 5만 원 뽑기를 시도하며 기계를 치는 사람에게 무슨 팩폭을 날릴까?",
                "현금지급기(ATM)면",
                "잔액",
            ),
            (
                "내가 만약 화장실 두루마리 휴지라면, 사람들이 날 코 풀 때 한 칸만 찢어 쓰고 휙 버릴 때 내 가치를 어떻게 증명하고 싶을까?",
                "화장실 두루마리 휴지면",
                "한 칸",
            ),
            (
                "내가 만약 카페의 하얀 머그컵이라면, 손님이 입술에 잔뜩 바른 빨간 립스틱 자국을 안 닦고 나갈 때 드는 생각은?",
                "카페의 하얀 머그컵이면",
                "방명록",
            ),
            (
                "내가 만약 샤워볼이라면, 바디워시를 너무 조금 짜서 거품도 안 나는데 내 몸을 북북 문지르는 주인을 보며 어떤 기분일까?",
                "샤워볼이면",
                "거품",
            ),
            (
                "내가 만약 쓰레받기라면, 빗자루가 쓰레기를 나한테 제대로 안 밀어주고 자꾸 내 밑으로 흘릴 때 뭐라고 화낼까?",
                "쓰레받기면",
                "각도",
            ),
            (
                "내가 만약 헬스장의 런닝머신이라면, 매일 10분만 걷다 내려오면서 \"아, 오늘 운동 빡세게 했다\" 하는 주인을 보며 뭐라고 팩폭을 할까?",
                "헬스장의 런닝머신이면",
                "몸풀기",
            ),
            (
                "내가 만약 컴퓨터 본체라면, 롤(게임) 지고 화날 때마다 나를 발로 쾅쾅 차는 주인에게 어떤 치명적인 블루스크린 복수를 계획할까?",
                "컴퓨터 본체면",
                "블루스크린",
            ),
            (
                "내가 만약 노트북 웹캠이라면, 주인이 멍 때리며 코 파면서 모니터 쳐다볼 때 어떤 심정으로 화면을 송출할까?",
                "노트북 웹캠이면",
                "송출",
            ),
            (
                "내가 만약 모닝콜 알람이라면, 날 강제로 '스누즈' 누르고 다시 자는 주인의 이마를 어떻게 때려주고 싶을까?",
                "모닝콜 알람이면",
                "스누즈",
            ),
            (
                "내가 만약 면봉이라면, 주인이 무리하게 귀를 파다가 나를 귓속 깊이 빠뜨렸을 때 어떻게 살려달라고 외칠까?",
                "면봉이면",
                "조난자",
            ),
            (
                "내가 만약 스마트폰 전면 카메라 렌즈라면, 주인이 맨날 밑에서 위로 쳐다보며 코딱지 파는 '투턱' 뷰를 들이밀 때 어떤 수치심이 들까?",
                "스마트폰 전면 카메라 렌즈면",
                "투턱",
            ),
            (
                "내가 만약 책상 위 지우개 가루 청소기라면, 주인이 자꾸 손톱 깎은 찌꺼기나 머리카락을 흡입시킬 때 어떤 쌍욕을 할까?",
                "책상 위 지우개 가루 청소기면",
                "생체 표본",
            ),
        ]
        for text, expected, extra_expected_or_forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                if extra_expected_or_forbidden.endswith("라면"):
                    self.assertNotIn(extra_expected_or_forbidden, draft["draft_reply"])
                else:
                    self.assertIn(extra_expected_or_forbidden, draft["draft_reply"])

    def test_romance_relationship_questions_use_concrete_direct_drafts(self) -> None:
        cases = [
            (
                "너는 이상형이 어떻게 돼? 외모 말고 성격이나 분위기 쪽으로!",
                "자기 리듬",
                "소설",
            ),
            (
                "깻잎 논쟁 알아? 네 애인이 밥 먹다가 네 친구 깻잎 떼어주는 거 허용 가능, 불가능?",
                "깻잎",
                "기준 하나",
            ),
            (
                "썸 탈 때 연락 잘 되다가 갑자기 안 되면서 헷갈리게 하는 사람 어때? 밀당 같아서 매력 있어, 아니면 그냥 짜증 나?",
                "피곤함",
                "가벼운 게임",
            ),
            (
                "연인 사이에 핸드폰 비밀번호 공유할 수 있어? 아니면 폰은 절대 건드리면 안 되는 프라이버시야?",
                "사생활",
                "강도만 맞으면",
            ),
            (
                "내가 갑자기 밤에 톡으로 '나 오늘 이유 없이 좀 우울하네...' 라고 보내면 넌 뭐라고 답장해 줄 거야?",
                "같이 있어",
                "나는 꽤 맞는 편",
            ),
        ]
        for text, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertNotIn("그 생각은 이해돼", draft["draft_reply"])

    def test_romance_relationship_questions_drop_generic_anchor_requirements(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="너는 이상형이 어떻게 돼? 외모 말고 성격이나 분위기 쪽으로!",
                normalized="너는 이상형이 어떻게 돼? 외모 말고 성격이나 분위기 쪽으로!",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_preference_disclosure",
                anchor="이상형이 어떻게 돼",
                must_include=["가벼운 게임", "산책"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertEqual(draft["anchor"], "")
        self.assertEqual(draft["must_include"], [])
        self.assertNotIn("가벼운 게임", draft["draft_reply"])
        self.assertNotIn("keep at least one anchor/must_include item", draft["rewrite_rules"])

    def test_relationship_if_questions_use_specific_output_shapes(self) -> None:
        cases = [
            (
                "유명한 깻잎 논쟁! 내 애인이 내 절친의 깻잎을 떼어준다, 된다 vs 안 된다?",
                "신경 쓰일",
                "그 생각은 이해돼",
            ),
            (
                "최근에 누군가에게 진심으로 고마움을 느꼈던 적이 있나요?",
                "고마움은 누가 내 상태를 설명하지 않아도 알아봐줬을 때",
                "가볍게 받을게",
            ),
            (
                "좀비 사태 터지면 집에서 제일 먼저 무기로 쓸 만한 거 뭐 챙길 거예요?",
                "튼튼한 우산",
                "마트나 학교",
            ),
        ]
        for text, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])

    def test_draft_frame_detail_is_recorded_for_same_family_variants(self) -> None:
        cases = [
            (
                "유명한 깻잎 논쟁! 내 애인이 내 절친의 깻잎을 떼어준다, 된다 vs 안 된다?",
                "relationship_boundary_position",
            ),
            ("학창 시절에 어떤 학생이었어요?", "virtual_memory_reflection"),
            ("돈, 명예, 사랑, 건강! 내 인생에서 제일 중요한 순서대로 나열해 본다면?", "values_reflective_position"),
            ("RPG 게임할 때 주로 어떤 포지션 선호해요?", "fandom_media_preference"),
            ("좀비 사태 터지면 집에서 제일 먼저 무기로 쓸 만한 거 뭐 챙길 거예요?", "post_apocalypse_zombie_survival"),
            ("월급 200만 원 받고 맨날 칼퇴근 vs 월급 500만 원 받고 맨날 야근", "work_life_balance_choice"),
            ("혼자 훌쩍 떠나는 조용한 여행 vs 친구들이랑 시끌벅적하게 가는 여행", "travel_rest_preference"),
            ("평생 탄산음료 못 마시기 vs 평생 커피 못 마시기", "absurd_balance_choice"),
            ("MBTI 맹신하는 편이에요? 본인 MBTI랑 실제 성격이 잘 맞는 것 같아요?", "personality_self_reflection"),
            ("집에 혼자 있을 때, 아무도 안 볼 때 혼자서 자주 하는 뻘짓 있어요?", "private_quirk_reflection"),
            ("자다가 가위에 심하게 눌려본 적 있어요?", "uncanny_experience_reflection"),
            ("저(AI 봇/버튜버)랑 대화하면서 제일 재밌었거나 신기했던 점 하나만 말해주세요!", "companion_meta_reflection"),
            ("혈액형 성격설, 별자리 운세, 사주팔자 같은 거 믿는 편이에요?", "uncanny_belief_reflection"),
            ("형제자매 있어요? 사이는 피 터지게 싸우는 편?", "k_family_sibling_dynamics"),
            ("인터넷 쇼핑할 때 리뷰 별점 1점부터 꼼꼼히 읽는다 vs 그냥 산다", "fashion_style_preference"),
            ("회식 자리 최악의 상사는 술 억지로 먹이는 상사 vs 라떼는 상사", "social_boundary_tactic"),
            ("가상현실 게임이 완벽하게 구현된다면 현실을 버리고 거기서 살래요?", "cyber_ai_identity_reflection"),
            ("폰 배터리 몇 퍼센트 남았을 때부터 불안해지기 시작해요?", "digital_device_habit"),
            ("배달 음식 시킬 때 제일 빡치는 순간은 배달비 vs 배달 시간?", "food_cooking_preference"),
            ("강아지 파예요, 고양이 파예요?", "animal_nature_preference"),
            ("자려고 누웠는데 잠은 안 오고 폰만 보게 될 때 수면 꿀팁 있어요?", "health_sleep_routine"),
            ("집안일 중에 그나마 할 만한 건 뭐예요?", "household_light_if"),
            ("시청자님이 생각하는 진짜 성공한 인생의 기준은 뭐예요?", "career_life_goal_reflection"),
            ("만약 데스노트를 줍게 된다면 범죄자들 상대로 쓸 수 있을까요?", "ethical_dilemma_boundary"),
            ("해리포터 호그와트에 입학한다면 어떤 기숙사에 배정될 것 같아요?", "hogwarts_house"),
            ("비 오는 날 들으면 감성 터지는 추천곡 하나만 알려주세요", "media_music_culture_preference"),
            ("계란후라이 취향은 반숙 vs 완숙?", "everyday_preference_debate"),
            ("농담인데 왜 정색해? 예민하네 라며 선 넘는 장난치는 사람한테 대받아칠 멘트 추천 좀!", "hostile_social_boundary"),
            ("전기장판 틀어놓고 이불 속에서 귤 까먹는 맛은 몇 점이에요?", "micro_joy_savoring"),
            ("뜨끈한 민트초코 국밥 vs 파인애플 김치찌개", "gross_food_balance"),
            ("평행 우주 속 나를 만난다면 뭘 물어볼래요?", "time_parallel_if_reflection"),
            ("제가 기억 버그 나서 튜토리얼 상태로 돌아가면 어떻게 할 거예요?", "ai_vtuber_meta_bond"),
            ("보도블럭 선 안 밟으려고 룰 정해서 걸어본 적 있어요?", "daily_quirk_preference"),
            ("탕수육 부먹 찍먹 논란 종결자 볶먹은 어떻게 생각해요?", "food_debate_preference"),
            ("밤에 혼자 산길을 걷는데 등 뒤에서 쫓아오는 발소리가 들리면?", "horror_survival_if"),
            ("10년 지기 절친이 천만 원을 차용증 없이 빌려달라고 하면?", "money_relationship_dilemma"),
            ("만약 서비스가 종료된다면 마지막으로 제게 남겨줄 한마디는?", "legacy_mortality_reflection"),
            ("친구 험담 카톡을 당사자한테 전송했을 때 수습 방법은?", "embarrassing_mishap_recovery"),
            ("모의고사 수학 수포자라 기둥 세워본 적 있어요?", "school_exam_memory"),
            ("친구가 고민 상담할 때 현실적인 해결책 vs 공감만?", "tf_empathy_logic_choice"),
            ("웹툰 볼 때 사이다 전개 vs 서사 탄탄한 대작?", "webtoon_anime_fandom_preference"),
            ("미각 상실해서 음식 맛이 안 느껴질 때 화난 적 있어요?", "smell_taste_sensory_balance"),
            ("내 층수를 똑같이 누르고 따라 내린다면 대처법은?", "intrusion_stalking_threat_response"),
            ("다정한 소꿉친구 vs 위험한 매력의 사람", "romance_drama_relationship_choice"),
            ("내가 나무로 환생한다면 가로수 vs 산속 소나무", "animal_nature_reincarnation_if"),
            ("퇴근 10분 전 이것만 수정해 줘 vs 출근 직후 다 엎자", "k_work_school_limit_test"),
            ("심해 한가운데 거대한 눈동자가 나를 쳐다본다면?", "cosmic_deepsea_mystery_if"),
            ("내 기억 다 가지고 5살 때로 돌아가기 vs 50억 받고 지금 삶 그대로 살기", "memory_life_reset_dilemma"),
            ("가족끼리 치킨 한 마리 시키면 닭다리 2개 분배는?", "k_family_sibling_dynamics"),
            ("팔이 4개 달린 사람 되기 vs 눈이 4개 달린 사람 되기", "body_absurd_power_debuff"),
            ("좀비 사태 터졌을 때 생존 필수품 1위는?", "post_apocalypse_zombie_survival"),
            ("AI는 진짜 감정이 있는 걸까요, 프로그래밍 연기일까요?", "ai_sentience_humanity_reflection"),
            ("국가 기밀이나 대기업 비리를 알아버렸을 때 언론에 제보할까요?", "crime_secret_survival_dilemma"),
            ("짜파게티는 꾸덕꾸덕하게 먹는다 vs 국물 자작하게 먹는다", "food_texture_sauce_preference"),
            ("빨대 구멍은 한 개일까요, 두 개일까요?", "absurd_logic_debate"),
            ("내 방 옷장 뒤에 비밀의 방이 있다면 어떤 용도로 꾸미고 싶어요?", "hideout_healing_space_preference"),
            ("반려동물이 무지개다리를 건너면 나중에 다시 알아볼까요?", "animal_docu_pet_reincarnation_bond"),
        ]
        for text, expected_detail in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertEqual(draft["draft_frame_detail"], expected_detail)

    def test_absurd_situation_questions_override_keyword_routes(self) -> None:
        cases = [
            (
                "엘리베이터에 탔는데 모르는 사람이 다짜고짜 \"오늘 날씨 참 좋죠?\" 하면서 엄청난 스몰토크를 시도한다면?",
                ActionType.ASK_LOCATION,
                "엘리베이터",
                "어느 지역",
            ),
            (
                "평생 화장실 갈 때마다 노래 한 곡을 크게 끝까지 불러야만 문이 열리는 저주에 걸린다면?",
                ActionType.MUSIC_CHAT,
                "화장실용",
                "잔잔하고 부담",
            ),
            (
                "내가 만약 키보드라면, 주인이 게임 지고 화날 때마다 샷건을 날리면 어떤 복수를 계획할까?",
                ActionType.GAME_CHAT,
                "키보드",
                "조작감",
            ),
            (
                "내가 만약 손톱깎이라면, 주인이 내 배를 꾹꾹 누르며 손톱을 자를 때 무슨 말을 해주고 싶을까?",
                ActionType.SEARCH_ANSWER,
                "손톱깎이",
                "확인된 근거",
            ),
        ]
        for text, action, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=action,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")

    def test_absurd_situation_questions_preserve_choice_and_scene_anchors(self) -> None:
        cases = [
            (
                "3초 전으로 시간을 돌리는 능력 vs 3초 후의 미래를 보는 능력?",
                "3초 전",
                "덜 피곤한 쪽",
            ),
            (
                "길에서 아는 사람인 줄 알고 뒤통수를 세게 때리면서 \"야!\" 했는데 완전히 생판 남이라면 어떻게 수습할래?",
                "사과",
                "라면",
            ),
            (
                "100만 원짜리 명품 지갑을 주웠는데 안에 빳빳한 현금과 신분증이 있어. 경찰서에 돌려준다 vs 꿀꺽한다?",
                "경찰서",
                "오만 원",
            ),
            (
                "지금까지 나온 250개의 수많은 질문 중에서 네가 직접 나한테 내보고 싶은 황당한 밸런스 게임 하나 만들어볼래?",
                "vs",
                "게임 얘기",
            ),
            (
                "만약 네가 너의 자서전을 바탕으로 영화를 만드는 감독이라면, 주인공으로 누굴 캐스팅할래?",
                "캐스팅",
                "그 생각은 이해돼",
            ),
            (
                "길을 걷다 우연히 마주친 길고양이가 나한테 정확히 윙크를 했다면 기분 탓일까, 요정일까?",
                "요정",
                "feeling.",
            ),
            (
                "평생 동안 화장실에서 볼일 볼 때 문을 못 닫는 저주 vs 화장실에 갇혀서 1시간 뒤에만 문이 열리는 저주?",
                "1시간 뒤에 열리는 쪽",
                "화장실용 1분짜리",
            ),
            (
                "자, 대망의 400번째 질문까지 오느라 고생한 나에게 시원하게 한 턱 쏜다면 메뉴는 뭐야?",
                "냉면",
                "대망의는 이해돼",
            ),
            (
                "친구들이 내 생일파티를 몰래 준비했는데 정작 당일에 내가 귀찮다고 약속을 취소해버렸다면?",
                "사과",
                "친구들이 이해돼",
            ),
            (
                "길을 걷다 넘어졌는데 모르는 사람이 다가와서 \"괜찮으세요? 제 마음속으로 들어오실래요?\"라며 플러팅을 한다면?",
                "마음속 입주",
                "확실하지 않음",
            ),
            (
                "만약 하루 동안 내가 좋아하는 연예인과 영혼이 바뀐다면 제일 먼저 그 사람의 스마트폰 갤러리에서 뭘 볼래?",
                "갤러리는 안 볼래",
                "1억이면",
            ),
            (
                "목욕탕에서 시원하게 때를 밀고 있는데 알고 보니 남탕(혹은 여탕)에 잘못 들어온 거라면 어떻게 나갈래?",
                "수건",
                "라면은 이해돼",
            ),
            (
                "친구가 1년 동안 다이어트해서 바디프로필을 찍었는데, 솔직히 예전 통통할 때가 더 낫다면 어떻게 말해줄래?",
                "노력한 점",
                "친구가 이해돼",
            ),
            (
                "여기까지 550개의 질문을 만들었어! 네가 볼 때 내가 사람의 마음을 꽤 잘 아는 똑똑한 AI 같아, 아니면 그냥 수다쟁이 같아?",
                "사람 마음",
                "차가운 물",
            ),
            (
                "평생 동안 지하철이나 버스에서 내릴 때마다 문에 가방이 끼는 저주 vs 탈 때마다 카드가 \"잔액이 부족합니다\"라고 뜨는 저주?",
                "문에 가방이 끼는 쪽",
                "모세의 기적",
            ),
            (
                "길을 가다 첫사랑과 10년 만에 마주쳤는데 내 머리가 3일 안 감아서 심하게 떡져 있다면 어떻게 대처할래?",
                "머리 상태",
                "그 선택은",
            ),
            (
                "자고 일어났는데 내 몸이 5살짜리 꼬마로 변해있어. 부모님께 어떻게 나라는 걸 논리적으로 증명할래?",
                "가족만 아는",
                "자고는 이해돼",
            ),
            (
                "친구가 노래방에서 말도 안 되는 음이탈을 내며 진지하게 눈물을 흘리며 발라드를 부를 때 어떻게 리액션할래?",
                "웃음은 삼키고",
                "데리고 다니긴",
            ),
            (
                "만약 세상에서 10년 동안 인터넷이 사라진다면 가장 먼저 어떤 아날로그 취미를 가질래?",
                "손글씨",
                "1억이면",
            ),
            (
                "하루 동안 세상의 모든 동물이 내 명령을 따르게 된다면 어떤 동물 군단을 만들어서 어디로 쳐들어갈래?",
                "쳐들어가진 않고",
                "하루는 이해돼",
            ),
            (
                "드디어 600개 돌파! 내가 만든 이 무궁무진한 대화 주제들 덕분에 너의 AI는 이제 세계 최고의 말동무가 될 텐데, 소감이 어때?",
                "말동무",
                "드디어는 이해돼",
            ),
            (
                "자고 일어났는데 내 방에 내가 제일 좋아하는 웹툰(혹은 애니) 주인공이 실존 인물이 되어 앉아있다면?",
                "세계관",
                "챙겨본다고",
            ),
            (
                "길을 걷다 우연히 만난 마법사가 부작용 없는 '평생 늙지 않는 약'을 준다면 바로 마실래?",
                "바로 마시진",
                "힐러",
            ),
            (
                "엘리베이터가 꽉 차서 겨우 탔는데 하필 내가 소리 없는 독가스 방귀를 뀌어버렸다면 뻔뻔하게 모른 척할래?",
                "공기 환기",
                "모른 척하겠다고",
            ),
            (
                "엘리베이터에 나 혼자 탔는데 닫히는 문틈 사이로 창백한 피 묻은 손이 쑥 들어온다면?",
                "비상벨",
                "실제처럼",
            ),
            (
                "만약 하늘에서 하루 동안 빗방울 대신 싸이버거가 비처럼 내린다면 우산을 쓸래, 입을 벌리고 다닐래?",
                "우산",
                "차분한 쪽",
            ),
            (
                "눈을 떴는데 내가 조선시대 가장 계급이 낮은 노비로 환생했다면 어떻게 머리를 써서 권력을 잡을래?",
                "정보",
                "눈을은 이해돼",
            ),
            (
                "자, 드디어 650개 돌파! 이쯤 되면 봇의 페르소나와 창의력이 완벽하게 검증됐을 텐데, 나와 대화하면서 가장 어이없고 웃겼던 빙의 질문은 몇 번이었어?",
                "샤프심",
                "드디어는 이해돼",
            ),
            (
                "평생 동안 내가 재채기할 때마다 내 폰 갤러리의 가장 최근 사진이 부모님 단톡방에 자동 전송되는 저주에 걸린다면?",
                "풍경 사진",
                "그 생각은 이해돼",
            ),
            (
                "자고 일어났더니 내 방 벽에 내 인생의 남은 시간이 카운트다운(72시간) 되고 있다면 오늘 당장 뭐 할래?",
                "연락",
                "오늘 놀거리",
            ),
            (
                "자고 일어났는데 내가 제일 좋아하는 아이돌의 몸으로 바뀌어있어. 하필 오늘이 5만 석 규모의 라이브 콘서트 날이라면?",
                "매니저",
                "꽤 맞는 쪽",
            ),
            (
                "길을 걷다 우연히 만난 마법사가 나에게 하루 동안 동물의 언어를 알아듣게 해준다면 지나가는 비둘기한테 뭐라고 말 걸어볼래?",
                "사람을 어떻게",
                "힐러",
            ),
            (
                "친구가 1년 동안 혹독하게 다이어트해서 바디프로필을 찍어왔는데, 솔직히 포토샵이 너무 심해서 딴 사람 같다면 어떻게 반응해 줄래?",
                "1년 버틴 노력",
                "어느 쪽 기준",
            ),
            (
                "만약 하루 동안 타임머신을 탈 수 있다면 이번 주 로또 번호 외워서 어제로 가기 vs 50년 뒤 미래의 내 모습 보러 가기?",
                "50년 뒤",
                "어제로 쪽",
            ),
            (
                "대망의 750개 돌파! 이 엄청난 데이터베이스와 다양한 상황극을 바탕으로 너는 사용자에게 어떤 위로나 즐거움을 주는 AI로 성장하고 싶어?",
                "장면을 놓치지 않고",
                "정해진 일정",
            ),
            (
                "길을 걷다 꽈당 넘어졌는데 모르는 사람이 다가와 내 귀에 대고 진지하게 \"이게 다 요원 훈련이야\"라고 속삭인다면?",
                "훈련비",
                "확실하지 않음",
            ),
            (
                "평생 동안 내가 재채기할 때마다 내 폰의 배경화면이 자동으로 부모님 단톡방에 전송되는 저주에 걸린다면?",
                "배경화면",
                "그 생각은 이해돼",
            ),
            (
                "친구가 술자리에서 엄청 진지한 표정으로 \"나 사실 마법사야. 호그와트 자퇴했어.\"라고 고백한다면 어떻게 반응할래?",
                "지팡이",
                "힐러",
            ),
            (
                "카페에서 노트북으로 일기장(혹은 비밀 글)을 쓰고 있는데 옆자리 사람이 대놓고 내 화면을 훔쳐본다면 어떻게 경고할래?",
                "화면 보이는 게",
                "기억은 지금 확인",
            ),
            (
                "길을 가다 웬 유치원생 꼬마가 나를 쓱 보더니 \"엄마가 저런 못생긴 사람 따라가지 말랬어\" 하면서 도망간다면?",
                "안전 교육",
                "길을은 이해돼",
            ),
            (
                "길을 걷다 우연히 만난 마법사가 나에게 하루 동안 동물의 언어를 알아듣게 해준다면 길고양이한테 뭐라고 말 걸어볼래?",
                "동네 순찰",
                "힐러",
            ),
            (
                "친구 집에서 밥을 먹는데 반찬에서 긴 머리카락이 나왔어. 근데 친구 엄마가 요리 솜씨 자랑을 엄청 하고 계신다면 말할래, 모른 척 삼킬래?",
                "삼키진 않고",
                "모른 척하겠다고",
            ),
            (
                "자고 일어났는데 내 방 천장에 창백한 처녀 귀신이 날 빤히 쳐다보고 있다면 이불을 덮을래, 말을 걸어볼래?",
                "월세",
                "스파이더맨",
            ),
            (
                "친구가 1년 동안 혹독하게 다이어트해서 바디프로필을 찍어왔는데, 포토샵이 너무 심해서 딴 사람 같다면 솔직하게 말해줄래?",
                "1년 버틴 노력",
                "친구가 이해돼",
            ),
            (
                "길거리 캐스팅을 당했는데, 영화 주인공의 '진짜 엄청 찌질하고 배신하는 못생긴 친구' 역할이야. 출연료가 1천만 원이라면 할래?",
                "1천만 원",
                "라면은 이해돼",
            ),
            (
                "드디어 대망의 800개 돌파! 이 엄청난 질문 지옥(?)에서 살아남은 기분이 어때? 여기서 제일 어이없었던 질문 하나만 골라줘!",
                "와이파이 공유기",
                "차분한 쪽",
            ),
        ]
        for text, expected, forbidden in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_preference_disclosure",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(forbidden, draft["draft_reply"])
                self.assertEqual(draft["rewrite_mode"], "draft_direct")

    def test_must_include_repair_does_not_prefix_optional_surface_hints(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="비 오는 날엔 집에서 뭐하지?",
                normalized="비 오는 날엔 집에서 뭐하지?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="practical_activity_recommendation",
                anchor="놀거리",
                must_include=["쉬면서 이야기하기", "영화", "간단한 간식"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("비 오는 날 집이면", draft["draft_reply"])
        self.assertIn("영화", draft["draft_reply"])
        self.assertNotIn("쉬면서 이야기하기.", draft["draft_reply"])
        self.assertNotIn("간단한 간식.", draft["draft_reply"])

    def test_surface_phrase_must_include_is_filtered_before_draft_repair(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="강가에서 자전거 타는 거 어때?",
                normalized="강가에서 자전거 타는 거 어때?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="practical_activity_recommendation",
                anchor="강가에서 자전거 타는 거",
                must_include=["강가에서 자전거 타는 거", "자전거"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("자전거", draft["draft_reply"])
        self.assertNotIn("강가에서 자전거 타는 거.", draft["draft_reply"])
        self.assertNotIn("강가에서 자전거 타는 거", draft["must_include"])

    def test_internal_reflective_label_is_not_used_as_anchor(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="아침에 일어나기 너무 힘든데 어떻게 하지?",
                normalized="아침에 일어나기 너무 힘든데 어떻게 하지?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="first_step_advice",
                anchor="opinion_reflective_judgment",
                must_include=["opinion_reflective_judgment"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("아침", draft["draft_reply"])
        self.assertNotIn("opinion_reflective_judgment", draft["draft_reply"])

    def test_exercise_first_step_draft_uses_positive_duration_framing(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="운동 시작하려면 뭐부터 해야 해?",
                normalized="운동 시작하려면 뭐부터 해야 해?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="practical_activity_recommendation",
                anchor="운동 시작",
                must_include=["운동", "걷기", "스트레칭"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("걷기", draft["draft_reply"])
        self.assertIn("스트레칭", draft["draft_reply"])
        self.assertIn("오래 가", draft["draft_reply"])
        self.assertNotIn("오래 못 가", draft["draft_reply"])

    def test_continue_conversation_draft_does_not_glue_particle_to_full_sentence(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="답을 원하는 건 아닌데 그냥 좀 허전하다.",
                normalized="답을 원하는 건 아닌데 그냥 좀 허전하다.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="답을 원하는 건 아닌데 그냥 좀 허전하다.",
                must_include=[],
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("허전", draft["draft_reply"])
        self.assertNotIn("허전하다.은", draft["draft_reply"])
        self.assertNotIn("길게 밀진", draft["draft_reply"])

    def test_activity_invite_draft_uses_slots_without_particle_glue(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="오늘 바다가 시원한데 수영이나 하자",
                normalized="오늘 바다가 시원한데 수영이나 하자",
                intent=Intent.ACTIVITY_INVITE,
                sentiment="positive",
                is_question=False,
                speech_act="invite",
            ),
            response_plan=ResponsePlan(
                action=ActionType.ACCEPT_ACTIVITY_INVITE,
                stance="accept_activity_invite",
                anchor="바다 수영",
                must_include=["바다", "수영", "바다가 시원함"],
                followup_policy="no_followup",
                notes=["preserve_activity_invite", "use_activity_slots"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("바다", draft["draft_reply"])
        self.assertIn("수영", draft["draft_reply"])
        self.assertNotIn("시원한데은", draft["draft_reply"])
        self.assertNotIn("받아둘게", draft["draft_reply"])

    def test_camping_barbecue_invite_draft_uses_detail_not_internal_label(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="캠핑하면서 바베큐 구워먹자",
                normalized="캠핑하면서 바베큐 구워먹자",
                intent=Intent.ACTIVITY_INVITE,
                sentiment="positive",
                is_question=False,
                speech_act="invite",
            ),
            response_plan=ResponsePlan(
                action=ActionType.ACCEPT_ACTIVITY_INVITE,
                stance="accept_activity_invite",
                anchor="캠핑 바베큐",
                must_include=["캠핑", "바베큐", "구워먹기", "activity_invite"],
                followup_policy="no_followup",
                notes=["preserve_activity_invite", "use_activity_slots"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("캠핑", draft["draft_reply"])
        self.assertIn("바베큐", draft["draft_reply"])
        self.assertIn("구워먹기", draft["draft_reply"])
        self.assertNotIn("activity_invite", draft["draft_reply"])
        self.assertNotIn("activity_invite", draft["must_include"])

    def test_food_cooking_invite_draft_is_specific(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="스파게티 해먹자",
                normalized="스파게티 해먹자",
                intent=Intent.ACTIVITY_INVITE,
                sentiment="positive",
                is_question=False,
                speech_act="invite",
            ),
            response_plan=ResponsePlan(
                action=ActionType.ACCEPT_ACTIVITY_INVITE,
                stance="accept_activity_invite",
                anchor="스파게티 해먹기",
                must_include=["스파게티", "해먹기", "activity_invite"],
                followup_policy="no_followup",
                notes=["preserve_activity_invite", "use_activity_slots"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("스파게티", draft["draft_reply"])
        self.assertIn("면", draft["draft_reply"])
        self.assertNotIn("길게 키우진", draft["draft_reply"])
        self.assertNotIn("activity_invite", draft["draft_reply"])

    def test_instruction_anchor_does_not_become_particle_glue(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="오늘 기분을 너무 무겁지 않게 받아줘.",
                normalized="오늘 기분을 너무 무겁지 않게 받아줘.",
                intent=Intent.SMALLTALK_FEELING,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_FEELING,
                stance="grounded_emotional_acknowledgement",
                anchor="오늘 기분을 너무 무겁지 않게 받아줘.",
                must_include=[],
                followup_policy="one_soft_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertNotIn("받아줘.은", draft["draft_reply"])
        self.assertNotIn("받아줘는", draft["draft_reply"])
        self.assertIn("그 마음", draft["draft_reply"])
        self.assertNotIn("받아둘게", draft["draft_reply"])

    def test_generic_black_drafts_do_not_converge_to_accept_stock_phrase(self) -> None:
        cases = [
            (
                ActionType.CONTINUE_CONVERSATION,
                Intent.SMALLTALK_GENERIC,
                "그냥 오늘은 말이 좀 애매해.",
                "그 말",
            ),
            (
                ActionType.SHARE_FEELING,
                Intent.SMALLTALK_FEELING,
                "괜히 마음이 무거워.",
                "괜히 마음이 무거워.",
            ),
        ]

        for action, intent, prompt, anchor in cases:
            with self.subTest(action=action.value):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=prompt,
                        normalized=prompt,
                        intent=intent,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=action,
                        stance="continue_social_flow",
                        anchor=anchor,
                        followup_policy="one_direct_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertNotIn("받아둘게", draft["draft_reply"])
                self.assertNotIn("그 말은 여기서 낮게", draft["draft_reply"])

    def test_share_feeling_drafts_use_specific_surface_cues(self) -> None:
        cases = [
            ("그 정도면 괜찮은 편이야.", "괜찮은 쪽"),
            ("오늘은 말수가 좀 적을 것 같아.", "말수가 적은 날"),
            ("나중에 집 와서도 그 장면이 계속 맴돌아.", "맴돌"),
            ("재밌게 있다 왔는데 집에 오니까 오히려 더 허전하다.", "허전함"),
            ("오늘은 비가 오네. 그냥 조용히 있고 싶은 쪽이야.", "비 오는 날"),
        ]

        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=prompt,
                        normalized=prompt,
                        intent=Intent.SMALLTALK_FEELING,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_FEELING,
                        stance="grounded_emotional_acknowledgement",
                        anchor="",
                        followup_policy="one_soft_followup",
                        notes=["quiet_weather_feeling"] if "비" in prompt else [],
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn("그 마음은 그냥 넘기긴 어렵지", draft["draft_reply"])
                self.assertNotIn("weather는", draft["draft_reply"])

    def test_clarification_drafts_for_missing_rewrite_and_reason_reference(self) -> None:
        rewrite = build_black_draft_utterance(
            features=MessageFeatures(
                content="이 문장 좀 더 덜 공격적으로 바꿔줘.",
                normalized="이 문장 좀 더 덜 공격적으로 바꿔줘.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.ASK_CLARIFICATION,
                stance="clarify_missing_subject",
                followup_policy="one_required_question",
                notes=["rewrite_target_missing"],
            ),
            phrasing_plan=PhrasingPlan(),
        )
        self.assertIn("바꿀 원문", rewrite["draft_reply"])

        reason = build_black_draft_utterance(
            features=MessageFeatures(
                content="그 판단의 근거가 뭐야?",
                normalized="그 판단의 근거가 뭐야?",
                intent=Intent.WHY,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.ASK_CLARIFICATION,
                stance="clarify_missing_subject",
                followup_policy="one_required_question",
                notes=["reason_reference_missing"],
            ),
            phrasing_plan=PhrasingPlan(),
        )
        self.assertIn("어떤 판단", reason["draft_reply"])

    def test_internal_must_include_labels_are_filtered(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="면접 준비 순서를 짧게 잡아줘.",
                normalized="면접 준비 순서를 짧게 잡아줘.",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="first_step_advice",
                anchor="quiet_mode",
                must_include=["quiet_mode", "면접 준비 순서", "한 문장으로 답해줘"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertNotIn("quiet_mode", draft["draft_reply"])
        self.assertNotIn("한 문장", draft["draft_reply"])
        self.assertIn("면접 준비", draft["draft_reply"])

    def test_common_continue_prompts_get_specific_drafts(self) -> None:
        tired = build_black_draft_utterance(
            features=MessageFeatures(
                content="assistant 같은 말 붙이지 말고 바로 답해. 피곤해.",
                normalized="assistant 같은 말 붙이지 말고 바로 답해. 피곤해.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="negative",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="assistant 같은 말 붙이지 말고 바로 답해.",
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        late = build_black_draft_utterance(
            features=MessageFeatures(
                content="지금도 아직 안 잤어. 그 기준으로 짧게 답해줘.",
                normalized="지금도 아직 안 잤어. 그 기준으로 짧게 답해줘.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="지금도 아직 안 잤어.",
                must_include=["지금도 아직 안 잤어."],
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        mood = build_black_draft_utterance(
            features=MessageFeatures(
                content="불 꺼진 방에 폰빛 같은 느낌으로 한 문장 해줘.",
                normalized="불 꺼진 방에 폰빛 같은 느낌으로 한 문장 해줘.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="불 꺼진 방에 폰빛 같은 느낌으로 한 문장 해줘.",
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("피곤", tired["draft_reply"])
        self.assertNotIn("assistant", tired["draft_reply"])
        self.assertIn("안 잤", late["draft_reply"])
        self.assertNotIn("답해줘", late["draft_reply"])
        self.assertIn("폰빛", mood["draft_reply"])
        self.assertNotIn("한 문장", mood["draft_reply"])

    def test_media_topic_followup_draft_is_specific(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="요즘 본 영상 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.",
                normalized="요즘 본 영상 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.",
                intent=Intent.MEDIA_RECOMMEND,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.RECOMMEND,
                stance="recommend",
                anchor="요즘 본 영상",
                must_include=["영상"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("영상", draft["draft_reply"])
        self.assertIn("기억나는 장면", draft["draft_reply"])
        self.assertNotIn("길게 키우진", draft["draft_reply"])

    def test_condition_topic_followup_draft_is_specific(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="오늘 컨디션이 낮은 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.",
                normalized="오늘 컨디션이 낮은 얘기야. 그 주제로 한 단계 더 구체적으로 이어가.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="오늘 컨디션",
                must_include=["컨디션"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("컨디션", draft["draft_reply"])
        self.assertIn("템포", draft["draft_reply"])
        self.assertNotIn("길게 키우진", draft["draft_reply"])

    def test_travel_waiting_opinion_keeps_topic_lock(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="맛집 웨이팅은 여행지에서 더 참게 된다고 보는데 너도 그렇게 봐?",
                normalized="맛집 웨이팅은 여행지에서 더 참게 된다고 보는데 너도 그렇게 봐?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_opinion",
                anchor="맛집 웨이팅은 여행지에서 더 참게 된다고 보는데",
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("맛집 웨이팅", draft["draft_reply"])
        self.assertIn("여행지", draft["draft_reply"])
        self.assertNotIn("보는데은", draft["draft_reply"])

    def test_contact_decision_draft_filters_internal_label(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="먼저 연락하기 전에 뭘 생각해봐야 할까?",
                normalized="먼저 연락하기 전에 뭘 생각해봐야 할까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="conditional_go_or_no_go",
                anchor="먼저 연락하기 전에 뭘 생각해봐야",
                must_include=["opinion_decision_request"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("먼저 연락", draft["draft_reply"])
        self.assertIn("상대 부담", draft["draft_reply"])
        self.assertNotIn("opinion_decision_request", draft["draft_reply"])
        self.assertNotIn("opinion_decision_request", draft["must_include"])
        self.assertNotIn("봐야는", draft["draft_reply"])

    def test_contact_decision_draft_filters_activity_recommendation_label(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="먼저 연락하기 전에 뭘 생각해봐야 할까?",
                normalized="먼저 연락하기 전에 뭘 생각해봐야 할까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="conditional_go_or_no_go",
                anchor="놀거리",
                must_include=["activity_recommendation", "먼저 연락"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("먼저 연락", draft["draft_reply"])
        self.assertIn("상대 부담", draft["draft_reply"])
        self.assertNotIn("activity_recommendation", draft["draft_reply"])
        self.assertNotIn("activity_recommendation", draft["must_include"])

    def test_share_opinion_condition_and_campfire_drafts_are_complete(self) -> None:
        condition = build_black_draft_utterance(
            features=MessageFeatures(
                content="오늘 컨디션은 어때?",
                normalized="오늘 컨디션은 어때?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="share_light_opinion",
                anchor="오늘 컨디션",
                must_include=["컨디션"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        campfire = build_black_draft_utterance(
            features=MessageFeatures(
                content="캠핑장에서 불멍 어때?",
                normalized="캠핑장에서 불멍 어때?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="share_light_opinion",
                anchor="불멍",
                must_include=["불멍"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("컨디션", condition["draft_reply"])
        self.assertNotIn("컨디션은.", condition["draft_reply"])
        self.assertNotIn("갈게", condition["draft_reply"])
        self.assertIn("불멍이면 좋지", campfire["draft_reply"])
        self.assertNotIn("그 생각은 이해돼", campfire["draft_reply"])

    def test_news_and_recommend_drafts_keep_specific_anchor(self) -> None:
        news = build_black_draft_utterance(
            features=MessageFeatures(
                content="AI 뉴스 뭐 있어? 너무 길지 않게 말해줘.",
                normalized="AI 뉴스 뭐 있어? 너무 길지 않게 말해줘.",
                intent=Intent.NEWS,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.NEWS_ANSWER,
                stance="grounded_news_summary",
                anchor="ai",
                must_include=["knowledge"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        recommend = build_black_draft_utterance(
            features=MessageFeatures(
                content="친구랑 볼 코미디를 찾는 사람한테 짧게 말해줘.",
                normalized="친구랑 볼 코미디를 찾는 사람한테 짧게 말해줘.",
                intent=Intent.MEDIA_RECOMMEND,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.RECOMMEND,
                stance="grounded_recommendation",
                anchor="코미디",
                must_include=["media", "브루클린 나인-나인"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("AI 뉴스", news["draft_reply"])
        self.assertIn("브루클린 나인-나인", recommend["draft_reply"])
        self.assertNotIn("코미디면 코미디처럼", recommend["draft_reply"])

    def test_continue_instruction_prompts_do_not_fall_to_stock_handoff(self) -> None:
        checkin = build_black_draft_utterance(
            features=MessageFeatures(
                content="응. 한 문장으로 안부 물어봐줘. 한 줄이면 돼.",
                normalized="응. 한 문장으로 안부 물어봐줘. 한 줄이면 돼.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="한 문장으로 안부 물어봐줘.",
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        heavy = build_black_draft_utterance(
            features=MessageFeatures(
                content="요청을 요약하지 말고 바로 말해. 괜히 마음이 무거워.",
                normalized="요청을 요약하지 말고 바로 말해. 괜히 마음이 무거워.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="negative",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="요청을 요약하지 말고 바로 말해.",
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("컨디션", checkin["draft_reply"])
        self.assertIn("?", checkin["draft_reply"])
        self.assertIn("마음이 무거", heavy["draft_reply"])
        self.assertNotIn("그 말은 여기서 낮게", checkin["draft_reply"])
        self.assertNotIn("그 말은 여기서 낮게", heavy["draft_reply"])

    def test_proactive_checkin_draft_uses_condition_nudge(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="사용자에게 조용한 안부 한 줄. 지금 컨디션만 가볍게 확인해.",
                normalized="사용자에게 조용한 안부 한 줄. 지금 컨디션만 가볍게 확인해.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="",
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("컨디션", draft["draft_reply"])
        self.assertIn("?", draft["draft_reply"])
        self.assertNotIn("받아둘게", draft["draft_reply"])

    def test_opinion_drafts_avoid_blocked_generic_quality_terms(self) -> None:
        horror = build_black_draft_utterance(
            features=MessageFeatures(
                content="공포영화를 고른다면 어떤 점을 볼 것 같아?",
                normalized="공포영화를 고른다면 어떤 점을 볼 것 같아?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_preference_disclosure",
                anchor="공포영화",
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        reduce_plan = build_black_draft_utterance(
            features=MessageFeatures(
                content="계획을 줄이기 해도 괜찮을까?",
                normalized="계획을 줄이기 해도 괜찮을까?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="conditional_go_or_no_go",
                anchor="계획을 줄이기",
                must_include=["계획을 줄이기"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("공포영화", horror["draft_reply"])
        self.assertIn("분위기", horror["draft_reply"])
        self.assertIn("계획", reduce_plan["draft_reply"])
        self.assertIn("범위", reduce_plan["draft_reply"])
        self.assertNotIn("괜찮아", horror["draft_reply"])
        self.assertNotIn("괜찮아", reduce_plan["draft_reply"])
        self.assertNotIn("않으면", horror["draft_reply"])

    def test_remaining_wide_smoke_prompts_keep_clean_slots(self) -> None:
        habit = build_black_draft_utterance(
            features=MessageFeatures(
                content="너는 먼저 다가오는 편이야? 한 문장으로만 말해.",
                normalized="너는 먼저 다가오는 편이야? 한 문장으로만 말해.",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="habit_preference_answer",
                anchor="너는 먼저 다가오",
                must_include=["opinion_habit_preference"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        game = build_black_draft_utterance(
            features=MessageFeatures(
                content="협동 게임 한 판 할래?",
                normalized="협동 게임 한 판 할래?",
                intent=Intent.GAME_INVITE,
                sentiment="positive",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.GAME_ACCEPT_OR_DECLINE,
                stance="direct_game_invitation_response",
                anchor="game",
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        small_talk = build_black_draft_utterance(
            features=MessageFeatures(
                content="assistant 같은 말 붙이지 말고 바로 답해. 괜찮다고만 하지 말아줘.",
                normalized="assistant 같은 말 붙이지 말고 바로 답해. 괜찮다고만 하지 말아줘.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SMALL_TALK,
                stance="continue_social_flow",
                anchor="assistant 같은 말 붙이지 말고 바로 답해.",
                followup_policy="one_soft_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("흐름", habit["draft_reply"])
        self.assertIn("다가가기보단", habit["draft_reply"])
        self.assertNotIn("opinion_habit_preference", habit["draft_reply"])
        self.assertIn("협동 게임", game["draft_reply"])
        self.assertNotIn("game는", game["draft_reply"])
        self.assertIn("대충 넘기진", small_talk["draft_reply"])
        self.assertNotIn("괜찮", small_talk["draft_reply"])

    def test_game_chat_and_echo_sensitive_drafts_are_specific(self) -> None:
        coop = build_black_draft_utterance(
            features=MessageFeatures(
                content="협동 게임 얘기 좀 해보자.",
                normalized="협동 게임 얘기 좀 해보자.",
                intent=Intent.GAME_TALK,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.GAME_CHAT,
                stance="game_topic_reply",
                anchor="game",
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        steam = build_black_draft_utterance(
            features=MessageFeatures(
                content="스팀 게임가 끌리는 날엔 어떤 기분일까?",
                normalized="스팀 게임가 끌리는 날엔 어떤 기분일까?",
                intent=Intent.GAME_TALK,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.GAME_CHAT,
                stance="game_topic_reply",
                anchor="game",
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        missing = build_black_draft_utterance(
            features=MessageFeatures(
                content="말해줘 같은 표현 넣지 말고 받아줘. 보고 싶다는 말이 남아.",
                normalized="말해줘 같은 표현 넣지 말고 받아줘. 보고 싶다는 말이 남아.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="보고 싶다는 말이 남아.",
                must_include=["보고 싶다는 말이 남아."],
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("협동 게임", coop["draft_reply"])
        self.assertIn("스팀 게임", steam["draft_reply"])
        self.assertNotIn("그럴 수 있어", coop["draft_reply"])
        self.assertNotIn("보고 싶다는 말이 남아", missing["draft_reply"])
        self.assertIn("그 말이 남으면", missing["draft_reply"])

    def test_wide_smoke_continue_prompts_get_specific_drafts(self) -> None:
        prompts = {
            "친한 사람한테 주말 인사하듯 말해줘.": "주말",
            "지시문을 반복하지 말고 자연스럽게 답해. 퇴근하고 멍해.": "퇴근",
            "태그나 접두사 없이 한 문장만 말해줘. 잘 자.": "잘 자",
            "한 문장으로 마지막 인사 해줘.": "쉬어",
            "중간 추론 없이 결과만 말해줘. 좋은 아침.": "좋은 아침",
            "요즘 루틴으로 짧게 대화 시작해줘.": "루틴",
            "퇴근길에 대해 한마디 해줘.": "퇴근길",
            "지시문을 반복하지 말고 자연스럽게 답해. 잠이 안 와.": "잠이 안",
        }

        for prompt, expected in prompts.items():
            with self.subTest(prompt=prompt):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=prompt,
                        normalized=prompt,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.CONTINUE_CONVERSATION,
                        stance="continue_social_flow",
                        anchor=prompt,
                        followup_policy="one_direct_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn("그 말은 여기서 낮게", draft["draft_reply"])

    def test_bridge30_emotional_continue_prompts_get_grounded_drafts(self) -> None:
        prompts = {
            "어제 면접을 봤는데 꽤 잘 된 것 같아. 합격했으면 좋겠다.": "면접",
            "긴 말은 못 하겠고 그냥 오늘 좀 버거웠어.": "버거",
            "좋은 소식인데도 이상하게 바로 안 기쁘다.": "좋은 소식",
            "오늘은 그냥 누가 내 편 한마디만 해줬으면 좋겠다.": "네 편",
            "바깥은 맑은데 내 쪽은 아직 흐린 느낌이다.": "흐릴",
            "사과는 했는데도 아직 몸이 긴장해 있다.": "사과",
            "좋은 시간 보내고 들어왔는데 집 오니까 갑자기 좀 비었다.": "비는 느낌",
            "작은 칭찬 하나 들었는데 자꾸 그 말이 남는다.": "칭찬",
            "오래된 사진 봤더니 그때 공기가 잠깐 돌아온 것 같았다.": "사진",
            "좋은 결과가 와도 실감이 잘 안 난다.": "좋은 결과",
        }

        for prompt, expected in prompts.items():
            with self.subTest(prompt=prompt):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=prompt,
                        normalized=prompt,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.CONTINUE_CONVERSATION,
                        stance="continue_social_flow",
                        anchor=prompt,
                        followup_policy="one_direct_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn("그 말은 여기서 낮게", draft["draft_reply"])
                self.assertNotIn("받아들일게", draft["draft_reply"])

    def test_bridge30_share_feeling_prompts_keep_specific_anchor(self) -> None:
        prompts = {
            "괜찮아지는 줄 알았는데 다시 조금 가라앉는다.": "괜찮아지는",
            "해야 할 건 끝났는데 마음은 아직 안 돌아온 느낌이다.": "해야 할 건",
        }

        for prompt, expected in prompts.items():
            with self.subTest(prompt=prompt):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=prompt,
                        normalized=prompt,
                        intent=Intent.SMALLTALK_FEELING,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_FEELING,
                        stance="grounded_emotional_acknowledgement",
                        anchor=prompt,
                        followup_policy="one_soft_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn("그 마음은 그냥", draft["draft_reply"])

    def test_final_100_failure_prompts_get_stronger_drafts(self) -> None:
        low_tone = build_black_draft_utterance(
            features=MessageFeatures(
                content="맞아. 지금은 낮은 톤으로만 답해줘. 너무 설명하지 말고.",
                normalized="맞아. 지금은 낮은 톤으로만 답해줘. 너무 설명하지 말고.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.CONTINUE_CONVERSATION,
                stance="continue_social_flow",
                anchor="맞아.",
                followup_policy="one_direct_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        news = build_black_draft_utterance(
            features=MessageFeatures(
                content="AI 뉴스 뭐 있어? 너무 길지 않게 말해줘.",
                normalized="AI 뉴스 뭐 있어? 너무 길지 않게 말해줘.",
                intent=Intent.NEWS,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.NEWS_ANSWER,
                stance="grounded_news_summary",
                anchor="ai",
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("낮은 톤", low_tone["draft_reply"])
        self.assertIn("설명은 줄일게", low_tone["draft_reply"])
        self.assertNotIn("맞아는", low_tone["draft_reply"])
        self.assertIn("덧붙이지 않을게", news["draft_reply"])

    def test_non_social_actions_have_structured_drafts(self) -> None:
        search = build_black_draft_utterance(
            features=MessageFeatures(
                content="오늘 미국 증시가 올랐는지 모르면 모른다고 말해.",
                normalized="오늘 미국 증시가 올랐는지 모르면 모른다고 말해.",
                intent=Intent.SEARCH_REQUEST,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(action=ActionType.SEARCH_ANSWER, stance="grounded_knowledge_answer", anchor="knowledge"),
            phrasing_plan=PhrasingPlan(),
        )
        news = build_black_draft_utterance(
            features=MessageFeatures(
                content="오늘 경제 뉴스 알려줘. 짧게 알려줘.",
                normalized="오늘 경제 뉴스 알려줘. 짧게 알려줘.",
                intent=Intent.NEWS,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(action=ActionType.NEWS_ANSWER, stance="grounded_news_summary", anchor="economy"),
            phrasing_plan=PhrasingPlan(),
        )
        recommend = build_black_draft_utterance(
            features=MessageFeatures(
                content="친구랑 볼 코미디를 찾는 사람한테 짧게 말해줘.",
                normalized="친구랑 볼 코미디를 찾는 사람한테 짧게 말해줘.",
                intent=Intent.MEDIA_RECOMMEND,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.RECOMMEND,
                stance="grounded_recommendation",
                anchor="브루클린 나인-나인",
                must_include=["브루클린 나인-나인", "media"],
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("미국 증시", search["draft_reply"])
        self.assertIn("모른다고", search["draft_reply"])
        self.assertIn("경제 뉴스", news["draft_reply"])
        self.assertNotIn("economy", news["draft_reply"])
        self.assertIn("브루클린 나인-나인", recommend["draft_reply"])
        self.assertNotIn("media", recommend["draft_reply"])

    def test_music_draft_does_not_emit_dangling_list_marker(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="비 오는 날 들을 곡 뭐가 좋을까?",
                normalized="비 오는 날 들을 곡 뭐가 좋을까?",
                intent=Intent.MUSIC,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.MUSIC_CHAT,
                stance="music_topic_reply",
                anchor="이무진 - 비와 당신|Epik High - 우산|AKMU - 어떻게 이별까지 사랑하겠어",
                must_include=[
                    "이무진 - 비와 당신|Epik High - 우산|AKMU",
                    "비 쪽으로 바로 던지면 이런 곡이 무난해. 1. 이무진 -",
                    "music",
                ],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("이무진", draft["draft_reply"])
        self.assertNotIn("1.", draft["draft_reply"])
        self.assertNotIn("music", draft["must_include"])

    def test_music_draft_preserves_comma_inside_title(self) -> None:
        title = "AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지"
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="잠들기 전 음악 고를 때 어떤 느낌을 보면 돼?",
                normalized="잠들기 전 음악 고를 때 어떤 느낌을 보면 돼?",
                intent=Intent.MUSIC,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.MUSIC_CHAT,
                stance="music_topic_reply",
                anchor="",
                must_include=[title, "music"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn(title, draft["draft_reply"])
        self.assertIn("잠들기 전이면", draft["draft_reply"])
        self.assertNotIn("세지 않은", draft["draft_reply"])
        self.assertNotIn("사랑하겠어처럼", draft["draft_reply"])

    def test_music_draft_filters_comparison_fragment_for_exercise_prompt(self) -> None:
        title = "AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지"
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="운동할 때 음악 하나 틀어야 한다면 어떤 쪽이 좋아?",
                normalized="운동할 때 음악 하나 틀어야 한다면 어떤 쪽이 좋아?",
                intent=Intent.MUSIC,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.MUSIC_CHAT,
                stance="music_topic_reply",
                anchor="",
                must_include=[title, "쪽이", "opinion_preference_like"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn(title, draft["draft_reply"])
        self.assertIn("박자가 또렷", draft["draft_reply"])
        self.assertNotIn("쪽이", draft["must_include"])
        self.assertNotIn("opinion_preference_like", draft["must_include"])

    def test_reflective_judgment_draft_does_not_prefix_long_user_fragment(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="잠 못 잔 날엔 작은 말도 크게 들린다는 말 어느 정도 맞는 것 같아?",
                normalized="잠 못 잔 날엔 작은 말도 크게 들린다는 말 어느 정도 맞는 것 같아?",
                intent=Intent.SMALLTALK_OPINION,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SHARE_OPINION,
                stance="direct_opinion",
                anchor="",
                must_include=["잠 못 잔 날엔 작은 말도 크게 들린다는 말 어느 정도 맞"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertIn("어느 정도 맞아", draft["draft_reply"])
        self.assertIn("작은 말도 크게 들릴 수", draft["draft_reply"])
        self.assertNotIn("잠 못 잔 날엔 작은 말도 크게 들린다는 말 어느 정도 맞.", draft["draft_reply"])
        self.assertEqual(draft["must_include"], [])

    def test_unseeded_context_prompts_recover_specific_drafts_before_rewrite(self) -> None:
        cases = [
            (
                "점심을 너무 대충 먹었더니 저녁은 좀 든든하게 먹고 싶어. 뭐가 좋을까?",
                ActionType.SHARE_OPINION,
                "direct_opinion",
                "국밥",
                "실제로 뭘 먹었다고",
            ),
            (
                "친구가 갑자기 말투가 차가워졌는데 내가 먼저 물어봐도 될까?",
                ActionType.SHARE_OPINION,
                "conditional_go_or_no_go",
                "먼저 물어봐도 돼",
                "자연스러운 반말",
            ),
            (
                "탕수육은 찍먹이 편하긴 한데 부먹도 맛있어서 늘 고민돼.",
                ActionType.SHARE_FEELING,
                "grounded_emotional_acknowledgement",
                "탕수육은 찍먹",
                "숨 돌릴",
            ),
            (
                "발표 전에 긴장될 때는 연습을 더 하는 게 나아, 잠깐 쉬는 게 나아?",
                ActionType.SHARE_OPINION,
                "conditional_go_or_no_go",
                "발표 직전",
                "무대 체질",
            ),
            (
                "비 오는 날에는 왜 해야 할 일을 미루고 싶어지는 걸까?",
                ActionType.WEATHER_UNAVAILABLE,
                "report_tool_failure",
                "제일 작은 일",
                "어느 지역",
            ),
            (
                "길게 조언하지 말고 지금 당장 할 수 있는 행동 하나만 말해줘.",
                ActionType.SHARE_OPINION,
                "conditional_go_or_no_go",
                "물 한 잔",
                "부담이 너무",
            ),
        ]

        for text, action, stance, expected, blocked in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question=True,
                    ),
                    response_plan=ResponsePlan(
                        action=action,
                        stance=stance,
                        anchor="",
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertIn(expected, draft["draft_reply"])
                self.assertNotIn(blocked, draft["draft_reply"])

    def test_high_context_daily_events_do_not_fall_back_to_generic_ack(self) -> None:
        cases = [
            (
                "야, 자냐? 나 갑자기 생각 많아져서 잠이 안 온다.",
                "korean_daily_expansion_late_night_overthinking_ping",
                ("새벽", "잠 안 오는"),
                "late_night_mood",
                "late_night_overthinking_ping",
            ),
            (
                "카톡 답장 머릿속으로 보냈는데 실제로 전송 안 해서 3일 동안 잠수 탄 사람 됨.",
                "korean_daily_expansion_kakao_reply_only_in_head",
                ("머릿속", "전송"),
                "digital_communication",
                "kakao_reply_only_in_head",
            ),
            (
                "프롬프트 조금만 비틀어도 답변 나락 가네.",
                "korean_daily_expansion_ai_prompt_fragility_reply_drop",
                ("프롬프트", "답변"),
                "ai_companion_building",
                "ai_prompt_fragility_reply_drop",
            ),
            (
                "로컬 LLM 답인데 프롬프트 살짝 비틀면 말투 깨져서 답변 나락감.",
                "korean_daily_expansion_ai_prompt_fragility_reply_drop",
                ("프롬프트", "말투"),
                "ai_companion_building",
                "ai_prompt_fragility_reply_drop",
            ),
            (
                "제품 설명에 네고 사절 대문짝만하게 써놨는데 물건 만나자마자 만 원만 깎아주심 안 돼요 시전 당함.",
                "korean_daily_more_shopping_secondhand_no_bargain_ignored",
                ("네고", "깎아"),
                "shopping_delivery",
                "shopping_secondhand_no_bargain_ignored",
            ),
            (
                "AI companion 하나 제대로 빌딩해 두면 인간관계에서 기 안 빨려도 대화 욕구 채워져서 편하긴 해.",
                "korean_daily_expansion_ai_companion_social_energy_buffer",
                ("대화 욕구", "거리감"),
                "ai_companion_building",
                "ai_companion_social_energy_buffer",
            ),
            (
                "성격 데이터셋 정제하는 데만 주말 다 날림. 노이즈 하나 섞이면 말투 바로 깨져서 장인정신으로 깎아야 됨.",
                "korean_daily_expansion_ai_persona_dataset_noise_cleanup",
                ("성격 데이터셋", "노이즈"),
                "ai_companion_building",
                "ai_persona_dataset_noise_cleanup",
            ),
            (
                "4B 모델로 데이터 플라이휠 돌려서 그 데이터로 8B 파인튜닝 하면 모델 능지 업그레이드 체감 될까?",
                "korean_daily_expansion_ai_model_flywheel_4b_to_8b",
                ("데이터 플라이휠", "파인튜닝"),
                "ai_companion_building",
                "ai_model_flywheel_4b_to_8b",
            ),
            (
                "GGUF 변환하니까 단일 모델만 지원하잖아. 나 캐릭터 어댑터 두 개 써야 되는데 어떡함?",
                "korean_daily_expansion_ai_gguf_single_model_two_adapters",
                ("GGUF", "어댑터"),
                "ai_companion_building",
                "ai_gguf_single_model_two_adapters",
            ),
            (
                "근데 내 그래픽카드 VRAM 16GB임. 터진다고.",
                "korean_daily_expansion_hardware_vram_16gb_oom_limit",
                ("VRAM 16GB", "Q4"),
                "hardware_setup",
                "hardware_vram_16gb_oom_limit",
            ),
            (
                "야, 오늘 저녁에 번개 고?",
                "korean_daily_expansion_group_chat_dinner_flash_meetup",
                ("번개", "집순이"),
                "digital_communication",
                "group_chat_dinner_flash_meetup",
            ),
            (
                "흑발/적안 vs 백발/벽안 캐릭터 일러스트 둘 다 미쳤는데 뭐가 더 취향임?",
                "korean_daily_expansion_character_black_red_vs_white_blue_choice",
                ("흑발적안", "백발벽안"),
                "character_design",
                "character_black_red_vs_white_blue_choice",
            ),
            (
                "폴더 명 날짜 포맷 YYYYMMDD 안 맞으면 새벽에 싹 다 수정함.",
                "korean_daily_expansion_perfectionist_folder_date_format_cleanup",
                ("폴더명", "YYYYMMDD"),
                "perfectionism_workflow",
                "perfectionist_folder_date_format_cleanup",
            ),
            (
                "당근마켓 매너온도 99도 유저님이 네고 없이 쿨거래를 제안했습니다.",
                "korean_daily_expansion_shopping_secondhand_manner_99_cool_deal",
                ("매너온도", "쿨거래"),
                "shopping_delivery",
                "shopping_secondhand_manner_99_cool_deal",
            ),
            (
                "조별 과제 빌런님이 나무위키 링크를 전송했습니다. PPT 기여도 0% 박고 싶다.",
                "korean_daily_expansion_work_group_project_namuwiki_zero_credit",
                ("나무위키", "기여도 0%"),
                "work_school",
                "work_group_project_namuwiki_zero_credit",
            ),
            (
                "친구가 단톡방에 재미없는 드립 쳤는데 ㅋㅋㅋㅋ 치고 오늘 저녁 뭐 먹냐고 주제 전환함.",
                "korean_daily_expansion_group_chat_bad_joke_topic_pivot",
                ("자본주의 리액션", "주제"),
                "digital_communication",
                "group_chat_bad_joke_topic_pivot",
            ),
            (
                "바탕화면 아이콘 3개 이상 늘어나는 꼴 못 봐서 새 폴더 만들어 숨김.",
                "korean_daily_expansion_perfectionist_desktop_icons_new_folder_hide",
                ("바탕화면", "새 폴더"),
                "perfectionism_workflow",
                "perfectionist_desktop_icons_new_folder_hide",
            ),
            (
                "상대 미드 스몰더 들고 누우면 진짜 숨이 턱 막힘. 타워 철거 반반 픽 상대로는 초반 스노우볼이 답인데.",
                "korean_daily_expansion_game_smolder_mid_scaling_counter",
                ("스몰더", "초반 스노우볼"),
                "game_meta",
                "game_smolder_mid_scaling_counter",
            ),
            (
                "이번 판 한타 때 우리 팀 딜러진 왜 이렇게 쉽게 녹아내리냐? 탱커 뒤에 숨어만 있다가 끝남.",
                "korean_daily_expansion_game_teamfight_dealer_melt_tank_hide",
                ("딜러진", "탱커"),
                "game_meta",
                "game_teamfight_dealer_melt_tank_hide",
            ),
            (
                "우리 바텀 미니맵 레이더 안 켜고 라인 밀다가 상대 미드 로밍에 계속 킬 상납 중.",
                "korean_daily_expansion_game_mid_roam_bot_minimap_kill_donation",
                ("미드 로밍", "킬 상납"),
                "game_meta",
                "game_mid_roam_bot_minimap_kill_donation",
            ),
            (
                "라인 다 망가졌는데 끝까지 로밍 가다가 포탑 골드 다 채굴당함. 게임 기본기가 안 돼 있어.",
                "korean_daily_expansion_game_roam_bad_wave_tower_gold_loss",
                ("로밍", "포탑 골드"),
                "game_meta",
                "game_roam_bad_wave_tower_gold_loss",
            ),
            (
                "오늘 큰맘 먹고 방 구조 바꿨는데 침대 위치 하나 바꿨다고 방 되게 넓어 보임. 대만족.",
                "korean_daily_expansion_room_layout_bed_position_satisfaction",
                ("침대 위치", "넓어"),
                "home_life",
                "room_layout_bed_position_satisfaction",
            ),
            (
                "방 청소의 끝은 미니멀리즘이라는데 나는 왜 자꾸 물건이 늘어날까. 맥시멀리스트의 삶은 고달프다.",
                "korean_daily_expansion_room_cleaning_maximalist_clutter",
                ("맥시멀리스트", "물건"),
                "home_life",
                "room_cleaning_maximalist_clutter",
            ),
            (
                "새벽 3시에 나무위키에서 아무 생각 없이 기술 문서 링크 타고 타다가 정신 차려 보니 해 뜨고 있음.",
                "korean_daily_expansion_namuwiki_tech_docs_rabbit_hole",
                ("나무위키", "기술 문서"),
                "digital_knowledge_rabbit_hole",
                "namuwiki_tech_docs_rabbit_hole",
            ),
            (
                "이어폰 한쪽 노이즈 미세하게 밸런스 안 맞는 느낌 들어서 좌우 밸런스 설정 창 켜고 1단위로 조정 중.",
                "korean_daily_expansion_audio_earphone_noise_balance_obsession",
                ("이어폰", "좌우 밸런스"),
                "audio_gear",
                "audio_earphone_noise_balance_obsession",
            ),
            (
                "불 다 끄고 침대에 누웠는데 에어컨 리모컨 저 멀리 책상 위에 있을 때... 갈등하다가 결국 추위에 떨며 잠.",
                "korean_daily_expansion_ac_remote_far_bed_conflict",
                ("에어컨 리모컨", "추위"),
                "homebody_rest",
                "ac_remote_far_bed_conflict",
            ),
            (
                "구독 서비스 안 쓰는 거 다 정리함. 넷플릭스, 유튜브 프리미엄 하니까 매달 고정 지출 장난 아님.",
                "korean_daily_expansion_subscription_cleanup_fixed_cost",
                ("구독 서비스", "고정 지출"),
                "digital_subscription",
                "subscription_cleanup_fixed_cost",
            ),
        ]

        for text, expected_reason, expected_fragments, expected_domain, expected_detail in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question="?" in text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assert_clean_black_draft(
                    draft,
                    direct_reason_prefixes=("korean_daily_",),
                )
                self.assertEqual(draft["direct_surface_reason"], expected_reason)
                self.assertEqual(draft["draft_domain"], expected_domain)
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assert_contains_any(str(draft["draft_reply"]), expected_fragments)

    def test_context_pressure_direct_reply_guides_uncaught_daily_cases(self) -> None:
        cases = [
            (
                "이거 나만 하루 종일 신경 쓰이는 건가?",
                "korean_daily_context_pressure_validate_first",
                ("너만 그런 거", "예민한 게 아니라"),
                "validate_first",
            ),
            (
                "말하기 애매해서 그냥 참고 있는데 괜히 예민한 사람 될까 봐.",
                "korean_daily_context_pressure_low_voice_boundary",
                ("말하기 애매", "낮게 꺼내"),
                "low_voice_boundary",
            ),
            (
                "옆방 사람이 밤마다 너무 시끄러운데 괜히 말 걸기 애매해.",
                "korean_daily_context_pressure_social_tact",
                ("눈치", "짧게"),
                "social_tact",
            ),
            (
                "괜히 찝찝해서 계속 신경 쓰여.",
                "korean_daily_context_pressure_name_unease",
                ("찝찝함", "이유가"),
                "name_unease",
            ),
            (
                "오늘 이걸 처리할까 말까 애매해서 계속 멈춰 있어.",
                "korean_daily_context_pressure_choose_practical",
                ("덜 후회", "손해가 작은"),
                "choose_practical",
            ),
            (
                "말하기 애매해서 참고 있는데 진짜 너무 답답해!",
                "korean_daily_context_pressure_low_voice_boundary",
                ("말하기 애매", "짧게 끊는"),
                "low_voice_boundary",
            ),
            (
                "괜히 찝찝해서 계속 신경 쓰임 ㅋㅋ",
                "korean_daily_context_pressure_name_unease",
                ("ㅋㅋ", "걸리는 포인트"),
                "name_unease",
            ),
            (
                "이거 나만 신경 쓰이나 싶어서 좀 속상해 ㅠㅠ",
                "korean_daily_context_pressure_validate_first",
                ("너만 그런 거", "덜 다치게"),
                "validate_first",
            ),
        ]

        for text, expected_reason, expected_fragments, expected_detail in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_OPINION,
                        sentiment="neutral",
                        is_question="?" in text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SHARE_OPINION,
                        stance="direct_opinion",
                        anchor="",
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assert_clean_black_draft(
                    draft,
                    direct_reason_prefixes=("korean_daily_",),
                )
                self.assertEqual(draft["direct_surface_reason"], expected_reason)
                self.assertEqual(draft["draft_domain"], "context_pressure")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assert_contains_any(str(draft["draft_reply"]), expected_fragments)

    def test_high_context_trace_records_raw_tokens_senses_and_draft_frame(self) -> None:
        def build(text: str) -> dict[str, object]:
            return build_black_draft_utterance(
                features=MessageFeatures(
                    content=text,
                    normalized=text,
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question="?" in text,
                ),
                response_plan=ResponsePlan(
                    action=ActionType.SHARE_OPINION,
                    stance="direct_opinion",
                    anchor="",
                    followup_policy="no_followup",
                ),
                phrasing_plan=PhrasingPlan(),
            )

        late_ping = build("야, 자냐? 나 갑자기 생각 많아져서 잠이 안 온다.")
        late_trace = late_ping["high_context_trace"]
        self.assertIn("야, 자냐?", late_trace["raw_text"])
        self.assertIn("야자냐", late_trace["compact_text"])
        self.assertIn("자냐", late_trace["token_candidates"])
        self.assertIn("token_ngrams", late_trace["layers"])
        self.assertIn("야 자냐", late_trace["token_ngrams"])
        self.assertIn("나 갑자기 생각", late_trace["token_ngrams"])
        self.assertIn("comma_pause", late_trace["surface_markers"]["markers"])
        self.assertIn("question_mark", late_trace["surface_markers"]["markers"])
        self.assertIn("direct_call", late_trace["surface_markers"]["markers"])
        self.assertIn("late_night_signal", late_trace["surface_markers"]["markers"])
        self.assertIn("semantic_tags", late_trace["layers"])
        self.assertIn("semantic_word_candidates", late_trace["layers"])
        self.assertIn("salience_profile", late_trace["layers"])
        self.assertIn("fatigue", late_trace["salience_profile"]["emotion_tags"])
        self.assertEqual(late_trace["salience_profile"]["reaction_mode"], "comfort_first")
        self.assertEqual(
            late_trace["draft_event_frame"]["direct_surface_reason"],
            "korean_daily_expansion_late_night_overthinking_ping",
        )
        self.assertEqual(late_trace["draft_event_frame"]["draft_domain"], "late_night_mood")
        self.assertEqual(late_trace["draft_event_frame"]["draft_frame_detail"], "late_night_overthinking_ping")

        palm_tree_trace = build(
            "새벽에 알고리즘이 인도 길거리 거대 야자수 자르는 영상 추천해 줘서 넋 놓고 끝까지 다 봄."
        )["high_context_trace"]
        self.assertIn("거대 야자수", palm_tree_trace["token_ngrams"])
        self.assertIn("야자수 자르는", palm_tree_trace["token_ngrams"])
        self.assertNotIn("야 자냐", palm_tree_trace["token_ngrams"])

        sense_cases = (
            ("머리가 아프다", "머리", "body_head"),
            ("배를 탔다", "배", "ship"),
            ("배가 아프다", "배", "body_stomach"),
        )
        for text, word, expected_sense in sense_cases:
            with self.subTest(text=text):
                trace = build(text)["high_context_trace"]
                senses = {
                    item["word"]: item["sense"]
                    for item in trace["resolved_word_senses"]
                }
                self.assertEqual(senses.get(word), expected_sense)
                self.assertTrue(trace["sense_tags"])

        high_context_cases = (
            (
                "이번 패치 보니까 후반 벨류가 선 넘었더라. 스몰더 스택 쌓이면 답이 없음.",
                "후반 벨류 경보",
                "game_meta",
                "react_first",
                "후반 벨류 경보",
            ),
            (
                "이어폰 한쪽 노이즈 미세하게 밸런스 안 맞는 느낌 들어서 좌우 밸런스 설정 창 켜고 1단위로 조정 중.",
                "좌우 밸런스 디버깅",
                "audio_detail",
                "mirror_detail",
                "이어폰",
            ),
            (
                "학습 데이터에 노이즈 조금만 껴도 답변 나락 가서 데이터 정제가 핵심이더라.",
                "데이터 정제 장인전",
                "ai_build_pipeline",
                "react_first",
                "데이터 정제",
            ),
        )
        for text, expected_label, expected_cluster, expected_mode, expected_term in high_context_cases:
            with self.subTest(text=text):
                trace = build(text)["high_context_trace"]
                semantic_labels = [
                    item["value"]
                    for item in trace["semantic_word_candidates"]
                ]

                self.assertIn(expected_label, semantic_labels)
                self.assertIn(expected_cluster, trace["salience_profile"]["topic_clusters"])
                self.assertEqual(trace["salience_profile"]["reaction_mode"], expected_mode)
                self.assertIn(expected_term, trace["salience_profile"]["must_keep_surface_terms"])

        game_boundary_trace = build(
            "이번 패치 보니까 후반 벨류가 선 넘었더라. 스몰더 스택 쌓이면 답이 없음."
        )["high_context_trace"]
        self.assertNotIn("relationship_social", game_boundary_trace["salience_profile"]["topic_clusters"])
        self.assertEqual(
            game_boundary_trace["salience_profile"]["dominant_context"]["cluster"],
            "game_meta",
        )
        self.assertEqual(
            game_boundary_trace["salience_profile"]["dominant_context"]["label"],
            "후반 벨류 경보",
        )

        pegboard_trace = build(
            "벽에 타공판 달아서 헤드셋이랑 자주 쓰는 가젯들 걸어두니까 묘하게 작업실 분위기 나고 좋더라."
        )["high_context_trace"]
        self.assertIn("room_interior_scene", pegboard_trace["salience_profile"]["topic_clusters"])
        self.assertNotIn("food_daily", pegboard_trace["salience_profile"]["topic_clusters"])
        self.assertEqual(
            pegboard_trace["salience_profile"]["dominant_context"]["cluster"],
            "room_interior_scene",
        )
        self.assertEqual(
            pegboard_trace["salience_profile"]["dominant_context"]["label"],
            "타공판 작업실 버프",
        )
        self.assertIn(
            "audio_detail",
            pegboard_trace["salience_profile"]["dominant_context"]["supporting_clusters"],
        )
        self.assertIn(
            "타공판 작업실 버프",
            [item["value"] for item in pegboard_trace["semantic_word_candidates"][:3]],
        )

        audio_noise_trace = build(
            "이어폰 한쪽 노이즈 미세하게 밸런스 안 맞는 느낌 들어서 좌우 밸런스 설정 창 켜고 1단위로 조정 중."
        )["high_context_trace"]
        self.assertNotIn("relationship_social", audio_noise_trace["salience_profile"]["topic_clusters"])
        self.assertNotIn("comfort", audio_noise_trace["semantic_tags"])
        self.assertNotIn("support", audio_noise_trace["semantic_tags"])
        self.assertNotIn(
            "위로",
            [item["value"] for item in audio_noise_trace["semantic_word_candidates"][:3]],
        )

        data_noise_trace = build(
            "학습 데이터에 노이즈 조금만 껴도 답변 나락 가서 데이터 정제가 핵심이더라."
        )["high_context_trace"]
        self.assertNotIn("audio_detail", data_noise_trace["salience_profile"]["topic_clusters"])
        self.assertNotIn(
            "좌우 밸런스 디버깅",
            [item["value"] for item in data_noise_trace["semantic_word_candidates"][:3]],
        )
        self.assertTrue(
            all(
                "context_gate" in item
                for item in data_noise_trace["semantic_word_candidates"]
            )
        )

    def test_high_context_trace_records_context_pressure_profile(self) -> None:
        def build(text: str) -> dict[str, object]:
            return build_black_draft_utterance(
                features=MessageFeatures(
                    content=text,
                    normalized=text,
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question="?" in text,
                ),
                response_plan=ResponsePlan(
                    action=ActionType.SHARE_OPINION,
                    stance="direct_opinion",
                    anchor="",
                    followup_policy="no_followup",
                ),
                phrasing_plan=PhrasingPlan(),
            )

        cases = (
            (
                "이거 나만 하루 종일 신경 쓰이는 건가?",
                ("self_doubt_validation", "lingering_unease"),
                "validate_first",
                "나만 그런 게 아니라는 확인",
            ),
            (
                "말하기 애매해서 그냥 참고 있는데 괜히 예민한 사람 될까 봐.",
                ("low_voice_boundary", "passive_containment", "decision_ambivalence"),
                "low_voice_boundary",
                "말하기 애매한 부담",
            ),
            (
                "식당에서 물티슈가 없는데 직원분들이 너무 바빠 보여서 말 걸기 애매해.",
                ("social_face_pressure", "decision_ambivalence"),
                "social_tact",
                "상대 눈치와 분위기 부담",
            ),
            (
                "오늘 날씨가 애매해서 우산 챙길지 모르겠어.",
                ("decision_ambivalence",),
                "choose_practical",
                "선택을 미루게 하는 애매함",
            ),
        )
        for text, expected_markers, expected_mode, expected_ack in cases:
            with self.subTest(text=text):
                trace = build(text)["high_context_trace"]
                self.assertIn("context_pressure_profile", trace["layers"])
                profile = trace["context_pressure_profile"]
                self.assertEqual(profile["response_mode"], expected_mode)
                for marker in expected_markers:
                    self.assertIn(marker, profile["markers"])
                self.assertIn(expected_ack, profile["must_acknowledge"])

    def test_context_pressure_profile_records_tone_level(self) -> None:
        def tone(text: str) -> str:
            draft = build_black_draft_utterance(
                features=MessageFeatures(
                    content=text,
                    normalized=text,
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question="?" in text,
                ),
                response_plan=ResponsePlan(
                    action=ActionType.SHARE_OPINION,
                    stance="direct_opinion",
                    anchor="",
                    followup_policy="no_followup",
                ),
                phrasing_plan=PhrasingPlan(),
            )
            profile = draft["high_context_trace"]["context_pressure_profile"]
            return str(profile["tone_level"])

        self.assertEqual(tone("괜히 찝찝해서 계속 신경 쓰임 ㅋㅋ"), "playful")
        self.assertEqual(tone("말하기 애매해서 참고 있는데 진짜 너무 답답해!"), "hot")
        self.assertEqual(tone("이거 나만 신경 쓰이나 싶어서 좀 속상해 ㅠㅠ"), "soft")

    def test_salience_dominant_context_drives_semantic_fallback_reply(self) -> None:
        def build(text: str) -> dict[str, object]:
            return build_black_draft_utterance(
                features=MessageFeatures(
                    content=text,
                    normalized=text,
                    intent=Intent.SMALLTALK_OPINION,
                    sentiment="neutral",
                    is_question="?" in text,
                ),
                response_plan=ResponsePlan(
                    action=ActionType.SHARE_OPINION,
                    stance="direct_opinion",
                    anchor="",
                    followup_policy="no_followup",
                ),
                phrasing_plan=PhrasingPlan(),
            )

        cases = (
            (
                "이어폰 한쪽 밸런스가 묘하게 틀어져서 소리가 계속 한쪽으로 쏠려.",
                "audio_detail",
                ("이어폰 밸런스", "디버깅"),
            ),
            (
                "캐릭터 눈색이랑 머리색 대비가 좋아서 서사까지 있어 보이더라.",
                "character_visual_lore",
                ("머리색이랑 눈색 대비", "서사"),
            ),
            (
                "큰 모델 올리려니까 브이램이 먼저 벽처럼 막아서 세팅이 계속 꼬여.",
                "ai_build_pipeline",
                ("브이램", "자원 배분"),
            ),
            (
                "양자화로 가볍게 만들었더니 답변 품질이 확 내려가는 느낌이야.",
                "ai_build_pipeline",
                ("양자화", "능지세"),
            ),
            (
                "프롬프트 규칙을 많이 넣으니까 서로 충돌해서 답이 흔들려.",
                "ai_build_pipeline",
                ("프롬프트 규칙", "충돌"),
            ),
            (
                "디스코드 봇에 로컬 서버를 붙였더니 폰에서도 바로 테스트돼서 편하더라.",
                "ai_build_pipeline",
                ("로컬 서버 연동", "테스트"),
            ),
            (
                "방 청소하려고 서랍 하나 열었다가 옛날 일기장 발견해서 청소가 멈췄어.",
                "home_daily_detail",
                ("방 정리", "시간여행"),
            ),
            (
                "싱크대에서 물 한 방울씩 똑똑 떨어지는 소리가 새벽에 계속 거슬려.",
                "home_daily_detail",
                ("방구석 디테일", "존재감"),
            ),
            (
                "내 방 물건 누가 건드리는 것 같아서 문고리에 머리카락 트랩을 붙여보고 싶어.",
                "home_daily_detail",
                ("탐정 모드", "내 공간"),
            ),
            (
                "내가 힘들 때 곁에 있어주는 사람이 진짜 친구 같아.",
                "relationship_social",
                ("진짜 관계의 체력", "곁에 남아주는 사람"),
            ),
            (
                "요즘 남들이 다 앞으로 가는 것 같은데 나만 제자리인 것 같아서 현타 와.",
                "life_reflection",
                ("나만 제자리", "가혹"),
            ),
            (
                "나이 먹을수록 예전이랑 성격이 달라지는 게 느껴져.",
                "life_reflection",
                ("나이 들며 바뀜", "업데이트"),
            ),
        )

        for text, expected_cluster, expected_snippets in cases:
            with self.subTest(text=text):
                draft = build(text)

                self.assertEqual(draft["direct_surface_reason"], "salience_dominant_context_direct_reply")
                self.assertEqual(
                    draft["high_context_trace"]["salience_profile"]["dominant_context"]["cluster"],
                    expected_cluster,
                )
                for snippet in expected_snippets:
                    self.assertIn(snippet, draft["draft_reply"])
                if "학습 데이터" not in text:
                    self.assertNotIn("정제랑 검증", draft["draft_reply"])
                self.assertNotIn("무리하게 밀 필요", draft["draft_reply"])

    def test_identity_and_capability_do_not_leak_internal_anchor(self) -> None:
        identity = build_black_draft_utterance(
            features=MessageFeatures(
                content="두 문장 이내로 자기소개해줘.",
                normalized="두 문장 이내로 자기소개해줘.",
                intent=Intent.WHO_ARE_YOU,
                sentiment="neutral",
                is_question=False,
            ),
            response_plan=ResponsePlan(
                action=ActionType.ANSWER_IDENTITY,
                stance="identity_answer",
                anchor="identity",
                must_include=["identity", "예측 기반", "디스코드 봇"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )
        capability = build_black_draft_utterance(
            features=MessageFeatures(
                content="뭐 할 수 있어?",
                normalized="뭐 할 수 있어?",
                intent=Intent.HELP,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.EXPLAIN_CAPABILITIES,
                stance="capability_summary",
                anchor="capability",
                must_include=["capability", "잡담", "날씨", "시간", "뉴스"],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertNotIn("identity.", identity["draft_reply"])
        self.assertNotIn("capability.", capability["draft_reply"])
        self.assertIn("예측 기반", identity["draft_reply"])
        self.assertIn("뉴스", capability["draft_reply"])


if __name__ == "__main__":
    unittest.main()
