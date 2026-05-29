from __future__ import annotations

import unittest
from types import SimpleNamespace

from predictive_bot.core.draft_nlg import (
    build_black_draft_utterance,
    _render_salience_dominant_direct_reply,
    _salience_profile,
    _slot_bank_selection,
    _text_signals,
)
from predictive_bot.core.draft_semantic_relations import infer_semantic_relations, rank_semantic_relations
from predictive_bot.core.draft_semantic_words import _input_tags, rank_semantic_candidates, rank_semantic_word
from predictive_bot.core.draft_slot_bank import DRAFT_SLOT_BANKS
from predictive_bot.core.draft_slot_selector import BertDraftSlotSelector, RuleDraftSlotSelector
from predictive_bot.core.draft_word_senses import build_word_sense_context_from_texts, resolve_word_senses
from predictive_bot.core.models import (
    ActionType,
    ConversationState,
    Intent,
    MessageFeatures,
    PhrasingPlan,
    ResponsePlan,
    TurnRecord,
)


class DraftSlotSelectorTests(unittest.TestCase):
    def test_rule_selector_picks_keyword_slots_with_trace(self) -> None:
        signals = _text_signals("오늘 점심 뭐 먹지?")
        selection = RuleDraftSlotSelector().select(
            signals,
            detail="daily_food_menu_pick",
            slot_bank=DRAFT_SLOT_BANKS["daily_food_menu_pick"],
        )

        self.assertEqual(selection.slots["dish"], "제육덮밥")
        self.assertEqual(selection.slots["setup"], "점심이면")
        self.assertTrue(selection.render())
        dish_trace = next(trace for trace in selection.trace if trace.slot == "dish")
        self.assertEqual(dish_trace.source, "keyword")
        self.assertIn("점심", dish_trace.matched_keywords)

    def test_rule_selector_is_stable_for_template_and_variants(self) -> None:
        signals = _text_signals("무임승차 빌런 이름 발표 피피티 기여도에 0%로 박아버림.")
        selector = RuleDraftSlotSelector()

        first = selector.select(
            signals,
            detail="workrevenge_group_project_zero_credit",
            slot_bank=DRAFT_SLOT_BANKS["workrevenge_group_project_zero_credit"],
        )
        second = selector.select(
            signals,
            detail="workrevenge_group_project_zero_credit",
            slot_bank=DRAFT_SLOT_BANKS["workrevenge_group_project_zero_credit"],
        )

        self.assertEqual(first.template, second.template)
        self.assertEqual(first.slots, second.slots)
        self.assertIn("무임승차 빌런", first.render())
        self.assertIn("기여도 0%", first.render())

    def test_draft_nlg_slot_bank_uses_selector_layer(self) -> None:
        selection = _slot_bank_selection(_text_signals("밤에 뭐 먹지? 배고픈데"), "daily_food_menu_pick")

        self.assertIn(selection.slots["dish"], selection.render())
        self.assertTrue(any(trace.slot == "$template" for trace in selection.trace))

    def test_context_pressure_slot_banks_choose_topic_and_pressure_words(self) -> None:
        validate = _slot_bank_selection(
            _text_signals("카톡 답장 때문에 이거 나만 신경 쓰이나?"),
            "context_pressure_validate_first",
        )
        self.assertEqual(validate.slots["topic_phrase"], "연락 쪽이")
        self.assertEqual(validate.slots["pressure"], "나만 그런가 싶은 감각")
        self.assertIn("연락", validate.render())

        boundary = _slot_bank_selection(
            _text_signals("말하기 애매해서 그냥 참고 있는데 괜히 예민한 사람 될까 봐."),
            "context_pressure_low_voice_boundary",
        )
        self.assertEqual(boundary.slots["burden"], "말하기 애매한 부담")
        self.assertIn("낮게 꺼내", boundary.render())

        practical = _slot_bank_selection(
            _text_signals("오늘 비 올지 말지 애매해서 우산 챙길지 계속 고민중."),
            "context_pressure_choose_practical",
        )
        self.assertEqual(practical.slots["choice"], "챙길지 말지")
        self.assertEqual(practical.slots["criterion"], "나중에 덜 젖고 덜 후회할 쪽")
        self.assertIn("덜 후회", practical.render())

    def test_bert_selector_adapter_preserves_current_fallback_shape(self) -> None:
        signals = _text_signals("비가 올지 말지 애매한데 우산 챙길까?")
        selection = BertDraftSlotSelector().select(
            signals,
            detail="daily_weather_umbrella_uncertainty",
            slot_bank=DRAFT_SLOT_BANKS["daily_weather_umbrella_uncertainty"],
        )

        self.assertEqual(selection.slots["weather"], "비 예보")
        self.assertIn("우산", selection.render())
        self.assertTrue(selection.trace)

    def test_semantic_word_slot_picks_nearest_metaphor(self) -> None:
        signals = _text_signals("족보가 팔만대장경인가? 모두의 할아버지라서 너무 길어.")
        selection = RuleDraftSlotSelector().select(
            signals,
            detail="semantic_comic_exaggeration_record",
            slot_bank=DRAFT_SLOT_BANKS["semantic_comic_exaggeration_record"],
        )

        self.assertEqual(selection.slots["metaphor"], "팔만대장경급")
        metaphor_trace = next(trace for trace in selection.trace if trace.slot == "metaphor")
        self.assertEqual(metaphor_trace.source, "semantic_word_bank")
        self.assertIn("huge_record", metaphor_trace.matched_keywords)
        self.assertIn("팔만", metaphor_trace.matched_keywords)

    def test_semantic_word_bank_reply_routes_through_draft_nlg(self) -> None:
        draft = build_black_draft_utterance(
            features=MessageFeatures(
                content="족보가 팔만대장경인가? 모두의 할아버지라서 그런가 너무 길어.",
                normalized="족보가 팔만대장경인가? 모두의 할아버지라서 그런가 너무 길어.",
                intent=Intent.SMALLTALK_GENERIC,
                sentiment="neutral",
                is_question=True,
            ),
            response_plan=ResponsePlan(
                action=ActionType.SMALL_TALK,
                stance="continue_light",
                anchor="",
                must_include=[],
                followup_policy="no_followup",
            ),
            phrasing_plan=PhrasingPlan(),
        )

        self.assertEqual(draft["rewrite_mode"], "draft_direct")
        self.assertEqual(draft["draft_frame_detail"], "semantic_comic_exaggeration_record")
        self.assertIn("족보", draft["draft_reply"])
        self.assertIn("팔만대장경급", draft["draft_reply"])

    def test_semantic_word_bank_covers_daily_money_weather_and_fandom(self) -> None:
        cases = (
            (
                "월급 들어왔는데 카드값이 퍼가요 당해서 통장을 스쳐 지나감.",
                ("salary", "vanish", "joke"),
                "월급 증발쇼",
            ),
            (
                "오늘 날씨 아침엔 춥고 낮엔 더워서 옷 선택 완전 망함.",
                ("weather", "unpredictable", "outfit", "frustration"),
                "날씨 룰렛",
            ),
            (
                "랜덤 굿즈 5개 샀는데 중복만 3개 나옴. 내 손은 왜 최애를 피해 가냐.",
                ("fandom", "random_goods", "duplicate", "frustration"),
                "뽑기 억까",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_covers_tech_pet_and_homebody(self) -> None:
        cases = (
            (
                "로컬 LLM 올렸더니 VRAM 16GB 꽉 차서 OOM 뜸. 4090 마렵다.",
                ("tech", "ai", "hardware", "vram", "oom"),
                "브이램 절벽",
            ),
            (
                "강아지 수제 간식 만들었는데 한 입 먹고 뱉음. 진짜 상전이다.",
                ("pet", "food_rejection", "cute_frustration"),
                "상전 리뷰 1점",
            ),
            (
                "이번 주말엔 침대 밖으로 한 발짝도 안 나가고 유튜브만 볼 예정.",
                ("homebody", "bed", "weekend", "joke"),
                "침대 지박령",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_candidate_ranking_keeps_high_context_domains_visible(self) -> None:
        cases = (
            (
                "이번 패치 보니까 후반 벨류가 선 넘었더라. 스몰더 스택 쌓이면 답이 없음.",
                "후반 벨류 경보",
                {"game", "meta", "scaling"},
            ),
            (
                "라인전 압박으로 초반 스노우볼 못 굴리면 타워 철거 반반 픽 상대로 답답해.",
                "스노우볼 계산서",
                {"game", "snowball", "lane_phase"},
            ),
            (
                "학습 데이터에 노이즈 조금만 껴도 답변 나락 가서 데이터 정제가 핵심이더라.",
                "데이터 정제 장인전",
                {"tech", "ai", "data_cleaning"},
            ),
        )

        for text, expected_value, expected_tags in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_candidates(_text_signals(text), limit=3)

                self.assertTrue(ranked)
                self.assertEqual(ranked[0].value, expected_value)
                self.assertTrue(expected_tags <= set(ranked[0].candidate_tags))

    def test_semantic_candidate_ranking_covers_scene_collocations_for_high_context(self) -> None:
        cases = (
            (
                "그래픽카드 서멀구리스 재도포 갈았더니 풀로드 온도가 5도 떨어짐. 이거 은근히 쾌감 있네.",
                "서멀 5도 보상",
                {"tech", "hardware", "thermal", "temperature_drop"},
            ),
            (
                "컴퓨터 본체 LED 불빛 유난히 거슬려서 전용 프로그램으로 아예 다 끄고 누드 감성으로 쓰는 중.",
                "LED 소등 평화",
                {"hardware", "pc_case", "led", "lighting"},
            ),
            (
                "외장 하드 정리하다가 10년 전 파일들 발견함. 백업의 중요성을 다시 한번 깨닫는 순간.",
                "백업 타임캡슐",
                {"digital", "storage", "backup", "old_files"},
            ),
            (
                "로컬 vLLM 서버 구동해 놓고 디스코드 봇으로 연동했더니 스마트폰으로도 테스트 가능해서 신세계임.",
                "로컬봇 손안의 실험실",
                {"tech", "ai", "local_server", "discord_bot"},
            ),
            (
                "VRAM 쪼개 쓰려고 양자화 포맷별로 속도 비교 중인데 비트 수 너무 낮추니까 확실히 능지 박살 남.",
                "양자화 능지세",
                {"tech", "ai", "quantization", "quality_loss"},
            ),
            (
                "인스타에서 미드센추리 모던 감성 조명 보고 홀린 듯 결제했는데 우리 집 인테리어랑 따로 놀아서 처치 곤란.",
                "미드센추리 조명불협",
                {"home", "interior", "lighting", "midcentury"},
            ),
            (
                "벽에 타공판 달아서 헤드셋이랑 자주 쓰는 가젯들 걸어두니까 묘하게 작업실 분위기 나고 좋더라.",
                "타공판 작업실 버프",
                {"home", "interior", "desk_setup", "pegboard"},
            ),
            (
                "아니 이 타이밍에 바론 치는 게 맞냐? 굳이 안 줘도 될 턴을 줘서 역전 빌미를 만드네.",
                "바론 역전빌미",
                {"game", "baron", "turn", "throw"},
            ),
            (
                "우리 팀 정글 동선 다 읽혔는데 끝까지 카정 가다가 퍼블 주네. 조용히 차단 박고 내 라인 집중한다.",
                "카정 퍼블 헌납",
                {"game", "jungle", "pathing", "counter_jungle"},
            ),
            (
                "상대 미드 로밍력 미쳤던데 우리 바텀 레이더 안 켜고 라인 밀다가 계속 더블킬 상납 중.",
                "로밍 레이더 꺼짐",
                {"game", "mid_roam", "bot_lane", "map_awareness"},
            ),
        )

        for text, expected_value, expected_tags in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_candidates(_text_signals(text), limit=5)

                self.assertTrue(ranked)
                self.assertEqual(ranked[0].value, expected_value)
                self.assertTrue(expected_tags <= set(ranked[0].candidate_tags), ranked[0])

    def test_semantic_candidate_ranking_keeps_reconnect_contact_as_mixed_emotion(self) -> None:
        text = "오랜만에 고등학교 동창한테 연락 왔는데 반가우면서도 묘하게 어색하더라."
        ranked = rank_semantic_candidates(_text_signals(text), limit=5)

        self.assertTrue(ranked)
        self.assertEqual(ranked[0].value, "오랜만 동창 어색함")
        self.assertTrue(
            {"relationship", "reconnect", "old_friend", "classmate", "awkward", "positive_mixed"}
            <= set(ranked[0].candidate_tags),
            ranked[0],
        )
        self.assertNotEqual(ranked[0].value, "연락")

    def test_salience_profile_uses_high_confidence_semantic_candidate_tags(self) -> None:
        profile = _salience_profile(
            signals=_text_signals("타공판에 헤드셋 걸어두니까 작업실 분위기 나고 좋더라."),
            semantic_tags=(),
            sense_tags=(),
            semantic_candidates=(
                SimpleNamespace(
                    value="타공판 작업실 버프",
                    score=6.2,
                    context_gate="accepted",
                    candidate_tags=("home", "interior", "desk_setup", "pegboard"),
                    matched_aliases=("타공판", "작업실"),
                ),
            ),
            must_preserve=[],
        )

        self.assertIn("room_interior_scene", profile["topic_clusters"])
        self.assertEqual(profile["dominant_context"]["cluster"], "room_interior_scene")
        self.assertEqual(profile["dominant_context"]["label"], "타공판 작업실 버프")
        self.assertIn("타공판 작업실 버프", profile["semantic_labels"])
        self.assertIn("타공판", profile["must_keep_surface_terms"])

    def test_salience_profile_ignores_low_confidence_candidate_tags(self) -> None:
        profile = _salience_profile(
            signals=_text_signals("오늘 그냥 별일 없었어."),
            semantic_tags=(),
            sense_tags=(),
            semantic_candidates=(
                SimpleNamespace(
                    value="바론 역전빌미",
                    score=1.8,
                    context_gate="accepted",
                    candidate_tags=("game", "baron", "turn"),
                    matched_aliases=("바론",),
                ),
            ),
            must_preserve=[],
        )

        self.assertNotIn("game_micro_decision", profile["topic_clusters"])
        self.assertEqual(profile["dominant_context"]["cluster"], "general")
        self.assertEqual(profile["dominant_context"]["label"], "")

    def test_salience_profile_ignores_single_alias_side_candidate_tags(self) -> None:
        profile = _salience_profile(
            signals=_text_signals("이번 패치 후반 벨류가 선 넘었더라."),
            semantic_tags=("game", "meta"),
            sense_tags=(),
            semantic_candidates=(
                SimpleNamespace(
                    value="손절 경계선",
                    score=3.9,
                    context_gate="accepted",
                    candidate_tags=("relationship", "friendship", "boundary", "hurt", "decision"),
                    matched_aliases=("선넘",),
                ),
            ),
            must_preserve=[],
        )

        self.assertIn("game_meta", profile["topic_clusters"])
        self.assertEqual(profile["dominant_context"]["cluster"], "game_meta")
        self.assertEqual(profile["dominant_context"]["label"], "")
        self.assertNotIn("relationship_social", profile["topic_clusters"])
        self.assertNotIn("decision_pressure", profile["emotion_tags"])

    def test_salience_profile_prefers_best_overlap_for_dominant_context(self) -> None:
        profile = _salience_profile(
            signals=_text_signals("이어폰 한쪽 밸런스가 묘하게 틀어져서 소리가 계속 한쪽으로 쏠려."),
            semantic_tags=("game", "meta", "balance", "audio", "earphones", "noise"),
            sense_tags=(),
            semantic_candidates=(
                SimpleNamespace(
                    value="좌우 밸런스 디버깅",
                    score=9.45,
                    context_gate="accepted",
                    candidate_tags=("audio", "earphones", "noise", "balance", "detail", "obsession"),
                    matched_aliases=("밸런스", "이어폰"),
                ),
            ),
            must_preserve=[],
        )

        self.assertIn("game_meta", profile["topic_clusters"])
        self.assertIn("audio_detail", profile["topic_clusters"])
        self.assertEqual(profile["dominant_context"]["cluster"], "audio_detail")
        self.assertEqual(profile["dominant_context"]["label"], "좌우 밸런스 디버깅")

    def test_salience_profile_preserves_reconnect_contact_context(self) -> None:
        text = "오랜만에 고등학교 동창한테 연락 왔는데 반가우면서도 묘하게 어색하더라."
        signals = _text_signals(text)
        ranked = rank_semantic_candidates(signals, limit=5)
        profile = _salience_profile(
            signals=signals,
            semantic_tags=_input_tags(signals),
            sense_tags=(),
            semantic_candidates=ranked,
            must_preserve=[],
        )
        direct = _render_salience_dominant_direct_reply(text)

        self.assertEqual(profile["dominant_context"]["cluster"], "relationship_social")
        self.assertEqual(profile["dominant_context"]["label"], "오랜만 동창 어색함")
        self.assertIn("오랜만", profile["must_keep_surface_terms"])
        self.assertIn("동창", profile["must_keep_surface_terms"])
        self.assertIn("반가움", direct)
        self.assertIn("어색함", direct)
        self.assertNotIn("연락는", direct)

    def test_semantic_candidate_ranking_keeps_generic_awkward_out_of_reconnect(self) -> None:
        text = "미용실 디자이너가 머리 마음에 드냐고 물어보면 어색하게 웃으며 네 하고 집 와서 후회함."
        ranked = rank_semantic_candidates(_text_signals(text), limit=5)
        direct = _render_salience_dominant_direct_reply(text)

        self.assertTrue(ranked)
        self.assertEqual(ranked[0].value, "미용실 자본주의 칭찬")
        self.assertNotEqual(ranked[0].value, "오랜만 동창 어색함")
        self.assertIn("미용실", direct)
        self.assertIn("분위기", direct)

    def test_salience_direct_replies_cover_high_context_daily_cues(self) -> None:
        cases = (
            (
                "퇴근길에 해 지는 노을 예쁘길래 사진 찍었는데, 눈으로 보는 그 감성이 안 담기네.",
                "scenery_mood",
                "노을 저장 실패",
                ("노을 사진", "카메라"),
            ),
            (
                "배달 앱 켜고 뭐 먹을지 고르다가 한 시간 지남. 결국 맨날 시키던 거 시켰다.",
                "food_daily",
                "아무거나 지뢰",
                ("메뉴 고르기", "익숙한 메뉴"),
            ),
            (
                "나 주말에 약속 취소되면 겉으로는 아쉽다 하는데 속으로는 은근히 신남. 집이 최고야.",
                "home_daily_detail",
                "약속취소 내적축제",
                ("약속 취소", "축제"),
            ),
            (
                "할 일 목록 짜는 데만 완벽하게 한 시간 쓰고, 정작 시작도 전에 지쳐서 눕는 엔딩.",
                "planning_burnout",
                "계획 장인 실천 파업",
                ("할 일 목록", "쉬운 한 칸"),
            ),
            (
                "오늘 아침에 급하게 나오느라 셔츠 단추 하나 밀려 끼운 거 지하철 거울 보고 알았음.",
                "commute_work",
                "셔츠 단추 대수치",
                ("셔츠 단추", "피해"),
            ),
            (
                "퇴근 10분 전에 메일 온 거 보고 모니터 조용히 끔. 내일의 내가 알아서 하겠지 뭐.",
                "commute_work",
                "퇴근길 폭탄",
                ("퇴근 직전 메일", "생존술"),
            ),
            (
                "마트 가서 과자 몇 개 집었는데 2만 원 나오더라. 요즘 물가 진짜 장난 아닌 듯.",
                "price_shock",
                "마트 물가 펀치",
                ("마트 물가", "영수증"),
            ),
            (
                "오늘 큰맘 먹고 방 대청소 했다. 이불 빨래까지 싹 돌렸더니 속이 다 시원하네.",
                "home_daily_detail",
                "대청소 이불빨래 보상",
                ("대청소랑 이불 빨래", "뽀송한 이불"),
            ),
            (
                "오늘 날씨 보니까 주말에 무조건 밖으로 놀러 나가야 하는 날씨임. 집에 있으면 손해야.",
                "weather_daily",
                "좋은 날씨 외출압박",
                ("좋은 날씨", "10분 산책"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_implicit_social_micro_cues(self) -> None:
        cases = (
            (
                "카톡 답장 머릿속으로 다 써놓고 전송 안 눌러서 하루 동안 잠수 탄 사람 됨.",
                "chat_social_misfire",
                "머릿속 답장 미전송",
                ("머릿속 답장", "현실 카톡방"),
            ),
            (
                "단톡방에 웃긴 짤 올렸는데 아무도 반응 안 해줘서 혼자 민망함.",
                "chat_social_misfire",
                "단톡방 정적공포",
                ("단톡방 무반응", "혼자 무대"),
            ),
            (
                "친구가 아무거나라더니 내가 고른 메뉴마다 싫대서 그냥 국밥집 데려감.",
                "food_daily",
                "아무거나 싫대 루프",
                ("메뉴 고르기", "지뢰 찾기"),
            ),
            (
                "비 온다더니 해 쨍쨍해서 우산만 하루 종일 짐 됨.",
                "weather_daily",
                "우산 벌칙아이템",
                ("우산", "벌칙 아이템"),
            ),
            (
                "지하철에서 내 앞 사람 내릴 줄 알고 대기 탔는데 종점까지 가더라.",
                "transit_micro_fail",
                "종점 눈치게임 완패",
                ("자리 눈치게임", "종점"),
            ),
            (
                "친구들이 여행 계획 아무거나 좋다더니 숙소 링크마다 딴지 걸어.",
                "planning_contradiction",
                "아무거나 딴지 루프",
                ("아무거나 딴지", "책임만 떠넘긴"),
            ),
            (
                "노래방 고음에서 삑사리 나서 조용히 볼륨 줄이고 흐느끼듯 부름.",
                "performance_embarrassment",
                "노래방 삑사리 생존술",
                ("노래방 삑사리", "생존술"),
            ),
            (
                "퇴근 5분 전에 상사가 잠깐 얘기 좀 할까 하면 속으로 욕하고 겉으론 웃음.",
                "social_mask_survival",
                "자본주의 미소 방어막",
                ("자본주의 미소", "퇴근 버튼"),
            ),
            (
                "스타벅스 닉네임 귀요미님 불려서 고개 숙이고 받아 옴.",
                "performance_embarrassment",
                "닉네임 공개처형",
                ("카페 닉네임 호출", "공개처형"),
            ),
            (
                "배달 기사님 가실 때까지 문 뒤에서 숨죽이고 기다렸다가 음식 가져옴.",
                "home_daily_detail",
                "택배 문앞 매복",
                ("문 앞 택배", "방구석 레이더"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_high_context_micro_social_and_daily_edges(self) -> None:
        cases = (
            (
                "오늘 팀장님 기분 안 좋아 보여서 탕비실 갈 때 숨 참고 지나감.",
                "social_mask_survival",
                "팀장 기분 레이더",
                ("팀장 기분 레이더", "숨 참고"),
            ),
            (
                "회사에서 다 같이 점심 먹자는 분위기인데 오늘은 혼밥하고 싶어서 핑계 찾는 중.",
                "food_daily",
                "강제 점심 핑계찾기",
                ("강제 점심", "혼밥"),
            ),
            (
                "친구가 고민 털어놨는데 해결책을 원하는지 공감을 원하는지 몰라서 일단 끄덕이는 중.",
                "relationship_social",
                "공감 해결책 갈림길",
                ("공감 해결책 갈림길", "듣기만 할까"),
            ),
            (
                "내가 농담쳤는데 상대가 진지하게 받아서 설명하다가 농담이 사망함.",
                "relationship_social",
                "농담 사망 설명회",
                ("농담 사망", "웃음이 증발"),
            ),
            (
                "카톡에 ㅋㅋ 하나만 치면 비웃는 것 같고 다섯 개 치면 과해 보여서 개수 고민함.",
                "chat_social_misfire",
                "ㅋㅋ 개수 눈치게임",
                ("ㅋㅋ 개수", "리액션도 사회생활"),
            ),
            (
                "오늘 회의에서 아무 생각 없이 고개 끄덕이다가 갑자기 질문 받아서 자동반사로 확인해보겠습니다 함.",
                "social_mask_survival",
                "회의 자동반사 방어",
                ("회의 자동반사", "확인해보겠습니다"),
            ),
            (
                "오랜만에 연락 온 지인이 안부 묻더니 바로 보험 얘기 꺼내서 마음의 문 닫힘.",
                "relationship_social",
                "보험 안부 배신",
                ("보험 안부", "마음의 문"),
            ),
            (
                "친구가 밥 산다더니 계산대 앞에서 갑자기 지갑 없다고 해서 내가 냄.",
                "relationship_social",
                "지갑 실종 계산대 덫",
                ("계산대 지갑", "신뢰 테스트"),
            ),
            (
                "마트에서 1+1이라길래 필요 없는 것까지 샀는데 집 와서 보니 유통기한 내일임.",
                "price_shock",
                "원플러스원 유통기한 함정",
                ("1+1 유통기한", "숙제"),
            ),
            (
                "노트북으로 발표하는데 카톡 미리보기로 상사 욕 보일까 봐 알림 끄느라 손 떨림.",
                "privacy_micro_tension",
                "발표 카톡미리보기 공포",
                ("카톡 미리보기", "발표 화면"),
            ),
            (
                "버스에서 내릴게요 세 번 말했는데 아무도 안 비켜줘서 다음 정거장까지 끌려감.",
                "transit_micro_fail",
                "버스 내릴게요 생존콜",
                ("버스 내릴게요", "다음 정거장"),
            ),
            (
                "비 오는 날 신발 안에 물 들어가서 하루 종일 양말이 축축함.",
                "weather_daily",
                "우산 없는 날의 젖은 양말",
                ("젖은 양말", "눅눅"),
            ),
            (
                "친구가 내 말 듣는 척하면서 폰만 보고 있어서 말하다가 기운 빠짐.",
                "relationship_social",
                "듣는 척 폰벽",
                ("듣는 척 폰벽", "혼자 방송"),
            ),
            (
                "혼자 카페 왔는데 옆자리 사람이 내 노트북 화면 계속 훔쳐봐서 신경 쓰임.",
                "privacy_micro_tension",
                "화면 훔쳐보기 레이더",
                ("노트북 화면 훔쳐보기", "남의 시선"),
            ),
            (
                "단톡방에 총대 메고 일정 정리했는데 다 읽고 아무도 답 안 해서 현타 옴.",
                "chat_social_misfire",
                "단톡 총대 무응답 현타",
                ("단톡 총대 무응답", "읽음 표시"),
            ),
            (
                "음식물 쓰레기 버리려고 봉투 들었는데 냄새 올라와서 인류애 잃음.",
                "home_daily_detail",
                "음쓰 최종보스",
                ("음식물 쓰레기", "정신력이 깎"),
            ),
            (
                "알람 5개 맞춰놨는데 전부 끄고 다시 자서 결국 지각함.",
                "body_health",
                "알람 전멸 지각엔딩",
                ("알람", "아침 방어전"),
            ),
            (
                "친구가 늦는다고 해서 괜찮다 했는데 사실 속으로는 이미 삐짐.",
                "relationship_social",
                "괜찮다 속삐짐",
                ("괜찮다 속삐짐", "지각 체크"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_subtext_daily_context_edges(self) -> None:
        cases = (
            (
                "친구가 괜찮다면서 말투가 갑자기 딱딱해졌는데, 내가 뭘 잘못했나 계속 눈치 보는 중.",
                "relationship_social",
                "말투 급냉 눈치게임",
                ("말투 급냉", "온도"),
            ),
            (
                "상사가 별거 아니라고 던진 일이 열어보니까 하루 통째로 먹는 업무 폭탄이었어.",
                "social_mask_survival",
                "별거아님 업무폭탄",
                ("업무 폭탄", "별거 아니다"),
            ),
            (
                "단톡방에서 다들 읽었는데 아무도 결정 안 해서 결국 내가 예약까지 함. 총대 또 나야.",
                "chat_social_misfire",
                "단톡 총대 무응답 현타",
                ("단톡 총대 무응답", "읽음 표시"),
            ),
            (
                "오랜만에 운동했더니 건강해진 느낌보다 내 몸이 폐업 직전이라는 사실만 알게 됐어.",
                "body_health",
                "오랜만 운동 폐업판정",
                ("오랜만 운동", "몸 상태 보고서"),
            ),
            (
                "칭찬 들었는데 좋긴 한데 더 잘해야 할 것 같아서 괜히 부담돼.",
                "relationship_social",
                "칭찬 부담 부스터",
                ("칭찬 부담", "기대치"),
            ),
            (
                "쉬려고 누웠는데 내일 할 일이 갑자기 생각나서 머릿속에서 업무 회의 열림.",
                "commute_work",
                "누웠는데 업무회의",
                ("누웠는데 업무회의", "뇌가 야근"),
            ),
            (
                "장난이라고 했는데 은근 선 넘은 말이라 웃고 넘겼는데 계속 생각남.",
                "relationship_social",
                "선넘은 장난 뒤끝",
                ("선 넘은 장난", "체크표시"),
            ),
            (
                "배달비 아끼려고 포장 주문했는데 왕복 걷다 보니 내가 배달원이 된 기분이야.",
                "food_daily",
                "포장주문 셀프배달",
                ("포장 주문 셀프배달", "다리로 결제"),
            ),
            (
                "새 옷 입고 나갔는데 아무도 못 알아봐서 괜히 나 혼자만 신경 쓴 사람 됨.",
                "fashion_social_detail",
                "새옷 무반응 민망",
                ("새 옷 무반응", "거울"),
            ),
            (
                "카페에서 조용히 쉬려고 했는데 옆자리 회의 소리 때문에 남의 회사 사정 다 알게 됨.",
                "cafe_noise_scene",
                "카페 옆자리 회의중계",
                ("옆자리 회의 중계", "강제 입사"),
            ),
            (
                "엘리베이터가 층마다 서서 엄청 바빴는데 내 인생이 슬로모션으로 흘러가는 줄.",
                "transit_micro_fail",
                "엘리베이터 층마다 정차",
                ("엘리베이터 층마다 정차", "슬로모션"),
            ),
            (
                "고민 듣고 바로 본인 얘기로 넘어가는 사람 만나면 상담 실패한 느낌 들어.",
                "relationship_social",
                "상담 자기얘기 납치",
                ("상담 자기얘기", "방향을 뺏긴"),
            ),
            (
                "배송비 때문에 장바구니 비웠다가 무료배송 채우려고 다시 필요 없는 걸 담는 중.",
                "money_spending",
                "배송비 채우기 역설",
                ("배송비 채우기", "소비를 추가"),
            ),
            (
                "기분 좋았는데 갑자기 과거 흑역사 떠올라서 혼자 조용히 데미지 입음.",
                "memory_mood_crash",
                "흑역사 급습 데미지",
                ("흑역사 급습", "오래된 흑역사"),
            ),
            (
                "네가 편해서 부탁하는 거야 라는 말, 사실 그냥 만만하다는 뜻 같아서 좀 그래.",
                "social_mask_survival",
                "편해서 부탁 만만함",
                ("편해서 부탁", "만만한 사람"),
            ),
            (
                "답장 늦어서 서운했는데 막상 답 오니까 나도 바로 답하기 싫어짐.",
                "relationship_social",
                "답장 늦음 복수심",
                ("답장 늦음", "작은 복수심"),
            ),
            (
                "회의에서 좋은 의견이라고 해놓고 아무도 내 의견 안 써서 칭찬이 공허해졌어.",
                "social_mask_survival",
                "공허한 칭찬 회의",
                ("공허한 칭찬", "허공의 박수"),
            ),
            (
                "스터디카페 왔는데 자리 잡고 커피 마신 것 말고 한 게 없음.",
                "planning_burnout",
                "스터디카페 커피엔딩",
                ("스터디카페 커피엔딩", "공부가 결석"),
            ),
            (
                "엄마한테 괜찮다고 했는데 좀 서운해서 방문 세게 닫아버렸어.",
                "relationship_social",
                "가족 서운 방문쾅",
                ("방문 쾅 서운함", "문소리"),
            ),
            (
                "혼자 밥 먹는 건 괜찮은데 직원이 몇 분이세요 물어보면 외로움이 실체화됨.",
                "food_daily",
                "혼밥 인원질문 외로움",
                ("혼밥 인원 질문", "외로움을 실체화"),
            ),
            (
                "오늘 너무 평범했는데 이상하게 그 평범함이 좀 고맙게 느껴졌어.",
                "life_reflection",
                "평범한 하루 고마움",
                ("평범한 하루 고마움", "안정감"),
            ),
            (
                "힘들다길래 위로해주고 싶은데 말이 가벼워 보일까 봐 못 보내는 중.",
                "relationship_social",
                "위로 문장 무게감",
                ("위로 문장", "말이 가벼워"),
            ),
            (
                "택배 도착 문자 보고 신났는데 뜯어보니 생각보다 별로라 흥미가 식었어.",
                "money_spending",
                "택배 기대 식음",
                ("택배 기대", "설렘"),
            ),
            (
                "내가 먼저 만나자고 해놓고 만나기 전엔 에너지 충전 안 돼서 후회 중.",
                "home_daily_detail",
                "선약 후회 에너지부족",
                ("선약 후회", "배터리"),
            ),
            (
                "카톡으로 장문 썼다가 너무 무거워 보여서 다 지우고 ㅋㅋ만 보냈어.",
                "chat_social_misfire",
                "장문 지우고 ㅋㅋ 방어",
                ("장문 지우고 ㅋㅋ", "웃음표"),
            ),
            (
                "계획은 완벽하게 짰는데 첫 번째 줄부터 시작하기 싫어서 멈췄어.",
                "planning_burnout",
                "계획 장인 실천 파업",
                ("할 일 목록", "한 칸"),
            ),
            (
                "새로 산 옷 입고 갔는데 아무도 몰라봐서 괜히 나 혼자 머쓱했어.",
                "fashion_social_detail",
                "새옷 무반응 민망",
                ("새 옷 무반응", "거울"),
            ),
            (
                "사과하려고 문장 쓰는데 변명처럼 보일까 봐 지웠다 썼다 하는 중이야.",
                "relationship_social",
                "사과 문장 변명공포",
                ("사과 문장", "해명처럼"),
            ),
            (
                "퇴근 직전에 메일 왔는데 오늘은 못 보겠어서 내일의 나한테 넘겼어.",
                "commute_work",
                "퇴근길 폭탄",
                ("퇴근 직전 메일", "내일의 나"),
            ),
            (
                "회의 중에 내 말만 자꾸 끊겨서 내가 투명인간 된 줄 알았어.",
                "relationship_social",
                "말 끊김 소외감",
                ("말 끊김", "존재감"),
            ),
            (
                "상사한테 네라고는 했는데 사실 뭘 하라는지 하나도 못 알아들었어.",
                "social_mask_survival",
                "회의 이해한 척 자동반사",
                ("네 했는데 이해 못함", "되묻"),
            ),
            (
                "카페에서 콘센트 자리 겨우 잡았는데 옆 사람이 충전기 꽂아도 되냐고 해서 애매했어.",
                "cafe_noise_scene",
                "카페 콘센트 양보 눈치",
                ("카페 콘센트", "쪼잔"),
            ),
            (
                "반품해야 하는 택배가 있는데 박스 다시 싸는 게 귀찮아서 현관에 일주일째 있어.",
                "money_spending",
                "반품 박스 방치",
                ("반품 박스", "미션 보스"),
            ),
            (
                "새벽에 감성글 길게 썼다가 아침에 보고 바로 삭제했어.",
                "memory_mood_crash",
                "새벽 감성글 아침검열",
                ("새벽 감성글", "아침의 내가"),
            ),
            (
                "약속 취소됐으면 했는데 진짜 취소되니까 또 살짝 서운하네.",
                "home_daily_detail",
                "약속취소 양가감정",
                ("약속 취소", "빈칸"),
            ),
            (
                "오늘 아무 일도 안 했는데 묘하게 하루를 낭비한 것 같아서 찝찝해.",
                "life_reflection",
                "무생산 하루 찝찝함",
                ("하루 낭비", "회복"),
            ),
            (
                "친구가 괜찮다는데 말끝이 짧아서 괜히 하루 종일 눈치 보게 돼.",
                "relationship_social",
                "말투 급냉 눈치게임",
                ("말투 급냉", "추리 지옥"),
            ),
            (
                "식당에서 혼자 먹는데 직원이 일행분 오세요 물어봐서 갑자기 머쓱했어.",
                "food_daily",
                "혼밥 인원질문 외로움",
                ("혼밥 인원 질문", "외로움"),
            ),
            (
                "계획표만 예쁘게 꾸미고 오늘 할 일은 하나도 못 했어.",
                "planning_burnout",
                "계획 장인 실천 파업",
                ("할 일 목록", "한 칸"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_high_context_subtext_v2_edges(self) -> None:
        cases = (
            (
                "친구 부탁 거절했는데 내가 나쁜 사람 된 기분이라 계속 마음에 걸려.",
                "relationship_social",
                "부탁 거절 죄책감",
                ("부탁 거절 죄책감", "경계를 지킨"),
            ),
            (
                "좋은 일 생겨서 말했는데 별거 아니지 소리 들으니까 기분이 확 식었어.",
                "relationship_social",
                "기쁨 깎는 별거아님",
                ("기쁨 깎는 말", "찬물"),
            ),
            (
                "친구가 좋은 회사 합격했다니까 축하는 하는데 속으로 나는 뭐하고 있나 살짝 씁쓸했어.",
                "relationship_social",
                "축하 속 비교현타",
                ("축하 속 비교현타", "복잡한 감정"),
            ),
            (
                "오랜만에 가족 단톡에 말 걸었는데 다들 읽고 아무 말 없어서 괜히 분위기 깬 사람 됨.",
                "chat_social_misfire",
                "가족단톡 무응답 소외",
                ("가족 단톡 무응답", "읽음만"),
            ),
            (
                "나 오늘은 혼자 있고 싶다고 했는데 막상 아무도 연락 안 오니까 또 좀 외롭네.",
                "relationship_social",
                "혼자있고싶다 연락공백",
                ("혼자 있고 싶은데 외로움", "공백"),
            ),
            (
                "상사가 오늘 고생했다 한마디 했는데 이상하게 피로가 조금 풀렸어.",
                "social_mask_survival",
                "고생했다 한마디 회복",
                ("고생했다 한마디", "누가 봤다는 느낌"),
            ),
            (
                "칭찬받았는데 바로 수정사항 열 개 붙어서 이게 칭찬인지 숙제인지 모르겠어.",
                "social_mask_survival",
                "칭찬 수정폭탄",
                ("칭찬 수정폭탄", "수정 폭탄"),
            ),
            (
                "대화 중에 내 농담만 매번 설명해야 해서 내가 재미없는 사람 된 느낌이야.",
                "relationship_social",
                "농담 설명 자괴감",
                ("농담 설명", "코드가 안 맞은"),
            ),
            (
                "오늘 하루 평범하게 지나갔는데 밤 되니까 이상하게 이대로 살아도 되나 싶더라.",
                "life_reflection",
                "평범한 하루 존재질문",
                ("이대로 살아도 되나", "인생 중간점검"),
            ),
            (
                "회사에서 내가 한 일인데 회의에서 다른 사람이 자연스럽게 자기 공처럼 말해서 벙쪘어.",
                "social_mask_survival",
                "공 가로채기 벙찜",
                ("내 공 가로채기", "기록"),
            ),
            (
                "단톡방에 힘들다고 썼다가 관심 구걸하는 것 같아서 바로 삭제했어.",
                "chat_social_misfire",
                "힘듦 삭제 관심구걸공포",
                ("힘들다 썼다 삭제", "신호를 보낼 곳"),
            ),
            (
                "새 물건 샀을 때보다 박스 뜯기 전까지 기다리는 시간이 더 설레는 것 같아.",
                "money_spending",
                "언박싱 전 설렘",
                ("언박싱 전 설렘", "쇼핑 도파민"),
            ),
            (
                "할 일 미뤄놓고 쉬는 건데 쉬는 것도 마음이 편하지 않아서 더 지쳐.",
                "planning_burnout",
                "쉬어도 죄책감",
                ("쉬어도 죄책감", "대기 모드"),
            ),
            (
                "아무 생각 없이 한 말에 친구 표정이 굳어서 그 뒤로 계속 머릿속에서 리플레이 돼.",
                "relationship_social",
                "말실수 리플레이",
                ("말실수 리플레이", "자동 재생"),
            ),
            (
                "오랜만에 산책 나갔는데 별거 아닌 바람 냄새에 기분이 좀 살아났어.",
                "scenery_mood",
                "산책 바람 회복",
                ("산책 바람 냄새", "밖의 리듬"),
            ),
            (
                "혼자 밥 먹는 건 괜찮은데 주변이 다 커플이라 괜히 내가 배경처럼 느껴졌어.",
                "food_daily",
                "혼밥 커플 배경화",
                ("혼밥 커플 배경", "주변 대비"),
            ),
            (
                "누가 나한테 괜찮아 보인다고 했는데 사실 안 괜찮아서 더 서러웠어.",
                "relationship_social",
                "괜찮아보임 서러움",
                ("괜찮아 보인다는 말", "속을 못 봐준다는"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_high_context_subtext_v3_edges(self) -> None:
        cases = (
            (
                "나 오늘 발표 끝냈어. 망하진 않은 것 같은데 그냥 누가 잘했다고 한마디만 해줬으면 좋겠어.",
                "relationship_social",
                "잘했다 한마디 요청",
                ("잘했다 한마디", "인정 욕구"),
            ),
            (
                "친구가 선물 고맙다고는 했는데 표정이 애매해서 괜히 내가 센스 없는 사람 된 느낌이야.",
                "relationship_social",
                "선물 리액션 눈치",
                ("선물 리액션", "표정 해석"),
            ),
            (
                "나 사실 그 약속 별로 안 가고 싶은데 먼저 잡은 거라 취소하자고 말하기가 너무 눈치 보여.",
                "relationship_social",
                "내가잡은약속 취소눈치",
                ("약속 취소 눈치", "체력이 바닥"),
            ),
            (
                "친구가 힘들다길래 들어줬는데, 세 시간째 같은 얘기라 내 에너지가 다 빨렸어. 나 나쁜 사람인가.",
                "relationship_social",
                "상담소진 죄책감",
                ("상담 소진", "감정 체력"),
            ),
            (
                "상사가 칭찬하는 척하면서 결국 더 어려운 일을 맡기는데 이거 좋게 봐야 돼?",
                "social_mask_survival",
                "칭찬포장 업무떠넘김",
                ("칭찬 포장 업무", "떠넘기기"),
            ),
            (
                "카톡에 ㅋㅋ만 보냈는데 너무 차갑게 보였을까 봐 다시 이모티콘 보낼까 고민 중.",
                "chat_social_misfire",
                "ㅋㅋ 온도보정 불안",
                ("ㅋㅋ 개수", "미세 조절"),
            ),
            (
                "오늘 별거 아닌 일로 칭찬받았는데 괜히 하루 종일 기분 좋아서 나 좀 단순한가 싶어.",
                "relationship_social",
                "작은칭찬 하루부스터",
                ("작은 칭찬", "하루를 살리는 연료"),
            ),
            (
                "친구가 나한테만 장난 세게 치는데 친해서 그런 건지 만만해서 그런 건지 모르겠어.",
                "relationship_social",
                "장난 친함만만함 경계",
                ("장난의 경계선", "만만해서"),
            ),
            (
                "내가 먼저 밥 먹자고 해놓고 막상 날짜 잡히니까 체력 없어서 도망가고 싶어.",
                "relationship_social",
                "내가잡은약속 취소눈치",
                ("약속 취소 눈치", "다시 날짜"),
            ),
            (
                "누가 내 얘기 잘 들어줘서 고마운데 너무 진지하게 받아줘서 오히려 민망했어.",
                "relationship_social",
                "진지한 경청 민망함",
                ("진지한 경청", "민망"),
            ),
            (
                "엄마가 밥 먹었냐고 물어본 건데 이상하게 오늘은 그게 위로처럼 들렸어.",
                "relationship_social",
                "밥먹었냐 위로",
                ("밥 먹었냐는 말", "돌봄"),
            ),
            (
                "나 오늘 아무 말 안 하고 있었는데 친구가 기분 안 좋아? 하고 물어서 좀 들킨 기분이야.",
                "relationship_social",
                "기분 들킨 방어막",
                ("기분 들킨 느낌", "방어막"),
            ),
            (
                "친구 좋은 소식 듣고 진심으로 축하했는데 집에 와서 혼자 비교하게 돼서 내가 싫어졌어.",
                "relationship_social",
                "축하 후 비교자책",
                ("축하 후 비교", "인간적인 반응"),
            ),
            (
                "단톡방에서 내가 말하면 대화가 끊기는 것 같아서 이제 그냥 리액션만 하게 돼.",
                "chat_social_misfire",
                "단톡방 정적공포",
                ("단톡방 무반응", "혼자 무대"),
            ),
            (
                "회사에서 별일 아닌 실수 했는데 다들 괜찮다 해서 더 창피했어.",
                "relationship_social",
                "괜찮다 더창피",
                ("괜찮다 해서 더 창피", "확대 재생"),
            ),
            (
                "좋아하는 사람이 내 스토리만 보고 답장은 안 해서 별거 아닌데 계속 신경 쓰여.",
                "relationship_social",
                "답장 온도차",
                ("답장 온도차", "내 말만 밀린"),
            ),
            (
                "나 진짜 괜찮은 척 잘하는 것 같아. 아무도 모르는 게 다행이면서도 좀 서럽네.",
                "relationship_social",
                "괜찮은척 고독",
                ("괜찮은 척", "조금은 알아줬으면"),
            ),
            (
                "선물 받았는데 취향은 아닌데 마음은 고마워서 리액션을 어떻게 해야 할지 모르겠어.",
                "relationship_social",
                "선물 리액션 눈치",
                ("선물 리액션", "취향 힌트"),
            ),
            (
                "오늘 운동 조금 했다고 뿌듯해하는 내가 웃기긴 한데 그래도 기분 좋다.",
                "body_health",
                "작은운동 셀프칭찬",
                ("작은 운동", "몸을 움직였다는 사실"),
            ),
            (
                "사람 만나고 오면 재밌었는데도 집 와서 혼자 아무 말도 안 하고 있어야 충전돼.",
                "relationship_social",
                "사회적 배터리 충전",
                ("사회적 배터리", "무음 모드"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_high_context_subtext_v4_edges(self) -> None:
        cases = (
            (
                "친구가 예전에 내가 좋아한다고 한 간식 기억해뒀다가 사왔는데 별거 아닌데 감동했어.",
                "relationship_social",
                "취향기억 작은감동",
                ("취향 기억", "저장해뒀다는 느낌"),
            ),
            (
                "생일 축하가 자정에 올 줄 알았는데 밤늦게 와서 별거 아닌데 좀 서운했어.",
                "relationship_social",
                "생일축하 지각서운",
                ("생일 축하", "제때 떠올렸나"),
            ),
            (
                "단톡방에서 나만 빼고 약속 잡은 걸 뒤늦게 알았는데 아무렇지 않은 척했어.",
                "relationship_social",
                "나만빼고 약속소외",
                ("나만 빼고 약속", "내 자리까지 빠진"),
            ),
            (
                "내가 도와준 일에 고맙다는 말도 없어서 티는 안 냈는데 기분이 좀 식었어.",
                "relationship_social",
                "도와줬는데 감사누락",
                ("고맙다는 말 누락", "당연한 사람처럼"),
            ),
            (
                "상대가 미안하다고 하니까 더 뭐라 못하겠는데 마음은 아직 안 풀렸어.",
                "relationship_social",
                "사과받아도 잔감정",
                ("사과 후 잔감정", "남은 감정"),
            ),
            (
                "내 얘기를 기억 못 하는 건 이해하는데 몇 번 말한 걸 또 물어보니까 좀 서운해.",
                "relationship_social",
                "반복설명 기억서운",
                ("반복 설명", "얼마나 담아뒀는지"),
            ),
            (
                "친구가 내 말 따라 웃긴 했는데 눈이 안 웃어서 괜히 민망했어.",
                "relationship_social",
                "눈안웃는 리액션",
                ("눈 안 웃는 리액션", "표정 온도"),
            ),
            (
                "좋은 소식 말했는데 다들 반응이 밋밋해서 말한 내가 괜히 민망해졌어.",
                "relationship_social",
                "좋은소식 밋밋반응",
                ("좋은 소식 밋밋 반응", "같이 기뻐해줄"),
            ),
            (
                "나도 축하받고 싶어서 얘기한 건 아닌데 아무도 안 물어봐서 살짝 섭섭했어.",
                "relationship_social",
                "축하욕구 미확인",
                ("축하받고 싶은 마음", "내 기쁨"),
            ),
            (
                "내가 싫다고 말한 걸 장난으로 계속하니까 웃으면서 넘기기가 힘들어.",
                "relationship_social",
                "싫다는데 장난반복",
                ("싫다는데 장난", "경계선 문제"),
            ),
            (
                "내가 말하고 있는데 상대가 시계만 봐서 빨리 끝내라는 신호처럼 느껴졌어.",
                "relationship_social",
                "시계보는 대화압박",
                ("대화 중 시계를 보는", "몸짓"),
            ),
            (
                "친구가 바쁘다더니 다른 모임 사진은 올려서 내가 쪼잔한가 싶으면서도 서운했어.",
                "relationship_social",
                "바쁘다더니 모임사진",
                ("바쁘다더니 모임 사진", "우선순위"),
            ),
            (
                "누가 내 이름 제대로 기억해줬는데 별것도 아닌데 이상하게 기분 좋더라.",
                "relationship_social",
                "이름기억 작은감동",
                ("이름 기억", "저장된 사람"),
            ),
            (
                "칭찬인 줄 알았는데 뒤에 부탁이 붙으니까 기분이 좀 애매했어.",
                "relationship_social",
                "칭찬뒤 부탁애매",
                ("칭찬 뒤 부탁", "포장지"),
            ),
            (
                "내가 먼저 연락 안 하면 아무도 안 하는 것 같아서 일부러 며칠 가만히 있었어.",
                "relationship_social",
                "먼저연락 실험공백",
                ("먼저 연락 실험", "관계도 멈추는지"),
            ),
            (
                "친구가 나한테만 편하게 막말하는데 친한 거랑 무례한 게 헷갈려.",
                "relationship_social",
                "편한말 무례경계",
                ("편한 말의 경계", "막 해도 된다는 뜻"),
            ),
            (
                "오랜만에 만났는데 예전 얘기만 하고 지금의 나는 별로 안 궁금한가 싶었어.",
                "relationship_social",
                "과거얘기 현재소외",
                ("예전 얘기만 하는 만남", "현재의 나도"),
            ),
            (
                "내가 괜찮다고 말했더니 진짜 괜찮은 줄 알고 넘어가서 좀 허무했어.",
                "relationship_social",
                "괜찮다 믿어버림 허무",
                ("괜찮다 믿어버림", "붙잡아주길"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_high_context_subtext_v5_edges(self) -> None:
        cases = (
            (
                "친구가 해결책 말하면 싫어하고 공감만 해달라는데, 뭐라 해야 할지 모르겠어.",
                "relationship_social",
                "답정너 위로압박",
                ("답정너 위로 압박", "해결책인지 편들기인지"),
            ),
            (
                "친구가 맨날 나한테만 하소연해서 듣다 보니 지쳐. 내가 감정 쓰레기통 된 것 같아.",
                "relationship_social",
                "감정쓰레기통 피로",
                ("감정 쓰레기통 피로", "용량이 찬"),
            ),
            (
                "시간 나면 보자고 하는데 뭔가 보험용 약속 같아서 내가 우선순위 낮은 사람 된 느낌이야.",
                "relationship_social",
                "보험용 약속서운",
                ("보험용 약속", "빈칸 채우기 후보"),
            ),
            (
                "도와줬으니까 이 정도는 해줘야지 하는 말 들으니까 고마움보다 빚진 느낌이 들어.",
                "relationship_social",
                "친절뒤 계산서",
                ("친절 뒤 계산서", "빚처럼 남는"),
            ),
            (
                "친구가 농담처럼 비교하는데 웃자고 한 말이라도 괜히 상처가 남아.",
                "relationship_social",
                "농담속 비교딜",
                ("농담 속 비교딜", "나를 깎으면"),
            ),
            (
                "둘이 있는데 침묵이 길어지면 내가 무슨 말 해야 할 것 같아서 괜히 불편해.",
                "relationship_social",
                "단둘이 침묵압박",
                ("단둘이 침묵", "내가 뭔가 해야"),
            ),
            (
                "답장 늦은 이유를 매번 해명해야 하는 것 같아서 카톡이 숙제처럼 느껴져.",
                "relationship_social",
                "읽씹해명 피로",
                ("읽씹 해명", "숙제처럼"),
            ),
            (
                "갑자기 친한 척하면서 사적인 걸 캐묻는데 왜 묻지 싶어서 경계하게 돼.",
                "relationship_social",
                "친한척 정보캐기",
                ("친한 척 정보 캐기", "경계심"),
            ),
            (
                "말 안 해도 기억해줬으면 했던 내가 좀 웃긴데, 안 기억해주니까 서운하더라.",
                "relationship_social",
                "기억해달라 말못함",
                ("기억해달라 말못함", "요구하면 부담"),
            ),
            (
                "위로해주고 싶었는데 타이밍 놓쳐서 말 못했어. 그때 말했어야 했나 계속 후회돼.",
                "relationship_social",
                "위로타이밍 놓침",
                ("위로 타이밍", "늦은 말"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_high_context_subtext_v6_inner_ambivalence(self) -> None:
        cases = (
            (
                "목표 달성했는데 끝나고 나니까 이상하게 허탈하고 빈 느낌이야.",
                "life_reflection",
                "목표달성 공허감",
                ("목표 달성 공허감", "긴장이 빠진 자리"),
            ),
            (
                "칭찬받았는데 내가 잘한 게 아니라 운이 좋았던 것 같아서 다음엔 들킬까 봐 불안해.",
                "relationship_social",
                "칭찬받고 운빨불안",
                ("칭찬받고 운빨 불안", "다음엔 들키는"),
            ),
            (
                "쉬는 날인데 잘 쉬어야 할 것 같아서 오히려 휴식도 숙제처럼 느껴져.",
                "life_reflection",
                "휴식예약 불안",
                ("휴식예약 불안", "휴식도 평가받는"),
            ),
            (
                "세일이라 샀는데 막상 집에 오니까 필요 없어서 괜히 산 듯해.",
                "money_spending",
                "할인득템 허무소비",
                ("할인한다고 산", "살 이유를 할인으로"),
            ),
            (
                "잘했다고 듣고 싶은데 칭찬해달라고 말하기 민망해서 그냥 가만히 있었어.",
                "relationship_social",
                "칭찬요청 민망함",
                ("칭찬요청 민망함", "확인받고 싶은 마음"),
            ),
            (
                "내가 고른 건데 결정하고 나서 계속 괜히 골랐나 싶고 책임지는 게 무서워.",
                "life_reflection",
                "혼자선택 책임불안",
                ("혼자선택 책임불안", "책임져야 한다는 감각"),
            ),
            (
                "좋은 일 생기면 기분 좋은데도 이러다 안 좋은 일 생길까 봐 괜히 불안해져.",
                "life_reflection",
                "좋은일 직후 불안",
                ("좋은 일이 생겼는데", "방어막"),
            ),
            (
                "관심받고 싶었는데 막상 주목받으니까 부담스럽고 숨고 싶어졌어.",
                "relationship_social",
                "관심받고 부담",
                ("관심받고 부담", "조명이 너무 밝아지는"),
            ),
            (
                "다이어트 한다고 공개 선언해놨는데 못 지키면 쪽팔릴 것 같아서 괜히 말했나 후회돼.",
                "planning_burnout",
                "계획공유 후회",
                ("계획공유 후회", "말한 순간 약속"),
            ),
            (
                "돈 아끼려다 싼 거 샀는데 품질 별로라 오히려 후회돼. 비싼 거 살 걸.",
                "money_spending",
                "가성비선택 후회",
                ("가성비선택 후회", "아낀 돈보다"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_salience_direct_replies_cover_high_context_subtext_v7_new_fifty_edges(self) -> None:
        cases = (
            (
                "좋아하는 취미를 일로 삼으면 언젠가 싫어질까 봐 살짝 겁나.",
                "life_reflection",
                "좋아하는일 직업화불안",
                ("좋아하는 일을 직업", "애정도 체력 싸움"),
            ),
            (
                "오늘은 진짜 쉬려고 했는데 쉬는 내내 밀린 일 생각나서 하나도 쉰 것 같지가 않아.",
                "planning_burnout",
                "쉬는중 일침투",
                ("쉬는 중 일 생각", "대기 모드"),
            ),
            (
                "사람 많은 곳은 재밌었는데 집에 오자마자 소리만 떠올라도 기가 빨려.",
                "relationship_social",
                "소리많은곳 방전",
                ("자극에 배터리", "바로 방전"),
            ),
            (
                "시작하기 전에는 설레는데 막상 시작하면 바로 부담으로 바뀌는 거 있지.",
                "life_reflection",
                "시작전 설렘부담",
                ("시작 전 설렘", "첫 발 떼기"),
            ),
            (
                "내가 좋아하는 플레이리스트를 남에게 들려주는 게 묘하게 심사받는 느낌이라 긴장돼.",
                "relationship_social",
                "취향공유 심사불안",
                ("플레이리스트", "심사받는 느낌"),
            ),
            (
                "내가 잘하고 있는지 모르겠을 때 누가 딱 한마디만 확신을 줬으면 좋겠어.",
                "relationship_social",
                "확신한마디 갈증",
                ("확신 한마디", "방향 확인"),
            ),
            (
                "계획은 완벽하게 짰는데 첫 줄 시작하기도 전에 이미 지쳤어.",
                "planning_burnout",
                "계획 장인 실천 파업",
                ("계획 첫 줄 방전", "제일 쉬운 한 칸"),
            ),
            (
                "내가 한 농담을 설명해야 하는 순간, 웃음보다 자존심이 먼저 사라져.",
                "relationship_social",
                "농담 설명 자괴감",
                ("농담 설명", "웃음이 증발"),
            ),
            (
                "작은 칭찬 하나 들었을 뿐인데 하루가 갑자기 살만해지는 내가 좀 단순한가.",
                "relationship_social",
                "작은칭찬 하루부스터",
                ("작은 칭찬", "마음이 아직 잘 반응"),
            ),
            (
                "새로운 사람 만나고 즐거웠어도 집에 오자마자 방전돼서 아무 말도 못 하겠어.",
                "relationship_social",
                "사회적 배터리 충전",
                ("사회적 배터리", "무음 모드"),
            ),
        )

        for text, expected_cluster, expected_label, expected_fragments in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                ranked = rank_semantic_candidates(signals, limit=5)
                profile = _salience_profile(
                    signals=signals,
                    semantic_tags=_input_tags(signals),
                    sense_tags=(),
                    semantic_candidates=ranked,
                    must_preserve=[],
                )
                direct = _render_salience_dominant_direct_reply(text)

                self.assertEqual(ranked[0].value, expected_label)
                self.assertEqual(profile["dominant_context"]["cluster"], expected_cluster)
                self.assertEqual(profile["dominant_context"]["label"], expected_label)
                for fragment in expected_fragments:
                    self.assertIn(fragment, direct)

    def test_semantic_candidate_ranking_gates_shared_noise_by_context(self) -> None:
        data_signals = _text_signals(
            "학습 데이터에 노이즈 조금만 껴도 답변 나락 가서 데이터 정제가 핵심이더라."
        )
        audio_signals = _text_signals(
            "이어폰 한쪽 노이즈 미세하게 밸런스 안 맞아서 좌우 밸런스 1단위로 조정 중."
        )
        data_ranked = rank_semantic_candidates(
            data_signals,
            limit=5,
        )
        audio_ranked = rank_semantic_candidates(
            audio_signals,
            limit=5,
        )
        audio_tags = set(_input_tags(audio_signals))

        self.assertEqual(data_ranked[0].value, "데이터 정제 장인전")
        self.assertNotIn("좌우 밸런스 디버깅", [item.value for item in data_ranked[:3]])
        self.assertEqual(audio_ranked[0].value, "좌우 밸런스 디버깅")
        self.assertEqual(audio_ranked[0].context_gate, "accepted")
        self.assertNotIn("data_cleaning", audio_tags)
        self.assertNotIn("comfort", audio_tags)
        self.assertNotIn("support", audio_tags)
        self.assertNotIn("relationship", audio_tags)
        self.assertIn("mismatch", audio_tags)
        self.assertNotIn("pain", audio_tags)

    def test_semantic_candidate_ranking_suppresses_generic_day_in_work_context(self) -> None:
        ranked = rank_semantic_candidates(
            _text_signals("오늘 일 너무 많아서 야근각이야."),
            limit=5,
        )

        self.assertTrue({"work", "task"} <= set(ranked[0].candidate_tags))
        self.assertFalse(
            any(item.value == "일" and "day" in item.candidate_tags for item in ranked[:3]),
            ranked[:3],
        )

    def test_semantic_candidate_ranking_covers_more_short_ambiguous_words(self) -> None:
        cases = (
            (
                "두 배로 힘들고 세 배로 피곤해.",
                "배수",
                {"number", "multiple"},
            ),
            (
                "믿었던 친구한테 배신당한 느낌이야.",
                "배신",
                {"betrayal", "hurt"},
            ),
            (
                "그 배우 연기 진짜 좋더라.",
                "배우",
                {"actor", "acting"},
            ),
            (
                "요즘 코딩 배우고 있는데 어렵다.",
                "배우다",
                {"learning", "study"},
            ),
            (
                "길 찾다가 골목에서 헤맸어.",
                "길",
                {"road", "navigation"},
            ),
            (
                "이 일을 해결할 길이 안 보인다.",
                "살길",
                {"method", "solution"},
            ),
            (
                "문제 문항이 너무 어려워.",
                "문제",
                {"problem", "question"},
            ),
            (
                "방 불 끄고 누웠어.",
                "불빛",
                {"light", "lighting"},
            ),
            (
                "불안해서 잠이 안 와.",
                "불안",
                {"anxiety", "worry"},
            ),
            (
                "모닥불 피워놓고 멍 때렸어.",
                "불",
                {"fire", "danger"},
            ),
        )

        for text, expected_value, expected_tags in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_candidates(_text_signals(text), limit=6)
                matched = next((item for item in ranked if item.value == expected_value), None)

                self.assertIsNotNone(matched, ranked)
                self.assertTrue(expected_tags <= set(matched.candidate_tags), matched)

    def test_semantic_word_bank_keeps_design_detail_out_of_animal_metaphors(self) -> None:
        ranked = rank_semantic_word(
            _text_signals("캐릭터 고유 컬러칩이랑 머리색 눈색 대비가 진짜 디테일 미쳤음."),
            desired_tags=("design", "color", "detail", "character", "visual"),
            default="기본 반응",
        )

        self.assertEqual(ranked.value, "컬러칩 과몰입")
        self.assertIn("design", ranked.matched_tags)

    def test_word_sense_bank_disambiguates_headache_from_hair(self) -> None:
        signals = _text_signals("머리가 아프다")
        senses = {(sense.word, sense.sense) for sense in resolve_word_senses(signals)}
        inferred_tags = set(_input_tags(signals))
        ranked = rank_semantic_word(
            signals,
            desired_tags=("short_word", "body", "pain", "headache", "health"),
            default="기본 반응",
        )

        self.assertIn(("머리", "body_head"), senses)
        self.assertIn("headache", inferred_tags)
        self.assertIn("health", inferred_tags)
        self.assertNotIn("beauty", inferred_tags)
        self.assertNotIn("haircut", inferred_tags)
        self.assertEqual(ranked.value, "두통")

    def test_word_sense_bank_keeps_hair_and_thinking_separate(self) -> None:
        hair_tags = set(_input_tags(_text_signals("머리 잘랐는데 완전 망했어")))
        thinking_tags = set(_input_tags(_text_signals("머리 좀 굴려봐. 아이디어가 필요해.")))

        self.assertIn("beauty", hair_tags)
        self.assertIn("hair", hair_tags)
        self.assertNotIn("headache", hair_tags)
        self.assertIn("thinking", thinking_tags)
        self.assertIn("idea", thinking_tags)
        self.assertNotIn("haircut", thinking_tags)

    def test_word_sense_context_disambiguates_ambiguous_current_phrase(self) -> None:
        cases = (
            (
                ("앞머리 망해서 모자 쓰고 미용실 다시 갈지 고민 중",),
                "hair_style",
                {"beauty", "hair"},
            ),
            (
                ("아이디어가 안 나와서 문제 해결 방향을 계속 고민 중",),
                "thinking_brain",
                {"thinking", "idea"},
            ),
            (
                ("아까부터 두통이 있고 컨디션이 영 별로야",),
                "body_head",
                {"headache", "health"},
            ),
        )

        for recent_texts, expected_sense, expected_tags in cases:
            with self.subTest(expected_sense=expected_sense):
                context = build_word_sense_context_from_texts(recent_texts)
                signals = _text_signals("머리 진짜 답 없다", sense_context=context)
                senses = {sense.sense for sense in resolve_word_senses(signals, context=context)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_sense, senses)
                self.assertTrue(expected_tags <= inferred_tags)

    def test_word_sense_phrase_boost_handles_high_context_collocations(self) -> None:
        cases = (
            ("그 배우 연기 진짜 좋더라", ("배", "actor_entertainment"), {"actor", "acting"}),
            ("요즘 코딩 배우고 있는데 어렵다", ("배", "learning_study"), {"learning", "study"}),
            ("믿었던 친구한테 배신당한 느낌이야", ("배", "betrayal"), {"betrayal", "hurt"}),
            ("두 배로 힘들고 세 배로 피곤해", ("배", "numeric_multiple"), {"number", "multiple"}),
            ("말이 너무 길어져서 미안", ("길", "length_long"), {"length", "long"}),
            ("길 찾다가 골목에서 헤맸어", ("길", "road_navigation"), {"road", "navigation"}),
            ("이 일을 해결할 길이 안 보인다", ("길", "method_solution"), {"method", "solution"}),
            ("내 묘비명 첫 문장을 뭐라고 쓸까", ("문", "sentence_language"), {"sentence", "writing"}),
            ("문제 문항이 너무 어려워", ("문", "problem_question"), {"problem", "question"}),
            ("방 불 끄고 누웠어", ("불", "light_lamp"), {"light", "lighting"}),
            ("불안해서 잠이 안 와", ("불", "anxiety_worry"), {"anxiety", "worry"}),
            ("모닥불 피워놓고 멍 때렸어", ("불", "fire_danger"), {"fire", "danger"}),
            ("오늘 하루가 너무 길었다", ("일", "day_time"), {"time", "day"}),
            ("회사 일이 너무 많이 쌓였어", ("일", "work_task"), {"work", "task"}),
            ("갑자기 무슨 일이 터진 거야", ("일", "event_happening"), {"event", "situation"}),
        )

        for text, expected_pair, expected_tags in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = {(sense.word, sense.sense): sense for sense in resolve_word_senses(signals)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_pair, senses)
                self.assertTrue(expected_tags <= inferred_tags)
                self.assertTrue(
                    any(cue.startswith("phrase:") for cue in senses[expected_pair].matched_cues),
                    senses[expected_pair].matched_cues,
                )

    def test_word_sense_local_window_cues_handle_separated_collocations(self) -> None:
        cases = (
            ("배 지금 너무 아파서 누워있어", ("배", "body_stomach"), {"stomach", "pain"}),
            ("머리가 오늘 계속 지끈거려서 집중이 안 돼", ("머리", "body_head"), {"headache", "health"}),
            ("머리 요즘 너무 복잡해서 생각이 안 정리돼", ("머리", "thinking_brain"), {"thinking", "mental"}),
            ("말 길이가 읽다 지칠 정도야", ("길", "length_long"), {"length", "long"}),
            ("방 문 좀 조용히 닫아줘", ("문", "door_object"), {"door", "object"}),
            ("회사 일이 계속 쌓여서 숨 막혀", ("일", "work_task"), {"work", "task"}),
            ("오늘 하루 일 진짜 길었다", ("일", "day_time"), {"time", "day"}),
            ("줄 지금 너무 길어서 기다리다 지쳤어", ("줄", "queue_line"), {"queue", "waiting"}),
            ("줄 이어폰이랑 같이 넣었더니 꺼낼 때마다 꼬여", ("줄", "cable_line"), {"cable", "wire"}),
            ("문장 줄 하나만 남기라면 뭐라고 쓰지", ("줄", "text_line"), {"text", "writing"}),
            ("문자 방금 보내고 답장 기다리는 중이야", ("문자", "message_text"), {"message", "chat"}),
            ("문자 한 글자만 새로 만든다면 자음을 넣고 싶어", ("문자", "written_character"), {"letter", "character"}),
            ("자리 버스에서 하나 났는데 앉을까 말까 눈치 봤어", ("자리", "seat_place"), {"seat", "place"}),
            ("자리 팀에서 애매해서 내 역할을 다시 잡아야 할 것 같아", ("자리", "position_role"), {"role", "position"}),
            ("등 지금 뻐근해서 기대기도 힘들어", ("등", "body_back"), {"back", "pain"}),
            ("등 형광등이 계속 깜빡거려서 거슬려", ("등", "light_object"), {"light", "lamp"}),
            ("선 친구가 농담이라고 넘어서 기분 나빠", ("선", "social_boundary"), {"boundary", "social"}),
            ("선 그림 그을 때마다 삐뚤어서 거슬려", ("선", "visual_line"), {"visual", "line"}),
            ("선 생일에 받은 게 센스 있어서 기분 좋았어", ("선", "gift"), {"gift", "present"}),
        )

        for text, expected_pair, expected_tags in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = {(sense.word, sense.sense): sense for sense in resolve_word_senses(signals)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_pair, senses)
                self.assertTrue(expected_tags <= inferred_tags)
                self.assertTrue(
                    any(cue.startswith("local_window:") for cue in senses[expected_pair].matched_cues),
                    senses[expected_pair].matched_cues,
                )

    def test_black_draft_uses_current_local_window_for_ambiguous_word_frame(self) -> None:
        cases = (
            (
                "배 지금 너무 아파서 누워있어",
                "contextual_stomach_state",
                "배 쪽",
                "body_stomach",
            ),
            (
                "머리 요즘 너무 복잡해서 생각이 안 정리돼",
                "contextual_thinking_state",
                "생각 쪽 머리",
                "thinking_brain",
            ),
            (
                "말투가 너무 차갑게 느껴져서 계속 신경 쓰여",
                "contextual_speech_state",
                "말이나 말투",
                "speech",
            ),
            (
                "발이 새 신발 때문에 계속 아파서 걷기 힘들어",
                "contextual_foot_state",
                "발 쪽",
                "body_foot",
            ),
            (
                "발표 생각만 하면 심장이 뛰어서 첫 문장이 안 나와",
                "contextual_presentation_state",
                "발표 쪽",
                "presentation",
            ),
            (
                "회사 업무가 계속 쌓여서 처리할 엄두가 안 나",
                "contextual_work_task_state",
                "업무",
                "work_task",
            ),
            (
                "갑자기 무슨 일이 터진 거야",
                "contextual_event_state",
                "상황",
                "event_happening",
            ),
            (
                "옆 사람이랑 눈 마주쳐서 너무 민망했어",
                "contextual_gaze_social_state",
                "시선",
                "gaze_attention",
            ),
            (
                "처음 가는 골목에서 길을 잃어서 계속 헤맸어",
                "contextual_road_navigation_state",
                "지도",
                "road_navigation",
            ),
            (
                "카톡 설명이 너무 길어져서 핵심이 하나도 안 보여",
                "contextual_long_text_state",
                "한 문장",
                "length_long",
            ),
            (
                "이 문제는 해결할 길이 안 보여서 멘붕이야",
                "contextual_solution_path_state",
                "돌파구",
                "method_solution",
            ),
            (
                "현관문이 안 열려서 문 앞에서 멈춰 섰어",
                "contextual_door_state",
                "현관",
                "door_object",
            ),
            (
                "첫 문장이 너무 어색해서 글이 안 써져",
                "contextual_sentence_state",
                "문장",
                "sentence_language",
            ),
            (
                "시험 문제 풀이가 계속 막혀서 정답을 못 찾겠어",
                "contextual_problem_question_state",
                "풀이",
                "problem_question",
            ),
            (
                "냄비에 불붙는 줄 알고 진짜 식은땀 났어",
                "contextual_fire_state",
                "화재",
                "fire_danger",
            ),
            (
                "방 불을 껐는데도 불빛이 계속 거슬려",
                "contextual_light_state",
                "조명",
                "light_lamp",
            ),
            (
                "내일 발표 때문에 불안해서 손이 떨려",
                "contextual_anxiety_state",
                "불안",
                "anxiety_worry",
            ),
            (
                "마트에서 장을 봐야 하는데 목록 정리부터 귀찮아",
                "contextual_shopping_market_state",
                "장보기",
                "shopping_market",
            ),
            (
                "장트러블 때문에 속이 안 좋아서 화장실만 찾는 중",
                "contextual_bowel_state",
                "장",
                "body_bowel",
            ),
            (
                "책 다음 장을 넘겨야 하는데 집중이 끊겼어",
                "contextual_page_chapter_state",
                "페이지",
                "book_page_chapter",
            ),
            (
                "긴 줄 기다리다 지쳤어",
                "contextual_queue_line_state",
                "기다림",
                "queue_line",
            ),
            (
                "이어폰 줄이 자꾸 꼬여서 꺼낼 때마다 빡쳐",
                "contextual_cable_line_state",
                "케이블",
                "cable_line",
            ),
            (
                "후기 첫 줄이 안 써져서 문장이 계속 맴돌아",
                "contextual_text_line_state",
                "한 줄",
                "text_line",
            ),
            (
                "문자 답장 타이밍 놓쳐서 반나절 뒤에 이제 봤다고 했어",
                "contextual_message_text_state",
                "답장",
                "message_text",
            ),
            (
                "한글 문자 하나를 더 만들 수 있다면 어떤 자음을 넣을까",
                "contextual_written_character_state",
                "자모",
                "written_character",
            ),
            (
                "버스에서 빈자리가 났는데 앉을까 말까 눈치 봤어",
                "contextual_seat_place_state",
                "눈치싸움",
                "seat_place",
            ),
            (
                "팀에서 내 자리가 애매해진 느낌이라 좀 불편해",
                "contextual_position_role_state",
                "포지션",
                "position_role",
            ),
            (
                "등이 계속 아파서 의자에 기대기도 힘들어",
                "contextual_back_state",
                "자세",
                "body_back",
            ),
            (
                "형광등이 계속 깜빡거려서 집중이 안 돼",
                "contextual_light_object_state",
                "조명",
                "light_object",
            ),
            (
                "친구가 농담이라면서 선 넘는 말 해서 정떨어졌어",
                "contextual_boundary_line_state",
                "무례",
                "social_boundary",
            ),
            (
                "그림 선이 계속 삐뚤게 그어져서 거슬려",
                "contextual_visual_line_state",
                "라인",
                "visual_line",
            ),
            (
                "생일 선물로 기프티콘 받았는데 센스 좋아서 기분 좋았어",
                "contextual_gift_state",
                "센스",
                "gift",
            ),
        )

        for text, expected_detail, expected_fragment, expected_sense in cases:
            with self.subTest(text=text):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                    state=ConversationState(user_id="current-local-window-frame-test"),
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(expected_fragment, draft["draft_reply"])
                self.assertIn(expected_sense, str(draft["high_context_trace"]))

    def test_current_word_sense_late_fallback_does_not_override_specific_frames(self) -> None:
        cases = (
            (
                "눈을 떠보니 마법이 존재하는 이세계야. 너는 어떤 클래스(직업)를 선택할 거야?",
                "contextual_gaze_social_state",
                "기록자",
            ),
            (
                "해야 할 일이 산더미인데 자꾸 미루게 되는 나를 당장 일하게 만들 팩폭 명언 하나 해줘.",
                "contextual_work_task_state",
                "5분만 해",
            ),
        )

        for text, blocked_detail, expected_fragment in cases:
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
                    state=ConversationState(user_id="current-word-sense-no-override-test"),
                )

                self.assertNotEqual(draft["draft_frame_detail"], blocked_detail)
                self.assertIn(expected_fragment, draft["draft_reply"])

    def test_word_sense_context_covers_daily_ambiguous_followups(self) -> None:
        cases = (
            (
                ("속상한 일이 있어서 속마음이 복잡하고 계속 서운해",),
                "속 진짜 답 없다",
                "inner_emotion",
                {"emotion", "inner_state"},
            ),
            (
                ("앱 속도가 느리고 로딩이 계속 버벅거려서 답답해",),
                "속 진짜 답 없다",
                "speed",
                {"speed", "performance"},
            ),
            (
                ("카톡 말투랑 그 한마디가 계속 신경 쓰여",),
                "말 진짜 답 없다",
                "speech",
                {"speech", "conversation"},
            ),
            (
                ("월말 마감이랑 연말 정산 때문에 정신없어",),
                "말 진짜 답 없다",
                "time_end",
                {"time", "period"},
            ),
            (
                ("손목이랑 손가락이 아파서 마우스 잡기 힘들어",),
                "손 진짜 답 없다",
                "body_hand",
                {"body", "hand"},
            ),
            (
                ("진상 손님 응대하다가 알바 멘탈이 나갔어",),
                "손 진짜 답 없다",
                "customer_service",
                {"customer", "service"},
            ),
            (
                ("환불도 안 되고 수수료까지 손해 봐서 빡침",),
                "손 진짜 답 없다",
                "loss",
                {"loss", "money"},
            ),
            (
                ("구두 신고 오래 걸었더니 발가락이 너무 아파",),
                "발 진짜 답 없다",
                "body_foot",
                {"body", "foot"},
            ),
            (
                ("내일 발표랑 피피티 때문에 긴장돼",),
                "발 진짜 답 없다",
                "presentation",
                {"presentation", "nervous"},
            ),
            (
                ("회사 업무랑 출근 때문에 요즘 너무 힘들어",),
                "일 진짜 답 없다",
                "work_task",
                {"work", "task"},
            ),
            (
                ("오늘 일 너무 많아서 야근각이야",),
                "일 진짜 답 없다",
                "work_task",
                {"work", "task"},
            ),
            (
                ("무슨 일 있었는지 상황이 꼬였어",),
                "일 진짜 답 없다",
                "event_happening",
                {"event", "situation"},
            ),
            (
                ("오늘 하루종일 쉬고 싶고 내일 아침도 천천히 보내고 싶어",),
                "일 진짜 답 없다",
                "day_time",
                {"time", "day"},
            ),
            (
                ("승마장에서 말 타다가 말발굽 소리에 놀랐어",),
                "말 진짜 답 없다",
                "horse",
                {"animal", "horse"},
            ),
            (
                ("차 막히고 주차도 안 돼서 운전이 너무 피곤해",),
                "차 진짜 답 없다",
                "car_vehicle",
                {"vehicle", "car"},
            ),
            (
                ("따뜻한 녹차랑 홍차 마시는 게 요즘 낙이야",),
                "차 진짜 답 없다",
                "tea_drink",
                {"drink", "tea"},
            ),
            (
                ("온도차랑 수준차가 너무 커서 비교하게 돼",),
                "차 진짜 답 없다",
                "difference_gap",
                {"difference", "gap"},
            ),
            (
                ("밤마다 새벽까지 잠이 안 와서 불면이 심해",),
                "밤 진짜 답 없다",
                "night_time",
                {"night", "sleep"},
            ),
            (
                ("군밤이랑 밤라떼 먹어봤는데 밤맛이 좀 애매해",),
                "밤 진짜 답 없다",
                "chestnut_food",
                {"food", "chestnut"},
            ),
            (
                ("친구한테 미안해서 사과해야 하는데 화해 타이밍을 놓쳤어",),
                "사과 진짜 답 없다",
                "apology",
                {"apology", "repair"},
            ),
            (
                ("사과주스랑 사과파이 먹었는데 과일 사과가 푸석했어",),
                "사과 진짜 답 없다",
                "apple",
                {"food", "apple"},
            ),
            (
                ("수학 숙제 문제 풀이가 안 돼서 정답을 못 찾겠어",),
                "풀 진짜 답 없다",
                "solve",
                {"solve", "problem"},
            ),
            (
                ("스트레스 풀고 기분 풀려고 나갔는데 더 피곤해졌어",),
                "풀 진짜 답 없다",
                "relax",
                {"relax", "emotion"},
            ),
            (
                ("딱풀로 종이 붙이는데 접착이 안 돼서 공예 망함",),
                "풀 진짜 답 없다",
                "glue",
                {"glue", "craft"},
            ),
            (
                ("풀밭이랑 잔디 관리하다가 잡초가 너무 많이 났어",),
                "풀 진짜 답 없다",
                "grass",
                {"nature", "grass"},
            ),
            (
                ("친구랑 오래 봐서 정이 들고 애착이 생겼어",),
                "정 진짜 답 없다",
                "affection_attachment",
                {"affection", "attachment"},
            ),
            (
                ("방 정리랑 파일 정리하다가 하루가 다 갔어",),
                "정 진짜 답 없다",
                "organize_cleanup",
                {"organize", "cleanup"},
            ),
            (
                ("퀴즈 정답이랑 해답을 보고도 이해가 안 됐어",),
                "정 진짜 답 없다",
                "correct_answer",
                {"answer", "correct"},
            ),
            (
                ("이번 패치 후반 벨류가 선 넘어서 밸런스가 이상해졌어",),
                "선 진짜 답 없다",
                "game_balance_boundary",
                {"game", "balance", "overpowered"},
            ),
            (
                ("그 말은 선 넘는 무례한 말이라 경계선을 지켜야 해",),
                "선 진짜 답 없다",
                "social_boundary",
                {"boundary", "line_crossing"},
            ),
            (
                ("그림 선 긋다가 직선이랑 곡선이 다 삐뚤어졌어",),
                "선 진짜 답 없다",
                "visual_line",
                {"visual", "line"},
            ),
            (
                ("생일 선물로 기프티콘을 받았는데 뭘 사줄지 고민돼",),
                "선 진짜 답 없다",
                "gift",
                {"gift", "present"},
            ),
            (
                ("두 배로 힘들고 세 배로 피곤해",),
                "배 진짜 답 없다",
                "numeric_multiple",
                {"number", "multiple"},
            ),
            (
                ("믿었던 친구한테 배신당한 느낌이야",),
                "배 진짜 답 없다",
                "betrayal",
                {"betrayal", "hurt"},
            ),
            (
                ("그 배우 연기 진짜 좋더라",),
                "배 진짜 답 없다",
                "actor_entertainment",
                {"actor", "acting"},
            ),
            (
                ("요즘 코딩 배우고 있는데 어렵다",),
                "배 진짜 답 없다",
                "learning_study",
                {"learning", "study"},
            ),
            (
                ("눈물 나고 울컥했어",),
                "눈 진짜 답 없다",
                "tears_emotion",
                {"emotion", "tears"},
            ),
            (
                ("눈높이랑 기대치가 너무 높아졌어",),
                "눈 진짜 답 없다",
                "standard_perspective",
                {"standard", "expectation"},
            ),
            (
                ("길 찾다가 골목에서 헤맸어",),
                "길 진짜 답 없다",
                "road_navigation",
                {"road", "navigation"},
            ),
            (
                ("말이 너무 길어져서 미안",),
                "길 진짜 답 없다",
                "length_long",
                {"length", "long"},
            ),
            (
                ("이 일을 해결할 길이 안 보인다",),
                "길 진짜 답 없다",
                "method_solution",
                {"method", "solution"},
            ),
            (
                ("문제 문항이 너무 어려워",),
                "문 진짜 답 없다",
                "problem_question",
                {"problem", "question"},
            ),
            (
                ("방 불 끄고 누웠어",),
                "불 진짜 답 없다",
                "light_lamp",
                {"light", "lighting"},
            ),
            (
                ("불안해서 잠이 안 와",),
                "불 진짜 답 없다",
                "anxiety_worry",
                {"anxiety", "worry"},
            ),
            (
                ("모닥불 피워놓고 멍 때렸어",),
                "불 진짜 답 없다",
                "fire_danger",
                {"fire", "danger"},
            ),
        )

        for recent_texts, current_text, expected_sense, expected_tags in cases:
            with self.subTest(expected_sense=expected_sense):
                context = build_word_sense_context_from_texts(recent_texts)
                signals = _text_signals(current_text, sense_context=context)
                senses = {sense.sense for sense in resolve_word_senses(signals, context=context)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_sense, senses)
                self.assertTrue(expected_tags <= inferred_tags)

    def test_black_draft_uses_recent_context_for_ambiguous_word_frame(self) -> None:
        cases = (
            (
                "앞머리 망해서 모자 쓰고 미용실 다시 갈지 고민 중",
                "contextual_hair_style_state",
                "머리 스타일",
            ),
            (
                "아이디어가 안 나와서 문제 해결 방향을 계속 고민 중",
                "contextual_thinking_state",
                "생각 쪽 머리",
            ),
            (
                "아까부터 두통이 있고 컨디션이 영 별로야",
                "contextual_headache_state",
                "머리 쪽",
            ),
        )

        for recent_text, expected_detail, expected_fragment in cases:
            with self.subTest(expected_detail=expected_detail):
                state = ConversationState(
                    user_id="context-word-sense-test",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content="머리 진짜 답 없다",
                        normalized="머리 진짜 답 없다",
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                    state=state,
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(expected_fragment, draft["draft_reply"])

    def test_black_draft_uses_expanded_recent_context_for_ambiguous_word_frame(self) -> None:
        cases = (
            (
                "속상한 일이 있어서 속마음이 복잡하고 계속 서운해",
                "속 진짜 답 없다",
                "contextual_inner_emotion_state",
                "속마음",
            ),
            (
                "앱 속도가 느리고 로딩이 계속 버벅거려서 답답해",
                "속 진짜 답 없다",
                "contextual_speed_state",
                "속도",
            ),
            (
                "카톡 말투랑 그 한마디가 계속 신경 쓰여",
                "말 진짜 답 없다",
                "contextual_speech_state",
                "말투",
            ),
            (
                "월말 마감이랑 연말 정산 때문에 정신없어",
                "말 진짜 답 없다",
                "contextual_time_end_state",
                "월말",
            ),
            (
                "진상 손님 응대하다가 알바 멘탈이 나갔어",
                "손 진짜 답 없다",
                "contextual_customer_service_state",
                "손님 응대",
            ),
            (
                "내일 발표랑 피피티 때문에 긴장돼",
                "발 진짜 답 없다",
                "contextual_presentation_state",
                "발표",
            ),
            (
                "회사 업무랑 출근 때문에 요즘 너무 힘들어",
                "일 진짜 답 없다",
                "contextual_work_task_state",
                "업무",
            ),
            (
                "차 막히고 주차도 안 돼서 운전이 너무 피곤해",
                "차 진짜 답 없다",
                "contextual_car_state",
                "운전",
            ),
            (
                "밤마다 새벽까지 잠이 안 와서 불면이 심해",
                "밤 진짜 답 없다",
                "contextual_night_state",
                "새벽",
            ),
            (
                "친구한테 미안해서 사과해야 하는데 화해 타이밍을 놓쳤어",
                "사과 진짜 답 없다",
                "contextual_apology_state",
                "화해",
            ),
            (
                "수학 숙제 문제 풀이가 안 돼서 정답을 못 찾겠어",
                "풀 진짜 답 없다",
                "contextual_solve_state",
                "문제",
            ),
        )

        for recent_text, current_text, expected_detail, expected_fragment in cases:
            with self.subTest(expected_detail=expected_detail):
                state = ConversationState(
                    user_id="expanded-context-word-sense-test",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=current_text,
                        normalized=current_text,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                    state=state,
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(expected_fragment, draft["draft_reply"])

    def test_black_draft_uses_recent_context_for_deictic_reference_frame(self) -> None:
        cases = (
            (
                "오늘 점심 마라탕이랑 돈까스 중에 고민 중",
                "그거 뭐가 나을까?",
                "contextual_reference_food",
                "먹는 얘기",
            ),
            (
                "친구가 카톡 말투가 너무 날카로워서 서운했어",
                "그 사람 좀 이상하지?",
                "contextual_reference_social",
                "그 사람",
            ),
            (
                "미용실 새로 생긴 데 예약할까 고민 중",
                "거기 괜찮을까?",
                "contextual_reference_place",
                "거기",
            ),
            (
                "내일 발표 피피티가 아직 반도 안 끝났어",
                "아까 그거 진짜 답 없다",
                "contextual_reference_task",
                "그 일",
            ),
            (
                "이번 웹툰 캐릭터 서사가 너무 매콤하더라",
                "그거 왜 이렇게 계속 생각나지?",
                "contextual_reference_media",
                "캐릭터",
            ),
            (
                "요즘 아무 이유 없이 우울하고 무기력해",
                "그게 좀 오래가네",
                "contextual_reference_emotion",
                "그 기분",
            ),
            (
                "쇼핑 앱 장바구니에 물건을 너무 많이 담아놨어",
                "이거 사도 되나?",
                "contextual_reference_item",
                "그 물건",
            ),
            (
                "어제 잠을 못 자서 컨디션이 너무 안 좋아",
                "그게 계속 힘드네",
                "contextual_reference_body",
                "컨디션",
            ),
        )

        for recent_text, current_text, expected_detail, expected_fragment in cases:
            with self.subTest(expected_detail=expected_detail):
                state = ConversationState(
                    user_id="context-reference-test",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=current_text,
                        normalized=current_text,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question="?" in current_text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                    state=state,
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(expected_fragment, draft["draft_reply"])

    def test_black_draft_uses_recent_context_for_choice_reference_frame(self) -> None:
        cases = (
            (
                "점심은 마라탕 vs 돈까스 중에 뭐가 나을까?",
                "난 후자가 나을 듯",
                "contextual_choice_second",
                "돈까스",
            ),
            (
                "치킨이랑 피자 중 하나만 먹어야 한다면?",
                "첫 번째로 갈래",
                "contextual_choice_first",
                "치킨",
            ),
            (
                "카톡으로 싸우기 vs 전화로 바로 풀기",
                "둘 다 별로인데",
                "contextual_choice_neither",
                "둘 다 별로",
            ),
            (
                "여름휴가 vs 겨울휴가 중에 하나만 고른다면?",
                "다른 거 없나",
                "contextual_choice_other",
                "다른 쪽",
            ),
        )

        for recent_text, current_text, expected_detail, expected_fragment in cases:
            with self.subTest(expected_detail=expected_detail):
                state = ConversationState(
                    user_id="context-choice-test",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=current_text,
                        normalized=current_text,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question="?" in current_text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                    state=state,
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(expected_fragment, draft["draft_reply"])

    def test_black_draft_uses_recent_context_for_discourse_flow_frame(self) -> None:
        cases = (
            (
                "친구한테 먼저 사과할까 고민 중이야",
                "그래도 먼저 연락하는 게 맞겠지",
                "contextual_discourse_concession",
                "부담되는 지점",
            ),
            (
                "내일 발표 자료가 아직 반도 안 끝났어",
                "그럼 일단 목차부터 정리해야겠다",
                "contextual_discourse_next_step",
                "작게 한 단계",
            ),
            (
                "오늘 저녁 치킨 먹을까 했어",
                "근데 너무 늦어서 좀 부담된다",
                "contextual_discourse_contrast",
                "걸리는 포인트",
            ),
            (
                "주말에 영화 보러 갈까 생각 중",
                "아니면 그냥 집에서 넷플릭스 볼까",
                "contextual_discourse_alternative",
                "방향을 바꾸",
            ),
            (
                "상사가 퇴근 직전에 일 던져서 짜증났어",
                "그니까 진짜 선 넘었지",
                "contextual_discourse_agreement",
                "그 결",
            ),
        )

        for recent_text, current_text, expected_detail, expected_fragment in cases:
            with self.subTest(expected_detail=expected_detail):
                state = ConversationState(
                    user_id="context-discourse-test",
                    recent_turns=[
                        TurnRecord(
                            user_text=recent_text,
                            bot_text="",
                            action=ActionType.SMALL_TALK,
                            decision_reason="test",
                        )
                    ],
                )
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=current_text,
                        normalized=current_text,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question="?" in current_text,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                    state=state,
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(expected_fragment, draft["draft_reply"])

    def test_black_draft_uses_current_turn_discourse_flow_frame(self) -> None:
        cases = (
            (
                "그 카페 좋긴 한데 사람이 너무 많아서 기 빨려",
                "current_turn_discourse_concession",
                "좋긴 한데",
                "그 장소",
            ),
            (
                "치킨도 괜찮긴 한데 지금 먹기엔 좀 부담돼",
                "current_turn_discourse_concession",
                "좋긴 한데",
                "메뉴",
            ),
            (
                "오늘 저녁 치킨 생각했는데 근데 너무 늦어서 부담된다",
                "current_turn_discourse_contrast",
                "근데",
                "메뉴",
            ),
            (
                "그 메뉴 말고 차라리 국밥 먹는 게 덜 피곤해",
                "current_turn_discourse_alternative",
                "차라리",
                "메뉴",
            ),
        )

        for text, expected_detail, first_fragment, second_fragment in cases:
            with self.subTest(expected_detail=expected_detail):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_GENERIC,
                        sentiment="neutral",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(first_fragment, draft["draft_reply"])
                self.assertIn(second_fragment, draft["draft_reply"])

    def test_black_draft_handles_mixed_topic_current_turn_frame(self) -> None:
        cases = (
            (
                "배고픈데 잠도 오고 내일 발표도 걱정돼",
                "mixed_hunger_sleep_task",
                "지금 흐름은",
                "조금 먹고",
            ),
            (
                "친구 답장도 늦고 일도 밀려서 멘탈이 별로야",
                "mixed_emotion_task",
                "지금 흐름은",
                "한 조각",
            ),
            (
                "비도 오고 몸도 축 처져서 저녁 메뉴 고르기도 귀찮아",
                "mixed_body_food",
                "지금 흐름은",
                "덜 무리",
            ),
            (
                "웹툰 완결 보고 마음이 먹먹한데 내일 출근 생각하니 더 싫다",
                "mixed_media_emotion",
                "지금 흐름은",
                "여운",
            ),
        )

        for text, expected_detail, first_fragment, second_fragment in cases:
            with self.subTest(expected_detail=expected_detail):
                draft = build_black_draft_utterance(
                    features=MessageFeatures(
                        content=text,
                        normalized=text,
                        intent=Intent.SMALLTALK_FEELING,
                        sentiment="negative",
                        is_question=False,
                    ),
                    response_plan=ResponsePlan(
                        action=ActionType.SMALL_TALK,
                        stance="continue_light",
                        anchor="",
                        must_include=[],
                        followup_policy="no_followup",
                    ),
                    phrasing_plan=PhrasingPlan(),
                )

                self.assertEqual(draft["rewrite_mode"], "draft_direct")
                self.assertEqual(draft["draft_frame_detail"], expected_detail)
                self.assertIn(first_fragment, draft["draft_reply"])
                self.assertIn(second_fragment, draft["draft_reply"])

    def test_word_sense_bank_disambiguates_bae_and_noon(self) -> None:
        stomach_signals = _text_signals("배가 아프다")
        ship_signals = _text_signals("배를 탔다")
        pear_signals = _text_signals("배를 깎아 먹었다")
        eye_signals = _text_signals("눈이 아프다")
        snow_signals = _text_signals("눈이 온다")
        sense_pairs = {
            text: {(sense.word, sense.sense) for sense in resolve_word_senses(signals)}
            for text, signals in (
                ("stomach", stomach_signals),
                ("ship", ship_signals),
                ("pear", pear_signals),
                ("eye", eye_signals),
                ("snow", snow_signals),
            )
        }
        stomach = rank_semantic_word(
            stomach_signals,
            desired_tags=("short_word", "body", "pain", "stomach", "health"),
            default="기본 반응",
        )
        ship = rank_semantic_word(
            ship_signals,
            desired_tags=("short_word", "vehicle", "ship"),
            default="기본 반응",
        )
        pear = rank_semantic_word(
            pear_signals,
            desired_tags=("short_word", "food", "fruit", "pear"),
            default="기본 반응",
        )
        eye = rank_semantic_word(
            eye_signals,
            desired_tags=("short_word", "body", "eye", "health"),
            default="기본 반응",
        )
        snow = rank_semantic_word(
            snow_signals,
            desired_tags=("short_word", "weather", "snow"),
            default="기본 반응",
        )

        self.assertIn(("배", "body_stomach"), sense_pairs["stomach"])
        self.assertIn(("배", "ship"), sense_pairs["ship"])
        self.assertIn(("배", "pear"), sense_pairs["pear"])
        self.assertIn(("눈", "eye"), sense_pairs["eye"])
        self.assertIn(("눈", "snow"), sense_pairs["snow"])
        self.assertEqual(stomach.value, "복통")
        self.assertEqual(ship.value, "배")
        self.assertIn("ship", ship.matched_tags)
        self.assertEqual(pear.value, "배")
        self.assertIn("pear", pear.matched_tags)
        self.assertEqual(eye.value, "눈")
        self.assertIn("eye", eye.matched_tags)
        self.assertEqual(snow.value, "눈옴")
        self.assertIn("snow", snow.matched_tags)

    def test_word_sense_bank_keeps_first_sight_apart_from_first_snow(self) -> None:
        romance_signals = _text_signals("첫눈에 반한다는 걸 믿어?")
        attraction_signals = _text_signals("첫눈에 확 끌리는 사람 만나본 적 있어?")
        snow_signals = _text_signals("첫눈 오는 날엔 괜히 설레더라.")

        romance_senses = {(sense.word, sense.sense) for sense in resolve_word_senses(romance_signals)}
        attraction_senses = {(sense.word, sense.sense) for sense in resolve_word_senses(attraction_signals)}
        snow_senses = {(sense.word, sense.sense) for sense in resolve_word_senses(snow_signals)}

        romance = rank_semantic_word(
            romance_signals,
            desired_tags=("short_word", "ambiguous", "relationship", "romance", "first_sight"),
            default="기본 반응",
        )
        attraction = rank_semantic_word(
            attraction_signals,
            desired_tags=("short_word", "ambiguous", "relationship", "romance", "first_sight"),
            default="기본 반응",
        )
        snow = rank_semantic_word(
            snow_signals,
            desired_tags=("short_word", "ambiguous", "weather", "snow", "winter"),
            default="기본 반응",
        )

        self.assertIn(("눈", "first_sight"), romance_senses)
        self.assertIn(("눈", "first_sight"), attraction_senses)
        self.assertIn(("눈", "snow"), snow_senses)
        self.assertEqual(romance.value, "첫눈 호감")
        self.assertIn("first_sight", romance.matched_tags)
        self.assertEqual(attraction.value, "첫눈 호감")
        self.assertIn("first_sight", attraction.matched_tags)
        self.assertEqual(snow.value, "눈옴")
        self.assertIn("snow", snow.matched_tags)

    def test_word_sense_bank_covers_high_context_daily_polysemy(self) -> None:
        cases = (
            (
                "다리가 아파서 오늘 계단은 못 오르겠어.",
                ("다리", "body_leg"),
                ("short_word", "ambiguous", "body", "leg", "pain"),
                "다리",
                "leg",
            ),
            (
                "한강 다리 위에서 찬바람 맞으니까 정신이 좀 들더라.",
                ("다리", "bridge_structure"),
                ("short_word", "ambiguous", "structure", "bridge", "place"),
                "다리",
                "bridge",
            ),
            (
                "내 작은 바람은 오늘 퇴근하고 아무한테도 연락 안 오는 거야.",
                ("바람", "wish_hope"),
                ("short_word", "ambiguous", "wish", "hope", "inner_state"),
                "바람",
                "wish",
            ),
            (
                "바람피운 건 진짜 신뢰를 박살내는 일이지.",
                ("바람", "relationship_cheating"),
                ("short_word", "ambiguous", "relationship", "cheating", "betrayal"),
                "바람피우다",
                "cheating",
            ),
            (
                "방문 닫고 조용히 혼자 있고 싶어.",
                ("문", "door_object"),
                ("short_word", "ambiguous", "object", "door", "home"),
                "문",
                "door",
            ),
            (
                "이 문장 하나가 너무 어색해서 계속 고치고 있어.",
                ("문", "sentence_language"),
                ("short_word", "ambiguous", "language", "sentence", "writing"),
                "문장",
                "sentence",
            ),
            (
                "퇴근길에 장보고 왔더니 두 손이 다 무겁다.",
                ("장", "shopping_market"),
                ("short_word", "ambiguous", "shopping", "market"),
                "장",
                "shopping",
            ),
            (
                "장이 안 좋아서 오늘은 매운 거 못 먹겠어.",
                ("장", "body_bowel"),
                ("short_word", "ambiguous", "body", "bowel", "stomach"),
                "장",
                "bowel",
            ),
            (
                "책 한 장 넘겼는데 벌써 졸리네.",
                ("장", "book_page_chapter"),
                ("short_word", "ambiguous", "book", "chapter", "page"),
                "장",
                "book",
            ),
            (
                "일이 너무 많이 쌓여서 오늘은 야근각이야.",
                ("일", "work_task"),
                ("short_word", "ambiguous", "work", "task", "office"),
                "일",
                "work",
            ),
            (
                "바람피운 건 진짜 신뢰를 박살내는 일이지.",
                ("일", "event_happening"),
                ("short_word", "ambiguous", "event", "incident", "situation"),
                "일",
                "event",
            ),
            (
                "단감 먹었더니 가을 온 느낌 나네.",
                ("감", "persimmon_fruit"),
                ("short_word", "ambiguous", "food", "fruit", "persimmon"),
                "감",
                "persimmon",
            ),
            (
                "이건 왠지 감이 와, 오늘은 되는 날이다.",
                ("감", "intuition_sense"),
                ("short_word", "ambiguous", "sense", "intuition", "feeling"),
                "감",
                "intuition",
            ),
            (
                "굴전 먹고 싶다, 비 오는 날엔 해산물이 당겨.",
                ("굴", "oyster_food"),
                ("short_word", "ambiguous", "food", "oyster", "seafood"),
                "굴",
                "oyster",
            ),
            (
                "동굴 입구가 너무 어두워서 살짝 쫄렸어.",
                ("굴", "cave_place"),
                ("short_word", "ambiguous", "place", "cave", "tunnel"),
                "굴",
                "cave",
            ),
            (
                "라면에 대파 송송 넣으면 갑자기 급이 달라져.",
                ("파", "green_onion_food"),
                ("short_word", "ambiguous", "food", "green_onion", "ingredient"),
                "파",
                "green_onion",
            ),
            (
                "나는 탕수육 찍먹파라 소스 부으면 마음이 아파.",
                ("파", "preference_team"),
                ("short_word", "ambiguous", "faction", "team", "preference"),
                "파",
                "faction",
            ),
            (
                "김밥에 김가루까지 뿌리면 그건 김 파티지.",
                ("김", "seaweed_food"),
                ("short_word", "ambiguous", "food", "seaweed", "ingredient"),
                "김",
                "seaweed",
            ),
            (
                "안경 김 서림 때문에 앞이 하나도 안 보여.",
                ("김", "steam_fog"),
                ("short_word", "ambiguous", "steam", "fog", "breath"),
                "김",
                "steam",
            ),
            (
                "등이 아파서 오늘은 의자에 기대기도 힘들다.",
                ("등", "body_back"),
                ("short_word", "ambiguous", "body", "back", "pain"),
                "등",
                "back",
            ),
            (
                "전등 꺼놓고 누우니까 바로 잠 올 것 같아.",
                ("등", "light_object"),
                ("short_word", "ambiguous", "object", "light", "lamp"),
                "등",
                "light",
            ),
            (
                "라면 김밥 만두 등등 다 먹고 싶어.",
                ("등", "etc_list"),
                ("short_word", "ambiguous", "etc", "list", "category"),
                "등",
                "etc",
            ),
            (
                "입맛이 없어서 오늘은 죽만 먹을래.",
                ("입", "mouth_body"),
                ("short_word", "ambiguous", "body", "mouth", "taste"),
                "입",
                "mouth",
            ),
            (
                "출입구 앞에서 기다릴게, 못 찾으면 전화해.",
                ("입", "entrance_place"),
                ("short_word", "ambiguous", "place", "entrance", "meeting_place"),
                "입구",
                "entrance",
            ),
            (
                "내 입장에서는 그 말이 조금 서운했어.",
                ("입", "stance_perspective"),
                ("short_word", "ambiguous", "stance", "perspective", "opinion"),
                "입장",
                "stance",
            ),
            (
                "목이 칼칼해서 오늘은 따뜻한 차 마셔야겠다.",
                ("목", "throat_body"),
                ("short_word", "ambiguous", "body", "throat", "neck"),
                "목",
                "throat",
            ),
            (
                "목소리가 좋아서 그 사람 말이 더 잘 들어와.",
                ("목", "voice_sound"),
                ("short_word", "ambiguous", "voice", "sound", "tone"),
                "목소리",
                "voice",
            ),
            (
                "오늘 목요일인 줄 알았는데 아직 화요일이라 멘탈 나감.",
                ("목", "weekday_thursday"),
                ("short_word", "ambiguous", "time", "weekday", "thursday"),
                "목요일",
                "thursday",
            ),
            (
                "올해 목표를 너무 크게 잡아서 벌써 숨 막힌다.",
                ("목", "goal_plan"),
                ("short_word", "ambiguous", "goal", "plan", "growth"),
                "목표",
                "goal",
            ),
            (
                "코막힘 때문에 커피 향도 하나도 안 나.",
                ("코", "nose_body"),
                ("short_word", "ambiguous", "body", "nose", "breathing"),
                "코",
                "nose",
            ),
            (
                "소스 코드 한 줄 때문에 두 시간째 디버깅 중이야.",
                ("코", "code_tech"),
                ("short_word", "ambiguous", "tech", "code", "programming"),
                "코드",
                "code",
            ),
        )

        for text, expected_sense, desired_tags, expected_word, expected_tag in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = {(sense.word, sense.sense) for sense in resolve_word_senses(signals)}
                ranked = rank_semantic_word(
                    signals,
                    desired_tags=desired_tags,
                    default="기본 반응",
                )
                self.assertIn(expected_sense, senses)
                self.assertEqual(ranked.value, expected_word)
                self.assertIn(expected_tag, ranked.matched_tags)

    def test_word_sense_bank_covers_more_daily_polysemy_words(self) -> None:
        cases = (
            ("감기약 먹고 누웠는데 또 복용 시간이 헷갈려.", ("약", "medicine_health"), {"medicine", "drug", "health"}),
            ("약속시간 늦을 것 같아서 연락을 먼저 해야겠어.", ("약", "appointment_promise"), {"appointment", "promise"}),
            ("멘탈이 약해서 작은 말에도 흔들리는 것 같아.", ("약", "weak_state"), {"weak", "fragile"}),
            ("병원 가야 할 정도로 목이 아픈지 모르겠어.", ("병", "illness_health"), {"illness", "health", "hospital"}),
            ("물병 뚜껑이 안 열려서 손목까지 아파.", ("병", "bottle_container"), {"bottle", "container"}),
            ("잠이 안 와서 새벽 내내 폰만 봤어.", ("잠", "sleep_state"), {"sleep", "rest", "night"}),
            ("화면 잠금 비밀번호를 또 까먹어서 멘붕이야.", ("잠", "lock_security"), {"lock", "security", "password"}),
            ("팔이 아파서 마우스 잡는 것도 싫어.", ("팔", "body_arm"), {"body", "arm", "pain"}),
            ("중고로 팔려고 올렸는데 아무도 연락이 없어.", ("팔", "sell_trade"), {"sell", "trade"}),
            ("8월인지 팔월인지 쓰는 방식만 헷갈려.", ("팔", "number_eight"), {"number", "eight"}),
            ("다이어트 중인데 뱃살이 안 빠져서 짜증나.", ("살", "body_fat"), {"body", "fat", "diet"}),
            ("이렇게라도 살아남으려면 오늘은 쉬어야 해.", ("살", "live_survive"), {"life", "survival", "living"}),
            ("스무살 때보다 지금이 더 정신이 없는 것 같아.", ("살", "age_years"), {"age", "years"}),
            ("축구공 차다가 발목이 삐끗했어.", ("공", "ball_sports"), {"sports", "ball", "play"}),
            ("점수가 0점이라 공점대 농담도 못 하겠어.", ("공", "zero_number"), {"number", "zero", "score"}),
            ("공용 와이파이가 너무 느려서 회의가 끊겼어.", ("공", "public_shared"), {"public", "shared", "official"}),
            ("게임 한판만 더 하자고 했다가 새벽 됐어.", ("판", "game_round"), {"game", "round", "match"}),
            ("간판 글씨가 너무 작아서 지나칠 뻔했어.", ("판", "plate_board"), {"plate", "board"}),
            ("괜히 말 꺼냈다가 판이 커져서 난장판 됐어.", ("판", "situation_scene"), {"situation", "scene", "mess"}),
            ("기차표 예매를 까먹어서 일정이 꼬였어.", ("표", "ticket_pass"), {"ticket", "transport", "reservation"}),
            ("시간표를 표로 정리하니까 그나마 보이네.", ("표", "chart_table"), {"table", "chart", "data"}),
            ("투표 한표 차이로 갈릴 수도 있다니까 묘하네.", ("표", "vote_ballot"), {"vote", "election", "ballot"}),
        )

        for text, expected_sense, expected_tags in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = {(sense.word, sense.sense): sense for sense in resolve_word_senses(signals)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_sense, senses)
                self.assertTrue(expected_tags <= inferred_tags)
                self.assertTrue(senses[expected_sense].matched_cues)

    def test_word_sense_bank_covers_even_more_daily_polysemy_words(self) -> None:
        cases = (
            ("물 한잔 마시고 정신 차려야겠다.", ("물", "water_drink"), {"water", "hydration"}),
            ("검은 티셔츠가 세탁하면서 물빠짐이 생겼어.", ("물", "dye_color_bleed"), {"dye", "color_bleed", "stain"}),
            ("몸에 열이 나서 체온을 재봤어.", ("열", "fever_heat"), {"fever", "body_heat", "temperature"}),
            ("열 번 말했는데도 또 까먹었어.", ("열", "number_ten"), {"number", "ten"}),
            ("파일을 열어 봤는데 내용이 비어 있어.", ("열", "open_action"), {"open_action", "access"}),
            ("30초만 기다리면 알람 울려.", ("초", "second_time_unit"), {"second_unit", "timer"}),
            ("초보 운전이라 골목길만 들어가도 긴장돼.", ("초", "beginner_novice"), {"beginner", "novice"}),
            ("양초 켜놓으니까 방 분위기가 좀 풀렸어.", ("초", "candle_light"), {"candle", "wick", "flame"}),
            ("내 방 정리하다가 예전 사진을 찾았어.", ("방", "room_space"), {"room", "indoor_space"}),
            ("오늘 방송 켜면 첫 멘트 뭐 하지?", ("방", "broadcast_stream"), {"broadcast", "streaming"}),
            ("해결 방안이 안 보여서 답답해.", ("방", "method_solution"), {"method", "solution"}),
            ("팔에 검은 점 하나가 생겨서 신경 쓰여.", ("점", "dot_mark"), {"dot", "mark", "spot"}),
            ("시험 점수가 생각보다 낮아서 멘탈 나갔어.", ("점", "score_grade"), {"score", "grade"}),
            ("편의점 들렀다가 우산을 두고 왔어.", ("점", "store_shop"), {"store", "shop"}),
            ("이 자세가 제일 편해서 계속 이렇게 앉아 있어.", ("편", "comfort_ease"), {"comfort", "ease"}),
            ("다음 편 올라오면 바로 볼 거야.", ("편", "episode_series"), {"episode", "series"}),
            ("네가 내 편 들어줘서 좀 살 것 같아.", ("편", "ally_side"), {"ally", "side"}),
            ("20대 초반엔 밤새도 괜찮았는데 지금은 아니야.", ("대", "age_group"), {"age_group", "generation"}),
            ("노트북 한 대 더 있으면 테스트가 편할 텐데.", ("대", "unit_counter"), {"counter_unit", "vehicle_unit"}),
            ("괜히 장난치다가 꿀밤 한 대 맞았어.", ("대", "hit_blow"), {"hit", "blow"}),
            ("이번에 상 받으면 진짜 뿌듯할 것 같아.", ("상", "award_prize"), {"award", "prize"}),
            ("책상 위에 컵을 올려놨다가 쏟았어.", ("상", "table_surface"), {"table_surface", "serving_table"}),
            ("요리하다가 손에 상처가 났어.", ("상", "wound_injury"), {"wound", "injury"}),
        )

        for text, expected_sense, expected_tags in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = {(sense.word, sense.sense): sense for sense in resolve_word_senses(signals)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_sense, senses)
                self.assertTrue(expected_tags <= inferred_tags)
                self.assertTrue(senses[expected_sense].matched_cues)

    def test_word_sense_bank_covers_more_short_daily_polysemy_words(self) -> None:
        cases = (
            ("보름달이 너무 밝아서 괜히 감성 올라왔어.", ("달", "moon_night_sky"), {"moon", "night_sky"}),
            ("다음 달 일정이 벌써 꽉 찼어.", ("달", "month_calendar"), {"month", "calendar"}),
            ("커피가 너무 달아서 한입 먹고 물렸어.", ("달", "sweet_taste"), {"sweet", "taste"}),
            ("환절기 철마다 목이 칼칼해져.", ("철", "season_period"), {"season", "period"}),
            ("철제 선반 모서리에 손을 긁었어.", ("철", "metal_material"), {"metal", "material"}),
            ("나 왜 아직도 철이 없는 것 같지.", ("철", "maturity_sense"), {"maturity", "adult_sense"}),
            ("집에 가자마자 뻗고 싶어.", ("집", "home_house"), {"home", "house"}),
            ("동네 맛집 웨이팅이 너무 길어.", ("집", "restaurant_shop"), {"restaurant", "food_place"}),
            ("수학 문제집 한 권을 또 샀어.", ("집", "book_collection"), {"workbook", "book_collection", "study"}),
            ("이번 주 일정이 너무 빡빡해.", ("주", "week_time"), {"week", "schedule"}),
            ("소주 한 잔 마시고 바로 얼굴 빨개졌어.", ("주", "alcohol_liquor"), {"alcohol", "liquor"}),
            ("주식 주가가 계속 떨어져서 멘탈 나갔어.", ("주", "stock_share"), {"stock_market", "equity"}),
            ("새 폰 샀는데 설정이 너무 귀찮아.", ("새", "new_fresh"), {"new", "fresh"}),
            ("창밖에서 새소리가 들려서 깼어.", ("새", "bird_animal"), {"bird", "animal"}),
            ("천장에서 물이 새는 것 같아서 불안해.", ("새", "leak_drip"), {"leak", "drip"}),
            ("말벌이 방 안에 들어와서 심장 멎는 줄.", ("벌", "bee_insect"), {"bee", "insect"}),
            ("지각해서 벌점 받으면 진짜 억울해.", ("벌", "punishment_penalty"), {"punishment", "penalty"}),
            ("정장 한 벌 사야 하는데 너무 비싸.", ("벌", "clothing_set"), {"clothing_set", "outfit_counter"}),
            ("건강검진에서 간수치가 높게 나왔어.", ("간", "liver_health"), {"liver", "health"}),
            ("국 간이 너무 세서 물을 더 넣었어.", ("간", "seasoning_salt"), {"seasoning", "saltiness"}),
            ("팀원 간 의견 차이가 커서 회의가 길어졌어.", ("간", "interval_between"), {"interval", "between"}),
            ("국 맛이 이상해서 상한 건지 걱정돼.", ("맛", "flavor_taste"), {"flavor", "taste"}),
            ("이 게임은 손맛이 살아서 계속 하게 돼.", ("맛", "enjoyment_vibe"), {"enjoyment", "fun"}),
            ("회사 근처 맛집 추천받았어.", ("맛", "restaurant_reputation"), {"restaurant", "food_reputation"}),
        )

        for text, expected_sense, expected_tags in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = {(sense.word, sense.sense): sense for sense in resolve_word_senses(signals)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_sense, senses)
                self.assertTrue(expected_tags <= inferred_tags)
                self.assertTrue(senses[expected_sense].matched_cues)

    def test_semantic_relations_infer_priority_tags_from_word_combinations(self) -> None:
        cases = (
            (
                "방금 스마트폰을 물에 빠뜨렸고 카메라에 습기가 차서 쌀통 효과가 궁금해.",
                "device_water_damage_practical_first",
                {"relation:device_water_damage", "water_damage", "priority_practical", "suppress_meta"},
                "practical_first",
            ),
            (
                "프라이팬 기름 불에 물 부으면 왜 위험한지 논리도 궁금해.",
                "oil_fire_water_misuse_practical_first",
                {"relation:oil_fire_water_misuse", "fire_danger", "water_misuse", "priority_practical"},
                "practical_first",
            ),
            (
                "감기약을 두 번 먹은 것 같아서 괜찮은지 불안해.",
                "medicine_double_dose_practical_first",
                {"relation:medicine_double_dose", "dosage_risk", "health_check", "priority_practical"},
                "practical_first",
            ),
            (
                "몸에 열이 나고 체온이 높아서 불안해.",
                "fever_body_check_practical_first",
                {"relation:fever_body_check", "body_risk", "health_check", "priority_practical"},
                "practical_first",
            ),
            (
                "사람은 많은데 내 편이 없어서 고독해.",
                "ally_loneliness_emotion_first",
                {"relation:ally_loneliness", "safe_person_need", "priority_emotion"},
                "emotion_stabilize",
            ),
            (
                "주식으로 남들은 돈 벌었다는데 나만 뒤처진 것 같아서 조급해.",
                "stock_fomo_judgment_brake",
                {"relation:stock_fomo", "money_fomo", "risk_control", "priority_judgment"},
                "judgment",
            ),
            (
                "마감 과제 파일이 날아간 것 같아서 복구부터 해야 해.",
                "deadline_file_loss_practical_first",
                {"relation:deadline_file_loss", "file_recovery", "priority_practical", "suppress_meta"},
                "practical_first",
            ),
            (
                "타임머신처럼 인과율을 따지고 싶은데 면접 가는 길에 버스를 놓쳤어.",
                "time_machine_emergency_practical_override",
                {"relation:meta_practical_conflict", "priority_practical", "real_world_first", "suppress_meta"},
                "practical_first",
            ),
            (
                "계좌이체를 다른 사람한테 잘못 보냈는데 은행 연락이 먼저인지 불안해.",
                "wrong_transfer_practical_first",
                {"relation:wrong_transfer", "transfer_error", "bank_contact", "priority_practical"},
                "practical_first",
            ),
            (
                "단톡에서 내 말만 씹힌 것 같아서 인간관계 상처가 커.",
                "group_chat_silence_emotion_first",
                {"relation:group_chat_silence", "hurt_stabilize", "priority_emotion"},
                "emotion_stabilize",
            ),
            (
                "친구가 읽씹인지 바쁜건지 모르겠고 폰만봐. 지금 단정 보류가 맞지?",
                "read_receipt_uncertainty_hold_judgment",
                {"relation:read_receipt_uncertainty_hold", "social_uncertainty", "priority_judgment"},
                "judgment",
            ),
            (
                "친구가 내 얘기 읽씹한 건지 바쁜건지 모르겠고 폰만봐, 단정해도 돼?",
                "read_receipt_uncertainty_hold_judgment",
                {"relation:read_receipt_uncertainty_hold", "hold_judgment", "priority_judgment"},
                "judgment",
            ),
            (
                "친구가 읽씹인지 바쁜건지 모르겠어. 폰만봐서 미치겠는데 단정 보류?",
                "read_receipt_uncertainty_hold_judgment",
                {"relation:read_receipt_uncertainty_hold", "social_uncertainty", "priority_judgment"},
                "judgment",
            ),
            (
                "상대가 읽고 답 없는 것 같은데 이거 읽씹이라고 봐도 돼?",
                "read_receipt_uncertainty_hold_judgment",
                {"relation:read_receipt_uncertainty_hold", "social_uncertainty", "priority_judgment"},
                "judgment",
            ),
            (
                "내가 예민한건지 상대가 무례한건지 모르겠어. 기분 확 상했는데 싸우지 않고 선을 어떻게 말해?",
                "relationship_boundary_polite_firm",
                {"relation:relationship_boundary_polite_firm", "relationship_boundary", "priority_judgment"},
                "judgment",
            ),
            (
                "애인 카톡 말투가 갑자기 차가워졌는데 불안해. 추궁하지 말고 짧게 확인해도 돼?",
                "relationship_kakao_tone_anxiety_check",
                {"relation:relationship_kakao_tone_anxiety_check", "short_check_question", "priority_practical"},
                "practical_first",
            ),
            (
                "배달 끊어야 하는데 오늘 너무 지쳐서 아무것도 못하겠어. 시켜도 합리적이야?",
                "delivery_tired_compromise_practical",
                {"relation:delivery_tired_compromise", "energy_budget_tradeoff", "priority_practical"},
                "practical_first",
            ),
            (
                "옆집이 매일 새벽에 쿵쾅거려서 미치겠어. 쪽지 먼저야 관리사무소 신고 먼저야?",
                "neighbor_noise_record_first_practical",
                {"relation:neighbor_noise_record_first", "neighbor_noise", "priority_practical"},
                "practical_first",
            ),
            (
                "착한 거짓말이 사람 덜 상처주면 괜찮은 건지, 솔직한 팩트가 장기적으로 나은 건지 어떻게 봐?",
                "white_lie_truth_tradeoff_judgment",
                {"relation:white_lie_truth_tradeoff", "truth_tact_tradeoff", "priority_judgment"},
                "judgment",
            ),
            (
                "면접 가는 버스 놓쳤고 뇌정지 왔어. 택시 먼저야 담당자 연락 먼저야 뭐부터 해?",
                "interview_missed_bus_practical_first",
                {"relation:interview_missed_bus", "arrival_risk", "priority_practical"},
                "practical_first",
            ),
            (
                "새 프로젝트를 맡았는데 아는 게 없어서 무능해 보일까 봐 첫 단추를 모르겠어.",
                "new_project_first_step_practical",
                {"relation:new_project_first_step", "work_uncertainty", "priority_practical"},
                "practical_first",
            ),
            (
                "천장에서 물이 새는데 사진이랑 영상부터 남겨야 할지 모르겠어.",
                "home_water_leak_practical",
                {"relation:home_water_leak", "home_repair", "evidence_first", "priority_practical"},
                "practical_first",
            ),
            (
                "새 폰 설정이 귀찮고 예전 폰이 그리워서 후회돼.",
                "new_phone_adjustment",
                {"relation:new_phone_adjustment", "adjustment_cost", "priority_emotion"},
                "emotion_stabilize",
            ),
            (
                "돈 모으고 싶은데 스트레스 받을 때 편의점 충동구매가 새서 결제 마찰 장치가 필요해.",
                "impulse_spending_payment_friction",
                {"relation:impulse_spending_payment_friction", "payment_friction", "priority_practical"},
                "practical_first",
            ),
            (
                "성공 기준이 남들 보여주기인지 덜 망가지는 삶인지 조건부터 써야겠어.",
                "success_standard_values",
                {"relation:success_standard_values", "personal_values", "priority_judgment"},
                "judgment",
            ),
            (
                "단맛 커피가 맛은 좋은데 기분이랑 컨디션이 둘 다 떨어지는 느낌이야.",
                "taste_condition_dual_read",
                {"relation:taste_condition_dual_read", "flavor_condition", "priority_judgment"},
                "judgment",
            ),
            (
                "가스 냄새가 위험한지 불안해서 창문과 관리사무소가 먼저인지 모르겠어.",
                "gas_smell_emergency_practical_first",
                {"relation:gas_smell_emergency", "gas_risk", "priority_practical"},
                "practical_first",
            ),
            (
                "가스레인지 한쪽만 불이 안 붙어서 점화장치 문제인가 봐.",
                "gas_stove_ignition_issue_practical",
                {"relation:gas_stove_ignition_issue", "home_maintenance", "gas_stove_check", "priority_practical"},
                "practical_first",
            ),
            (
                "가스레인지 디자인이 예뻐서 사진 저장했는데 점화장치 후기가 별로래.",
                "appliance_design_review_judgment",
                {"relation:appliance_design_review_judgment", "home_appliance", "purchase_judgment", "priority_practical"},
                "practical_first",
            ),
            (
                "요즘 가스비 너무 올라서 보일러 켜기 무서워.",
                "heating_bill_anxiety_practical",
                {"relation:heating_bill_anxiety", "utility_bill_pressure", "heating_budget", "priority_practical"},
                "practical_first",
            ),
            (
                "기름값이 너무 올라서 주유소 갈 때마다 지갑이 아파.",
                "living_cost_pressure_practical",
                {"relation:living_cost_pressure", "cost_of_living", "budget_pressure", "priority_practical"},
                "practical_first",
            ),
            (
                "요즘 물가 너무 올라서 마트 가기 무서워.",
                "living_cost_pressure_practical",
                {"relation:living_cost_pressure", "cost_of_living", "budget_pressure", "priority_practical"},
                "practical_first",
            ),
            (
                "은행 문자 링크를 눌렀는지 계좌가 불안해서 비밀번호부터 막아야 할 것 같아.",
                "phishing_link_account_lock_practical",
                {"relation:phishing_link_account_lock", "account_security", "priority_practical"},
                "practical_first",
            ),
            (
                "지갑을 잃어버려서 자책보다 카드 정지와 분실 신고가 먼저야.",
                "lost_wallet_card_stop_practical",
                {"relation:lost_wallet_card_stop", "lost_card", "priority_practical"},
                "practical_first",
            ),
            (
                "열 번 검색했고 열이 나는 것 같아서 파일 열기보다 체온부터 재야 해.",
                "heat_polysemy_fever_first",
                {"relation:heat_polysemy_fever_first", "wordplay_disambiguation", "priority_practical"},
                "practical_first",
            ),
            (
                "단어 뜻만으로는 부족하고 방안과 방 안 같은 단어 사이 관계도 같이 봐야 해.",
                "semantic_relation_map_meta",
                {"relation:semantic_relation_map_meta", "word_relation_graph", "priority_meta"},
                "meta",
            ),
        )

        for text, expected_relation, expected_tags, expected_priority in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = resolve_word_senses(signals)
                inferred_tags = set(_input_tags(signals))
                relations = {
                    relation.name: relation
                    for relation in infer_semantic_relations(
                        signals,
                        base_tags=tuple(inferred_tags),
                        resolved_senses=senses,
                    )
                }

                self.assertIn(expected_relation, relations)
                self.assertEqual(relations[expected_relation].priority, expected_priority)
                self.assertTrue(expected_tags <= inferred_tags)
                self.assertGreater(relations[expected_relation].score, 0.0)
                self.assertGreater(relations[expected_relation].confidence, 0.0)
                self.assertTrue(relations[expected_relation].positive_evidence)

    def test_semantic_relations_do_not_fire_on_false_positive_contexts(self) -> None:
        cases = (
            ("요즘 가스비 너무 올라서 보일러 켜기 무서워.", "gas_smell_emergency_practical_first"),
            ("가스레인지 한쪽만 불이 안 붙어서 점화장치 문제인가 봐.", "gas_smell_emergency_practical_first"),
            ("기름값이 너무 올라서 주유소 갈 때마다 지갑이 아파.", "oil_fire_action_practical_first"),
            ("게임에서 사고가 나서 보험 아이템을 써야 할지 고민이야.", "car_accident_first_steps_practical"),
            ("약속 자리에서 술 한잔 마셨는데 분위기 때문에 불안했어.", "medicine_alcohol_check_practical"),
            ("회의 링크 눌렀는데 권한 없음 떠서 비밀번호를 다시 확인했어.", "phishing_link_account_lock_practical"),
            ("지갑 사정 안 좋아서 이번 시즌 굿즈는 패스하려고.", "lost_wallet_card_stop_practical"),
            ("책상 위가 난장판이라 정리부터 해야겠어.", "table_award_cleanup_first"),
            ("상대가 서운하다고 하는데 내 논리는 맞는 것 같아, 바로 반박하지 말고 뭐가 서운했는지 먼저 물어봐?", "speech_conflict_first_sentence"),
            ("체온이 계속 높고 해열제 먹은 시간도 헷갈려서 검색만 하고 있는데 지금 뭘 먼저 확인해야 해?", "heat_polysemy_fever_first"),
            ("가스 냄새가 위험한지 불안해서 창문과 관리사무소가 먼저인지 모르겠어.", "heating_bill_anxiety_practical"),
            ("보일러가 갑자기 에러 코드 띄우면서 온수가 안 나와서 멘붕이야.", "heating_bill_anxiety_practical"),
            ("기름불이 붙은 것 같아서 불안한데 물 부어도 돼?", "living_cost_pressure_practical"),
            ("계곡 물가 산책하다가 마트 간식 생각났어.", "living_cost_pressure_practical"),
            ("가스 냄새가 위험한지 불안해서 창문과 관리사무소가 먼저인지 모르겠어.", "gas_stove_ignition_issue_practical"),
            ("가스레인지 기름때가 안 닦여서 청소가 짜증나.", "gas_stove_ignition_issue_practical"),
            ("가스레인지 디자인이 예뻐서 사진 저장했는데 점화장치 후기가 별로래.", "gas_stove_ignition_issue_practical"),
            ("가스레인지 한쪽만 불이 안 붙어서 점화장치 문제인가 봐.", "appliance_design_review_judgment"),
            ("이번 신작 캐릭터 디자인 예뻐서 사진 저장했는데 후기가 별로래.", "appliance_design_review_judgment"),
            ("가스레인지 기름때가 안 닦여서 청소 후기를 찾아봤어.", "appliance_design_review_judgment"),
        )

        for text, forbidden_relation in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = resolve_word_senses(signals)
                inferred_tags = set(_input_tags(signals))
                relation_names = {
                    relation.name
                    for relation in infer_semantic_relations(
                        signals,
                        base_tags=tuple(inferred_tags),
                        resolved_senses=senses,
                    )
                }

                self.assertNotIn(forbidden_relation, relation_names)

    def test_semantic_relations_do_not_fire_on_expanded_false_positive_contexts(self) -> None:
        cases = (
            ("가스라이팅 당한 건가 싶어서 불안하고 창문 열어놓고 멍때렸어.", "gas_smell_emergency_practical_first"),
            ("가스비 고지서 보고 불안해서 창문 열고 한숨 쉬었어.", "gas_smell_emergency_practical_first"),
            ("보일러 요금 때문에 불안해서 관리사무소 앱만 봤어.", "gas_smell_emergency_practical_first"),
            ("가스레인지 디자인이 예뻐서 사진 저장했는데 점화장치 후기가 별로래.", "gas_smell_emergency_practical_first"),
            ("헬륨가스 풍선 영상 보다가 불안하긴 한데 창문 열면 괜찮나 싶었어.", "gas_smell_emergency_practical_first"),
            ("프라이팬 사고 싶은데 기름값도 올라서 물가가 불안해.", "oil_fire_water_misuse_practical_first"),
            ("피부 기름 때문에 물세안이 맞는지 불안해서 검색했어.", "oil_fire_water_misuse_practical_first"),
            ("기름진 음식 먹고 물 많이 마셨는데 속이 불편해.", "oil_fire_action_practical_first"),
            ("주유소 기름 냄새가 싫어서 물티슈로 손 닦았어.", "oil_fire_action_practical_first"),
            ("기름때 제거하려고 물에 불리는 게 맞나 찾아봤어.", "oil_fire_water_misuse_practical_first"),
            ("차 사고 싶어서 보험료랑 사진 후기만 계속 보고 있어.", "car_accident_first_steps_practical"),
            ("게임에서 차 사고 보험 아이템 쓰는 퀘스트가 있더라.", "car_accident_first_steps_practical"),
            ("광고 사진 보고 차 사고 싶은 마음이 커졌어.", "car_accident_first_steps_practical"),
            ("보험 광고가 너무 많아서 차 사진만 봐도 피곤해.", "car_accident_first_steps_practical"),
            ("중고차 사고 싶은데 과실 이력 조회가 어렵네.", "car_accident_first_steps_practical"),
            ("약속 전에 술 한잔할까 했는데 분위기가 불안해.", "medicine_alcohol_check_practical"),
            ("예약한 약속 자리에서 음주 얘기가 나와서 괜히 불편했어.", "medicine_alcohol_check_practical"),
            ("절약하려고 술약속 줄이는 중인데 마음이 불안해.", "medicine_alcohol_check_practical"),
            ("공약 발표 보고 한잔 마셨다는 댓글이 많아서 웃겼어.", "medicine_alcohol_check_practical"),
            ("약간 취한 연기 하는 영상 봤는데 진짜 술은 아니래.", "medicine_alcohol_check_practical"),
            ("회의 링크 눌렀는데 비밀번호 입력하래서 다시 확인했어.", "phishing_link_account_lock_practical"),
            ("유튜브 링크 눌렀는데 계정 비밀번호를 까먹은 것뿐이야.", "phishing_link_account_lock_practical"),
            ("숙소 링크 타고 예약 페이지 들어갔는데 권한 없음이 떴어.", "phishing_link_account_lock_practical"),
            ("문서 링크마다 접근 권한이 달라서 계정 설정을 봤어.", "phishing_link_account_lock_practical"),
            ("줌 링크가 안 열려서 비밀번호를 동료한테 물어봤어.", "phishing_link_account_lock_practical"),
            ("지갑 사정이 안 좋아서 카드 정리를 해야겠어.", "lost_wallet_card_stop_practical"),
            ("지갑 열 때마다 카드값 생각나서 자책하게 돼.", "lost_wallet_card_stop_practical"),
            ("게임 지갑이 털려서 카드 덱을 다시 짜야 해.", "lost_wallet_card_stop_practical"),
            ("지갑 방어한다고 카드 할인만 찾아보고 있어.", "lost_wallet_card_stop_practical"),
            ("카드뉴스 만들다가 지갑 사진을 잃어버린 줄 알았어.", "lost_wallet_card_stop_practical"),
            ("열 번 봐도 파일 여는 버튼 위치를 모르겠어.", "fever_body_check_practical_first"),
            ("열정이 과해서 체온 얘기까지 농담으로 나왔어.", "fever_body_check_practical_first"),
            ("열쇠 잃어버려서 불안한데 체온은 멀쩡해.", "fever_body_check_practical_first"),
            ("파일을 열어야 하는데 몸 상태 말고 노트북이 문제야.", "fever_body_check_practical_first"),
            ("고열량 간식 먹고 죄책감이 들어.", "fever_body_check_practical_first"),
            ("달 사진 보니까 이번 달 마감 생각나서 낭만이 깨졌어.", "moon_month_deadline_first_task"),
            ("달달한 커피 마시며 마감 얘기했더니 불안해졌어.", "moon_month_deadline_first_task"),
            ("방송 방 제목 정리하다가 내 방 청소 생각났어.", "room_cleanup_first_action"),
            ("해결 방안 정리하는 문서가 난장판이야.", "room_cleanup_first_action"),
            ("책상을 샀는데 상 받는 상상이 들어서 웃겼어.", "table_award_cleanup_first"),
            ("새 폰 케이스가 예뻐서 예전 폰 사진을 봤어.", "new_phone_adjustment"),
            ("친구 애인이 추천한 드라마 욕만 나오게 재미없더라.", "friend_partner_complaint_boundary"),
            ("편의점 결제 장치가 고장나서 줄이 길었어.", "impulse_spending_payment_friction"),
            ("한 편만 보려다 다음 편 예고가 별로라 바로 껐어.", "episode_binge_control"),
            ("부모님이 좋아하는 가치관 책을 선물했어.", "parent_value_conflict_boundary"),
            ("회사 동료가 개인적인 취향을 물어봤는데 그냥 스몰톡이었어.", "coworker_private_boundary"),
            ("동물이 나오는 영화랑 고양이 얘기로 대화했어.", "pet_talk_care_first"),
            ("로또 번호 통계 얘기하다가 세무 상담 유튜브를 봤어.", "lottery_practical_first"),
            ("복권 긁는 장면이 있는 영화 보고 사표 장면도 웃겼어.", "lottery_practical_first"),
            ("타임머신 영화 보다가 면접 장면에서 버스 놓치는 클리셰가 나왔어.", "time_machine_emergency_practical_override"),
        )

        self.assertEqual(len(cases), 50)
        for text, forbidden_relation in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = resolve_word_senses(signals)
                inferred_tags = set(_input_tags(signals))
                relation_names = {
                    relation.name
                    for relation in infer_semantic_relations(
                        signals,
                        base_tags=tuple(inferred_tags),
                        resolved_senses=senses,
                    )
                }

                self.assertNotIn(forbidden_relation, relation_names)

    def test_semantic_relations_do_not_fire_on_relation_bank_v2_false_positives(self) -> None:
        cases = (
            (
                "카톡 말투 분석 영상에서 차가워졌다는 예문을 확인해봤어.",
                "relationship_kakao_tone_anxiety_check",
            ),
            (
                "친구가 읽씹이라는 단어 뜻 예문에서 바쁜건지 모르겠다는 문장을 봤어.",
                "read_receipt_uncertainty_hold_judgment",
            ),
            (
                "캐릭터가 예민하고 상대가 무례해서 기분 확 상하는 장면인데 선 넘는 연출이 좋았어.",
                "relationship_boundary_polite_firm",
            ),
            (
                "배달 앱 광고에서 돈 아끼는 원칙을 말하길래 오늘 봤어.",
                "delivery_tired_compromise_practical",
            ),
            (
                "옆집 새벽배송 소음방지상품 광고에 쪽지랑 관리사무소 예시가 나오더라.",
                "neighbor_noise_record_first_practical",
            ),
            (
                "착한 거짓말 영화리뷰에서 상처랑 솔직함 중 뭐가 나은건지 어떻게 봐야 하는지 얘기하더라.",
                "white_lie_truth_tradeoff_judgment",
            ),
            (
                "면접 장면에서 버스 놓쳤고 담당자 연락 먼저 하는 클리셰가 나오더라.",
                "interview_missed_bus_practical_first",
            ),
        )

        for text, forbidden_relation in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = resolve_word_senses(signals)
                inferred_tags = set(_input_tags(signals))
                relation_names = {
                    relation.name
                    for relation in infer_semantic_relations(
                        signals,
                        base_tags=tuple(inferred_tags),
                        resolved_senses=senses,
                    )
                }

                self.assertNotIn(forbidden_relation, relation_names)

    def test_semantic_relations_do_not_fire_on_expanded_cue_boundary_false_positives(self) -> None:
        cases = (
            ("회의 링크 눌렀는데 권한 없음 뜨네, 이건 그냥 접근 권한 문제지?", "phishing_link_account_lock_practical"),
            ("유튜브 링크 눌렀는데 영상이 안 떠서 답답해.", "phishing_link_account_lock_practical"),
            ("숙소 링크 확인하려는데 예약 페이지가 느리게 열려.", "phishing_link_account_lock_practical"),
            ("문서 링크 공유했는데 상대가 못 열었다고 해.", "phishing_link_account_lock_practical"),
            ("나무위키 링크 타고 읽다 보니 시간이 다 사라졌어.", "phishing_link_account_lock_practical"),
            ("링크마다 광고가 너무 많아서 집중이 깨져.", "phishing_link_account_lock_practical"),
            ("초대 링크가 만료돼서 다시 보내달라고 해야겠다.", "phishing_link_account_lock_practical"),
            ("기술문서 링크를 저장해놨는데 어디 있는지 모르겠어.", "phishing_link_account_lock_practical"),
            ("카드뉴스 디자인이 예뻐서 저장했어.", "lost_wallet_card_stop_practical"),
            ("카드게임에서 덱을 잘못 짜서 졌어.", "lost_wallet_card_stop_practical"),
            ("생일 카드 문구를 뭐라고 써야 할지 모르겠어.", "lost_wallet_card_stop_practical"),
            ("교통카드 충전하려고 했는데 잔액이 부족했어.", "lost_wallet_card_stop_practical"),
            ("명함 카드 사진이 흐릿하게 찍혔어.", "lost_wallet_card_stop_practical"),
            ("카드값이 너무 많이 나와서 지갑이 아파.", "lost_wallet_card_stop_practical"),
            ("팬아트에 불꽃 효과 넣었더니 너무 과해 보여.", "oil_fire_action_practical_first"),
            ("아이돌 팬인데 티켓팅 실패해서 멘탈이 나갔어.", "oil_fire_action_practical_first"),
            ("프라이팬 코팅이 벗겨져서 새로 살까 고민돼.", "oil_fire_action_practical_first"),
            ("팬 소음이 커서 컴퓨터가 일하는 티를 너무 내.", "oil_fire_action_practical_first"),
            ("불닭볶음면에 물을 덜 버려서 맛이 애매해졌어.", "oil_fire_water_misuse_practical_first"),
            ("게임에서 불 속성 캐릭터가 팬덤에서 제일 인기야.", "oil_fire_action_practical_first"),
            ("노트북 물건너 온 택배가 아직 통관 중이래.", "device_water_damage_practical_first"),
            ("노트북으로 물가 자료를 정리해야 하는데 귀찮아.", "device_water_damage_practical_first"),
            ("노트북 화면에 물결무늬 배경화면 깔았더니 예뻐.", "device_water_damage_practical_first"),
            ("물리 과제 파일을 노트북에 저장했는데 어디 갔지.", "device_water_damage_practical_first"),
            ("휴대폰 물량이 부족해서 예약 배송이 늦어진대.", "device_water_damage_practical_first"),
            ("차 사고 싶어서 보험료랑 유지비를 계산해봤어.", "car_accident_first_steps_practical"),
            ("사고 싶은 차 사진 저장해놓고 매일 보고 있어.", "car_accident_first_steps_practical"),
            ("자동차 보험료 광고가 너무 과장돼 보여.", "car_accident_first_steps_practical"),
            ("게임에서 사고가 나서 보험 아이템을 썼어.", "car_accident_first_steps_practical"),
            ("차 사고 영상 보니까 운전이 더 무서워졌어.", "car_accident_first_steps_practical"),
            ("약속 시간이 두 번 바뀌어서 헷갈려.", "medicine_double_dose_practical_first"),
            ("약간 감기 기운이 있는 것 같아서 따뜻한 물 마셨어.", "medicine_double_dose_practical_first"),
            ("공약을 두 번이나 바꾼 정치인 얘기를 봤어.", "medicine_double_dose_practical_first"),
            ("예약을 또 먹통으로 만들어서 앱이 짜증나.", "medicine_double_dose_practical_first"),
            ("절약하려고 편의점 행사만 확인하고 있어.", "medicine_double_dose_practical_first"),
            ("프로젝트 이름만 새로 정하고 실제 시작은 못 했어.", "new_project_first_step_practical"),
            ("새 프로젝트 소개 영상을 봤는데 디자인이 예쁘더라.", "new_project_first_step_practical"),
            ("프로젝트 파일 정리하다가 폴더 이름 때문에 막혔어.", "new_project_first_step_practical"),
            ("마감 없는 개인 프로젝트라 천천히 해도 될 것 같아.", "new_project_first_step_practical"),
            ("프로젝트 굿즈가 예뻐서 사고 싶어졌어.", "new_project_first_step_practical"),
            ("책 펴놓고 휴대폰 충전만 하고 있어, 공부 얘기는 아니야.", "study_phone_first_action"),
            ("휴대폰 사진 정리하려고 책상에 앉았는데 귀찮아.", "study_phone_first_action"),
            ("폰 케이스를 치워놓고 어디 뒀는지 모르겠어.", "study_phone_first_action"),
            ("전자책 펴놓고 폰트 크기만 계속 바꾸는 중이야.", "study_phone_first_action"),
            ("책 표지 디자인이 완성도 높아서 감탄했어.", "perfectionism_sixty_point_start"),
            ("완성도 높은 영화라서 다시 보고 싶어.", "perfectionism_sixty_point_start"),
            ("완벽한 날씨라 산책 가고 싶어.", "perfectionism_sixty_point_start"),
            ("단톡방에 공지 올렸는데 다들 확인만 했나 봐.", "group_chat_silence_emotion_first"),
            ("단톡방에서 밈 보냈는데 반응 없어서 그냥 웃겼어.", "group_chat_silence_emotion_first"),
            ("부모님 선물 고르는데 누가 맞는지보다 취향이 문제야.", "parent_value_conflict_boundary"),
        )

        self.assertEqual(len(cases), 50)
        for text, forbidden_relation in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = resolve_word_senses(signals)
                inferred_tags = set(_input_tags(signals))
                relation_names = {
                    relation.name
                    for relation in infer_semantic_relations(
                        signals,
                        base_tags=tuple(inferred_tags),
                        resolved_senses=senses,
                    )
                }

                self.assertNotIn(forbidden_relation, relation_names)

    def test_semantic_relations_infer_expanded_hard_positive_boundaries(self) -> None:
        cases = (
            ("가스 새는 냄새가 나는 듯해서 밸브 잠그고 밖으로 나가야 하나?", "gas_smell_emergency_practical_first", "practical_first"),
            ("주방에서 이상한 가스 냄새 맡았는데 불 켜지 말고 환기부터 해야 해?", "gas_smell_emergency_practical_first", "practical_first"),
            ("가스 누출 느낌이 나서 스위치 건드리면 안 되는 거 맞지?", "gas_smell_emergency_practical_first", "practical_first"),
            ("냄새가 가스 같으면 창문 열고 바로 나가야 하는 상황이야?", "gas_smell_emergency_practical_first", "practical_first"),
            ("후라이팬에 식용유 불 붙었는데 물 말고 뚜껑이 먼저야?", "oil_fire_water_misuse_practical_first", "practical_first"),
            ("튀김하다가 팬에서 불길 올라왔어, 물 붓지 말고 가스부터 꺼?", "oil_fire_water_misuse_practical_first", "practical_first"),
            ("기름 두른 냄비에 불 붙은 것 같은데 소화기 찾아야 해?", "oil_fire_action_practical_first", "practical_first"),
            ("식용유 화재면 물 뿌리면 안 되는 거지?", "oil_fire_water_misuse_practical_first", "practical_first"),
            ("맥북에 커피 쏟았는데 켜보지 말고 전원부터 꺼야 해?", "device_water_damage_practical_first", "practical_first"),
            ("태블릿이 물에 젖었는데 충전하면 안 되지?", "device_water_damage_practical_first", "practical_first"),
            ("아이패드에 물을 쏟아서 화면이 이상해, 말리고 수리점 가야 해?", "device_water_damage_practical_first", "practical_first"),
            ("키보드에 음료 쏟았는데 계속 두드리면 안 되지?", "device_water_damage_practical_first", "practical_first"),
            ("카드사 문자 링크 눌렀는데 계정 잠그고 비번 바꿔야 해?", "phishing_link_account_lock_practical", "practical_first"),
            ("택배조회 링크가 수상해서 눌렀는데 결제내역 확인해야 해?", "phishing_link_account_lock_practical", "practical_first"),
            ("인증번호 입력한 것 같아서 계좌랑 카드부터 막아야 하나?", "phishing_link_account_lock_practical", "practical_first"),
            ("문자에 온 로그인 링크 누른 뒤 계정이 불안해졌어, 비밀번호부터 바꿔?", "phishing_link_account_lock_practical", "practical_first"),
            ("신용카드를 잃어버린 것 같아, 사용내역 보고 정지부터 해야 해?", "lost_wallet_card_stop_practical", "practical_first"),
            ("체크카드가 사라졌는데 찾기 전에 분실신고부터 걸어야 해?", "lost_wallet_card_stop_practical", "practical_first"),
            ("지갑을 버스에 두고 내린 것 같은데 카드 막는 게 먼저야?", "lost_wallet_card_stop_practical", "practical_first"),
            ("신분증이랑 카드 든 파우치를 잃어버렸어, 신고부터야?", "lost_wallet_card_stop_practical", "practical_first"),
            ("주차장에서 차를 긁었는데 사진 찍고 보험사 연락 먼저야?", "car_accident_first_steps_practical", "practical_first"),
            ("차끼리 부딪혔는데 괜찮냐고 확인하고 사진 남겨야 해?", "car_accident_first_steps_practical", "practical_first"),
            ("후진하다 접촉한 것 같은데 말싸움보다 블랙박스랑 위치 사진부터야?", "car_accident_first_steps_practical", "practical_first"),
            ("사람은 안 다친 것 같은데 사고 현장 사진부터 남기는 게 맞아?", "car_accident_first_steps_practical", "practical_first"),
            ("감기약 먹은 줄 모르고 또 먹을 뻔했어, 추가 복용 멈춰야 하지?", "medicine_double_dose_practical_first", "practical_first"),
            ("진통제를 방금 먹었는지 헷갈리는데 하나 더 먹으면 안 되지?", "medicine_double_dose_practical_first", "practical_first"),
            ("약 봉지를 보니 같은 약을 두 알 먹은 듯해, 약국에 전화해야 해?", "medicine_double_dose_practical_first", "practical_first"),
            ("복용 시간이 기억 안 나는데 또 먹지 말고 기록부터 봐야 해?", "medicine_double_dose_practical_first", "practical_first"),
            ("약 먹고 소주 마신 게 찜찜해, 더 마시지 말고 확인해야 해?", "medicine_alcohol_check_practical", "practical_first"),
            ("진통제 먹은 날 술 마셔도 되는지 모르겠어, 약사한테 물어봐야 해?", "medicine_alcohol_check_practical", "practical_first"),
            ("몸이 뜨겁고 오한이 있는데 체온계부터 찾는 게 맞아?", "fever_body_check_practical_first", "practical_first"),
            ("열감이 있고 해열제 먹은 시간이 기억 안 나, 시간 확인부터야?", "fever_body_check_practical_first", "practical_first"),
            ("천장 누수인지 물방울 떨어져, 사진 찍고 아래 물건 치워야 해?", "home_water_leak_practical", "practical_first"),
            ("윗집에서 물 새는 것 같은데 영상 남기고 관리실 연락해야 해?", "home_water_leak_practical", "practical_first"),
            ("벌레가 아니라 벌이 방에 들어온 것 같아, 잡으려 하지 말고 거리 둬야 해?", "bee_room_safety", "practical_first"),
            ("말벌 같은 게 커튼 쪽에 붙어있어, 문 닫고 멀어져야 하지?", "bee_room_safety", "practical_first"),
            ("처음 맡은 업무라 아무것도 모르겠어, 마감이랑 산출물부터 물어봐야 해?", "new_project_first_step_practical", "practical_first"),
            ("신규 과제 받았는데 모르는 용어가 많아, 질문 리스트 먼저 만들까?", "new_project_first_step_practical", "practical_first"),
            ("공부하려고 앉았는데 휴대폰 알림만 보고 있어, 알림 끄고 타이머부터야?", "study_phone_first_action", "practical_first"),
            ("책상에 앉자마자 폰부터 열어, 10분만 잠그는 게 맞지?", "study_phone_first_action", "practical_first"),
            ("완벽한 계획 세우다 하루 다 갔어, 초안 먼저 만드는 게 맞아?", "perfectionism_sixty_point_start", "practical_first"),
            ("처음부터 잘하려다 아무것도 못 냈어, 1차본부터 던져야 해?", "perfectionism_sixty_point_start", "practical_first"),
            ("답이 늦어서 읽씹인지 모르겠는데 바로 추궁하지 말아야 해?", "read_receipt_uncertainty_hold_judgment", "judgment"),
            ("카톡 확인한 것 같은데 답이 없어서 확정하고 싶어, 보류가 맞아?", "read_receipt_uncertainty_hold_judgment", "judgment"),
            ("단체방에서 내 질문만 지나간 것 같아, 상처 키우기보다 가볍게 다시 물어봐?", "group_chat_silence_emotion_first", "emotion_stabilize"),
            ("단톡에서 답이 안 와서 민망한데 길게 해명하지 말아야 해?", "group_chat_silence_emotion_first", "emotion_stabilize"),
            ("이별하고 밤에 긴 카톡 쓰고 있어, 보내지 말고 저장해야 하지?", "breakup_long_message_emotion_first", "emotion_stabilize"),
            ("전애인한테 새벽에 긴 메시지 보내고 싶어, 내일 낮에 다시 봐야 해?", "breakup_long_message_emotion_first", "emotion_stabilize"),
            ("엄마가 내 선택을 애 취급해서 상처야, 논쟁 말고 여기까지만 해야 해?", "parent_value_conflict_boundary", "emotion_stabilize"),
            ("아빠랑 가치관 얘기하다 또 싸울 것 같아, 대화 한계부터 세워야 해?", "parent_value_conflict_boundary", "emotion_stabilize"),
        )

        self.assertEqual(len(cases), 50)
        for text, expected_relation, expected_priority in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = resolve_word_senses(signals)
                inferred_tags = set(_input_tags(signals))
                ranked = rank_semantic_relations(
                    infer_semantic_relations(
                        signals,
                        base_tags=tuple(inferred_tags),
                        resolved_senses=senses,
                    )
                )

                self.assertTrue(ranked)
                self.assertEqual(ranked[0].name, expected_relation)
                self.assertEqual(ranked[0].priority, expected_priority)

    def test_semantic_relation_arbiter_prefers_specific_polysemy_over_generic_relation(self) -> None:
        text = "열 번 검색했는데 열이 나는 것 같고 파일도 열어야 해서 머리가 꼬여. 지금은 체온부터 재는 게 맞아?"
        signals = _text_signals(text)
        senses = resolve_word_senses(signals)
        inferred_tags = set(_input_tags(signals))
        ranked = rank_semantic_relations(
            infer_semantic_relations(
                signals,
                base_tags=tuple(inferred_tags),
                resolved_senses=senses,
            )
        )

        self.assertTrue(ranked)
        self.assertEqual(ranked[0].name, "heat_polysemy_fever_first")
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertIn("cue:열 번", ranked[0].positive_evidence)
        self.assertIn("cue:파일", ranked[0].positive_evidence)

    def test_semantic_relation_arbiter_separates_heating_from_general_living_cost(self) -> None:
        cases = (
            (
                "기름값보다 공과금이 무서워서 히터 켜는 시간을 줄여야겠어.",
                "heating_bill_anxiety_practical",
            ),
            (
                "식비는 아직 버티겠는데 보일러 난방비가 불안해서 온도를 낮춰야겠어.",
                "heating_bill_anxiety_practical",
            ),
            (
                "가스비보다 이번 주 식비랑 기름값 쪽이 더 신경 쓰여.",
                "living_cost_pressure_practical",
            ),
            (
                "물가 때문에 이번달생활비가 흔들려서 마트 가기 겁나.",
                "living_cost_pressure_practical",
            ),
            (
                "보일러 얘기처럼 들리지만 사실은 주유비랑 식비가 먼저 터졌어.",
                "living_cost_pressure_practical",
            ),
        )

        for text, expected_relation in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = resolve_word_senses(signals)
                inferred_tags = set(_input_tags(signals))
                ranked = rank_semantic_relations(
                    infer_semantic_relations(
                        signals,
                        base_tags=tuple(inferred_tags),
                        resolved_senses=senses,
                    )
                )

                self.assertTrue(ranked)
                self.assertEqual(ranked[0].name, expected_relation)

    def test_word_sense_bank_keeps_new_senses_inside_compound_prompts(self) -> None:
        cases = (
            ("논리적으로 카페인이 효율적인지도 궁금한데 지금 머리가 띵해서 물 한잔 마시고 판단해도 되는지 실전 조언 줘.", ("물", "water_drink"), {"water", "hydration"}),
            ("세탁 원리를 따져보고 싶은데 검은 셔츠에 물빠짐이 생겨서 기분이 확 상했어, 지금 뭘 해야 해?", ("물", "dye_color_bleed"), {"dye", "color_bleed", "stain"}),
            ("확률로는 별일 아닐 수도 있겠지만 고열이 나는 것 같아서 불안해, 체온부터 재야 해?", ("열", "fever_heat"), {"fever", "body_heat", "temperature"}),
            ("열 번 설명했는데도 안 통하면 내가 문제인지 상대가 문제인지 감정적으로 빡쳐, 어떻게 말해야 해?", ("열", "number_ten"), {"number", "ten"}),
            ("파일을 열어 봐야 논리적으로 확인이 되는데 갑자기 무서워서 미루는 중이야, 뭐부터 눌러?", ("열", "open_action"), {"open_action", "access"}),
            ("30초 안에 답해야 하는 상황이라 판단이 흐려졌어, 감정보다 실전으로 첫 문장만 줘.", ("초", "second_time_unit"), {"second_unit", "timer"}),
            ("초보 운전이라 사고 확률을 계산해도 골목길만 보면 손이 떨려, 지금은 어떤 판단이 맞아?", ("초", "beginner_novice"), {"beginner", "novice"}),
            ("양초 켜놓고 위로받고 싶은데 불 조심 논리도 신경 쓰여, 마음은 불안하고 실전 팁도 필요해.", ("초", "candle_light"), {"candle", "wick", "flame"}),
            ("내 방 정리가 인생 통제감이랑 무슨 관계인지 철학도 궁금한데 너무 답답해서 지금 첫 행동만 정해줘.", ("방", "room_space"), {"room", "indoor_space"}),
            ("오늘 방송 켜면 사람들이 재미없어할까 봐 불안한데 알고리즘 논리보다 첫 멘트가 먼저 필요해.", ("방", "broadcast_stream"), {"broadcast", "streaming"}),
            ("해결 방안이 안 보이면 감정부터 달래야 하는지 논리부터 세워야 하는지 모르겠어, 실전 순서 줘.", ("방", "method_solution"), {"method", "solution"}),
            ("팔에 검은 점 하나가 생긴 게 별일 아닐 확률도 알지만 괜히 불안해, 지금 뭘 확인해야 해?", ("점", "dot_mark"), {"dot", "mark", "spot"}),
            ("시험 점수가 낮은 게 내 능력의 증거인지 감정이 무너져서 모르겠어, 다음 행동부터 말해줘.", ("점", "score_grade"), {"score", "grade"}),
            ("편의점에서 충동구매한 게 합리적 보상인지 낭비인지 따지고 싶은데 기분은 좀 나아졌어, 어떻게 봐야 해?", ("점", "store_shop"), {"store", "shop"}),
            ("마음 편하게 쉬는 게 생산성인지 게으름인지 계속 따지다가 더 지쳤어, 지금은 어떻게 쉬어야 해?", ("편", "comfort_ease"), {"comfort", "ease"}),
            ("다음 편을 보는 게 도피인지 회복인지 논리적으로 헷갈리는데 오늘은 너무 지쳤어, 봐도 돼?", ("편", "episode_series"), {"episode", "series"}),
            ("사람은 많은데 내 편이 없다는 생각이 들어서 철학이고 뭐고 외로워, 지금 어떻게 버텨?", ("편", "ally_side"), {"ally", "side"}),
            ("20대 초반엔 괜찮았던 루틴이 지금은 안 먹혀서 내가 늙은 건지 시스템이 틀린 건지 모르겠어.", ("대", "age_group"), {"age_group", "generation"}),
            ("노트북 한 대 더 사는 게 효율 투자일지 충동구매일지 감정이 섞여서 판단이 안 돼.", ("대", "unit_counter"), {"counter_unit", "vehicle_unit"}),
            ("장난치다 한 대 맞은 게 별일 아닌지 선 넘은 건지 기분이 나빠서 논리적으로 정리하고 싶어.", ("대", "hit_blow"), {"hit", "blow"}),
            ("상 받으면 인정 욕구가 채워지는 건지 진짜 성취인지 모르겠고, 기대하다 실망할까 봐 불안해.", ("상", "award_prize"), {"award", "prize"}),
            ("책상 위가 엉망이면 머리도 복잡해지는 게 과학인지 핑계인지 모르겠어, 지금 어디부터 치워?", ("상", "table_surface"), {"table_surface", "serving_table"}),
            ("상처가 났는데 별거 아닌지 병원 갈 일인지 판단이 안 되고 겁나, 지금 첫 확인 뭐야?", ("상", "wound_injury"), {"wound", "injury"}),
        )

        for text, expected_sense, expected_tags in cases:
            with self.subTest(text=text):
                signals = _text_signals(text)
                senses = {(sense.word, sense.sense): sense for sense in resolve_word_senses(signals)}
                inferred_tags = set(_input_tags(signals))

                self.assertIn(expected_sense, senses)
                self.assertTrue(expected_tags <= inferred_tags)
                self.assertTrue(senses[expected_sense].matched_cues)

    def test_semantic_word_bank_covers_commute_chat_sleep_and_media(self) -> None:
        cases = (
            (
                "오늘 버스 카드 찍는데 잔액이 부족합니다 떠서 등줄기에 식은땀 쫙 남.",
                ("payment", "transit", "card", "panic"),
                "잔액부족 식은땀",
            ),
            (
                "카톡에 ㅋㅋ 몇 개 붙일지 고민하는 거 나만 진심임?",
                ("chat", "kakao", "nuance", "overthinking"),
                "카톡 ㅋ개수 심리전",
            ),
            (
                "약속 취소됐으면 좋겠다고 속으로 빌었는데 진짜 취소돼서 내적 축제 열림.",
                ("social", "appointment", "homebody", "relief"),
                "약속취소 내적축제",
            ),
            (
                "넷플릭스 뭐 볼지 고르다가 1시간 지나서 결국 아무것도 못 보고 잠.",
                ("media", "netflix", "choice_paralysis", "time_loss"),
                "넷플릭스 선택장애 엔딩",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_covers_work_chores_relationship_and_small_happiness(self) -> None:
        cases = (
            (
                "메일에 첨부파일 확인 부탁드립니다 적어놓고 파일 첨부 안 해서 재송부함.",
                ("work", "email", "mistake", "embarrassment"),
                "첨부파일 유령",
            ),
            (
                "설거지 하기 싫어서 싱크대에 그릇 산맥이 쌓여 있음.",
                ("chores", "dishwashing", "annoyance", "home"),
                "설거지 보스몹",
            ),
            (
                "애인이 내 절친 깻잎 떼어주면 질투 버튼 바로 눌릴 듯.",
                ("relationship", "jealousy", "debate", "boundary"),
                "깻잎 경보",
            ),
            (
                "오늘 퇴근길에 노을이 너무 예뻐서 사진 찍었는데 하나도 안 담김.",
                ("scenery", "photo", "sunset", "frustration"),
                "노을 저장 실패",
            ),
            (
                "오늘 하루 중 제일 기분 좋았던 순간은 편의점에서 좋아하는 간식 발견한 거야. 소확행 인정.",
                ("positive", "small_happiness", "daily", "warm"),
                "소확행 저장완료",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_covers_debates_fashion_family_fear_and_reflection(self) -> None:
        cases = (
            (
                "민트초코는 치약 맛 아니냐? 민초파랑 반민초파 갈리는 거 개웃김.",
                ("food", "taste_debate", "mint_choco", "preference"),
                "민초 전선",
            ),
            (
                "탕수육은 부먹이냐 찍먹이냐 이건 진짜 소스 논쟁임.",
                ("food", "taste_debate", "sauce", "preference"),
                "부먹찍먹 휴전선",
            ),
            (
                "옷장 앞에서 15분 서 있었는데 입을 옷이 하나도 없어.",
                ("fashion", "clothes", "closet", "choice_paralysis"),
                "옷장 블랙홀",
            ),
            (
                "명절에 친척들이 취업 결혼 잔소리 시작하면 방어 멘트부터 준비함.",
                ("family", "holiday", "nagging", "defense"),
                "명절 잔소리 방패",
            ),
            (
                "가위눌렸는데 몸이 안 움직여서 새벽에 진짜 무서웠어.",
                ("fear", "sleep_paralysis", "night", "body"),
                "가위눌림 정지화면",
            ),
            (
                "친구랑 다퉜는데 내가 먼저 사과해야 할지 자존심 때문에 고민돼.",
                ("relationship", "conflict", "apology", "timing"),
                "사과 타이밍 싸움",
            ),
            (
                "우주 끝 생각하면 내가 너무 작게 느껴져서 존재 현타 옴.",
                ("philosophy", "existential", "space", "small_self", "reflective"),
                "우주적 현타",
            ),
            (
                "너 같은 AI 봇이 말동무처럼 느껴질 때가 있어서 좀 신기해.",
                ("ai_companion", "relationship", "comfort", "rapport"),
                "AI 친구감",
            ),
            (
                "좀비 사태 터지면 대형마트로 도망가서 생존해야지.",
                ("survival", "zombie", "fear", "action"),
                "좀비 생존본능",
            ),
            (
                "치약 짜고 칫솔에 물 묻히는지 안 묻히는지 이거 은근 논쟁임.",
                ("routine", "toothbrush", "habit", "taste_debate"),
                "치약 물논쟁",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_covers_fandom_camping_books_audio_and_home_details(self) -> None:
        cases = (
            (
                "이번 최애 장르 팝업스토어 평일 웨이팅 몇 시간인지 눈치게임 중.",
                ("fandom", "popup_store", "waiting", "goods"),
                "팝업 웨이팅 눈치게임",
            ),
            (
                "해외 직구 굿즈가 한 달 만에 세관 통과해서 택배 조회 새로고침 중.",
                ("fandom", "overseas_order", "customs", "delivery"),
                "세관 통과 기도",
            ),
            (
                "차박 가서 노지에서 별 보면서 불멍했는데 완전 힐링이었다.",
                ("camping", "car_camping", "stars", "healing"),
                "차박 별멍",
            ),
            (
                "캠핑장 옆 텐트가 매너타임 안 지키고 새벽까지 고성방가해서 밤샘.",
                ("camping", "noise", "manner_time", "frustration"),
                "매너타임 붕괴",
            ),
            (
                "알라딘 중고서점에서 절판 희귀본 발견해서 심봤다 싶었어.",
                ("book", "used_bookstore", "rare_find", "joy"),
                "중고서점 심봤다",
            ),
            (
                "고전 소설 세 페이지 읽고 단어 뜻 몰라서 폰 켰어. 문해력 큰일남.",
                ("book", "reading", "difficulty", "self_reflection"),
                "문해력 재활훈련",
            ),
            (
                "노이즈 캔슬링 헤드폰 바꿨더니 지하철 소음 완전 차단됨.",
                ("audio", "noise_canceling", "commute", "quiet"),
                "노캔 고요버프",
            ),
            (
                "스마트워치 샀는데 알림 확인이랑 폰 찾기만 쓰고 있음.",
                ("gadget", "smartwatch", "usage", "daily"),
                "스마트워치 알림기계",
            ),
            (
                "혼술 각이라 편의점에서 맥주랑 닭강정 사 와서 세팅 완료.",
                ("food", "drink", "alone", "night"),
                "혼술 세팅완료",
            ),
            (
                "싱크대 물 한 방울씩 똑똑 떨어지는 소리 때문에 새벽에 일어남.",
                ("home", "sound", "obsession", "night"),
                "물방울 새벽집착",
            ),
            (
                "샤워하다가 내가 머리를 감았나 린스를 했나 기억 안 나서 추리함.",
                ("routine", "shower", "forgetfulness", "daily"),
                "샤워 기억상실",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_covers_laundry_admin_health_retail_and_home_bills(self) -> None:
        cases = (
            (
                "세탁기 종료음 들렸는데 빨래 꺼내기 귀찮아서 그냥 방치 중.",
                ("laundry", "chores", "procrastination", "home"),
                "세탁기 종료음 방치",
            ),
            (
                "흰옷 입고 짜장 먹다가 얼룩 튀어서 하루 망함.",
                ("clothes", "stain", "food_accident", "embarrassment"),
                "흰옷 얼룩참사",
            ),
            (
                "비밀번호 계속 틀려서 계정 잠기고 재설정 지옥에 빠짐.",
                ("digital", "password", "login", "frustration"),
                "비밀번호 리셋지옥",
            ),
            (
                "무료체험 끊어놓고 까먹어서 정기결제 자동결제 됐어.",
                ("money", "subscription", "recurring_payment", "regret"),
                "구독결제 유령",
            ),
            (
                "주민센터 가서 등본이랑 초본 발급받는데 서류 던전 같더라.",
                ("admin", "documents", "public_office", "errand"),
                "주민센터 서류던전",
            ),
            (
                "치과 드릴 위잉 소리만 들어도 무서워서 몸이 굳음.",
                ("health", "dentist", "fear", "sound"),
                "치과 드릴공포",
            ),
            (
                "버스 도착정보는 곧 도착이라는데 정류장에서 10분째 안 와.",
                ("commute", "bus", "waiting", "delay"),
                "버스 도착 희망고문",
            ),
            (
                "무인계산대에서 바코드 안 찍혀서 뒤에 사람들 눈치 보임.",
                ("retail", "self_checkout", "awkward", "payment"),
                "무인계산대 버벅임",
            ),
            (
                "분리수거 하려는데 라벨 떼고 플라스틱 종이 나누는 게 퍼즐임.",
                ("chores", "recycling", "sorting", "home"),
                "분리수거 퍼즐",
            ),
            (
                "이번 달 관리비 고지서 보고 전기세 때문에 현실 펀치 맞음.",
                ("home", "bills", "maintenance_fee", "money"),
                "관리비 현실고지서",
            ),
            (
                "식물 물 주는 타이밍 모르겠어. 과습일까 말라 죽을까 고민됨.",
                ("plant", "care", "watering", "home"),
                "식물 물주기 딜레마",
            ),
            (
                "머리 식히려고 동네 한 바퀴 산책했더니 기분이 좀 리셋됨.",
                ("walk", "neighborhood", "healing", "routine"),
                "동네 산책 리셋",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_covers_cooking_beauty_exercise_performance_family_and_devices(self) -> None:
        cases = (
            (
                "밥물 계량 잘못해서 진밥도 된밥도 아닌 이상한 밥 됐어.",
                ("cooking", "rice", "mistake", "texture"),
                "밥물 계량참사",
            ),
            (
                "남은 피자 에어프라이어에 돌렸더니 다시 바삭하게 부활함.",
                ("cooking", "air_fryer", "leftover", "food"),
                "에어프라이어 부활술",
            ),
            (
                "피부가 갑자기 뒤집어져서 트러블 올라오고 얼굴 컨디션 비상임.",
                ("beauty", "skin", "trouble", "condition"),
                "피부 뒤집힘 비상",
            ),
            (
                "운동복까지 입었는데 헬스 가기 귀찮아서 준비만 하고 끝남.",
                ("exercise", "procrastination", "routine", "failure"),
                "운동복 입고 끝",
            ),
            (
                "콘서트 티켓팅 광탈함. 좌석 예매 전쟁 진짜 피의 전쟁이네.",
                ("performance", "ticketing", "competition", "panic"),
                "티켓팅 피의전쟁",
            ),
            (
                "엄마 부재중 전화가 3통 와 있으면 괜히 심장 철렁함.",
                ("family", "missed_call", "panic", "care"),
                "엄마 부재중전화 경보",
            ),
            (
                "와이파이 방구석만 안 터져서 공유기 앞에서만 인터넷 됨.",
                ("digital", "wifi", "connection", "home"),
                "와이파이 음영구역",
            ),
            (
                "블루투스 끊긴 줄 모르고 노래가 폰 스피커로 크게 나왔어.",
                ("digital", "bluetooth", "connection", "embarrassment"),
                "블루투스 배신",
            ),
            (
                "급하게 과제 출력하려는데 프린터 종이걸림 떠서 멘붕 옴.",
                ("device", "printer", "paper_jam", "work"),
                "프린터 종이걸림 저주",
            ),
            (
                "파일명이 최종 진짜최종 최최종 수정본까지 가서 나도 뭐가 뭔지 모르겠어.",
                ("digital", "file", "versioning", "work"),
                "최종진짜최종 파일",
            ),
            (
                "브라우저 탭을 99개 열어놨더니 어디서 뭐 보던 건지 모르겠음.",
                ("digital", "browser", "tabs", "clutter"),
                "탭 99개 숲",
            ),
            (
                "클라우드 동기화 됐는지 불안해서 백업 파일 계속 확인함.",
                ("digital", "cloud", "sync", "file"),
                "클라우드 동기화 불신",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_covers_short_reaction_words(self) -> None:
        cases = (
            (
                "이거 완전 개꿀인데?",
                ("short_word", "positive", "lucky"),
                "개꿀",
            ),
            (
                "오늘 일 처리 진짜 노답이라 말이 안 나옴.",
                ("short_word", "negative", "hopeless"),
                "노답",
            ),
            (
                "남들은 다 갓생 사는데 나만 침대라서 현타 온다.",
                ("short_word", "self_reflection", "low_energy"),
                "현타",
            ),
            (
                "갑자기 파일 날아가서 멘붕 왔어.",
                ("short_word", "panic", "confusion"),
                "멘붕",
            ),
            (
                "오늘 진짜 억까 당한 기분.",
                ("short_word", "bad_luck", "frustration"),
                "억까",
            ),
            (
                "이 말투 은근 킹받음.",
                ("short_word", "annoyance", "playful"),
                "킹받음",
            ),
            (
                "답장이 없어서 뭔가 찝찝해.",
                ("short_word", "uneasy", "relationship"),
                "찝찝",
            ),
            (
                "약속 취소돼서 속으로 내적환호함.",
                ("short_word", "relief", "happy"),
                "내적환호",
            ),
            (
                "사람 많은 모임 다녀오면 기빨림 장난 아님.",
                ("short_word", "introvert", "fatigue"),
                "기빨림",
            ),
            (
                "오늘 할 일 다 끝내서 뿌듯해.",
                ("short_word", "pride", "positive"),
                "뿌듯",
            ),
            (
                "새벽 감성 노래 듣다가 마음이 좀 먹먹해짐.",
                ("short_word", "deep_feeling", "sad"),
                "먹먹",
            ),
            (
                "할 일 많은데 또 쇼츠 보면서 딴짓함.",
                ("short_word", "distraction", "procrastination"),
                "딴짓",
            ),
            (
                "여름엔 수박 차갑게 썰어 먹는 게 최고지.",
                ("short_word", "food", "fruit", "summer"),
                "수박",
            ),
            (
                "발표 끝나고 박수 받으면 진짜 뿌듯할 것 같아.",
                ("short_word", "applause", "praise", "reaction"),
                "박수",
            ),
            (
                "사과 깎아 먹고 싶은데 냉장고에 없네.",
                ("short_word", "food", "fruit"),
                "사과",
            ),
            (
                "오늘은 라면 끓여 먹자.",
                ("short_word", "food", "meal", "noodle"),
                "라면",
            ),
            (
                "커피 없으면 오전이 안 굴러가.",
                ("short_word", "drink", "coffee", "energy"),
                "커피",
            ),
            (
                "답장 안 와서 괜히 걱정된다.",
                ("short_word", "emotion", "worry"),
                "걱정",
            ),
            (
                "힘들다니까 위로 한마디만 해줘.",
                ("short_word", "comfort", "support"),
                "위로",
            ),
            (
                "내일 발표라 응원 좀 해줘.",
                ("short_word", "support", "encouragement"),
                "응원",
            ),
            (
                "우산 안 챙겼는데 비 올 것 같아.",
                ("short_word", "object", "umbrella"),
                "우산",
            ),
            (
                "오늘 비 오네. 빗소리는 좋은데 나가기 싫다.",
                ("short_word", "weather", "rain"),
                "비옴",
            ),
            (
                "퇴근길 노을이 진짜 예쁘더라.",
                ("short_word", "scenery", "sunset"),
                "노을",
            ),
            (
                "지갑 어디 뒀는지 모르겠어.",
                ("short_word", "object", "wallet"),
                "지갑",
            ),
            (
                "허리가 아파서 의자에 오래 못 앉아 있겠어.",
                ("short_word", "body", "pain", "back"),
                "허리",
            ),
            (
                "안경에 김 서려서 앞이 안 보여.",
                ("short_word", "object", "glasses"),
                "안경",
            ),
        )

        for text, desired_tags, expected in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_short_ambiguous_words(self) -> None:
        cases = (
            (
                "배가 고파서 꼬르륵거려.",
                ("short_word", "ambiguous", "body", "stomach", "hunger"),
                "배",
                "hunger",
            ),
            (
                "항구에서 배를 타고 섬에 들어갔어.",
                ("short_word", "ambiguous", "vehicle", "ship"),
                "배",
                "ship",
            ),
            (
                "배 깎아 먹었는데 진짜 달더라.",
                ("short_word", "ambiguous", "food", "fruit", "pear"),
                "배",
                "pear",
            ),
            (
                "렌즈 오래 꼈더니 눈이 너무 아파.",
                ("short_word", "ambiguous", "body", "eye", "vision"),
                "눈",
                "eye",
            ),
            (
                "첫눈 오면 괜히 기분 좋아져.",
                ("short_word", "ambiguous", "weather", "snow"),
                "눈",
                "snow",
            ),
            (
                "말을 너무 세게 해서 분위기 싸해졌어.",
                ("short_word", "ambiguous", "speech", "talk"),
                "말",
                "speech",
            ),
            (
                "제주도 가서 말 타보고 싶어.",
                ("short_word", "ambiguous", "animal", "horse"),
                "말",
                "horse",
            ),
            (
                "밤늦게 자면 다음 날 망해.",
                ("short_word", "ambiguous", "time", "night"),
                "밤",
                "night",
            ),
            (
                "군밤 냄새 맡으니까 겨울 느낌 난다.",
                ("short_word", "ambiguous", "food", "chestnut"),
                "밤",
                "chestnut",
            ),
            (
                "차 타고 드라이브 가고 싶다.",
                ("short_word", "ambiguous", "vehicle", "car"),
                "차",
                "car",
            ),
            (
                "따뜻한 차 마시니까 좀 진정된다.",
                ("short_word", "ambiguous", "drink", "tea"),
                "차",
                "tea",
            ),
            (
                "우리 둘 입장 차가 꽤 큰 것 같아.",
                ("short_word", "ambiguous", "difference", "gap"),
                "차",
                "gap",
            ),
            (
                "문제 풀다가 머리 터질 것 같아.",
                ("short_word", "ambiguous", "solve", "release"),
                "풀",
                "solve",
            ),
            (
                "나는 찍먹파라 소스 붓는 거 못 참아.",
                ("short_word", "ambiguous", "faction", "team"),
                "파",
                "faction",
            ),
            (
                "입김 때문에 안경에 김 서려서 앞이 안 보여.",
                ("short_word", "ambiguous", "steam", "fog"),
                "김",
                "steam",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_more_short_ambiguous_words(self) -> None:
        cases = (
            (
                "사과 깎아 먹었는데 진짜 달더라.",
                ("short_word", "ambiguous", "food", "fruit", "apple"),
                "사과",
                "apple",
            ),
            (
                "친구한테는 내가 먼저 사과해야 할 것 같아.",
                ("short_word", "ambiguous", "social", "apology", "relationship"),
                "사과",
                "apology",
            ),
            (
                "오래 걸었더니 다리가 너무 아파.",
                ("short_word", "ambiguous", "body", "leg", "pain"),
                "다리",
                "leg",
            ),
            (
                "한강 다리 위에서 야경 보면 기분 좋아져.",
                ("short_word", "ambiguous", "structure", "bridge", "place"),
                "다리",
                "bridge",
            ),
            (
                "머리가 지끈거려서 아무 생각도 안 나.",
                ("short_word", "ambiguous", "body", "head", "pain"),
                "머리",
                "head",
            ),
            (
                "앞머리 자를까 말까 계속 고민 중이야.",
                ("short_word", "ambiguous", "beauty", "hair", "style"),
                "머리",
                "hair",
            ),
            (
                "이 문제는 머리 굴려도 답이 안 나와.",
                ("short_word", "ambiguous", "thinking", "idea", "brain"),
                "머리",
                "thinking",
            ),
            (
                "등이 결려서 의자에 오래 못 앉아 있겠어.",
                ("short_word", "ambiguous", "body", "back", "pain"),
                "등",
                "back",
            ),
            (
                "방 전등 꺼야 하는데 일어나기 귀찮아.",
                ("short_word", "ambiguous", "object", "light", "lamp"),
                "등",
                "light",
            ),
            (
                "커피, 우유, 과자 등등 챙겨서 갈게.",
                ("short_word", "ambiguous", "etc", "list", "category"),
                "등",
                "etc",
            ),
            (
                "손목이 아파서 키보드 치는 게 힘들어.",
                ("short_word", "ambiguous", "body", "hand", "finger"),
                "손",
                "hand",
            ),
            (
                "오늘 진상 손님 때문에 멘탈이 털렸어.",
                ("short_word", "ambiguous", "customer", "service", "social"),
                "손님",
                "customer",
            ),
            (
                "괜히 샀다가 손해 본 기분이야.",
                ("short_word", "ambiguous", "loss", "money", "regret"),
                "손해",
                "loss",
            ),
            (
                "문지방에 발가락 찧어서 눈물 날 뻔했어.",
                ("short_word", "ambiguous", "body", "foot", "pain"),
                "발",
                "foot",
            ),
            (
                "내일 발표라 벌써부터 긴장돼.",
                ("short_word", "ambiguous", "work", "school", "presentation", "nervous"),
                "발표",
                "presentation",
            ),
            (
                "길거리에서 우연히 예쁜 가게를 발견했어.",
                ("short_word", "ambiguous", "place", "road", "navigation"),
                "길",
                "road",
            ),
            (
                "글이 너무 길어서 읽다가 지쳤어.",
                ("short_word", "ambiguous", "length", "long", "too_long"),
                "길",
                "long",
            ),
            (
                "현관문 닫았는지 계속 신경 쓰여.",
                ("short_word", "ambiguous", "object", "door", "home"),
                "문",
                "door",
            ),
            (
                "이 문장은 말투가 좀 딱딱해 보여.",
                ("short_word", "ambiguous", "language", "sentence", "writing"),
                "문장",
                "sentence",
            ),
            (
                "퇴근하고 마트 가서 장 봐야 해.",
                ("short_word", "ambiguous", "shopping", "market", "grocery"),
                "장",
                "market",
            ),
            (
                "매운 거 먹고 장 트러블 와서 고생 중.",
                ("short_word", "ambiguous", "body", "bowel", "stomach"),
                "장",
                "bowel",
            ),
            (
                "책 다음 장 넘기기 아까울 정도로 몰입했어.",
                ("short_word", "ambiguous", "book", "chapter", "page"),
                "장",
                "chapter",
            ),
            (
                "비 오는 날엔 파전 먹어야지.",
                ("short_word", "ambiguous", "food", "pancake", "korean_food"),
                "전",
                "pancake",
            ),
            (
                "며칠 전 얘기인데 아직도 생각나.",
                ("short_word", "ambiguous", "time", "before", "past"),
                "전",
                "before",
            ),
            (
                "콘서트 티켓팅은 진짜 피의 전쟁이야.",
                ("short_word", "ambiguous", "competition", "battle", "intense"),
                "전쟁",
                "battle",
            ),
            (
                "회 먹을 때 초장 찍어 먹는 편이야.",
                ("short_word", "ambiguous", "food", "sashimi", "seafood"),
                "회",
                "sashimi",
            ),
            (
                "회의실 들어가자마자 머리가 하얘졌어.",
                ("short_word", "ambiguous", "work", "meeting", "office"),
                "회의",
                "meeting",
            ),
            (
                "내 방 청소하다가 옛날 일기장을 발견했어.",
                ("short_word", "ambiguous", "place", "room", "home"),
                "방",
                "room",
            ),
            (
                "이럴 때 대처법이나 방법이 뭐가 좋을까?",
                ("short_word", "ambiguous", "method", "solution", "advice"),
                "방법",
                "method",
            ),
            (
                "발표 끝나고 상 받으면 진짜 뿌듯할 것 같아.",
                ("short_word", "ambiguous", "award", "praise", "achievement"),
                "상",
                "award",
            ),
            (
                "집밥 한상 차려 먹고 싶다.",
                ("short_word", "ambiguous", "table", "meal", "food"),
                "상",
                "table",
            ),
            (
                "오늘 내 상태가 좀 이상하게 멍해.",
                ("short_word", "ambiguous", "condition", "state", "status"),
                "상태",
                "state",
            ),
            (
                "장기 기증 같은 얘기는 생각할수록 묵직해.",
                ("short_word", "ambiguous", "body", "organ", "health"),
                "장기",
                "organ",
            ),
            (
                "장기자랑 나가면 뭘 해야 할지 모르겠어.",
                ("short_word", "ambiguous", "talent", "skill", "performance"),
                "장기",
                "talent",
            ),
            (
                "이건 장기적으로 봐야 하는 계획 같아.",
                ("short_word", "ambiguous", "time", "long_term", "plan"),
                "장기",
                "long_term",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_body_senses_and_everyday_words(self) -> None:
        cases = (
            (
                "매운 거 먹고 속이 쓰려.",
                ("short_word", "ambiguous", "body", "stomach", "pain"),
                "속",
                "stomach",
            ),
            (
                "속마음은 좀 서운한데 겉으론 괜찮은 척했어.",
                ("short_word", "ambiguous", "emotion", "inner_feeling", "mind"),
                "속",
                "inner_feeling",
            ),
            (
                "인터넷 속도가 너무 느려서 답답해.",
                ("short_word", "ambiguous", "speed", "pace", "digital"),
                "속도",
                "speed",
            ),
            (
                "입안이 헐어서 밥 먹기 힘들어.",
                ("short_word", "ambiguous", "body", "mouth", "pain"),
                "입",
                "mouth",
            ),
            (
                "카페 입구에서 기다릴게.",
                ("short_word", "ambiguous", "place", "entrance", "meeting_place"),
                "입구",
                "entrance",
            ),
            (
                "내 입장에서는 좀 억울해.",
                ("short_word", "ambiguous", "stance", "perspective", "opinion"),
                "입장",
                "stance",
            ),
            (
                "목이 칼칼해서 감기 올 것 같아.",
                ("short_word", "ambiguous", "body", "throat", "health"),
                "목",
                "throat",
            ),
            (
                "목소리 음색이 너무 좋아.",
                ("short_word", "ambiguous", "voice", "sound", "tone"),
                "목소리",
                "voice",
            ),
            (
                "오늘 목요일인 줄 알고 착각했어.",
                ("short_word", "ambiguous", "time", "weekday", "thursday"),
                "목요일",
                "thursday",
            ),
            (
                "올해 목표를 너무 크게 잡았나 봐.",
                ("short_word", "ambiguous", "goal", "plan", "growth"),
                "목표",
                "goal",
            ),
            (
                "팔꿈치가 아파서 운동 쉬어야겠어.",
                ("short_word", "ambiguous", "body", "arm", "pain"),
                "팔",
                "arm",
            ),
            (
                "안 쓰는 물건 중고로 팔아야겠어.",
                ("short_word", "ambiguous", "sell", "marketplace", "money"),
                "팔다",
                "sell",
            ),
            (
                "이건 내 팔자인가 싶어서 웃겨.",
                ("short_word", "ambiguous", "fate", "fortune", "life"),
                "팔자",
                "fate",
            ),
            (
                "코막힘 때문에 잠을 못 잤어.",
                ("short_word", "ambiguous", "body", "nose", "breathing"),
                "코",
                "nose",
            ),
            (
                "코드 디버깅하다가 새벽 됐어.",
                ("short_word", "ambiguous", "tech", "code", "programming"),
                "코드",
                "code",
            ),
            (
                "비트코인 샀다가 계좌가 파래졌어.",
                ("short_word", "ambiguous", "money", "coin", "crypto"),
                "코인",
                "coin",
            ),
            (
                "귀가 먹먹해서 이어폰 빼야겠어.",
                ("short_word", "ambiguous", "body", "ear", "hearing"),
                "귀",
                "ear",
            ),
            (
                "퇴근 후 귀가길이 너무 길어.",
                ("short_word", "ambiguous", "commute", "home_return", "route"),
                "귀가",
                "home_return",
            ),
            (
                "혼자 공포 영화 보다가 귀신 나와서 껐어.",
                ("short_word", "ambiguous", "fear", "ghost", "horror"),
                "귀신",
                "ghost",
            ),
            (
                "충전기 줄이 자꾸 꼬여서 짜증나.",
                ("short_word", "ambiguous", "object", "cable", "string"),
                "줄",
                "cable",
            ),
            (
                "식당 대기줄이 너무 길어.",
                ("short_word", "ambiguous", "queue", "waiting", "line"),
                "줄",
                "queue",
            ),
            (
                "첫 문장 한 줄을 못 쓰겠어.",
                ("short_word", "ambiguous", "language", "sentence_line", "writing"),
                "한 줄",
                "sentence_line",
            ),
            (
                "팔에 검은 점 하나 생겼어.",
                ("short_word", "ambiguous", "mark", "dot", "spot"),
                "점",
                "dot",
            ),
            (
                "타로점 봤는데 은근 맞아서 소름.",
                ("short_word", "ambiguous", "fortune", "divination", "mystic"),
                "점",
                "fortune",
            ),
            (
                "이 답변은 10점 만점에 몇 점일까?",
                ("short_word", "ambiguous", "score", "rating", "evaluation"),
                "점수",
                "score",
            ),
            (
                "물 한잔 마시니까 좀 살 것 같아.",
                ("short_word", "ambiguous", "drink", "water", "hydration"),
                "물",
                "water",
            ),
            (
                "중요한 물건을 잃어버렸어.",
                ("short_word", "ambiguous", "object", "belongings", "item"),
                "물건",
                "object",
            ),
            (
                "하나만 물어봐도 돼?",
                ("short_word", "ambiguous", "ask", "question", "conversation"),
                "질문",
                "question",
            ),
            (
                "불 나면 제일 먼저 뭐 챙겨야 하지?",
                ("short_word", "ambiguous", "fire", "danger", "heat"),
                "불",
                "fire",
            ),
            (
                "방 불빛이 너무 밝아서 잠이 안 와.",
                ("short_word", "ambiguous", "light", "lighting", "room"),
                "불빛",
                "light",
            ),
            (
                "괜히 불안해서 계속 확인하게 돼.",
                ("short_word", "ambiguous", "emotion", "anxiety", "worry"),
                "불안",
                "anxiety",
            ),
            (
                "위층 소리 때문에 잠을 못 잤어.",
                ("short_word", "sound", "noise", "sensory"),
                "소리",
                "sound",
            ),
            (
                "오늘 기분이 이상하게 가라앉아.",
                ("short_word", "emotion", "mood", "condition"),
                "기분",
                "mood",
            ),
            (
                "마음이 복잡해서 정리가 안 돼.",
                ("short_word", "emotion", "mind", "heart"),
                "마음",
                "mind",
            ),
            (
                "심장 철렁해서 수명 줄어든 줄.",
                ("short_word", "body", "panic", "heart"),
                "심장",
                "panic",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_daily_polysemy_expansion(self) -> None:
        cases = (
            (
                "라면 면발이 꼬들꼬들해야 진짜 맛있어.",
                ("short_word", "ambiguous", "food", "noodle", "meal"),
                "면",
                "noodle",
            ),
            (
                "책상 표면이 거칠어서 마우스가 잘 안 움직여.",
                ("short_word", "ambiguous", "surface", "texture", "object"),
                "표면",
                "surface",
            ),
            (
                "내일 면접장 가는 생각만 해도 긴장돼.",
                ("short_word", "ambiguous", "work", "interview", "nervous"),
                "면접",
                "interview",
            ),
            (
                "만약 하루가 30시간이면 남는 시간에 뭐 할까?",
                ("short_word", "ambiguous", "condition", "if", "hypothetical"),
                "만약",
                "hypothetical",
            ),
            (
                "그 농담은 솔직히 선 넘는 말이었어.",
                ("short_word", "ambiguous", "boundary", "line_crossing", "social"),
                "선",
                "line_crossing",
            ),
            (
                "이 그림은 곡선이 부드러워서 색감이 살아.",
                ("short_word", "ambiguous", "line", "shape", "visual"),
                "선",
                "line",
            ),
            (
                "친구 생일 선물로 기프티콘 보낼까?",
                ("short_word", "ambiguous", "gift", "relationship", "positive"),
                "선물",
                "gift",
            ),
            (
                "오늘 노을 색감이 진짜 미쳤어.",
                ("short_word", "ambiguous", "color", "visual", "taste"),
                "색",
                "color",
            ),
            (
                "오늘 안색이 너무 창백해 보여.",
                ("short_word", "ambiguous", "complexion", "face", "body"),
                "색",
                "complexion",
            ),
            (
                "비누향 나는 향수가 제일 좋아.",
                ("short_word", "ambiguous", "scent", "smell", "sensory"),
                "향",
                "scent",
            ),
            (
                "이 프로젝트 방향성이 조금 애매해.",
                ("short_word", "ambiguous", "direction", "route", "orientation"),
                "방향",
                "direction",
            ),
            (
                "이 집 떡볶이는 단맛이 강한데 맛있어.",
                ("short_word", "ambiguous", "taste", "food", "sensory"),
                "맛",
                "taste",
            ),
            (
                "이 게임은 보는 맛보다 직접 하는 재미가 커.",
                ("short_word", "ambiguous", "fun", "interest", "enjoyment"),
                "재미",
                "fun",
            ),
            (
                "키보드 누르는 손맛이 좋아서 계속 타자 치게 돼.",
                ("short_word", "ambiguous", "feel", "tactile", "skill"),
                "손맛",
                "feel",
            ),
            (
                "정 들어서 쉽게 버리지를 못하겠어.",
                ("short_word", "ambiguous", "affection", "attachment", "relationship"),
                "정",
                "affection",
            ),
            (
                "생각이 너무 많아서 정리가 안 돼.",
                ("short_word", "ambiguous", "organize", "cleanup", "clarity"),
                "정리",
                "organize",
            ),
            (
                "이 문제 정답이 뭔지 모르겠어.",
                ("short_word", "ambiguous", "answer", "correct", "judgement"),
                "정답",
                "answer",
            ),
            (
                "오늘 정신이 없어서 카톡 답장을 깜빡했어.",
                ("short_word", "ambiguous", "mind", "focus", "mental_state"),
                "정신",
                "mind",
            ),
            (
                "기운이 없어서 아무것도 못 하겠어.",
                ("short_word", "ambiguous", "energy", "vitality", "body"),
                "기운",
                "energy",
            ),
            (
                "오늘 묘한 기운이 이상하게 느껴져.",
                ("short_word", "ambiguous", "vibe", "omen", "mood"),
                "기운",
                "vibe",
            ),
            (
                "방 공기가 답답해서 환기해야겠어.",
                ("short_word", "ambiguous", "air", "weather", "breathing"),
                "공기",
                "air",
            ),
            (
                "회의실 공기가 싸해져서 아무도 말을 못 했어.",
                ("short_word", "ambiguous", "atmosphere", "mood", "social"),
                "분위기",
                "atmosphere",
            ),
            (
                "지하철에서 창가 자리가 비어서 바로 앉았어.",
                ("short_word", "ambiguous", "seat", "place", "transit"),
                "자리",
                "seat",
            ),
            (
                "팀에서 내 포지션과 역할이 애매해.",
                ("short_word", "ambiguous", "position", "role", "job"),
                "자리",
                "role",
            ),
            (
                "오늘 옷차림은 꾸안꾸 느낌으로 갔어.",
                ("short_word", "ambiguous", "outfit", "fashion", "appearance"),
                "차림",
                "outfit",
            ),
            (
                "집밥 상차림이 너무 든든해서 기분 좋아.",
                ("short_word", "ambiguous", "meal_setting", "table", "food"),
                "상차림",
                "meal_setting",
            ),
            (
                "그 말 듣고 표정 관리가 안 됐어.",
                ("short_word", "face", "expression", "emotion"),
                "표정",
                "expression",
            ),
            (
                "이번 판은 초반부터 말렸어.",
                ("short_word", "ambiguous", "game_round", "match", "play"),
                "판",
                "game_round",
            ),
            (
                "괜히 말 꺼냈다가 판이 커졌어.",
                ("short_word", "ambiguous", "situation", "mess", "state"),
                "판",
                "situation",
            ),
            (
                "사진 각도랑 구도가 진짜 중요하더라.",
                ("short_word", "ambiguous", "angle", "visual", "geometry"),
                "각",
                "angle",
            ),
            (
                "이건 퇴근각이다. 더 못 버티겠어.",
                ("short_word", "ambiguous", "chance", "slang", "timing"),
                "각",
                "slang",
            ),
            (
                "이건 법적으로 문제가 될 수도 있어.",
                ("short_word", "ambiguous", "law", "rule", "society"),
                "법",
                "law",
            ),
            (
                "잠 잘 오는 방법이나 노하우 있어?",
                ("short_word", "ambiguous", "method", "how_to", "advice"),
                "방법",
                "method",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_life_action_and_text_polysemy(self) -> None:
        cases = (
            (
                "요즘 야식 때문에 살이 쪘어.",
                ("short_word", "ambiguous", "body", "weight", "flesh"),
                "살",
                "weight",
            ),
            (
                "스무 살 때로 돌아가면 뭘 할까?",
                ("short_word", "ambiguous", "age", "life_stage", "number"),
                "살",
                "age",
            ),
            (
                "무인도에서도 살 수 있을 것 같아?",
                ("short_word", "ambiguous", "life", "survival", "living"),
                "살다",
                "survival",
            ),
            (
                "이 가방 살까 말까 계속 고민 중이야.",
                ("short_word", "ambiguous", "buy", "shopping", "money"),
                "사다",
                "buy",
            ),
            (
                "감기 기운인지 열이 나는 것 같아.",
                ("short_word", "ambiguous", "body", "fever", "temperature"),
                "열",
                "fever",
            ),
            (
                "창문 열어봤는데 바람이 너무 차.",
                ("short_word", "ambiguous", "open", "door", "action"),
                "열다",
                "open",
            ),
            (
                "요즘 열정이 예전 같지 않아.",
                ("short_word", "ambiguous", "passion", "motivation", "energy"),
                "열정",
                "passion",
            ),
            (
                "이 답변 문장틀은 조금 더 쪼개야 해.",
                ("short_word", "ambiguous", "template", "frame", "structure"),
                "틀",
                "template",
            ),
            (
                "방금 네 해석은 조금 틀린 것 같아.",
                ("short_word", "ambiguous", "wrong", "mistake", "answer"),
                "틀리다",
                "wrong",
            ),
            (
                "비 오는 날엔 잔잔한 노래 틀어놓고 싶어.",
                ("short_word", "ambiguous", "play", "turn_on", "media"),
                "틀다",
                "play",
            ),
            (
                "문틈으로 바람이 계속 들어와.",
                ("short_word", "ambiguous", "gap", "crack", "space"),
                "틈",
                "gap",
            ),
            (
                "틈틈이 영어 공부라도 해야 하는데.",
                ("short_word", "ambiguous", "free_time", "spare_time", "routine"),
                "틈",
                "free_time",
            ),
            (
                "이 계획은 빈틈이 너무 많아 보여.",
                ("short_word", "ambiguous", "weakness", "loophole", "gap"),
                "허점",
                "weakness",
            ),
            (
                "아침 햇빛이 눈부셔서 바로 깼어.",
                ("short_word", "ambiguous", "light", "visual", "brightness"),
                "빛",
                "light",
            ),
            (
                "그래도 아직 희망이 조금은 보이는 것 같아.",
                ("short_word", "ambiguous", "hope", "positive", "future"),
                "희망",
                "hope",
            ),
            (
                "대출 때문에 빚이 늘어서 걱정돼.",
                ("short_word", "ambiguous", "debt", "money", "burden"),
                "빚",
                "debt",
            ),
            (
                "오랜만에 그림 그렸는데 생각보다 괜찮아.",
                ("short_word", "ambiguous", "art", "drawing", "creative"),
                "그림",
                "drawing",
            ),
            (
                "방 구석 검은 그림자가 괜히 신경 쓰여.",
                ("short_word", "ambiguous", "shadow", "visual", "uncanny"),
                "그림자",
                "shadow",
            ),
            (
                "글쓰기 시작하려고 했는데 첫 문장이 안 나와.",
                ("short_word", "ambiguous", "writing", "text", "language"),
                "글",
                "writing",
            ),
            (
                "커뮤 게시글 올렸다가 댓글이 너무 많이 달렸어.",
                ("short_word", "ambiguous", "post", "sns", "online"),
                "게시글",
                "post",
            ),
            (
                "내 글씨체가 너무 악필이라 나도 못 알아봐.",
                ("short_word", "ambiguous", "handwriting", "writing", "visual"),
                "글씨",
                "handwriting",
            ),
            (
                "여행 갈 짐을 아직 하나도 못 챙겼어.",
                ("short_word", "ambiguous", "luggage", "belongings", "travel"),
                "짐",
                "luggage",
            ),
            (
                "그 말이 계속 마음의 짐처럼 남아.",
                ("short_word", "ambiguous", "burden", "pressure", "emotion"),
                "짐",
                "burden",
            ),
            (
                "요즘 체력이랑 근력이 너무 떨어진 것 같아.",
                ("short_word", "ambiguous", "strength", "body", "energy"),
                "힘",
                "strength",
            ),
            (
                "오늘 하루가 너무 힘들어서 못 버티겠어.",
                ("short_word", "ambiguous", "hardship", "fatigue", "emotion"),
                "힘들다",
                "hardship",
            ),
            (
                "어제 꿈에서 이상한 터널을 봤어.",
                ("short_word", "ambiguous", "dream", "sleep", "unconscious"),
                "꿈",
                "dream",
            ),
            (
                "어릴 때 장래희망은 뭐였어?",
                ("short_word", "ambiguous", "goal", "aspiration", "future"),
                "꿈",
                "aspiration",
            ),
            (
                "이 단어가 무슨 뜻인지 모르겠어.",
                ("short_word", "ambiguous", "meaning", "language", "interpretation"),
                "뜻",
                "meaning",
            ),
            (
                "그 사람이 무슨 의도로 그런 말을 했을까?",
                ("short_word", "ambiguous", "intention", "motive", "mind"),
                "의도",
                "intention",
            ),
            (
                "뜻밖의 선물을 받아서 기분이 좋아졌어.",
                ("short_word", "ambiguous", "surprise", "unexpected", "event"),
                "뜻밖",
                "surprise",
            ),
            (
                "머릿결이 좋아 보여서 부럽다.",
                ("short_word", "ambiguous", "texture", "grain", "surface"),
                "결",
                "texture",
            ),
            (
                "오늘 저녁 메뉴를 빨리 결정해야 해.",
                ("short_word", "ambiguous", "decision", "choice", "judgement"),
                "결정",
                "decision",
            ),
            (
                "그 영화 결말은 아직도 여운이 남아.",
                ("short_word", "ambiguous", "ending", "story", "media"),
                "결말",
                "ending",
            ),
            (
                "드디어 과제가 끝났어.",
                ("short_word", "ambiguous", "end", "finish", "time"),
                "끝",
                "end",
            ),
            (
                "커피 끝맛이 고소해서 마음에 들어.",
                ("short_word", "ambiguous", "aftertaste", "taste", "food"),
                "끝맛",
                "aftertaste",
            ),
            (
                "검은 옷에 고양이 털이 너무 많이 묻었어.",
                ("short_word", "ambiguous", "hair", "fur", "body"),
                "털",
                "fur",
            ),
            (
                "이불 먼지 털고 나니까 좀 개운해.",
                ("short_word", "ambiguous", "shake_off", "clean", "action"),
                "털다",
                "shake_off",
            ),
            (
                "오늘 회의에서 멘탈이 탈탈 털렸어.",
                ("short_word", "ambiguous", "loss", "defeat", "mental"),
                "털리다",
                "loss",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_social_body_and_sensory_polysemy(self) -> None:
        cases = (
            (
                "친구가 아무렇지 않게 던진 말에 상처받았어.",
                ("short_word", "ambiguous", "emotion", "hurt", "relationship"),
                "상처",
                "hurt",
            ),
            (
                "요리하다가 손가락이 베여서 상처가 났어.",
                ("short_word", "ambiguous", "body", "injury", "wound"),
                "상처",
                "injury",
            ),
            (
                "머릿속으로 이미 퇴사하고 제주도 사는 상상까지 끝냈어.",
                ("short_word", "ambiguous", "imagination", "fantasy", "mind"),
                "상상",
                "imagination",
            ),
            (
                "택배 상자 뜯을 때 테이프 찢는 손맛이 좋아.",
                ("short_word", "ambiguous", "box", "package", "object"),
                "상자",
                "box",
            ),
            (
                "요즘 고민이 많아서 진지하게 상담 좀 받고 싶어.",
                ("short_word", "ambiguous", "advice", "counseling", "talk"),
                "상담",
                "counseling",
            ),
            (
                "내일 입을 상의는 셔츠가 나을까 티셔츠가 나을까?",
                ("short_word", "ambiguous", "clothes", "top", "fashion"),
                "상의",
                "top",
            ),
            (
                "이건 혼자 정하지 말고 친구랑 상의하고 결정하자.",
                ("short_word", "ambiguous", "consult", "discuss", "decision"),
                "상의하다",
                "consult",
            ),
            (
                "내 작은 바람은 그냥 오늘 하루 조용히 지나가는 거야.",
                ("short_word", "ambiguous", "wish", "hope", "desire"),
                "바람",
                "wish",
            ),
            (
                "연인이 바람피우면 바로 손절할 것 같아.",
                ("short_word", "ambiguous", "relationship", "cheating", "betrayal"),
                "바람피우다",
                "cheating",
            ),
            (
                "이번 달 식비가 지난달보다 두 배로 늘었어.",
                ("short_word", "ambiguous", "number", "multiple", "comparison"),
                "배수",
                "multiple",
            ),
            (
                "믿었던 친구한테 배신당한 느낌이라 마음이 안 좋아.",
                ("short_word", "ambiguous", "relationship", "betrayal", "hurt"),
                "배신",
                "betrayal",
            ),
            (
                "그 영화 주연 배우 연기가 진짜 좋더라.",
                ("short_word", "ambiguous", "entertainment", "actor", "media"),
                "배우",
                "actor",
            ),
            (
                "요즘 새 언어를 배우고 싶어서 앱을 깔았어.",
                ("short_word", "ambiguous", "learning", "study", "skill"),
                "배우다",
                "learning",
            ),
            (
                "방 청소하다가 옛날 일기장을 발견했어.",
                ("short_word", "ambiguous", "discover", "find", "surprise"),
                "발견",
                "discover",
            ),
            (
                "영어 발음이 자꾸 꼬여서 말하기가 민망해.",
                ("short_word", "ambiguous", "speech", "pronunciation", "language"),
                "발음",
                "pronunciation",
            ),
            (
                "내일 발표라서 피피티를 계속 고치고 있어.",
                ("short_word", "ambiguous", "presentation", "public_speaking", "work"),
                "발표",
                "presentation",
            ),
            (
                "계단 내려가다가 발목을 접질렀어.",
                ("short_word", "ambiguous", "body", "ankle", "pain"),
                "발목",
                "ankle",
            ),
            (
                "예전 실수가 계속 내 발목을 잡는 것 같아.",
                ("short_word", "ambiguous", "obstacle", "hold_back", "problem"),
                "발목잡다",
                "obstacle",
            ),
            (
                "단톡방 분위기 이상해서 눈치 보다가 그냥 조용히 있었어.",
                ("short_word", "ambiguous", "social", "reading_room", "attention"),
                "눈치",
                "reading_room",
            ),
            (
                "그 장면 보고 눈물이 날 뻔했어.",
                ("short_word", "ambiguous", "emotion", "tears", "sad"),
                "눈물",
                "tears",
            ),
            (
                "내 눈높이가 너무 높아서 기대치가 문제인가 봐.",
                ("short_word", "ambiguous", "perspective", "standard", "expectation"),
                "눈높이",
                "standard",
            ),
            (
                "계속 선 넘는 친구는 결국 손절하는 게 맞을까?",
                ("short_word", "ambiguous", "relationship", "cut_off", "boundary"),
                "손절",
                "cut_off",
            ),
            (
                "생일에 받은 손편지가 아직도 제일 기억에 남아.",
                ("short_word", "ambiguous", "letter", "gift", "sentimental"),
                "손편지",
                "letter",
            ),
            (
                "힘들 때 조용히 챙겨주는 손길이 진짜 오래 남더라.",
                ("short_word", "ambiguous", "touch", "care", "affection"),
                "손길",
                "care",
            ),
            (
                "머릿속에 생각이 너무 많아서 잠이 안 와.",
                ("short_word", "ambiguous", "mind", "thought", "overthinking"),
                "머릿속",
                "thought",
            ),
            (
                "자기 전에 머리맡에 물 한 컵은 꼭 둬.",
                ("short_word", "ambiguous", "bedside", "sleep", "room"),
                "머리맡",
                "bedside",
            ),
            (
                "그 말 듣고 가슴이 먹먹해졌어.",
                ("short_word", "ambiguous", "emotion", "heart", "feeling"),
                "가슴",
                "heart",
            ),
            (
                "계단 뛰어올랐더니 가슴이 답답하고 숨이 막혀.",
                ("short_word", "ambiguous", "body", "chest", "health"),
                "가슴",
                "chest",
            ),
            (
                "요즘 술을 자주 마셨더니 간 수치가 걱정돼.",
                ("short_word", "ambiguous", "body", "liver", "health"),
                "간",
                "liver",
            ),
            (
                "찌개 간이 조금 싱거워서 소금을 더 넣었어.",
                ("short_word", "ambiguous", "taste", "seasoning", "food"),
                "간",
                "seasoning",
            ),
            (
                "친구가 자꾸 내 반응을 간보는 것 같아서 불편해.",
                ("short_word", "ambiguous", "test", "probe", "social"),
                "간보다",
                "probe",
            ),
            (
                "오늘은 왠지 일이 잘 풀릴 것 같은 감이 와.",
                ("short_word", "ambiguous", "sense", "intuition", "feel"),
                "감",
                "intuition",
            ),
            (
                "가을엔 단감이나 홍시가 제일 맛있어.",
                ("short_word", "ambiguous", "food", "persimmon", "fruit"),
                "감",
                "persimmon",
            ),
            (
                "내 감정선이 요즘 너무 쉽게 흔들리는 것 같아.",
                ("short_word", "ambiguous", "emotion", "feeling", "inner_state"),
                "감정",
                "feeling",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_core_daily_polysemy_batch_two(self) -> None:
        cases = (
            (
                "오늘 할 일이 너무 많아서 업무부터 쳐내야 해.",
                ("short_word", "ambiguous", "work", "task", "duty"),
                "일",
                "task",
            ),
            (
                "오늘 하루가 유난히 길게 느껴졌어.",
                ("short_word", "ambiguous", "time", "day", "date"),
                "일",
                "day",
            ),
            (
                "무슨 일 있었어? 표정이 좀 안 좋아 보여.",
                ("short_word", "ambiguous", "event", "incident", "situation"),
                "일",
                "incident",
            ),
            (
                "퇴근하면 바로 집에 가서 쉬고 싶어.",
                ("short_word", "ambiguous", "place", "home", "rest"),
                "집",
                "home",
            ),
            (
                "이번 설정집 컬러칩 디테일이 진짜 좋더라.",
                ("short_word", "ambiguous", "book", "collection", "compiled_volume"),
                "집",
                "collection",
            ),
            (
                "집중이 안 돼서 같은 문장만 계속 읽고 있어.",
                ("short_word", "ambiguous", "focus", "attention", "study"),
                "집중",
                "focus",
            ),
            (
                "오늘 방송 켜면 뭐부터 이야기할까?",
                ("short_word", "ambiguous", "broadcast", "streaming", "vtuber"),
                "방송",
                "broadcast",
            ),
            (
                "조용한 회의실에서 방귀 뀔까 봐 긴장했어.",
                ("short_word", "ambiguous", "body", "fart", "embarrassment"),
                "방귀",
                "fart",
            ),
            (
                "좀비 사태면 일단 문 잠그고 방어부터 해야지.",
                ("short_word", "ambiguous", "defense", "protect", "strategy"),
                "방어",
                "defense",
            ),
            (
                "이 문제는 혼자 풀기엔 좀 막혔어.",
                ("short_word", "ambiguous", "problem", "issue", "question"),
                "문제",
                "problem",
            ),
            (
                "인증 문자 안 와서 로그인 못 하고 있어.",
                ("short_word", "ambiguous", "chat", "text_message", "notification"),
                "문자",
                "text_message",
            ),
            (
                "회사 조직문화가 묘하게 답답해.",
                ("short_word", "ambiguous", "culture", "society", "habit"),
                "문화",
                "culture",
            ),
            (
                "친구가 거짓말한 걸 알아서 신뢰가 흔들려.",
                ("short_word", "ambiguous", "lie", "deception", "relationship"),
                "거짓말",
                "lie",
            ),
            (
                "그 말투가 은근 비꼬는 느낌이라 기분이 별로야.",
                ("short_word", "ambiguous", "tone", "speech_style", "language"),
                "말투",
                "tone",
            ),
            (
                "밤 11시에 치킨 시키려는 나를 누가 좀 말려줘.",
                ("short_word", "ambiguous", "stop", "dissuade", "advice"),
                "말리다",
                "dissuade",
            ),
            (
                "빨래를 말려야 하는데 건조대가 꽉 찼어.",
                ("short_word", "ambiguous", "dry", "laundry", "hair"),
                "말리다",
                "dry",
            ),
            (
                "그 영화 마지막 장면은 아직도 기억나.",
                ("short_word", "ambiguous", "scene", "memory", "media"),
                "장면",
                "scene",
            ),
            (
                "내 장점이 뭔지 잘 모르겠어서 자신감이 떨어져.",
                ("short_word", "ambiguous", "strength", "advantage", "positive"),
                "장점",
                "advantage",
            ),
            (
                "장례식장 갈 때 무슨 말을 해야 할지 어렵더라.",
                ("short_word", "ambiguous", "funeral", "death", "legacy"),
                "장례",
                "funeral",
            ),
            (
                "농담했는데 상대가 정색해서 분위기 싸해졌어.",
                ("short_word", "ambiguous", "expression", "serious_face", "social"),
                "정색",
                "serious_face",
            ),
            (
                "친구들이랑 여행비 정산하는 게 제일 귀찮아.",
                ("short_word", "ambiguous", "money", "settlement", "split_bill"),
                "정산",
                "settlement",
            ),
            (
                "정류장에서 버스 기다리는데 20분째 안 와.",
                ("short_word", "ambiguous", "commute", "bus_stop", "waiting"),
                "정류장",
                "bus_stop",
            ),
            (
                "면접 복장으로 정장을 입어야 할까?",
                ("short_word", "ambiguous", "clothes", "suit", "formal"),
                "정장",
                "suit",
            ),
            (
                "그 정도면 꽤 괜찮은 결과 아닌가?",
                ("short_word", "ambiguous", "degree", "extent", "measure"),
                "정도",
                "degree",
            ),
            (
                "내 성격이 너무 예민한 건지 모르겠어.",
                ("short_word", "ambiguous", "personality", "trait", "identity"),
                "성격",
                "personality",
            ),
            (
                "다시 태어나면 성별이 달라도 괜찮을 것 같아.",
                ("short_word", "ambiguous", "gender", "identity", "body"),
                "성별",
                "gender",
            ),
            (
                "이번 프로젝트 성과가 생각보다 좋아서 뿌듯해.",
                ("short_word", "ambiguous", "result", "achievement", "work"),
                "성과",
                "achievement",
            ),
            (
                "이 노트북 성능이면 로컬 모델도 돌릴 수 있나?",
                ("short_word", "ambiguous", "performance", "tech", "hardware"),
                "성능",
                "performance",
            ),
            (
                "그때 기억이 아직도 선명하게 떠올라.",
                ("short_word", "ambiguous", "memory", "recall", "past"),
                "기억",
                "memory",
            ),
            (
                "이 기계는 버튼이 너무 많아서 어렵다.",
                ("short_word", "ambiguous", "machine", "device", "tech"),
                "기계",
                "machine",
            ),
            (
                "이번 기회를 놓치면 후회할 것 같아.",
                ("short_word", "ambiguous", "opportunity", "chance", "future"),
                "기회",
                "opportunity",
            ),
            (
                "기차 타고 부산 가는 길이 은근 설레.",
                ("short_word", "ambiguous", "vehicle", "train", "travel"),
                "기차",
                "train",
            ),
            (
                "그 뉴스 기사 제목만 봐도 어질어질하더라.",
                ("short_word", "ambiguous", "news", "article", "media"),
                "기사",
                "article",
            ),
            (
                "배달 기사님이 문 앞에 두고 가셨어.",
                ("short_word", "ambiguous", "driver", "delivery", "service"),
                "기사님",
                "driver",
            ),
            (
                "어느 기준으로 봐야 맞는 건지 헷갈려.",
                ("short_word", "ambiguous", "standard", "criterion", "judgement"),
                "기준",
                "standard",
            ),
            (
                "콘서트 표 예매하려고 대기 중이야.",
                ("short_word", "ambiguous", "ticket", "reservation", "travel"),
                "표",
                "ticket",
            ),
            (
                "엑셀 표로 정리하면 훨씬 보기 좋겠다.",
                ("short_word", "ambiguous", "table_chart", "spreadsheet", "data"),
                "표",
                "spreadsheet",
            ),
            (
                "퇴근길에 버스 타고 멍 때리는 시간이 좋아.",
                ("short_word", "ambiguous", "ride", "vehicle", "commute"),
                "타다",
                "ride",
            ),
            (
                "햇볕에 팔이 새까맣게 타버렸어.",
                ("short_word", "ambiguous", "burn", "sunburn", "heat"),
                "타다",
                "sunburn",
            ),
            (
                "아침에 믹스커피 타 마시고 정신 차렸어.",
                ("short_word", "ambiguous", "mix", "drink", "coffee"),
                "타다",
                "mix",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_core_daily_polysemy_batch_three(self) -> None:
        cases = (
            (
                "아침에 창문 열었더니 새소리가 들려서 기분 좋았어.",
                ("short_word", "ambiguous", "animal", "bird", "nature"),
                "새",
                "bird",
            ),
            (
                "새로 산 이어폰 케이스가 생각보다 예쁘더라.",
                ("short_word", "ambiguous", "new", "fresh", "object"),
                "새것",
                "fresh",
            ),
            (
                "새벽 감성 타서 괜히 옛날 노래만 들었어.",
                ("short_word", "ambiguous", "time", "dawn", "night"),
                "새벽",
                "dawn",
            ),
            (
                "밤하늘에 별빛이 진짜 선명해서 한참 봤어.",
                ("short_word", "ambiguous", "sky", "star", "night"),
                "별",
                "star",
            ),
            (
                "그 카페는 기대했는데 맛이 별로였어.",
                ("short_word", "ambiguous", "negative", "dislike", "evaluation"),
                "별로",
                "dislike",
            ),
            (
                "오늘 별일 없었어? 표정이 좀 지쳐 보여.",
                ("short_word", "ambiguous", "event", "unusual", "daily"),
                "별일",
                "unusual",
            ),
            (
                "보름달이 너무 밝아서 사진 찍고 싶더라.",
                ("short_word", "ambiguous", "sky", "moon", "night"),
                "달",
                "moon",
            ),
            (
                "이번 달 식비가 생각보다 많이 나왔어.",
                ("short_word", "ambiguous", "time", "month", "calendar"),
                "달",
                "month",
            ),
            (
                "벽에 선반 달아서 책 올려두니까 방이 좀 정리됐어.",
                ("short_word", "ambiguous", "attach", "mount", "object", "home"),
                "달다",
                "attach",
            ),
            (
                "그 케이크 너무 달아서 커피가 꼭 필요했어.",
                ("short_word", "ambiguous", "taste", "sweet", "food"),
                "달다",
                "sweet",
            ),
            (
                "감기약 먹었는데도 목이 아직 칼칼해.",
                ("short_word", "ambiguous", "medicine", "health", "body"),
                "약",
                "medicine",
            ),
            (
                "주말 약속이 갑자기 취소돼서 솔직히 조금 좋았어.",
                ("short_word", "ambiguous", "appointment", "promise", "social"),
                "약속",
                "appointment",
            ),
            (
                "대략 약 10분 정도 걸릴 것 같아.",
                ("short_word", "ambiguous", "approximate", "number", "estimate"),
                "약",
                "approximate",
            ),
            (
                "요즘 잔병이 많아서 컨디션 관리가 필요해.",
                ("short_word", "ambiguous", "illness", "health", "body"),
                "병",
                "illness",
            ),
            (
                "책상 위 물병을 또 어디다 뒀는지 모르겠어.",
                ("short_word", "ambiguous", "bottle", "container", "drink"),
                "병",
                "bottle",
            ),
            (
                "목이 계속 아파서 병원 가야 하나 고민돼.",
                ("short_word", "ambiguous", "hospital", "health", "place"),
                "병원",
                "hospital",
            ),
            (
                "오늘 첫 끼니라서 밥을 든든하게 먹고 싶어.",
                ("short_word", "ambiguous", "food", "rice", "meal"),
                "밥",
                "meal",
            ),
            (
                "요즘 밥값이 너무 올라서 점심값이 부담돼.",
                ("short_word", "ambiguous", "money", "meal_cost", "spending"),
                "밥값",
                "meal_cost",
            ),
            (
                "이 일도 결국 먹고살기 위한 밥벌이잖아.",
                ("short_word", "ambiguous", "work", "living", "money"),
                "밥벌이",
                "living",
            ),
            (
                "오늘은 혼술하면서 맥주 한 캔만 마시고 싶어.",
                ("short_word", "ambiguous", "alcohol", "drink", "social"),
                "술",
                "alcohol",
            ),
            (
                "회식 술자리는 가기 전부터 기가 빨려.",
                ("short_word", "ambiguous", "drinking_gathering", "social", "relationship"),
                "술자리",
                "drinking_gathering",
            ),
            (
                "이건 손기술이 좋아야 깔끔하게 만들 수 있겠다.",
                ("short_word", "ambiguous", "skill", "technique", "ability"),
                "기술",
                "technique",
            ),
            (
                "자기 전에 일기를 쓰다 보면 생각이 정리돼.",
                ("short_word", "ambiguous", "writing", "text", "language"),
                "쓰다",
                "writing",
            ),
            (
                "이 도구는 캠핑 갈 때 써먹기 좋겠다.",
                ("short_word", "ambiguous", "use", "tool", "action"),
                "쓰다",
                "use",
            ),
            (
                "아메리카노가 너무 써서 물을 더 탔어.",
                ("short_word", "ambiguous", "taste", "bitter", "food"),
                "쓰다",
                "bitter",
            ),
            (
                "미세먼지 심해서 마스크 쓰고 나가야겠어.",
                ("short_word", "ambiguous", "wear", "head", "object"),
                "쓰다",
                "wear",
            ),
            (
                "주말 내내 드라마 보고 유튜브만 봤어.",
                ("short_word", "ambiguous", "see", "watch", "media"),
                "보다",
                "watch",
            ),
            (
                "치킨보다 피자가 오늘은 더 나을 것 같아.",
                ("short_word", "ambiguous", "compare", "than", "judgement"),
                "보다",
                "compare",
            ),
            (
                "새로 생긴 카페 한번 가보고 싶어.",
                ("short_word", "ambiguous", "try", "experience", "action"),
                "보다",
                "try",
            ),
            (
                "찬물로 샤워했더니 손발이 너무 차가워.",
                ("short_word", "ambiguous", "cold", "temperature", "weather"),
                "차다",
                "cold",
            ),
            (
                "엘리베이터가 사람으로 꽉 차서 못 탔어.",
                ("short_word", "ambiguous", "full", "filled", "state"),
                "차다",
                "full",
            ),
            (
                "화나서 괜히 의자 다리를 발로 차버렸어.",
                ("short_word", "ambiguous", "kick", "body_action", "motion"),
                "차다",
                "kick",
            ),
            (
                "면접 볼 때 시계를 차고 가는 게 나을까?",
                ("short_word", "ambiguous", "wear", "accessory", "body"),
                "차다",
                "accessory",
            ),
            (
                "그 말은 맞는 말이라 반박하기 어렵더라.",
                ("short_word", "ambiguous", "correct", "answer", "judgement"),
                "맞다",
                "correct",
            ),
            (
                "새로 산 셔츠 사이즈가 딱 맞아서 기분 좋아.",
                ("short_word", "ambiguous", "fit", "clothes", "size"),
                "맞다",
                "fit",
            ),
            (
                "비 맞고 들어왔더니 머리가 다 젖었어.",
                ("short_word", "ambiguous", "hit", "impact", "pain"),
                "맞다",
                "hit",
            ),
            (
                "렌즈 끼고 오래 있었더니 눈이 뻑뻑해.",
                ("short_word", "ambiguous", "wear", "accessory", "body"),
                "끼다",
                "wear",
            ),
            (
                "출근길 지하철에 끼어서 숨도 못 쉬겠더라.",
                ("short_word", "ambiguous", "stuck", "crowded", "commute"),
                "끼다",
                "stuck",
            ),
            (
                "아침 첫 끼를 놓치면 하루 종일 힘이 없어.",
                ("short_word", "ambiguous", "meal", "routine", "food"),
                "끼니",
                "meal",
            ),
            (
                "엑셀 빈칸을 다 채우는 게 생각보다 오래 걸려.",
                ("short_word", "ambiguous", "space", "compartment", "place"),
                "칸",
                "space",
            ),
            (
                "이번 웹툰은 컷 전환이 좋아서 한 컷마다 힘이 있더라.",
                ("short_word", "ambiguous", "comic_panel", "frame", "media"),
                "칸",
                "comic_panel",
            ),
            (
                "모르는 번호로 전화를 걸어야 해서 괜히 긴장돼.",
                ("short_word", "ambiguous", "call", "phone", "contact"),
                "걸다",
                "call",
            ),
            (
                "코트를 옷걸이에 걸어두고 바로 누웠어.",
                ("short_word", "ambiguous", "hang", "object", "home"),
                "걸다",
                "hang",
            ),
            (
                "감기 걸린 듯해서 오늘은 일찍 쉬려고.",
                ("short_word", "ambiguous", "catch", "illness", "health"),
                "걸리다",
                "catch",
            ),
            (
                "퇴근길 버스가 오래 걸려서 지쳤어.",
                ("short_word", "ambiguous", "time_taken", "delay", "waiting"),
                "걸리다",
                "time_taken",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_core_daily_polysemy_batch_four(self) -> None:
        cases = (
            (
                "퇴근길에 꽃다발 하나 샀는데 기분이 확 좋아졌어.",
                ("short_word", "ambiguous", "flower", "nature", "pretty"),
                "꽃",
                "flower",
            ),
            (
                "이번 일만 잘 넘기면 앞으로 꽃길만 걸었으면 좋겠다.",
                ("short_word", "ambiguous", "good_future", "success", "wish"),
                "꽃길",
                "success",
            ),
            (
                "화분 잎이 축 처져서 물을 줘야 할 것 같아.",
                ("short_word", "ambiguous", "leaf", "plant", "nature"),
                "잎",
                "leaf",
            ),
            (
                "이 문제의 뿌리가 뭔지부터 찾아야 할 것 같아.",
                ("short_word", "ambiguous", "root", "plant", "origin"),
                "뿌리",
                "origin",
            ),
            (
                "책상에 부딪혀서 다리에 멍이 들었어.",
                ("short_word", "ambiguous", "bruise", "body", "pain"),
                "멍",
                "bruise",
            ),
            (
                "오늘은 아무 생각 없이 창밖 보면서 멍때리고 싶어.",
                ("short_word", "ambiguous", "spacing_out", "blank_mind", "rest"),
                "멍때림",
                "spacing_out",
            ),
            (
                "그 옷 스타일 진짜 멋있더라.",
                ("short_word", "ambiguous", "cool", "style", "charm"),
                "멋",
                "style",
            ),
            (
                "친구가 내 물건을 제멋대로 써서 좀 짜증났어.",
                ("short_word", "ambiguous", "arbitrary", "selfish", "annoyance"),
                "멋대로",
                "selfish",
            ),
            (
                "요즘 점심값이 너무 올라서 밥값 계산할 때마다 놀라.",
                ("short_word", "ambiguous", "price", "money", "cost"),
                "값",
                "price",
            ),
            (
                "비싸긴 했는데 이 정도면 돈값은 하는 것 같아.",
                ("short_word", "ambiguous", "value", "worth", "judgement"),
                "값어치",
                "worth",
            ),
            (
                "방바닥에 빨래가 쌓여 있어서 걸어다니기도 힘들어.",
                ("short_word", "ambiguous", "floor", "place", "home"),
                "바닥",
                "floor",
            ),
            (
                "오늘은 체력이 완전 바닥이라 아무것도 못 하겠어.",
                ("short_word", "ambiguous", "rock_bottom", "low_state", "emotion"),
                "바닥",
                "rock_bottom",
            ),
            (
                "방 벽지가 오래돼서 분위기가 너무 칙칙해.",
                ("short_word", "ambiguous", "wall", "place", "home"),
                "벽",
                "wall",
            ),
            (
                "공부하다가 한계의 벽에 막힌 느낌이 들었어.",
                ("short_word", "ambiguous", "obstacle", "limit", "frustration"),
                "벽",
                "obstacle",
            ),
            (
                "오늘은 집 밖으로 한 발짝도 나가기 싫어.",
                ("short_word", "ambiguous", "outside", "outing", "place"),
                "밖",
                "outside",
            ),
            (
                "눈앞에 버스가 지나가서 그대로 놓쳤어.",
                ("short_word", "ambiguous", "front", "position", "place"),
                "앞",
                "front",
            ),
            (
                "요즘 내 앞날이 어떻게 될지 괜히 걱정돼.",
                ("short_word", "ambiguous", "future", "life", "worry"),
                "앞날",
                "future",
            ),
            (
                "뒤쪽 자리에 앉으니까 화면이 잘 안 보였어.",
                ("short_word", "ambiguous", "behind", "back_position", "place"),
                "뒤",
                "behind",
            ),
            (
                "난 뒤끝 없는 줄 알았는데 계속 생각나더라.",
                ("short_word", "ambiguous", "grudge", "lingering_feeling", "relationship"),
                "뒤끝",
                "grudge",
            ),
            (
                "머리 위에 조명이 너무 밝아서 눈이 아팠어.",
                ("short_word", "ambiguous", "above", "top", "position"),
                "위",
                "above",
            ),
            (
                "매운 거 먹었더니 위가 아파서 죽겠어.",
                ("short_word", "ambiguous", "stomach_organ", "health", "body"),
                "위",
                "stomach_organ",
            ),
            (
                "책상 밑에 이어폰이 굴러들어갔어.",
                ("short_word", "ambiguous", "under", "bottom", "position"),
                "밑",
                "under",
            ),
            (
                "안경에 김 서려서 앞이 하나도 안 보여.",
                ("short_word", "ambiguous", "steam", "fog", "sensory"),
                "김",
                "steam",
            ),
            (
                "김밥에 김이 너무 질겨서 먹기 힘들었어.",
                ("short_word", "ambiguous", "seaweed", "food", "side_dish"),
                "김",
                "seaweed",
            ),
            (
                "소금빵 냄새 맡자마자 그냥 지나칠 수가 없었어.",
                ("short_word", "ambiguous", "bread", "food", "snack"),
                "빵",
                "bread",
            ),
            (
                "그 말 듣고 진짜 빵 터져서 한참 웃었어.",
                ("short_word", "ambiguous", "burst_sound", "joke", "reaction"),
                "빵",
                "burst_sound",
            ),
            (
                "사람이 너무 많아서 숨이 막히는 줄 알았어.",
                ("short_word", "ambiguous", "breath", "body", "pressure"),
                "숨",
                "breath",
            ),
            (
                "무서워서 문 뒤에 숨어 있다가 들킬 뻔했어.",
                ("short_word", "ambiguous", "hide", "avoid", "action"),
                "숨",
                "hide",
            ),
            (
                "버스 놓칠까 봐 뛰었더니 땀이 엄청 났어.",
                ("short_word", "ambiguous", "sweat", "body", "heat"),
                "땀",
                "sweat",
            ),
            (
                "이건 진짜 땀 흘려 노력한 티가 나더라.",
                ("short_word", "ambiguous", "effort", "hard_work", "labor"),
                "땀",
                "effort",
            ),
            (
                "가방을 손에 들고 뛰려니까 너무 불편했어.",
                ("short_word", "ambiguous", "hold", "carry", "object"),
                "들다",
                "hold",
            ),
            (
                "이사 비용이 생각보다 많이 들어서 당황했어.",
                ("short_word", "ambiguous", "cost", "money", "expense"),
                "들다",
                "cost",
            ),
            (
                "문득 그런 생각이 들어서 잠이 안 왔어.",
                ("short_word", "ambiguous", "thought", "feeling", "mind"),
                "들다",
                "thought",
            ),
            (
                "넘어질 뻔해서 난간을 겨우 붙잡았어.",
                ("short_word", "ambiguous", "catch", "grab", "action"),
                "잡다",
                "grab",
            ),
            (
                "친구랑 다음 주 약속을 잡아야 해.",
                ("short_word", "ambiguous", "schedule", "appointment", "plan"),
                "잡다",
                "schedule",
            ),
            (
                "이제 방향을 잡고 하나씩 해보면 될 것 같아.",
                ("short_word", "ambiguous", "focus", "direction", "plan"),
                "잡다",
                "direction",
            ),
            (
                "화나서 책상을 치고 싶었는데 겨우 참았어.",
                ("short_word", "ambiguous", "hit", "impact", "action"),
                "치다",
                "hit",
            ),
            (
                "키보드로 타자 치는 소리가 너무 크게 들려.",
                ("short_word", "ambiguous", "typing", "keyboard", "text"),
                "치다",
                "typing",
            ),
            (
                "어릴 때 피아노 치는 걸 배워보고 싶었어.",
                ("short_word", "ambiguous", "play_music", "instrument", "hobby"),
                "치다",
                "play_music",
            ),
            (
                "샷 빼고 달달하게 해달라고 말하면 이상할까?",
                ("short_word", "ambiguous", "remove", "exclude", "action"),
                "빼다",
                "remove",
            ),
            (
                "이번 달에는 진짜 살 빼려고 운동 시작했어.",
                ("short_word", "ambiguous", "lose_weight", "diet", "body"),
                "빼다",
                "lose_weight",
            ),
            (
                "라면에 계란을 넣어 먹으면 훨씬 든든해.",
                ("short_word", "ambiguous", "put_in", "add", "action"),
                "넣다",
                "add",
            ),
            (
                "가방을 의자 위에 올려놓고 그냥 나왔어.",
                ("short_word", "ambiguous", "put_down", "place_object", "action"),
                "놓다",
                "put_down",
            ),
            (
                "타이밍을 놓쳐서 말을 못 꺼냈어.",
                ("short_word", "ambiguous", "miss", "lose_chance", "regret"),
                "놓치다",
                "miss",
            ),
            (
                "머릿속에서 같은 노래가 계속 맴돌아.",
                ("short_word", "ambiguous", "rotate", "go_around", "motion"),
                "돌다",
                "rotate",
            ),
            (
                "오늘 일정이 너무 많아서 진짜 돌겠어.",
                ("short_word", "ambiguous", "crazy", "dizzy", "state"),
                "돌다",
                "crazy",
            ),
            (
                "사람들이 밀어서 지하철 안쪽으로 떠밀렸어.",
                ("short_word", "ambiguous", "push", "action", "force"),
                "밀다",
                "push",
            ),
            (
                "답장이 너무 밀려서 어디부터 해야 할지 모르겠어.",
                ("short_word", "ambiguous", "backlog", "delay", "task"),
                "밀리다",
                "backlog",
            ),
            (
                "문이 안 열려서 손잡이를 세게 당겼어.",
                ("short_word", "ambiguous", "pull", "action", "force"),
                "당기다",
                "pull",
            ),
            (
                "비 오니까 갑자기 칼국수가 너무 땡겨.",
                ("short_word", "ambiguous", "craving", "food", "desire"),
                "땡기다",
                "craving",
            ),
            (
                "방문 닫고 혼자 조용히 있고 싶어.",
                ("short_word", "ambiguous", "close", "door", "action"),
                "닫다",
                "close",
            ),
            (
                "그 말 듣고 마음이 닫힌 느낌이 들었어.",
                ("short_word", "ambiguous", "blocked", "closed_off", "emotion"),
                "닫히다",
                "closed_off",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_core_daily_polysemy_batch_five(self) -> None:
        cases = (
            (
                "카페에서 유리잔에 담긴 물 한잔을 받았어.",
                ("short_word", "ambiguous", "cup", "drink", "container"),
                "잔",
                "cup",
            ),
            (
                "재고가 조금 남아있어서 잔량부터 확인해야 해.",
                ("short_word", "ambiguous", "remaining", "leftover", "amount"),
                "잔",
                "remaining",
            ),
            (
                "그 장면이 머릿속에 계속 떠올라서 잔상이 남았어.",
                ("short_word", "ambiguous", "afterimage", "memory_echo", "lingering"),
                "잔상",
                "afterimage",
            ),
            (
                "회사에 이상한 소문이 돌기 시작해서 분위기가 묘해.",
                ("short_word", "ambiguous", "rumor", "social", "talk"),
                "소문",
                "rumor",
            ),
            (
                "오랜만에 친구 근황 소식 들으니까 반갑더라.",
                ("short_word", "ambiguous", "news", "update", "contact"),
                "소식",
                "news",
            ),
            (
                "뭔가 이상한 느낌이 들어서 수상한 낌새를 챘어.",
                ("short_word", "ambiguous", "sign", "hint", "suspicion"),
                "낌새",
                "sign",
            ),
            (
                "둘이 단둘이 있으니까 너무 어색하고 불편했어.",
                ("short_word", "ambiguous", "discomfort", "awkward", "inconvenience"),
                "불편",
                "discomfort",
            ),
            (
                "자기 전에 화면 끄고 알람 끄고 누웠어.",
                ("short_word", "ambiguous", "turn_off", "device", "light"),
                "끄다",
                "turn_off",
            ),
            (
                "캠핑장에서 모닥불을 완전히 꺼야 해서 물을 부었어.",
                ("short_word", "ambiguous", "extinguish", "fire", "safety"),
                "끄다",
                "extinguish",
            ),
            (
                "방이 어두워서 불 켜고 컴퓨터 켰어.",
                ("short_word", "ambiguous", "turn_on", "device", "light"),
                "켜다",
                "turn_on",
            ),
            (
                "밥 먹으면서 유튜브 켜고 영상 보려고 했어.",
                ("short_word", "ambiguous", "open_app", "play_media", "digital"),
                "켜다",
                "open_app",
            ),
            (
                "노트북에 스티커를 붙였더니 분위기가 좀 살았어.",
                ("short_word", "ambiguous", "attached", "stick", "object"),
                "붙다",
                "attached",
            ),
            (
                "면접 붙었다는 연락 받고 심장이 터지는 줄 알았어.",
                ("short_word", "ambiguous", "pass_exam", "success", "result"),
                "붙다",
                "pass_exam",
            ),
            (
                "친구가 하루 종일 찰싹 붙어다녀서 웃기긴 했어.",
                ("short_word", "ambiguous", "cling", "social", "close"),
                "붙다",
                "cling",
            ),
            (
                "폰을 바닥에 떨어뜨려서 액정 깨질 뻔했어.",
                ("short_word", "ambiguous", "fall_drop", "drop", "object"),
                "떨어지다",
                "fall_drop",
            ),
            (
                "퇴근길에 배터리 떨어져서 길 찾기 못 할 뻔했어.",
                ("short_word", "ambiguous", "run_out", "shortage", "depleted"),
                "떨어지다",
                "run_out",
            ),
            (
                "시험 떨어졌다는 결과 보고 하루 종일 멍했어.",
                ("short_word", "ambiguous", "fail", "rejection", "result"),
                "떨어지다",
                "fail",
            ),
            (
                "그 드립 듣고 웃음이 터져서 한참 웃었어.",
                ("short_word", "ambiguous", "burst_laugh", "laughter", "reaction"),
                "터지다",
                "burst_laugh",
            ),
            (
                "퇴근 5분 전에 문제가 터져서 다들 멘탈 나갔어.",
                ("short_word", "ambiguous", "explode_problem", "incident", "crisis"),
                "터지다",
                "explode_problem",
            ),
            (
                "퇴근길에 차가 너무 막혀서 버스에서 녹아내렸어.",
                ("short_word", "ambiguous", "blocked", "traffic", "route"),
                "막히다",
                "traffic",
            ),
            (
                "너무 어이없어서 말문이 막히고 할 말이 없었어.",
                ("short_word", "ambiguous", "speechless", "stunned", "reaction"),
                "막히다",
                "speechless",
            ),
            (
                "친구랑 얘기하고 나니까 마음이 풀려서 좀 편해졌어.",
                ("short_word", "ambiguous", "solved", "relief", "release"),
                "풀리다",
                "relief",
            ),
            (
                "걷다가 신발끈이 풀려서 중간에 멈춰 섰어.",
                ("short_word", "ambiguous", "undone", "loosened", "object"),
                "풀리다",
                "undone",
            ),
            (
                "주말엔 외출해서 산책이라도 하고 싶어.",
                ("short_word", "ambiguous", "go_out", "outing", "outside"),
                "나가다",
                "go_out",
            ),
            (
                "월급 들어오자마자 카드값으로 돈이 다 나갔어.",
                ("short_word", "ambiguous", "money_outflow", "spending", "vanish"),
                "나가다",
                "money_outflow",
            ),
            (
                "요즘 물가가 너무 올라서 점심값도 부담돼.",
                ("short_word", "ambiguous", "price_up", "increase", "money"),
                "오르다",
                "price_up",
            ),
            (
                "주말에 산 오르는 길이 생각보다 너무 힘들었어.",
                ("short_word", "ambiguous", "climb", "go_up", "movement"),
                "오르다",
                "climb",
            ),
            (
                "지하철 내릴 역을 지나쳐서 다시 돌아왔어.",
                ("short_word", "ambiguous", "get_off", "transit", "commute"),
                "내리다",
                "get_off",
            ),
            (
                "밤새 비가 내리는 소리 들으면서 잠들었어.",
                ("short_word", "ambiguous", "precipitation", "rain_snow", "weather"),
                "내리다",
                "precipitation",
            ),
            (
                "배달 온 국이 다 식어서 조금 아쉬웠어.",
                ("short_word", "ambiguous", "cool_down", "food_temp", "food"),
                "식다",
                "cool_down",
            ),
            (
                "그 일 이후로 마음이 식어버린 느낌이야.",
                ("short_word", "ambiguous", "feelings_cool", "interest_fade", "relationship"),
                "식다",
                "feelings_cool",
            ),
            (
                "갑자기 화면에 이상한 팝업이 떠서 당황했어.",
                ("short_word", "ambiguous", "appear", "popup", "digital"),
                "뜨다",
                "appear",
            ),
            (
                "아침에 눈 뜨자마자 알람부터 확인했어.",
                ("short_word", "ambiguous", "wake_open_eyes", "wake", "sleep"),
                "뜨다",
                "wake_open_eyes",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_core_daily_polysemy_batch_six(self) -> None:
        cases = (
            (
                "퇴근길 노을이 예뻐서 사진 찍고 한참 봤어.",
                ("short_word", "ambiguous", "photo", "camera", "record"),
                "찍다",
                "photo",
            ),
            (
                "점심 메뉴 하나만 찍어줘, 나 진짜 못 고르겠어.",
                ("short_word", "ambiguous", "choose", "pick", "mark"),
                "찍다",
                "choose",
            ),
            (
                "탕수육은 소스 찍어 먹어야 바삭함이 살아.",
                ("short_word", "ambiguous", "dip_sauce", "food", "taste"),
                "찍다",
                "dip_sauce",
            ),
            (
                "에어컨 필터 갈아야 하는데 계속 미루고 있어.",
                ("short_word", "ambiguous", "replace", "maintenance", "device"),
                "갈다",
                "replace",
            ),
            (
                "아침마다 원두 갈아서 커피 내려 마시면 기분 좋더라.",
                ("short_word", "ambiguous", "grind", "food_prep", "cooking"),
                "갈다",
                "grind",
            ),
            (
                "그 말 듣고 너무 분해서 이를 갈았어.",
                ("short_word", "ambiguous", "resentment", "anger", "grudge"),
                "갈다",
                "resentment",
            ),
            (
                "빈칸 채워 넣는 문제만 나오면 이상하게 긴장돼.",
                ("short_word", "ambiguous", "fill", "empty_space", "amount"),
                "채우다",
                "fill",
            ),
            (
                "폰 배터리 채우려고 충전기부터 찾았어.",
                ("short_word", "ambiguous", "charge", "battery", "device"),
                "채우다",
                "charge",
            ),
            (
                "편의점에서 삼각김밥으로 허기 채우고 왔어.",
                ("short_word", "ambiguous", "satisfy_hunger", "hunger", "food"),
                "채우다",
                "satisfy_hunger",
            ),
            (
                "주말에 방을 싹 비워서 공간을 좀 만들었어.",
                ("short_word", "ambiguous", "empty", "clear_space", "cleanup"),
                "비우다",
                "empty",
            ),
            (
                "오늘은 생각 비우고 멍때리고 싶어.",
                ("short_word", "ambiguous", "clear_mind", "rest", "mental"),
                "비우다",
                "clear_mind",
            ),
            (
                "종이 찢고 다시 쓰려니까 괜히 손만 바빠졌어.",
                ("short_word", "ambiguous", "tear", "paper_clothes", "object"),
                "찢다",
                "tear",
            ),
            (
                "오늘 무대는 진짜 찢었다고 해도 될 정도였어.",
                ("short_word", "ambiguous", "slay", "impressive", "praise"),
                "찢다",
                "slay",
            ),
            (
                "이번 장비 가격 선 넘어서 지갑 찢어지는 줄 알았어.",
                ("short_word", "ambiguous", "wallet_pain", "spending", "money"),
                "찢다",
                "wallet_pain",
            ),
            (
                "엘리베이터 버튼 누르고 멍하니 기다렸어.",
                ("short_word", "ambiguous", "press_button", "button", "device"),
                "누르다",
                "press_button",
            ),
            (
                "화나는 감정을 꾹 참고 눌러 담았어.",
                ("short_word", "ambiguous", "suppress", "emotion_control", "hold_back"),
                "누르다",
                "suppress",
            ),
            (
                "어제 가위눌려서 몸이 안 움직이는 느낌이었어.",
                ("short_word", "ambiguous", "sleep_paralysis", "fear", "body"),
                "눌리다",
                "sleep_paralysis",
            ),
            (
                "가방에 눌려서 과자가 다 부서졌어.",
                ("short_word", "ambiguous", "pressed", "pressure", "object"),
                "눌리다",
                "pressed",
            ),
            (
                "빨래 개면서 옷 접는 게 은근 귀찮아.",
                ("short_word", "ambiguous", "fold", "clothes_paper", "object"),
                "접다",
                "fold",
            ),
            (
                "이 취미는 돈이 너무 들어서 접을까 고민 중이야.",
                ("short_word", "ambiguous", "give_up", "quit", "decision"),
                "접다",
                "give_up",
            ),
            (
                "비 오기 전에 우산 펴고 나가서 다행이었어.",
                ("short_word", "ambiguous", "unfold", "open_object", "object"),
                "펴다",
                "unfold",
            ),
            (
                "하루 종일 앉아 있었더니 허리 펴는 순간 살 것 같아.",
                ("short_word", "ambiguous", "relax", "straighten", "body"),
                "펴다",
                "relax",
            ),
            (
                "김치전 뒤집다가 반으로 찢어져서 마음 아팠어.",
                ("short_word", "ambiguous", "flip", "cooking", "object"),
                "뒤집다",
                "flip",
            ),
            (
                "막판 한타로 판을 뒤집어서 역전승했어.",
                ("short_word", "ambiguous", "reverse_result", "turnaround", "game"),
                "뒤집다",
                "reverse_result",
            ),
            (
                "웹툰 보다가 재미없는 회차는 그냥 스킵해서 넘겼어.",
                ("short_word", "ambiguous", "turn_page", "skip", "media"),
                "넘기다",
                "turn_page",
            ),
            (
                "친구 실수라서 이번엔 그냥 넘어가주기로 했어.",
                ("short_word", "ambiguous", "let_pass", "tolerate", "social"),
                "넘기다",
                "let_pass",
            ),
            (
                "고기 씹다가 볼 안쪽을 깨물어서 아팠어.",
                ("short_word", "ambiguous", "chew", "eat", "mouth"),
                "씹다",
                "chew",
            ),
            (
                "카톡 읽씹 당하니까 괜히 신경 쓰이더라.",
                ("short_word", "ambiguous", "ignore_message", "chat", "social"),
                "씹다",
                "ignore_message",
            ),
            (
                "약이 너무 써서 물이랑 꿀꺽 삼켰어.",
                ("short_word", "ambiguous", "swallow", "eat", "body"),
                "삼키다",
                "swallow",
            ),
            (
                "하고 싶은 말이 많았는데 그냥 말을 삼켰어.",
                ("short_word", "ambiguous", "hold_back_words", "emotion", "restraint"),
                "삼키다",
                "hold_back_words",
            ),
            (
                "친구한테 직접 묻고 싶었는데 타이밍을 놓쳤어.",
                ("short_word", "ambiguous", "ask", "question", "conversation"),
                "묻다",
                "ask",
            ),
            (
                "흰 셔츠에 커피 얼룩이 묻어서 하루 종일 신경 쓰였어.",
                ("short_word", "ambiguous", "stain", "attached_dirt", "object"),
                "묻다",
                "stain",
            ),
            (
                "타임캡슐을 땅에 묻어두면 나중에 재밌을 것 같아.",
                ("short_word", "ambiguous", "bury", "hide", "ground"),
                "묻다",
                "bury",
            ),
            (
                "외출 전에 선크림 바르는 걸 자꾸 까먹어.",
                ("short_word", "ambiguous", "apply", "skin", "beauty"),
                "바르다",
                "apply",
            ),
            (
                "식빵에 버터 발라 먹으면 단순한데 맛있어.",
                ("short_word", "ambiguous", "spread_sauce", "food", "taste"),
                "바르다",
                "spread_sauce",
            ),
            (
                "라면 국물이 튀어서 흰옷에 점처럼 묻었어.",
                ("short_word", "ambiguous", "splatter", "food_accident", "stain"),
                "튀다",
                "splatter",
            ),
            (
                "그 사람은 옷 색이 너무 튀어서 멀리서도 눈에 띄었어.",
                ("short_word", "ambiguous", "stand_out", "noticeable", "visual"),
                "튀다",
                "stand_out",
            ),
            (
                "책상 위에 커피 흘려서 키보드까지 젖었어.",
                ("short_word", "ambiguous", "spill", "liquid", "accident"),
                "흘리다",
                "spill",
            ),
            (
                "회의 전에 정보를 슬쩍 흘린 사람이 있는 것 같아.",
                ("short_word", "ambiguous", "leak_secret", "careless_talk", "social"),
                "흘리다",
                "leak_secret",
            ),
            (
                "비가 새서 창틀 밑에 물이 고였어.",
                ("short_word", "ambiguous", "leak", "water", "home"),
                "새다",
                "leak",
            ),
            (
                "어제 게임하다가 밤새서 오늘 완전 멍해.",
                ("short_word", "ambiguous", "stay_up_all_night", "sleep_loss", "night"),
                "새다",
                "stay_up_all_night",
            ),
            (
                "밀린 일이 잔뜩 쌓여서 어디서부터 해야 할지 모르겠어.",
                ("short_word", "ambiguous", "pile_up", "backlog", "task"),
                "쌓이다",
                "pile_up",
            ),
            (
                "말 안 하고 참다 보니 불만이 계속 쌓인 것 같아.",
                ("short_word", "ambiguous", "emotion_accumulate", "resentment", "stress"),
                "쌓이다",
                "emotion_accumulate",
            ),
            (
                "아침에 잠에서 깨자마자 폰부터 봤어.",
                ("short_word", "ambiguous", "wake_up", "sleep", "morning"),
                "깨다",
                "wake_up",
            ),
            (
                "설거지하다가 접시 깨먹어서 조용히 치웠어.",
                ("short_word", "ambiguous", "break_object", "break", "accident"),
                "깨다",
                "break_object",
            ),
            (
                "술이 깨고 나니까 어제 보낸 카톡이 무서워졌어.",
                ("short_word", "ambiguous", "sober_up", "recover", "alcohol"),
                "깨다",
                "sober_up",
            ),
            (
                "폰 액정 깨져서 터치할 때마다 마음이 아파.",
                ("short_word", "ambiguous", "broken", "object", "damage"),
                "깨지다",
                "broken",
            ),
            (
                "그 일 이후로 관계가 깨질까 봐 걱정돼.",
                ("short_word", "ambiguous", "relationship_break", "breakup", "social"),
                "깨지다",
                "relationship_break",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_core_daily_polysemy_batch_seven(self) -> None:
        cases = (
            (
                "오늘 점심에 김밥 먹고 커피까지 마셨어.",
                ("short_word", "ambiguous", "eat", "food", "meal"),
                "먹다",
                "eat",
            ),
            (
                "감기약 먹고 좀 누워 있어야겠다.",
                ("short_word", "ambiguous", "take_medicine", "medicine", "health"),
                "먹다",
                "take_medicine",
            ),
            (
                "회의에서 한소리 먹고 멘탈이 살짝 나갔어.",
                ("short_word", "ambiguous", "criticism", "scolded", "social"),
                "먹다",
                "criticism",
            ),
            (
                "나이 먹을수록 밤새는 게 힘들어.",
                ("short_word", "ambiguous", "age", "time", "life_stage"),
                "먹다",
                "age",
            ),
            (
                "친구한테 생일 선물 받았어.",
                ("short_word", "ambiguous", "receive", "gift", "message"),
                "받다",
                "receive",
            ),
            (
                "요즘 상담받으면서 생각 정리하는 중이야.",
                ("short_word", "ambiguous", "service", "help", "care"),
                "받다",
                "service",
            ),
            (
                "그 말 듣고 상처받아서 하루 종일 조용했어.",
                ("short_word", "ambiguous", "hurt", "stress", "impact"),
                "받다",
                "hurt",
            ),
            (
                "세탁기 돌리고 나니까 집안일 한 것 같아.",
                ("short_word", "ambiguous", "run_device", "machine", "routine"),
                "돌리다",
                "run_device",
            ),
            (
                "의자를 한 바퀴 돌려서 방향을 바꿨어.",
                ("short_word", "ambiguous", "rotate", "motion", "object"),
                "돌리다",
                "rotate",
            ),
            (
                "자꾸 책임을 남한테 돌리는 사람이 제일 싫어.",
                ("short_word", "ambiguous", "shift_blame", "topic_change", "social"),
                "돌리다",
                "shift_blame",
            ),
            (
                "과제 내야 하는데 아직 첫 장도 못 썼어.",
                ("short_word", "ambiguous", "submit", "pay", "school_work"),
                "내다",
                "submit",
            ),
            (
                "회의에서 의견 내는 게 은근 긴장돼.",
                ("short_word", "ambiguous", "express", "sound", "idea"),
                "내다",
                "express",
            ),
            (
                "갑자기 화내서 분위기가 싸해졌어.",
                ("short_word", "ambiguous", "anger", "emotion_expression", "reaction"),
                "화내다",
                "anger",
            ),
            (
                "오늘 회사에서 큰일 났어.",
                ("short_word", "ambiguous", "happen", "incident", "event"),
                "나다",
                "happen",
            ),
            (
                "냉장고에서 이상한 냄새가 나.",
                ("short_word", "ambiguous", "sensory", "smell_sound", "notice"),
                "나다",
                "sensory",
            ),
            (
                "마스크 오래 썼더니 피부에 여드름 났어.",
                ("short_word", "ambiguous", "body", "skin", "symptom"),
                "나다",
                "body",
            ),
            (
                "치킨은 반씩 나눠 먹어야 싸움이 안 나.",
                ("short_word", "ambiguous", "share", "split", "food"),
                "나누다",
                "share",
            ),
            (
                "친구랑 오랜만에 깊은 얘기를 나눴어.",
                ("short_word", "ambiguous", "conversation", "talk", "relationship"),
                "나누다",
                "conversation",
            ),
            (
                "퀴즈 정답 맞췄을 때 은근 기분 좋더라.",
                ("short_word", "ambiguous", "guess", "correct", "answer"),
                "맞추다",
                "guess",
            ),
            (
                "약속 시간을 서로 맞춰서 다시 잡았어.",
                ("short_word", "ambiguous", "align", "adjust", "schedule"),
                "맞추다",
                "align",
            ),
            (
                "내 취향에 맞춘 커스텀 키보드 갖고 싶어.",
                ("short_word", "ambiguous", "customize", "fit", "personalize"),
                "맞추다",
                "customize",
            ),
            (
                "지갑 잃어버려서 하루 종일 정신없었어.",
                ("short_word", "ambiguous", "lost_object", "forget", "belongings"),
                "잃다",
                "lost_object",
            ),
            (
                "요즘 의욕을 잃어서 뭐든 하기 싫어.",
                ("short_word", "ambiguous", "lose_confidence", "motivation", "emotion"),
                "잃다",
                "lose_confidence",
            ),
            (
                "낯선 동네에서 길을 잃고 한참 헤맸어.",
                ("short_word", "ambiguous", "lost_way", "route", "confusion"),
                "잃다",
                "lost_way",
            ),
            (
                "잃어버린 이어폰 찾으려고 방을 다 뒤졌어.",
                ("short_word", "ambiguous", "search", "find", "object"),
                "찾다",
                "search",
            ),
            (
                "주말에 단골 카페 찾아가서 쉬고 왔어.",
                ("short_word", "ambiguous", "visit", "go_to", "place"),
                "찾다",
                "visit",
            ),
            (
                "며칠 쉬니까 드디어 페이스를 찾은 것 같아.",
                ("short_word", "ambiguous", "regain", "recover", "state"),
                "찾다",
                "regain",
            ),
            (
                "인스타에 사진 올렸더니 친구들이 바로 반응했어.",
                ("short_word", "ambiguous", "upload", "post", "sns"),
                "올리다",
                "upload",
            ),
            (
                "노래가 좋아서 볼륨을 올렸어.",
                ("short_word", "ambiguous", "increase", "raise", "level"),
                "올리다",
                "increase",
            ),
            (
                "택배 상자를 책상 위에 올려놓고 까먹었어.",
                ("short_word", "ambiguous", "put_on", "place_object", "action"),
                "올리다",
                "put_on",
            ),
            (
                "빙판길에서 넘어져서 무릎이 아파.",
                ("short_word", "ambiguous", "fall_down", "body", "accident"),
                "넘어지다",
                "fall_down",
            ),
            (
                "다음 화로 넘어가고 싶은데 광고가 너무 길어.",
                ("short_word", "ambiguous", "move_next", "transition", "media"),
                "넘어가다",
                "move_next",
            ),
            (
                "친구 말에 속아 넘어가서 쓸데없는 걸 샀어.",
                ("short_word", "ambiguous", "fooled", "persuaded", "social"),
                "넘어가다",
                "fooled",
            ),
            (
                "중고로 올린 의자가 바로 팔렸어.",
                ("short_word", "ambiguous", "sold", "marketplace", "shopping"),
                "팔리다",
                "sold",
            ),
            (
                "사람들 앞에서 넘어져서 너무 쪽팔렸어.",
                ("short_word", "ambiguous", "embarrassment", "shame", "social"),
                "팔리다",
                "embarrassment",
            ),
            (
                "그 방법이 먹혀들어서 분위기가 좀 풀렸어.",
                ("short_word", "ambiguous", "effective", "works", "result"),
                "먹히다",
                "effective",
            ),
            (
                "변명은 하나도 안 먹혀서 그냥 사과했어.",
                ("short_word", "ambiguous", "not_working", "rejected", "frustration"),
                "먹히다",
                "not_working",
            ),
            (
                "주말에 밀린 빨래를 한꺼번에 빨았어.",
                ("short_word", "ambiguous", "wash_clothes", "laundry", "chores"),
                "빨다",
                "wash_clothes",
            ),
            (
                "빨대로 음료를 쪽쪽 빨아먹었어.",
                ("short_word", "ambiguous", "suck", "drink", "mouth"),
                "빨다",
                "suck",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_core_daily_polysemy_batch_eight(self) -> None:
        cases = (
            (
                "수업 시간에 딴짓하다가 선생님한테 걸렸어.",
                ("short_word", "ambiguous", "caught", "detected", "embarrassment"),
                "걸리다",
                "caught",
            ),
            (
                "목에 생선 가시가 걸려서 물만 마시는 중이야.",
                ("short_word", "ambiguous", "stuck", "blocked", "object"),
                "걸리다",
                "stuck",
            ),
            (
                "이번엔 진짜 목숨 걸고 티켓팅 해볼 거야.",
                ("short_word", "ambiguous", "bet", "risk", "commitment"),
                "걸다",
                "bet",
            ),
            (
                "처음 보는 사람한테 먼저 말 거는 게 아직 어렵더라.",
                ("short_word", "ambiguous", "start_conversation", "talk", "social"),
                "걸다",
                "start_conversation",
            ),
            (
                "수학 문제 풀다가 머리가 하얘졌어.",
                ("short_word", "ambiguous", "solve", "problem", "answer"),
                "풀다",
                "solve",
            ),
            (
                "스트레스 풀고 싶어서 매운 떡볶이를 시켰어.",
                ("short_word", "ambiguous", "release", "emotion", "stress"),
                "풀다",
                "release",
            ),
            (
                "여행 다녀와서 짐 풀기도 전에 침대에 누웠어.",
                ("short_word", "ambiguous", "unpack", "loosen", "object"),
                "풀다",
                "unpack",
            ),
            (
                "문 앞을 의자로 막아서 못 들어가게 했어.",
                ("short_word", "ambiguous", "block", "prevent", "defense"),
                "막다",
                "block",
            ),
            (
                "야식 충동을 막고 싶었는데 결국 라면 끓였어.",
                ("short_word", "ambiguous", "stop_urge", "self_control", "emotion"),
                "막다",
                "stop_urge",
            ),
            (
                "출근길 사람들한테 떠밀려서 지하철에 탔어.",
                ("short_word", "ambiguous", "pushed_back", "crowd", "movement"),
                "밀리다",
                "pushed_back",
            ),
            (
                "이번 달 카드값이 밀려서 마음이 불편해.",
                ("short_word", "ambiguous", "overdue", "late_payment", "money"),
                "밀리다",
                "overdue",
            ),
            (
                "비 오는 날이라 뜨끈한 국물이 당긴다.",
                ("short_word", "ambiguous", "craving", "food", "desire"),
                "당기다",
                "craving",
            ),
            (
                "회의 일정을 한 시간 앞당겨서 다시 잡았어.",
                ("short_word", "ambiguous", "move_earlier", "schedule", "time"),
                "당기다",
                "move_earlier",
            ),
            (
                "이번에 산 셔츠가 생각보다 마음에 들어.",
                ("short_word", "ambiguous", "like", "preference", "satisfaction"),
                "들다",
                "like",
            ),
            (
                "어제는 눕자마자 바로 잠들었어.",
                ("short_word", "ambiguous", "fall_asleep", "sleep", "night"),
                "들다",
                "fall_asleep",
            ),
            (
                "이제 그 일은 마음에서 내려놓고 싶어.",
                ("short_word", "ambiguous", "let_go", "release", "emotion"),
                "놓다",
                "let_go",
            ),
            (
                "회의 내내 정신 놓고 멍때렸어.",
                ("short_word", "ambiguous", "zone_out", "mental", "tired"),
                "정신놓다",
                "zone_out",
            ),
            (
                "요즘 새 게임에 푹 빠져 있어서 잠을 못 자.",
                ("short_word", "ambiguous", "fall_into", "addiction", "interest"),
                "빠지다",
                "fall_into",
            ),
            (
                "회의 명단에서 내 이름이 빠져서 당황했어.",
                ("short_word", "ambiguous", "excluded", "missing", "list"),
                "빠지다",
                "excluded",
            ),
            (
                "운동 시작하고 살이 조금 빠져서 기분 좋아.",
                ("short_word", "ambiguous", "lose_weight", "body", "diet"),
                "빠지다",
                "lose_weight",
            ),
            (
                "폰을 새 모델로 바꿨는데 배터리가 오래가.",
                ("short_word", "ambiguous", "change", "replace", "object"),
                "바꾸다",
                "change",
            ),
            (
                "답답해서 산책으로 기분전환 좀 하고 왔어.",
                ("short_word", "ambiguous", "change_mood", "refresh", "emotion"),
                "바꾸다",
                "change_mood",
            ),
            (
                "안 입는 옷을 싹 버렸더니 방이 넓어졌어.",
                ("short_word", "ambiguous", "throw_away", "trash", "cleanup"),
                "버리다",
                "throw_away",
            ),
            (
                "믿었던 사람이 날 버리고 떠난 느낌이야.",
                ("short_word", "ambiguous", "abandon", "relationship", "emotion"),
                "버리다",
                "abandon",
            ),
            (
                "새벽 감성에 홀려서 비싼 걸 사버렸어.",
                ("short_word", "ambiguous", "done_impulsively", "regret", "action"),
                "해버리다",
                "done_impulsively",
            ),
            (
                "화나는 걸 꾹 참고 버텼더니 머리가 아파.",
                ("short_word", "ambiguous", "endure", "patience", "emotion"),
                "참다",
                "endure",
            ),
            (
                "회의 중이라 화장실을 계속 참았어.",
                ("short_word", "ambiguous", "hold_back_body", "body", "urge"),
                "참다",
                "hold_back_body",
            ),
            (
                "오늘은 아무 이유 없이 울고 싶어.",
                ("short_word", "ambiguous", "cry", "sadness", "emotion"),
                "울다",
                "cry",
            ),
            (
                "아침 알람이 울렸는데 못 듣고 계속 잤어.",
                ("short_word", "ambiguous", "ring", "sound", "alarm"),
                "울리다",
                "ring",
            ),
            (
                "그 영화 마지막 장면이 마음을 울렸어.",
                ("short_word", "ambiguous", "move_heart", "emotion", "touching"),
                "울리다",
                "move_heart",
            ),
            (
                "친구 드립이 너무 웃겨서 배 잡고 웃었어.",
                ("short_word", "ambiguous", "laugh", "laughter", "reaction"),
                "웃다",
                "laugh",
            ),
            (
                "낯선 사람이 살짝 미소 지어줘서 기분이 풀렸어.",
                ("short_word", "ambiguous", "smile", "soft_reaction", "social"),
                "웃다",
                "smile",
            ),
            (
                "지하철에서 꾸벅 졸다가 내릴 역을 놓쳤어.",
                ("short_word", "ambiguous", "doze", "sleepy", "fatigue"),
                "졸다",
                "doze",
            ),
            (
                "오늘은 아무것도 안 하고 쉬어야 할 것 같아.",
                ("short_word", "ambiguous", "rest", "break", "recovery"),
                "쉬다",
                "rest",
            ),
            (
                "노래를 너무 불렀더니 목이 쉬었어.",
                ("short_word", "ambiguous", "hoarse", "voice", "throat"),
                "쉬다",
                "hoarse",
            ),
            (
                "주말에 친구랑 같이 놀러 가기로 했어.",
                ("short_word", "ambiguous", "hang_out", "play", "social"),
                "놀다",
                "hang_out",
            ),
            (
                "하루 종일 빈둥거리며 놀고먹었어.",
                ("short_word", "ambiguous", "idle", "slack_off", "time_waste"),
                "놀다",
                "idle",
            ),
            (
                "내일 비 온다니까 우산 챙겨야겠다.",
                ("short_word", "ambiguous", "pack", "belongings", "prepare"),
                "챙기다",
                "pack",
            ),
            (
                "친구가 세세하게 챙겨줘서 좀 감동했어.",
                ("short_word", "ambiguous", "care_for", "relationship", "support"),
                "챙기다",
                "care_for",
            ),
            (
                "가방에서 지갑을 꺼냈는데 카드가 없더라.",
                ("short_word", "ambiguous", "take_out", "object", "action"),
                "꺼내다",
                "take_out",
            ),
            (
                "그 얘기를 먼저 꺼내기가 너무 어려워.",
                ("short_word", "ambiguous", "bring_up_topic", "conversation", "social"),
                "꺼내다",
                "bring_up_topic",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")

    def test_semantic_word_bank_disambiguates_cho_and_cleanup_micro_contexts(self) -> None:
        cases = (
            (
                "초콜릿 하나 먹었더니 당이 좀 채워졌어.",
                ("short_word", "ambiguous", "food", "sweet", "chocolate", "sugar"),
                "초콜릿 당충전",
                "chocolate",
            ),
            (
                "촛불 켜놓고 멍때리니까 마음이 좀 가라앉아.",
                ("short_word", "ambiguous", "object", "candle", "mood", "light"),
                "촛불 멍",
                "candle",
            ),
            (
                "초보 운전이라 골목길만 들어가도 식은땀 나.",
                ("short_word", "ambiguous", "beginner", "driving", "fear", "car"),
                "초보운전 식은땀",
                "beginner",
            ),
            (
                "30초만 늦었어도 버스 놓칠 뻔했어.",
                ("short_word", "ambiguous", "time", "seconds", "rush", "commute"),
                "30초 지각 공포",
                "seconds",
            ),
            (
                "방 정리하다가 옛날 사진 발견했어.",
                ("short_word", "ambiguous", "organize", "cleanup", "room", "tidy"),
                "정리하다",
                "organize",
            ),
        )

        for text, desired_tags, expected_value, expected_tag in cases:
            with self.subTest(text=text):
                ranked = rank_semantic_word(
                    _text_signals(text),
                    desired_tags=desired_tags,
                    default="기본 반응",
                )

                self.assertEqual(ranked.value, expected_value)
                self.assertIn(expected_tag, ranked.matched_tags)
                self.assertEqual(ranked.source, "semantic_word_bank")


if __name__ == "__main__":
    unittest.main()
