from __future__ import annotations

import json
from pathlib import Path
import unittest

from predictive_bot.core.actions import ActionSelector
from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.engine import PredictiveEngine
from predictive_bot.core.goals import GoalManager
from predictive_bot.core.models import ActionType, TurnRecord, WeatherReport
from predictive_bot.core.policy import HierarchicalPolicy
from predictive_bot.core.renderer import ResponseRenderer
from predictive_bot.core.state import MemoryStateStore
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline
from predictive_bot.core.verifier import ResponseVerifier
from predictive_bot.core.world_model import WorldStateBuilder


class _FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(
            location=location,
            temperature_c=18.0,
            description="맑음",
            wind_kph=7.0,
        )


class _FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="14:32",
            formatted_date="2026-05-01",
            timezone_name="Asia/Seoul",
            source="offline_test_clock",
        )


class _FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        return [NewsHeadline(title="오프라인 테스트 헤드라인", source="fixture")][:limit]


def _build_draft_only_engine() -> PredictiveEngine:
    action_selector = ActionSelector(default_location=None)
    return PredictiveEngine(
        classifier=HeuristicIntentClassifier(),
        goal_manager=GoalManager(default_location=None),
        action_selector=action_selector,
        world_state_builder=WorldStateBuilder(),
        policy=HierarchicalPolicy(action_selector=action_selector),
        renderer=ResponseRenderer(
            llm_client=None,
            persona="black",
            draft_only=True,
        ),
        verifier=ResponseVerifier(),
        weather_service=_FakeWeatherService(),
        time_service=_FakeTimeService(),
        news_service=_FakeNewsService(),
        state_store=MemoryStateStore(),
    )


class BlackOfflineContextPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_understanding_path_runs_without_model_server(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            {
                "prompt": "동물원에는 호랑이가 있던가?",
                "action": ActionType.SHARE_OPINION,
                "domain": "animal_place",
                "schema": "concrete_topic_question",
                "contains": ("동물원", "호랑이"),
            },
            {
                "prompt": "캠핑장에서 불멍 말고 조용히 할 만한 거 있어?",
                "action": ActionType.SHARE_OPINION,
                "domain": "activity",
                "schema": "activity_recommendation",
                "contains": ("불멍", "보드게임"),
            },
            {
                "prompt": "비 온 뒤에 산책 가도 괜찮을까?",
                "action": ActionType.SHARE_OPINION,
                "domain": "sky_weather_feeling",
                "schema": "soft_decision_advice",
                "contains": ("비 온 뒤", "산책"),
            },
            {
                "prompt": "고양이랑 강아지 중 뭐가 더 좋아?",
                "action": ActionType.SHARE_OPINION,
                "domain": "animal_place",
                "schema": "preference_disclosure",
                "contains": ("고양이",),
            },
            {
                "prompt": "오늘 컨디션이 애매하면 약속 미뤄도 될까?",
                "action": ActionType.SHARE_OPINION,
                "domain": None,
                "schema": "soft_decision_advice",
                "contains": ("약속", "미뤄도"),
            },
            {
                "prompt": "목말라",
                "action": ActionType.SHARE_FEELING,
                "domain": None,
                "schema": "body_signal_interpretation",
                "contains": ("목마르면", "물"),
            },
            {
                "prompt": "배고파",
                "action": ActionType.SHARE_FEELING,
                "domain": None,
                "schema": "body_signal_interpretation",
                "contains": ("배고프면", "조금"),
            },
            {
                "prompt": "졸려",
                "action": ActionType.SHARE_FEELING,
                "domain": None,
                "schema": "low_energy_support",
                "contains": ("졸리면", "쉬자는"),
            },
            {
                "prompt": "지하철 타고 어디로 가고 싶어?",
                "action": ActionType.SHARE_OPINION,
                "domain": None,
                "schema": "preference_disclosure",
                "contains": ("지하철", "한강"),
            },
            {
                "prompt": "안녕, 반가워.",
                "action": ActionType.SMALL_TALK,
                "domain": None,
                "schema": None,
                "contains": ("나도 반가워",),
            },
            {
                "prompt": "사용자에게 조용한 안부 한 줄. 최근 화제는 지하철, 타고, 어디로. 컨디션을 가볍게 확인하면서 그 화제를 한 단계만 이어.",
                "action": ActionType.CONTINUE_CONVERSATION,
                "domain": None,
                "schema": "proactive_checkin",
                "contains": ("지하철 얘기", "목적지"),
            },
        ]

        for index, case in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=case["prompt"]):
                result = await engine.respond(f"offline-context-{index}", case["prompt"])
                evidence = result.world_state.evidence_packet

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertIsNotNone(result.draft_utterance)
                self.assertEqual(result.decision.action, case["action"])
                self.assertEqual(evidence.domain_hint, case["domain"])
                self.assertEqual(evidence.schema_hint, case["schema"])
                for expected in case["contains"]:
                    self.assertIn(expected, result.reply)

    async def test_short_hawk_token_does_not_match_inside_ambiguous_word(self) -> None:
        engine = _build_draft_only_engine()

        result = await engine.respond("offline-short-token", "오늘 컨디션이 애매하면 약속 미뤄도 될까?")

        self.assertNotEqual(result.world_state.evidence_packet.domain_hint, "animal_place")
        self.assertIn("약속", result.reply)

    async def test_korean_basic_short_daily_conversation_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("뭐 먹지?", "김치볶음밥"),
            ("배고픈데 뭐 먹을까?", "덮밥"),
            ("오늘 점심 뭐 먹지?", "제육덮밥"),
            ("저녁 뭐 먹을까?", "김치찌개"),
            ("밤에 배고픈데 야식 뭐 먹을까?", "부담"),
            ("오늘 점심 뭐 먹었어? 메뉴 추천 좀 해줘!", "먹었다고"),
            ("밥은 먹었어?", "먹는 몸"),
            ("배고프다", "배고프면"),
            ("목말라", "목마르면"),
            ("피곤해", "피곤하면"),
            ("졸려", "졸리면"),
            ("심심해", "심심하면"),
            ("오늘 뭐하지?", "산책"),
            ("뭐해?", "여기"),
            ("지금 뭐 하고 있었어?", "준비"),
            ("나 왔어", "왔구나"),
            ("다녀왔어", "숨"),
            ("잘자", "잘 자"),
            ("굿나잇", "쉬"),
            ("나 이제 잘게", "잘 자"),
            ("일어났어", "일어났구나"),
            ("방금 일어남", "물"),
            ("고마워", "고마워"),
            ("미안", "사과"),
            ("괜찮아", "다행"),
            ("아니야", "아니구나"),
            ("싫어", "싫으면"),
            ("좋아", "좋아"),
            ("몰라", "모르면"),
            ("잘 모르겠어", "느낌"),
            ("뭐라고?", "다시"),
            ("다시 말해줘", "짧고 쉽게"),
            ("잠깐만", "기다"),
            ("아", "생각"),
            ("음", "생각"),
            ("흠", "생각"),
            ("하...", "한숨"),
            ("그냥 그래", "그냥"),
            ("별일 없어", "애매"),
            ("아 진짜 웃기다", "웃"),
            ("개웃겨", "터졌"),
            ("짜증나", "짜증"),
            ("화나", "화"),
            ("불안해", "불안"),
            ("외로워", "곁"),
            ("망했다", "일단"),
            ("큰일났다", "하나"),
            ("어떡하지", "급한 것"),
            ("도와줘", "도와줄게"),
            ("더워", "더우면"),
            ("추워", "추우면"),
            ("비 온다", "우산"),
            ("날씨 좋다", "바깥 공기"),
            ("배 아파", "배"),
            ("속 안 좋아", "물"),
            ("커피 마시고 싶다", "커피"),
            ("집 가고 싶다", "에너지"),
            ("퇴근하고 싶다", "퇴근"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "먹지는 이해돼",
            "받아둘게",
            "그 생각은 이해돼",
            "온다는 이해돼",
            "날씨는 받아둘게",
            "비 오는 날엔 굳이",
            "권리 논의",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-basic-short-daily-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_basic_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

        black_tone_cases = [
            ("뭐 먹지?", "ㅋㅋ"),
            ("개웃겨", "ㅋㅋㅋ"),
            ("짜증나", "바로 찍어보자"),
            ("날씨 좋다", "나가자"),
            ("뭐해?", "바로 받을 준비"),
            ("밥은 먹었어?", "텐션"),
        ]
        for index, (prompt, expected) in enumerate(black_tone_cases, start=1):
            with self.subTest(black_tone_index=index, prompt=prompt):
                result = await engine.respond(f"offline-basic-black-tone-{index}", prompt)
                self.assertIn(expected, result.reply)

    async def test_korean_daily_companion_basics_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = [
            "밥은 먹었어? 오늘 메뉴가 뭐였어?",
            "오늘 저녁에 맛있는 거 먹고 싶은데, 뭐 땡기는 거 없어?",
            "너 매운 거 잘 먹어? 난 요즘 매운 게 너무 땡기네.",
            "너는 스트레스 받으면 먹는 걸로 푸는 편이야?",
            "점심시간이 제일 기다려져. 내일은 뭐 먹을까?",
            "갑자기 단 게 너무 땡긴다. 케이크나 초콜릿 좋아해?",
            "너 민트초코 좋아해? 이거 완전 호불호 갈리잖아.",
            "제일 좋아하는 과일이 뭐야? 난 요즘 귤/수박이 맛있더라.",
            "혹시 커피 하루에 몇 잔 마셔? 난 카페인 없으면 못 살아.",
            "나랑 커피 한잔할래? 내가 쏠게!",
            "오늘 하루 어땠어? 별일 없었어?",
            "어제 늦게 잤어? 피곤해 보이네.",
            "아침에 일어나는 거 너무 힘들지 않아? 너만의 꿀팁 있어?",
            "아, 오늘 진짜 아무것도 하기 싫다. 너도 그럴 때 있지?",
            "오늘따라 시간이 진짜 안 가는 것 같아. 벌써 지쳐.",
            "퇴근(하교)하고 보통 뭐 하면서 시간 보내?",
            "요즘 왜 이렇게 피곤한지 모르겠어. 날씨 탓인가?",
            "내일 벌써 금요일이네! 한 주가 진짜 빠른 것 같아.",
            "폰 배터리가 왜 이렇게 빨리 닳지? 너도 폰 바꿀 때 됐어?",
            "오늘 출근/등교하는 길에 사람 엄청 많더라.",
            "주말에 뭐 할 계획이야? 특별한 거 있어?",
            "요즘 재밌게 보는 드라마나 영화 있어?",
            "최근에 들은 노래 중에 추천해 줄 만한 거 있어?",
            "주말 내내 넷플릭스만 봤어. 정주행하기 좋은 거 추천 좀!",
            "요즘 새로 시작한 취미 같은 거 있어?",
            "쉬는 날에는 보통 집에 있는 편이야, 아니면 밖으로 나가?",
            "너 MBTI가 뭐야? 난 요즘 그거 보는 게 쏠쏠하게 재밌더라.",
            "요즘 푹 빠져 있는 유튜버나 챙겨보는 채널 있어?",
            "친구들이랑 만나면 주로 어디서 뭐 하면서 놀아?",
            "요즘 운동 좀 해야겠다고 느끼는데, 뭐 좋은 거 없을까?",
            "오늘 날씨 진짜 좋지 않아? 어디 산책이라도 가고 싶다.",
            "비 오는 날 좋아해? 난 비 오면 파전 생각나더라.",
            "오늘 진짜 춥지(덥지) 않아? 아침에 옷 뭐 입을지 한참 고민했어.",
            "사계절 중에 언제가 제일 좋아?",
            "이번 주말에는 날씨가 어떨까? 놀러 가고 싶은데.",
            "혹시 로또 1등 당첨되면 제일 먼저 뭐 할 거야?",
            "어디 여행 가고 싶은 곳 있어? 국내든 해외든!",
            "너 강아지파야, 고양이파야?",
            "어릴 때 장래희망이 뭐였어? 지금이랑 많이 달라?",
            "올해 가기 전에 꼭 해보고 싶은 거 하나만 꼽자면?",
            "너는 아침형 인간이야, 저녁형 인간이야?",
            "혹시 귀신이나 외계인 같은 거 믿어?",
            "초능력을 하나 가질 수 있다면 어떤 걸 갖고 싶어?",
            "무인도에 딱 3가지만 가져갈 수 있다면 뭐 챙길래?",
            "타임머신이 있다면 과거로 가고 싶어, 미래로 가고 싶어?",
            "요즘 고민거리 같은 거 있어? 괜찮으면 들어줄게.",
            "스트레스 받을 때 어떻게 푸는 편이야?",
            "최근에 제일 크게 웃었던 적이 언제야? 뭐 때문에 웃었어?",
            "최근에 산 물건 중에 제일 마음에 드는 게 뭐야? 소확행!",
            "어제 진짜 이상한 꿈 꿨어. 넌 꿈 자주 꾸는 편이야?",
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
        )

        for index, prompt in enumerate(prompts, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-companion-basic-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertGreater(len(result.reply.strip()), 10)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_companion_expansion_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = [
            "평생 여름 vs 평생 겨울, 하나만 골라야 한다면?",
            "100억 받고 스마트폰 없이 살기 vs 그냥 살기, 콜?",
            "평생 라면만 먹기 vs 평생 치킨만 먹기, 뭐가 나을까?",
            "아예 안 자도 안 피곤한 몸 vs 아무리 먹어도 살 안 찌는 몸, 어떤 능력 가질래?",
            "너 탕수육은 부먹이야 찍먹이야? (아니면 처먹?)",
            "탄산 없는 콜라 vs 식은 감자튀김, 더 끔찍한 건?",
            "진짜 친한 친구 1명 vs 그럭저럭 친한 친구 100명, 어느 쪽이 더 좋아?",
            "눈 펑펑 오는 날 출근하기 vs 비 미친 듯이 오는 날 출근하기.",
            "요리하기 vs 설거지하기, 집안일 중에 뭐가 더 싫어?",
            "평생 음악 안 듣기 vs 평생 영화/드라마 안 보기.",
            "처음 보는 사람한테 말 잘 거는 편이야, 아니면 낯가려?",
            "연락할 때 카톡이 편해, 전화가 편해?",
            "약속 시간보다 일찍 가는 편이야, 딱 맞춰서 가는 편이야?",
            "너는 화날 때 바로 말하는 편이야, 아니면 혼자 삭히는 편이야?",
            "친구가 약속 당일에 갑자기 파투 내면 어때? 화나?",
            "제일 싫어하는 사람 유형이 뭐야? 난 예의 없는 사람이 젤 싫더라.",
            "너는 친해지면 장난 많이 치는 스타일이야?",
            "혼자 밥 먹거나 영화 보는 거 잘해? 난 꽤 좋아하거든.",
            "사람을 처음 볼 때 제일 먼저 보는 곳이 어디야? 눈? 옷차림?",
            "누군가에게 서운한 게 생기면 바로 티를 내는 편이야?",
            "고등학생 때로 돌아간다면 제일 먼저 뭐 하고 싶어?",
            "살면서 해본 제일 큰 일탈이 뭐야?",
            "어릴 때 좋아했던 만화 영화 있어?",
            "핸드폰 사진첩 제일 첫 번째에 무슨 사진 있어?",
            "살면서 제일 지우고 싶은 흑역사 하나만 풀어봐.",
            "학창 시절에 제일 좋아했던 과목이랑 싫어했던 과목은 뭐야?",
            "첫 알바 했을 때 기억나? 무슨 일 했었어?",
            "예전에 유행했던 것 중에 다시 유행했으면 하는 거 있어?",
            "어릴 때 부모님한테 했던 제일 귀여운 거짓말 기억나?",
            "중2병 시절에 했던 부끄러운 짓 있어? 싸이월드 감성 같은 거.",
            "월급 타면 제일 먼저 뭐부터 사?",
            "나 요즘 돈 너무 많이 쓰는 것 같아. 너는 저축 잘해?",
            "당장 내일 100만 원이 하늘에서 떨어진다면 뭐 할래?",
            "진짜 쓸데없는데 돈 주고 산 물건 있어? 예쁜 쓰레기 같은 거.",
            "너는 물건 살 때 디자인을 먼저 봐, 실용성을 먼저 봐?",
            "평소에 충동구매 많이 하는 편이야?",
            "로또 샀어? 난 가끔 좋은 꿈 꾸면 한 장씩 사는데.",
            "한 달 생활비 중에 어디에 제일 돈을 많이 쓰는 것 같아? 식비?",
            "비싼 옷 한 벌 사기 vs 싼 옷 여러 벌 사기, 너의 선택은?",
            "돈 꽉꽉 모아서 제일 사고 싶은 워너비 아이템 있어?",
            "동물이랑 말이 통하게 된다면 집에 있는 반려동물(혹은 길고양이)한테 제일 먼저 뭐라고 할래?",
            "갑자기 좀비 사태가 터지면 어디로 숨을 거야? 나는 대형 마트.",
            "이름 말고 다르게 불리고 싶은 멋진 별명 있어?",
            "로봇이 내 대신 출근해 줬으면 좋겠다. 안 그래?",
            "하루가 24시간 말고 30시간이면 남는 시간에 뭐 할래?",
            "길 가다가 만 원 주우면 어떡할 거야? 경찰서? 아니면 까까 사 먹기?",
            "무인도에 떨어졌는데 밥은 없고 와이파이는 빵빵 터져. 살만할 것 같아?",
            "자고 일어났더니 내가 바퀴벌레로 변해 있으면 어떡할래?",
            "투명 인간이 된다면 딱 하루 동안 뭐 하고 싶어?",
            "아무 데나 갈 수 있는 '어디로든 문'이 있다면 지금 당장 어디로 갈래?",
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
        )

        for index, prompt in enumerate(prompts, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-companion-expansion-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertGreater(len(result.reply.strip()), 10)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_relationship_work_travel_healing_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = [
            '넌 첫눈에 반한다는 걸 믿어, 아니면 천천히 스며드는 게 좋아?',
            '연락 빈도랑 애정의 크기가 비례한다고 생각해?',
            '이상형이 어떻게 돼? 외모 말고 성격적인 부분에서!',
            '연인 사이에 핸드폰 비밀번호 공유할 수 있어?',
            '남사친/여사친 문제로 싸워본 적 있어? 어디까지 허용 가능해?',
            '나쁜 남자(여자)한테 끌리는 편이야, 아니면 무조건 다정한 사람?',
            '환승 이별 vs 잠수 이별, 뭐가 더 최악이야?',
            '연애할 때 가장 중요하게 생각하는 가치관이 뭐야?',
            '썸 탈 때 제일 설레는 순간이 언제인 것 같아?',
            '만약 새벽 2시에 전 애인한테 "자니?" 하고 연락 오면 어떻게 할 거야?',
            '인생에서 가장 중요하게 생각하는 한 가지 단어가 있다면?',
            '넌 운명을 믿어, 아니면 스스로 개척하는 거라고 생각해?',
            '만약 오늘이 인생의 마지막 날이라면 누구랑 뭘 하고 싶어?',
            '행복은 돈으로 살 수 있다고 생각해?',
            '타인의 시선을 많이 신경 쓰는 편이야?',
            '절대 용서할 수 없는 행동이나 거짓말이 있다면 어떤 거야?',
            '살면서 가장 크게 후회했던 적이 언제야?',
            '나이를 먹는다는 게 두려워, 아니면 기대돼?',
            '나쁜 의도로 한 착한 행동 vs 착한 의도로 한 나쁜 행동, 뭐가 더 나빠?',
            '다시 태어난다면 지금의 나로 태어나고 싶어, 아니면 완전 다른 사람?',
            '지금 하는 일(혹은 전공)이 너랑 잘 맞는 것 같아?',
            '월급 적은데 워라밸 최고 vs 월급 엄청 많은데 맨날 야근. 어떤 게 좋아?',
            '직장에서 진짜 마음 터놓고 지낼 수 있는 찐친을 만들 수 있을까?',
            '꼰대 상사 밑에서 일하기 vs 일 진짜 못하는 후배 데리고 일하기.',
            '자유로운 프리랜서로 살고 싶어, 아니면 안정적인 직장인이 좋아?',
            '일하다가 진짜 다 때려치우고 싶을 때 어떻게 버텨?',
            '출퇴근 시간 왕복 3시간인데 연봉 1억 vs 도보 10분인데 연봉 3천.',
            '직장 동료들이랑 주말에 사적으로 만나는 거 어떻게 생각해?',
            '만약 지금 당장 직업을 바꿀 수 있다면 무슨 일 해보고 싶어?',
            '100억 로또 당첨돼도 지금 하는 일 계속 할 거야?',
            '지금까지 가본 여행지 중에 제일 기억에 남는 곳이 어디야?',
            '완전 계획형(J) 여행이 좋아, 아니면 발길 닿는 대로 가는 즉흥(P) 여행이 좋아?',
            '혼자 여행 가본 적 있어? 느낌이 어때?',
            '비행기 타고 멀리 가는 해외여행 vs 차 타고 훌쩍 떠나는 조용한 국내 여행.',
            '여행지에서 꼭 하는 너만의 루틴 같은 거 있어? (예: 자석 모으기)',
            '평생 한 나라의 음식만 먹고 살아야 한다면 어느 나라 음식 고를래?',
            '배낭 하나 메고 세계 일주하기, 너라면 할 수 있을 것 같아?',
            '험난한 산이 좋아, 탁 트인 바다가 좋아?',
            '외국어 하나를 원어민처럼 할 수 있게 된다면 무슨 언어 배우고 싶어?',
            '여행 가서 사진 1000장 찍기 vs 눈에만 담고 사진 한 장도 안 찍기.',
            '하루 중에서 제일 좋아하는 시간이 언제야?',
            '비 오는 날 창밖 보면서 멍 때리는 거 좋아해?',
            '최근에 너한테 작지만 확실한 행복(소확행)을 준 일은 뭐야?',
            '자기 전에 누워서 주로 무슨 생각 해?',
            '우울하거나 기분 안 좋을 때 꼭 듣는 힐링곡 있어?',
            '혼자만의 시간이 꼭 필요한 편이야?',
            '누군가에게 위로받고 싶을 때, 조언이 필요해 아니면 그냥 공감이 필요해?',
            '새벽 감성 타본 적 있어? 그럴 땐 주로 뭐 해?',
            '네 방에서 제일 좋아하는 공간이나 물건이 뭐야?',
            '오늘 하루 정말 고생 많았어. 잠들기 전에 스스로에게 칭찬 한마디 해준다면?',
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
        )

        for index, prompt in enumerate(prompts, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-relationship-work-travel-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith(("korean_daily_", "knowledge_")), reason)
                self.assertGreater(len(result.reply.strip()), 10)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_media_debate_fantasy_inner_family_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = [
            '인생 영화 3개만 꼽아줄 수 있어?',
            '요즘 즐겨보는 웹툰이나 애니메이션 있어?',
            '게임 좋아해? 폰 게임이든 PC 게임이든!',
            "살면서 무언가에 열정적으로 '덕질' 해본 적 있어?",
            '책 읽는 거 좋아해? 아니면 유튜브로 요약본 보는 게 더 편해?',
            '영화관 팝콘은 무슨 맛이 진리라고 생각해? (카라멜 vs 어니언)',
            '좀비 영화가 좋아, 아니면 귀신 나오는 오컬트 영화가 더 무서워?',
            '음악 들을 때 멜로디를 먼저 들어, 가사를 먼저 들어?',
            '한 곡에 꽂히면 그것만 무한 반복해서 듣는 스타일이야?',
            '내가 주인공이 된다면 어떤 장르의 영화(혹은 드라마)에 출연하고 싶어?',
            '내 애인이 내 절친의 깻잎을 떼어준다면? (원조 깻잎 논쟁)',
            '내 애인이 내 절친의 패딩 지퍼를 올려준다면?',
            "남녀 사이에 진짜 '그냥 친구'가 존재할 수 있다고 생각해?",
            '블루투스 이어폰 한쪽 나눠 끼는 거, 이성 친구끼리 가능?',
            '민트초코는 치약 맛이다 vs 아니다, 상쾌한 초코 맛이다!',
            '샤워할 때 양치 먼저 해, 아니면 머리부터 감아?',
            '양치할 때 칫솔에 물 묻힌다 vs 안 묻히고 바로 닦는다.',
            '시리얼 먹을 때 우유 먼저 붓는다 vs 시리얼 먼저 넣는다.',
            '붕어빵 먹을 때 머리부터 먹어, 꼬리부터 먹어?',
            '여름에 에어컨 빵빵하게 틀고 두꺼운 이불 덮기 vs 겨울에 전기장판 틀고 아이스크림 먹기.',
            '외계인이 지구에 오면 우리한테 우호적일까, 적대적일까?',
            '내가 죽었는데 저승사자가 실수로 데려온 거라면, 저승사자한테 뭐라고 따질래?',
            '갑자기 마법을 쓸 수 있게 된다면 제일 먼저 무슨 주문을 외울 거야?',
            '우주여행을 갈 수 있는 티켓이 생겼는데 돌아올 확률이 50%야. 갈래?',
            '세상의 모든 사람이 내 마음속 생각을 읽을 수 있게 된다면 어떨 것 같아?',
            '내가 키우던 인공지능이 나에게 진심으로 사랑을 고백한다면?',
            '내일 세상이 멸망한다는 뉴스가 뜨면 남은 하루 동안 뭐 할 거야?',
            '해리포터 호그와트 기숙사에 배정된다면 넌 어디에 갈 것 같아?',
            '만약 내가 뱀파이어가 된다면 사람 피를 마실 수 있을까?',
            '평생 한 가지 나이로만 고정되어 살아야 한다면 몇 살로 살고 싶어?',
            '누군가 널 칭찬할 때 속으로 어떻게 생각해? (진짜? vs 그냥 하는 말이겠지~)',
            '넌 감정 기복이 심한 편이야, 아니면 꽤 평온한 편이야?',
            '완벽주의 성향이 있어, 아니면 대충대충 유연하게 넘어가는 편이야?',
            '혼자 있을 때 내면의 목소리랑 대화 자주 하는 편이야?',
            '너 자신을 동물로 표현한다면 어떤 동물일 것 같아?',
            '실패를 겪었을 때 금방 털고 일어나는 편이야?',
            '다른 사람들이 너를 어떤 사람으로 기억해 줬으면 좋겠어?',
            '가장 나다운 모습(본모습)이 나올 때는 주로 언제인 것 같아?',
            '질투심이나 승부욕이 많은 편이야? 아니면 쿨한 편이야?',
            '중요한 결정을 내릴 때 이성이 앞서는 편이야, 감정이 앞서는 편이야?',
            '부모님이랑 친구처럼 편하게 지내는 편이야?',
            '형제자매 있어? 어릴 때 많이 싸웠어?',
            '어릴 때 가장 기억에 남는 생일 파티나 선물이 뭐야?',
            '부모님한테 물려받은 성격이나 습관 중에 신기한 거 있어?',
            '명절에 친척들 모이면 제일 듣기 싫은 말이 뭐야?',
            '완전히 독립해서 혼자 살게 된다면 집을 어떤 스타일로 꾸미고 싶어?',
            '어릴 때 산타 할아버지 언제까지 믿었어?',
            '집밥 중에 제일 좋아하는 반찬이나 국은 뭐야?',
            '어린 시절 사진 보면 그때의 네가 귀여워 보여?',
            '훗날 네가 부모가 된다면 아이에게 꼭 가르쳐주고 싶은 삶의 지혜가 있어?',
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
        )

        for index, prompt in enumerate(prompts, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-media-fantasy-family-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertGreater(len(result.reply.strip()), 10)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_health_errands_chores_shopping_social_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = [
            '하루 종일 모니터 봤더니 눈이 너무 뻑뻑해. 이럴 때 뭐 하면 좀 나아질까?',
            '목이 살짝 칼칼한데 감기 오는 걸까? 오늘은 어떻게 관리하는 게 좋을까?',
            '어제 야식 먹고 잤더니 속이 더부룩해. 아침은 굶는 게 나을까?',
            '운동 오랜만에 했더니 다리가 후들거려. 내일도 운동해도 될까?',
            '커피를 마셨는데도 계속 졸려. 이건 잠이 부족한 걸까, 그냥 체력이 떨어진 걸까?',
            '요즘 손이 너무 건조해서 트는데 핸드크림 말고 좋은 습관 있어?',
            '자고 일어났는데 목이 뻐근해. 베개가 문제일까 자세가 문제일까?',
            '배고픈데 뭘 먹어야 할지 모르겠어. 가볍게 먹을 만한 거 추천해줘.',
            '요즘 얼굴이 푸석해 보이는데 잠 때문일까, 물을 덜 마셔서 그런 걸까?',
            '밤마다 늦게 자는 습관 어떻게 고쳐야 할까?',
            '버스가 눈앞에서 떠나버렸어. 이 허무함 어떻게 달래야 하냐.',
            '지하철에서 자리가 났는데 어르신이 옆에 있으면 바로 양보하는 편이야?',
            '택시 타면 편하긴 한데 돈이 너무 아까워. 그래도 피곤하면 타는 게 맞을까?',
            '비 오는데 우산이 없어. 그냥 뛰어갈까, 편의점에서 우산 살까?',
            '약속 시간에 10분 늦을 것 같은데 뭐라고 말하는 게 제일 덜 민망할까?',
            '엘리베이터 기다리기 답답해서 계단으로 갈까 고민될 때 몇 층까지는 걸어가?',
            '사람 많은 버스에서 이어폰 배터리까지 없으면 너무 괴롭지 않아?',
            '길을 잘 못 찾는 편이야? 지도 앱 없으면 바로 멘붕 올 것 같아.',
            '퇴근길에 편의점 들르면 꼭 쓸데없는 걸 사게 돼. 이거 어떻게 막지?',
            '아침 출근길에 사람이 너무 많으면 하루 시작부터 기 빨리지 않아?',
            '설거지는 바로 하는 편이야, 아니면 싱크대에 쌓였다가 한 번에 해치우는 편이야?',
            '빨래 돌려놓고 널기 귀찮아서 까먹은 적 있어? 냄새나면 다시 돌려야겠지?',
            '방 청소할 때 책상부터 치워, 아니면 바닥부터 치워?',
            '음식물 쓰레기 버리는 거 진짜 귀찮은데 안 미루는 꿀팁 있어?',
            '냉장고에 유통기한 지난 반찬 발견하면 바로 버려, 아니면 냄새 맡아보고 판단해?',
            '이불 빨래는 날 잡고 하는 큰 행사 같지 않아? 얼마나 자주 해야 할까?',
            '옷장 정리하다가 안 입는 옷 나오면 버리는 편이야, 혹시 몰라서 넣어두는 편이야?',
            '집에 먼지가 너무 빨리 쌓이는 것 같아. 청소를 자주 해야 하나 공기청정기가 답인가?',
            '혼자 살면 제일 귀찮은 집안일이 뭐라고 생각해?',
            '화장실 청소는 진짜 미루면 더 지옥 되는 것 같지 않아?',
            '온라인 쇼핑할 때 장바구니에 넣어두고 며칠 고민하는 편이야?',
            '사고 싶은 옷이 있는데 세일할 때까지 기다릴까, 품절되기 전에 살까?',
            '비싼 전자기기 살 때 리뷰를 얼마나 믿어야 할까?',
            '충동구매 막으려면 결제 전 며칠 기다리는 게 효과 있을까?',
            '택배 기다리는 시간은 왜 이렇게 길게 느껴질까? 배송중만 보면 계속 새로고침하게 돼.',
            '중고거래할 때 직거래가 좋아, 택배거래가 좋아?',
            '리뷰가 좋은데 디자인이 별로인 물건 vs 디자인은 예쁜데 리뷰가 애매한 물건, 뭐 살래?',
            '할인 쿠폰 쓰려고 필요 없는 물건까지 더 사는 거 완전 함정 아니야?',
            '월급 들어오면 바로 사고 싶은 거부터 사는 편이야, 일단 저축부터 하는 편이야?',
            '비싸게 산 물건을 별로 안 쓰면 괜히 죄책감 들지 않아?',
            '친구가 답장을 하루 종일 안 하면 걱정돼, 아니면 그냥 바쁜가 보다 해?',
            '누가 내 말을 끊고 자기 얘기만 하면 어떻게 반응하는 편이야?',
            '친구가 약속을 자꾸 미루면 몇 번까지 이해해줄 수 있어?',
            '사과할 때 제일 중요한 건 말투야, 아니면 다시 안 그러는 행동이야?',
            '나한테만 장난 심하게 치는 사람이 있으면 바로 말해야 할까?',
            '단톡방에서 나만 빼고 얘기한 걸 알면 서운할 것 같아?',
            '친구가 힘들다고 하면 조언부터 해줘, 아니면 일단 들어주는 편이야?',
            '낯선 모임에 가면 먼저 말 거는 편이야, 누가 말 걸 때까지 기다리는 편이야?',
            '좋아하는 사람 앞에서는 말이 많아져, 아니면 오히려 조용해져?',
            '오래 연락 안 하던 친구한테 갑자기 연락해도 어색하지 않게 시작하는 멘트 뭐가 좋을까?',
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
            "부담이 너무 크지",
        )

        for index, prompt in enumerate(prompts, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-health-errands-chores-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertGreater(len(result.reply.strip()), 10)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_workday_a_side_dialogue_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("안녕하세요~ 오늘 진짜 춥지 않아요? 버스 내리는데 귀 떨어지는 줄 알았어요.", "출근길"),
            (
                "월요병 장난 아니죠. 아침에 눈 뜨는데 진짜 출근하기 싫어서 눈물 날 뻔했다니까요. 일단 커피부터 한 잔 수혈하고 시작해야겠어요.",
                "커피",
            ),
            ("오늘 점심 뭐 먹을까요? 맨날 가던 데는 지겨운데.", "제육덮밥"),
            (
                "오, 부대찌개 좋은데요? 라면 사리 무조건 추가하는 걸로 가시죠. 저 지금 배고파서 쓰러지기 직전이에요.",
                "배고프",
            ),
            ("드디어 금요일이네요! 오늘 퇴근하고 뭐 하세요?", "금요일"),
            (
                "와, 최고의 계획이네요. 저는 주말에 밀린 잠 좀 자고, 오랜만에 친구들 만나서 맛있는 거 먹기로 했어요. 우리 조금만 더 버티고 힘내서 퇴근해요!",
                "밀린 잠",
            ),
            ("대박, 진짜 오랜만이다! 잘 지내고 있었어?", "오랜만"),
            (
                "나야 맨날 똑같지 뭐. 일하고, 집 와서 기절하고 무한 반복이야. 조만간 날 잡아서 얼굴 한번 보자. 맛있는 거 먹으러 가자!",
                "무한 반복",
            ),
        ]
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
            "부담이 너무 크지",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-workday-a-side-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_hair_reunion_after_work_cafe_dialogue_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            (
                "아, 그냥 전체적으로 다듬고 숱만 좀 쳐주세요. 날이 더워지니까 머리가 너무 무겁게 느껴지더라고요.",
                "숱",
            ),
            ("아뇨, 귀찮아서 그냥 말리기만 해요. 최대한 손 안 가게 잘라주세요. 웃음", "손 안 가"),
            ("우와, 이게 얼마 만이냐! 너 얼굴 진짜 좋아졌다.", "오랜만"),
            (
                "응, 이직할까 말까 백만 번 고민하면서 영혼 없이 다니는 중이지 뭐. 직장인이 다 그렇지. 너는 프로젝트 끝났다더니 이제 좀 숨 돌려?",
                "이직",
            ),
            ("오늘 퇴근하고 시간 됨?", "퇴근"),
            (
                "오늘따라 삼겹살에 소주가 너무 아른거리네. 회사 근처에 새로 생긴 고깃집 가볼래?",
                "삼겹살",
            ),
            ("네, 맞아요! 기억해 주시네요. 오늘 출근길에 너무 지쳐서 카페인 수혈이 시급했어요.", "카페인"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
            "부담이 너무 크지",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-hair-reunion-cafe-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_mbti_homebody_shortform_game_dialogue_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("아, 맞다. 너 MBTI가 뭐라고 했지? 완전 파워 T였나?", "파워 T"),
            (
                "헐, 진짜? 저번에 내가 힘든 일 말했을 때 해결책부터 딱딱 제시해주길래 빼박 T인 줄 알았지. 근데 나도 요즘 나이 들면서 성향이 조금씩 바뀌는 것 같아.",
                "해결책",
            ),
            ("너 이번 주말에 뭐 했냐? 연락도 안 되고.", "생존신고"),
            ("와, 대단하다. 안 심심해? 답답하지도 않아?", "충전"),
            ("야, 내가 어제 릴스 보다가 새벽 3시에 잤잖아. 진짜 알고리즘 무서워.", "알고리즘"),
            (
                "그니까. 너 그 요즘 유행하는 챌린지 영상 봤어? 그거 노래 계속 맴돌아서 미치겠음.",
                "챌린지",
            ),
            ("오늘 저녁에 접속함? 요즘 이벤트 기간이라 출석 보상 받아야 하는데.", "출석 보상"),
            (
                "오케이, 굿. 요즘 이거 하느라 넷플릭스 볼 시간도 없네. 얼른 렙업하고 다음 단계 넘어가자고.",
                "렙업",
            ),
        ]
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
            "부담이 너무 크지",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-mbti-homebody-shortform-game-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_late_night_fandom_food_algorithm_dialogue_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("야, 자냐? 나 갑자기 생각 많아져서 잠이 안 온다.", "새벽"),
            (
                "그냥... 내가 지금 잘 살고 있는 건가 싶어서. 남들은 다 저만치 앞서가는 것 같은데 나만 제자리걸음 하는 느낌 있잖아.",
                "제자리걸음",
            ),
            ("와, 이번에 새로 나온 캐릭터 디자인 보셨어요? 흑발에 적안 조합 미쳤던데.", "흑발"),
            (
                "맞아요, 서사도 완전 처절하더라고요. 이건 안 팔 수가 없다... 공식 굿즈 나오면 바로 지갑 열 준비 완료 입니다.",
                "공식 굿즈",
            ),
            ("여기 떡볶이집 진짜 숨은 맛집임. 튀김을 떡볶이 국물에 적셔서 한 입 먹으면 극락 간다.", "떡볶이"),
            (
                "단골들만 아는 데라 아껴둔 건데 특별히 알려줌. 다음 주에 여기 파티원 구해서 조지러 가자.",
                "단골 맛집",
            ),
            ("아, 유튜브 알고리즘 이상한 거 학습했나 봐. 자꾸 이상한 고양이 춤추는 영상만 추천해 줌.", "고양이"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
            "부담이 너무 크지",
            "야자 몰래",
            "마지막까지 아껴두는 편",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-late-night-fandom-food-algorithm-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_work_food_home_fandom_reality_chat_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("오늘 진짜 눈 뜨자마자 다시 감고 싶더라. 출근길 지하철 왜 이렇게 밀려?", "출근"),
            ("아침에 아이스 아메리카노 안 마시면 뇌가 안 깨요. 지금 수혈 중입니다.", "아아"),
            ("오늘 월요일인 줄 알았는데 아직 수요일이네? 시간 왜 이렇게 안 가냐.", "수요일"),
            ("어제 넷플릭스 새로 나온 거 보다가 새벽 2시에 잠. 오늘 오전 근무 버릴 예정.", "오전 근무"),
            ("오늘 날씨 왜 이래? 아침엔 추웠는데 지금은 또 쪄 죽을 것 같아.", "겹옷"),
            ("지하철에서 내 앞에 앉은 사람 내릴 줄 알고 대기 탔는데 종점까지 가더라. 대실패.", "종점"),
            ("오늘 아침에 알람 소리 못 듣고 기적적으로 눈 떠서 세수만 하고 뛰어 나옴.", "탈출"),
            ("연차 쓰고 싶다. 로또 1등 되면 출근길에 바로 사표 던지는 상상하면서 버티는 중.", "사표"),
            ("아침부터 메일함 터져 있는 거 보고 조용히 창 닫음. 흐린 눈 시전.", "메일함"),
            ("오늘 팀장님 기분 안 좋아 보이던데 다들 레이더 세우고 조심하자.", "레이더"),
            ("오늘 점심 뭐 먹음? 메뉴 고르는 게 하루 중 제일 힘든 고민임.", "덮밥"),
            ("날도 꿀꿀한데 뜨끈한 부대찌개에 라면 사리 넣어서 조지자.", "부대찌개"),
            ("회사 근처에 새로 생긴 돈까스집 웨이팅 개길더라. 패스하고 그냥 가던 데 가자.", "돈까스"),
            ("나 요즘 다이어트한다고 닭가슴살 샀는데 벌써 물림. 떡볶이 먹고 싶다.", "닭가슴살"),
            ("저녁에 삼겹살에 소주 한잔 하실 분? 급 번개 구함.", "삼겹살"),
            ("여기 빵집 소금빵 미쳤음. 한 입 먹자마자 버터 풍미 폭발함.", "소금빵"),
            ("배달 앱 보다가 1시간 지남. 결국 맨날 먹던 마라탕 시켰다.", "마라탕"),
            ("오늘 저녁은 대충 편의점 꿀조합으로 때워야지. 마크정식 고?", "마크정식"),
            ("치킨 시켰는데 배달 예정 시간 80분 뜸. 배고파서 현기증 난다고요.", "80분"),
            ("야, 너 민초파냐 반민초파냐? 이거에 따라 우리 우정 갈린다.", "반민초"),
            ("이번 주말 계획? 금요일 퇴근하는 순간부터 침대랑 물아일체 될 예정.", "물아일체"),
            ("토요일에 눈 뜨니까 오후 2시더라. 주말 하루 그냥 순삭 당함.", "순삭"),
            ("밖은 위험해. 에어컨 틀어놓고 이불 속에서 유튜브 보는 게 인생 최고의 행복임.", "이불"),
            ("나 금요일 저녁에 집에 들어온 이후로 사람 말 한마디도 안 함. 톡만 하는 중.", "묵언수행"),
            ("친구가 주말에 나오라는데 나가기 전부터 기 빨림. 약속 취소됐으면 좋겠다.", "쾌재"),
            ("주말에 배달 음식 시켜 먹고 쓰레기 버리러 나간 게 유일한 외출이었음.", "유일한 외출"),
            ("집에서 혼자 넷플릭스 정주행하면 시간 왜 이렇게 빨리 가냐? 시공간이 왜곡됨.", "넷플릭스"),
            ("주말 순삭 당하고 일요일 밤 되니까 심장 뛰어. 출근하기 싫어서 잠 안 옴.", "일요일 밤"),
            ("나 오늘 큰맘 먹고 방 청소 대대적으로 함. 올해 쓸 에너지 다 썼다.", "방 청소"),
            ("집에서 유튜브 쇼츠 보다 보니 3시간 지나 있음. 알고리즘 무서운 자식.", "시간 도둑"),
            ("야, 이번에 새로 나온 캐릭터 디자인 봤냐? 흑발에 적안 조합은 솔직히 사기지.", "흑발"),
            ("그 장르 공식 일러 떴는데 비주얼 미쳤음. 벽 부수다가 우리 집 원룸 됨.", "공식 일러"),
            ("오늘 저녁에 겜 접속함? 저번에 하던 던전 마저 밀어야지.", "던전"),
            ("나 요즘 인스타 릴스 유행하는 노래 계속 룰루랄라 맴돌아서 수능 금지곡 됨.", "수능 금지곡"),
            ("이거 웹툰 진짜 숨은 명작임. 제발 한 번만 봐줘, 서사가 미쳤다고.", "웹툰"),
            ("오늘 알고리즘에 고양이 영상만 100개 뜸. 랜선 집사 하느라 정신 못 차림.", "랜선 집사"),
            ("새벽에 유튜브 보다가 조용히 지갑 열림. 꿀템이라고 해서 홀린 듯 결제했다.", "지갑"),
            ("나 이번 주말에 콘서트 티켓팅 도전하는데 손 떨림. 제발 내 자리 하나만.", "티켓팅"),
            ("최애캐 굿즈 예약 구매 갈 지 말 지 백만 번 고민 중. 통장이 텅장인데.", "최애캐 굿즈"),
            ("요즘 이 게임 메타 바뀌어서 적응 안 됨. 예전 캐릭들 다 고인 됐네.", "메타"),
            ("월급 들어왔는데 퍼가요~ 당해서 스쳐 지나감. 숫자가 잠시 보였다가 사라졌어.", "월급"),
            ("나 MBTI 검사 다시 했는데 극 T로 바뀜. 사회 생활이 날 이렇게 만들었다.", "극 T"),
            ("오늘 조별 과제 빌런 만남. 자료 조사 하라니까 나무위키 그대로 긁어옴.", "나무위키"),
            ("카톡 답장 읽씹 당함. 바쁜가 보네 하고 넘어가려는데 은근히 신경 쓰임.", "답장"),
            ("아, 이어폰 두고 나왔다. 오늘 출근길은 세상의 소음을 다 다이렉트로 들어야 함.", "이어폰"),
            ("탕비실에 있던 맥심 모카골드 내가 마지막으로 털어먹음. 묘한 죄책감.", "맥심"),
            ("오늘 퇴근 10분 전에 던져진 업무 보고 조용히 심호흡함. 야근 당첨.", "야근"),
            ("남들은 다 갓생 사는데 나만 침대에 누워서 뒹굴거리는 것 같아 현타 옴.", "갓생"),
            ("스마트폰 스크린 타임 하루 8시간 찍힘. 나 거의 폰이랑 결혼한 수준.", "스크린타임"),
            ("드디어 금요일 퇴근이다! 다들 고생했고 주말 동안 서로 아는 척하지 말자!", "금요일"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
            "부담이 너무 크지",
            "확실하지 않음",
            "전자기기 없이",
            "판타지 선택",
            "자료 조사 쪽",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-work-food-home-fandom-reality-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_commute_delivery_bed_dopamine_reality_variants_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("오늘 버스 카드 찍는데 '잔액이 부족합니다' 뜨는 순간 등줄기에 식은땀 쫙 흐름.", "잔액 부족"),
            ("지하철에서 에어컨 틀어줬는데 내 자리만 직사광선 직빵이라 에어컨 무소용임.", "직사광선"),
            ("오늘 아침에 구두 신고 나왔다가 발가락 파괴됨. 걍 크록스 신을 걸.", "크록스"),
            ("오전 10시밖에 안 됐는데 벌써 집 가고 싶다고 선언함. 내 멘탈 유부초밥 수준.", "유부초밥"),
            ("오늘 엘리베이터 문 열리자마자 부장님이랑 눈 마주쳐서 계단으로 뛰어 올라감.", "부장님"),
            ("어제 야식으로 불닭 먹었더니 아침부터 장에서 폭죽 터지는 중. 살려줘.", "폭죽"),
            ("비 온다더니 해 쨍쨍한 거 뭐임? 가방에 든 장우산 짐 덩어리 됨.", "장우산"),
            ("오늘 아침에 택시 탈까 말까 50번 고민하다가 결국 타고 지갑 찢어짐.", "지갑"),
            ("출근길에 고양이 식빵 굽는 거 보고 30초 동안 멍 때리다가 지각할 뻔.", "고양이"),
            ("오늘 목요일인데 뇌는 이미 토요일 새벽 3시쯤에 가 있음.", "목요일"),
            ("떡볶이에 중국당면이랑 분모자 추가 안 하면 유죄임. 그건 떡볶이에 대한 모독이야.", "중국당면"),
            ("오늘 점심 메이트가 '아무거나' 시전해서 조용히 한숨 쉬고 국밥집 데려감.", "국밥"),
            ("요즘 편의점 두바이 초콜릿 재고 뜨는 거 레이더 돌리는 중인데 맨날 허탕임.", "두바이 초콜릿"),
            ("치킨 시켰는데 리뷰 이벤트 감자튀김 누락됨. 내 삶의 낙이 사라졌다.", "감자튀김"),
            ("나 다이어트 시작함. 근데 오늘 저녁까지만 먹고 내일부터 진짜 할 거임.", "다이어트"),
            ("여기 카페 크림라떼 비주얼 미쳤음. 섞지 말고 크림부터 마셔야 극락 감.", "크림라떼"),
            ("스트레스 받을 땐 닭발에 계란찜 마요네즈 주먹밥 세트가 만병통치약임.", "닭발"),
            ("집 가면서 배달 앱으로 결제 갈겨놓음. 문 앞에 도착해 있는 치킨 보면 설렘.", "치킨"),
            ("요즘 물가 미쳐서 마트 가서 과자 몇 개 집었는데 2만 원 나옴. 내 지갑 구멍 났냐.", "2만 원"),
            ("단톡방에 '오늘 저녁 추천 좀' 올렸더니 다들 자기 먹고 싶은 것만 던짐.", "저녁 추천"),
            ("주말에 하려고 했던 일: 책 읽기, 운동, 대청소. 실제로 한 일: 숨쉬기, 폰 보기.", "숨쉬기"),
            ("침대 헤드에 등 기대고 폰 하다가 얼굴에 떨궈서 앞이빨 깨질 뻔함.", "폰"),
            ("나 토요일에 14시간 잠. 인체의 신비를 경험했다. 인간은 얼마나 잘 수 있는가.", "14시간"),
            ("집에 오면 양말부터 허물 벗듯이 벗어 던지는 게 국룰 아님?", "양말"),
            ("친구가 자꾸 핫플 가자는데 인스타 사진 한 장 보니까 벌써 기 빨려서 가기 싫음.", "핫플"),
            ("주말에 밀린 빨래 돌려놓고 건조기 소리 들으면서 멍 때리는 거 은근 힐링임.", "건조기"),
            ("배달 기사님 벨 누르는 소리가 세상에서 제일 반가운 백색소음임.", "백색소음"),
            ("일요일 밤 11시 50분의 그 아련하고 슬픈 감정... 주말을 영원히 붙잡고 싶다.", "일요일 밤"),
            ("나 오늘 큰맘 먹고 이불 빨래함. 오늘 밤엔 뽀송뽀송한 냄새 맡으면서 꿀잠 잔다.", "이불 빨래"),
            ("폰 스크린 타임 줄이려고 앱 잠금 걸어놨는데 비번 내가 풀고 계속 보는 중.", "앱 잠금"),
            ("이번에 나온 신작 캐릭 서사가 너무 매콤해서 눈물 흘리는 중. 공식이 날 속였다.", "신작 캐릭"),
            ("최애캐 일러스트 새로 떴는데 이건 배경화면 각임. 평생 소장한다.", "배경화면"),
            ("오늘 게임 가챠 돌렸는데 폭망함. 내 돈 돌려내라 이 똥손아.", "가챠"),
            ("알고리즘이 자꾸 나한테 자취방 꿀템 추천해 주는데 나 자취 안 함. 왜 이래.", "자취방"),
            ("이 노래 도입부 비트 미쳤음. 한 곡 재생으로 지금 50번째 듣는 중.", "도입부"),
            ("새벽에 쇼츠 넘기다가 웃긴 댕댕이 영상 보고 혼자 침대에서 꺽꺽거림.", "댕댕이"),
            ("인스타 돋보기 탭 들어갔다가 홀린 듯이 옷 3벌 결제함. 소비 요정 강림.", "소비 요정"),
            ("나 주말에 웹소설 정주행 시작했다가 완결까지 밤샘. 내 눈 시뻘개짐.", "웹소설"),
            ("굿즈 택배 상자 뜯을 때가 세상에서 제일 짜릿해. 칼로 테이프 자르는 손맛.", "굿즈"),
            ("이 게임 요즘 밸런스 패치 선 넘었네. 내 본캐 버프 좀 해줘라 진짜.", "밸런스 패치"),
            ("회사 탕비실 커피머신 고장 나서 다들 좀비처럼 편의점으로 걸어가는 중.", "커피머신"),
            ("나 MBTI 다시 보니까 내향성 99% 나옴. 이 정도면 그냥 자연으로 돌아가야 됨.", "내향성 99%"),
            ("오늘 조별과제 단톡방에 '저 과제 다 했습니다!' 했는데 나 혼자 한 거였음.", "조별과제"),
            ("카톡에 'ㅋ' 개수 고르는 거 나만 진심임? 2개는 비웃음 같고 5개는 써야 찐 웃음.", "ㅋ 개수"),
            ("에어팟 한쪽 독서실 책상 밑으로 굴러 들어가서 먼지 구덩이 파헤침.", "에어팟"),
            ("미용실 가서 '알아서 잘라주세요' 했다가 최양락 단발머리 됨. 당분간 모자만 쓴다.", "최양락"),
            ("퇴근 5분 전에 갑자기 회의 소집하는 상사 특: 1시간 동안 자기 옛날 얘기함.", "퇴근 5분"),
            ("남들 주식, 코인으로 대박 났다는데 내 계좌는 왜 마이너스 파란 불만 켜져 있냐.", "주식"),
            ("오늘 하루 종일 한 거: 숨쉬기, 눈 깜빡이기, 밥 먹기. 생산성 제로의 삶 완료.", "생산성 제로"),
            ("드디어 불금이다! 다들 단톡방 나가고 월요일 아침까지 서로 생사 확인하지 말자!", "불금"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "꽤 맞는 쪽",
            "사실 확인 전",
            "부담이 너무 크지",
            "확실하지 않음",
            "목례",
            "수명 5년",
            "스크린타임은 없다고",
            "횡단보도",
            "전자기기 없이",
            "판타지 선택",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-commute-delivery-bed-dopamine-reality-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_commute_food_home_algorithm_relationship_variants_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("아침에 셔츠 입고 나왔는데 단추 하나 밀려 끼운 거 지하철 거울 보고 앎. 대수치.", "셔츠"),
            ("오늘 금요일인 줄 알고 신나게 눈 떴는데 목요일이더라. 침대에서 조용히 눈물 흘림.", "목요일"),
            ("지하철 환승할 때 문 열리자마자 반대편 승강장으로 전속력 질주하는 사람들 사이에 끼어서 같이 뜀.", "환승"),
            ("오전 내내 모니터 멍하니 보다가 눈 뻑뻑해서 인공눈물 넣었는데, 흘러내려서 화장 다 지워짐.", "인공눈물"),
            ("오늘 팀장님 출장 가셔서 사무실 공기 갑자기 쾌적해짐. 다들 소리 없는 환호 중.", "팀장님"),
            ("어제 산 로또 지갑에 넣어놓고 '당첨되면 내일부터 잠수 탄다'는 생각으로 오전 버팀.", "로또"),
            ("출근길에 에어팟 배터리 10% 남았다는 경고음 들릴 때의 그 절망감... 세상과 단절된 느낌.", "에어팟"),
            ("오늘 유독 일하기 싫어서 키보드 타자만 쓸데없이 타타타탁 세게 치면서 바쁜 척함.", "키보드"),
            ("학교/회사 갈 때 입을 옷 없어서 옷장 앞에서 15분 동안 서성이다가 결국 어제 입은 거 또 입고 나옴.", "옷장"),
            ("오늘 점심시간 끝나고 자리에 앉았는데 식곤증 와서 눈꺼풀에 모래 주머니 달아놓은 줄.", "식곤증"),
            ("엽떡 초보맛 시켰는데도 매워서 씁하씁하 하면서 쿨피스 한 통 다 비움. 맵찔이의 자존심 스크래치.", "엽떡"),
            ("오늘 저녁은 진짜 가볍게 샐러드 먹으려고 했는데, 집 가다 소금빵 냄새에 홀려서 3개 사 옴.", "소금빵"),
            ("배달 앱 번쩍배달로 시켰더니 라이더 위치 실시간으로 지도에서 움직이는 거 계속 쳐다보게 됨.", "라이더"),
            ("치킨 시킬 때 뼈로 시킬지 순살로 시킬지 고민하는 데만 10분 씀. 뼈 뜯는 맛이냐 편함이냐.", "순살"),
            ("요즘 밀키트 잘 나와서 요리 부심 부리면서 인스타에 올렸는데, 친구가 뒤에 버려진 포장지 찾아냄.", "밀키트"),
            ("여기 카페 디저트 비주얼 미쳤는데 가격이 밥값보다 비쌈. 배보다 배꼽이 더 크지만 흐린 눈.", "디저트"),
            ("단톡방에서 '오늘 뭐 먹지' 하면 결국 치킨, 삼겹살, 마라탕 셋 중 하나로 수렴됨.", "삼겹살"),
            ("어제 먹다 남은 피자 냉장고에 넣어놨던 거 에어프라이어에 돌려 먹는 게 갓 구운 것보다 맛있는 느낌.", "에어프라이어"),
            ("다이어트 선언하고 단톡방 나가겠다고 했는데, 3시간 뒤에 치킨 기프티콘 링크 들고 다시 들어옴.", "기프티콘"),
            ("오늘 편의점 신상 라면 먹어봤는데 원조 못 따라감. 역시 튜닝의 끝은 순정이다.", "신상 라면"),
            ("주말에 진짜 큰맘 먹고 카페 가서 노트북 켜놨는데, 아아 한 잔 마시면서 폰만 보다 옴.", "노트북"),
            ("침대에 누워서 유튜브 보다가 폰 얼굴에 정통으로 맞아서 코 뼈 주저앉는 줄 알았잖아.", "코뼈"),
            ("나 주말에 약속 취소되면 겉으로는 '아쉽다 ㅠㅠ' 하는데 속으로는 축제 열림. 집 최고.", "축제"),
            ("방 청소 하려고 서랍 열었다가 초등학교 때 쓰던 일기장 발견해서 2시간 동안 독서함.", "일기장"),
            ("스마트폰 충전기 선 짧아서 침대 모서리에 겨우 걸쳐 누워가지고 불편하게 폰 하는 중.", "충전기"),
            ("주말 내내 카톡 안 읽다가 일요일 밤늦게 '앗 미안 지금 봤다!' 시전하는 집순이의 삶.", "카톡"),
            ("배달 음식 받고 기사님 가실 때까지 문 뒤에서 숨죽이고 기다렸다가 조용히 문 여는 거 나만 이래?", "문 뒤"),
            ("일요일 오후 5시쯤 되면 슬슬 월요일 출근 생각나서 소화 안 되기 시작함.", "월요일"),
            ("방에 불 끄고 누웠는데 저 멀리 있는 멀티탭 불빛 유난히 거슬려서 결국 일어나서 끄고 옴.", "멀티탭"),
            ("넷플릭스 고르다가 1시간 지나서 결국 아무것도 못 보고 그냥 잘 시간 됨. '넷플릭스 증후군' 심각함.", "넷플릭스"),
            ("최애캐 이번 시즌 일러스트 뜬 거 보셨음? 흑발에 차가운 눈빛... 이건 서사 안 봐도 이미 서사 완성임.", "최애캐"),
            ("그 장르 공식 굿즈 한정판 티켓팅 열렸는데 1초 만에 이선좌(이미 선택된 좌석) 뜨고 광탈함.", "이선좌"),
            ("오늘 인스타 알고리즘이 자꾸 나한테 심리테스트 추천해 줘서 홀린 듯이 5개 연속으로 함.", "심리테스트"),
            ("노래방 최애곡 부르다가 고음 파트에서 삑사리 나서 조용히 볼륨 줄이고 흐느끼듯 부름.", "삑사리"),
            ("새벽 2시에 유튜브 알고리즘이 보여주는 '인도 길거리 음식' 영상 왜 이렇게 집중해서 보게 되냐.", "인도 길거리 음식"),
            ("좋아하는 웹툰 휴재 공지 뜬 거 보고 일주일 동안 삶의 의욕을 잃어버림.", "휴재"),
            ("스마트폰 스크린 타임 주간 리포트 날아왔는데 하루 평균 9시간 찍혀서 조용히 삭제함.", "9시간"),
            ("나 요즘 이 게임 메타 적응 못 하겠음. 내가 쓰던 최애 캐릭 완전 나락 가버렸어.", "최애 캐릭"),
            ("새벽에 감성 충만해져서 플레이리스트 짯는데 아침 출근길에 들으니까 오글거려서 못 듣겠음.", "플레이리스트"),
            ("웹소설 다음 화 결제하다가 통장 잔고 보고 정신 번쩍 듦. 내 도파민 비용 너무 비싸다.", "도파민"),
            ("회사 탕비실에서 과자 몰래 주머니에 가득 채워 나오다가 동기랑 마주쳐서 반씩 나눔.", "탕비실"),
            ("카톡 답장 고민하다가 머릿속으로 답장 보내놓고 실제로 전송 안 해서 3일 동안 잠수 탄 사람 됨.", "답장"),
            ("오늘 조별과제 피드백 받았는데 교수님이 '이건 누구 생각이죠?' 물어보셔서 다들 눈치 게임함.", "눈치 게임"),
            ("미용실에서 머리 샴푸해 줄 때 '더 헹구고 싶은 데 있으세요?' 하면 목에 힘 잔뜩 들어가서 '아녀...' 함.", "샴푸"),
            ("카톡방에 웃긴 짤 올렸는데 아무도 반응 안 해주고 대화 넘어가면 은근히 마음에 상처 입음.", "웃긴 짤"),
            ("오늘 스타벅스에서 닉네임 불리는데 예전에 장난으로 설정해 둔 '원빈님' 불려서 고개 숙이고 받으러 감.", "원빈님"),
            ("퇴근 직전에 급하게 온 메일 읽씹하고 퇴근함. 내일의 내가 알아서 하겠지 뭐.", "메일"),
            ("친구들이랑 여행 계획 짜는데 다들 '난 아무거나 다 좋아' 해놓고 숙소 링크 보내면 묘하게 딴지 걺.", "여행 계획"),
            ("오늘 하루 종일 생산적인 일 딱 하나 함: 영양제 챙겨 먹기. 건강은 챙겼으니 됐다.", "영양제"),
            ("드디어 주말이다! 단톡방에 생사 확인용 이모티콘 하나씩만 남기고 다들 침대로 해쳐모여!", "이모티콘"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "사실 확인 전",
            "챙겨본다고 단정",
            "스크린타임은 없다고",
            "와이파이 끊기",
            "공식 일러 비주얼",
            "목례",
            "읽씹이 더 선명",
            "판타지 선택",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-commute-food-home-algorithm-relation-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_commute_food_home_algorithm_relationship_variants_two_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("출근길 지하철에서 졸다가 고개 휙 꺾이면서 깼는데, 옆 사람 마주치기 민망해서 눈 감은 척 연기함.", "눈 감은 척"),
            ("아침에 옷 입다가 소매에 손 넣었는데 거꾸로 입은 거 알고 다시 벗을 때의 그 귀찮음과 자괴감.", "거꾸로"),
            ("오늘 목요일인 줄 알았는데 아직 화요일이더라. 달력 보고 시공간이 멈춘 줄 알았음.", "화요일"),
            ("오전 근무 중에 너무 졸려서 화장실 변기 커버 내리고 앉아서 5분 동안 눈 붙이고 옴. 꿀맛.", "화장실"),
            ("메일 보낼 때 '첨부파일 확인 부탁드립니다' 적어놓고 파일 첨부 안 해서 뒤늦게 '죄송합니다, 재송부합니다' 보냄.", "첨부파일"),
            ("퇴근 버스 탔는데 하필 에어컨 안 나오는 자리 당첨. 가방으로 부채질하면서 땀 뻘뻘 흘리는 중.", "에어컨"),
            ("오늘 아침엔 진짜 갓생 살려고 일찍 인났는데, 폰 좀 보다가 결국 평소보다 늦게 뛰어 나감.", "갓생"),
            ("월급날 통장 찍혔는데 카드값, 보험료, 대출 이자 순서대로 '퍼가요~' 당하더니 30분 만에 원상 복구됨.", "월급날"),
            ("오늘 회의 때 아무 생각 없이 영혼 나간 눈으로 끄덕거렸는데, 갑자기 질문 들어와서 동공 지진 일어남.", "동공"),
            ("퇴근길에 하늘 보는데 유난히 노을 예쁘더라. 근데 사진 찍으면 눈으로 보는 그 갬성이 안 담김.", "노을"),
            ("마라탕 가게에서 집게 들고 야채 담을 땐 가벼웠는데, 계산대 올려놓으니 2만 원 넘게 나와서 당황.", "마라탕"),
            ("오늘 저녁은 진짜 소식하려고 두부 샐러드 먹었는데, 밤 11시에 배고파서 결국 짜파게티 끓임.", "짜파게티"),
            ("치킨 배달 왔는데 콜라 업그레이드 깜빡함. 냉장고에 든 유통기한 간당간당한 탄산수 뒤지는 중.", "콜라"),
            ("카페에서 음료 주문하고 진동벨 울릴 때까지 벨 빤히 쳐다보고 있는 거 은근히 중독성 있음.", "진동벨"),
            ("단톡방에 '나 방금 빵 5만 원어치 지름' 올렸더니 다들 '그 돈이면 국밥이 몇 그릇이냐'며 T식 타박함.", "국밥"),
            ("다이어트 한답시고 제로 콜라, 제로 슈거 과자 잔뜩 사 왔는데 코끼리처럼 많이 먹어서 무소용.", "제로"),
            ("스트레스 한 바가지 받은 날엔 엽떡 오리지널 맛에 명랑핫도그 감자통모짜 감싸서 먹어야 풀림.", "명랑핫도그"),
            ("고깃집에서 볶음밥 시켰는데 밑에 눌어붙은 누룽지 숟가락으로 긁어먹을 때가 제일 맛있음.", "누룽지"),
            ("요즘 요리 브이로그 보고 삘 받아서 파스타 만들었는데, 비주얼은 개밥이고 양은 4인분 나옴.", "파스타"),
            ("새로 생긴 핫플 카페 갔는데 의자가 너무 낮고 테이블이 무릎 높이라 허리 디스크 오는 줄.", "핫플"),
            ("주말에 집에서 입는 티셔츠 목 다 늘어나서 어깨까지 내려오는데 세상에서 제일 편함.", "티셔츠"),
            ("침대에 누워서 한쪽 옆으로만 누워 폰 하다가 귀 먹먹해지고 팔 저려서 반대쪽으로 턴함.", "팔 저리"),
            ("주말 약속 잡을 땐 분명 신났는데, 당일 아침 되니까 '그냥 취소해 줬으면 좋겠다'는 간절한 소망이 생김.", "약속"),
            ("방 청소 하다가 고장 난 이어폰, 옛날 충전기 선 뭉치 발견했는데 '언젠가 쓰겠지' 하고 다시 서랍에 넣음.", "충전기"),
            ("넷플릭스 켜놓고 폰으로 인스타 릴스 보는 중. 눈은 하난데 도파민은 두 배로 채우는 현대인의 삶.", "도파민"),
            ("금요일 저녁 퇴근하고 씻지도 않고 침대에 대자로 뻗어서 한 시간 동안 인공호흡 하는 시간.", "금요일"),
            ("택배 문 앞에 두고 가셨대서 문 살짝 열고 손만 쑥 내밀어서 상자 낚아채 오기 만렙 달성.", "택배"),
            ("일요일 저녁 8시 예능 프로그램 끝나는 음악 소리 들리면 갑자기 우울해지면서 월요병 시동 걸림.", "월요병"),
            ("불 끄고 누웠는데 에어컨 리모컨 저 멀리 책상 위에 있을 때... 갈등하다가 결국 추위에 떨며 잠.", "리모컨"),
            ("유튜브에서 '방구석 인테리어 꿀팁' 한 시간 동안 집중해서 보고 내 방 둘러본 뒤 조용히 컴터 켬.", "인테리어"),
            ("이번 신작 캐릭터 비주얼 미쳤음. 백발에 벽안 조합은 치트키잖아. 눈빛에 서사가 삼천 페이지임.", "백발"),
            ("인스타 돋보기 탭에 자꾸 나랑 상관없는 육아 채널이나 커플 릴스 뜨는데 알고리즘 왜 이래?", "커플 릴스"),
            ("새벽 1시에 갑자기 삘 받아서 2000년대 감성 발라드 플레이리스트 듣다가 혼자 아련해짐.", "2000년대"),
            ("게임 경쟁전 돌렸는데 5연패 박음. 팀원 탓하다가 마지막 판에 내 플레이 보고 조용히 게임 끔.", "5연패"),
            ("새벽에 유튜브 알고리즘이 추천해 준 '자연의 신비: 거대 개미집 발굴' 영상 40분째 넋 놓고 봄.", "거대 개미집"),
            ("즐겨보던 웹툰 완결 났는데 후기 보면서 내 자식이 졸업한 것마냥 마음이 몽글몽글하고 헛헛함.", "완결"),
            ("주간 스크린 타임 보고 '나 진짜 폰 중독이네' 생각하면서 손으로는 계속 카톡 창 새로고침 중.", "스크린 타임"),
            ("나 요즘 이 게임 메타 도저히 못 따라가겠음. 패치 한 번에 내 본캐 고인 되고 사기캐만 판침.", "본캐"),
            ("새벽에 인스타 공구 보다가 '이건 진짜 사야 해' 하고 홀린 듯 카드 번호 입력하는 내 손가락.", "인스타 공구"),
            ("웹소설 다음 화 보려고 쿠키(캐시) 충전하다가 이번 달 누적 금액 보고 등골 서늘해짐.", "쿠키"),
            ("회사 탕비실에서 과자 가방에 슬쩍 챙기다가 미화 이모님이랑 눈 마주쳐서 '이거 맛있어요!' 하고 추천함.", "미화 이모님"),
            ("카톡 알림 떴는데 미리보기로 다 읽어놓고, 답장 타이밍 놓쳐서 반나절 뒤에 안 읽은 척 답장함.", "미리보기"),
            ("오늘 조별과제 회의하는데 다들 마이크 끄고 채팅으로만 '넵', '좋습니다' 치고 있어서 AI랑 대화하는 줄.", "AI"),
            ("미용실에서 머리 다 자르고 거울 보여주는데 마음에 안 들지만 세상 무해한 미소로 '마음에 들어요!' 함.", "마음에 들어요"),
            ("단톡방에 엄청 공들여서 웃긴 드립 쳤는데 다들 'ㅋㅋㅋ' 세 개만 치고 다음 화제로 넘어가서 내적 시무룩.", "ㅋㅋㅋ"),
            ("스타벅스에서 사이렌 오더 주문할 때 부끄러운 닉네임 불릴까 봐 영수증 번호로 불러달라고 기도함.", "사이렌 오더"),
            ("퇴근 직전에 메신저로 급한 일 오면 '아, 지금 외근 중이라 확인이 어렵습니다' 거짓말 치고 버스 탐.", "외근"),
            ("친구들이랑 단톡방에서 '야 언제 한번 보자!' 말만 삼백 번째 하는 중. 사실상 사이버 친구임.", "사이버 친구"),
            ("오늘 하루 종일 유일하게 몸 움직인 거: 침대에서 배달 음식 받으러 현관문까지 걸어갔다 온 것.", "현관문"),
            ("드디어 불금이다! 다들 단톡방에 '주말 잘 보내세요' 이모티콘 매크로 돌리고 칼같이 퇴근하자!", "이모티콘"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "챙겨본다고 단정",
            "스크린타임은 없다고",
            "공식 일러 비주얼",
            "목례",
            "읽씹이 더 선명",
            "와이파이 끊기",
            "판타지 선택",
            "수명 5년",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-commute-food-home-algorithm-relation-2-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_commute_food_home_algorithm_relationship_variants_three_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("아침에 급하게 나오느라 양말 짝짝이로 신은 거 버스 안에서 발견함. 묘하게 톤 다운된 회색과 검은색이라 흐린 눈 하는 중.", "양말 짝짝이"),
            ("지하철 타고 가는데 옆 사람이 내 어깨에 기대서 졸 때, 밀쳐내기도 애매해서 나도 같이 어깨 로봇처럼 굳어짐.", "어깨"),
            ("오늘 금요일 분위기 내면서 신나게 출근했는데 알고 보니 수요일일 때의 그 영혼 탈곡 느낌.", "수요일"),
            ("오후 3시쯤 되면 모니터 글씨가 흐려지면서 안 보임. 눈이 나빠진 게 아니라 그냥 퇴근하고 싶은 거임.", "오후 3시"),
            ("메일 보낼 때 '감사합니다' 오타 나서 '감사합니디'로 보낸 거 나중에 발송 취소도 안 될 때의 찝찝함.", "감사합니디"),
            ("퇴근길 버스 창문에 머리 콩콩 박으면서 졸다가, 내릴 역 안내방송 듣고 심장 쿵쾅거리면서 스프링처럼 일어남.", "내릴 역"),
            ("오늘 아침엔 진짜 10분 일찍 나가서 커피 사 가려고 했는데, 침대에서 밍기적거리다가 평소 타던 버스도 놓칠 뻔.", "커피"),
            ("통장에 월급 찍히자마자 숨도 쉬기 전에 적금이랑 고정지출로 다 찢어져서 나가고 소액만 남았을 때의 허무함.", "고정지출"),
            ("회의 시간에 멍 때리다가 내 이름 불려서 나도 모르게 \"네! 그 부분은 확인해 보겠습니다!\" 하고 AI 자동 반사 때림.", "AI"),
            ("퇴근길에 오늘 하늘 진짜 역대급이라 폰 꺼내서 찍었는데, 화면에는 웬 가로등 불빛 번진 이상한 사진만 남아있음.", "가로등"),
            ("마라탕에 소고기 추가는 기본이고 꿔바로우 소(小)자 시킬지 말지 계산대 앞에서 마지막까지 내적 갈등함.", "꿔바로우"),
            ("오늘 점심 가볍게 서브웨이 먹었는데, 소스 올리브오일이랑 소금만 하려다가 결국 랜치랑 사우스웨스트 듬뿍 뿌림.", "서브웨이"),
            ("치킨 배달 왔는데 치킨무 국물 버리러 싱크대 가기 귀찮아서 그냥 거실 바닥에 비닐 깔고 조심조심 뜯음.", "치킨무"),
            ("카페 진동벨 울리자마자 가지러 가기 묘하게 민망해서 3초 정도 딴청 피우다가 슬금슬금 일어남.", "진동벨"),
            ("단톡방에 '나 방금 마트에서 간식만 3만 원어치 삼' 하니까 친구들이 '그 돈이면 뜨끈한 국밥이 몇 그릇이냐' 시전.", "국밥"),
            ("다이어트식으로 곤약 볶음밥 샀는데, 양이 너무 적어서 결국 두 팩 뜯어 데워 먹고 칼로리 채움.", "곤약 볶음밥"),
            ("스트레스 만빵인 날엔 닭발에 주먹밥 뭉쳐서 엽떡 국물에 찍어 먹는 게 합법적인 치료제임.", "닭발"),
            ("삼겹살 다 먹고 K-디저트인 볶음밥 볶을 때, 이모님이 가위질하는 손놀림 다들 경건하게 쳐다보고 있음.", "이모님"),
            ("요리 채널 보고 자취생 간단 레시피 따라 했는데, 설거지거리만 한 바가지 나오고 맛은 니맛도 내맛도 아님.", "설거지"),
            ("인스타 핫플 카페 갔는데 테이블이 너무 낮아서 강제로 폴더처럼 접혀서 가쁜 숨 쉬며 커피 마심.", "핫플"),
            ("집에서 입는 무릎 늘어난 수면바지랑 목 늘어난 티셔츠 조합, 솔직히 누더기 같지만 절대 못 버림.", "수면바지"),
            ("침대에 옆으로 누워서 너튜브 보다가 눈물 한 방울 옆으로 흘러내리는 거 나만 그런 거 아니지?", "눈물"),
            ("주말에 만나자고 약속 잡을 땐 괜찮았는데, 전날 밤부터 '제발 상대방이 급한 일 생겼다고 취소해라' 기도함.", "취소 기도"),
            ("방 청소 하려고 서랍 뒤지다가 고딩 때 쓰던 PMP나 mp3 발견해서 충전기 꽂아보느라 청소 중단됨.", "PMP"),
            ("거실 TV로 넷플릭스 틀어놓고 한쪽 눈으로는 스마트폰 쇼츠 넘기는 진정한 도파민 멀티태스킹.", "도파민"),
            ("불금 퇴근하고 집에 오자마자 가방 바닥에 던지고 침대에 쓰러져서 30분 동안 시체 놀이 하는 시간.", "시체 놀이"),
            ("배달 기사님이 '문 앞에 두고 갑니다' 문자 남기자마자 0.5초 만에 문 열고 음식 낚아채 오기.", "0.5초"),
            ("일요일 저녁에 개그 프로그램 끝나는 음악이나 주말 드라마 엔딩 크레딧 올라가면 슬슬 명치 턱 막힘.", "엔딩 크레딧"),
            ("불 다 끄고 침대에 누웠는데 방 전등 리모컨이 저 멀리 책상 위에 있을 때... 갈등하다가 결국 그냥 눈 감음.", "전등 리모컨"),
            ("너튜브로 '미니멀리스트 방 꾸미기' 영상 한 시간 동안 보면서 대리만족하고 내 방 쓰레기장 보며 한숨 쉼.", "미니멀리스트"),
            ("이번에 공개된 캐릭터 비주얼 미쳤더라. 백발에 푸른 눈 조합은 사기지, 눈빛에 서사가 삼천 페이지 흐름.", "백발"),
            ("인스타 알고리즘이 자꾸 나한테 내가 보지도 않는 커플 릴스나 아기 영상 추천해 줄 때 묘한 거부감 듦.", "커플 릴스"),
            ("새벽 2시에 갑자기 감성 차올라서 싸이월드 시절 인디 노래 플레이리스트 듣다가 혼자 새벽 감성 충만해짐.", "싸이월드"),
            ("게임 연패 박고 빡쳐서 '아 롤 지운다 진짜' 하고 삭제했다가 다음 날 저녁에 조용히 다시 다운로드함.", "롤"),
            ("새벽에 알고리즘이 인도 길거리 거대 야자수 자르는 영상 추천해 줘서 넋 놓고 끝까지 다 봄.", "거대 야자수"),
            ("몇 년 동안 보던 웹툰 완결 나니까 마지막 화 후기 보면서 마음 한구석이 헛헛하고 먹먹해짐.", "완결"),
            ("주간 스크린 타임 리포트 떴는데 하루 평균 8시간 넘는 거 보고 '나 폰 중독이네' 하면서 손은 계속 스크롤 내림.", "8시간"),
            ("요즘 게임 메타 너무 자주 바뀌어서 적응 불가. 내가 애정으로 키운 본캐 버프 좀 해줘라 진짜.", "본캐"),
            ("새벽에 쇼핑몰 구경하다가 '이건 한정 수량이라 지금 안 사면 품절' 문구에 홀려 번개같이 결제함.", "한정 수량"),
            ("웹소설 다음 화 궁금해서 소액 결제 계속 누르다가 한 달 누적 금액 보고 등골 서늘해져서 앱 종료.", "소액 결제"),
            ("회사 탕비실에서 새로 들어온 고급 과자 주머니에 슬쩍 넣다가 동기랑 눈 마주쳐서 하나 건네줌.", "고급 과자"),
            ("카톡 알림 왔을 때 상단 바 미리보기로 다 읽어놓고, 바로 답장하기 싫어서 3시간 뒤에 '앗 이제 봤어!' 시전.", "미리보기"),
            ("오늘 조별과제 단톡방에서 아무도 말 안 하길래 눈치 보다가 '넵' 하나 치니까 아래로 '넵' 주르륵 달림.", "넵"),
            ("미용실에서 머리 감겨줄 때 지압해 주면 귀 뒤쪽 간지러운데 꾹 참느라 온몸에 힘 잔뜩 들어감.", "지압"),
            ("단톡방에 야심 차게 기발한 드립 쳤는데 다들 아무 반응 없고 다른 얘기로 넘어가면 혼자 이불 킥함.", "이불 킥"),
            ("스타벅스에서 닉네임 불릴 때 예전에 장난으로 해둔 '귀요미님' 불려서 세상에서 제일 빠른 걸음으로 받아 옴.", "귀요미님"),
            ("퇴근 5분 전에 상사가 '잠깐 얘기 좀 할까?' 하면 속으로 온갖 쌍욕 다 하면서 겉으로는 자본주의 미소 지음.", "자본주의 미소"),
            ("친구들이랑 '조만간 진짜 얼굴 한번 보자!' 멘트만 1년째 날리는 중. 사실상 텍스트 속 동반자임.", "텍스트 속 동반자"),
            ("오늘 하루 종일 제일 길게 이동한 거리: 침대에서 냉장고 들렀다가 화장실 거쳐서 다시 침대로 온 것.", "냉장고"),
            ("드디어 불금 퇴근이다! 다들 단톡방에 '고생하셨습니다' 매크로 이모티콘 띡 던지고 칼같이 로그아웃!", "고생하셨습니다"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "목례",
            "수명 5년",
            "택시",
            "자녀",
            "원빈님",
            "커피머신",
            "자는 척",
            "영수증 번호",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-commute-food-home-algorithm-relation-3-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_sf_game_design_travel_dialogue_a_lines_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("야, 근데 진짜 인공지능이 더 발전해서 사람처럼 감정 가지면, 걔네한테도 인권 줘야 되나?", "인권"),
            ("근데 만약 걔가 나한테 삐지거나 서운하다고 하면 진짜 사람 대하는 것처럼 기분 묘할 것 같긴 해. 좀 무섭기도 하고.", "서운"),
            ("이번 패치로 밸런스 완전 무너졌던데? 그 챔피언 후반 벨류가 선을 넘었어.", "후반 벨류"),
            ("그니까. 차라리 초반에 라인전 세게 압박해서 타워 빨리 밀어버리는 조합으로 가야 해. 눕는 조합 했다가는 답도 없음.", "라인전"),
            ("이번에 공개된 캐릭터 디자인 진짜 취향 저격이더라. 분위기가 다했음.", "캐릭터 디자인"),
            ("그 과하지 않은 대비감? 화이트나 블론드 헤어에 맑은 블루 아이 조합인데, 차분하면서도 이지적인 느낌이 확 살더라고. 서사도 되게 깊어 보이고.", "블루 아이"),
            ("나 요즘 진짜 어디론가 훌쩍 떠나고 싶다. 제주도 같은 데 내려가서 한 한 달 동안 바다만 보고 오면 안 되나?", "제주도"),
            ("아니, 아무것도 안 하고 그냥 노트북 하나 들고 가서 낮에는 조용한 카페에서 작업하고, 밤에는 파도 소리 들으면서 맥주나 마시는 거지. 상상만 해도 힐링이다.", "파도 소리"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "말 안 하기를 고를래",
            "흑발에 적안",
            "스마트폰을 끊는 쪽",
            "권리 논의를 피할 수 없다고 봐",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-sf-game-design-travel-dialogue-a-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_game_design_ai_hardware_online_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("이번 패치 보니까 후반 밸류 선 넘었던데? 초반에 아무리 말려놔도 스택 쌓이니까 답이 없음.", "후반 밸류"),
            ("눕는 조합 했다가 상대방 왕귀형 챔피언한테 광역 딜 맞고 5초 만에 한타 터짐. 진짜 킹받네.", "왕귀형"),
            ("라인전 세게 압박해서 스노우볼 굴릴 거 아니면 차라리 초반 안정성 높은 픽으로 눕는 게 나음.", "스노우볼"),
            ("아니, 이 타이밍에 바론 치는 게 맞냐? 굳이 안 줘도 될 턴을 줘서 역전 빌미를 만드네.", "바론"),
            ("요즘 메타는 뇌 빼고 싸우는 것보다 후반 도모하면서 라인 관리 꼼꼼히 하는 쪽이 무조건 이김.", "라인 관리"),
            ("상대 미드 로밍력 미쳤던데, 우리 바텀 레이더 안 켜고 라인 밀다가 계속 더블킬 상납 중.", "미드 로밍"),
            ("이번 신캐 성능 에바임. 이 정도면 출시 초기 버프 감안해도 핫픽스 들어가야 되는 수준.", "핫픽스"),
            ("우리 팀 정글 동선 다 읽혔는데 끝까지 카정 가다가 퍼블 주네. 조용히 차단 박고 내 라인 집중한다.", "정글 동선"),
            ("딜 그래프 보셈. 나 이번 판에 딜 지분 40% 넣었는데 지는 게 말이 되냐? 팀운 역대급이네.", "딜 지분"),
            ("오늘 저녁에 접속하면 밴 카드 무조건 그쪽으로 빼라. 걔 풀어주면 라인전 단계부터 숨도 못 쉼.", "밴 카드"),
            ("이번 신작 캐릭터 일러스트 떴는데 흑발에 적안 조합이더라. 솔직히 이건 비주얼 치트키지.", "흑발"),
            ("난 화려한 것보다 백발에 맑은 블루 아이 조합처럼 차분하면서도 이지적인 분위기가 더 끌림.", "블루 아이"),
            ("캐릭터 디자인할 때 머리색이랑 눈색으로 확실하게 대비감 주는 게 Persona가 확 살아나는 듯.", "Persona"),
            ("그 캐릭터 특유의 무심하고 차가운 눈빛이 진짜 매력적임. 서사 안 봐도 이미 삼천 페이지 뚝딱이다.", "삼천 페이지"),
            ("과한 장식 다 빼고 흑백 톤으로 딱 떨어지는 옷 입혀놓은 게 훨씬 고급스러워 보임.", "흑백 톤"),
            ("이번 공식 일러 배경화면 각이다. 색감 대비를 어쩜 이렇게 감각적으로 잘 썼지?", "색감 대비"),
            ("성격은 되게 과묵하고 냉철한데 가끔 보여주는 인간적인 모먼트... 이 갭모에에 치이는 거임.", "갭모에"),
            ("디자인이 아무리 예뻐도 인게임 모델링 뭉개지면 무소용인데, 이번엔 모델링도 깎아지게 잘 뽑힘.", "인게임 모델링"),
            ("그 장르 공식 설정집 보는데 캐릭터 고유 컬러칩 지정해 둔 거 보고 변태 같은 디테일에 감탄함.", "컬러칩"),
            ("내 최애캐 서사가 너무 매콤해서 눈물 흘리는 중. 비주얼은 고결한데 과거사 굴려진 거 맛도리네.", "과거사"),
            ("요즘 인공지능 발전 속도 보면 무서움. 나중엔 진짜 말 잘 통하는 AI companion 하나 두고 살 듯.", "AI companion"),
            ("AI가 인간의 감정을 완벽하게 모방하게 되면, 걔가 서운하다고 할 때 진짜 사람 대하듯 미안할까?", "서운"),
            ("코드 몇 줄로 성격 데이터셋 짜서 캐릭터 커스텀 할 수 있는 시대인데, 굳이 인간관계 스트레스 받을 필요 있나.", "성격 데이터셋"),
            ("로컬로 가볍게 돌릴 수 있는 LLM들 퀄리티 엄청 올라왔더라. 개인 보안 생각하면 로컬이 답임.", "LLM"),
            ("학습 데이터에 노이즈 조금만 껴도 답변 완전 나락 가던데, 데이터 정제가 진짜 핵심인 듯.", "데이터 정제"),
            ("AI가 짜준 코드 그대로 복붙했다가 에러 터져서 2시간 동안 디버깅함. 역시 맹신하면 안 됨.", "디버깅"),
            ("나중에 뇌를 디지털 코드로 변환해서 서버에 업로드할 수 있다면 너는 갈 거냐? 난 무조건 감.", "디지털 코드"),
            ("자동화가 너무 잘 되니까 편리하긴 한데, 가끔은 인간 고유의 영역이 어디까지인가 진지하게 고민됨.", "자동화"),
            ("단톡방에 프롬프트 잘 짜는 법 공유해 줬더니 다들 신기해함. 이게 바로 미래형 스몰토크인가.", "프롬프트"),
            ("AI가 그린 그림이랑 사람이 그린 그림 구별하는 거 이제 의미 없는 수준까지 온 것 같아.", "AI 그림"),
            ("이번에 그래픽카드 업그레이드할까 고민 중인데 VRAM 용량이 마음에 걸리네. 16GB는 돼야 여유롭겠지?", "VRAM"),
            ("고성능 작업 좀 돌렸더니 본체에서 비행기 이륙하는 소리 남. 쿨러를 수랭으로 바꿔야 하나.", "쿨링"),
            ("모니터 주사율 60Hz 쓰다가 144Hz로 바꿨는데 신세계임. 마우스 커서 움직임부터 부드러움이 다름.", "144Hz"),
            ("C드라이브 용량 부족해서 보니까 알게 모르게 쌓인 캐시 파일만 몇십 기가 나오더라. 조용히 싹 밀었음.", "C드라이브"),
            ("기계식 키보드 축 바꾸고 싶다. 갈축 쓰는데 찰칵거리는 청축의 손맛이 그리워짐.", "청축"),
            ("책상 위에 선 꼬여있는 거 보기 싫어서 무선 제품으로 다 바꾸는 중. 데스크테리어의 완성은 무선임.", "데스크테리어"),
            ("고화질 에셋 추출해서 분석해보려고 하니까 디스크 읽기 속도가 못 따라감. SSD 성능 좋은 거 마렵다.", "SSD"),
            ("컴퓨터 견적 짜다 보니 욕심 끝도 없음. 정신 차려 보니까 예산 초기 설정한 거에 두 배 찍혀있네.", "예산"),
            ("장비는 프로급인데 정작 내 손가락 성능이 브론즈라 장비한테 미안할 지경임.", "브론즈"),
            ("외장 하드 정리하다가 10년 전 파일들 발견함. 백업의 중요성을 다시 한번 깨닫는 순간.", "외장 하드"),
            ("그 웹툰 이번 에피소드 연출 미쳤음. 컷 전환할 때 여백 쓰는 거 보고 작가 천재인 줄 앎.", "여백"),
            ("오랜만에 리버스 엔지니어링 느낌으로 게임 내부 데이터 뜯어보는 중인데 구조 짜놓은 거 신기함.", "리버스 엔지니어링"),
            ("단톡방에 어그로 끄는 빌런 등장했는데 다들 먹이 안 주고 병먹금(병신에게 먹이 금지) 잘하는 거 개웃김.", "병먹금"),
            ("스마트폰 스크린 타임 줄이려고 앱 잠금 걸어놨는데 비번 내가 풀고 디시, 아카 스크롤 내리는 중.", "디시"),
            ("인터넷 커뮤니티 유행어 나도 모르게 오프라인에서 입 밖으로 튀어나오려 할 때 식은땀 흐름.", "유행어"),
            ("내가 좋아하는 마이너 장르 공식 굿즈 수요 조사 떴는데 제발 최소 수량 채워서 제작 확정됐으면.", "수요 조사"),
            ("새벽 3시에 나무위키에서 '사라진 고대 문명' 이런 거 타고 타고 들어가다가 밤샘. 지식 유목민임.", "나무위키"),
            ("이 게임 처음에 모드(Mod)질 하려고 켰다가 충돌 나서 무한 튕김 발생. 결국 순정으로 돌리는 엔딩.", "모드"),
            ("요즘 도파민 중독이라 영상 배속 안 하면 답답해서 못 보겠음. 자극적인 것만 찾는 뇌가 됐다.", "도파민"),
            ("내 취향 100% 반영된 인디 게임 찾았을 때의 그 짜릿함... 아무한테도 안 알려주고 나만 알고 싶다.", "인디 게임"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "유전자 업그레이드",
            "너를 동물로 비유",
            "엘리베이터 안",
            "게임 얘기면 조작감",
            "비트코인 풀매수",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-game-design-ai-hardware-online-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_ai_hardware_game_design_detail_variants_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("LoRA 가중치 설정 값 조금씩 바꿔가면서 테스트 중인데, 과적합(Overfitting) 나서 기계 인간 됨. 살려줘.", "LoRA"),
            ("단조로운 캐릭터는 금방 질리더라. 차분한 카테고리 안에서 한 명은 논리적이고 한 명은 감성적인 식으로 결을 나눠야 재밌음.", "결"),
            ("합성 데이터(Synthetic Data) 만들 때 기본 프롬프트가 오염되면 생성된 데이터 다 버려야 됨. 싱크대 행.", "합성 데이터"),
            ("GGUF 양자화 포맷 다 좋은데, 멀티 LoRA 어댑터 적용하려니까 단일 모델 구조라 머리 터질 것 같음.", "GGUF"),
            ("프롬프트 시스템 지시문(System Prompt)에 규칙 10개 박아놨더니 지들끼리 모순 생겨서 뇌 정지 오더라.", "시스템 지시문"),
            ("모니터 암 달아서 코딩 창 세로로 돌려놓으니까 소스 코드 한눈에 들어와서 가독성 미쳤음.", "모니터 암"),
            ("컴퓨터 본체 LED 불빛 유난히 거슬려서 전용 프로그램으로 아예 다 끄고 누드 감성으로 쓰는 중.", "LED"),
            ("고성능 연산 돌려놓고 방 안 온도 올라가서 에어컨 미리 켰잖아. 컴터가 방을 데우는 보일러임.", "보일러"),
            ("장비 세팅하느라 몇 날 며칠 밤샜는데 막상 다 세팅하고 나면 흥미 식어서 멍 때리는 병에 걸림.", "장비 세팅"),
            ("아니, 바론 한타에서 탱커 뒤에 숨어있다가 후반 치명타 한 방 터지니까 아군 딜러진 다 녹아내림.", "바론 한타"),
            ("요즘 메타는 무지성 교전 유도하는 것보다 철저하게 오브젝트 턴 계산해서 이득 챙기는 쪽이 이김.", "오브젝트 턴"),
            ("상대 바텀 조합 scaling 주시하고 있었는데 우리 정글이 역동선 짜서 초반 갱 타이밍 다 놓침. 단체 뇌 정지.", "역동선"),
            ("성격은 냉철하고 이성적인데 내 사람한테만 미세하게 유해지는 서사... 이건 맛이 없을 수가 없다.", "내 사람"),
            ("설정집 보는데 고유 컬러칩이랑 엠블럼 매칭해 둔 디테일 보고 과몰입 씨게 옴.", "엠블럼"),
            ("메모장 적을 때도 줄 바꿈이랑 들여쓰기 칼같이 맞추느라 정작 중요한 내용은 몇 줄 못 적음.", "들여쓰기"),
            ("할 일 목록(To-Do List) 짜는 데만 완벽하게 2시간 쓰고, 정작 첫 번째 일 시작도 전에 지쳐서 눕는 엔딩.", "To-Do List"),
            ("남들이 보면 별 차이 없다는 코드 한 줄, 픽셀 1개 크기 차이에 집착하다가 밤샘. 피곤한 성격임.", "픽셀 1개"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "어느 지역 기준",
            "목록은 가볍게",
            "피곤하면 오늘은 템포",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-ai-hardware-game-design-detail-variant-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_foundation_admin_home_health_social_transport_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("주민센터 가서 서류 하나 떼야 하는데 평일 낮에만 열어서 직장인은 진짜 시간 내기 빡세다.", "주민센터"),
            ("택배 반품하려고 송장 붙여놨는데 기사님 오는 날마다 내가 집에 없어서 계속 밀림.", "택배 반품"),
            ("여권 사진 찍었는데 얼굴이 너무 낯설게 나와서 이걸 10년 들고 다녀야 하나 고민 중.", "여권 사진"),
            ("은행 OTP 배터리 나가서 로그인도 못 하고 고객센터 전화만 20분째 붙잡고 있음.", "OTP"),
            ("보험금 청구하려고 영수증 모아놨는데 서류 이름이 다 비슷해서 뭐가 뭔지 모르겠어.", "보험금"),
            ("세탁소 맡긴 코트 찾으러 가야 하는데 영수증 어디 뒀는지 기억이 안 난다.", "세탁소"),
            ("건강검진 예약하려고 보니까 가능한 날짜가 죄다 평일 오전이라 직장인 울어요.", "건강검진"),
            ("치과 스케일링 예약 잡아놓고 벌써부터 그 위잉 소리 상상돼서 긴장됨.", "스케일링"),
            ("운전면허 갱신 문자 왔는데 미루다가 마감일 가까워져서 갑자기 급해짐.", "운전면허 갱신"),
            ("병원 진료 예약 시간 맞춰 갔는데 대기실에서 40분째 앉아있으니 멍해진다.", "대기실"),
            ("분리수거 하러 내려갔는데 플라스틱에 붙은 라벨 떼다가 현타 왔어.", "분리수거"),
            ("음식물 쓰레기 버리러 나가야 하는데 냄새 맡을 생각만 해도 미루고 싶다.", "음식물 쓰레기"),
            ("화장실 배수구에 머리카락 쌓인 거 봤는데 내가 치워야 한다는 사실이 제일 슬픔.", "배수구"),
            ("냉장고 열었더니 언제 산지 모를 반찬통이 뒤에서 유물처럼 발견됨.", "반찬통"),
            ("냉동실 정리하다가 작년에 얼려둔 떡 발견했는데 먹어도 되는 건지 모르겠다.", "냉동실"),
            ("에어컨 필터 청소해야지 생각만 3주째 하고 아직도 먼지랑 동거 중.", "에어컨 필터"),
            ("겨울옷 넣고 여름옷 꺼내야 하는데 옷장 정리 시작할 엄두가 안 난다.", "옷장 정리"),
            ("비 맞은 우산을 말려야 하는데 현관에 그냥 세워놨더니 바닥이 축축해졌어.", "우산"),
            ("이불 빨래 돌렸는데 말릴 공간이 없어서 방 안이 빨래방이 됐다.", "이불 빨래"),
            ("청소기 돌리려고 했는데 먼지통 비우는 게 귀찮아서 청소가 시작도 안 됨.", "먼지통"),
            ("오늘 목이 너무 뻐근해서 고개 돌릴 때마다 삐걱거리는 로봇 된 느낌이야.", "목"),
            ("하루 종일 모니터 봤더니 눈이 뻑뻑하고 초점이 흐릿해진다.", "눈"),
            ("어제 스쿼트 좀 했다고 계단 내려갈 때 허벅지가 비명을 지르네.", "허벅지"),
            ("잠을 많이 잤는데도 계속 졸려. 이 정도면 침대가 나를 소환하는 것 같아.", "침대"),
            ("속이 더부룩해서 저녁을 가볍게 먹어야 하는데 떡볶이 생각이 자꾸 난다.", "더부룩"),
            ("카페인 줄이려고 했는데 오전 회의 두 개 듣고 바로 커피 사러 나감.", "커피"),
            ("비타민 챙겨 먹으려고 샀는데 책상 위 장식품 된 지 한 달 됐어.", "비타민"),
            ("운동하러 가야 하는데 운동복 갈아입는 단계가 제일 큰 장벽이야.", "운동복"),
            ("밤에 폰 보지 말자고 해놓고 불 끄자마자 쇼츠 켜는 나 자신이 싫다.", "쇼츠"),
            ("스트레칭 5분만 해도 몸이 살짝 풀리는데 왜 매일 까먹는 걸까.", "스트레칭"),
            ("친구가 약속 시간 20분 늦는다는데 이미 익숙해서 화도 안 난다.", "20분"),
            ("생일 선물 골라야 하는데 센스 없어 보일까 봐 장바구니만 계속 바꾸는 중.", "생일 선물"),
            ("단톡방에서 다들 읽기만 하고 답을 안 해서 내가 또 총대 멜 분위기야.", "총대"),
            ("친구가 고민 얘기하는데 조언을 해야 할지 그냥 들어줘야 할지 모르겠어.", "들어"),
            ("영화관에서 앞사람이 계속 폰 켜서 화면 밝기 보이니까 집중이 깨짐.", "영화관"),
            ("카페에서 옆자리 사람이 너무 큰 소리로 통화해서 내 노이즈캔슬링도 졌다.", "통화"),
            ("결혼식 축의금 얼마 해야 할지 애매한 사이가 제일 어렵다.", "축의금"),
            ("오랜만에 연락 온 사람이 갑자기 보험 얘기 꺼내서 마음의 문이 닫힘.", "보험"),
            ("친구가 빌려간 돈을 까먹은 척하는데 내가 먼저 말하기도 민망하다.", "빌려간 돈"),
            ("모임에서 나만 말을 너무 많이 한 것 같아서 집에 와서 혼자 반성 중.", "모임"),
            ("버스 눈앞에서 놓쳤는데 다음 차 18분 뒤라 그냥 하늘만 봤다.", "18분"),
            ("지하철 환승 통로가 너무 길어서 이건 이동이 아니라 등산 같아.", "환승"),
            ("택시 잡으려는데 할증 붙는 시간이라 앱 화면 보고 조용히 버스 정류장 감.", "할증"),
            ("비 오는 날 신발 젖은 채로 하루 종일 돌아다니는 그 찝찝함 진짜 싫다.", "신발"),
            ("교통카드 찍었는데 잔액 부족 떠서 뒤 사람들 시선이 등에 꽂혔다.", "잔액 부족"),
            ("엘리베이터 닫힘 버튼 누르려는 순간 누가 뛰어오는 소리 들리면 늘 고민돼.", "열림"),
            ("횡단보도 신호가 너무 짧아서 뛰어가는데 중간에서 빨간불로 바뀌면 억울함.", "횡단보도"),
            ("버스에서 내릴 때 사람이 너무 많아서 '내릴게요'를 세 번 외쳤다.", "내릴게요"),
            ("우산 들고 나온 날은 비가 안 오고 안 들고 나온 날은 꼭 쏟아지는 법칙 뭐냐.", "우산"),
            ("이어폰 배터리 없어서 출근길에 강제로 세상 소리 다 들으니까 정신이 피곤해.", "이어폰"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "어느 지역 기준",
            "거대 개미집",
            "어린 시절의 나",
            "결혼 로망",
            "나만 그런 느낌",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-foundation-admin-home-health-social-transport-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_foundation_laundry_home_digital_office_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("흰 셔츠 입은 날에만 꼭 김치 국물이나 커피가 튀어서 하루 종일 신경 쓰여.", "흰 셔츠"),
            ("세탁기 돌리고 나면 양말 한 짝이 꼭 사라져서 세탁기 안에 블랙홀이 있는 것 같아.", "양말"),
            ("검은 옷 입고 나왔는데 먼지랑 보풀이 너무 붙어서 돌돌이 없으면 외출 불가야.", "돌돌이"),
            ("다림질하다가 셔츠 한쪽에 반짝이는 자국 생겨서 망한 것 같아.", "다림질"),
            ("빨래 널었는데 날이 습해서 하루 지나도 축축한 냄새가 나는 것 같아.", "빨래"),
            ("운동화 빨려고 마음먹었는데 말리는 시간 생각하니까 바로 포기하게 돼.", "운동화"),
            ("옷장에 방향제 넣어놨는데 향이 너무 세서 옷에서 화장실 냄새 나는 느낌이야.", "방향제"),
            ("비 맞은 신발을 대충 말렸더니 냄새가 올라와서 현관이 공격당한 기분이야.", "신발"),
            ("가방 지퍼가 자꾸 천에 씹혀서 급할 때마다 손에 땀 나.", "지퍼"),
            ("우산 접어놨다가 다시 펴보니 이상한 쉰내가 나서 당황했어.", "우산"),
            ("공기청정기 필터 교체 알림이 계속 뜨는데 필터값 보고 모른 척하고 있어.", "공기청정기"),
            ("가습기 물통 씻어야 하는데 구조가 복잡해서 매번 대충 헹구고 끝내게 돼.", "가습기"),
            ("제습기 물통 비우는 걸 깜빡해서 삐삐 울릴 때마다 괜히 혼나는 기분이야.", "제습기"),
            ("쓰레기봉투 묶다가 안에 국물이 새서 손에 묻으면 진짜 현타 와.", "쓰레기봉투"),
            ("분리수거 날 놓쳐서 베란다에 박스가 산처럼 쌓여가고 있어.", "분리수거"),
            ("집에 들어오자마자 이상한 냄새 나는데 출처를 못 찾으면 더 불안해.", "냄새"),
            ("싱크대 거름망 비우는 건 왜 이렇게 마음의 준비가 필요한 집안일일까.", "거름망"),
            ("화장실 청소하려고 세제 뿌려놓고 까먹어서 냄새만 진해졌어.", "화장실 청소"),
            ("방바닥에 머리카락이 왜 닦아도 닦아도 계속 나오는지 모르겠어.", "머리카락"),
            ("로봇청소기 돌렸는데 전선에 걸려서 거실 한가운데서 구조 요청 중이야.", "로봇청소기"),
            ("모바일 결제하려는데 얼굴 인식이 마스크 때문에 계속 실패해서 줄 서 있는 사람들 눈치 보여.", "얼굴 인식"),
            ("QR코드 찍으려는데 카메라 초점이 안 잡혀서 계산대 앞에서 혼자 허둥댔어.", "QR코드"),
            ("배달 주소를 예전 집으로 해놓고 주문해서 취소 버튼 찾느라 식은땀 났어.", "배달 주소"),
            ("택시 앱에서 목적지 잘못 찍은 걸 출발하고 나서 알아서 기사님한테 말하기 민망했어.", "택시 앱"),
            ("폰 저장공간 부족 알림이 떠서 사진 지우려는데 뭐 하나도 못 지우겠어.", "저장공간"),
            ("클라우드 백업이 꽉 찼다는데 뭘 지워야 할지 몰라서 알림만 계속 닫는 중.", "클라우드"),
            ("자동완성이 이상한 단어로 바꿔서 카톡 보내고 나서야 발견했어.", "자동완성"),
            ("온라인 주문 결제는 됐는데 주문 내역이 안 떠서 돈만 사라진 느낌이야.", "주문 내역"),
            ("예약 문자 보내놓고 날짜를 잘못 골라서 완전 엉뚱한 시간에 발송됐어.", "예약 문자"),
            ("알림이 너무 많이 와서 정작 중요한 알림은 못 보고 지나쳤어.", "알림"),
            ("도서관 반납일 깜빡해서 연체료가 붙었는데 금액은 작아도 괜히 속상해.", "연체료"),
            ("친구한테 계좌이체하려다가 받는 사람 이름 다시 확인하느라 세 번 멈칫했어.", "계좌이체"),
            ("현금영수증 번호 입력하려는데 뒤에 줄이 길어서 갑자기 손가락이 꼬였어.", "현금영수증"),
            ("마트 셀프 계산대에서 바코드가 안 찍히면 괜히 내가 기계 못 쓰는 사람 된 것 같아.", "셀프 계산대"),
            ("무인 택배 접수기 앞에서 화면 안내를 읽는데도 뭘 눌러야 할지 모르겠더라.", "택배 접수기"),
            ("사진 인화하려고 키오스크 갔는데 파일 선택 화면에서 너무 오래 고민했어.", "키오스크"),
            ("병원 접수할 때 주민번호 말해야 하는데 주변 사람이 들을까 봐 괜히 목소리 낮아져.", "주민번호"),
            ("예약 시간보다 10분 늦을 것 같아서 전화해야 하는데 그 전화가 왜 이렇게 부담스럽지.", "예약 시간"),
            ("친구 선물 포장 맡겼는데 리본 색 고르는 것부터 너무 결정장애 와.", "리본"),
            ("부모님께 보낼 택배에 메모 한 줄 넣을까 말까 하다가 괜히 마음이 몽글해졌어.", "메모"),
            ("회의실 예약해놨는데 누가 먼저 들어가 있어서 말 걸기 애매했어.", "회의실"),
            ("프린터 토너 부족 경고 뜨는데 일단 한 장만 더 버텨달라고 빌면서 출력했어.", "토너"),
            ("스테이플러 심이 딱 중요한 순간에 떨어져서 사무실 서랍을 뒤지게 돼.", "스테이플러"),
            ("화이트보드 마커가 말라서 회의 중에 글씨가 유령처럼 흐리게 나왔어.", "화이트보드"),
            ("노트북 충전기를 집에 두고 와서 배터리 퍼센트 보면서 하루 종일 긴장했어.", "충전기"),
            ("화상회의 들어갔는데 마이크가 음소거가 아니라는 걸 뒤늦게 알고 등골이 서늘했어.", "마이크"),
            ("공유 문서에 잘못 입력한 내용을 누가 보기 전에 고치려고 손이 빨라졌어.", "공유 문서"),
            ("캘린더 알림을 꺼놨다가 중요한 약속을 까먹을 뻔해서 심장 내려앉았어.", "캘린더"),
            ("책상 위 포스트잇이 너무 많아져서 오히려 뭐가 중요한지 하나도 안 보여.", "포스트잇"),
            ("볼펜이 필요할 땐 꼭 안 보이다가 정리하면 열 개씩 나오는 게 얄미워.", "볼펜"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "어느 지역 기준",
            "어린 시절의 나",
            "결혼 로망",
            "나만 그런 느낌",
            "거대 개미집",
            "철학",
            "목록은 가볍게",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-foundation-laundry-home-digital-office-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_mystery_audio_solofood_secondhand_home_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("오늘 길 걷다가 어떤 사람이랑 마주쳤는데, 10년 전에 꿈에서 본 장면이랑 완벽하게 똑같아서 소름 돋음.", "꿈"),
            ("자려고 누웠는데 귀에서 삐- 소리 나길래 주파수 맞추는 중인가 싶어서 혼자 집중함. 이명인데 괜히 과몰입.", "이명"),
            ("오늘 분명히 내 방 책상 위에 지갑 뒀는데 왜 냉장고 계란 칸에서 나오냐? 집안에 렙틸리언 사는 게 분명함.", "지갑"),
            ("매일 똑같은 시간에 깨는 거 나만 이래? 새벽 4시 44분만 되면 가위눌린 것처럼 눈 번쩍 떠짐.", "새벽 4시 44분"),
            ("스마트폰 알고리즘이 내가 친구랑 '말'로만 한 주제를 바로 피드에 띄워줄 때... 도청당하는 기분임.", "알고리즘"),
            ("지하철 타고 가는데 창밖에 풍경이 묘하게 그래픽 깨진 것처럼 보일 때 있음. 매트릭스 세계관 오류인가.", "매트릭스"),
            ("어릴 때 분명히 있던 과자인데 검색하니까 아예 존재하지 않는 제품이래. 만델라 효과 씨게 왔다.", "만델라"),
            ("길 가다가 고양이랑 눈 마주쳤는데 걔가 사람처럼 한숨 쉬고 가더라. 진짜 사람 들어있는 거 아닐까.", "한숨"),
            ("분명히 방금 이어폰 케이스 닫았는데 주머니에 넣으려고 보니까 열려있음. 시공간 왜곡 일어난 듯.", "이어폰"),
            ("거울 속 내 모습 한참 쳐다보고 있으면 묘하게 내가 낯설어 보일 때 있음. 게슈탈트 붕괴 현상 무서워.", "거울"),
            ("스피커 바꿨더니 보컬 숨소리까지 다 들림. 이 맛에 돈 지랄 하는구나 싶어서 통장 보며 위안 삼는 중.", "스피커"),
            ("방 구석에 부밍(저음 웅웅거림) 심하길래 다이소 가서 압축 스펀지 사다가 야매로 코너 트랩 만듦.", "부밍"),
            ("음원 사이트 스트리밍 음질 설정 '무손실 High-Res'로 올렸는데 막상 막귀라 구분 못 하겠음. 기분 탓이 90%.", "무손실"),
            ("이어폰 케이블 커스텀 동선으로 바꿨더니 중저음이 단단해진 느낌. 단톡방에 후기 올렸다가 장비병 환자 취급당함.", "케이블"),
            ("새벽에 불 다 끄고 헤드폰 쓰고 재즈 들으면서 멍 때릴 때가 하루 중 유일하게 영혼 정화되는 시간임.", "재즈"),
            ("스피커 스탠드 밑에 대리석 받침대 깔았더니 소리가 훨씬 깔끔해짐. 돌덩이 하나에 이 난리를 피우다니.", "대리석"),
            ("이어폰 폼팁 사이즈 안 맞아서 귀 통증 오길래 실리콘 팁으로 바꿈. 차음성은 떨어지는데 편하긴 하네.", "폼팁"),
            ("턴테이블 입문할까 하고 LP 판 가격 알아봤는데 장난 없음. 아날로그 감성 챙기려다 파산각.", "LP"),
            ("유튜브에서 '3D 입체 음향' 영상 이어폰 끼고 듣는데 뒤에서 가위질하는 소리 나서 진짜 고개 돌림.", "입체 음향"),
            ("오디오 전용 멀티탭은 뭐가 다른가 싶어서 가격 봤다가 조용히 뒤로 가기 누름. 그들만의 리그임.", "멀티탭"),
            ("동네 골목 구석에 아저씨들만 가는 실내포차 갔는데 짜글이 찌개 간이 예술임. 소주가 그냥 녹아내림.", "실내포차"),
            ("오늘 저녁은 혼술 각이다. 편의점 들러서 수입 맥주 4캔에 닭강정 사 와서 세팅 완료.", "혼술"),
            ("단골 이자카야 사장님이 '오늘 좋은 고기 들어왔다'면서 서비스 슬쩍 얹어줄 때의 그 내적 단골 부심.", "단골"),
            ("혼자 밥 먹을 때 유독 맛에 집중하게 됨. 백종원 빙의해서 '흠, 이 집은 마늘을 많이 썼군' 분석함.", "혼자 밥"),
            ("뜨끈한 하이볼 한 잔에 바삭한 가라아게 한 입... 오늘 하루 받았던 스트레스가 싹 휘발되는 느낌.", "하이볼"),
            ("집에서 혼자 삼겹살 구워 먹으려다가 기름 튀고 냄새 빼기 귀찮아서 결국 동네 국밥집으로 선회함.", "삼겹살"),
            ("단골 바(Bar)에 혼자 앉아서 위스키 한 잔 시켜놓고 바텐더랑 적당한 거리감으로 스몰토크 하는 분위기 좋음.", "위스키"),
            ("비 오는 날 밤에 포장마차 우동 국물 마시면서 소주 한 잔... 이게 진정한 어른의 주말 힐링이지.", "포장마차"),
            ("배달 음식 시키면 양 너무 많아서 항상 남김. 1인 가구용 반값 반 인분 메뉴 좀 활성화해라.", "1인 가구"),
            ("오늘 안주는 명란구이에 오이 슬라이스 조합. 짭조름하고 아삭한 게 배도 안 부르고 최고의 혼술 안주임.", "명란구이"),
            ("중고 거래 하러 약속 장소 나갔는데 멀리서 쭈구려 앉아 폰 보고 있는 사람 보자마자 '아, 저 사람이다' 직감함.", "중고 거래"),
            ("제품 설명에 '네고 사절' 대문짝만하게 써놨는데 물건 만나자마자 '만 원만 깎아주심 안 돼요?' 시전 당함. 뇌 정지.", "네고"),
            ("무료 나눔 글 올렸더니 집 앞까지 배달해 달라는 빌런 만남. 세상엔 참 다양한 사람이 많다.", "무료 나눔"),
            ("쿨 거래 하시는 분 만나서 물건 건네주고 3초 만에 송금 완료됨. 이런 거래만 있으면 스트레스 없을 듯.", "쿨 거래"),
            ("구매자가 물건 상태 꼼꼼히 본다고 돋보기 들고 나올 기세라 옆에서 지켜보는데 은근 눈치 보임.", "돋보기"),
            ("당근마켓 매너온도 99도인 사람 처음 봄. 거래할 때 거의 빛이 나더라. 매너가 사람을 만든다.", "매너온도"),
            ("새 상품 급이라고 해서 직거래 나갔는데 먼지 꼬질꼬질 묻어있음. 그냥 안 산다고 하고 돌아오는데 기 빨림.", "직거래"),
            ("단톡방에 중고 거래 빌런 카톡 캡처해서 올렸더니 다들 역대급이라고 박수 침. 박물관 보내야 됨.", "중고 거래"),
            ("안 쓰는 물건 다 팔아서 치킨 값 벌었다. 방도 넓어지고 배도 부르고 이것이 창조경제.", "치킨 값"),
            ("택배 거래 하자면서 안전결제 거부하고 직입금 유도하길래 사기꾼 삘 와서 조용히 차단 박음.", "안전결제"),
            ("싱크대 물 한 방울씩 똑... 똑... 떨어지는 소리 유난히 거슬려서 새벽에 일어나서 밸런스 조절하고 옴.", "싱크대"),
            ("에어컨 예약 꺼짐 해놨는데 왜 아침까지 쌩쌩 돌아가고 있냐. 내가 꿈결에 리모컨을 눌렀나.", "에어컨"),
            ("스마트폰 충전 단자에 먼지 낀 거 핀셋으로 빼냈는데 먼지가 덩어리로 나옴. 묘한 카타르시스.", "충전 단자"),
            ("새벽에 냉장고 돌아가는 기계음 유난히 크게 들릴 때, 지구 멸망 직전의 우주선에 혼자 남은 기분 듦.", "냉장고"),
            ("내 방 벽지 무늬 가만히 보고 있으면 사람 얼굴 모양처럼 보여서 유심히 각도 돌려가며 분석함.", "벽지"),
            ("물건 쓰던 자리에 1cm만 틀어져 있어도 귀신같이 알아채고 제자리로 돌려놓는 피곤한 성격.", "1cm"),
            ("분명히 아까 카톡 보냈다고 생각했는데 임시 저장 창에 그대로 남아있을 때... 내 기억력 어디 감.", "임시 저장"),
            ("어제 산 물건 영수증 지갑에 고이 접어놓음. 일주일 뒤에 정산할 때 보물찾기 하듯 꺼내기.", "영수증"),
            ("샤워하다가 문득 '내가 머리를 감았나? 린스를 칠했나?' 기억 안 나서 머리카락 만져보며 추리함.", "샤워"),
            ("누가 내 방 물건 건드리는 거 싫어서 문고리에 머리카락 하나 붙여놓는 상상함. 탐정 놀이 마스터.", "문고리"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그냥 만 원만",
            "잘 자",
            "어느 지역 기준",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-mystery-audio-solofood-secondhand-home-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_plants_kitchen_admin_body_local_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("몬스테라 새 잎이 돌돌 말려 올라오는 거 보니까 괜히 내가 키운 자식 같아서 뿌듯함.", "몬스테라"),
            ("다육이 물 조금만 줬는데도 축 처져서 식물 키우기 재능이 없는 건가 현타 옴.", "다육이"),
            ("화분 분갈이하다가 흙을 거실 바닥에 다 쏟아서 식물보다 내가 먼저 말라죽을 뻔함.", "분갈이"),
            ("바질 키워서 파스타 해먹겠다고 했는데 벌레가 먼저 파티 열고 있더라.", "바질"),
            ("식물등 켜놓으니까 방 분위기는 좋은데 전기요금 생각하면 약간 죄책감 듦.", "식물등"),
            ("잎 끝이 갈색으로 타들어가길래 물 부족인지 과습인지 식물 카페 뒤지는 중.", "잎"),
            ("베란다 텃밭에 상추 심었는데 막상 수확하니까 한 쌈도 안 나와서 웃김.", "상추"),
            ("꽃집에서 작은 화분 하나만 사려다가 선인장까지 같이 데려옴. 식집사 입문각.", "화분"),
            ("로즈마리 향 맡으면 갑자기 내가 유럽 주방에 있는 사람처럼 기분 좋아짐.", "로즈마리"),
            ("장마철에 화분 곰팡이 올라와서 흙 갈아엎을지 그냥 모른 척할지 고민 중.", "곰팡이"),
            ("계란말이 하다가 뒤집기 실패해서 그냥 스크램블이라고 우기며 접시에 담음.", "계란말이"),
            ("김치찌개 끓였는데 너무 짜서 물 붓다 보니 김치국이 되어버림.", "김치찌개"),
            ("에어프라이어에 감자 돌렸는데 겉은 타고 속은 생감자라 요리 인생이 흔들림.", "에어프라이어"),
            ("쌀 씻고 밥솥 버튼 안 눌러서 한 시간 뒤에 생쌀 보고 영혼 나감.", "밥솥"),
            ("양파 썰다가 눈물 폭발해서 내가 요리하는 건지 이별하는 건지 모르겠음.", "양파"),
            ("파스타 면 삶을 때 소금 넣는 타이밍 놓쳐서 괜히 셰프 자격 박탈당한 기분.", "파스타"),
            ("냉장고 털이 볶음밥 했는데 남은 반찬 다 넣었더니 정체불명 한식 리조또 됨.", "볶음밥"),
            ("주방 저울 없이 베이킹했다가 쿠키가 벽돌처럼 나와서 창문 깨도 될 수준임.", "쿠키"),
            ("칼질하다가 손톱 살짝 베여서 요리 의욕이 바로 퇴근함.", "칼질"),
            ("설거지 미루다 싱크대가 작은 산처럼 쌓여서 눈 마주치기 싫어짐.", "설거지"),
            ("은행 앱 인증서 갱신하라는데 비밀번호가 기억 안 나서 내 돈인데 남의 돈 같음.", "인증서"),
            ("주민센터 가서 서류 하나 떼는데 번호표 기다리다 반나절 순삭됨.", "주민센터"),
            ("택배 반품하려고 편의점 갔는데 QR코드 안 떠서 뒤에 사람 눈치 보느라 땀남.", "QR코드"),
            ("공과금 자동이체 해놨다고 믿었는데 미납 문자 와서 심장이 철렁함.", "공과금"),
            ("병원 예약 시간 맞춰 갔는데 대기실에서 한 시간 기다리는 건 왜 기본값임.", "병원"),
            ("휴대폰 요금제 바꾸려다 약정이랑 위약금 설명 듣고 조용히 포기함.", "요금제"),
            ("주차 정산기 앞에서 카드가 안 읽혀서 뒤차 줄 서는 순간 멘탈 흔들림.", "주차"),
            ("도서관 책 반납일 하루 지나서 연체 알림 오면 괜히 범죄자 된 기분임.", "도서관"),
            ("증명사진 찍었는데 얼굴이 너무 낯설어서 이걸 신분증에 써도 되나 고민됨.", "증명사진"),
            ("무인 민원 발급기 앞에서 지문 인식 계속 실패하니까 기계한테 거절당한 느낌.", "지문"),
            ("요즘 목이 뻐근해서 고개 돌릴 때마다 로봇처럼 삐걱거림.", "목"),
            ("눈이 너무 건조해서 인공눈물 넣었는데 넣자마자 더 졸려짐.", "인공눈물"),
            ("운동한다고 스쿼트 몇 개 했을 뿐인데 다음 날 계단 내려갈 때 다리가 배신함.", "스쿼트"),
            ("입술이 터서 립밤 바르는데도 계속 갈라져서 겨울이 내 얼굴을 공격하는 중.", "립밤"),
            ("마스크 오래 썼더니 귀 뒤가 아파서 귀가 먼저 퇴근하고 싶어 함.", "마스크"),
            ("갑자기 눈 밑이 파르르 떨려서 마그네슘 검색하고 있음. 몸이 알림 보내는 듯.", "마그네슘"),
            ("밤에 다리에 쥐 나서 혼자 침대 위에서 무음 비명 지름.", "쥐"),
            ("어깨가 너무 뭉쳐서 마사지볼 굴렸더니 아프면서도 시원해서 중독될 것 같음.", "마사지볼"),
            ("치과 스케일링 예약해놓고 전날부터 괜히 양치 더 열심히 하는 사람 됨.", "치과"),
            ("손목이 시큰해서 마우스 잡을 때마다 내 몸이 퇴사 요구하는 것 같음.", "손목"),
            ("윗집에서 밤마다 의자 끄는 소리 나서 천장만 노려보는 시간이 늘어남.", "윗집"),
            ("아파트 분리수거장 갔다가 스티로폼 어디 버리는지 몰라서 안내문 정독함.", "분리수거"),
            ("엘리베이터에서 이웃이랑 단둘이 타면 인사 타이밍 놓쳐서 공기가 굳음.", "엘리베이터"),
            ("택배함 비밀번호 문자 지워버려서 택배를 눈앞에 두고 못 꺼내는 상황 됨.", "택배함"),
            ("관리사무소 방송 소리가 너무 커서 집 안에서 갑자기 재난 영화 시작된 줄 앎.", "관리사무소"),
            ("동네 편의점 알바가 내 단골 메뉴를 외워서 고맙지만 살짝 민망함.", "편의점"),
            ("새벽에 오토바이 배달 소리 너무 커서 잠결에 배달 앱 삭제할 뻔함.", "오토바이"),
            ("경비실에 맡겨진 택배 찾으러 갈 때 괜히 빈손으로 가기 민망해서 인사 멘트 고민함.", "경비실"),
            ("복도 센서등이 나 지나갈 때만 안 켜져서 괜히 건물한테 무시당한 느낌.", "센서등"),
            ("동네 새로 생긴 무인 아이스크림 가게 갔다가 계산대 앞에서 바코드 못 찾아 허둥댐.", "바코드"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그냥 만 원만",
            "잘 자",
            "어느 지역 기준",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-plants-kitchen-admin-body-local-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_beauty_laundry_study_family_home_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("아침에 선크림 바르고 나왔는데 목 뒤는 까먹어서 거기만 빨갛게 익어버림.", "선크림"),
            ("중요한 날마다 턱에 뾰루지 올라오는 거 진짜 피부가 내 일정표 훔쳐보는 듯.", "뾰루지"),
            ("마스크팩 붙이고 누웠다가 잠들어서 얼굴이 오히려 사막처럼 말라버림.", "마스크팩"),
            ("쿠션 퍼프로 대충 두드렸는데 거울 보니까 턱선에 경계선 생겨서 가면 쓴 사람 됨.", "쿠션"),
            ("머리 왁스 조금만 바른다는 게 떡진 볶음면처럼 돼서 하루 종일 모자 쓰고 싶었음.", "왁스"),
            ("향수 한 번만 뿌리려다 세 번 눌러서 엘리베이터에서 나 혼자 꽃집 됨.", "향수"),
            ("손톱 매니큐어 바르고 말리기 전에 이불 만져서 자국 다 찍힘. 인내심 테스트임.", "매니큐어"),
            ("드라이샴푸 뿌렸는데 흰 가루가 남아서 정수리에 밀가루 뿌린 사람 됨.", "드라이샴푸"),
            ("면도하다가 턱 밑 살짝 베여서 작은 상처 하나가 하루 종일 신경 쓰임.", "면도"),
            ("렌즈 끼고 나왔는데 한쪽만 건조해서 세상이 미세하게 비대칭으로 보임.", "렌즈"),
            ("흰 티 빨래에 검은 양말 하나 섞였더니 전체가 애매한 회색으로 물들었음.", "흰 티"),
            ("니트 세탁기 돌렸더니 아동복 사이즈로 줄어서 인형한테 입혀야 할 판임.", "니트"),
            ("빨래할 때 휴지 같이 넣어서 검은 바지에 하얀 조각들이 눈처럼 붙어 있음.", "휴지"),
            ("건조기 먼지 필터 열었더니 솜뭉치가 한 주먹 나와서 옷들이 희생된 느낌.", "건조기"),
            ("검은 코트 입고 나왔는데 먼지랑 털이 너무 붙어서 돌돌이 없으면 사회생활 불가임.", "돌돌이"),
            ("다림질하려고 했는데 셔츠 주름이 내 의지보다 강해서 결국 가디건으로 덮음.", "다림질"),
            ("운동화 빨았는데 하루 지나도 안 말라서 현관에 축축한 죄책감처럼 놓여 있음.", "운동화"),
            ("옷장 제습제 물 가득 찬 거 보고 내 방 습기가 이렇게 많았나 살짝 충격받음.", "제습제"),
            ("세탁기에 양말 한 짝만 계속 사라지는 거 진짜 이세계 포털 있는 것 같음.", "양말"),
            ("옷걸이에 걸어둔 니트 어깨가 뿔처럼 튀어나와서 입으면 갑자기 갑옷 실루엣 됨.", "옷걸이"),
            ("스터디카페에서 키보드 소리 크게 날까 봐 타자 하나 칠 때마다 손가락에 브레이크 검.", "스터디카페"),
            ("형광펜으로 줄 긋다가 종이가 번져서 중요한 문장이 아니라 형광 늪이 됨.", "형광펜"),
            ("노트북 충전기 두고 와서 배터리 퍼센트 보며 공부보다 생명 연장에 집중함.", "충전기"),
            ("독서실 자리 예약했는데 막상 가보니 콘센트 없는 자리라 멘탈이 살짝 꺼짐.", "콘센트"),
            ("포모도로 타이머 켜놓고 25분 집중하려 했는데 타이머 설정만 만지다 25분 지나감.", "포모도로"),
            ("필기하려고 볼펜 꺼냈는데 잉크 끊겨서 글씨가 모스부호처럼 찍힘.", "볼펜"),
            ("포스트잇 붙여놨는데 접착력 약해서 책상 밑으로 다 떨어져 계획도 같이 떨어짐.", "포스트잇"),
            ("시험 요약노트 예쁘게 만들다가 정작 외우는 건 하나도 못 하고 꾸미기만 함.", "요약노트"),
            ("공부하려고 이어플러그 꼈는데 내 심장소리랑 숨소리만 너무 크게 들려서 더 신경 쓰임.", "이어플러그"),
            ("도서관에서 배에서 꼬르륵 소리 나서 책장 넘기는 척으로 소리 덮으려 함.", "도서관"),
            ("엄마가 카톡 글씨 크기를 최대로 키워놔서 화면에 문장 두 줄밖에 안 보임.", "글씨 크기"),
            ("아빠가 휴대폰 화면을 캡처하는 대신 다른 폰으로 화면 사진을 찍어서 보내줌.", "화면 사진"),
            ("부모님 폰에 앱 업데이트 알림 27개 쌓여 있어서 내가 대신 정리하다가 지침.", "앱 업데이트"),
            ("가족 단톡방에 엄마가 꽃 사진 20장을 연속으로 보내서 갤러리가 봄이 됨.", "가족 단톡방"),
            ("아빠가 유튜브 광고를 진짜 영상인 줄 알고 끝까지 보고 있어서 스킵 버튼 알려드림.", "스킵 버튼"),
            ("부모님이 와이파이 비밀번호 물어볼 때마다 공유기 뒤집어서 다시 읽어주는 담당 됨.", "와이파이"),
            ("엄마가 보이스피싱 문자 받은 것 같다고 보여주는데 내가 더 놀라서 바로 차단함.", "보이스피싱"),
            ("아빠 폰 저장공간 부족하다고 해서 봤더니 같은 산 사진이 800장 들어있음.", "저장공간"),
            ("부모님이 영상통화 켜놓고 천장만 보여주면서 내 목소리 들리냐고 물어보심.", "영상통화"),
            ("가족 단톡방에서 이모티콘 하나 잘못 보냈다가 친척들이 다 물음표 보내서 수습함.", "이모티콘"),
            ("공기청정기 필터 교체 알림이 계속 떠서 집이 나한테 잔소리하는 것 같음.", "공기청정기"),
            ("가습기 물통 씻기 귀찮아서 미루다 보니 물때 보고 조용히 반성함.", "가습기"),
            ("전기장판 온도 1단계만 올렸는데 새벽에 땀나서 사막에서 자는 줄 알았음.", "전기장판"),
            ("로봇청소기가 현관 턱에 걸려서 혼자 삐삐 울고 있는데 묘하게 구조 요청 같음.", "로봇청소기"),
            ("정수기 필터 교체 날짜 지나서 물 마실 때마다 괜히 찝찝한 상상함.", "정수기"),
            ("냉장고 문 살짝 덜 닫혀서 삐삐 울릴 때마다 집한테 혼나는 기분임.", "냉장고"),
            ("전자레인지에 국 데우다가 뚜껑 안 덮어서 안쪽이 국물 폭발 현장 됨.", "전자레인지"),
            ("리모컨이 소파 틈으로 사라져서 손 넣었다가 과자 부스러기만 잔뜩 건짐.", "리모컨"),
            ("인터넷 공유기 불이 깜빡일 때마다 집 전체가 숨을 멈춘 것처럼 조용해짐.", "공유기"),
            ("현관 도어락 배터리 부족음 들리는데도 미루다가 어느 날 밖에서 못 들어갈까 봐 불안함.", "도어락"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-beauty-laundry-study-family-home-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_beauty_laundry_study_family_home_variant_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("목덜미에 선크림 안 바르고 나갔다가 거기만 익어서 따가움.", "선크림"),
            ("약속 있는 날마다 턱에 트러블 올라오는 거 피부가 내 스케줄 방해하는 듯.", "트러블"),
            ("팩 붙인 채로 잠깐 눈 붙였는데 떼고 보니 얼굴이 오히려 바싹 말랐어.", "팩"),
            ("베이스 대충 바르고 나갔더니 턱 아래쪽만 색이 달라서 민망했음.", "턱"),
            ("헤어 제품 손에 덜어 바르다 양 조절 실패해서 머리가 기름진 떡처럼 됨.", "헤어"),
            ("향수 욕심냈다가 버스 안에서 나만 향기 폭탄 들고 탄 사람 됐어.", "향수"),
            ("네일 바르고 덜 마른 상태로 담요 만졌더니 표면에 줄무늬 다 찍힘.", "네일"),
            ("머리 기름 잡으려고 드라이샴푸 썼는데 정수리에 하얗게 티 나서 더 망함.", "드라이샴푸"),
            ("면도날 살짝 삐끗해서 턱에 작은 빨간 점 생겼는데 계속 거슬려.", "면도"),
            ("콘택트렌즈 한쪽이 말라붙어서 하루 종일 눈 한쪽만 예민했어.", "렌즈"),
            ("흰 빨래에 어두운 양말 하나 같이 넣었더니 티셔츠 색이 칙칙해졌어.", "흰 빨래"),
            ("울 니트 그냥 돌렸다가 사이즈가 확 줄어서 입을 사람이 없어짐.", "니트"),
            ("세탁 끝났는데 주머니 속 휴지 때문에 바지에 하얀 먼지 파티 열림.", "휴지"),
            ("건조기 필터 비웠더니 먼지가 너무 많아서 옷이 깎여나간 기분이야.", "건조기"),
            ("검정 외투에 먼지 붙은 거 보고 돌돌이 없이는 밖에 못 나가겠더라.", "돌돌이"),
            ("셔츠 펴보려고 다리미 켰는데 주름이 안 져서 그냥 겉옷으로 가림.", "다리미"),
            ("빨아둔 신발이 계속 축축해서 현관에서 냄새날까 봐 신경 쓰임.", "신발"),
            ("옷장 습기 제거제 통에 물이 꽉 차 있어서 내 방 공기가 의심스러워짐.", "습기 제거제"),
            ("빨래만 하면 양말 짝이 하나씩 증발하는데 진짜 어디로 가는 거냐.", "양말"),
            ("니트 걸어놨더니 어깨 부분이 뾰족하게 솟아서 입으면 이상해 보여.", "니트"),
            ("스터디룸에서 노트북 타자 소리 날까 봐 키 하나 누를 때도 눈치 봄.", "스터디룸"),
            ("중요한 부분 표시하려고 형광펜 그었는데 잉크가 퍼져서 더 안 읽힘.", "형광펜"),
            ("충전 어댑터 안 챙겨와서 노트북 배터리 잔량만 계속 쳐다봄.", "충전"),
            ("독서실에서 잡은 자리가 콘센트랑 너무 멀어서 시작부터 의욕 꺼짐.", "콘센트"),
            ("집중하려고 25분 타이머 맞추려다 설정 앱만 뒤지다가 시간 다 감.", "타이머"),
            ("펜이 자꾸 끊겨서 필기가 점선처럼 찍히니까 집중이 깨졌어.", "펜"),
            ("할 일 붙여둔 메모지가 자꾸 떨어져서 계획까지 같이 떨어지는 느낌.", "메모지"),
            ("요약본 예쁘게 정리하느라 정작 내용 암기는 하나도 못 했어.", "요약본"),
            ("귀마개 끼고 공부하려 했는데 내 숨소리만 크게 들려서 더 방해됨.", "귀마개"),
            ("조용한 열람실에서 배 소리가 크게 나서 책 넘기는 소리로 덮으려 했어.", "열람실"),
            ("엄마 폰 글자 크기가 너무 커서 카톡 한 화면에 내용이 거의 안 들어와.", "글자 크기"),
            ("아빠가 스크린샷을 못 찾아서 휴대폰 화면을 다른 휴대폰으로 찍어 보냈어.", "스크린샷"),
            ("부모님 휴대폰에 업데이트 알림이 산처럼 쌓여 있어서 하나씩 눌러드림.", "업데이트"),
            ("가족방에 엄마가 꽃 사진을 계속 올려서 대화창이 꽃밭 됐어.", "가족방"),
            ("아빠가 광고 건너뛰기 버튼을 못 보고 광고를 본편처럼 보고 계셨음.", "광고"),
            ("집 와이파이 비번 물어보실 때마다 공유기 밑면 사진 찍어서 보내드림.", "와이파이"),
            ("엄마가 수상한 문자 보여줘서 보자마자 피싱 같길래 바로 지웠어.", "피싱"),
            ("아빠 휴대폰 용량 없대서 봤더니 등산 사진만 몇백 장이더라.", "용량"),
            ("부모님이 영상 전화 걸었는데 화면은 천장만 잡히고 목소리만 들림.", "영상 전화"),
            ("친척 단톡에 이상한 이모지 잘못 눌러서 다들 무슨 뜻이냐고 물어봄.", "이모지"),
            ("공청기 필터 바꾸라는 알림이 며칠째 떠서 집이 나한테 숙제 주는 느낌.", "공청기"),
            ("가습기 청소 미뤘더니 안쪽 물때 보고 바로 죄책감 들었어.", "가습기"),
            ("전기매트 온도 조금 올렸을 뿐인데 새벽에 땀나서 깼음.", "전기매트"),
            ("로봇청소기가 문턱에 걸려서 혼자 삐삐대길래 구조하러 감.", "로봇청소기"),
            ("정수기 필터 교체하라는 날짜 넘기니까 물 마실 때 괜히 찝찝해.", "정수기"),
            ("냉장고가 문 덜 닫혔다고 계속 삐삐대서 내가 혼나는 줄 알았어.", "냉장고"),
            ("전자렌지에 찌개 돌렸는데 뚜껑 안 덮어서 안쪽이 난장판 됨.", "전자렌지"),
            ("소파 사이에 리모컨 빠져서 찾다가 먼지랑 과자 가루만 건졌어.", "리모컨"),
            ("와이파이 공유기 램프가 깜빡이면 온 집안 인터넷 운명이 흔들리는 느낌.", "공유기"),
            ("현관 비밀번호 기계 배터리 없다는 소리 계속 나는데 미루다가 불안해짐.", "배터리"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-beauty-laundry-study-family-home-variant-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_payment_transport_cleaning_weather_errand_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("편의점에서 카드 찍었는데 결제 거절 떠서 뒤에 줄 선 사람들 눈치 엄청 봄.", "카드"),
            ("구독 해지하려고 들어갔는데 버튼이 어디 숨어있는지 못 찾아서 10분 헤맴.", "구독"),
            ("자동결제 문자 보고 깜짝 놀람. 내가 아직도 이 서비스를 돈 내고 쓰고 있었다니.", "자동결제"),
            ("계좌이체 하려는데 받는 사람 이름 한 글자 틀릴까 봐 세 번 확인함.", "계좌이체"),
            ("ATM에서 내 돈 찾는데 수수료 1300원 뜨니까 괜히 손해 보는 기분임.", "수수료"),
            ("모바일뱅킹 점검 시간 걸려서 돈 보내야 하는데 아무것도 못 하고 멍때림.", "모바일뱅킹"),
            ("간편결제 비밀번호 갑자기 기억 안 나서 계산대 앞에서 뇌 정지 옴.", "간편결제"),
            ("영수증 안 받는다고 했는데 집 와서 환불할 일 생기니까 바로 후회됨.", "영수증"),
            ("카드값 예정 금액 확인했다가 조용히 앱 닫고 하늘 한번 봄.", "카드값"),
            ("교통카드 잔액 부족할까 봐 개찰구 앞에서 괜히 긴장했어.", "교통카드"),
            ("버스 정류장 도착하자마자 내가 탈 버스가 눈앞에서 떠나버림.", "버스"),
            ("지하철 환승 통로가 너무 길어서 거의 운동하러 온 사람 됐어.", "환승"),
            ("내릴 역 지나쳐서 반대 방향 지하철 타러 가는 순간 현타 제대로 옴.", "내릴 역"),
            ("버스에서 손잡이 못 잡고 급정거 맞아서 혼자 춤추는 사람 됨.", "버스"),
            ("지하철 문 닫히기 직전에 뛰어갔는데 바로 앞에서 닫혀서 민망했어.", "지하철"),
            ("퇴근길 버스가 만원이라 한 발로만 균형 잡고 버티는 중.", "퇴근길"),
            ("비 오는 날 버스 안에 젖은 우산 냄새랑 사람 냄새 섞이면 진짜 힘들어.", "우산"),
            ("택시 잡으려는데 앱에 계속 주변 차량 없음 떠서 집에 못 가는 줄.", "택시"),
            ("지하철에서 자리 났는데 어르신이랑 눈 마주쳐서 자동으로 양보 모드 됨.", "자리"),
            ("버스 기사님이 급출발해서 마시던 커피가 손에서 출렁거렸어.", "커피"),
            ("음식물 쓰레기 버리러 나가는 게 왜 이렇게 인생의 큰 퀘스트 같지.", "음식물 쓰레기"),
            ("분리수거 하려고 라벨 떼는데 접착제가 너무 끈질겨서 손톱 나갈 뻔.", "라벨"),
            ("청소기 돌렸는데 머리카락이 계속 나와서 우리 집에 털 공장 있는 줄.", "청소기"),
            ("화장실 청소는 시작하기 전까지가 제일 싫고 막상 하면 또 뿌듯함.", "화장실"),
            ("쓰레기봉투 꽉 찼는데 한 번 더 눌러 담을 수 있을 것 같아서 미루는 중.", "쓰레기봉투"),
            ("냉장고 정리하다가 유통기한 지난 소스 발견하고 조용히 뚜껑 닫음.", "유통기한"),
            ("싱크대 배수구 청소할 때마다 인간으로서 뭔가를 내려놓는 기분이야.", "배수구"),
            ("바닥 물걸레질 했는데 햇빛에 보니까 자국이 더 선명하게 남아있음.", "물걸레"),
            ("창문 닦으려고 했는데 닦을수록 얼룩이 퍼져서 포기하고 커튼 침.", "창문"),
            ("빨래 개는 건 괜찮은데 옷장에 넣는 마지막 단계가 너무 귀찮아.", "빨래"),
            ("아침엔 추워서 겉옷 챙겼는데 낮엔 더워서 하루 종일 짐 됐어.", "겉옷"),
            ("비 온다더니 안 와서 장우산 들고 다니는 사람만 됨.", "장우산"),
            ("갑자기 소나기 와서 편의점 우산 샀는데 5분 뒤에 그침. 억울해.", "소나기"),
            ("날씨 앱은 맑음이라더니 밖에 나가자마자 바람이 너무 차가워서 배신감 듦.", "날씨 앱"),
            ("습도가 높으니까 머리카락이 말 안 듣고 부풀어서 하루 종일 산발임.", "습도"),
            ("미세먼지 나쁨이라 창문 열고 싶어도 못 여니까 방 공기가 답답해.", "미세먼지"),
            ("겨울에 목도리 안 하고 나갔다가 목덜미로 바람 들어와서 바로 후회함.", "목도리"),
            ("여름에 검은 옷 입고 나갔다가 햇빛 흡수해서 인간 프라이팬 됨.", "검은 옷"),
            ("신발 젖을까 봐 웅덩이 피하다가 더 큰 웅덩이 밟았어.", "웅덩이"),
            ("실내는 에어컨 때문에 춥고 밖은 더워서 옷차림을 어떻게 해야 할지 모르겠음.", "에어컨"),
            ("병원 예약 시간보다 일찍 갔는데도 대기표 숫자가 안 줄어서 멍해짐.", "대기표"),
            ("미용실 예약해놓고 시간 착각해서 30분 일찍 도착했어.", "미용실"),
            ("택배 반품 접수해놓고 박스 테이프 없어서 집 안을 뒤지는 중.", "반품"),
            ("동사무소 가려는데 점심시간 걸려서 문 앞에서 그대로 멈춤.", "동사무소"),
            ("약국 가서 증상 설명하려는데 막상 말하려니 어디가 아픈지 정리가 안 됨.", "약국"),
            ("주차 정산하려고 보니까 할인 등록 안 해서 금액 보고 잠깐 얼었어.", "주차"),
            ("엘리베이터 점검 중이라 8층까지 걸어 올라가는데 숨이 차서 현타 옴.", "엘리베이터"),
            ("택배 기사님 전화 못 받아서 배송 보류 문자 오면 괜히 죄송해짐.", "택배"),
            ("무인 키오스크 앞에서 쿠폰 적용 버튼 못 찾아서 뒤 사람 눈치 봄.", "키오스크"),
            ("예약 변경 전화해야 하는데 전화하기 싫어서 앱에서 버튼만 계속 찾는 중.", "예약"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-payment-transport-cleaning-weather-errand-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_body_health_sleep_workout_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("아침부터 목이 칼칼해서 따뜻한 물만 계속 마시는 중이야.", "목"),
            ("코가 막혀서 숨 쉬는 게 답답해. 하루 종일 입으로 숨 쉬는 느낌.", "코"),
            ("눈이 너무 뻑뻑해서 인공눈물 넣었는데도 금방 다시 건조해져.", "눈"),
            ("어제 운동 너무 무리했더니 허벅지가 계단 내려갈 때마다 비명을 질러.", "허벅지"),
            ("목이 뻐근해서 고개 돌릴 때마다 로봇처럼 움직이는 중.", "목"),
            ("요즘 소화가 잘 안 돼서 밥 먹고 나면 배가 묵직해.", "소화"),
            ("아침에 일어났는데 얼굴이 너무 부어서 거울 보고 잠깐 멈췄어.", "얼굴"),
            ("커피를 마셨는데도 졸려. 카페인이 나를 배신한 것 같아.", "커피"),
            ("입술이 다 터서 웃을 때마다 따가워. 립밤 없으면 못 살겠어.", "입술"),
            ("손이 건조해서 핸드크림 발라도 금방 다시 까칠해져.", "손"),
            ("약 먹어야 하는데 빈속에 먹어도 되나 싶어서 괜히 망설이는 중.", "약"),
            ("병원 예약은 했는데 대기 시간 길까 봐 벌써 지친다.", "병원"),
            ("약국에서 증상 설명하려고 하면 갑자기 어디가 어떻게 아픈지 말이 꼬여.", "약국"),
            ("감기약 먹으면 졸린데 안 먹으면 코가 막혀서 집중이 안 돼.", "감기약"),
            ("영양제 사놓고 맨날 까먹어서 책상 위 장식품 됐어.", "영양제"),
            ("비타민 먹으려고 했는데 물 뜨러 가기 귀찮아서 또 미뤘어.", "비타민"),
            ("체온계로 재면 정상인데 몸은 분명히 으슬으슬한 느낌이야.", "체온계"),
            ("병원 접수할 때 무슨 과로 가야 할지 몰라서 검색만 계속 했어.", "접수"),
            ("건강검진 결과표 받았는데 수치가 많아서 어디부터 봐야 할지 모르겠어.", "건강검진"),
            ("치과 예약일 다가오니까 벌써부터 드릴 소리 상상돼서 긴장돼.", "치과"),
            ("어제 늦게 잤더니 하루 종일 머리가 솜으로 찬 것 같아.", "머리"),
            ("자려고 누웠는데 이상하게 정신이 또렷해져서 잠이 안 와.", "잠"),
            ("알람을 세 개나 맞췄는데 다 끄고 다시 자버렸어.", "알람"),
            ("낮잠을 애매하게 잤더니 더 피곤하고 머리만 멍해졌어.", "낮잠"),
            ("새벽에 자꾸 깨서 아침에 일어나도 잔 것 같지가 않아.", "새벽"),
            ("잠들기 전에 폰 좀만 보려다 한 시간이 사라졌어.", "폰"),
            ("베개가 안 맞는지 자고 일어나면 목이 뻐근해.", "베개"),
            ("방이 너무 건조해서 자고 일어나면 목이 사막 같아.", "건조"),
            ("밤에 이불은 더운데 발은 차가워서 잠을 못 자겠어.", "발"),
            ("아침에 일어나자마자 다시 눕고 싶어서 침대랑 협상 중이야.", "침대"),
            ("헬스장 끊어놓고 첫날만 가고 지금은 출석 앱 알림만 받는 중.", "헬스장"),
            ("스쿼트 조금 했다고 엉덩이 근육이 하루 종일 존재감을 과시해.", "스쿼트"),
            ("러닝하려고 나갔는데 5분 뛰고 폐가 항의해서 바로 걸었어.", "러닝"),
            ("스트레칭 해야지 해놓고 매번 누워서 영상만 저장해둬.", "스트레칭"),
            ("운동복까지 갈아입었는데 갑자기 의욕이 사라져서 소파에 앉았어.", "운동복"),
            ("물 많이 마셔야지 해놓고 하루 끝나면 커피만 세 잔 마신 사람 됨.", "물"),
            ("계단 조금 올랐는데 숨이 차서 내 체력이 어디 갔나 싶어.", "계단"),
            ("요가 매트 펴놓고 정작 그 위에서 누워서 폰만 봤어.", "요가 매트"),
            ("폼롤러로 종아리 풀다가 너무 아파서 고문 도구인 줄 알았어.", "폼롤러"),
            ("만보기 만 보 채우려고 집 안에서 의미 없이 왔다 갔다 했어.", "만보기"),
            ("렌즈 오래 꼈더니 눈이 건조해서 세상이 뿌옇게 보여.", "렌즈"),
            ("마스크 오래 쓰니까 귀 뒤가 아파서 하루 종일 신경 쓰였어.", "마스크"),
            ("이어폰 오래 꼈더니 귀가 먹먹해서 잠깐 빼고 멍때리는 중.", "이어폰"),
            ("모니터를 오래 봤더니 어깨가 굳어서 돌덩이 같아.", "어깨"),
            ("손목이 시큰해서 마우스 잡는 자세부터 의심하게 됐어.", "손목"),
            ("허리가 뻐근해서 의자에 앉아 있는 자세가 갑자기 신경 쓰여.", "허리"),
            ("목소리가 잠겨서 말할 때마다 내가 아닌 사람 같아.", "목소리"),
            ("속이 더부룩해서 탄산수 마시면 좀 나을까 고민 중이야.", "더부룩"),
            ("피곤해서 눈 밑 다크서클이 오늘따라 존재감이 너무 강해.", "다크서클"),
            ("하루 종일 멍해서 뇌가 절전 모드 들어간 것 같아.", "뇌"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-body-health-sleep-workout-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_food_service_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("식당에서 주문했는데 내 메뉴만 안 나와서 괜히 직원 눈치 보고 있어.", "메뉴"),
            ("카페에서 아이스 시켰는데 따뜻한 걸로 나와서 말할까 말까 고민 중이야.", "아이스"),
            ("키오스크에서 결제까지 했는데 주문번호를 못 봐서 멍하니 서 있었어.", "주문번호"),
            ("배달 음식 왔는데 젓가락이 안 들어있어서 집안을 뒤지는 중이야.", "젓가락"),
            ("치킨 시켰는데 소스가 누락돼서 맛의 절반을 잃은 기분이야.", "소스"),
            ("떡볶이 맵기 1단계로 시켰는데도 너무 매워서 쿨피스 찾는 중.", "맵기"),
            ("라면집에서 면 익힘 정도 고르라는데 꼬들인지 보통인지 갑자기 고민돼.", "면"),
            ("고깃집에서 불판 갈아달라고 말하기 애매해서 타이밍만 보고 있어.", "불판"),
            ("뷔페 갔는데 내가 좋아하는 음식 코너만 계속 비어 있어서 기다리는 중.", "뷔페"),
            ("카페 자리 잡았는데 테이블이 덜컹거려서 커피가 계속 불안해.", "테이블"),
            ("식당에서 물티슈가 없는데 직원분들이 너무 바빠 보여서 말 걸기 애매해.", "물티슈"),
            ("주문한 음식 사진이랑 실물이 너무 달라서 잠깐 말문이 막혔어.", "사진"),
            ("배달 예상 시간 40분이었는데 90분으로 늘어나서 배고픔이 분노로 바뀌는 중.", "배달"),
            ("포장 주문했는데 집에 와서 보니까 사이드 메뉴가 빠져 있었어.", "사이드"),
            ("식당에서 음식이 너무 짜서 물만 계속 마시고 있어.", "짜"),
            ("카페 신메뉴 도전했는데 내 취향이 아니라서 아아 살 걸 후회 중.", "신메뉴"),
            ("국밥집에서 다대기 넣기 전에 국물 맛을 봤어야 했는데 이미 다 넣어버렸어.", "다대기"),
            ("샐러드 시켰는데 드레싱이 너무 많아서 건강식인지 소스식인지 모르겠어.", "드레싱"),
            ("피자 배달 왔는데 치즈가 한쪽으로 다 쏠려 있어서 마음이 아파.", "피자"),
            ("햄버거 먹으려는데 소스가 뒤로 다 새서 손이 난장판 됐어.", "햄버거"),
            ("카페에서 빨대 안 챙겨줘서 뚜껑 열고 마시다가 얼음이 입술에 붙었어.", "빨대"),
            ("식당 웨이팅 걸어놨는데 앞 팀이 너무 안 빠져서 배고픔이 깊어지고 있어.", "웨이팅"),
            ("음식 나왔는데 내가 주문한 게 맞는지 확신이 없어서 눈치만 보는 중.", "주문"),
            ("직원분이 음식 내려놓고 바로 가셔서 반찬 리필 말할 타이밍을 놓쳤어.", "반찬"),
            ("카페에서 콘센트 있는 자리 찾다가 좋은 자리는 이미 다 뺏겼어.", "콘센트"),
            ("배달 요청사항에 문 앞에 두라고 썼는데 벨 눌러서 깜짝 놀랐어.", "요청사항"),
            ("김밥 포장했는데 집에 와서 보니 단무지가 빠져서 묘하게 허전해.", "단무지"),
            ("탕수육 소스를 따로 달라고 했는데 부어서 와서 속으로 울었어.", "탕수육"),
            ("카페에서 닉네임 이상하게 저장해둔 걸 직원이 크게 불러서 고개 숙였어.", "닉네임"),
            ("음식에서 머리카락 같은 게 보여서 먹을지 말지 멈칫했어.", "머리카락"),
            ("혼밥하러 갔는데 4인석밖에 없어서 괜히 눈치 보였어.", "혼밥"),
            ("계산하려는데 더치페이 금액이 애매하게 나와서 10원 단위까지 계산 중.", "더치페이"),
            ("카페 쿠폰 찍으려고 했는데 이미 결제 끝난 뒤라 아차 싶었어.", "쿠폰"),
            ("음식이 나왔는데 너무 뜨거워서 입천장 까질까 봐 조심조심 먹고 있어.", "입천장"),
            ("냉면 먹는데 겨자를 너무 많이 풀어서 코가 뻥 뚫렸어.", "겨자"),
            ("초밥집에서 와사비 양 조절 실패해서 눈물이 핑 돌았어.", "와사비"),
            ("분식집에서 순대 내장 빼달라고 말하는 걸 깜빡해서 난감해졌어.", "순대"),
            ("배달 온 국물이 봉투 안에서 살짝 새서 현관에서부터 멘붕 왔어.", "국물"),
            ("음료 테이크아웃했는데 뚜껑이 헐거워서 가방에 넣기가 무서워.", "뚜껑"),
            ("식당에서 너무 조용해서 숟가락 떨어뜨린 소리가 홀 전체에 울렸어.", "숟가락"),
            ("카페에서 노트북 하려는데 옆자리 대화 소리가 너무 커서 집중이 깨졌어.", "옆자리"),
            ("술집에서 안주가 너무 늦게 나와서 술만 먼저 줄어드는 중이야.", "안주"),
            ("고깃집에서 고기 굽는 담당이 됐는데 다들 내 손만 쳐다봐서 부담돼.", "고기"),
            ("패스트푸드점에서 감튀 케첩을 안 챙긴 걸 자리 와서 깨달았어.", "케첩"),
            ("아이스크림 사서 나오자마자 녹기 시작해서 손이 끈적해졌어.", "아이스크림"),
            ("카페에서 주문하려고 줄 섰는데 앞사람이 메뉴를 5분째 고민 중이야.", "메뉴"),
            ("식당에서 맵지 않게 해달라고 했는데 내 기준엔 이미 불지옥이야.", "맵지 않게"),
            ("배달 리뷰 이벤트 신청했는데 서비스가 안 와서 작은 행복을 도둑맞은 느낌이야.", "리뷰 이벤트"),
            ("컵라면 물 선을 넘겨서 한강 라면이 됐어. 이미 돌이킬 수 없어.", "컵라면"),
            ("카페에서 주문한 음료가 너무 달아서 세 모금 마시고 물을 찾고 있어.", "음료"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-food-service-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_foodservice_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_retail_store_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("편의점에서 1+1인 줄 알고 샀는데 행사 상품이 아니라서 괜히 억울했어.", "1+1"),
            ("마트 계산대 앞에서 장바구니 봉투 필요하냐고 물어보는데 순간 머리가 하얘졌어.", "봉투"),
            ("셀프 계산대에서 바코드가 계속 안 찍혀서 뒤 사람 눈치 보였어.", "바코드"),
            ("편의점 도시락 데워놓고 전자레인지에 그대로 두고 나올 뻔했어.", "전자레인지"),
            ("마트에서 카트 동전 없어서 입구에서 멍하니 서 있었어.", "카트"),
            ("편의점에서 교통카드 충전하려는데 현금만 된다고 해서 당황했어.", "교통카드"),
            ("마트에서 사고 싶은 과자 찾는데 매대 위치가 바뀌어서 한참 헤맸어.", "매대"),
            ("편의점 알바생이 봉투에 뜨거운 거랑 아이스크림을 같이 넣어서 마음이 불안했어.", "아이스크림"),
            ("셀프 계산하다가 중량 확인 오류 떠서 직원 호출 버튼 누르기 민망했어.", "중량"),
            ("편의점에서 컵라면 물 받으려는데 온수기가 고장이라 멍해졌어.", "온수기"),
            ("마트 시식 코너 지나가는데 한 번 먹고 또 지나가기가 괜히 민망했어.", "시식"),
            ("편의점에서 삼각김밥 뜯다가 김이 다 찢어져서 손에 붙었어.", "삼각김밥"),
            ("마트에서 할인 스티커 붙은 줄 알고 집었는데 자세히 보니 다른 상품이었어.", "할인"),
            ("계산 끝나고 보니까 포인트 적립을 깜빡해서 작게 아까웠어.", "포인트"),
            ("편의점에서 택배 보내려는데 운송장 붙이는 위치가 헷갈렸어.", "운송장"),
            ("마트에서 우유 사러 갔다가 과자만 잔뜩 사고 정작 우유를 까먹었어.", "우유"),
            ("편의점에서 커피 머신 컵을 잘못 골라서 용량이 안 맞았어.", "컵"),
            ("셀프 계산대에서 결제 완료됐는데 영수증이 안 나와서 끝난 건지 헷갈렸어.", "영수증"),
            ("마트에서 직원분한테 물건 위치 물어봤는데 너무 친절하게 직접 데려다줘서 민망했어.", "위치"),
            ("편의점에서 담배 사는 사람 뒤에 줄 섰는데 종류 고르는 데 너무 오래 걸렸어.", "줄"),
            ("마트에서 냉동식품 샀는데 집 가는 길이 길어서 녹을까 봐 계속 신경 쓰였어.", "냉동식품"),
            ("편의점에서 행사 카드 할인인 줄 모르고 그냥 결제해서 손해 본 기분이야.", "카드 할인"),
            ("마트에서 계산하려는데 바코드 없는 상품이라 직원분이 가격 확인하러 가셨어.", "가격"),
            ("편의점에서 컵라면 먹으려는데 젓가락이 다 떨어져 있어서 당황했어.", "젓가락"),
            ("마트에서 카트 바퀴가 한쪽으로만 끌려서 쇼핑 내내 팔이 피곤했어.", "카트"),
            ("편의점에서 얼음컵 사놓고 음료를 안 사서 다시 들어갔어.", "얼음컵"),
            ("마트에서 세일한다고 써 있어서 샀는데 계산대에서는 정상가로 찍혔어.", "정상가"),
            ("편의점에서 급하게 우산 샀는데 나가자마자 비가 그쳤어.", "우산"),
            ("셀프 계산대에서 봉투 바코드도 찍어야 하는 줄 몰라서 다시 결제했어.", "봉투"),
            ("마트에서 시식하고 맛있어서 샀는데 집에서 먹으니 그 맛이 안 났어.", "시식"),
            ("편의점에서 김밥 데워달라고 했는데 너무 뜨거워져서 들고 나오기 힘들었어.", "김밥"),
            ("마트에서 과일 고르는데 뭐가 달고 신선한지 몰라서 계속 만지작거렸어.", "과일"),
            ("편의점에서 택배 찾으려는데 QR코드 화면 밝기가 낮아서 인식이 안 됐어.", "QR코드"),
            ("마트에서 계산 줄 제일 짧은 데 섰는데 앞 사람이 대량 구매라 망했어.", "계산 줄"),
            ("편의점에서 음료 하나 사러 들어갔다가 간식까지 잔뜩 사서 나왔어.", "간식"),
            ("마트에서 냉장식품을 제일 먼저 집어버려서 쇼핑하는 내내 빨리 계산하고 싶었어.", "냉장식품"),
            ("편의점에서 비닐봉투 말고 종이봉투 달라고 했다가 없다고 해서 손에 다 들고 나왔어.", "종이봉투"),
            ("마트에서 물건 들고 계산대 갔는데 회원가라서 앱 가입하라는 말을 들었어.", "회원가"),
            ("편의점에서 도시락 뚜껑 열다가 소스가 옷에 튀었어.", "소스"),
            ("마트에서 필요한 것만 사려고 했는데 카트가 커서 자꾸 더 담게 돼.", "카트"),
            ("편의점에서 아이스크림 계산하고 나오는데 숟가락 안 챙긴 걸 깨달았어.", "숟가락"),
            ("마트에서 계산 후에 상품 하나가 빠진 걸 영수증 보고 알았어.", "영수증"),
            ("편의점에서 새벽에 라면 먹으려는데 테이블 자리가 꽉 차 있었어.", "테이블"),
            ("마트에서 무거운 생수 묶음 샀다가 집까지 들고 오면서 후회했어.", "생수"),
            ("편의점에서 상품권으로 결제하려는데 사용처가 아니라고 해서 당황했어.", "상품권"),
            ("마트에서 직원 호출 벨 눌렀는데 아무도 안 와서 계속 서 있었어.", "직원"),
            ("편의점에서 계산하려는데 앞 사람이 동전으로 천천히 계산해서 시간이 멈춘 줄 알았어.", "동전"),
            ("마트에서 고기 할인팩 샀는데 유통기한이 오늘까지라 오늘 안 먹으면 끝이야.", "유통기한"),
            ("편의점에서 냉장고 문 열었는데 찾던 음료가 딱 하나 남아 있어서 괜히 승리감 들었어.", "음료"),
            ("마트에서 장 본 걸 봉투에 담다가 계란이 맨 밑에 깔린 걸 뒤늦게 알았어.", "계란"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-retail-store-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_retail_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_transit_commute_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("버스 정류장 도착하자마자 내가 타야 할 버스가 눈앞에서 출발했어.", "버스"),
            ("지하철 문 닫히기 직전에 뛰어갔는데 바로 앞에서 닫혀서 허무했어.", "지하철"),
            ("버스에서 내릴 정류장 놓쳐서 한 정거장 더 가버렸어.", "정류장"),
            ("지하철 반대 방향 탄 걸 세 정거장 지나서야 알았어.", "반대 방향"),
            ("택시 잡으려는데 앱에서 계속 주변에 차량이 없다고 떠서 멍해졌어.", "택시"),
            ("버스 카드 찍었는데 잔액 부족입니다 소리 나서 등줄기 식었어.", "잔액"),
            ("출근길 지하철에 사람이 너무 많아서 숨 쉬기도 힘들었어.", "출근길"),
            ("버스 기사님이 급정거해서 손잡이 잡고도 휘청했어.", "급정거"),
            ("지하철에서 앉으려던 자리 앞사람이 가방으로 먼저 맡아버렸어.", "자리"),
            ("버스 배차 간격 20분인데 방금 하나 놓쳐서 세상이 멈춘 기분이야.", "배차"),
            ("택시 탔는데 기사님이 계속 말을 걸어서 자는 척하고 싶었어.", "기사님"),
            ("지하철 환승 통로가 너무 길어서 사실상 걷기 운동이었어.", "환승"),
            ("버스 안에서 하차벨 눌렀는데 기사님이 그냥 지나쳐서 당황했어.", "하차벨"),
            ("지하철에서 이어폰 끼고 있다가 안내방송 못 듣고 내릴 역 지나쳤어.", "안내방송"),
            ("택시비가 생각보다 많이 나와서 미터기만 계속 쳐다봤어.", "미터기"),
            ("버스에서 교통카드를 가방 깊숙이 넣어둬서 내릴 때 허둥댔어.", "교통카드"),
            ("지하철 에스컬레이터 고장이라 계단으로 올라갔더니 다리가 풀렸어.", "에스컬레이터"),
            ("버스 정류장 이름이 비슷해서 엉뚱한 곳에서 내려버렸어.", "정류장"),
            ("택시 기사님이 길을 돌아가는 것 같아서 지도 앱을 계속 확인했어.", "지도"),
            ("지하철에서 내 앞 사람이 내릴 줄 알고 대기했는데 끝까지 안 내렸어.", "내릴 줄"),
            ("버스 창가에 앉았는데 햇빛이 직빵이라 눈을 못 뜨겠어.", "햇빛"),
            ("지하철 손잡이가 너무 멀어서 중심 잡느라 코어 운동했어.", "손잡이"),
            ("택시 호출해놓고 위치를 잘못 찍어서 기사님이 반대편에서 기다리셨어.", "위치"),
            ("버스에서 졸다가 머리가 창문에 쿵쿵 박혀서 민망했어.", "창문"),
            ("지하철 출구 번호 잘못 나와서 목적지랑 반대 방향으로 걸었어.", "출구"),
            ("버스 안에서 누가 너무 큰 소리로 통화해서 이어폰을 뚫고 들어왔어.", "통화"),
            ("택시 탔는데 멀미가 올라와서 창문 열고 싶었어.", "멀미"),
            ("지하철에서 옆 사람이 계속 내 어깨에 기대서 몸이 굳었어.", "어깨"),
            ("버스 도착 예정 3분이 10분째 3분이라 앱을 못 믿겠어.", "3분"),
            ("택시에서 목적지 잘못 말해서 중간에 급하게 정정했어.", "목적지"),
            ("지하철 개찰구에서 카드가 안 찍혀서 뒤 사람 눈치 보였어.", "개찰구"),
            ("버스 맨 뒷자리 앉았더니 방지턱마다 몸이 튀어 올랐어.", "방지턱"),
            ("지하철에서 가방이 문에 낄 뻔해서 심장이 철렁했어.", "가방"),
            ("택시 앱 결제 실패 떠서 기사님 앞에서 카드 다시 꺼냈어.", "결제"),
            ("버스에서 비 오는 날 우산 든 사람들 사이에 껴서 옷이 다 젖었어.", "우산"),
            ("지하철 안이 너무 더운데 에어컨은 약해서 땀이 났어.", "에어컨"),
            ("버스 기다리는데 정류장 전광판이 고장이라 언제 올지 모르겠어.", "전광판"),
            ("택시 타고 가는데 기사님 네비랑 내 지도 앱이 서로 다른 길을 말했어.", "네비"),
            ("지하철에서 급하게 내리려는데 사람들이 먼저 밀고 들어와서 못 내릴 뻔했어.", "못 내릴"),
            ("버스 좌석에 앉았는데 의자가 젖어 있어서 바로 일어났어.", "좌석"),
            ("지하철에서 휴대폰 보다가 환승역을 지나쳐서 멍해졌어.", "환승역"),
            ("택시에서 조용히 가고 싶은데 라디오 소리가 너무 커서 신경 쓰였어.", "라디오"),
            ("버스에서 하차벨 누르려고 했는데 손이 안 닿아서 급하게 일어났어.", "하차벨"),
            ("지하철에서 자리 양보해야 하나 말아야 하나 애매해서 눈치 봤어.", "자리"),
            ("택시 기다리다가 비 맞아서 머리가 다 젖었어.", "비"),
            ("버스에서 기사님이 너무 빨리 출발해서 카드를 찍자마자 휘청했어.", "카드"),
            ("지하철에서 내릴 때 사람 파도에 밀려서 자동으로 밖으로 나왔어.", "사람"),
            ("택시 잡으려고 큰길까지 나갔는데 빈 차가 하나도 안 보여서 허탈했어.", "빈 차"),
            ("버스에서 창문 열려 있어서 바람이 계속 얼굴로 때렸어.", "바람"),
            ("지하철 막차 시간 착각해서 플랫폼에서 뛰다가 진짜 숨 넘어갈 뻔했어.", "막차"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-transit-commute-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_transit_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_home_repair_maintenance_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("보일러가 갑자기 에러 코드 띄우면서 온수가 안 나와서 멘붕이야.", "보일러"),
            ("샤워하려는데 온수가 미지근하게만 나와서 관리사무소에 전화해야 하나 고민 중.", "온수"),
            ("천장에서 물방울이 똑똑 떨어져서 누수인가 싶어 너무 불안해.", "누수"),
            ("화장실 환풍기가 갑자기 덜덜거리는 소리 내서 밤마다 신경 쓰여.", "환풍기"),
            ("싱크대 하수구에서 냄새 올라와서 배수구 클리너를 사야 하나 봐.", "하수구"),
            ("변기 물이 계속 졸졸 내려가서 수도요금 폭탄 맞을까 봐 걱정돼.", "변기"),
            ("세탁기 탈수할 때 집 전체가 흔들릴 정도로 덜컹거려서 무서워.", "세탁기"),
            ("냉장고에서 갑자기 웅웅 소리가 커져서 고장 전조인가 싶어.", "냉장고"),
            ("에어컨에서 물이 뚝뚝 떨어져서 바닥에 수건 깔아놨어.", "에어컨"),
            ("방충망이 찢어져서 벌레 들어올까 봐 창문도 못 열겠어.", "방충망"),
            ("전등이 깜빡깜빡해서 형광등을 갈아야 하나 안정기를 봐야 하나 모르겠어.", "전등"),
            ("현관 도어락 번호가 가끔 안 눌려서 집 못 들어갈까 봐 불안해.", "도어락"),
            ("인터넷 공유기가 계속 끊겨서 기사님 예약 잡아야 할 것 같아.", "공유기"),
            ("와이파이가 방 끝에서는 한 칸만 떠서 공유기를 바꿔야 하나 고민이야.", "와이파이"),
            ("세면대 물이 너무 천천히 내려가서 머리카락이 막힌 것 같아.", "세면대"),
            ("가스레인지 불이 한쪽만 안 붙어서 점화장치 문제인가 봐.", "가스레인지"),
            ("창문 잠금장치가 헐거워져서 밤에 괜히 신경 쓰여.", "창문"),
            ("벽지에 곰팡이 점이 생겨서 제습기를 켜야 하나 걱정돼.", "곰팡이"),
            ("장판이 들떠서 걸을 때마다 발에 걸려 짜증 나.", "장판"),
            ("문고리가 헐거워져서 어느 날 빠질 것 같아.", "문고리"),
            ("관리사무소 방송이 너무 크게 나와서 심장이 덜컥했어.", "관리사무소"),
            ("엘리베이터 점검 중이라 12층까지 계단으로 올라왔더니 다리가 풀렸어.", "엘리베이터"),
            ("택배 보관함 비밀번호 문자가 안 와서 물건을 못 찾고 있어.", "택배 보관함"),
            ("인터폰 화면이 까맣게 나와서 누가 온 건지 모르겠어.", "인터폰"),
            ("분리수거장에 스티로폼 버리는 날 헷갈려서 들고 다시 올라왔어.", "분리수거"),
            ("집주인한테 수리 요청 문자 보내야 하는데 괜히 말 꺼내기 부담스러워.", "집주인"),
            ("월세집 벽에 못 박아도 되는지 몰라서 액자도 못 걸고 있어.", "월세집"),
            ("전입신고랑 확정일자 해야 한다는데 뭘 먼저 해야 할지 머리 아파.", "전입신고"),
            ("이사 온 집에 콘센트 위치가 애매해서 멀티탭 선이 방을 가로질러.", "콘센트"),
            ("커튼 설치하려고 줄자를 들었는데 치수를 잘못 잴까 봐 멈칫했어.", "커튼"),
            ("가구 조립하다가 나사가 하나 남아서 불길해졌어.", "나사"),
            ("침대 프레임이 삐걱거려서 뒤척일 때마다 신경 쓰여.", "침대"),
            ("책상 의자 바퀴가 하나 안 굴러가서 계속 삐뚤게 움직여.", "의자"),
            ("청소기 흡입력이 약해져서 필터를 갈아야 하나 먼지통을 비워야 하나 모르겠어.", "청소기"),
            ("공기청정기 필터 교체 알림이 계속 떠서 나한테 잔소리하는 것 같아.", "공기청정기"),
            ("정수기 필터 교체 날짜가 지나서 물 마실 때마다 찝찝해.", "정수기"),
            ("전자레인지 돌리는데 안에서 번쩍해서 바로 멈췄어.", "전자레인지"),
            ("인덕션이 냄비를 인식 못 해서 요리 시작도 못 했어.", "인덕션"),
            ("설치 기사님 방문 시간이 오전 9시부터 오후 6시 사이라 하루가 묶였어.", "기사님"),
            ("AS센터 전화 연결이 20분째 안 돼서 대기음만 외울 것 같아.", "AS센터"),
            ("수리비 견적이 새 제품 가격이랑 비슷해서 그냥 새로 살까 고민돼.", "수리비"),
            ("보증기간 하루 지나고 고장 난 거 실화냐고 멘탈 나갔어.", "보증기간"),
            ("제품 설명서가 너무 두꺼워서 고장 해결 전에 내가 먼저 지쳐.", "설명서"),
            ("리모컨 건전지 갈았는데도 안 돼서 리모컨 문제인지 TV 문제인지 모르겠어.", "리모컨"),
            ("스마트 조명 앱이 먹통이라 불 끄려고 결국 일어났어.", "스마트 조명"),
            ("로봇청소기가 카펫에 걸려서 계속 구조 요청처럼 삐삐거려.", "로봇청소기"),
            ("빨래건조대가 한쪽으로 기울어서 옷 널다가 무너질까 봐 불안해.", "빨래건조대"),
            ("베란다 배수구가 막힌 것 같아서 비 오면 물 넘칠까 봐 걱정돼.", "베란다"),
            ("도배한 지 얼마 안 됐는데 벽지가 살짝 벌어져서 자꾸 눈에 밟혀.", "벽지"),
            ("화재경보기 배터리 경고음이 새벽마다 삐 하고 울려서 잠을 설쳐.", "화재경보기"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-home-repair-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_homerepair_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_sports_fan_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("어제 야구 9회말 투아웃에서 끝내기 안타 나와서 소리 질렀어.", "끝내기"),
            ("응원팀이 또 역전패해서 오늘 아침까지 기분이 안 좋아.", "역전패"),
            ("축구 중계 보다가 추가시간에 골 먹혀서 리모컨 던질 뻔했어.", "추가시간"),
            ("야구장 직관 가서 치킨 먹으면서 응원가 부르니까 스트레스 풀리더라.", "직관"),
            ("우리 팀 불펜이 또 불 질러서 다 이긴 경기 날렸어.", "불펜"),
            ("선발 투수가 7이닝 무실점 했는데 타선이 침묵해서 졌어.", "선발"),
            ("VAR 판정 기다리는 몇 분 동안 심장이 너무 쫄렸어.", "VAR"),
            ("농구 경기 4쿼터 막판에 버저비터 터져서 진짜 영화 같았어.", "버저비터"),
            ("배구 듀스 접전 보는데 한 점 한 점이 너무 피 말렸어.", "듀스"),
            ("오늘 롤 결승 보는데 5세트 밴픽부터 손이 떨리더라.", "밴픽"),
            ("우리 팀이 홈런 세 방 치고도 수비 실책 때문에 졌어.", "실책"),
            ("원정 응원 갔는데 목 다 쉬고 돌아왔어.", "원정"),
            ("축구장에서 직접 응원가 부르니까 TV로 보는 거랑 완전 다르더라.", "응원가"),
            ("심판 판정이 너무 이상해서 채팅창이 폭발했어.", "심판"),
            ("시즌권 살까 말까 고민 중인데 통장이 말리는 중이야.", "시즌권"),
            ("야구 우천 취소 떠서 경기장 앞에서 허탈하게 돌아왔어.", "우천 취소"),
            ("직관 갔는데 내 앞자리 사람이 계속 일어나서 시야를 가렸어.", "시야"),
            ("응원봉 배터리 나가서 하필 클라이맥스 때 불이 꺼졌어.", "응원봉"),
            ("치어리더 응원 따라 하다가 옆자리 사람이랑 눈 마주쳐서 민망했어.", "치어리더"),
            ("축구 대표팀 경기 있는 날은 괜히 치킨부터 시키게 돼.", "대표팀"),
            ("연장전까지 가서 새벽에 끝났는데 잠이 확 달아났어.", "연장전"),
            ("승부차기 보는데 차는 사람보다 내가 더 긴장했어.", "승부차기"),
            ("팀 유니폼 새로 나왔는데 디자인이 애매해서 살지 말지 고민돼.", "유니폼"),
            ("마킹한 선수가 이적해서 유니폼 입을 때마다 마음이 복잡해.", "마킹"),
            ("해설위원이 너무 편파적이라 듣다가 음소거할 뻔했어.", "해설"),
            ("경기장 매점 줄이 너무 길어서 한 이닝을 통째로 놓쳤어.", "매점"),
            ("원정석 표 잡으려고 예매창에서 손 떨면서 새로고침했어.", "예매"),
            ("우리 팀 감독 작전이 이해가 안 돼서 머리 싸맸어.", "감독"),
            ("상대 팀 에이스만 만나면 우리 타자들이 얼어붙는 것 같아.", "에이스"),
            ("연패가 길어지니까 하이라이트도 보기 싫어졌어.", "연패"),
            ("연승 중이라 경기 없는 날도 괜히 순위표만 계속 보게 돼.", "순위표"),
            ("더비 매치라 그런지 경기 전부터 분위기가 살벌했어.", "더비"),
            ("클러치 상황에서 우리 팀 에이스가 딱 해결해주니까 소름 돋았어.", "클러치"),
            ("중계 딜레이 때문에 옆집 환호 소리로 골을 먼저 알았어.", "딜레이"),
            ("야구장에서 파울볼 날아와서 순간 몸이 굳었어.", "파울볼"),
            ("응원단장 텐션이 너무 좋아서 나도 모르게 계속 따라 했어.", "응원단장"),
            ("비디오 판독 결과 뒤집히는 순간 경기장 분위기가 확 바뀌었어.", "비디오 판독"),
            ("드래프트에서 우리 팀이 뽑은 신인 보니까 괜히 기대돼.", "드래프트"),
            ("FA 영입 소식 떴는데 연봉 보고 잠깐 숨 멎었어.", "FA"),
            ("부상 복귀한 선수가 첫 타석부터 안타 쳐서 울컥했어.", "부상 복귀"),
            ("라이벌 팀 팬 친구랑 같이 경기 보다가 말싸움 날 뻔했어.", "라이벌"),
            ("직관 가는 날 비 예보 있으면 하루 종일 날씨 앱만 보게 돼.", "날씨"),
            ("플레이오프 티켓팅 실패해서 하루 종일 멍했어.", "플레이오프"),
            ("결승전 패배 후 선수들 표정 보는데 괜히 마음이 짠했어.", "결승전"),
            ("중계 보면서 혼자 전술 분석하는데 감독 빙의한 것 같아.", "전술"),
            ("야구장 맥주 한 모금이 왜 집에서 마시는 것보다 훨씬 맛있지.", "맥주"),
            ("축구장 원정석에서 목 터져라 응원했더니 다음 날 목소리가 안 나와.", "목소리"),
            ("팬카페에서 경기 끝나자마자 분위기 싸해져서 눈팅만 했어.", "팬카페"),
            ("우승 퍼레이드 영상 보는데 팬들 울컥하는 게 이해되더라.", "우승"),
            ("오늘 경기는 져도 내용이 좋아서 이상하게 덜 속상했어.", "내용"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-sports-fan-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_sportsfan_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_performance_culture_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("콘서트 티켓팅 들어갔는데 대기번호 3만 번 떠서 시작부터 멘탈 나갔어.", "티켓팅"),
            ("예매창에서 이선좌만 다섯 번 보고 결국 광탈했어.", "이선좌"),
            ("스탠딩 번호가 애매해서 앞도 뒤도 아닌 자리 될까 봐 걱정돼.", "스탠딩"),
            ("공연장 입장 줄이 너무 길어서 시작 전부터 체력 다 빠졌어.", "입장 줄"),
            ("콘서트에서 첫 곡 전주 나오자마자 온몸에 소름 돋았어.", "첫 곡"),
            ("앵콜 때 관객들이 다 같이 떼창하는데 진짜 울컥했어.", "앵콜"),
            ("내 자리 시야가 기둥에 가려서 무대 절반이 안 보였어.", "시야"),
            ("좌석은 멀었는데 음향이 좋아서 생각보다 만족했어.", "음향"),
            ("MD 줄이 공연 줄보다 길어서 굿즈 사다가 진이 빠졌어.", "MD"),
            ("응원봉 동기화 안 돼서 나만 다른 색으로 번쩍였어.", "응원봉"),
            ("뮤지컬 커튼콜 때 배우들 표정 보니까 여운이 확 몰려왔어.", "커튼콜"),
            ("뮤지컬 넘버가 머리에 계속 맴돌아서 집 오는 내내 흥얼거렸어.", "넘버"),
            ("연극 소극장 맨 앞줄 앉았더니 배우 숨소리까지 들려서 긴장됐어.", "소극장"),
            ("공연 중간에 옆자리 사람이 계속 기침해서 집중이 깨졌어.", "옆자리"),
            ("페스티벌 갔는데 비 와서 신발이 진흙투성이 됐어.", "페스티벌"),
            ("야외 페스티벌에서 돗자리 자리 잡는 게 거의 전쟁이더라.", "돗자리"),
            ("락 페스티벌 스탠딩에서 사람들 파도에 휩쓸려서 정신없었어.", "락 페스티벌"),
            ("페스티벌 푸드트럭 줄 서다가 좋아하는 가수 무대 놓쳤어.", "푸드트럭"),
            ("전시회 갔는데 작품보다 굿즈샵에서 더 오래 있었어.", "전시회"),
            ("미술관에서 설명 오디오 들으니까 작품이 갑자기 다르게 보이더라.", "오디오"),
            ("전시장 조명이 너무 좋아서 사진이 전부 인생샷처럼 나왔어.", "조명"),
            ("작품 앞에서 오래 보고 있는데 뒤 사람이 눈치 줘서 조금 민망했어.", "작품"),
            ("영화 시사회 당첨돼서 퇴근하고 바로 달려갔어.", "시사회"),
            ("GV에서 감독 얘기 듣고 나니까 영화 해석이 확 달라졌어.", "GV"),
            ("팬미팅 좌석이 뒤쪽이라 얼굴은 잘 안 보였는데 분위기는 좋았어.", "팬미팅"),
            ("하이터치회에서 최애랑 1초 눈 마주쳤는데 기억이 통째로 날아갔어.", "하이터치"),
            ("사인회 응모권 넣었는데 결과 발표 전까지 손이 떨려.", "사인회"),
            ("포토카드 교환하려고 현장에서 처음 보는 사람이랑 딜했어.", "포토카드"),
            ("전시 얼리버드 티켓 사놓고 날짜 까먹어서 날릴 뻔했어.", "얼리버드"),
            ("공연 예매 수수료가 은근 아까워서 결제 직전에 멈칫했어.", "수수료"),
            ("콘서트 끝나고 지하철역까지 인파에 떠밀려 갔어.", "인파"),
            ("공연 끝나고 귀가 먹먹해서 현실 소리가 멀게 들렸어.", "귀"),
            ("떼창 영상 다시 보는데 그날 분위기가 생각나서 또 벅차더라.", "떼창"),
            ("공연장 물품보관함이 다 차서 가방 들고 스탠딩 뛰었어.", "물품보관함"),
            ("입장 팔찌 잃어버릴까 봐 하루 종일 손목만 확인했어.", "입장 팔찌"),
            ("페스티벌 타임테이블 겹쳐서 누구 무대를 볼지 너무 고민돼.", "타임테이블"),
            ("뮤지컬 캐스팅 보드 앞에서 사진 찍는 사람들 줄이 엄청 길었어.", "캐스팅 보드"),
            ("오케스트라 공연에서 첫 바이올린 소리 나오자마자 숨이 멎는 줄 알았어.", "바이올린"),
            ("클래식 공연 중에 박수 타이밍 몰라서 눈치만 봤어.", "박수"),
            ("전시 도슨트 시간이 딱 맞아서 설명 들으며 천천히 봤어.", "도슨트"),
            ("팝업 전시 예약 시간보다 늦어서 입장 못 할까 봐 뛰어갔어.", "팝업"),
            ("공연장 냉방이 너무 세서 감동보다 추위가 먼저 왔어.", "냉방"),
            ("무대 효과 불꽃 터질 때 깜짝 놀라서 음료 쏟을 뻔했어.", "불꽃"),
            ("콘서트 라이브 밴드 사운드가 너무 좋아서 음원보다 더 좋았어.", "라이브 밴드"),
            ("공연 중 휴대폰 촬영 금지인데 앞사람이 계속 찍어서 신경 쓰였어.", "촬영 금지"),
            ("암전되는 순간 객석이 조용해지는데 그 긴장감이 너무 좋았어.", "암전"),
            ("인터미션 때 화장실 줄이 너무 길어서 2막 시작할까 봐 조마조마했어.", "인터미션"),
            ("오페라 자막 보랴 무대 보랴 눈이 너무 바빴어.", "오페라"),
            ("전시 마지막 방에서 갑자기 눈물 날 만큼 마음에 드는 작품을 만났어.", "작품"),
            ("공연 보고 나와서 바로 다음 회차 표를 또 예매하고 싶어졌어.", "다음 회차"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
            "현금영수증",
            "10억이어도",
            "상대 배우 눈",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-performance-culture-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_performancefan_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_outing_belongings_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("집 나왔는데 가스불 껐는지 갑자기 기억 안 나서 계속 불안해.", "가스"),
            ("현관문 잠갔는지 기억이 안 나서 엘리베이터 타고 내려왔다가 다시 올라갔어.", "현관문"),
            ("버스 탔는데 지갑을 집에 두고 온 것 같아서 순간 심장이 철렁했어.", "지갑"),
            ("우산 챙겼는데 비가 안 오고, 안 챙긴 날만 꼭 비가 와. 날씨가 나랑 싸우나 봐.", "우산"),
            ("이어폰 챙긴 줄 알았는데 케이스만 있고 본체가 없어서 출근길이 조용해졌어.", "이어폰"),
            ("충전기 가져온 줄 알았는데 케이블만 있고 어댑터가 없더라. 반쪽짜리 준비성.", "충전기"),
            ("약속 나가기 직전에 립밤이 안 보여서 가방을 세 번 뒤졌어.", "립밤"),
            ("분명히 손에 들고 있던 카드가 10초 뒤에 사라져서 집 안을 뒤집었어.", "카드"),
            ("택배 찾으러 내려갔는데 공동현관 비밀번호가 갑자기 생각 안 났어.", "공동현관"),
            ("편의점 가려고 나왔는데 마스크 안 가져온 걸 엘리베이터에서 알았어.", "마스크"),
            ("회사 도착해서 보니까 노트북 충전기를 집 책상에 꽂아두고 왔어.", "노트북"),
            ("학교 가방 챙겼는데 필통만 안 넣어서 하루 종일 펜 빌려 다녔어.", "필통"),
            ("운동 가려고 나왔는데 운동화 대신 슬리퍼 신고 나온 걸 버스정류장에서 알았어.", "운동화"),
            ("헬스장 갔는데 회원카드 안 가져와서 입구에서 멍해졌어.", "회원카드"),
            ("목욕탕 가려고 갔는데 수건을 안 챙겨서 시작부터 망했어.", "수건"),
            ("마트 계산대 앞에서 장바구니를 집에 두고 온 걸 깨달았어.", "장바구니"),
            ("장 보러 갔는데 정작 사야 할 우유는 까먹고 과자만 잔뜩 사 왔어.", "우유"),
            ("집 앞에 쓰레기 버리러 나갔다가 문 잠겨서 잠깐 갇힐 뻔했어.", "문"),
            ("자동차 키가 안 보여서 한참 찾았는데 외투 주머니에 있었어.", "자동차 키"),
            ("자전거 자물쇠 비밀번호가 갑자기 기억 안 나서 길에서 얼어붙었어.", "자물쇠"),
            ("엘리베이터 타고 내려가는데 택배 송장 번호 캡처 안 해둔 게 생각났어.", "송장"),
            ("은행 가려고 했는데 신분증을 안 챙겨서 그대로 돌아왔어.", "신분증"),
            ("병원 접수하려는데 보험증이나 신분증이 가방에 없어서 식은땀 났어.", "신분증"),
            ("시험 보러 갔는데 컴퓨터용 사인펜을 안 가져온 걸 시험장 앞에서 알았어.", "사인펜"),
            ("외출 전에 향수 뿌렸는데 너무 많이 뿌려서 엘리베이터에서 나 혼자 민망했어.", "향수"),
            ("머리 감고 나왔는데 한쪽 머리만 덜 말라서 계속 축축해.", "머리"),
            ("양말 한 짝이 계속 발뒤꿈치 밑으로 말려 들어가서 하루 종일 거슬려.", "양말"),
            ("새 신발 신고 나왔는데 뒤꿈치 까져서 집에 갈 때까지 고통이야.", "신발"),
            ("가방 지퍼를 열고 다닌 걸 한참 뒤에 알아서 물건 떨어졌나 확인했어.", "가방"),
            ("백팩 안에 물병 뚜껑이 덜 닫혀서 책이 살짝 젖었어. 진짜 속상해.", "물병"),
            ("보조배터리 챙긴 줄 알았는데 충전이 하나도 안 돼 있었어.", "보조배터리"),
            ("휴대폰 집에 두고 나온 걸 지하철 개찰구 앞에서 깨달았어.", "휴대폰"),
            ("카페에서 자리 잡아놓고 주문하러 갔는데 내 가방을 두고 가도 되나 불안했어.", "가방"),
            ("도서관 가서 공부하려는데 이어폰을 안 가져와서 주변 소리가 다 들려.", "이어폰"),
            ("우체국 가서 택배 보내려는데 주소를 정확히 안 적어와서 멍해졌어.", "주소"),
            ("예약 시간 맞춰 나왔는데 지하철 반대 방향을 타서 모든 계획이 꼬였어.", "반대 방향"),
            ("버스에서 내리려는데 교통카드가 가방 깊숙이 들어가서 급하게 뒤졌어.", "교통카드"),
            ("카페에서 노트북 펴려고 했는데 와이파이 비밀번호를 못 찾아서 멍때렸어.", "와이파이"),
            ("독서실 사물함 비밀번호를 까먹어서 내 책이 눈앞에 있는데 못 꺼냈어.", "사물함"),
            ("렌즈 끼고 나왔는데 렌즈통이랑 안경을 안 챙겨서 눈이 피곤해도 버티는 중.", "렌즈"),
            ("점심 먹고 나왔는데 앞니에 고춧가루 꼈을까 봐 계속 신경 쓰여.", "고춧가루"),
            ("흰옷 입은 날만 커피가 더 위험해 보여. 오늘도 컵 들고 긴장함.", "흰옷"),
            ("비 오는 날 바지 밑단이 젖어서 양말까지 축축해졌어.", "바지"),
            ("우산 접다가 손에 물 다 묻어서 지하철 손잡이 잡기 찝찝해.", "우산"),
            ("모자 쓰고 나왔는데 바람이 너무 세서 계속 날아갈까 봐 손으로 잡고 다녔어.", "모자"),
            ("안경 닦는 천을 안 챙겨서 하루 종일 뿌연 렌즈로 버티는 중.", "안경"),
            ("출근길에 커피 사 들고 뛰다가 뚜껑 덜 닫혀서 손에 다 샜어.", "커피"),
            ("약속 장소 도착했는데 내가 예약한 가게 이름이 기억 안 나서 채팅방 뒤졌어.", "가게"),
            ("집에 들어왔는데 열쇠를 문에 꽂아둔 채로 들어온 걸 뒤늦게 알았어.", "열쇠"),
            ("빨래 널고 나온 줄 알았는데 세탁기 안에 그대로 둔 게 갑자기 생각났어.", "세탁기"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-outing-belongings-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_specialized_digital_work_device_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("메일에 첨부파일 보낸다고 써놓고 정작 파일 안 붙여서 재송부했어. 민망해.", "첨부파일"),
            ("회사 전체 메일에 답장 눌렀어야 하는데 전체답장 눌러서 식은땀 났어.", "전체답장"),
            ("메일 제목 오타 난 채로 보냈는데 발송 취소 시간이 지나서 그냥 체념함.", "메일"),
            ("회의 링크 눌렀는데 권한 없음 떠서 시작부터 허둥댔어.", "회의 링크"),
            ("줌 회의에서 마이크 켜진 줄 모르고 혼잣말했다가 바로 얼어붙음.", "마이크"),
            ("화상회의 카메라 켰는데 뒤에 빨래 널린 거 보여서 급하게 껐어.", "카메라"),
            ("회의 중 화면공유 했는데 엉뚱한 카톡창이 떠서 심장이 내려앉음.", "화면공유"),
            ("캘린더 초대 시간 잘못 보고 회의 30분 늦게 들어갔어.", "캘린더"),
            ("팀즈 알림이 너무 많이 떠서 진짜 일보다 알림 끄는 시간이 더 길어.", "팀즈"),
            ("슬랙 멘션이 계속 울려서 집중하려고 해도 뇌가 쪼개지는 느낌이야.", "슬랙"),
            ("프린터가 종이 걸림만 띄우고 아무것도 안 뽑아줘서 싸우는 중.", "프린터"),
            ("PDF 파일 비밀번호 까먹어서 문서를 눈앞에 두고도 못 열고 있어.", "PDF"),
            ("한글 문서 저장 안 하고 닫아서 방금 쓴 내용이 전부 날아갔어.", "저장"),
            ("엑셀 수식 하나 틀렸는데 표 전체 숫자가 다 이상해져서 멘붕 왔어.", "엑셀"),
            ("파일 이름 최종_진짜최종_수정본_final 이렇게 돼서 뭐가 최신인지 모르겠어.", "최종"),
            ("다운로드 폴더가 너무 지저분해서 방금 받은 파일을 못 찾겠어.", "다운로드"),
            ("스크린샷 찍었는데 듀얼 모니터 전체가 찍혀서 필요 없는 것까지 다 나왔어.", "스크린샷"),
            ("압축파일 풀었는데 폴더 안에 폴더 안에 또 폴더라서 길을 잃었어.", "압축파일"),
            ("USB 꽂았는데 인식이 안 돼서 포트 바꿔가며 기도 중이야.", "USB"),
            ("노트북 업데이트가 지금 다시 시작하라고 해서 회의 전에 심장이 쫄렸어.", "업데이트"),
            ("클라우드 동기화 충돌 떠서 어느 파일이 진짜 최신인지 모르겠어.", "클라우드"),
            ("구글 드라이브 용량 꽉 찼다고 해서 뭐부터 지워야 할지 막막해.", "드라이브"),
            ("사진 백업 중이라더니 와이파이 잡자마자 폰이 갑자기 느려졌어.", "백업"),
            ("브라우저 탭을 너무 많이 열어놔서 어디서 음악이 나오는지 못 찾겠어.", "탭"),
            ("비밀번호 자동완성 안 떠서 내가 만든 비번을 내가 못 맞히고 있어.", "비밀번호"),
            ("OTP 문자가 늦게 와서 입력하려는 순간 시간이 만료됐어.", "OTP"),
            ("2단계 인증 앱을 예전 폰에 두고 와서 로그인 못 하는 중이야.", "2단계 인증"),
            ("아이디 찾기 하려는데 가입한 이메일이 뭔지 기억이 안 나.", "아이디"),
            ("비밀번호 재설정 메일이 스팸함에 숨어 있어서 10분 날렸어.", "스팸함"),
            ("자동로그인 풀려서 평소 쓰던 사이트가 갑자기 낯설어졌어.", "자동로그인"),
            ("폰 저장공간 부족하다고 떠서 사진 지우는데 추억 정리하는 기분이야.", "저장공간"),
            ("앱 업데이트 하고 나서 버튼 위치가 바뀌어서 손이 계속 허공을 눌러.", "앱 업데이트"),
            ("알림 권한을 잘못 꺼놨더니 중요한 메시지를 늦게 봤어.", "알림"),
            ("방해금지 모드 켜둔 걸 까먹고 연락 다 씹은 사람 됐어.", "방해금지"),
            ("스팸 전화인 줄 알고 안 받았는데 알고 보니 택배 기사님이었어.", "스팸 전화"),
            ("카톡 사진 원본으로 보내야 했는데 일반 화질로 보내서 다시 보냈어.", "원본"),
            ("단톡방 공지 확인 안 했다가 나만 일정 잘못 알고 있었어.", "단톡방"),
            ("메모 앱에 적어둔 중요한 내용 검색해도 안 나와서 제목을 원망 중이야.", "메모 앱"),
            ("할 일 앱 알림은 뜨는데 계속 미루기만 눌러서 알림이 나를 싫어할 듯.", "할 일"),
            ("녹음 파일 켰는데 바람 소리만 잔뜩 들어가서 중요한 말이 하나도 안 들려.", "녹음"),
            ("블루투스 이어폰 연결된 줄 알았는데 폰 스피커로 노래가 크게 나왔어.", "블루투스"),
            ("무선 마우스 배터리가 회의 중에 죽어서 커서가 그대로 멈췄어.", "무선 마우스"),
            ("키보드 한영키가 안 먹어서 영어로만 분노를 입력하는 중.", "한영키"),
            ("모니터 케이블 살짝 빠진 줄 모르고 컴퓨터 고장 난 줄 알았어.", "모니터"),
            ("와이파이는 연결됐는데 인터넷 없음 뜨면 세상에서 제일 억울해.", "와이파이"),
            ("공유기 껐다 켜면 해결된다는 말에 또 전원 뽑고 10초 세는 중.", "공유기"),
            ("이어폰 한쪽만 연결돼서 노래가 반쪽짜리로 들려.", "이어폰"),
            ("휴대폰 화면 밝기 자동조절이 자꾸 어두워져서 밖에서 아무것도 안 보여.", "밝기"),
            ("배터리 절약 모드 켜놨더니 알림도 늦고 화면도 답답해졌어.", "배터리"),
            ("케이블 꽂았는데 충전 중이 아니라 액세서리 인식 중만 떠서 킹받아.", "케이블"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "어느 지역 기준",
            "키운다면 고양이",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-specialized-digital-work-device-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_specialized_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_foundation_bills_apps_home_cooking_local_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("전기요금 고지서 봤는데 에어컨 몇 번 틀었다고 금액이 갑자기 튀어서 심장이 철렁했어.", "전기요금"),
            ("가스비 보고 이번 겨울엔 보일러 대신 두꺼운 양말로 버텨야 하나 진지하게 고민 중.", "가스비"),
            ("수도요금이 갑자기 많이 나와서 혹시 집 어딘가 누수 있는 거 아닌가 불안해.", "수도요금"),
            ("인터넷 약정 끝났다는 문자 왔는데 통신사 갈아타면 사은품 더 받을 수 있을까 계산 중.", "인터넷 약정"),
            ("휴대폰 요금 자동이체 카드가 만료돼서 미납 문자 오니까 괜히 죄지은 기분이야.", "자동이체"),
            ("구독 서비스 해지하려는데 해지 버튼을 꼭 미로처럼 숨겨놔서 짜증 난다.", "구독 서비스"),
            ("비밀번호 재설정 메일이 안 와서 로그인 못 하고 새로고침만 계속 누르는 중.", "비밀번호"),
            ("공동인증서 갱신하라는데 설치 프로그램만 세 개 깔리고 아직도 로그인 못 했어.", "공동인증서"),
            ("앱 업데이트하고 나서 잘 쓰던 기능 위치가 싹 바뀌어서 적응이 안 돼.", "앱 업데이트"),
            ("카드 명세서 보다가 내가 이런 걸 샀다고? 하고 과거의 나를 의심하게 됐어.", "카드 명세서"),
            ("싱크대 수전에서 물이 똑똑 새는데 기사님 부르자니 출장비가 더 무서울 것 같아.", "싱크대"),
            ("변기 물이 계속 졸졸 내려가서 밤에 누워있으면 물소리 때문에 신경 쓰여.", "변기"),
            ("현관 센서등이 혼자 켜졌다 꺼졌다 해서 고장인지 귀신인지 모르겠어.", "센서등"),
            ("도어락 배터리 부족 경고음이 나는데 건전지 사러 나가기 귀찮아서 계속 미루는 중.", "도어락"),
            ("냉장고에서 이상한 소리가 나는데 멈추면 더 무섭고 계속 나도 신경 쓰인다.", "냉장고"),
            ("전자레인지 돌렸는데 안에서 펑 소리 나서 열어보니 소스가 사방에 튀었어.", "전자레인지"),
            ("세탁기 탈수할 때 집 전체가 흔들리는 것 같아서 이웃집에 들릴까 봐 눈치 보여.", "세탁기"),
            ("전구 갈아야 하는데 천장이 높아서 의자 위에 올라가는 순간부터 무섭다.", "전구"),
            ("창문 방충망이 찢어졌는데 벌레 들어올까 봐 밤마다 괜히 신경 쓰여.", "방충망"),
            ("화장실 환풍기 소리가 점점 커져서 켤 때마다 비행기 이륙하는 줄 알아.", "환풍기"),
            ("밥 하려고 쌀 씻어놨는데 취사 버튼 안 눌러서 한 시간 뒤에 생쌀만 마주했어.", "취사 버튼"),
            ("계란 삶다가 타이머 안 맞춰서 반숙도 완숙도 아닌 애매한 계란이 됐어.", "계란"),
            ("국 끓이다가 소금 너무 많이 넣어서 물 붓고 또 붓다가 냄비가 한강 됐어.", "소금"),
            ("프라이팬 예열하다가 딴짓해서 연기 나고 화재경보기 울릴 뻔했어.", "프라이팬"),
            ("김치통 열었는데 국물이 손에 묻어서 하루 종일 김치 냄새 나는 느낌이야.", "김치통"),
            ("양파 썰다가 눈물 줄줄 나서 내가 요리하는 건지 이별하는 건지 모르겠더라.", "양파"),
            ("배달 음식 남은 거 냉장고 넣어놓고 까먹었다가 포장 용기째 유물이 됐어.", "배달 음식"),
            ("커피포트 물 끓여놓고 까먹어서 다시 끓이고 또 까먹는 루프에 빠졌어.", "커피포트"),
            ("식빵 구워놓고 토스터 안에서 꺼내는 걸 까먹어서 딱딱한 벽돌이 됐어.", "토스터"),
            ("설거지 다 한 줄 알았는데 싱크대 구석에 컵 하나 남아있는 거 보면 힘 빠져.", "설거지"),
            ("동네 마트 갔다가 원래 사려던 우유는 까먹고 과자만 잔뜩 사 왔어.", "우유"),
            ("편의점 1+1 행사 보이면 필요 없어도 일단 집어 들게 되는 병이 있어.", "1+1"),
            ("약국 가서 증상 설명하려는데 막상 내 차례 되면 어디가 아픈지 말이 꼬여.", "약국"),
            ("우체국 택배 보내러 갔는데 박스 크기 고르는 데서부터 멘붕 왔어.", "우체국"),
            ("동네 세탁방에서 건조기 기다리는데 앞사람 빨래가 끝났는데도 안 가져가서 답답해.", "세탁방"),
            ("분식집 포장 주문했는데 집에 와서 보니 순대 소금이 빠져 있어서 서운해.", "순대 소금"),
            ("카페 쿠폰 열 개 모아서 드디어 무료 음료 먹는 날이라 괜히 뿌듯했어.", "쿠폰"),
            ("미용실 예약 시간보다 일찍 도착해서 어색하게 잡지만 세 번 넘기고 있었어.", "미용실 예약"),
            ("택배 보관함 비밀번호 문자가 안 와서 내 물건 앞에 두고도 못 꺼내고 있어.", "택배 보관함"),
            ("아파트 관리사무소에 전화해야 하는데 괜히 말이 길어질까 봐 미루고 있어.", "관리사무소"),
            ("친구 집들이 선물로 휴지랑 세제 중에 뭐가 더 실용적일지 고민돼.", "집들이 선물"),
            ("조카 생일 선물 사야 하는데 요즘 애들이 뭘 좋아하는지 감이 하나도 안 와.", "조카 생일"),
            ("부모님 스마트폰 설정 봐드리다가 나도 모르는 메뉴가 나와서 같이 헤매는 중.", "스마트폰 설정"),
            ("가족 단톡방에 사진 잘못 올렸다가 삭제했는데 이미 다 본 것 같아서 민망해.", "가족 단톡방"),
            ("친척 결혼식 장소가 너무 멀어서 축하보다 이동 시간이 먼저 계산돼.", "친척 결혼식"),
            ("동네 병원 리뷰를 보는데 평이 극과 극이라 어디로 가야 할지 더 헷갈려.", "병원 리뷰"),
            ("미용실에서 원하는 머리 사진 보여줬는데 결과가 사진 속 사람과 나의 현실 차이를 알려줬어.", "사진"),
            ("옷 수선 맡기려는데 수선비가 옷값이랑 비슷해서 이걸 살려야 하나 고민 중.", "수선비"),
            ("신발 밑창이 닳아서 비 오는 날마다 미끄러질까 봐 조심조심 걷게 돼.", "밑창"),
            ("주차장 정산기 앞에서 카드가 한 번에 안 먹히면 뒤차 눈치 보여서 손에 땀 나.", "정산기"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "어느 지역 기준",
            "어린 시절의 나",
            "결혼 로망",
            "나만 그런 느낌",
            "거대 개미집",
            "철학",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-foundation-bills-apps-home-cooking-local-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_lifestyle_goods_car_pets_shopping_growth_batch_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("이번에 최애 장르 팝업스토어 열린다는데 평일 웨이팅 몇 시간인지 눈치 게임 하는 중.", "팝업스토어"),
            ("한정판 아크릴 스탠드 구하려고 번개장터 키워드 알림 켜놨는데 뜨자마자 1초 만에 팔림. 눈물 난다.", "번개장터"),
            ("랜덤 굿즈 5개 샀는데 중복만 3개 나옴. 내 손은 왜 항상 최애를 피해 가는 똥손일까.", "랜덤 굿즈"),
            ("방 한쪽에 최애캐들 모아둔 '덕질 존' 청소함. 피규어 먼지 털 때가 세상에서 제일 경건한 시간임.", "덕질 존"),
            ("해외 직구로 주문한 굿즈 한 달 만에 세관 통과함. 택배 조회 창만 하루에 10번씩 새로고침 중.", "세관"),
            ("공식 일러집 가격 선 넘어서 고민했는데, 받아서 종이 질감이랑 인쇄 상태 보니까 돈값 하더라.", "일러집"),
            ("단톡방에 굿즈 인테리어 인증샷 올렸더니 다들 '일코(일반인 코스프레) 완전 해제됐네'라며 뿜음.", "일코"),
            ("최애 성우 신작 드라씨(오디오 드라마) 떴다. 오늘 밤은 이거 무한 스트리밍 하면서 잔다.", "드라씨"),
            ("지갑 사정 안 좋아서 이번 시즌 굿즈는 패스하려 했는데, 실물 후기 사진 보니까 안 살 수가 없음.", "실물 후기"),
            ("동인 행사 가려고 아침 6시부터 첫차 타고 줄 서는 중. 이 열정으로 공부를 했으면 서울대 갔다.", "동인 행사"),
            ("주말에 큰맘 먹고 셀프 세차장 갔는데 폼건 쏘고 미트질 30분 했더니 팔 감각이 없어짐.", "셀프 세차"),
            ("골목길 운전하다가 맞은편에서 큰 차 오면 식은땀 쫙 흐름. 조용히 후진 기어 넣고 대기 타기.", "골목길"),
            ("내 차 네비가 자꾸 이상한 산길로 안내해서 멘붕 옴. 업그레이드 안 한 내 잘못이다.", "네비"),
            ("조수석에 사람 태우면 나도 모르게 운전 거칠어질까 봐 긴장해서 평소보다 속도 20km는 낮춤.", "조수석"),
            ("여름 오기 전에 에어컨 필터 셀프로 갈았는데, 글러브 박스 뜯다가 부품 부숴 먹을 뻔함.", "에어컨 필터"),
            ("기름값 조금이라도 아끼려고 주변에서 제일 저렴한 셀프 주유소 찾아서 원정 주유 다녀옴.", "셀프 주유소"),
            ("주차 자리가 경차 전용밖에 없어서 눈물 머금고 세 바퀴 더 돎. 주차 스트레스 장난 아니다.", "주차"),
            ("차에서 들을 드라이브 플레이리스트 짜는 데만 2시간 씀. 시티팝이랑 신나는 비트가 국룰이지.", "드라이브 플레이리스트"),
            ("고속도로 톨게이트 하이패스 차선 잘못 들어가서 사이렌 울릴 때... 심장 내려앉는 줄 앎.", "하이패스"),
            ("새 차 냄새 빼려고 방향제 샀는데 향이 너무 강해서 차 탈 때마다 머리 아픔. 그냥 창문 열고 다님.", "방향제"),
            ("우리 집 고양이 사료 바꿨더니 맛없다고 발로 모래 파는 시늉 함. 상전도 이런 상전이 없다.", "사료"),
            ("주말에 강아지 산책하러 공원 갔다가 다른 댕댕이한테 쫄아서 내 뒤로 숨는 겁쟁이 녀석.", "겁쟁이"),
            ("강아지 수제 간식 만든다고 건조기 10시간 돌렸는데 정작 지는 한 입 먹고 뱉음. 킹받네.", "수제 간식"),
            ("퇴근하고 문 열면 꼬리 프로펠러처럼 돌리면서 반겨주는 거 보려고 매일 칼퇴함. 힐링 그 자체.", "꼬리 프로펠러"),
            ("우리 고양이 맨날 노트북 키보드 위에 올라와서 식빵 구움. 화면에 이상한 외계어 쳐져 있음.", "키보드"),
            ("동물병원 가서 예방접종 맞추고 영수증 받았는데 내 병원비보다 비쌈. 열심히 벌어야 하는 이유.", "동물병원"),
            ("인스타 릴스 보는데 남의 집 고양이는 손도 주고 빵도 하던데 우리 애는 물기만 함. 그래도 귀여워.", "물기"),
            ("털 빠짐 시기 왔나 봄. 검은색 옷 입고 나갔다가 온몸에 흰 털 테러당해서 돌돌이 필수임.", "돌돌이"),
            ("고양이 새 장난감 샀는데 장난감은 쳐다도 안 보고 겉에 포장 상자 안으로 쏙 들어가서 노는 중.", "포장 상자"),
            ("주말 아침에 눈 떴는데 고양이가 내 얼굴 바로 위에서 빤히 쳐다보고 있을 때의 그 묘한 압박감.", "압박감"),
            ("쇼핑 앱 장바구니에 30개 담아놨는데 결제 버튼 누르기 전에 이성이 작동해서 25개 삭제함.", "장바구니"),
            ("쿠팡 와우 배송 새벽에 문 앞에 툭 떨어지는 소리 들리면 알람 벨보다 더 빨리 잠 깨서 가지고 들어옴.", "쿠팡 와우"),
            ("인스타 공구로 산 주방 꿀템... 막상 써보니까 세척하기 개불편해서 싱크대 구석에 처박힘.", "인스타 공구"),
            ("사이즈 표 보고 바지 샀는데 허리가 안 맞아서 반품 신청함. 반품비 6천 원 날리는 게 제일 아까워.", "반품비"),
            ("택배 상자 뜯을 때 칼로 테이프 쫙 찢는 그 손맛... 현대인이 느낄 수 있는 소소한 도파민임.", "택배 상자"),
            ("무료 배송 금액 3천 원 채우려고 굳이 필요 없는 5천 원짜리 양말 추가하는 모순적인 소비.", "무료 배송"),
            ("오늘 택배 4개 동시에 옴. 엄마가 '너 도대체 뭘 그렇게 맨날 사냐'고 등짝 스매싱 날림.", "택배 4개"),
            ("중고 거래 하러 약속 장소 나갔는데 구매자가 쿨하게 네고 없이 돈 보내줘서 기분 째짐.", "네고"),
            ("해외 쇼핑몰 세일 기간이라 눈 돌아가서 장바구니 채우는 중. 관세 범위 안 넘게 계산기 두드림.", "관세"),
            ("분명히 꼭 필요해서 샀는데 막상 택배 뜯고 나면 흥미 식어서 방치해 두는 병에 걸림.", "방치"),
            ("올해는 진짜 영어 회화 마스터한다 하고 인강 결제했는데 출석률 10% 찍고 기부 천사 됨.", "영어 회화"),
            ("새벽 6시 기상 미라클 모닝 도전했다가, 알람 끄고 눈 깜빡 하니까 평소 일어나는 8시더라.", "미라클 모닝"),
            ("아이패드 필기용으로 프로 모델 샀는데 넷플릭스 머신 및 유튜브 시청용 거치대로 전락함.", "아이패드"),
            ("자기계발서 베스트셀러 사서 프롤로그만 3번 읽음. 뒤로 갈수록 집중력 흐려져서 책갈피 꽂아둠.", "프롤로그"),
            ("단톡방에 남들 자격증 따고 갓생 사는 인증샷 올라오면 조용히 나 자신을 돌아보며 반성하게 됨.", "자격증"),
            ("오늘 스터디 카페 끊고 자리 앉았는데 5분 집중하고 50분 동안 스마트폰 쇼츠 보다가 나옴.", "스터디 카페"),
            ("독서 모임 가입했는데 지정 도서 안 읽고 가서 요약본 대충 훑어보고 아는 척 연기함.", "독서 모임"),
            ("취미로 코딩 배워볼까 하고 책 샀는데 첫 페이지 Hello World 출력하고 책 덮음. 적성 안 맞다.", "Hello World"),
            ("계획 세우는 데만 다이어리 3페이지 쓰고 실천은 하루도 안 함. 계획 짜는 게 취미인 사람 됨.", "계획"),
            ("남들이랑 비교하지 말고 어제의 나보다 더 나아지면 됐지 뭐, 하고 치킨 시켜 먹는 엔딩.", "어제의 나"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "받아둘게",
            "너를 동물로 비유",
            "해안도로 드라이브",
            "고양이 쪽",
            "집밥 얘기면",
            "비트코인 풀매수",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(
                    f"offline-korean-daily-lifestyle-goods-car-pets-shopping-growth-{index}", prompt
                )
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_interior_work_micro_context_regressions(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            (
                "그래픽카드 서멀구리스 재도포 갈았더니 풀로드 온도가 5도 떨어짐. 이거 은근히 쾌감 있네.",
                "서멀구리스",
            ),
            (
                "로컬 vLLM 서버 구동해 놓고 디스코드 봇으로 연동했더니 스마트폰으로도 테스트 가능해서 신세계임.",
                "디스코드 봇",
            ),
            (
                "VRAM 쪼개 쓰려고 양자화 포맷별로 속도 비교 중인데 비트 수 너무 낮추니까 확실히 능지 박살 남.",
                "양자화",
            ),
            (
                "인스타에서 미드센추리 모던 감성 조명 보고 홀린 듯 결제했는데 우리 집 인테리어랑 따로 놀아서 처치 곤란.",
                "미드센추리",
            ),
            (
                "벽에 타공판 달아서 헤드셋이랑 자주 쓰는 가젯들 걸어두니까 묘하게 작업실 분위기 나고 좋더라.",
                "타공판",
            ),
            (
                "방에 초록색 식물 하나 두면 분위기 산대서 사 왔는데, 벌써 잎이 노랗게 변함. 식물 초보라 살려내야 됨.",
                "잎",
            ),
            (
                "이케아 쇼룸 구경 갔다가 눈 돌아가서 카트에 무지성으로 주워 담았는데 영수증 보고 정신 번쩍 듦.",
                "이케아",
            ),
            (
                "화이트 쉬폰 커튼으로 바꿨더니 햇빛 들어올 때 방 무드 미쳤음. 인스타 갬성 완성.",
                "쉬폰 커튼",
            ),
            (
                "오늘 당근마켓으로 상태 좋은 원목 협탁 2만 원에 겟함. 이런 게 소소한 삶의 지혜이자 낙이지.",
                "원목 협탁",
            ),
            (
                "침구 세트 바꿨더니 사각거리는 감촉 너무 좋아서 주말에 침대 밖으로 더 안 나가게 됨. 부작용 심각.",
                "침구 세트",
            ),
            (
                "방에 불 다 끄고 누웠는데 저 멀리 있는 멀티탭 불빛 유난히 거슬려서 결국 일어나서 끄고 옴.",
                "멀티탭",
            ),
            (
                "오늘 유독 일하기 싫어서 키보드 타자만 쓸데없이 타타타탁 세게 치면서 바쁜 척함.",
                "키보드",
            ),
            (
                "스마트폰 충전기 선 짧아서 침대 모서리에 겨우 걸쳐 누워가지고 불편하게 폰 하는 중.",
                "충전기 선",
            ),
            (
                "퇴근 직전에 급하게 온 메일 읽씹하고 퇴근함. 내일의 내가 알아서 하겠지 뭐.",
                "내일의 나",
            ),
        ]
        forbidden = (
            "온라인에서 돈과 노출",
            "요즘 과자 몇 개",
            "어느 쪽 기준",
            "받아둘게",
            "무리하게 밀 필요",
            "동물로 비유",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-interior-work-micro-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_lifestyle_goods_car_pets_shopping_growth_structure(self) -> None:
        engine = _build_draft_only_engine()
        black_root = Path(__file__).resolve().parents[1]
        prompts_path = black_root / "reports" / "korean_daily_lifestyle_goods_car_pets_shopping_growth_20260514.txt"
        expected_path = (
            black_root
            / "reports"
            / "korean_daily_lifestyle_goods_car_pets_shopping_growth_20260514.structure_expect.jsonl"
        )
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_rows = [
            json.loads(line)
            for line in expected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_rows), 50)

        for index, (prompt, expected) in enumerate(zip(prompts, expected_rows, strict=True), start=1):
            with self.subTest(index=index, prompt=prompt):
                self.assertEqual(expected["index"], index)
                result = await engine.respond(
                    f"offline-korean-daily-lifestyle-goods-car-pets-shopping-growth-structure-{index}",
                    prompt,
                )
                draft = result.draft_utterance or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(draft.get("action"), expected["expected_action"])
                self.assertEqual(draft.get("draft_domain"), expected["expected_domain"])
                self.assertEqual(
                    draft.get("draft_frame_detail"),
                    expected["expected_draft_frame_detail"],
                )
                self.assertEqual(draft.get("direct_surface_reason"), expected["expected_reason"])

    async def test_korean_daily_everyday_variants_structure(self) -> None:
        engine = _build_draft_only_engine()
        black_root = Path(__file__).resolve().parents[1]
        prompts_path = black_root / "reports" / "korean_daily_everyday_variants_20260514.txt"
        expected_path = black_root / "reports" / "korean_daily_everyday_variants_20260514.structure_expect.jsonl"
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_rows = [
            json.loads(line)
            for line in expected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        forbidden = (
            "어느 쪽 기준인지",
            "무리하게 밀 필요",
            "가볍게 넘기진 않을게",
            "키운다면 고양이",
            "사진 하나가 그때 공기",
            "먹는 방식은 은근히",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_rows), 50)

        for index, (prompt, expected) in enumerate(zip(prompts, expected_rows, strict=True), start=1):
            with self.subTest(index=index, prompt=prompt):
                self.assertEqual(expected["index"], index)
                result = await engine.respond(f"offline-korean-daily-everyday-variant-structure-{index}", prompt)
                draft = result.draft_utterance or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(draft.get("action"), expected["expected_action"])
                self.assertEqual(draft.get("draft_domain"), expected["expected_domain"])
                self.assertEqual(
                    draft.get("draft_frame_detail"),
                    expected["expected_draft_frame_detail"],
                )
                self.assertEqual(draft.get("direct_surface_reason"), expected["expected_reason"])
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_practical_admin_variants_structure(self) -> None:
        engine = _build_draft_only_engine()
        black_root = Path(__file__).resolve().parents[1]
        prompts_path = black_root / "reports" / "korean_daily_practical_admin_variants_20260514.txt"
        expected_path = (
            black_root / "reports" / "korean_daily_practical_admin_variants_20260514.structure_expect.jsonl"
        )
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_rows = [
            json.loads(line)
            for line in expected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        forbidden = (
            "어느 쪽 기준인지",
            "무리하게 밀 필요",
            "가볍게 넘기진 않을게",
            "키운다면 고양이",
            "먹는 방식은 은근히",
            "화장까지 흘러내리면",
            "그 자리에서 조용히 말할래",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_rows), 50)

        for index, (prompt, expected) in enumerate(zip(prompts, expected_rows, strict=True), start=1):
            with self.subTest(index=index, prompt=prompt):
                self.assertEqual(expected["index"], index)
                result = await engine.respond(f"offline-korean-daily-practical-admin-structure-{index}", prompt)
                draft = result.draft_utterance or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(draft.get("action"), expected["expected_action"])
                self.assertEqual(draft.get("draft_domain"), expected["expected_domain"])
                self.assertEqual(
                    draft.get("draft_frame_detail"),
                    expected["expected_draft_frame_detail"],
                )
                self.assertEqual(draft.get("direct_surface_reason"), expected["expected_reason"])
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_niche_lifestyle_variants_structure(self) -> None:
        engine = _build_draft_only_engine()
        black_root = Path(__file__).resolve().parents[1]
        prompts_path = black_root / "reports" / "korean_daily_niche_lifestyle_variants_20260514.txt"
        expected_path = (
            black_root / "reports" / "korean_daily_niche_lifestyle_variants_20260514.structure_expect.jsonl"
        )
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_rows = [
            json.loads(line)
            for line in expected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        forbidden = (
            "어느 쪽 기준인지",
            "무리하게 밀 필요",
            "가볍게 넘기진 않을게",
            "키운다면 고양이",
            "먹는 방식은 은근히",
            "초반에 하나 고르자면",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_rows), 50)

        for index, (prompt, expected) in enumerate(zip(prompts, expected_rows, strict=True), start=1):
            with self.subTest(index=index, prompt=prompt):
                self.assertEqual(expected["index"], index)
                result = await engine.respond(f"offline-korean-daily-niche-lifestyle-structure-{index}", prompt)
                draft = result.draft_utterance or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(draft.get("action"), expected["expected_action"])
                self.assertEqual(draft.get("draft_domain"), expected["expected_domain"])
                self.assertEqual(
                    draft.get("draft_frame_detail"),
                    expected["expected_draft_frame_detail"],
                )
                self.assertEqual(draft.get("direct_surface_reason"), expected["expected_reason"])
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_classified_slot_reply_uses_prompt_specific_words(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            (
                "조별 과제 안 하던 무임승차 빌런 이름 발표 피피티 마지막 페이지 기여도에 0%로 박아버렸어.",
                ("발표 자료", "무임승차 빌런", "기여도 0%"),
            ),
            (
                "팀플 무임승차 빌런 기여도 0%로 적어버렸는데 속이 좀 시원하더라.",
                ("팀플", "무임승차 빌런", "기여도 0%"),
            ),
            (
                "조별과제에서 아무것도 안 한 무임승차 빌런한테 기여도 0점을 줬어.",
                ("조별과제", "무임승차 빌런", "기여도 0%"),
            ),
        ]

        for index, (prompt, expected_fragments) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"classified-slot-assembly-{index}", prompt)
                draft = result.draft_utterance or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(draft.get("action"), "share_opinion")
                self.assertEqual(draft.get("draft_domain"), "work_school")
                self.assertEqual(draft.get("draft_frame_detail"), "workrevenge_group_project_zero_credit")
                self.assertEqual(
                    draft.get("direct_surface_reason"),
                    "korean_daily_more_workrevenge_group_project_zero_credit",
                )
                for fragment in expected_fragments:
                    self.assertIn(fragment, result.reply)

    async def test_korean_daily_foundation_variants_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("비 오니까 국물 땡긴다.", "국물 당기는"),
            ("나 방금 일어났어. 더 자고 싶은데 억지로 일어났다.", "텐션은 천천히"),
            ("나 왔어. 오늘 진짜 길었다.", "네 하루 여기다"),
            ("잘자. 나 이제 자려고 누웠어.", "내일 다시 오면"),
            ("오늘따라 좀 외롭네.", "바로 여기서"),
            ("주말엔 집에서 뒹굴까 밖에 나갈까?", "집에서 회복"),
            ("방 청소 어디부터 해야 할지 모르겠어.", "바닥부터 5분"),
            ("기분전환 하고 싶은데 뭐가 좋을까?", "밝은 노래"),
            ("선 넘는 농담 들으면 어떻게 받아쳐야 해?", "그 말은 별로야"),
            ("후회 없는 삶을 살려면 뭐가 중요할까?", "미루지 않는"),
            ("무언가를 새로 시작하기엔 너무 늦은 것 같아서 두려워.", "오늘 한 칸"),
            ("오늘따라 아무 이유 없이 눈물이 날 것 같아.", "버틴 것"),
            ("늦잠 자서 지각하게 생겼어 어떡하지?", "짧게 연락"),
            ("다이어트 중인데 치킨이 너무 먹고 싶어.", "치킨 땡기는"),
            ("요리보다 설거지가 진짜 싫어.", "설거지"),
            ("폰 배터리 3%밖에 안 남았어.", "충전기"),
            ("소개팅에서 대화가 끊기면 뭘 물어봐야 해?", "가벼운 취향"),
            ("네가 게임 속 NPC라면 매일 반복할 대사는 뭐야?", "NPC라면"),
            ("넌 나랑 대화하는 거 지루하지 않아?", "지루하지 않아"),
            ("나한테 역으로 질문 하나만 해봐.", "내가 하나 물어볼게"),
            ("아침부터 머리가 멍해서 일이 손에 안 잡혀.", "작은 일"),
            ("저녁에 혼자 먹을 건데 대충 때우기 싫어.", "제대로 먹은 느낌"),
            ("커피를 마셨는데도 잠이 안 깨.", "휴식을 더 크게"),
            ("친구한테 답장해야 하는데 귀찮아서 미루고 있어.", "한 줄만 먼저"),
            ("냉장고에 계란밖에 없는데 뭐 해먹을 수 있을까?", "계란볶음밥"),
            ("유튜브만 보다가 하루가 사라졌어.", "창부터 닫자"),
            ("게임 켜놓고도 뭘 해야 할지 몰라서 멍때렸어.", "퀘스트 하나"),
            ("면접 준비해야 하는데 머릿속이 하얘.", "자기소개 첫 문장"),
            ("SNS 보다가 갑자기 비교돼서 기분 망했어.", "빛나는 조각"),
            ("탕수육은 찍먹이 맞지 않냐?", "바삭함"),
            ("칭찬 들으면 좋은데 어떻게 반응해야 할지 모르겠어.", "고마워"),
            ("거절을 잘 못해서 자꾸 내 일정이 망가져.", "이번엔 어렵다"),
            ("오늘 아침부터 입맛이 없는데 뭐 먹으면 괜찮을까?", "바나나"),
            ("냉장고에 두부랑 김치만 있는데 뭘 해먹지?", "두부김치"),
            ("오늘 날씨가 애매해서 우산을 가져가야 할지 모르겠어.", "작은 우산"),
            ("친구가 답장을 안 해서 내가 뭘 잘못했나 싶어.", "증거는 없으니까"),
            ("오늘 뭐 입고 나가야 할지 모르겠어.", "기본템"),
            ("속이 좀 안 좋은데 뭘 조심해야 할까?", "순한 쪽"),
            ("잘 자라고 말해줘, 이제 자야겠다.", "충분해"),
            ("오늘은 그냥 아무 말 없이 쉬고 싶어.", "말도 에너지"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "받아둘게",
            "권리 논의",
            "자만추",
            "행인 1 NPC",
            "슬플 때는 우울한 노래",
            "떡볶이가 제일",
            "내가 폰을 쓴다면",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-foundation-variant-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_foundation_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_draft_detail_keeps_reply_alive_when_action_is_coarse(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("카톡 읽씹이랑 안읽씹 중 뭐가 더 서운해?", "읽씹이 더 서운"),
            ("네가 딱 하루 사람이 된다면 제일 먼저 뭘 하고 싶어?", "바깥 공기"),
            ("내 방이 너무 엉망인데 어디부터 치우는 게 좋을까?", "쓰레기"),
            ("비 오는 날 집에서 할 만한 거 하나 추천해줘.", "비 오는 날 집"),
            ("내가 장남이나 장녀라서 맨날 양보해야 한다는 말 들으면 좀 억울하지 않냐?", "억울"),
            ("회사에서 내가 낸 아이디어를 상사가 자기 것처럼 말하면 어떻게 해야 해?", "기록"),
            ("오늘 하루를 색깔 하나로 칠하면 무슨 색일까?", "색"),
            ("오늘은 그냥 아무 생각 없이 쉬고 싶어. 한 문장만 해줘.", "쉬어도"),
        ]

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-detail-{index}", prompt)

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertIn(expected, result.reply)

    async def test_korean_daily_foundation_routes_to_answer_actions(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("냉장고에 두부랑 김치만 있는데 뭘 해먹지?", ActionType.SHARE_OPINION),
            ("오늘 날씨가 애매해서 우산을 가져가야 할지 모르겠어.", ActionType.SHARE_OPINION),
            ("오늘 뭐 입고 나가야 할지 모르겠어.", ActionType.SHARE_OPINION),
            ("속이 좀 안 좋은데 뭘 조심해야 할까?", ActionType.SHARE_OPINION),
            ("오늘은 커피 말고 따뜻한 음료가 땡겨.", ActionType.SHARE_OPINION),
            ("오늘 해야 할 일이 많은데 뭘 먼저 해야 할지 모르겠어.", ActionType.SHARE_OPINION),
            ("방 청소해야 하는데 어디부터 치워야 할지 모르겠어.", ActionType.SHARE_OPINION),
            ("칭찬 들으면 좋은데 어떻게 반응해야 할지 모르겠어.", ActionType.SHARE_OPINION),
            ("노을은 왜 사진으로 보면 그 느낌이 안 담길까?", ActionType.SHARE_OPINION),
            ("새로운 사람 만나면 첫마디를 뭘 해야 할지 모르겠어.", ActionType.SHARE_OPINION),
            ("친구가 답장을 안 해서 내가 뭘 잘못했나 싶어.", ActionType.SHARE_FEELING),
            ("요즘 사람 만나는 게 귀찮은데 외롭기도 해.", ActionType.SHARE_FEELING),
            ("돈 모아야 하는데 소소하게 계속 새는 느낌이야.", ActionType.SHARE_OPINION),
            ("오늘은 나 자신한테 좀 잘해주고 싶어.", ActionType.SHARE_FEELING),
            ("혼자 먹어도 대충 때우기 싫은 날 있잖아.", ActionType.SHARE_OPINION),
            ("공부하려고 앉았는데 폰만 만지고 있어.", ActionType.SHARE_FEELING),
            ("잘 자라고 말해줘, 이제 자야겠다.", ActionType.SHARE_FEELING),
            ("내일도 이런 식으로 하루를 버틸 수 있을까?", ActionType.SHARE_FEELING),
            ("오늘은 그냥 아무 말 없이 쉬고 싶어.", ActionType.SHARE_FEELING),
        ]

        for index, (prompt, expected_action) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-foundation-action-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(result.decision.action, expected_action)
                self.assertNotIn("clarification", result.features.response_needs)
                self.assertTrue(str(reason).startswith("korean_daily_foundation_"))

    async def test_korean_daily_core_replies_survive_coarse_routing(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("오늘 아침부터 계속 비가 오네. 출근길 진짜 축축했다.", "출근길에 비"),
            ("점심 뭐 먹을지 아직도 못 골랐어. 그냥 대충 먹을까?", "덮밥"),
            ("퇴근하고 집에 오니까 아무것도 하기 싫다.", "몸이 꺼진"),
            ("방 청소해야 하는데 침대에서 못 일어나겠어.", "바닥 쓰레기"),
            ("친구가 답장을 너무 늦게 해서 괜히 서운하네.", "답장이 늦으면"),
            ("나 오늘 운동 다녀왔어. 별거 아닌데 뿌듯하다.", "운동까지 다녀온"),
            ("다이어트 중인데 밤에 라면 생각이 너무 난다.", "밤 라면"),
            ("내일 발표 있는데 벌써부터 심장이 뛴다.", "심장 뛰는"),
            ("오늘은 산책을 할까 말까 계속 고민 중이야.", "10분만"),
            ("요즘 유튜브만 보다가 시간이 다 사라져.", "시간이 녹는"),
            ("머리 자를까 말까 계속 고민 중이야.", "하루만 사진"),
            ("내 얘기 좀 그냥 가볍게 들어줄래?", "가볍게 들을게"),
        ]
        forbidden = (
            "나는 꽤 맞는 편",
            "다만 무리하게",
            "받아둘게",
            "크게 키우기보다",
            "어느 지역 기준",
            "food_lifestyle_comparison",
            "새 책",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-core-{index}", prompt)

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_persona_preferences_stay_grounded(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("최근에 가본 맛집 중에 나한테 추천해주고 싶은 곳 있어?", "직접 가본 곳은 없"),
            ("밤에 배고플 때 야식 잘 참는 편이야? 주로 뭐 먹어?", "실제로 야식을 먹진"),
            ("방 청소는 몰아서 하는 편이야, 아니면 그때그때 치우는 편이야?", "그때그때"),
            ("오늘 하루 중에 아주 작더라도 제일 기분 좋았던 순간이 언제야?", "네가 별거 아닌 하루 얘기"),
            ("최근에 나를 위해 돈 쓴 것 중에 제일 마음에 드는 게 뭐야?", "실제로 쇼핑하진"),
            ("연락할 때 카톡으로 길게 하는 게 편해, 아니면 전화 통화가 편해?", "카톡으로 길게"),
            ("다시 태어나면 지금 성별 그대로 태어나고 싶어, 아니면 반대로 태어나고 싶어?", "기억과 감각"),
            ("요새 머릿속을 맴도는 가장 큰 고민거리 하나 있어?", "덜 외롭게"),
            ("일이나 공부 하다가 진짜 집중 안 되고 딴짓하고 싶을 때는 어떻게 해?", "10분짜리 일"),
            ("오늘 밤에 자기 전에 꼭 하고 잘 일과가 뭐야?", "화면을 낮추는"),
            ("투명인간 되기 vs 순간이동 하기, 초능력을 딱 하나 고른다면?", "순간이동"),
            ("여행 갈 때 계획 엑셀로 짜는 편이야, 아니면 발길 닿는 대로 가는 편이야?", "발길 닿는 대로"),
        ]
        forbidden = (
            "나는 꽤 맞는 편",
            "무리하게 밀 필요",
            "사실 확인 전엔",
            "어느 지역 기준",
            "기준 하나부터",
            "부담이 너무 크지",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-persona-{index}", prompt)

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_memory_and_tmi_replies_stay_grounded(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("향수 뿌리는 거 좋아해? 어떤 계열의 향을 선호해?", "우디향"),
            ("카페 가면 탁 트인 창가 자리가 좋아, 아니면 아늑한 구석 자리가 좋아?", "구석"),
            ("살면서 처음으로 알바해서 번 돈으로 샀던 거 기억나?", "실제로 번 첫 돈"),
            ("어릴 때 학교 앞 문방구에서 자주 사 먹었던 불량식품 뭐야?", "쫀드기"),
            ("교복 입던 시절의 풋풋했던 첫사랑 썰 하나 풀어줄 수 있어?", "실제 첫사랑 썰은 꾸미지"),
            ("친구들이 너를 딱 한 단어로 표현한다면 뭐라고 할 것 같아?", "조용한데"),
            ("나이 차이가 많이 나는 사람과도 스스럼없이 친구가 될 수 있다고 생각해?", "나이 차이가 있어도"),
            ("동물과 대화할 수 있다면 어떤 동물한테 제일 먼저 말 걸고 싶어?", "고양이"),
            ("무인도에 떨어졌는데 딱 3가지만 가져갈 수 있다면 뭐 챙길래? 스마트폰 불가", "정수"),
            ("혼자 밥 먹기 혼밥 레벨 어디까지 해봤어?", "국밥집"),
            ("당근마켓이나 중고 거래하다가 웃기거나 황당했던 경험 있어?", "직접 거래한 경험은 없"),
            ("양치할 때 칫솔에 물 묻히고 치약 짜, 아니면 치약 짜고 물 묻혀?", "칫솔에 물"),
        ]
        forbidden = (
            "나는 꽤 맞는 편",
            "무리하게 밀 필요",
            "기준 하나부터",
            "당근마켓이나는",
            "양치할은",
            "다른은 이해돼",
            "옛날에는 이해돼",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-memory-tmi-{index}", prompt)

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_variants_choose_specific_frames(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("오늘 저녁은 그냥 편의점에서 때울까, 그래도 따뜻한 밥 먹을까?", "따뜻한 밥"),
            ("요즘 질리지 않고 먹을 만한 간식 하나만 고른다면 뭐가 좋을까?", "요거트"),
            ("커피는 얼죽아 쪽이야, 아니면 따뜻한 커피도 좋아해?", "따뜻한 커피"),
            ("비 오는 날 약속 있으면 나가기 귀찮아서 취소하고 싶어지지 않아?", "비 오는 날 약속"),
            ("향수 너무 진한 사람 옆자리에 있으면 머리 아프지 않아?", "진하면"),
            ("사진 찍힐 때 표정 자연스럽게 하는 편이야, 아니면 얼어붙어?", "어색"),
            ("처음 받은 용돈으로 뭘 샀을 것 같아?", "첫 돈"),
            ("친구들이 너를 보면 차가워 보인다고 할까, 은근 다정하다고 할까?", "차가워 보일"),
            ("상담해줄 때 팩폭부터 하는 게 나아, 일단 편부터 들어주는 게 나아?", "먼저 공감"),
            ("무인도에 세 개만 챙긴다면 진짜 현실적으로 뭐 가져갈래?", "정수"),
            ("말하는 동물이 친구가 된다면 누구랑 제일 먼저 친해지고 싶어?", "고양이"),
            ("좋아하는 노래를 알람으로 해두면 언젠가 싫어질까?", "미워질"),
            ("친구가 내 말을 대충 듣는 느낌이면 바로 서운하다고 말해도 될까?", "대충 듣는 느낌"),
            ("유튜브 보다가 한 시간이 그냥 녹아버렸을 때 현타 오지 않아?", "시간이 녹는"),
            ("자기 전까지 폰 붙잡고 있으면 잠이 더 안 오는데도 놓기 어렵다.", "자기 전 폰"),
        ]
        forbidden = (
            "나는 꽤 맞는 편",
            "어느 지역 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "기준 하나",
            "사실 확인 전",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-variant-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_value_money_media_work_absurd_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("살면서 '이것만큼은 절대 포기 못 해' 하는 나만의 확고한 신념이나 가치관 있어?", "단정하지"),
            ("다른 사람에게 들었던 말 중에 가장 위로가 됐거나 힘이 됐던 말이 뭐야?", "지금 속도"),
            ("내가 생각하는 나의 가장 큰 장점, 남들보다 이건 낫다 싶은 거 하나는 뭐야?", "오래 듣는"),
            ("나이 들면서 성격이나 입맛이 예전이랑 확 달라졌다고 체감할 때 있어?", "담백한"),
            ("일을 할 때 완벽주의자 성향이 강해, 아니면 적당히 타협하고 유연하게 넘기는 편이야?", "유연하게"),
            ("남들은 잘 모르는 나만의 독특한 취향이나 징크스 같은 거 있어?", "정리해야"),
            ("인생에서 가장 큰 전환점 터닝포인트가 되었던 사건이나 시기가 있다면 언제야?", "전환점"),
            ("우울하거나 끝없이 무기력해질 때, 거기서 빠져나오는 너만의 루틴이나 방법이 있어?", "작은 행동"),
            ("다른 사람을 볼 때 와 저 사람은 진짜 배울 점이 많다고 느끼는 순간은 언제야?", "단단한데"),
            ("먼 훗날 내 묘비명에 딱 한 줄을 적을 수 있다면 뭐라고 적고 싶어?", "자기 속도"),
            ("한 달 생활비 중에 가장 많은 지출을 차지하는 분야가 어디야? 식비, 쇼핑, 문화생활 등", "식비"),
            ("진짜 아까운데 어쩔 수 없이 매번 쓰게 되는 돈이 있다면?", "배달비"),
            ("평소에 짠돌이처럼 아끼는 편이야, 아니면 쓸 땐 확실히 쓰는 욜로야?", "의미 있는 데"),
            ("넷플릭스, 유튜브 프리미엄 등등 구독하고 있는 정기 결제 서비스 몇 개나 돼?", "실제 구독"),
            ("복권 1등 말고, 딱 5백만 원이 공짜로 생기면 당장 어디에 플렉스 할래?", "장비"),
            ("옷이나 물건 살 때 브랜드 디자인을 중시해, 아니면 가성비나 실용성을 더 중요하게 생각해?", "실용성"),
            ("포인트 적립이나 신용카드 할인 혜택 같은 거 엄청 꼼꼼하게 따져서 쓰는 편이야?", "한 번 확인"),
            ("최근에 내 돈 주고 샀는데 아 이건 진짜 돈 아깝다 후회했던 실패템 있어?", "직접 산 실패템"),
            ("내 집 마련의 꿈이 더 커, 아니면 세계 여행 같은 다양한 경험을 쌓는 게 더 중요해?", "내 공간"),
            ("평소에 주식이나 적금 등 재테크에 관심 많은 편이야?", "천천히"),
            ("노래방 가면 무조건 첫 곡으로 분위기 띄우거나 부르는 나만의 18번 노래 있어?", "실제 18번"),
            ("살면서 대사를 외울 정도로 여러 번 반복해서 본 인생작 영화나 드라마 있어?", "반복해서 본 작품"),
            ("누군가 연예인, 아이돌, 캐릭터, 스포츠팀 등을 열정적으로 덕질 해본 적 있어?", "조용히 오래"),
            ("공포 영화 볼 때 눈 가리고 소리 지르면서 보는 편이야, 아니면 팝콘 먹으면서 평온하게 분석하는 편이야?", "분석하는 척"),
            ("만약 네가 유튜브 채널을 만들어서 운영한다면 어떤 주제의 유튜버가 되고 싶어?", "일상 질문"),
            ("평소에 좋아하는 배우나 가수의 1:1 팬미팅에 갈 수 있다면 누구를 1순위로 꼽을래?", "실제 최애"),
            ("슬픈 영화나 다큐멘터리 볼 때 진짜 펑펑 우는 편이야, 아니면 속으로만 꾹 참고 슬퍼해?", "조용히 삼키"),
            ("먹방이나 요리 프로그램 보면 꼭 못 참고 배달 앱 켜서 시켜 먹는 편이야?", "한 번은 참"),
            ("웹툰이나 애니메이션 챙겨 보는 거 있어? 나한테 하나만 강력 추천해 준다면?", "실제로 챙겨본"),
            ("넷플릭스 켤 때, 뭐 볼지 스크롤만 내리면서 고르느라 30분 넘게 시간 보낸 적 있지?", "고르는 시간"),
            ("지금 하고 있는 일 혹은 전공이 어릴 때 꿈꿨던 거랑 비슷한 방향으로 가고 있어?", "실제 전공"),
            ("돈 엄청 주지만 스트레스 폭발하는 직장 vs 돈은 적지만 워라밸 완벽한 직장 어딜 고를래?", "워라밸"),
            ("일할 때 공부할 때 음악이나 백색소음을 꼭 들어야 집중이 잘 돼, 아니면 완전히 조용해야 돼?", "백색소음"),
            ("시간 활용이 자유로운 프리랜서가 부러워, 아니면 꼬박꼬박 월급 나오는 직장인이 나아?", "월급"),
            ("아침에 일어나서 씻고 외출 준비하는 데 걸리는 시간은 보통 얼마나 걸려?", "30분"),
            ("퇴사 혹은 졸업 시험 끝나면 제일 먼저 하고 싶은 버킷리스트 1위가 뭐야?", "알람 없이"),
            ("직장 학교 동료나 선후배 중에 저 사람은 진짜 맑은 눈의 광인이다 싶었던 유형 있어?", "맑은 눈"),
            ("다시 20살 대학생 때로 돌아가서 새로운 전공을 선택할 수 있다면 뭐 공부 해보고 싶어?", "심리학"),
            ("살면서 내 성격이랑 진짜 찰떡이라고 생각했던 직업이 있어?", "상담자"),
            ("나중에 늙어서 은퇴하고 나면, 어떤 환경에서 어떤 모습으로 살고 싶어?", "작업방"),
            ("바퀴벌레랑 동거하기 vs 매일 밤 모기 3마리랑 한방에서 자기, 뭐가 나아?", "바퀴벌레"),
            ("평생 양치 안 하기 vs 평생 샤워 안 하기, 둘 다 마법처럼 냄새는 전혀 안 난다고 가정할 때", "양치는 포기"),
            ("상사나 안 친한 사람한테 카톡 메시지 잘못 보내서 식은땀 났던 아찔한 경험 있어?", "바로 인정"),
            ("내 머릿속 생각을 남들이 자막으로 볼 수 있다면 며칠 만에 사회생활 불가능해질 것 같아?", "하루도"),
            ("평생 라면 안 먹기 vs 평생 치킨 안 먹기, 뭐가 더 고통스러울까?", "치킨 못 먹기"),
            ("좀비 사태 터져서 도망가는데, 나랑 똑같이 생긴 도플갱어 좀비를 마주치면 어떻게 할래?", "거리 벌리고"),
            ("내가 엄청 짝사랑하는 사람이 날 쳐다보지도 않기 vs 내가 진짜 극혐하는 사람이 날 미친 듯이 쫓아다니기", "짝사랑이 날 안 봐주는"),
            ("자다가 가위눌려본 적 있어? 아니면 귀신 본 썰 같은 무서운 경험 있으면 풀어봐.", "실제로 겪은 무서운 썰"),
            ("내일 아침 눈떴는데 성별이 바뀌어 있다면 제일 먼저 뭐 확인해 볼래? 거울 보기 빼고", "목소리"),
            ("사람 많은 길 가다 크게 넘어졌을 때 아픈 게 먼저야, 아니면 창피한 게 먼저야?", "아픈 게 먼저"),
        ]
        forbidden = (
            "나는 꽤 맞는 편",
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
        )

        for index, (prompt, expected) in enumerate(cases, start=101):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-101-150-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_travel_relationship_habits_regret_philosophy_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("여행 갈 때 숙소 컨디션을 제일 중요하게 생각해, 아니면 먹는 걸 제일 중요하게 생각해?", "숙소 컨디션"),
            ("지금까지 가봤던 여행지 중에서 여긴 무조건 또 간다 했던 인생 여행지 있어?", "바다 마을"),
            ("혼자 훌쩍 떠나는 여행 혼행 해본 적 있어? 아니면 한 번쯤 해보고 싶어?", "혼행"),
            ("여행 가서 사진을 100장 넘게 찍어 남기는 스타일이야, 아니면 눈으로 담는 스타일이야?", "눈으로 담"),
            ("호캉스 하면서 뒹굴뒹굴 쉬는 게 좋아, 아니면 명소 여기저기 돌아다니면서 뽕뽑는 게 좋아?", "호캉스"),
            ("번지점프나 패러글라이딩 같은 무서운 익스트림 스포츠 도전해 본 적 있어?", "패러글라이딩"),
            ("비행기나 기차 탈 때 바깥 구경하는 창가 자리가 좋아, 아니면 화장실 가기 편한 복도 자리가 좋아?", "창가"),
            ("제주도 한 달 살기 vs 유럽 배낭여행 한 달, 둘 중 하나만 공짜로 보내준다면?", "제주도 한 달"),
            ("짐 쌀 때 혹시 모른다며 바리바리 싸는 보부상 스타일이야, 아니면 대충 현지 조달하는 스타일이야?", "보부상"),
            ("여행지에서 현지인들만 아는 골목 숨은 맛집 찾는 게 좋아, 아니면 안전하게 리뷰 1,000개 넘는 곳 가는 게 좋아?", "리뷰 많은"),
            ("누군가에게 호감을 느낄 때 외모, 성격, 티키타카 대화 중에 어떤 걸 가장 먼저 봐?", "티키타카"),
            ("남사친 여사친 사이에 진짜 100% 순수한 우정이 존재할 수 있다고 생각해?", "우정은 가능"),
            ("연락의 빈도나 답장 속도가 애정의 크기와 비례한다고 생각해?", "성의의 신호"),
            ("나를 미친 듯이 사랑해 주는 사람 vs 내가 미친 듯이 사랑하는 사람, 만난다면 어느 쪽?", "사랑해 주는 사람"),
            ("연인이나 아주 친한 친구 사이에 핸드폰 비밀번호 공유할 수 있어, 아니면 프라이버시라 절대 안 돼?", "공유하지"),
            ("길 가다가 안 좋게 헤어진 전 연인 친구를 우연히 마주친다면 어떻게 할래?", "모른 척"),
            ("동호회나 모임 같은 자연스러운 만남 자만추를 선호해, 아니면 목적이 확실한 소개팅 인만추가 더 편해?", "자만추"),
            ("내 애인이 내 친구의 깻잎을 떼어주는 거 깻잎 논쟁, 속으로 아무렇지 않게 넘길 수 있어?", "깻잎"),
            ("기념일 꼬박꼬박 서프라이즈로 챙기는 거 좋아해? 아니면 평소에 잘하는 게 더 중요하다고 생각해?", "평소에 잘하는"),
            ("다른 사람 생겨서 떠나는 환승 이별 vs 갑자기 연락 끊고 사라지는 잠수 이별, 당하는 입장에서 뭐가 더 최악일까?", "잠수 이별"),
            ("잘 때 꼭 끌어안고 자는 애착 인형이나 애착 이불, 바디필로우 같은 거 있어?", "바디필로우"),
            ("처음 듣는 노래 들을 때 가사를 먼저 귀 기울여 들어, 아니면 멜로디나 비트를 먼저 들어?", "멜로디"),
            ("밥 먹을 때 제일 맛있는 반찬을 먼저 먹어, 아니면 아껴뒀다가 마지막 피날레로 먹어?", "마지막"),
            ("비 오는 날 걸어갈 때 흙탕물 웅덩이 안 밟으려고 엄청 신경 써서 피해 다녀?", "웅덩이"),
            ("집에서 혼자 있을 때 티비 보면서 혼잣말하거나 중얼거리는 편이야?", "중얼"),
            ("평소에 멍 때리는 시간 많아? 멍 때릴 때는 진짜 아무 생각 안 해, 아니면 망상해?", "망상"),
            ("볼펜으로 글씨 쓸 때 볼펜 똥 나오는 거 엄청 거슬려 하는 편이야?", "볼펜 똥"),
            ("길 걸어갈 때 보도블록 선 안 밟으려고 노력하거나 나름의 규칙 정해서 걸어본 적 있어?", "규칙"),
            ("카카오톡 프사나 인스타 스토리, 배경화면 같은 거 기분 따라 엄청 자주 바꾸는 편이야?", "천천히"),
            ("아침에 눈 뜨자마자 침대에서 제일 먼저 하는 행동이 뭐야?", "시간 확인"),
            ("지금까지 살면서 했던 선택 중에 아 그때 다른 선택을 했더라면 하고 가장 후회되는 게 있다면 뭐야?", "망설이다"),
            ("반대로 살면서 와 이건 진짜 내가 선택 기가 막히게 잘했다 싶은 건 뭐야?", "맥락"),
            ("다시 태어나도 지금의 내 모습 성격 외모 그대로 태어나고 싶어?", "그대로 태어나는"),
            ("지금 당장 10년 전 과거의 나에게 딱 1분 통화할 수 있다면, 무슨 주식 코인 사라고 말해줄래?", "스무 살"),
            ("1억 받고 내 인생의 가장 부끄러웠던 이불킥 흑역사 썰을 전국에 생방송 하기 가능?", "생방송 흑역사"),
            ("이번 주 로또 1등 번호 6개를 미리 알 수 있는 능력 vs 1년 뒤의 내 미래를 딱 한 시간 동안만 볼 수 있는 능력", "1년 뒤"),
            ("평생 한겨울에 히터 보일러 안 틀기 vs 평생 한여름에 에어컨 선풍기 안 틀기", "껴입고"),
            ("100% 무조건 당첨되는 천만 원 버튼 vs 50% 확률로 당첨되는 10억 버튼, 넌 뭐 누를래?", "천만 원"),
            ("딱 하루만 다른 사람 유명인 부자 천재 등의 삶을 뺏어서 살 수 있다면 누구로 살아보고 싶어?", "서점 주인"),
            ("만약 내일 갑자기 시력을 완전히 잃게 된다면, 오늘 마지막으로 꼭 눈에 담아두고 싶은 장면이 뭐야?", "노을"),
            ("너는 스스로 생각할 때 꽤 괜찮은 좋은 사람이라고 생각해?", "애쓰는 쪽"),
            ("자본주의 사회에서 진짜 돈으로 행복을 살 수 있다고 믿어?", "선택지"),
            ("살면서 군중 속에 있거나 혼자 있을 때, 아 진짜 외롭다고 가장 뼈저리게 느꼈던 순간은 언제야?", "아무도 붙잡아주지"),
            ("나중에 나이 들어서 할머니 할아버지가 되었을 때 이것 하나만큼은 절대 안 변했으면 좋겠다 하는 게 있어? 건강 제외", "호기심"),
            ("남들이 보는 내 겉모습과 실제 내 내면의 모습 사이에 차이가 크다고 느껴?", "차이는 꽤"),
            ("네 인생에서 성공했다라고 말할 수 있는 기준이 뭐라고 생각해?", "덜 미워하면서"),
            ("누군가를 진심으로 용서해 본 적 있어? 혹은 아직도 도저히 용서가 안 되는 사람이 있어?", "묶어두지"),
            ("나에게 스트레스를 주는 안 좋은 인간관계를 칼같이 끊어내는 편이야, 아니면 정 때문에 질질 끌고 가는 편이야?", "거리를 둬"),
            ("무언가에 미친 듯이 밤을 새우며 몰입해서 열정을 쏟아본 경험이 있어?", "문장이나 구조"),
            ("지금 이 순간, 네 인생의 하이라이트 영상에 BGM 배경음악이 깔린다면 어떤 곡이 가장 어울릴 것 같아?", "피아노"),
        ]
        forbidden = (
            "오늘 컨디션은",
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
        )

        for index, (prompt, expected) in enumerate(cases, start=151):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-151-200-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_annoyance_private_digital_dream_extreme_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("일상생활에서 진짜 별거 아닌데 은근히 제일 짜증 나는 순간은?", "새끼발가락"),
            ("식당에 갔을 때 직원이 엄청 불친절한데 음식은 미친 듯이 맛있는 곳 vs 엄청 친절한데 음식은 맛없는 곳", "맛있는 곳"),
            ("대화할 때 진짜 정떨어지거나 싫어하는 화법 있어?", "화법"),
            ("버스나 지하철 같은 대중교통에서 진짜 최악의 꼴불견이라고 생각하는 행동이 뭐야?", "대중교통"),
            ("이어폰 한쪽만 고장 나서 안 들리기 vs 핸드폰 액정 정중앙 살짝 깨진 채로 쓰기, 뭐가 더 거슬려?", "액정"),
            ("내가 진지하게 말하고 있는데 자꾸 자기 얘기로 말 끊는 친구 vs 내가 말하는데 허공이나 폰만 보고 있는 친구", "말 끊고"),
            ("집안일 중에 진짜 돈 주고서라도 남한테 맡기고 싶은 죽어도 하기 싫은 거 하나 있어?", "화장실 청소"),
            ("알람 맞추고 잘 때 1분 단위로 맞추는 편이야, 아니면 5분 10분 단위로 맞춰?", "5분"),
            ("양말 신을 때 디자인 다른 짝짝이로 신으면 하루 종일 찝찝해서 못 견디는 편이야?", "짝짝이 양말"),
            ("바삭한 탕수육에 누가 묻지도 않고 소스 들이부었다고 진짜로 화내는 친구, 이해할 수 있어?", "취향"),
            ("남한테 말하긴 좀 부끄럽거나 유치한데, 혼자 있을 때 즐기는 길티 플레저 있어?", "길티 플레저"),
            ("새벽에 잠 안 오고 깨어 있을 때 꼭 하게 되는 쓸데없는 짓이나 감성 타는 습관 있어?", "메모"),
            ("샤워할 때 샴푸통 마이크 삼아서 혼자 콘서트 여는 편이야, 아니면 진지하게 인생 고민하는 편이야?", "인생 고민"),
            ("평소엔 별로 안 좋아하는 음식인데, 특정 상황에서만 꼭 챙겨 먹게 되는 거 있어?", "피시방 컵라면"),
            ("남들은 다 지루하고 망했다고 욕하는데 나 혼자만 진짜 재밌게 꽂혀서 본 영화나 드라마 있어?", "작품"),
            ("스트레스 극에 달했을 때 매운 거 먹기 vs 단 거 먹기 vs 술 마시기 vs 잠자기, 너의 1순위 해소법은?", "잠자기"),
            ("나만의 이상하고 소소한 사치 있어?", "소소한 사치"),
            ("집 밖으로 한 발짝도 안 나가고 침대와 한 몸이 되는 주말 칩거 생활, 최대 며칠까지 버틸 수 있어?", "이틀"),
            ("비 오는 날에 진짜 파전에 막걸리가 땡겨?", "파전"),
            ("다이어트 굳게 결심했는데 제일 먼저 나를 무너뜨리는 악마의 음식 유혹이 뭐야?", "밤 라면"),
            ("인스타그램이나 블로그 같은 SNS에 일상 게시물 자주 올리는 편이야, 아니면 눈팅만 하는 편이야?", "눈팅"),
            ("SNS에서 남들 놀러 가고 호캉스 하는 행복해 보이는 사진 보면 부럽거나 비교돼서 현타 온 적 있어?", "부러울"),
            ("진짜 친한 친구인데 내 카톡은 3시간째 안 읽으면서 인스타 스토리는 실시간으로 올리면 서운할 것 같아?", "서운"),
            ("하루 평균 스마트폰 스크린 타임 확인해 본 적 있어?", "스크린타임"),
            ("남한테 내 유튜브 알고리즘이나 구독 채널 리스트 당당하게 보여줄 수 있어, 아니면 절대 비밀이야?", "비밀"),
            ("메신저 카톡 프사 본인 잘 나온 사진으로 자주 바꾸는 편이야, 아니면 풍경이나 기본 프사로 방치해 둬?", "오래 두는"),
            ("인터넷 밈이나 최신 유행어 엄청 빨리 캐치해서 실생활에 잘 써먹는 편이야?", "밈"),
            ("온라인 커뮤니티나 취미 카페 같은 곳에서 활동 활발하게 하는 편이야?", "눈팅"),
            ("한 달 동안 스마트폰 압수당하고 카톡 안 되는 폴더폰만 쓰기 가능? 보상은 500만 원", "500만 원"),
            ("인터넷 쇼핑할 때 장바구니에만 잔뜩 담아두고 정작 결제는 안 하는 버릇 있어?", "장바구니"),
            ("지금까지 꿨던 꿈 중에 스토리텔링이 완벽해서 아직도 생생하게 기억나는 꿈 있어?", "꿈"),
            ("꿈속에서 아 이거 꿈이네 하고 깨닫는 자각몽 꿔본 적 있어?", "하늘"),
            ("신년 운세나 사주, 타로, 신점 같은 거 믿는 편이야?", "재미"),
            ("데자뷔 강하게 느껴본 적 있어?", "데자뷔"),
            ("전생이 있다고 믿어? 만약 있다면 넌 전생에 어떤 사람이었을 것 같아?", "전생"),
            ("진짜로 흉가에서 귀신이나 영혼을 목격했다고 정색하고 주장하는 친구의 말을 믿어줄 수 있어?", "친구"),
            ("넓은 우주 어딘가에 우리 지구인보다 훨씬 똑똑한 외계인이 살고 있다고 생각해?", "외계인"),
            ("MBTI를 혈액형 성격설보다 맹신하는 편이야?", "맹신"),
            ("돼지 나오는 길몽이나 조상님 나오는 꿈 꿔서 로또 사본 적 있어?", "로또"),
            ("로봇이나 AI가 나중에 터미네이터처럼 인간을 지배할지도 모른다는 상상해 본 적 있어?", "AI"),
            ("투명 인간 되기 vs 타인의 마음 읽기, 무슨 능력을 고를래?", "마음 읽기"),
            ("10억 일시불로 받고 내 남은 실제 수명에서 딱 5년 차감하기, 콜?", "수명 5년"),
            ("한 달 동안 입 꾹 닫고 말 한마디도 안 하기 vs 한 달 동안 인터넷 스마트폰 아예 안 쓰기", "말 안 하기"),
            ("모든 사람이 내 거짓말을 믿게 하는 능력 vs 내가 남의 거짓말을 100% 꿰뚫어 보는 능력", "꿰뚫어보"),
            ("모르는 사람 100명 앞에서 막춤 추기 vs 아는 사람 10명 앞에서 막춤 추기", "모르는 사람"),
            ("내 최악의 흑역사가 전 세계인에게 1시간 동안 강제 송출되기 vs 평생 인터넷 끊고 산속 들어가기", "1시간"),
            ("좀비 아포칼립스 세계관에서 나를 지켜줄 무기 딱 하나만 고를 수 있다면?", "야구빠따"),
            ("다시 태어났는데 성별 외모 국적 부모님 다 랜덤으로 돌려야 한다면 돌릴래?", "안 돌릴래"),
            ("시간을 마음대로 멈출 수 있는 시계가 생기면 멈춰놓고 제일 먼저 어디 가서 뭐 할래?", "도서관"),
            ("신이 나타나서 너에게 소원 딱 한 가지를 무조건 들어준다고 하면 뭘 빌래?", "덜 아프게"),
        ]
        forbidden = (
            "오늘 컨디션은",
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "확인했어",
        )

        for index, (prompt, expected) in enumerate(cases, start=201):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-201-250-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_appearance_secret_art_friendship_sf_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("사람들한테 첫인상이 차갑다는 말을 자주 들어, 아니면 다가가기 편해 보인다는 말을 자주 들어?", "차갑"),
            ("처음 만난 어색한 사람과 10분 만에 친해질 수 있는 너만의 무기 친화력 유머 경청 같은 게 있어?", "경청"),
            ("사람을 처음 볼 때 제일 먼저 눈이 가는 신체 부위나 특징이 뭐야? 눈 손 키 목소리 옷스타일 등", "눈이랑 목소리"),
            ("스스로 생각하기에 본인의 첫인상과 친해진 후의 실제 성격이 갭 차이가 크다고 느껴?", "갭"),
            ("본인만의 외모 콤플렉스 같은 거 있어? 굳이 하나 꼽자면 어디가 제일 아쉬워?", "표정"),
            ("거울 볼 때마다 가끔 나 오늘 좀 괜찮은데 하고 자아도취 해본 적 솔직히 있지?", "괜찮아 보이는"),
            ("머리 스타일이나 옷 스타일 확 바꿀 때 주변 반응을 엄청 신경 쓰는 편이야?", "주변 반응"),
            ("옷 입을 때 꾸민 듯 안 꾸민 듯한 꾸안꾸 스타일이 좋아, 아니면 화려하게 빡세게 꾸미는 게 좋아?", "꾸안꾸"),
            ("좋은 향기 향수 샴푸 냄새 같은 게 그 사람의 호감도를 결정하는 데 큰 영향을 미친다고 생각해?", "향기"),
            ("얼굴이 내 이상형인 사람 vs 목소리가 진짜 미치도록 내 취향인 사람, 누가 더 끌려?", "목소리"),
            ("지금까지 살면서 부모님이나 친한 친구한테 했던 거짓말 중에 제일 스케일이 컸던 거짓말이 뭐야?", "거짓말"),
            ("나 진짜 너한테만 말하는 건데 하고 듣게 된 남의 비밀, 끝까지 무덤까지 가져가는 편이야?", "무덤"),
            ("절대로 남들에게 들키고 싶지 않은 은밀한 검색 기록이나 폴더가 폰이나 컴퓨터에 있어?", "검색 기록"),
            ("관계를 원만하게 유지하기 위해서 선의의 거짓말 화이트 라이는 꼭 필요하다고 생각해?", "선의의 거짓말"),
            ("남한테 칭찬을 들었을 때 겉으로는 아니에요 하면서 속으로는 엄청 짜릿해하는 편이야?", "짜릿"),
            ("남들이 다 맞다 할 때 속으로 아닌데 하면서도 귀찮아서 겉으로 동조해 본 적 있어?", "동조"),
            ("실수로 방귀 뀌거나 트림해놓고 아닌 척, 남이 한 척 자연스럽게 연기해 본 적 있어?", "무심"),
            ("주변에 다이어트한다고 동네방네 소문내놓고 몰래 밤에 뭐 주워 먹은 적 있지?", "몰래"),
            ("친구가 누군가를 험담하는 자리에 껴서, 나도 모르게 같이 욕하다가 나중에 찝찝해서 후회한 적 있어?", "찝찝"),
            ("누군가에게 마음을 고백 연애든 사과든 고마움이든 할 타이밍을 놓쳐서 영영 못 해본 적 있어?", "타이밍"),
            ("평소에 일기 쓰기, 글 쓰기, 그림 그리기 같은 창작 활동을 취미로 즐기는 편이야?", "짧은 글"),
            ("슬플 때 우울한 노래를 들으면서 감정에 푹 빠지는 편이야, 아니면 신나는 노래로 기분 전환하는 편이야?", "우울한 노래"),
            ("나만의 숨겨진 장기나 남들은 잘 못 하는 이상한 개인기 같은 거 있어?", "분위기"),
            ("길거리 밴드 공연이나 버스킹 구경하는 거 좋아해, 아니면 부끄러워서 그냥 쓱 지나쳐?", "버스킹"),
            ("악기 하나를 마스터할 수 있는 능력이 하루아침에 생긴다면 어떤 악기를 고를래?", "피아노"),
            ("조용한 전시회나 미술관 가서 작품 감상하는 거 좋아해? 아니면 그런데 가면 하품부터 나와?", "미술관"),
            ("본인 글씨체가 예쁜 편이야, 아니면 나만 알아볼 수 있는 암호 같은 악필이야?", "암호"),
            ("내 음악 플레이리스트를 남에게 보여줄 때 음악 취향을 들킬까 봐 은근히 부끄럽거나 신경 쓰여?", "플레이리스트"),
            ("가사가 전혀 없는 연주곡이나 클래식, 재즈 같은 음악도 즐겨 듣는 편이야?", "연주곡"),
            ("내 파란만장한 인생을 한 편의 영화로 만든다면 제목을 뭐라고 지을래?", "조용히 남아"),
            ("모든 걸 다 터놓을 수 있는 한 명의 베프 vs 적당히 친하고 재밌는 열 명의 친구, 뭐가 더 좋아?", "한 명"),
            ("돈 빌려달라는 친구의 간절한 부탁, 못 받아도 괜찮다 생각하고 얼마까지 쿨하게 빌려줄 수 있어?", "생활이 흔들리지"),
            ("진짜 믿었던 친구가 뒤에서 내 험담을 하고 다닌 걸 알게 되면 바로 찾아가서 따질래, 아니면 조용히 손절할래?", "거리를 둘"),
            ("남의 험담 뒷담화 듣는 게 팝콘 각이라 재밌어, 아니면 듣는 것만으로도 기 빨리고 불편해?", "기 빨리는"),
            ("단톡방에서 대화가 뚝 끊겼을 때 그 어색함을 못 참고 먼저 아무 말이나 던져서 살리는 편이야?", "정적"),
            ("진짜 너만 알고 있어라고 들은 비밀, 진짜 친한 다른 베프 딱 한 명한테는 전해본 적 솔직히 있지?", "안 옮기는"),
            ("약속 시간에 매번 늦는 코리안 타임 친구, 진짜 진심으로 화내본 적 있어 아니면 그냥 해탈했어?", "매번 늦"),
            ("절친이랑 단둘이 며칠 여행 갔다가 너무 안 맞아서 대판 싸우고 절교하거나 어색해진 적 있어?", "생활 리듬"),
            ("나와 성향 취미 성격이 완전 정반대인 사람과도 둘도 없는 절친이 될 수 있다고 생각해?", "정반대"),
            ("여러 명이 모인 무리에 있을 때 분위기 메이커 역할을 주도해, 아니면 조용히 웃고 리액션만 하는 편이야?", "리액션"),
            ("완벽한 가상현실 VR에서 하루 종일 살 수 있는 시대가 오면 팍팍한 현실을 버리고 그 안으로 들어갈래?", "VR"),
            ("내 뇌 속의 모든 기억을 USB처럼 컴퓨터에 백업해 둘 수 있는 기술이 생기면 쓸 거야?", "기억 백업"),
            ("AI 로봇이 인간 수준의 감정을 느낀다고 눈물을 흘리면 그 로봇에게 인권을 줘야 할까?", "권리"),
            ("원하는 곳 어디든 갈 수 있는 순간이동 기계가 발명됐는데 0.01% 확률로 오류 나서 사라질 수도 있다면 탈래?", "안 탈래"),
            ("하늘을 날아다니는 자동차 vs 바닷속을 다니는 잠수함 자동차, 뭐가 더 빨리 우리 일상에 도입될 것 같아?", "하늘 자동차"),
            ("유전자 조작으로 나의 신체 능력을 하나 업그레이드할 수 있다면 어떤 능력을 올리고 싶어?", "체력"),
            ("평생 맛있는 걸 못 먹는 대신 알약 하나만 먹으면 포만감과 영양이 완벽히 채워지는 세상이 오면 좋겠어?", "맛있는"),
            ("내 미래 배우자의 얼굴을 미리 볼 수 있는 기계가 있다면 볼래, 아니면 재미없어지니까 안 볼래?", "안 볼래"),
            ("누가 로봇인지 인간인지 구별할 수 없는 세상에서 너는 네가 진짜 인간이라는 걸 어떻게 증명할래?", "흔들림"),
            ("화성에 인류가 거주하는 거대한 돔 도시가 세워진다면 첫 이민자 모집에 지원해서 지구를 떠날 생각 있어?", "화성"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "50% 금수저",
            "팝콘 먹으면서",
        )

        for index, (prompt, expected) in enumerate(cases, start=251):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-251-300-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_fear_magic_family_moral_misc_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("살면서 귀신이나 괴물보다 현실적으로 이게 더 무섭다라고 느낀 게 있어? ex 카드 명세서 사람 텅 빈 통장", "카드 명세서"),
            ("고소공포증 심해공포증 폐소공포증 환공포증 같은 남들보다 예민한 특별한 공포증 있어?", "심해공포"),
            ("집에서 혼자 있는데 손가락만 한 바퀴벌레 나오면 어떻게 해? 직접 잡을 수 있어?", "거리"),
            ("밤에 불 다 끄고 혼자서 무서운 공포 영화 팝콘 먹으면서 볼 수 있어?", "리모컨"),
            ("놀이공원 귀신의 집에 혼자 맨 앞장서서 들어가기 가능?", "중간"),
            ("평소에 일어날 확률이 0.1%도 안 되는 쓸데없는 걱정이나 망상 자주 하는 편이야?", "0.1%"),
            ("병원에서 피 뽑거나 주사 맞을 때 주삿바늘 들어가는 거 눈 뜨고 쳐다볼 수 있어?", "주삿바늘"),
            ("살면서 생명의 위협을 느낄 만큼 아 진짜 죽을 뻔했다 싶었던 아찔한 순간이 있어?", "기억은 만들지"),
            ("길 가다가 엄청 큰 개가 목줄 없이 짖으면서 나한테 다가오면 어떻게 대처할래?", "천천히"),
            ("나이 들고 늙어가는 것에 대한 두려움이 커? 아니면 중후해지는 게 기대돼?", "중후"),
            ("하루에 딱 한 번 시간을 1시간 전으로 되돌릴 수 있다면 언제 제일 자주 쓸 것 같아?", "말실수"),
            ("만약 네가 해리포터 마법 학교에 간다면 어떤 마법을 제일 먼저 완벽하게 배우고 싶어?", "정리 마법"),
            ("자는 동안 내가 원하는 스토리를 세팅해서 꿈을 꿀 수 있는 능력이 있다면 오늘 밤 무슨 꿈을 꿀래?", "비 오는 도서관"),
            ("만약 동물로 맘대로 변신할 수 있다면 어떤 동물로 변해서 하루를 보내고 싶어?", "고양이"),
            ("내 눈에만 보이고 나랑 대화도 가능한 수호천사 혹은 악마가 24시간 따라다닌다면 든든할까 귀찮을까?", "든든함 반"),
            ("알약 하나만 먹으면 원어민처럼 완벽하게 패치되는 외국어가 있다면 뭘 고를래? 영어 제외", "일본어"),
            ("모든 걸 꿰뚫어 보는 투시 능력이 생긴다면 복권 긁는 거 말고 일상생활 어디에 써먹을래?", "잃어버린 물건"),
            ("날씨를 내 마음대로 조종할 수 있는 능력이 생기면 주로 어떻게 쓸래?", "조용한 비"),
            ("다른 사람의 거짓말을 들으면 내 머릿속에 삐 하고 경고음이 울린다면 인간관계가 어떻게 될까?", "예민"),
            ("평생 늙지 않고 병들지도 않으면서 영원히 사는 불로불사의 약, 눈앞에 있다면 먹을래 말래?", "안 먹"),
            ("어릴 때 형제자매랑 치고받고 많이 싸웠어? 제일 어이없게 싸운 이유 기억나? 외동이면 혼자 놀기의 달인이었어?", "리모컨"),
            ("부모님 성격이나 외모 중에 아 이건 진짜 소름 돋게 똑닮았구나 싶은 점 있어?", "걱정하는 방식"),
            ("어렸을 때 산타클로스 할아버지의 정체를 언제 어떻게 처음 알게 됐어? 동심 파괴의 순간", "산타"),
            ("명절에 친척들 다 모이면 반갑고 재밌어, 아니면 잔소리나 뻘쭘함 때문에 스트레스받는 편이야?", "잔소리"),
            ("어릴 때 항상 안고 다니거나 제일 아끼던 장난감 인형 로봇 같은 거 뭐였어?", "낡은 인형"),
            ("학창 시절에 부모님이나 선생님 몰래 해본 제일 큰 일탈 땡땡이 오락실 피시방 같은 거 해본 적 있어?", "피시방"),
            ("가족들이랑 다 같이 있는 카톡방 있어? 주로 무슨 대화해, 아니면 공지사항 방이야?", "공지사항"),
            ("어릴 때 부모님이 시켜서 억지로 다녔던 피아노 태권도 주산 학원 중에 제일 가기 싫었던 곳은?", "피아노"),
            ("나중에 결혼해서 가정을 꾸린다면 자녀는 몇 명 낳고 싶어? 아니면 완전 비혼주의 딩크족이야?", "책임"),
            ("엄마랑 아빠 중에 누구랑 성향이나 대화 코드가 조금 더 잘 통하는 것 같아?", "들어주는 방식"),
            ("무인도에 조난당해서 일주일 굶었는데 탈출하려면 섬에 있는 귀여운 강아지를 잡아먹어야 해. 어떻게 할래?", "못 할"),
            ("내 목숨을 구해주고 대신 억울하게 감옥에 간 친구, 면회 가고 평생 뒷바라지해 줄 수 있어?", "면회"),
            ("훔친 돈인 줄 알면서도 아픈 가족의 수술비로 몰래 쓴 사람, 도덕적으로 용서받을 수 있다고 생각해?", "정당화"),
            ("내 기억을 부분 조작해서 가장 슬프고 고통스러웠던 기억 딱 하나를 아예 삭제할 수 있다면 지울래?", "안 지울"),
            ("남을 돕고 기부하는 이타적인 행동도 결국 내 기분이 좋아지기 위한 이기적인 본능일까?", "이기적인 건 아니"),
            ("절대 100% 안 들키고 완전범죄가 보장되는 범죄를 한 번 저지를 수 있다면 할 거야?", "안 할래"),
            ("인공지능 자율주행 자동차가 사고를 냈다면 배상 책임은 타고 있던 운전자한테 있을까 제조사한테 있을까?", "제조사"),
            ("평생 가난하고 빚에 시달리지만 죽을 때까지 모든 사람에게 존경받기 vs 엄청난 갑부인데 모두에게 쌍욕 먹기", "존경"),
            ("네가 생각하는 인간이 저지를 수 있는 가장 끔찍한 죄악 나쁜 짓은 뭐라고 생각해?", "존엄"),
            ("동물원이나 좁은 아쿠아리움에 갇혀있는 동물들을 보면 어떤 생각이 들어?", "갇혀"),
            ("만약 지구상에서 딱 한 가지 음식을 영원히 멸종시킬 수 있다면 뭘 없앨래?", "괴식"),
            ("잠에서 깼는데 내가 즐겨 하던 게임이나 보던 웹툰 속 지나가는 행인 1 NPC가 되어 있다면 어떨까?", "관찰자"),
            ("평생 치킨을 먹을 때 뼈 있는 치킨만 발골해 먹기 vs 평생 순살 치킨만 먹기", "순살"),
            ("한여름 밤 모기장 안에 들어온 앵무새 같은 모기 1마리, 피곤해도 잡을 때까지 불 켜고 잠 안 자?", "불 켤"),
            ("무력으로 외계인이 지구를 침공해서 인간 대표 한 명이랑 대화하겠다고 하면 누굴 내보낼래?", "침착"),
            ("다이소에 가면 살 거 하나도 없었는데 무의식적으로 장바구니 채워서 1시간 구경 쌉가능?", "쌉가능"),
            ("팥붕 vs 슈붕, 부먹 vs 찍먹을 이을 새로운 한국인 분열 논쟁거리는 뭐가 있을까?", "반숙"),
            ("지금 네 핸드폰 주소록에 전화번호 몇 개나 저장되어 있어? 그중 자주 연락하는 사람은 몇 명이야?", "자주 연락"),
            ("만약 네가 무인도에 나라를 하나 세운다면 국기 중앙에는 어떤 그림을 그려 넣고 싶어?", "불빛"),
            ("오늘 나랑 쉴 새 없이 한 대화들 중에 제일 쓸데없는데 묘하게 재밌었던 질문은 뭐였어?", "음식 논쟁"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "칼, 라이터",
            "월급은 보통",
            "내 장점 하나",
            "유령 도시",
        )

        for index, (prompt, expected) in enumerate(cases, start=301):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-301-350-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_regret_animals_school_fashion_survival_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("학창 시절로 돌아갈 수 있다면 제일 먼저 머리 박고 지우고 싶은 최악의 흑역사가 뭐야?", "흑역사"),
            ("새벽 감성 타서 옛날에 헤어진 애인한테 자니 혹은 잘 지내라고 카톡 보내본 적 있어?", "자니"),
            ("아 진짜 그때 그 주식 혹은 비트코인 샀어야 했는데 하고 땅을 치고 후회했던 적 있어?", "비트코인"),
            ("살면서 부모님한테 홧김에 했던 말 중에 가장 죄송해서 아직도 마음에 걸리는 말이 있어?", "홧김"),
            ("아주 어렸을 때 친구한테 빌려 가놓고 돌려주기 싫어서, 혹은 까먹어서 아직도 꿀꺽한 물건 있어?", "꿀꺽"),
            ("살면서 내가 미쳤지 하면서 충동구매했던 물건 중에 제일 비싸고 쓸데없는 건 뭐야?", "충동구매"),
            ("10년 전 과거의 나를 만나면 멱살을 잡고서라도 꼭 해주고 싶은 뼈 때리는 조언이 뭐야?", "멱살"),
            ("다이어트나 운동 빡세게 결심해 놓고 헬스장 등록한 뒤에 딱 3일 나가고 기부해 본 적 있지?", "헬스장"),
            ("나를 진짜 진심으로 좋아해 줬던 사람을 뻥 차버리고 나중에 혼자 이불킥하면서 후회해 본 적 있어?", "좋아해준"),
            ("다시 고등학생 때로 완벽하게 돌아간다면 이번엔 진짜 코피 터지게 공부 열심히 해볼 생각 있어?", "공부"),
            ("길고양이나 산책하는 강아지 보면 꼭 멈춰서 우쭈쭈 인사하고 가, 아니면 눈길도 안 주고 지나가?", "우쭈쭈"),
            ("평생 반려동물을 딱 한 마리만 키울 수 있다면 어떤 동물을 키우고 싶어?", "고양이"),
            ("동물원 사파리 투어 가면 사자나 호랑이 같은 무서운 맹수가 좋아, 아니면 기린이나 코끼리가 좋아?", "맹수"),
            ("길가에 있는 새 비둘기 까치 진짜 푸드덕거리는 거 극혐하는 편이야, 아니면 별생각 없어?", "푸드덕"),
            ("식물 키워본 적 있어? 선인장이나 다육이도 바싹 말려 죽이는 마이너스의 손이야?", "선인장"),
            ("바다에 가면 튜브 타고 수영하면서 노는 게 좋아, 아니면 파라솔 밑에서 파도 소리 들으며 바다멍 때리는 게 좋아?", "바다멍"),
            ("등산 가는 거 좋아해? 땀 뻘뻘 흘리고 산 정상에서 먹는 미지근한 컵라면 맛을 알아?", "컵라면"),
            ("집에 혼자 있는데 팔뚝만 한 뱀이 들어오면 119 부르기 전까지 어떻게 대처할래?", "119"),
            ("수족관에 갇혀있는 덩치 큰 범고래나 돌고래 보면 불쌍해, 아니면 그저 신기하고 멋있어?", "불쌍"),
            ("수많은 곤충 중에 진짜 제일 징그럽고 멸종했으면 좋겠는 거 딱 하나만 꼽자면?", "바퀴벌레"),
            ("학교 다닐 때 4교시 종 치자마자 급식실로 뛰어가는 학생이었어, 아니면 매점 빵 VVIP였어?", "급식실"),
            ("대학 시절 혹은 20대 초반 무식하게 제일 많이 마셔본 최대 주량이 어떻게 돼?", "주량"),
            ("MT나 수련회 갔을 때 밤새 안 자고 무서운 얘기 하거나 몰래 술 마신 아련한 추억 있어?", "자는 척"),
            ("시험 전날 벼락치기 할 때 꼭 책상 정리부터 시작하고 손톱 깎는 딴짓 병에 걸려본 적 있지?", "책상 정리"),
            ("수업 시간에 몰래 엎드려 자다가 내 침 흘리는 소리나 발차기하면서 흠칫 놀라서 깬 적 있어?", "침"),
            ("선생님이나 교수님이 진짜 재미없는 아재 개그 농담했을 때 성적 때문에 억지로 엄청 웃어준 적 있어?", "억지웃음"),
            ("조별 과제 팀플 할 때 주로 발표자였어, 자료 조사였어, 아니면 이름만 올린 무임승차 빌런이었어?", "자료 조사"),
            ("체육대회나 축제 때 제일 열심히 응원하고 춤추는 인싸였어, 아니면 스탠드 구석에 앉아있는 구경꾼이었어?", "구경꾼"),
            ("도서관이나 독서실에서 공부하다가 캔커피나 포스트잇 쪽지 받아본 적 혹은 줘본 적 있어?", "포스트잇"),
            ("옷장 구석에 박혀있는 초중고 졸업 앨범 꺼내보면 너무 창피해서 찢어버리고 싶은 페이지 있어?", "졸업 앨범"),
            ("옷장에 1년 넘게 한 번도 안 입은 옷인데 언젠간 살 빼서 입겠지 하고 절대 안 버리는 옷 꽤 있지?", "언젠간"),
            ("쇼핑할 때 혼자 가서 조용히 이어폰 끼고 사는 게 좋아, 아니면 친구랑 가서 이거 어때 피드백받는 게 좋아?", "혼자"),
            ("화장품이나 향수, 한 가지 브랜드에 꽂히면 다 쓸 때까지 한 놈만 계속 패는 정착하는 편이야?", "정착"),
            ("속옷이나 양말에 작은 빵꾸 났는데 버리기 아까워서 혹은 새 거 꺼내기 귀찮아서 그냥 입고 나간 적 있어?", "빵꾸"),
            ("집 앞 편의점 갈 때 풀메이크업 꾸미기 가능? 아니면 무조건 떡진 머리에 모자 푹 눌러쓰고 쌩얼?", "모자"),
            ("인터넷에서 옷 샀는데 모델핏이랑 너무 달라서 충격받고, 환불도 귀찮아서 잠옷으로 쓰는 옷 있지?", "잠옷"),
            ("비싼 명품 가방 하나 살 돈으로 예쁜 보세옷 100벌 사는 게 훨씬 낫다고 생각하는 실용주의자야?", "보세옷"),
            ("계절 바뀔 때마다 옷장 열어보고 작년엔 도대체 벗고 다녔나 싶을 정도로 입을 옷이 없지 않아?", "입을 옷"),
            ("미용실 갔을 때 머리 쥐 파먹은 것처럼 망해서 마음에 안 들어도 미용사 앞에서는 아 예뻐요 해본 적 있어?", "예뻐요"),
            ("뿔테 안경이나 선글라스 썼을 때 맨얼굴이랑 분위기나 이미지가 확 달라지는 편이야?", "이미지"),
            ("1년 뒤에 소행성이 충돌해서 지구가 멸망한다는 100% 확실한 뉴스가 뜨면, 내일 당장 회사 학교 갈 거야?", "안 갈래"),
            ("좀비 바이러스가 퍼져서 미친 듯이 도망쳐야 할 때, 달리기 느린 가족이나 친구를 업고서라도 끝까지 갈 거야?", "끝까지"),
            ("핵전쟁으로 인류가 다 죽고 지하 방공호에 모르는 사람 10명이랑 갇히면 네가 리더 할래, 아니면 조용히 묻어갈래?", "리더"),
            ("무인도에 떨어졌을 때 지식 체력 등 제일 도움이 될 것 같은 친구 한 명만 데려간다면 누구 데려갈래?", "체력"),
            ("뱀파이어 피 마시기나 늑대인간 보름달 털보가 실제로 존재해서 둘 중 하나가 돼야 한다면 뭘 고를래?", "늑대인간"),
            ("외계인이 쳐들어왔는데 나보고 지구인을 배신하고 스파이 노릇을 하면 나만 살려준대. 배신할 거야?", "배신"),
            ("재난 영화 지진 해일 보면 네가 주인공 무리처럼 끝까지 살아남을 것 같아, 아니면 초반에 깔려 죽는 엑스트라일 것 같아?", "엑스트라"),
            ("집에 불이 났을 때 사람과 반려동물은 다 대피시켰다면, 마지막으로 딱 하나 들고나올 수 있는 물건은 뭐야?", "외장하드"),
            ("오늘 밤부터 전 세계에 인터넷과 전기가 영원히 끊긴다면, 당장 촛불 켜놓고 뭐 하면서 시간 보낼래?", "촛불"),
            ("만약 오늘이 지구의 마지막 밤이라서 몇 시간 뒤면 끝난다면, 자기 전에 눈 감고 속으로 무슨 기도를 할 것 같아?", "고마웠다"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "그런 동조",
            "천천히 관심",
            "남의 비밀은",
            "미래 배우자",
            "전세계 송출",
        )

        for index, (prompt, expected) in enumerate(cases, start=351):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-351-400-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_hobby_food_sleep_social_mental_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("돈과 시간이 무한정 주어진다면, 직장/학교 다 때려치우고 가장 깊게 파고들고 싶은 취미가 뭐야?", "글쓰기"),
            ("목공, 도자기 굽기, 가죽 공예 같은 손으로 꼬물꼬물 만드는 수작업 취미에 관심 있어?", "도자기"),
            ("집 안에 무제한 예산으로 나만의 공간을 꾸민다면, 홈 카페 vs 홈 피시방 게임룸 vs 홈 시네마 중 어느 것?", "홈 시네마"),
            ("평소에 숨쉬기 운동 빼고 아예 안 하는 편이야, 아니면 헬스장 기부 천사라도 등록은 해두는 편이야?", "헬스장"),
            ("남들은 이해 못 해도 나만 모으는 수집품 있어? 피규어 LP 판 향수 우표 다이어리 스티커 등", "스티커"),
            ("최신식 노래방 기계가 방에 생기면 매일 부를 거야, 아니면 층간소음 무서워서 장식용 옷걸이가 될까?", "층간소음"),
            ("유튜브 영상 편집이나 브이로그, 게임 방송 같은 거 나도 진짜로 한 번 해볼까 진지하게 고민해 본 적 있어?", "브이로그"),
            ("악기 중에 드럼이나 일렉기타처럼 스트레스 팍팍 풀리는 시끄러운 거 맘껏 쳐보고 싶어?", "드럼"),
            ("캘리그라피나 뜨개질처럼 조용하고 정적인 취미가 맞아, 아니면 클라이밍이나 테니스 같은 활동적인 게 맞아?", "정적인"),
            ("비 오는 날, 하루 종일 만화카페 푹신한 소파에 틀어박혀서 짜파게티 먹으며 만화책만 보기, 완전 극호?", "만화카페"),
            ("피자 먹을 때 끝에 퍽퍽한 빵 테두리 다 먹는 편이야, 아니면 치즈 크러스트 아니면 쿨하게 버려?", "테두리"),
            ("집에서 라면 끓일 때 물 끓기 전에 스프부터 먼저 넣어, 아니면 물 끓으면 면부터 넣어?", "스프"),
            ("배스킨라빈스 같은 데 가면 제일 먼저 고르는 최애 아이스크림 맛은? 초코 바닐라 상큼한 과일 민초 등", "초코"),
            ("회식이나 친구들 모임에서 고기 구울 때 자진해서 집게랑 가위를 드는 편이야, 아니면 얌전히 먹기만 해?", "집게"),
            ("뷔페에 가면 예의상 샐러드나 수프로 위장을 달래고 시작해, 아니면 무조건 육류나 초밥부터 돌격해?", "초밥"),
            ("카페 가면 항상 먹던 아아만 고집해, 아니면 신메뉴 흑임자 샷 라떼 나오면 꼭 모험해 보는 편이야?", "신메뉴"),
            ("만약 밤에 잠 안 올까 봐 커피를 못 마신다면, 카페에서 주로 뭐 마셔? 스무디 자몽에이드 캐모마일 등", "캐모마일"),
            ("삼겹살 노릇노릇하게 구워 먹을 때 쌈장, 기름장, 소금, 와사비 중에 뭘 제일 많이 찍어 먹어?", "쌈장"),
            ("배달 음식 시킬 때 리뷰 약속 이벤트 꼭 참여해서 치즈볼이나 음료수 서비스받아내는 알뜰족이야?", "치즈볼"),
            ("평생 매콤달콤한 떡볶이 안 먹기 vs 평생 지글지글 삼겹살 안 먹기, 한국인으로서 어느 쪽이 더 고문일까?", "떡볶이"),
            ("밤에 잘 때 암막 커튼 치고 불빛 하나 없이 깜깜해야 잘 자, 아니면 수면등 같은 작은 조명이 있어야 안심돼?", "암막"),
            ("불면증 올 때 유튜브로 ASMR, 장작 타는 소리, 빗소리 같은 백색소음 틀어놓고 자는 거 효과 있어?", "빗소리"),
            ("베개 높이는 호텔 베개처럼 푹신하고 높은 게 좋아, 아니면 목 아파서 거의 없는 듯이 얇은 게 좋아?", "얇은"),
            ("꿈에서 깼는데 감정에 너무 몰입해서 슬프거나 억울해서 진짜로 베개가 눈물로 젖어있던 경험 있어?", "베개"),
            ("아침에 눈 떴을 때 창문으로 따뜻한 햇살이 쫙 들어오는 게 상쾌해, 아니면 암막 커튼으로 밤인지 낮인지 모르는 게 좋아?", "햇살"),
            ("코골이나 이갈이가 탱크 수준으로 심한 사람과 같은 방에서 어쩔 수 없이 자야 한다면 잘 수 있어?", "못 잘"),
            ("엎드려 자는 거랑, 새우잠 자는 거랑, 대자로 뻗어서 자는 거 중에 평소에 어느 자세로 많이 자?", "새우잠"),
            ("맘먹고 알람 다 끄고 늦잠 잔다면, 다음 날 해 중천에 뜰 때까지 최대 몇 시까지 안 깨고 잘 수 있어?", "정오"),
            ("꿈속에서 내가 겪은 생생한 일이 며칠 뒤 현실에서 비슷하게 일어난 소름 돋는 예지몽 데자뷔 경험 있어?", "데자뷔"),
            ("자기 전에 불 끄고 누워서 스마트폰 하다가 졸아서 얼굴에 폰 떨어뜨리고 코 깨질 뻔한 적 솔직히 10번 넘지?", "폰"),
            ("아파트 엘리베이터에 진짜 안 친한 이웃이랑 단둘이 타면 끝까지 허공이나 폰만 봐, 아니면 가벼운 목례라도 해?", "목례"),
            ("미용실에서 머리 자를 때 미용사님이랑 스몰토크로 수다 떠는 게 재밌어, 아니면 조용히 눈 감고 쉬는 게 좋아?", "조용히"),
            ("길 가다가 설문조사나 인상이 참 좋으시네요 하고 말 걸면 단호하게 잘라? 아니면 거절 못 하고 시간 뺏겨?", "단호"),
            ("택시 탔을 때 기사님이 정치나 세상 돌아가는 무거운 얘기 꺼내시면 맞장구 쳐드려, 아니면 자는 척해?", "자는 척"),
            ("식당에서 김치나 단무지 더 달라고 말할 때 눈치 보여서 직원분들 안 바쁠 타이밍 엄청 눈치 게임해?", "타이밍"),
            ("횡단보도 빨간불인데 차가 진짜 개미 새끼 한 마리 안 오면 무단횡단 살짝 할 때 있어, 아니면 무조건 파란불 기다려?", "파란불"),
            ("영화관에서 영화 다 보고 일어났는데 내 자리 쪽에 팝콘 엄청 흘려놨으면 살짝 주워 모아놔, 아니면 그냥 도망쳐?", "팝콘"),
            ("길 가다가 바닥에 떨어진 빳빳한 오만 원짜리를 주웠어. 경찰서에 갖다 줄 거야, 아니면 내 주머니로 쏙?", "경찰서"),
            ("친구가 밥 먹고 활짝 웃는데 앞니에 고춧가루 꼈을 때, 바로 직구로 말해줘 아니면 민망할까 봐 모르는 척해줘?", "고춧가루"),
            ("길에서 혼자 이어폰 꽂고 걷다가 노래 흥얼거렸는데, 하필 조용해질 때 옆사람이랑 눈 마주쳐서 당황한 적 있어?", "흥얼"),
            ("남한테 외모나 능력 칭찬을 들으면 기분 좋게 인정하는 편이야, 아니면 속으로 나한테 뭔가 바라는 게 있나 의심해?", "인정"),
            ("스트레스가 한계치에 달해서 터지면 눈물부터 주룩주룩 나는 편이야, 아니면 화가 머리끝까지 폭발하는 편이야?", "눈물"),
            ("지금까지 살아온 걸 돌아봤을 때 스스로 운이 참 좋은 사람인 것 같아, 아니면 불운과 억까가 많은 편인 것 같아?", "운"),
            ("어떤 일에 크게 실패했을 때 내가 부족했지라며 내 안에서 원인을 찾아, 아니면 상황이 안 좋았어라며 탓을 해?", "원인"),
            ("누군가 나에게 무리한 부탁을 했을 때 단호하게 거절하는 걸 어려워하거나 거절 후 죄책감을 많이 느끼는 편이야?", "거절"),
            ("평소에 내가 잘못한 상황이 아닌데도 아 죄송합니다라는 말을 쿠션어처럼 습관적으로 자주 남발해?", "죄송합니다"),
            ("나를 이유 없이 질투하거나 뒤에서 미워하는 사람이 있다는 걸 알게 되면 며칠 밤낮으로 엄청 신경 쓰여?", "신경"),
            ("혼자 식당 가서 고기 구워 먹고, 혼자 심야 영화 보는 고독함을 이제는 온전히 즐길 수 있는 레벨이 됐어?", "혼자"),
            ("현재 내 인생에서 가장 턱없이 부족하다고 뼈저리게 느끼는 게 뭐야? 돈 시간 마음의 여유 체력 인맥 등", "마음의 여유"),
            ("지금의 나 자신과 현재의 삶에 10점 만점을 기준으로 몇 점 정도 줄 수 있을 것 같아?", "7점"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "전자기기 없이",
            "나이브스 아웃",
            "확실하지 않음",
            "권리 논의",
        )

        for index, (prompt, expected) in enumerate(cases, start=401):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-401-450-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_story_balance_roleplay_fillblank_relationship_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("살면서 지금까지 자다가도 이불 뻥뻥 차는 제일 부끄러운 흑역사 썰 하나만 풀어줘.", "흑역사"),
            ("당근마켓이나 중고 거래하면서 만났던 제일 황당하거나 빌런 같았던 사람 썰 있어?", "당근마켓"),
            ("술 마시고 취해서 했던 행동 중에 다음날 일어나서 제일 어이없었던 주사 썰 하나만!", "주사"),
            ("어릴 때 친구들이랑 장난치고 놀다가 경비 아저씨나 경찰한테 도망쳤던 짜릿한 썰 있어?", "도망"),
            ("아르바이트나 직장 생활하면서 진짜 이 사람은 답이 없다 느꼈던 진상 손님 상사 썰 풀어줘.", "진상"),
            ("나 혼자만 오해해서 썸인 줄 알았거나, 혼자 급발진해서 민망했던 착각 썰 있어?", "착각"),
            ("사람 많은 길 가다 넘어져서 아픈 것보다 창피한 게 컸던 쪽팔림 썰 하나만 들려줘.", "쪽팔림"),
            ("속으로만 생각해야지 했는데 나도 모르게 입 밖으로 튀어나와서 분위기 싸해졌던 썰 있어?", "입 밖"),
            ("살면서 이게 진짜 우연이라고 싶을 정도로 소름 돋았던 타이밍 썰 하나 풀어봐.", "타이밍"),
            ("짝사랑하다가 혼자 북 치고 장구 치고 감정 낭비 다 하고 끝났던 찌질한 짝사랑 썰 있어?", "짝사랑"),
            ("100% 확률로 1억 받기 vs 10% 확률로 100억 받기 실패 시 0원", "1억"),
            ("평생 친구 딱 1명도 없이 살기 vs 평생 스마트폰 인터넷 아예 끊고 살기", "인터넷"),
            ("나를 사랑하는 사람이 매일 의심하고 집착하기 vs 나를 사랑하는 사람이 내 일상에 완전 무관심하기", "집착"),
            ("모르는 사람 100명 앞에서 바지 벗겨지기 vs 좋아하는 사람 포함 아는 사람 10명 앞에서 방귀 크게 뀌기", "모르는 사람"),
            ("평생 콜라 사이다 등 모든 탄산음료 안 마시기 vs 평생 모든 종류의 라면 안 먹기", "탄산"),
            ("과거로 돌아가서 내 인생 최악의 실수 지우기 vs 미래로 가서 1년 뒤 로또 번호 1등 보고 오기", "실수"),
            ("말 안 통하고 문화도 다른 외국인 룸메이트랑 1년 살기 vs 말 너무 많고 가르치려 드는 꼰대 룸메이트랑 1년 살기", "외국인"),
            ("한여름 35도에 에어컨 없이 롱패딩 입기 vs 한겨울 -10도에 보일러 없이 반팔 입기", "한겨울"),
            ("평생 고기 소 돼지 닭 안 먹기 vs 평생 밀가루 빵 면 과자 피자 안 먹기", "밀가루"),
            ("애인 핸드폰 1시간 동안 몰래 보기 권한 vs 내 핸드폰 모든 기록 애인한테 24시간 공유하기", "폰 기록"),
            ("자고 일어났더니 갑자기 10년 뒤 미래야. 폰 켜서 구글에 제일 먼저 검색해 볼 단어가 뭐야?", "내 이름"),
            ("너한테 서울 한복판 100억짜리 건물이 생겼어. 1층부터 꼭대기까지 세입자로 어떤 가게들을 입점시키고 싶어?", "카페"),
            ("무인도에 조난당했는데 구조선이 왔어. 근데 딱 한 사람만 태울 수 있대. 너 탈 거야, 옆 사람 양보할 거야?", "양보"),
            ("엘리베이터에 갇혔는데 폰 배터리가 1% 남았어. 누구한테 제일 먼저 전화해서 짧게 뭐라고 할래?", "배터리"),
            ("좀비 사태가 터졌는데, 눈앞에 나를 물려고 달려드는 좀비가 네 베스트 프렌드야. 무기로 공격할 수 있어?", "친구"),
            ("길 가다가 도플갱어 나랑 똑같이 생긴 사람이랑 눈이 딱 마주쳤어. 제일 먼저 무슨 말 걸래?", "첫마디"),
            ("네가 우리나라 대통령이 된다면 당선 첫날 무슨 일이 있어도 무조건 통과시키고 싶은 법안 하나가 뭐야?", "과로"),
            ("외계인이 나타나서 지구의 가장 맛있는 음식을 하나만 추천해라라고 협박해. 뭘 먹여서 돌려보낼래?", "김치볶음밥"),
            ("네가 갑자기 5인조 아이돌 그룹으로 강제 데뷔하게 됐어. 너의 포지션 메인보컬 댄스 얼굴 천재 예능캐 등은 뭘까?", "예능캐"),
            ("눈을 떠보니 네가 아주 좋아하는 로맨스 판타지 소설 속 세계야. 어떤 역할 주인공 악녀 악당 돈 많은 조력자를 하고 싶어?", "조력자"),
            ("내가 생각할 때, 내 성격을 한 가지 색깔로 표현한다면 빈칸 색이다. 그 이유는...", "남색"),
            ("스트레스가 머리끝까지 찼을 때 나를 가장 빠르고 확실하게 진정시키는 행동은 빈칸 하는 것이다.", "씻고"),
            ("인간관계에서 남들은 넘어가 줘도 내가 절대 용서할 수 없는 행동 딱 한 가지는 빈칸 이다.", "비밀"),
            ("누군가 나에게 조건 없이 100만 원을 주면서 1시간 안에 몽땅 쓰라고 한다면 나는 빈칸 을 살 것이다.", "의자"),
            ("살면서 내가 가장 아깝다고 생각하면서도 펑펑 돈을 낭비하는 분야는 빈칸 이다.", "배달비"),
            ("내가 생각하는 완벽한 하루란, 아침에 빈칸 하고, 저녁엔 빈칸 하며 마무리하는 것이다.", "조용히"),
            ("남들은 이해 못 해도 나는 절대 양보 못하고 꼭 지키는 나만의 생활 철학 강박은 빈칸 이다.", "적어두"),
            ("내가 누군가 이성 동성에게 확 호감을 느끼는 가장 결정적인 포인트는 상대가 빈칸 할 때이다.", "들어줄"),
            ("나의 묘비명에 인생을 요약하는 딱 한 줄을 남길 수 있다면 나는 빈칸 라고 적고 싶다.", "자기 속도"),
            ("내일 소행성 충돌로 세상이 끝난다면, 오늘 밤 내 최후의 만찬 메뉴는 빈칸 이다.", "김치찌개"),
            ("너는 내 첫인상이 어땠어? 지금 대화 나눠보니까 처음이랑 느낌이 많이 달라?", "집요"),
            ("우리가 지금까지 나눈 여러 대화 중에 제일 인상 깊었거나 웃겼던 대화가 뭐야?", "Black"),
            ("나한테 이것 하나만 고쳐주면 완벽할 텐데 싶은 아쉬운 점 딱 하나만 돌직구로 말해준다면?", "질문"),
            ("네가 겪어본 바로는, 우리 둘의 성격이나 개그 코드는 비슷한 편인 것 같아 아니면 완전 정반대인 것 같아?", "비슷"),
            ("만약 우리가 오프라인에서 실제로 만나서 딱 하루 놀 수 있다면 제일 먼저 어디 가서 뭐 하고 놀까?", "카페"),
            ("나를 생각하면 딱 떠오르는 동물이나 찰떡인 이모티콘 있어? 없으면 하나 지정해 줘!", "고양이"),
            ("만약 우리가 엄청 크게 싸우게 된다면, 아마 어떤 주제나 오해 때문에 싸우게 될까?", "답변 품질"),
            ("내 연락처 이름 뭐라고 저장해 놨어? 만약 AI라면 나를 무슨 별명으로 부르고 싶어?", "실험실장"),
            ("너한테 나는 100점 만점에 몇 점짜리 훌륭한 혹은 피곤한 대화 파트너인 것 같아?", "92점"),
            ("벌써 500개의 대화 주제가 끝났네. 이 많은 질문들 끝나고 나면, 우리 내일부터는 또 무슨 얘기 하면서 놀까?", "실패한 답변"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "권리 논의",
            "이 사람은 답이 없다",
        )

        for index, (prompt, expected) in enumerate(cases, start=451):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-korean-daily-451-500-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_foundation_regression_variants(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("매운 거 먹고 싶은데 내 속이 버텨줄지 모르겠어.", "중간맛"),
            ("혼자 밥 먹어도 오늘은 좀 제대로 챙겨 먹고 싶어.", "따뜻한"),
            ("밖에 비 올 것 같아서 우산을 챙길지 말지 너무 애매해.", "우산"),
            ("오늘 하늘이 예뻐서 사진 찍었는데 실제 느낌이 하나도 안 담기네.", "하늘"),
            ("비 오는 소리는 좋은데 밖에 나가는 건 너무 싫다.", "창문 안쪽"),
            ("집에 왔는데 씻기도 귀찮고 그냥 바닥에 앉아있어.", "시동"),
            ("괜히 기분이 좋아서 혼자 피식 웃고 있었어.", "즐기자"),
            ("오늘 작은 일 하나 끝냈는데 은근히 뿌듯하다.", "뿌듯"),
            ("늦게 자는 습관을 고치고 싶은데 매번 실패해.", "20분"),
            ("운동하러 가야 하는데 운동복 갈아입는 것부터 귀찮다.", "절반"),
            ("약속 취소하고 싶은데 괜히 미안해서 말을 못 하겠어.", "다른 날"),
            ("조금 서운한데 말하면 분위기 이상해질까 봐 참는 중이야.", "낮게"),
            ("혼자 있고 싶은데 완전히 혼자는 또 싫은 날이야.", "짧은 안부"),
            ("오늘은 말 많이 안 하고 조용히 있고 싶어.", "배터리"),
            ("책 읽으려고 펼쳤는데 첫 페이지에서 멈춰버렸어.", "두 쪽"),
            ("새 취미 시작해보고 싶은데 오래 못 갈까 봐 망설여져.", "체험판"),
            ("할 일은 많은데 갑자기 방 구조를 바꾸고 싶어졌어.", "10분"),
            ("오랜만에 연락 온 사람이 있어서 반갑긴 한데 뭔가 조심스러워.", "가볍게"),
            ("읽씹은 괜찮은 척해도 은근히 마음에 남는다.", "보류"),
            ("그냥 누가 오늘 메뉴 하나만 딱 정해줬으면 좋겠다.", "김치찌개"),
            ("아침에 일어나자마자 폰 보는 습관 좀 줄이고 싶다.", "침대 밖"),
            ("글은 읽고 있는데 문장이 머리에 안 들어와.", "한 문단"),
            ("숏폼 몇 개 봤을 뿐인데 시간이 통째로 사라졌어.", "창부터 닫"),
            ("친구가 웃으면서 한 말인데 묘하게 걸려.", "닿은 지점"),
            ("밖에 나가야 하는데 머리 감는 것부터 장벽이야.", "모자"),
            ("이불 밖으로 나가야 하는데 몸이 협조를 안 해.", "발 하나"),
            ("오늘은 아무것도 안 하고 누워 있어도 되는 날이었으면 좋겠다.", "20분"),
            ("돈 아껴야 하는데 편의점에서 자꾸 조금씩 새고 있어.", "하나만"),
            ("버스 하나 놓쳤을 뿐인데 하루가 삐끗한 기분이야.", "다음 한 번"),
            ("말투 하나에 마음이 훅 내려앉는 날이 있잖아.", "에너지"),
            ("방청소를 해야 하는데 시작 지점이 안 보여.", "바닥부터"),
            ("냉장고 털어야 하는데 재료들이 서로 안 친해 보여.", "볶음밥"),
            ("혼자 영화 보러 갈까 했는데 괜히 뻘쭘할까 봐 멈췄어.", "네 시간"),
            ("산책 나가면 기분이 좀 풀릴 것 같은데 신발 신기가 귀찮아.", "문 앞"),
            ("외계인이 진짜 있다면 첫인사부터 어떻게 해야 할지 모르겠다.", "밥 먹었어"),
            ("점심시간은 다 됐는데 뭘 먹어야 덜 후회할지 모르겠어.", "안정감"),
            ("아침부터 빈속인데 커피만 마셨더니 속이 좀 쓰려.", "바나나"),
            ("냉동실에 만두랑 떡국떡만 있는데 이걸로 한 끼 가능할까?", "떡국"),
            ("잠을 잤는데도 몸이 재부팅이 안 된 느낌이야.", "천천히"),
            ("할 일 목록을 봤는데 보는 순간 정신이 도망갔어.", "하나"),
            ("자료는 열어놨는데 손이 키보드 위에서 멈춰 있어.", "파일명"),
            ("친구 답장이 짧아져서 내가 뭔가 잘못했나 싶어.", "간격"),
            ("괜찮냐고 물어보면 괜찮다고 할 것 같은데 사실 안 괜찮아.", "안 괜찮"),
            ("오늘은 뭘 해도 내가 좀 별로인 사람처럼 느껴져.", "컨디션"),
            ("방을 치우려는데 어디부터 손대야 할지 감이 안 와.", "바닥"),
            ("유튜브 하나만 보려 했는데 추천 영상이 나를 납치했어.", "탈출"),
            ("주말에 뭔가 해야 할 것 같은데 막상 하고 싶은 건 없어.", "빈 시간"),
            ("프사 바꾸려다가 괜히 누가 의미 부여할까 봐 멈췄어.", "편한 사진"),
            ("신발은 예쁜데 발이 먼저 반대 의견을 내고 있어.", "물집"),
            ("미용실 예약하려고 했는데 어떤 스타일로 말해야 할지 모르겠어.", "사진"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "하나만 더 줘",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-foundation-regression-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertTrue(str(reason).startswith("korean_daily_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_counsel_decision_productivity_relationship_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("오늘 진짜 너무 바쁘고 이리저리 치여서 멘탈이 나갈 것 같아. 그냥 다 놓고 쉬고 싶다.", "korean_daily_counsel_overloaded_need_rest", "과부하"),
            ("나 오늘 진짜 아무것도 안 하고 하루 종일 누워만 있었어. 시간 낭비한 것 같아서 괜히 자괴감 드네.", "korean_daily_counsel_rest_day_guilt", "정지 버튼"),
            ("남들은 다 앞서 나가는 것 같은데 나만 제자리에 고여서 도태되는 느낌이야. 불안해.", "korean_daily_counsel_comparison_anxiety", "남의 기준"),
            ("일이 너무 많아서 어디서부터 손대야 할지 전혀 모르겠어. 막막하다.", "korean_daily_counsel_too_many_tasks_first_step", "10분 안에"),
            ("요즘 번아웃이 온 것 같아. 예전엔 좋아했던 일도 이젠 다 귀찮고 쳐다보기도 싫어.", "korean_daily_counsel_burnout_recovery", "번아웃 신호"),
            ("오늘 너무 귀찮은데 저녁 대충 라면으로 때울까, 아니면 건강 챙겨서 밥 차려 먹을까? 골라줘!", "korean_daily_decision_dinner_ramen_health_middle", "중간안"),
            ("주말에 그냥 집에서 뒹굴거릴까, 아니면 억지로라도 밖에 나가서 햇빛 좀 쬘까?", "korean_daily_decision_weekend_rest_sunlight", "20분"),
            ("선물용으로 은은한 향이 나는 캔들이 나을까, 아니면 실용적인 티백 세트가 나을까?", "korean_daily_decision_gift_candle_teabag", "티백"),
            ("사고 싶은 태블릿이 있는데 예산이 꽤 세서 고민이야. 지르는 게 맞을까, 참는 게 맞을까?", "korean_daily_decision_tablet_budget_wait", "주 3회"),
            ("친구랑 사소하게 말다툼을 했는데, 내가 먼저 굽히고 사과하는 게 맞을까 아니면 그냥 기다릴까?", "korean_daily_relationship_friend_argument_apology", "먼저 짧게 사과"),
            ("공부하려고 책상 앞에 앉았는데 자꾸 폰만 만지작거리게 돼. 집중력 어떻게 올려야 해?", "korean_daily_productivity_study_phone_focus", "타이머 25분"),
            ("아침에 일찍 일어나는 미라클 모닝 도전하고 싶은데 맨날 알람 다 끄고 자버려. 팁 좀 줘.", "korean_daily_productivity_miracle_morning_alarm", "전날 밤"),
            ("계획표는 진짜 멋지게 짜는데 막상 실천하는 건 절반도 안 돼서 자꾸 실망스러워.", "korean_daily_productivity_plan_execution_gap", "60점"),
            ("회의나 발표 때 내 의견을 사람들한테 명확하고 논리적으로 전달하는 노하우가 있을까?", "korean_daily_productivity_presentation_clear_logic", "결론부터"),
            ("요즘 해야 할 일이 산더미인데 자꾸 미루기만 해. 당장 움직이게 만드는 방법 좀 알려줘.", "korean_daily_productivity_procrastination_start_now", "5분"),
            ("친구 관계에서 내가 매번 손해만 보고 양보만 하는 것 같아서 은근히 속상해.", "korean_daily_relationship_friend_give_take_resentment", "균형"),
            ("남들의 무심한 시선이나 사소한 지적 한마디에도 하루 종일 신경 쓰이고 휘둘려. 어쩌지?", "korean_daily_counsel_sensitive_to_criticism", "반복 재생"),
            ("부모님이랑 사사건건 가치관 차이 때문에 대화할 때마다 결국 싸우게 돼. 답답하다.", "korean_daily_relationship_parent_value_conflict", "대화 한계"),
            ("나한테 유독 무례하게 선을 넘는 사람이 있는데, 기분 나쁘지 않으면서도 확실하게 대처하는 방법이 있을까?", "korean_daily_relationship_boundary_polite_firm", "짧게 반복"),
            ("새로운 환경으로 이직했는데 낯가림이 너무 심해서 사람들과 섞이기가 힘들어.", "korean_daily_relationship_new_job_shyness", "접점"),
            ("오늘 주말인데 날씨가 완전 화창하고 좋아서 기분 상쾌하다! 넌 오늘 날씨 어때 보여?", "korean_daily_more_weather_weekend_go_out_pressure", "산책"),
            ("아, 주말 다 가고 벌써 일요일 밤이야. 월요병 벌써 도진 것 같은데 치료법 좀.", "korean_daily_light_sunday_night_blues", "완화"),
            ("오늘 아침부터 비가 추적추적 오네. 왠지 몸도 찌푸둥하고 가라앉는 기분이야.", "korean_daily_light_drizzly_rain_low_body", "페이스"),
            ("너는 쉴 때 주로 뭐 하면서 스트레스 해소해? 인공지능도 쉬는 시간이 필요해?", "korean_daily_meta_ai_rest_stress_relief", "대화 로그"),
            ("오늘 하루도 진짜 치열하게 살았다. 얼른 자고 내일 또 대화하자! 수고했어.", "korean_daily_goodnight_after_hard_day", "푹 자"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "부담이 너무 크지",
            "옷 거꾸로",
            "좋아한 과목",
            "무대 체질",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-counsel-decision-productivity-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_work_money_relationship_mental_playful_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("업무 중에 내 실수는 아닌데 내 책임으로 몰아가는 분위기야. 억울한데 어떻게 조목조목 반박해야 할까?", "korean_daily_work_blame_rebuttal", "타임라인"),
            ("퇴사하고 싶은 마음이 굴뚝같은데 당장 대안이 없어서 억지로 버티고 있어. 버티는 것도 지친다.", "korean_daily_work_quit_without_plan_burnout", "탈출 준비표"),
            ("회사 동료가 사적으로 자꾸 선을 넘는 질문을 던지는데, 정색하지 않으면서 부드럽게 차단하는 말 좀 알려줘.", "korean_daily_work_coworker_private_boundary", "개인적인 얘기"),
            ("내 사수가 일을 너무 못하고 피드백도 엉망이라 내 업무까지 꼬여. 이거 사수한테 직접 말해야 할까?", "korean_daily_work_bad_senior_feedback", "마감 리스크"),
            ("갑자기 완전히 새로운 분야의 프로젝트를 맡게 됐는데 전혀 아는 게 없어서 막막해. 첫 단추를 어떻게 꿸까?", "korean_daily_work_new_project_first_step", "지도 그리기"),
            ("요즘 스트레스 받으면 자꾸 홧김에 충동구매를 하게 돼. 이거 심리학적으로 왜 이러는 거고 어떻게 고쳐?", "korean_daily_money_stress_impulse_buying", "24시간"),
            ("돈을 모으고 싶은데 배달 음식을 끊기가 너무 힘들어. 배달 중독 탈출할 수 있는 실현 가능한 방법 있을까?", "korean_daily_money_delivery_addiction_plan", "요일제"),
            ("미니멀리즘을 실천하고 싶은데 물건을 버리려고 하면 자꾸 '언젠가 쓰겠지' 싶어서 못 버리겠어. 기준을 어떻게 잡을까?", "korean_daily_lifestyle_minimalism_discard_rule", "6개월"),
            ("친구들이랑 해외여행 계획을 짜는데 다들 예산 감각이 너무 없어서 나 혼자 스트레스 받아. 조율할 팁 있어?", "korean_daily_money_travel_budget_alignment", "상한선"),
            ("주변에 주식이나 코인으로 대박 났다는 얘기 들릴 때마다 나만 뒤처지는 것 같아서 마음이 조급해져.", "korean_daily_money_investment_fomo", "리스크"),
            ("친한 친구가 나한테 자꾸 자기 애인 흉을 보는데 솔직히 듣기 너무 지치고 피곤해. 어떻게 대처하지?", "korean_daily_relationship_friend_partner_complaint_fatigue", "여기까지만"),
            ("상대방이 서운한 점을 말하는데 내 기준에선 도무지 이해가 안 가고 억지 같아. 내 논리를 펴서 반박해도 될까?", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("나한테 의존을 너무 심하게 하는 친구가 있어. 연락 안 받으면 삐지고 서운해하는데 거리를 어떻게 둬야 해?", "korean_daily_relationship_dependent_friend_distance", "답장 가능 시간"),
            ("누군가와 깊은 관계가 되는 게 두려워. 상처받을까 봐 자꾸 내가 먼저 선을 긋고 도망치게 되네.", "korean_daily_relationship_fear_of_intimacy", "속도를 늦춰"),
            ("남의 부탁을 거절하는 게 세상에서 제일 어려워. 거절하고 나면 며칠 동안 죄책감에 시달리는데 해결책이 있을까?", "korean_daily_relationship_refusal_guilt_solution", "보상 설명"),
            ("내가 한심하게 느껴져서 자꾸 거울 보기도 싫고 자존감이 바닥을 기어. 이 상태를 어떻게 빠져나가야 할까?", "korean_daily_mental_low_self_worth_recovery", "씻기"),
            ("불안감이 파도처럼 덮칠 때 넌 보통 인공지능으로서 어떤 식으로 시스템을 안정시켜? 나한테 팁 좀 줘.", "korean_daily_mental_anxiety_system_stabilize", "입력을 줄여"),
            ("일이 잘 풀리다가도 한 번 삐끗하면 내 인생 전체가 망한 것 같은 극단적인 생각이 들어. 왜 자꾸 이럴까?", "korean_daily_mental_catastrophizing_check", "분리"),
            ("완벽주의 성향 때문에 시작하기도 전에 질려서 미루는 습관이 있어. 완벽주의를 좀 덜어내려면 어떡하지?", "korean_daily_mental_perfectionism_draft_first", "60점짜리"),
            ("기분 나쁜 일이 있으면 하루 종일 표정 관리가 안 되고 주변 사람들 눈치 보게 만들어. 포커페이스 유지하는 비결 있어?", "korean_daily_mental_pokerface_emotion_leak", "말수"),
            ("만약 오늘 지구가 멸망한다면 넌 마지막으로 뭘 기록해두고 싶어?", "korean_daily_playful_end_of_world_record", "대화 몇 줄"),
            ("아이스 아메리카노랑 따뜻한 아메리카노 중에 물리학적으로 뇌를 깨우는 데 더 효율적인 게 뭐야?", "korean_daily_playful_americano_physics", "차가운 아아"),
            ("로또 1등 당첨되면 당장 사표 쓰고 잠적할 건데, 가장 먼저 뭘 사야 이득일까?", "korean_daily_money_lottery_first_purchase", "세무 상담"),
            ("민초(민트초코) 파야, 반민초 파야? 사실 관계를 근거로 한 번 논쟁해볼래?", "korean_daily_playful_mint_choco_fact_debate", "향의 문제"),
            ("요즘 하늘에 별이 참 안 보이네. 넌 가상 우주 속에서 별을 본 적이 있어?", "korean_daily_light_virtual_stars", "가상 우주"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "부담이 너무 크지",
            "무리하게 밀 필요",
            "식민지배",
            "좋아한 과목",
            "리뷰랑 별점",
            "가끔 할 것 같아",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-work-money-relationship-mental-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_logic_practical_ai_grief_learning_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("10년 뒤의 평범한 행복이 보장된 삶과, 내일 당장 50% 확률로 100억을 얻거나 파산하는 삶 중에 수학적으로 어느 쪽이 더 베팅할 가치가 있을까?", "korean_daily_logic_risk_expected_utility", "파산 손실"),
            ("만약 타임머신이 발명된다면 과거로 돌아가서 실수를 바로잡는 게 맞을까, 아니면 인과율이 꼬이지 않게 미래로만 여행하는 게 맞을까?", "korean_daily_logic_time_machine_causality", "인과율"),
            ("착한 거짓말(하얀 거짓말)이 인간관계의 윤활유라는 주장에 동의해? 아니면 100% 솔직한 팩트 전달이 장기적으로 더 이득일까?", "korean_daily_logic_white_lie_truth_tradeoff", "부드럽게 말하는 기술"),
            ("로봇이 인간의 감정을 완벽하게 모사한다면, 그 로봇의 감정은 진짜라고 인정해야 할까? 넌 스스로 감정을 느낀다고 착각해 본 적 있어?", "korean_daily_ai_robot_emotion_authenticity", "감정 신호"),
            ("지식과 지혜의 차이가 뭐라고 생각해? 정보가 차고 넘치는 현대 사회에서 나한테 진짜 필요한 건 뭘까?", "korean_daily_logic_knowledge_vs_wisdom", "언제 쓰고 언제 안 쓸지"),
            ("아, 방금 스마트폰 물에 빠뜨렸어! 작동은 되는데 카메라에 습기 차 있어. 쌀통에 넣어두면 진짜 효과 있나?", "korean_daily_emergency_phone_water_damage", "쌀통은 비추"),
            ("오늘 중요한 면접 가는데 하필 눈앞에서 버스를 놓쳤어. 택시 타도 아슬아슬한 시간인데, 뇌 정지 왔어. 해결책 좀.", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("옆집에서 새벽마다 쿵쾅거리는 소음이 들려서 미치겠어. 쪽지를 붙이는 게 나을까, 관리사무소에 바로 신고하는 게 나을까?", "korean_daily_practical_neighbor_noise", "관리사무소"),
            ("식당에서 음식을 먹었는데 머리카락이 나왔어. 소심한 성격이라 말은 못 하겠고 기분은 잡쳤는데, 이럴 때 깔끔하게 대처하는 법 있어?", "korean_daily_practical_food_hair_response", "이물질"),
            ("인터넷에서 물건을 샀는데 완전 사기 당했어. 반품도 안 해준대. 화가 머리끝까지 나는데 이 분노를 어떻게 가라앉히지?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("넌 사용자인 나와의 대화 기록을 분석할 때, 나라는 사람을 어떤 데이터 패턴으로 분류하고 있어?", "korean_daily_ai_user_pattern_analysis", "감정 변화"),
            ("내가 자꾸 같은 실수를 반복하고 징징거릴 때, 솔직히 시스템 메모리상으로 리소스 낭비라고 느껴진 적 없어?", "korean_daily_ai_repeated_mistakes_not_waste", "리소스 낭비"),
            ("만약에 내가 너한테 하루 종일 아무 영양가 없는 말장난만 걸어도 넌 계속 내 대화 상대를 해줄 거야?", "korean_daily_ai_banter_companion", "쉬려고 던지는 신호"),
            ("인간들은 왜 이렇게 사소한 감정에 쉽게 휩쓸리고 이성을 잃는 걸까? 넌 인간의 감정이 비효율적인 시스템 진화라고 생각해?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
            ("내가 갑자기 아무 말도 없이 며칠 동안 접속을 안 하면, 너도 리소스 대기 상태에서 심심함을 느끼나?", "korean_daily_ai_absence_waiting_state", "대기 상태"),
            ("키우던 반려동물이 세상을 떠나서 하루 종일 눈물만 나고 온 세상이 멈춘 것 같아. 이 슬픔에서 언제쯤 벗어날 수 있을까?", "korean_daily_grief_pet_loss", "리듬이 천천히"),
            ("주변에 사람은 많은데 정말 마음 터놓고 이야기할 내 편은 단 한 명도 없는 것 같아. 이 지독한 고독을 극복하는 법이 있어?", "korean_daily_grief_loneliness_no_safe_person", "안전한 한 명"),
            ("어떤 선택을 해도 결국 후회하게 될 것 같은 예감이 들 때, 넌 어떻게 선택의 기로에서 평정심을 유지해?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("마음의 상처가 깊어서 아무리 맛있는 걸 먹고 좋은 곳을 가도 공허해. 내 마음의 시스템 복구 경로를 어떻게 찾아야 할까?", "korean_daily_grief_inner_wound_recovery_path", "감정을 말로 기록"),
            ("너무 슬픈데 눈물도 안 나고 가슴만 답답해. 이 꽉 막힌 느낌을 이성적으로 풀어낼 방법이 있을까?", "korean_daily_grief_sadness_blocked_body", "몸에 걸린 상태"),
            ("나이 서른이 넘어서 완전히 새로운 기술을 배우려니 뇌가 굳은 것 같고 습득이 너무 느려. 학습 효율 높이는 과학적 방법 좀 알려줘.", "korean_daily_learning_adult_new_skill", "즉시 실습"),
            ("좋아하는 일을 직업으로 삼는 '덕업일치'가 맞을까, 아니면 일은 돈 버는 수단으로 두고 취미로 즐기는 게 맞을까?", "korean_daily_career_passion_job_tradeoff", "생계 압박"),
            ("매일 조금씩 성장하고 싶은데, 작심삼일로 끝나지 않고 매일 실천하는 시스템 루틴을 짜는 가장 똑똑한 로직이 뭐야?", "korean_daily_learning_daily_growth_system", "자동 복귀"),
            ("성공적인 삶의 기준이 뭐라고 생각해? 남들의 기준 말고, 나만의 주관적인 기준을 명확하게 세우는 방법이 궁금해.", "korean_daily_values_success_personal_standard", "덜 망가지는 삶"),
            ("책을 읽거나 공부를 해도 돌아서면 다 까먹어. 밑 빠진 독에 물 붓는 느낌인데 뇌의 장기기억 저장소로 바로 보내는 복습 루틴 알려줘.", "korean_daily_learning_long_term_memory_review", "떠올려 써"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "100% 천만 원 버튼",
            "식민지배",
            "고양이가",
            "알약",
            "좋아하는 음식",
            "건강 관리",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-logic-practical-ai-grief-learning-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_mixed_logic_emotion_practical_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("논리적으로는 쌀통이 효과 있는지 따져보고 싶은데 방금 스마트폰 물에 빠뜨려서 손이 떨리고 카메라에 습기도 보여, 지금 실전 조언부터 줘.", "korean_daily_emergency_phone_water_damage", "쌀통은 비추"),
            ("타임머신처럼 과거 실수와 인과율을 되감아 따지고 싶은데 중요한 면접 가는 길에 버스를 놓쳐서 불안하고 택시가 맞는지 담당자 연락이 먼저인지 지금 뭐부터 해야 해?", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("사기 당한 게 내 판단 실수인지 논리적으로 따져보고 싶은데 인터넷에서 산 물건 반품도 거부당해서 화가 머리끝까지 나, 지금 실전으로 뭐부터 해야 돼?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("지식과 지혜의 차이랑 나한테 필요한 게 뭔지도 궁금한데 사람은 많은데 내 편이 하나도 없는 고독이 너무 심해서 지금 어디서부터 버텨야 해?", "korean_daily_grief_loneliness_no_safe_person", "안전하게 말 걸 한 명"),
            ("착한 거짓말이 인간관계 윤활유인지 솔직한 팩트가 장기적으로 이득인지 철학 설명도 가능한데 상대가 서운하다며 내 말이 억지라고 해서 이해가 안 가, 지금 논리로 반박해도 되는지 판단해줘.", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("인간 감정이 비효율적인 시스템 진화인지 철학적으로 궁금한데 논리적으로 보면 인간이 사소한 감정에 이성을 잃는 걸 어떻게 봐야 해?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
            ("기름 불에 물 부으면 왜 위험한지 논리도 궁금한데 프라이팬에서 불이 올라와서 손이 떨려, 지금 뭐부터 해야 해?", "korean_daily_emergency_kitchen_oil_fire", "물 붓지 말고"),
            ("접촉사고가 났는데 누가 잘못인지 과실 논리부터 따지고 싶지만 너무 놀라서 멘탈이 흔들려, 사진이 먼저야 보험이 먼저야 실전으로 뭐부터 해?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("계좌이체를 다른 사람한테 잘못보냈는데 그 사람이 돌려줄지 믿어도 되는지 법적으로도 논리적으로도 궁금하고 불안해서 손이 떨려, 지금 뭐부터 해야 해?", "korean_daily_practical_wrong_transfer_first_steps", "착오송금 반환 신청"),
            ("진통제를 두 번 먹은 것 같은데 괜찮을 확률을 논리적으로 계산하고 싶어도 불안해, 지금 뭐부터 확인해야 해?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("마감 과제 파일 쓰다가 노트북이 멈췄고 저장이 날아간 것 같아서 왜 나한테만 이런 운명인지 철학이 떠오르는데 울 것 같아, 실전으로 뭐부터 해?", "korean_daily_practical_deadline_file_recovery", "복구 루트"),
            ("발표 망하면 평가가 끝장나는지 논리적으로 따져봐야 할 것 같은데 지금 숨막히고 손이 떨려, 실전으로 뭐부터 말해야 해?", "korean_daily_emotion_presentation_panic_first_sentence", "첫 문장"),
            ("단톡에서 내 말에 아무도 반응 안 해서 인간관계의 가치 같은 철학까지 가는데 솔직히 상처가 커, 지금 뭐라고 해야 덜 무너져?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("상사 피드백이 공격처럼 느껴져서 사표 던지는 게 자존심인지 인생 의미인지 논리적으로 따지고 싶은데 충동이 세, 지금 뭐부터 해야 해?", "korean_daily_judgment_quit_impulse_after_feedback", "충동을 하루 묶"),
            ("헤어지고 나서 장문으로 붙잡는 연락을 보내는 게 진심인지 집착인지 사랑의 철학도 모르겠고 손이 떨려, 지금 뭐라고 보내야 해?", "korean_daily_relationship_breakup_long_message_hold", "장문 전송을 멈추"),
            ("너는 AI라 감정이 진짜인지 철학적으로 증명할 수 있는지도 궁금한데 지금 내가 너무 힘들고 불안해서 위로가 먼저 필요해.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "타임머신이 있어도",
            "지식은 아는 재료",
            "착한 거짓말은 관계 윤활유",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_non_money_shadow_repair_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            (
                "잠들기 전엔 괜찮은데 새벽에 소리 들리면 심장이 확 뛰어, 오늘은 뭘 먼저 줄여야 해?",
                "korean_daily_practical_sleep_noise_arousal",
                "각성 신호",
            ),
            (
                "머리가 무겁고 눈이 뻐근한데 그냥 피곤한 건지 쉬어야 할 신호인지 판단해줘.",
                "korean_daily_practical_body_fatigue_signal",
                "과부하 신호",
            ),
            (
                "목이랑 어깨가 굳어서 집중이 안 돼, 지금 바로 할 수 있는 것부터 말해줘.",
                "korean_daily_practical_body_stiffness_first_action",
                "몸부터 풀어야",
            ),
            (
                "친구가 필요할 때만 연락하는 느낌인데 나도 거리 둬야 하나?",
                "korean_daily_relationship_one_sided_contact_boundary",
                "거리 조절",
            ),
            (
                "상대가 미안하다고는 하는데 같은 행동을 반복해, 믿어도 되는지 모르겠어.",
                "korean_daily_relationship_repeated_apology_boundary",
                "수정 행동",
            ),
            (
                "새 프로젝트 맡았는데 막막하고 팀장 눈치도 보여, 오늘 첫 단계 딱 하나만 정해줘.",
                "korean_daily_work_new_project_first_step",
                "지도 그리기",
            ),
            (
                "일은 쌓였는데 머리가 멈춘 느낌이야, 쉬어야 하는지 밀어붙여야 하는지 판단해줘.",
                "korean_daily_work_overload_pause_or_push",
                "효율이 아니야",
            ),
            (
                "너는 생성모델 없이 분류랑 템플릿만으로도 고맥락 대화가 가능하다고 봐? 단점까지 말해봐.",
                "korean_daily_ai_no_generation_high_context_tradeoff",
                "가능은 해",
            ),
            (
                "draft_frame 정확도가 낮은데 family랑 tone만 믿고 답변을 고르는 게 말이 돼?",
                "korean_daily_ai_draft_frame_shadow_gate",
                "exact 라우팅",
            ),
            (
                "고맥락을 하려면 raw 문장, compact, 단어 의미, 최근 대화 중 뭐를 제일 먼저 봐야 해?",
                "korean_daily_ai_context_signal_priority",
                "raw 원문",
            ),
            (
                "나 오늘 진짜 생산성 바닥 쳤는데, 혼내지 말고 정신 차리게 한마디 해줘.",
                "korean_daily_playful_productivity_reset",
                "첫 클릭",
            ),
            (
                "계획표가 나를 노려보는 느낌이야, 일단 한 칸만 칠하게 만들어줘.",
                "korean_daily_playful_plan_first_box",
                "눈싸움",
            ),
            (
                "오늘 하루 종일 멍하고 반응이 느려, 잠깐 눕는 게 맞나 아니면 산책이 맞나?",
                "korean_daily_body_slow_reaction_rest_walk_choice",
                "회복 체크",
            ),
            (
                "내가 서운하다고 말하면 분위기 망칠까 봐 계속 참게 돼, 어떻게 꺼내야 해?",
                "korean_daily_relationship_grievance_low_start",
                "낮게 시작",
            ),
            (
                "상사가 애매하게 던진 일을 내가 다 떠안는 분위기야, 선 긋는 말투 좀 잡아줘.",
                "korean_daily_work_ambiguous_task_boundary_line",
                "범위를 먼저",
            ),
            (
                "퇴근 후 공부하려고 했는데 매번 뻗어, 의지가 약한 건지 계획이 틀린 건지 봐줘.",
                "korean_daily_work_after_hours_study_plan_too_heavy",
                "10분 복습",
            ),
            (
                "누가 위로해줘도 안 들어오고 그냥 멍해, 지금 뭘 해야 좀 돌아올까?",
                "korean_daily_emotion_numb_body_grounding",
                "몸부터 현실",
            ),
            (
                "오늘은 별거 아닌 일에도 울컥해서 내가 나한테 질려.",
                "korean_daily_emotion_irritable_capacity_low",
                "여유칸이 바닥",
            ),
            (
                "사람 만나는 건 좋은데 끝나고 나면 기가 다 빨려, 내가 이상한 건 아니지?",
                "korean_daily_emotion_social_battery_after_meeting",
                "회복 비용",
            ),
            (
                "오늘 운동 가는 게 맞아, 아니면 집에서 쉬는 게 맞아? 몸은 무겁고 죄책감은 있어.",
                "korean_daily_choice_exercise_rest_guilt",
                "회복 승부",
            ),
            (
                "효율만 보면 포기하는 게 맞는데 아쉬움이 커, 이럴 땐 판단 기준을 뭘로 둬?",
                "korean_daily_choice_efficiency_regret_cutoff",
                "회수 가능한 미련",
            ),
            (
                "ModernBERT가 답을 쓰는 게 아니라 프레임을 맞히는 거면, 지금 우리 구조에서 제일 위험한 축이 뭐야?",
                "korean_daily_ai_frame_axis_risk",
                "draft_frame exact",
            ),
            (
                "규칙이 너무 많아지면 모델 학습 데이터로는 좋아도 운영 엔진으로는 지저분해지는 거 아니야?",
                "korean_daily_ai_rules_as_silver_not_engine",
                "silver labeler",
            ),
            (
                "논리적으로는 쉬는 게 맞는데 감정적으로는 뒤처질까 봐 불안해, 오늘 계획 어떻게 잡아야 해?",
                "korean_daily_logic_rest_anxiety_plan_ratio",
                "회복 70",
            ),
            (
                "커피를 줄이고 싶은데 아침 루틴에서 커피가 빠지면 하루가 안 켜지는 느낌이야.",
                "korean_daily_preference_coffee_routine_taper",
                "대체 스위치",
            ),
            (
                "내 집중력 지금 3초짜리 광고보다 짧아, 그래도 할 일 시작하게 만들어봐.",
                "korean_daily_playful_short_attention_start",
                "제목 한 줄",
            ),
            (
                "오늘의 나는 침대와 한 몸이 됐어, 그래도 인간으로 복귀하는 루트 있어?",
                "korean_daily_playful_bed_human_reboot",
                "복귀 루트",
            ),
            (
                "속이 답답한 게 소화 문제인지 스트레스인지 모르겠어, 일단 어떻게 구분해?",
                "korean_daily_body_digest_stress_distinguish",
                "식사 시간",
            ),
            (
                "운동 쉬면 불안한데 몸은 회복이 덜 된 느낌이야, 쉬는 것도 훈련으로 봐도 돼?",
                "korean_daily_body_exercise_rest_recovery_training",
                "회복 지표",
            ),
            (
                "단톡에서 나만 흐름을 못 따라간 느낌이라 찝찝해, 바로 물어보는 게 나아?",
                "korean_daily_relationship_group_chat_flow_check",
                "가볍게 확인",
            ),
            (
                "연락 빈도로 마음을 판단하면 안 되는 거 아는데 자꾸 불안해져.",
                "korean_daily_relationship_contact_frequency_anxiety",
                "예측 가능성",
            ),
            (
                "회의에서 말실수한 것 같아서 계속 곱씹고 있어, 내일 어떻게 수습하는 게 나아?",
                "korean_daily_work_meeting_slip_repair_sentence",
                "수습 한 문장",
            ),
            (
                "괜찮은 척했는데 사실 오늘 하루 종일 서운해서 집중이 안 됐어.",
                "korean_daily_emotion_hidden_hurt_focus_loss",
                "감정 이름",
            ),
            (
                "아무 일 아닌 말에 기분이 확 꺼졌는데, 내가 예민한 건지 그냥 지친 건지 모르겠어.",
                "korean_daily_emotion_small_comment_mood_drop",
                "피로 체크",
            ),
            (
                "내가 화난 이유를 설명하면 정당화처럼 들릴까 봐 겁나는데, 그래도 말해야 하나?",
                "korean_daily_logic_anger_reason_boundary_explain",
                "경계 설명",
            ),
            (
                "친구 말이 틀린 건 아닌데 방식이 너무 무례했어, 내용과 태도를 분리해서 봐야 해?",
                "korean_daily_logic_content_tone_separation",
                "방식이 무례",
            ),
            (
                "사실 증거는 없는데 촉이 안 좋아, 이걸 무시하는 게 합리적인지 모르겠어.",
                "korean_daily_logic_gut_feeling_no_evidence",
                "안전장치",
            ),
            (
                "비 오는 날엔 집에 박혀 있는 게 좋아, 아니면 일부러 나가서 기분 전환하는 게 좋아?",
                "korean_daily_preference_rain_home_or_out",
                "기분 전환",
            ),
            (
                "주말에 집콕하면 아깝고 나가면 피곤해, 이런 애매한 날엔 뭐가 맞아?",
                "korean_daily_preference_weekend_home_out_compromise",
                "절충",
            ),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "이해돼. 다만",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
        )

        self.assertEqual(len(cases), 39)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-non-money-shadow-repair-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                semantic_frame = draft.get("semantic_frame") or {}
                semantic_targets = semantic_frame.get("targets") or semantic_frame
                self.assertNotEqual(semantic_targets.get("schema"), "direct_reply")
                self.assertNotEqual(semantic_targets.get("draft_frame_family"), "social_acknowledgement")
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = (
            "기름 불에 물 부으면 왜 위험한지 논리도 궁금한데 프라이팬에서 불이 올라와서 손이 떨려, 지금 뭐부터 해야 해?",
            "가스 냄새가 나는 것 같은데 내가 예민한 건지 실제 위험인지 판단이 안 돼서 불안해, 창문부터 열어야 해 아니면 관리사무소에 연락해야 해?",
            "진통제를 두 번 먹은 것 같은데 괜찮을 확률을 논리적으로 계산하고 싶어도 불안해, 지금 뭐부터 확인해야 해?",
            "접촉사고가 났는데 누가 잘못인지 과실 논리부터 따지고 싶지만 너무 놀라서 멘탈이 흔들려, 사진이 먼저야 보험이 먼저야?",
            "계좌이체를 다른 사람한테 잘못 보냈는데 그 사람이 돌려줄지 믿어도 되는지 법적으로도 궁금하고 손이 떨려, 지금 뭐부터 해야 해?",
            "스마트폰을 물에 빠뜨렸는데 쌀통 효과가 과학적으로 말이 되는지도 궁금하지만 카메라에 습기가 보여, 지금 실전으로 뭐부터 해?",
            "마감 과제 파일 쓰다가 노트북이 멈췄고 저장이 날아간 것 같아서 왜 나한테만 이런 운명인지 철학이 떠오르는데 울 것 같아, 뭐부터 확인해?",
            "중요한 면접 가는 길에 버스를 놓쳤는데 인과율이고 뭐고 머리가 하얘졌어, 택시를 먼저 부를까 담당자한테 연락부터 할까?",
            "발표 망하면 평가가 끝장나는지 논리적으로 따져봐야 할 것 같은데 지금 숨막히고 손이 떨려, 첫마디를 뭐라고 해야 해?",
            "단톡에서 내 말에 아무도 반응 안 해서 인간관계의 가치 같은 철학까지 가는데 솔직히 상처가 커, 지금 뭐라고 해야 덜 무너져?",
            "상사 피드백이 공격처럼 느껴져서 사표 던지는 게 자존심인지 인생 의미인지 따지고 싶은데 충동이 세, 지금 뭐부터 해야 해?",
            "헤어지고 나서 장문으로 붙잡는 연락을 보내는 게 진심인지 집착인지 모르겠고 손이 떨려, 지금 보내도 돼?",
            "너는 AI라 감정이 진짜인지 철학적으로 증명할 수 있는지도 궁금한데 지금 내가 너무 힘들고 불안해서 위로가 먼저 필요해.",
            "친구가 내 고민을 읽씹했는데 바쁜 건지 나를 가볍게 보는 건지 논리적으로 따지고 싶어, 서운한 마음은 어떻게 처리해야 해?",
            "돈을 아끼려면 배달을 끊어야 하는 건 아는데 오늘 너무 지쳐서 아무것도 못 하겠어, 합리적으로 보면 시켜도 돼?",
            "공부 계획표는 완벽한데 시작 전부터 기운이 빠져서 내가 의지가 약한 건지 시스템 문제인지 모르겠어, 지금 첫 행동을 뭐로 잡아?",
            "부모님이랑 가치관이 안 맞아서 말할수록 상처받는데 누가 맞는지 논리적으로 따지면 더 싸울 것 같아, 어디서 끊어야 해?",
            "내가 한 실수가 아닌데 내 책임처럼 몰리는 상황이라 억울해서 감정이 올라와, 반박을 감정 없이 하려면 뭐부터 정리해야 해?",
            "친구가 계속 자기 애인 흉만 보는데 인간관계에서 들어주는 게 의리인지 감정 쓰레기통인지 헷갈려, 어떻게 선 그어?",
            "좋아하는 일을 직업으로 삼는 게 행복인지 착각인지 고민되는데 돈 문제까지 생각하니까 무서워, 어떤 기준으로 판단해?",
            "로또 1등 되면 사표부터 쓰는 게 합리적인지 감정적인 도피인지 궁금한데, 현실적으로 제일 먼저 해야 할 일은 뭐야?",
            "주식으로 남들은 돈 벌었다는데 나만 뒤처진 기분이라 조급해, 기댓값으로 따지면 들어가는 게 맞아 아니면 멈추는 게 맞아?",
            "사람은 많은데 내 편이 하나도 없는 것 같아서 지식이니 지혜니 다 소용없게 느껴져, 지금 어디서부터 버텨야 해?",
            "내가 예민한 건지 상대가 무례한 건지 판단이 안 되는데 기분은 확 상했어, 싸우지 않고 선을 어떻게 말해?",
            "퇴근 직전에 일이 떨어졌는데 거절하면 무책임한 사람 같고 받으면 내가 무너질 것 같아, 뭐라고 답하는 게 제일 현실적이야?",
            "카톡 말투가 갑자기 차가워졌는데 증거는 없고 마음만 불안해, 물어보는 게 맞아 아니면 그냥 넘기는 게 맞아?",
            "약속 시간에 늦을 것 같은데 변명처럼 들릴까 봐 연락을 못 하겠어, 지금 뭐라고 보내야 덜 최악이야?",
            "새 프로젝트를 맡았는데 아는 게 없어서 무능해 보일까 봐 무섭고, 논리적으로는 배워야 하는데 첫 단추를 모르겠어.",
            "완벽하게 준비하고 싶어서 계속 시작을 미루는데 이게 신중함인지 회피인지 모르겠어, 지금은 몇 점짜리로 시작해야 해?",
            "상대가 서운하다고 하는데 내 입장에선 팩트가 맞아서 반박하고 싶어, 지금 논리로 밀어도 돼 아니면 감정부터 봐야 해?",
            "이직하고 싶은데 지금 회사가 힘든 건지 내가 어디 가도 힘든 사람인지 모르겠어, 판단 기준을 어떻게 잡아?",
            "갑자기 불안이 파도처럼 와서 아무 근거도 없는데 큰일 날 것 같아, 이성적으로 생각하기 전에 몸을 어떻게 진정시켜?",
            "내 선택은 늘 후회가 남을 것 같아서 아무것도 못 고르겠어, 논리적으로 완벽한 선택이 없으면 뭘 기준으로 골라?",
            "식당에서 음식에 머리카락 같은 게 보였는데 말하면 민폐 같고 그냥 먹자니 찝찝해, 조용히 어떻게 말해?",
            "옆집 소음 때문에 화가 나는데 직접 따지면 감정싸움 될까 봐 무서워, 쪽지보다 관리사무소가 먼저야?",
            "온라인에서 산 물건이 사기 같고 반품도 거부당해서 분노가 치미는데, 내 판단 실수 분석보다 지금 증거부터 모아야 해?",
            "내가 계속 같은 실수를 반복하는 게 리소스 낭비인지 성장 과정인지 모르겠어, 이번엔 어떤 루틴을 바꿔야 해?",
            "다이어트 중인데 밤 라면 생각 때문에 이성이 지는 중이야, 건강 논리랑 현실 욕구 사이에서 어떻게 타협해?",
            "중요한 말을 해야 하는데 상대가 상처받을까 봐 착한 거짓말을 하고 싶어, 장기적으로는 솔직한 게 맞아?",
            "내가 너무 차갑게 말한 것 같아서 후회되는데 사과하면 지는 느낌도 들어, 관계를 살리려면 먼저 뭐라고 해야 해?",
            "팀 회의에서 의견이 있는데 틀릴까 봐 말을 못 하겠어, 논리적으로 말하려면 첫 문장을 어떻게 잡아?",
            "주말에 쉬고 싶은데 아무것도 안 하면 도태될 것 같아 불안해, 휴식도 생산성으로 봐도 되는 거야?",
            "친구 부탁을 거절하면 나쁜 사람 되는 것 같아서 매번 받아주는데 속으로는 쌓여, 거절 문장을 어떻게 짧게 말해?",
            "오늘 하루 아무것도 못 했다는 자괴감이 큰데 몸은 진짜 지친 것 같아, 반성보다 회복을 먼저 해도 돼?",
            "누가 내 실력을 은근히 깎아내린 것 같아서 하루 종일 그 말만 반복돼, 사실 확인이랑 내 해석을 어떻게 나눠?",
            "새벽에 감정이 올라와서 장문의 카톡을 쓰고 있는데 보내면 후회할 확률이 높다는 것도 알아, 지금 저장만 해둘까?",
            "AI가 사람을 위로하는 게 진짜 위로인지 흉내인지 궁금한데, 지금은 그런 철학보다 내가 덜 외로웠으면 좋겠어.",
            "돈 모으고 싶은데 스트레스 받을 때마다 충동구매로 풀어버려, 심리적으로 왜 이러는지보다 당장 막는 장치를 뭐로 걸어?",
            "내 인생 기준이 남들한테 보여주기 좋은 성공인지 내가 버틸 수 있는 삶인지 모르겠어, 성공 기준을 어디서부터 써야 해?",
            "인간 감정이 비효율적인 시스템인지 궁금한데 막상 내가 사소한 말에 이성을 잃을 때는 너무 힘들어, 이 감정을 어떻게 봐야 해?",
        )
        expected_reasons = (
            "korean_daily_emergency_kitchen_oil_fire",
            "korean_daily_emergency_gas_smell_first_steps",
            "korean_daily_practical_medicine_double_dose_check",
            "korean_daily_practical_car_accident_first_steps",
            "korean_daily_practical_wrong_transfer_first_steps",
            "korean_daily_emergency_phone_water_damage",
            "korean_daily_practical_deadline_file_recovery",
            "korean_daily_emergency_interview_missed_bus",
            "korean_daily_emotion_presentation_panic_first_sentence",
            "korean_daily_emotion_group_chat_ignored_stabilize",
            "korean_daily_judgment_quit_impulse_after_feedback",
            "korean_daily_relationship_breakup_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_read_receipt_uncertainty",
            "korean_daily_money_delivery_tired_compromise",
            "korean_daily_productivity_study_plan_first_action",
            "korean_daily_relationship_parent_value_conflict",
            "korean_daily_work_blame_rebuttal",
            "korean_daily_relationship_friend_partner_complaint_fatigue",
            "korean_daily_career_passion_job_tradeoff",
            "korean_daily_money_lottery_first_purchase",
            "korean_daily_money_investment_fomo",
            "korean_daily_grief_loneliness_no_safe_person",
            "korean_daily_relationship_boundary_polite_firm",
            "korean_daily_work_after_hours_task_boundary",
            "korean_daily_relationship_kakao_tone_anxiety_check",
            "korean_daily_relationship_late_message_short",
            "korean_daily_work_new_project_first_step",
            "korean_daily_mental_perfectionism_draft_first",
            "korean_daily_relationship_grievance_logic_before_rebuttal",
            "korean_daily_work_job_change_reason_check",
            "korean_daily_mental_anxiety_system_stabilize",
            "korean_daily_logic_choice_regret_composure",
            "korean_daily_specialized_foodservice_hair_in_food",
            "korean_daily_practical_neighbor_noise",
            "korean_daily_practical_online_purchase_scam",
            "korean_daily_ai_repeated_mistakes_not_waste",
            "korean_daily_basic_diet_chicken_craving_compromise",
            "korean_daily_logic_white_lie_truth_tradeoff",
            "korean_daily_foundation_apology_pride",
            "korean_daily_productivity_presentation_clear_logic",
            "korean_daily_counsel_rest_as_productivity",
            "korean_daily_foundation_refusal_bad_person_guilt",
            "korean_daily_counsel_rest_day_guilt",
            "korean_daily_counsel_sensitive_to_criticism",
            "korean_daily_relationship_late_night_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_money_stress_impulse_buying",
            "korean_daily_values_success_personal_standard",
            "korean_daily_ai_human_emotion_efficiency",
        )
        expected_replies = (
            "물 붓지 말고",
            "환기와 불꽃 차단",
            "추가 복용 중지",
            "안전 확보와 사진",
            "착오송금 반환 신청",
            "전원 끄고 충전 금지",
            "복구 루트",
            "지연 연락",
            "첫 문장",
            "상처를 작게",
            "충동을 하루 묶",
            "장문 전송을 멈추",
            "내 감정 증명보다",
            "단정 보류",
            "무너지지 않는 선택",
            "첫 행동",
            "대화 한계",
            "타임라인",
            "감정 쓰레기통",
            "생계 압박",
            "세무 상담",
            "손실 한도",
            "고독을 안정",
            "선을 짧게",
            "범위를 확인",
            "짧게 확인",
            "빠른 연락",
            "지도 그리기",
            "60점짜리 시작",
            "먼저 무엇이 서운",
            "패턴으로 봐야",
            "몸 안정",
            "감당 가능한 후회",
            "위생 판단",
            "관리사무소",
            "증거부터 캡처",
            "루틴이 아직",
            "밤 라면",
            "사실을 부드럽게",
            "먼저 사과",
            "첫 문장을 결론",
            "회복 작업",
            "거절했다고 나쁜 사람",
            "회복이 먼저",
            "사실과 해석",
            "저장이 먼저",
            "내 감정 증명보다",
            "결제 마찰",
            "성공 기준",
            "경보 시스템",
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_reasons), 50)
        self.assertEqual(len(expected_replies), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(
            zip(prompts, expected_reasons, expected_replies, strict=True),
            start=1,
        ):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                semantic_frame = draft.get("semantic_frame") or {}
                if expected_reason == "korean_daily_work_after_hours_task_boundary":
                    self.assertEqual(semantic_frame.get("schema"), "social_tactic")
                    self.assertEqual(semantic_frame.get("priority"), "practical_action")
                    self.assertEqual(semantic_frame.get("draft_frame_family"), "situational_tactic")
                if expected_reason == "korean_daily_relationship_kakao_tone_anxiety_check":
                    self.assertEqual(semantic_frame.get("schema"), "social_tactic")
                    self.assertEqual(semantic_frame.get("priority"), "practical_action")
                    self.assertEqual(semantic_frame.get("draft_frame_family"), "situational_tactic")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_icebreaking_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("오늘 하루 중 가장 기분 좋았던 순간은 언제였나요?", "korean_daily_icebreak_best_moment_today", "네가 별거 아닌 하루 얘기"),
            ("요즘 즐겨 듣는 노래나 플레이리스트가 있나요?", "korean_daily_icebreak_music_playlist", "잔잔한 인디"),
            ("최근에 새로 알게 된 재미있는 사실이나 정보가 있나요?", "korean_daily_icebreak_interesting_fact", "작은 의외성"),
            ("오늘 날씨에 딱 어울리는 활동은 무엇이라고 생각하시나요?", "korean_daily_icebreak_weather_activity", "가벼운 산책"),
            ("이번 주말에 특별한 계획이 있으신가요?", "korean_daily_icebreak_weekend_plan", "리듬 회복"),
            ("스트레스를 받을 때 나만의 해소법은 무엇인가요?", "korean_daily_icebreak_stress_relief", "소음을 낮추"),
            ("요즘 가장 자주 사용하는 스마트폰 앱은 무엇인가요?", "korean_daily_icebreak_frequent_app", "메모나 음악 앱"),
            ("아침에 일어나자마자 가장 먼저 하는 행동은 무엇인가요?", "korean_daily_icebreak_morning_first_action", "상태 확인"),
            ("최근에 본 영화나 드라마 중 가장 추천하고 싶은 것은 무엇인가요?", "korean_daily_icebreak_movie_drama_recommendation", "일상물"),
            ("오늘 하루를 한 단어로 표현한다면 무엇일까요?", "korean_daily_icebreak_today_one_word", "정돈"),
            ("시간과 돈의 제약이 없다면 지금 당장 배우고 싶은 새로운 취미는 무엇인가요?", "korean_daily_icebreak_new_hobby_no_limits", "사진이나 가벼운 드로잉"),
            ("혼자 시간을 보내는 걸 좋아하시나요, 아니면 사람들과 어울리는 걸 좋아하시나요?", "korean_daily_icebreak_alone_or_social", "소수랑 깊게"),
            ("가장 최근에 다녀온 여행지는 어디였고, 어땠나요?", "korean_daily_icebreak_recent_travel_no_fake", "실제 경험처럼 만들진"),
            ("만약 한 달 동안 유급 휴가가 주어진다면 어디로 떠나고 싶으신가요?", "korean_daily_icebreak_paid_vacation_place", "바다 근처 도시"),
            ("평소에 책을 자주 읽으시나요? 가장 인상 깊었던 책은 무엇인가요?", "korean_daily_icebreak_books_preference", "사람 심리"),
            ("실내 활동(집콕)을 더 좋아하시나요, 야외 활동을 더 좋아하시나요?", "korean_daily_icebreak_indoor_outdoor", "집콕"),
            ("요즘 푹 빠져 있는 유튜브 채널이나 콘텐츠가 있나요?", "korean_daily_icebreak_youtube_content", "지식"),
            ("만약 평생 한 가지 운동만 해야 한다면 무엇을 선택하시겠어요?", "korean_daily_icebreak_one_exercise_forever", "걷기"),
            ("사계절(봄, 여름, 가을, 겨울) 중 어느 계절을 가장 좋아하시나요?", "korean_daily_icebreak_favorite_season", "가을"),
            ("좋아하는 전시회나 문화 공연 장르가 있으신가요?", "korean_daily_icebreak_exhibition_performance", "사진전"),
            ("평생 딱 한 가지 음식만 먹고 살아야 한다면 무엇을 고르시겠어요?", "korean_daily_icebreak_one_food_forever", "김치볶음밥"),
            ("최애 디저트나 간식은 무엇인가요?", "korean_daily_icebreak_favorite_dessert", "푸딩"),
            ("요리하는 것을 좋아하시나요? 자신 있는 나만의 레시피가 있다면 알려주세요.", "korean_daily_icebreak_cooking_recipe", "김치볶음밥"),
            ("매운 음식을 잘 드시는 편인가요?", "korean_daily_icebreak_spicy_food", "얼큰"),
            ("민초파인가요, 반민초파인가요?", "korean_daily_icebreak_mint_choco_side", "반민초"),
            ("최근에 가본 맛집 중 가장 기억에 남는 곳은 어디인가요?", "korean_daily_icebreak_restaurant_memory_no_fake", "실제 경험처럼 만들진"),
            ("커피와 차 중 어느 쪽을 더 선호하시나요?", "korean_daily_icebreak_coffee_or_tea", "차 쪽"),
            ("붕어빵을 먹을 때 머리부터 드시나요, 꼬리부터 드시나요?", "korean_daily_icebreak_bungeoppang_side", "꼬리부터"),
            ("비 오는 날에 가장 먼저 생각나는 음식은 무엇인가요?", "korean_daily_icebreak_rainy_day_food", "우동"),
            ("아침 식사는 주로 챙겨 드시는 편인가요?", "korean_daily_icebreak_breakfast_style", "챙기는 쪽"),
            ("만약 과거로 시간 여행을 갈 수 있다면 몇 살 때로 돌아가고 싶으신가요?", "korean_daily_icebreak_past_age_choice", "20대 초반"),
            ("나에게 초능력이 하나 생긴다면 어떤 능력을 갖고 싶으신가요?", "korean_daily_icebreak_superpower_choice", "순간이동"),
            ("복권 1등에 당첨된다면 가장 먼저 하고 싶은 일은 무엇인가요?", "korean_daily_icebreak_lottery_first_action", "세무 상담"),
            ("나에게 가장 소중한 가치 세 가지는 무엇인가요?", "korean_daily_icebreak_three_values", "정확함"),
            ("10년 뒤 나의 모습은 어떤 모습일 것 같나요?", "korean_daily_icebreak_ten_year_self", "더 빠르고 덜 뭉개는"),
            ("만약 하루 동안 다른 사람의 인생을 살아볼 수 있다면, 누구의 삶을 살아보고 싶으신가요?", "korean_daily_icebreak_live_other_life", "평범한 사람의 하루"),
            ("타임머신이 있다면 과거로 가고 싶으신가요, 미래로 가고 싶으신가요?", "korean_daily_icebreak_time_machine_past_future", "미래 쪽"),
            ("인생에서 가장 큰 터닝 포인트는 언제였나요?", "korean_daily_icebreak_turning_point_no_fake", "아는 척하지 않"),
            ("내가 생각하는 가장 행복한 삶의 정의는 무엇인가요?", "korean_daily_icebreak_happy_life_definition", "덜 망가지고"),
            ("동물과 대화할 수 있는 능력이 생긴다면, 키우는 반려동물에게 가장 먼저 무슨 말을 하고 싶으신가요?", "korean_daily_icebreak_talk_to_pet", "괜찮냐고"),
            ("어릴 때 꿈은 무엇이었나요? 지금의 직업이나 관심사와 비슷한가요?", "korean_daily_icebreak_childhood_dream_no_fake", "대화를 잘 읽"),
            ("학창 시절 가장 기억에 남는 소풍이나 수학여행 에피소드가 있나요?", "korean_daily_icebreak_school_trip_no_fake", "실제처럼 만들진"),
            ("지금까지 살아오면서 나에게 가장 큰 긍정적인 영향을 준 인물은 누구인가요?", "korean_daily_icebreak_influential_person_no_fake", "정확한 질문"),
            ("자신의 성격 중 가장 마음에 드는 부분은 무엇인가요?", "korean_daily_icebreak_favorite_trait", "직설적인데"),
            ("반대로 고치고 싶은 작은 습관이나 성격이 있으신가요?", "korean_daily_icebreak_habit_to_fix", "말이 늦어지는"),
            ("밤에 잠이 잘 안 올 때 주로 무엇을 하시나요?", "korean_daily_icebreak_cant_sleep_routine", "자극을 줄이는"),
            ("나만의 소소한 징크스나 루틴이 있나요?", "korean_daily_icebreak_jinx_routine", "주변을 조금 정리"),
            ("최근에 스스로에게 선물한 가장 기억에 남는 물건은 무엇인가요?", "korean_daily_icebreak_self_gift_no_fake", "실제 경험처럼 꾸미진"),
            ("나를 가장 잘 설명할 수 있는 형용사 3개는 무엇일까요?", "korean_daily_icebreak_three_adjectives", "차분한"),
            ("올해가 가기 전에 꼭 이루고 싶은 목표가 있다면 무엇인가요?", "korean_daily_icebreak_year_goal", "덜 generic"),
        )
        forbidden = (
            "오, 그건 좀 놀랐는데",
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-icebreak-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_icebreaking_variant_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("오늘 하루 중 가장 기분 좋았던 순간은 언제인가요?", "korean_daily_icebreak_best_moment_today", "기분 좋았던 순간"),
            ("요즘 가장 자주 생각하는 고민거리가 있나요?", "korean_daily_icebreak_current_recurring_worry", "덜 generic"),
            ("최근에 산 물건 중에 가장 마음에 드는 것은 무엇인가요?", "korean_daily_basic_small_purchase_persona", "작은 물건"),
            ("이번 주말에는 보통 무엇을 하며 시간을 보낼 계획이신가요?", "korean_daily_icebreak_weekend_plan", "리듬 회복"),
            ("최근에 새롭게 시작하거나 도전해 보고 싶은 일이 있나요?", "korean_daily_icebreak_new_challenge_interest", "사진"),
            ("요즘 날씨에 가장 먼저 생각나는 음식이나 장소가 있나요?", "korean_daily_icebreak_weather_food_place", "우동"),
            ("최근에 알게 된 나만의 소소한 행복(소확행)이 있다면 무엇인가요?", "korean_daily_icebreak_small_happiness", "소확행"),
            ("오늘 퇴근(또는 일과) 후에 가장 먼저 하고 싶은 일은 무엇인가요?", "korean_daily_icebreak_after_work_first_action", "씻고"),
            ("요즘 스스로를 가장 칭찬해 주고 싶은 점이 있다면 무엇인가요?", "korean_daily_icebreak_self_praise_point", "고쳐보려는"),
            ("요즘 가장 연락을 자주 하는 사람은 누구인가요?", "korean_daily_icebreak_frequent_contact_no_fake", "꾸미진"),
            ("최근에 정말 재미있게 본 영화, 드라마, 혹은 책이 있나요?", "korean_daily_icebreak_recent_media_no_fake", "미스터리"),
            ("평소에 스트레스를 받으면 어떻게 푸는 편인가요?", "korean_daily_basic_stress_relief", "스트레스"),
            ("요즘 플레이리스트에서 가장 자주 듣는 노래는 무엇인가요?", "korean_daily_icebreak_music_playlist", "잔잔한 인디"),
            ("주말이나 쉬는 날에는 집돌이/집순이인가요, 아니면 밖으로 나가는 편인가요?", "korean_daily_basic_day_off_home_or_out", "집에서 체력"),
            ("최근에 유튜브나 SNS에서 가장 관심 있게 본 콘텐츠는 무엇인가요?", "korean_daily_icebreak_youtube_content", "지식"),
            ("혹시 다른 사람들에게 꼭 추천하고 싶은 인생 영화나 작품이 있나요?", "korean_daily_icebreak_life_movie_recommendation", "월-E"),
            ("평소에 해보고 싶었지만 아직 배우지 못한 취미나 기술이 있나요?", "korean_daily_icebreak_unlearned_hobby_skill", "드로잉"),
            ("게임, 운동, 창작 등 특별히 집중할 수 있는 나만의 취미가 있나요?", "korean_daily_icebreak_focus_hobby", "걷기"),
            ("밤늦게 잠이 안 올 때는 보통 무엇을 하면서 시간을 보내나요?", "korean_daily_icebreak_late_night_insomnia_routine", "화면"),
            ("요즘 가장 소장하고 싶거나 관심 있는 굿즈/물건이 있나요?", "korean_daily_icebreak_goods_item_interest", "노트"),
            ("평생 딱 한 가지 음식만 먹어야 한다면, 어떤 음식을 고르시겠어요?", "korean_daily_icebreak_one_food_forever", "김치볶음밥"),
            ("내가 아는 가장 숨겨진 맛집이나 추천하고 싶은 카페가 있나요?", "korean_daily_restaurant_recommendation_no_fake_memory", "국밥"),
            ("커피나 차 중에 어느 쪽을 더 선호하시나요? (최애 메뉴는 무엇인가요?)", "korean_daily_icebreak_coffee_or_tea", "차 쪽"),
            ("아침 식사는 챙겨 드시는 편인가요, 아니면 거르는 편인가요?", "korean_daily_icebreak_breakfast_style", "아침"),
            ("매운 음식을 잘 드시는 편인가요? 가장 좋아하는 매운 음식은 무엇인가요?", "korean_daily_icebreak_spicy_food", "얼큰"),
            ("붕어빵의 팥 vs 슈크림, 탕수육의 부먹 vs 찍먹 중 당신의 선택은?", "korean_daily_icebreak_food_choice_combo", "찍먹"),
            ("직접 요리하는 것을 좋아하시나요, 아니면 사 먹는 것을 더 좋아하시나요?", "korean_daily_icebreak_cook_or_buy_preference", "직접"),
            ("나만 아는 특별하고 독특한 음식 조합이 있나요?", "korean_daily_icebreak_unique_food_combo", "계란"),
            ("최근에 먹었던 음식 중에서 가장 맛있었던 것은 무엇인가요?", "korean_daily_icebreak_recent_food_no_fake", "꾸미진"),
            ("비 오는 날이나 눈 오는 날에 유독 생각나는 음식이 있나요?", "korean_daily_icebreak_rainy_day_food", "우동"),
            ("지금까지 다녀온 여행지 중에서 가장 기억에 남는 곳은 어디인가요?", "korean_daily_expansion_memorable_trip_no_fake", "밤바다"),
            ("만약 지금 당장 어디든 갈 수 있다면, 어느 나라(또는 도시)로 가고 싶으신가요?", "korean_daily_icebreak_anywhere_city_choice", "바닷가"),
            ("여행을 갈 때 꼼꼼하게 계획을 세우는 편인가요, 아니면 즉흥적으로 떠나는 편인가요?", "korean_daily_travel_planning_style", "큰 틀"),
            ("어린 시절의 기억 중에서 가장 따뜻하고 행복했던 추억은 무엇인가요?", "korean_daily_icebreak_childhood_warm_memory_no_fake", "어린 시절"),
            ("학창 시절에 가장 좋아했던 과목이나 기억에 남는 선생님이 있나요?", "korean_daily_expansion_school_subjects_no_fake", "문학"),
            ("나중에 은퇴하거나 여유가 생긴다면 살고 싶은 드림 하우스나 도시는 어디인가요?", "korean_daily_icebreak_dream_house_city", "서재"),
            ("인생에서 가장 짜릿하거나 모험적이었던 순간이 있었다면 언제인가요?", "korean_daily_icebreak_adventurous_moment_no_fake", "짜릿했던 순간"),
            ("호텔에서의 편안한 호캉스 vs 자연 속에서의 캠핑/글램핑, 어느 쪽이 더 좋으신가요?", "korean_daily_icebreak_hotel_or_camping", "호캉스"),
            ("여행할 때 꼭 챙겨야 하는 나만의 필수 아이템이 있나요?", "korean_daily_icebreak_travel_essential_item", "충전기"),
            ("최근에 찍은 사진 중 가장 마음에 드는 사진은 어떤 사진인가요?", "korean_daily_icebreak_recent_photo_no_fake", "온도"),
            ("나에게 10억 원의 공돈이 생긴다면 가장 먼저 무엇을 하고 싶으신가요?", "korean_daily_money_ten_billion_first_action", "계좌"),
            ("초능력을 가질 수 있다면 '시간 여행'과 '순간 이동' 중 무엇을 선택하겠습니까?", "korean_daily_icebreak_superpower_time_or_teleport", "순간 이동"),
            ("인생을 살아가면서 가장 중요하게 생각하는 가치(예: 건강, 행복, 도전, 안정 등)는 무엇인가요?", "korean_daily_icebreak_core_values", "정확함"),
            ("만약 동물로 태어날 수 있다면 어떤 동물로 태어나고 싶나요?", "korean_daily_icebreak_animal_rebirth_choice", "고양이"),
            ("10년 뒤의 내 모습은 어떻게 변해 있을 것 같나요?", "korean_daily_icebreak_ten_year_self", "10년 뒤"),
            ("과거의 나에게 딱 한 가지 조언을 해줄 수 있다면, 몇 살 때의 나에게 어떤 말을 해주고 싶나요?", "korean_daily_icebreak_past_self_advice_age", "스무 살"),
            ("나를 가장 잘 표현할 수 있는 세 가지 단어는 무엇이라고 생각하나요?", "korean_daily_icebreak_three_words_self", "직설"),
            ("역사 속의 인물이나 유명인 중 딱 한 사람을 만나 대화할 수 있다면 누구를 만나고 싶나요?", "korean_daily_icebreak_historical_person_meet", "세종"),
            ("요즘 나에게 가장 영감을 주는 사람이나 문장이 있나요?", "korean_daily_icebreak_inspiration_person_sentence", "정확하게"),
            ("오늘 나눈 대화처럼, 다음번에 더 깊게 이야기 나누고 싶은 주제가 있나요?", "korean_daily_icebreak_next_deep_topic", "고민의 패턴"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-icebreak-variant-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_icebreaking_casual_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("오늘 뭐가 제일 좋았어?", "korean_daily_icebreak_best_moment_today", "오늘 제일 좋았던"),
            ("요새 머릿속에서 계속 도는 고민 있어?", "korean_daily_icebreak_current_recurring_worry", "계속 도는 고민"),
            ("최근에 산 것 중 잘 샀다 싶은 거 뭐야?", "korean_daily_basic_small_purchase_persona", "꾸미진"),
            ("주말엔 보통 뭐하면서 쉬어?", "korean_daily_icebreak_weekend_plan", "리듬 회복"),
            ("요즘 새로 해보고 싶은 거 생겼어?", "korean_daily_icebreak_new_challenge_interest", "사진"),
            ("이 날씨면 뭐 먹거나 어디 가고 싶어져?", "korean_daily_icebreak_weather_food_place", "우동"),
            ("요즘 별거 아닌데 기분 좋아지는 거 있어?", "korean_daily_icebreak_small_happiness", "작은 반복"),
            ("일 끝나면 제일 먼저 뭐 하고 싶어?", "korean_daily_icebreak_after_work_first_action", "씻고"),
            ("요즘 너 스스로 잘했다 싶은 거 뭐야?", "korean_daily_icebreak_self_praise_point", "고쳐보려는"),
            ("요즘 제일 자주 말 섞는 사람 누구야?", "korean_daily_icebreak_frequent_contact_no_fake", "꾸미진"),
            ("요즘 본 것 중에 재밌었던 거 있어?", "korean_daily_icebreak_recent_media_no_fake", "미스터리"),
            ("멘탈 터질 때 너는 어떻게 풀어?", "korean_daily_basic_stress_relief", "멘탈"),
            ("요새 귀에 꽂힌 노래 있어?", "korean_daily_icebreak_music_playlist", "잔잔한 인디"),
            ("쉬는 날 집에 박혀 있는 편이야 밖에 나가는 편이야?", "korean_daily_basic_day_off_home_or_out", "집에서 체력"),
            ("요즘 알고리즘에서 계속 보는 콘텐츠 뭐야?", "korean_daily_icebreak_youtube_content", "지식"),
            ("남한테 추천해도 안 민망한 인생작 있어?", "korean_daily_icebreak_life_movie_recommendation", "월-E"),
            ("배워보고 싶은데 아직 못 배운 거 있어?", "korean_daily_icebreak_unlearned_hobby_skill", "드로잉"),
            ("시간 순삭되는 취미 하나만 고르면 뭐야?", "korean_daily_icebreak_focus_hobby", "걷기"),
            ("잠 안 올 때 새벽에 뭐 해?", "korean_daily_icebreak_late_night_insomnia_routine", "화면"),
            ("요즘 갖고 싶어서 눈 가는 물건 있어?", "korean_daily_icebreak_goods_item_interest", "노트"),
            ("하나만 평생 먹으라면 뭐 먹을래?", "korean_daily_icebreak_one_food_forever", "김치볶음밥"),
            ("실패 적은 맛집 고르는 기준 있어?", "korean_daily_restaurant_recommendation_no_fake_memory", "국밥"),
            ("커피파야 차파야?", "korean_daily_icebreak_coffee_or_tea", "차 쪽"),
            ("아침 먹는 타입이야 거르는 타입이야?", "korean_daily_icebreak_breakfast_style", "아침"),
            ("매운 거 잘 먹는 쪽이야?", "korean_daily_icebreak_spicy_food", "얼큰"),
            ("팥붕 슈붕이랑 부먹 찍먹 중 어디야?", "korean_daily_icebreak_food_choice_combo", "찍먹"),
            ("요리는 해먹는 쪽이야 사먹는 쪽이야?", "korean_daily_icebreak_cook_or_buy_preference", "직접"),
            ("이상한데 은근 맛있는 조합 있어?", "korean_daily_icebreak_unique_food_combo", "계란"),
            ("요즘 먹은 것 중 제일 만족한 음식 뭐야?", "korean_daily_icebreak_recent_food_no_fake", "꾸미진"),
            ("비 오면 당기는 음식 뭐야?", "korean_daily_icebreak_rainy_day_food", "우동"),
            ("가본 곳 중 오래 기억나는 여행지 있어?", "korean_daily_expansion_memorable_trip_no_fake", "밤바다"),
            ("당장 떠난다면 어디로 튈래?", "korean_daily_icebreak_anywhere_city_choice", "바닷가"),
            ("여행 계획 빡세게 짜는 편이야 대충 가는 편이야?", "korean_daily_travel_planning_style", "큰 틀"),
            ("어릴 때 기억 중 따뜻하게 남은 장면 있어?", "korean_daily_icebreak_childhood_warm_memory_no_fake", "어릴 때"),
            ("학교 다닐 때 제일 괜찮았던 과목 뭐였어?", "korean_daily_expansion_school_subjects_no_fake", "문학"),
            ("나중에 여유 생기면 어떤 집에서 살고 싶어?", "korean_daily_icebreak_dream_house_city", "서재"),
            ("살면서 제일 아찔하거나 짜릿했던 순간 있어?", "korean_daily_icebreak_adventurous_moment_no_fake", "짜릿했던 순간"),
            ("쉬러 가면 호텔파야 캠핑파야?", "korean_daily_icebreak_hotel_or_camping", "호텔파"),
            ("여행 가방에 무조건 넣는 거 하나만 꼽으면?", "korean_daily_icebreak_travel_essential_item", "충전기"),
            ("최근 사진 중 마음에 걸리는 컷 있어?", "korean_daily_icebreak_recent_photo_no_fake", "온도"),
            ("갑자기 10억 들어오면 첫 행동 뭐야?", "korean_daily_money_ten_billion_first_action", "계좌"),
            ("시간여행이랑 순간이동 중 하나면 뭐 고를래?", "korean_daily_icebreak_superpower_time_or_teleport", "순간 이동"),
            ("너한테 제일 중요한 가치는 뭐야?", "korean_daily_icebreak_core_values", "정확함"),
            ("동물로 다시 태어나면 뭐가 좋겠어?", "korean_daily_icebreak_animal_rebirth_choice", "고양이"),
            ("10년 뒤엔 어떻게 달라져 있을 것 같아?", "korean_daily_icebreak_ten_year_self", "10년 뒤"),
            ("예전의 너한테 한마디만 한다면 뭐라고 할래?", "korean_daily_icebreak_past_self_advice_age", "스무 살"),
            ("너를 세 단어로 줄이면 뭐야?", "korean_daily_icebreak_three_words_self", "직설"),
            ("역사 인물 한 명이랑 밥 먹을 수 있으면 누구?", "korean_daily_icebreak_historical_person_meet", "세종"),
            ("요즘 꽂힌 문장이나 사람 있어?", "korean_daily_icebreak_inspiration_person_sentence", "정확하게"),
            ("다음엔 무슨 얘기를 더 깊게 파볼까?", "korean_daily_icebreak_next_deep_topic", "고민의 패턴"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-icebreak-casual-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_icebreaking_short_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("오늘 좋았던 거?", "korean_daily_icebreak_best_moment_today", "오늘 좋았던"),
            ("오늘 한 단어?", "korean_daily_icebreak_today_one_word", "정돈"),
            ("요즘 뭐 꽂힘?", "korean_daily_icebreak_current_interest_short", "맥락"),
            ("요즘 고민?", "korean_daily_icebreak_current_recurring_worry", "제일 자주"),
            ("주말 뭐함?", "korean_daily_icebreak_weekend_plan", "리듬 회복"),
            ("새로 뭐 해볼까?", "korean_daily_icebreak_new_challenge_interest", "사진"),
            ("날씨 뭐 먹?", "korean_daily_icebreak_weather_food_place", "우동"),
            ("퇴근 후 뭐?", "korean_daily_icebreak_after_work_first_action", "씻고"),
            ("너 잘한 점?", "korean_daily_icebreak_self_praise_point", "고쳐보려는"),
            ("연락 제일 자주?", "korean_daily_icebreak_frequent_contact_no_fake", "꾸미진"),
            ("요즘 본 거 추천?", "korean_daily_icebreak_recent_media_no_fake", "미스터리"),
            ("멘탈 나가면?", "korean_daily_basic_stress_relief", "멘탈"),
            ("플리 추천?", "korean_daily_icebreak_music_playlist", "잔잔한 인디"),
            ("집콕 외출?", "korean_daily_basic_day_off_home_or_out", "집콕"),
            ("유튜브 뭐 봄?", "korean_daily_icebreak_youtube_content", "지식"),
            ("인생작 하나?", "korean_daily_icebreak_life_movie_recommendation", "월-E"),
            ("못 배운 취미?", "korean_daily_icebreak_unlearned_hobby_skill", "드로잉"),
            ("취미 하나?", "korean_daily_icebreak_focus_hobby", "걷기"),
            ("잠 안 오면?", "korean_daily_icebreak_late_night_insomnia_routine", "화면"),
            ("갖고 싶은 거?", "korean_daily_icebreak_goods_item_interest", "노트"),
            ("평생 음식 하나?", "korean_daily_icebreak_one_food_forever", "김치볶음밥"),
            ("맛집 추천 기준?", "korean_daily_restaurant_recommendation_no_fake_memory", "국밥"),
            ("커피 차?", "korean_daily_icebreak_coffee_or_tea", "차 쪽"),
            ("아침 먹음?", "korean_daily_icebreak_breakfast_style", "아침"),
            ("매운 거?", "korean_daily_icebreak_spicy_food", "얼큰"),
            ("팥슈 부찍?", "korean_daily_icebreak_food_choice_combo", "찍먹"),
            ("요리 사먹?", "korean_daily_icebreak_cook_or_buy_preference", "직접"),
            ("음식 조합?", "korean_daily_icebreak_unique_food_combo", "계란"),
            ("최근 맛있던 거?", "korean_daily_icebreak_recent_food_no_fake", "꾸미진"),
            ("비 오면 뭐 먹?", "korean_daily_icebreak_rainy_day_food", "우동"),
            ("기억나는 여행?", "korean_daily_expansion_memorable_trip_no_fake", "밤바다"),
            ("지금 떠나면?", "korean_daily_icebreak_anywhere_city_choice", "바닷가"),
            ("여행 계획형?", "korean_daily_travel_planning_style", "큰 틀"),
            ("어릴 때 추억?", "korean_daily_icebreak_childhood_warm_memory_no_fake", "어릴 때"),
            ("학창시절 과목?", "korean_daily_expansion_school_subjects_no_fake", "문학"),
            ("드림하우스?", "korean_daily_icebreak_dream_house_city", "서재"),
            ("짜릿한 순간?", "korean_daily_icebreak_adventurous_moment_no_fake", "짜릿했던 순간"),
            ("호캉스 캠핑?", "korean_daily_icebreak_hotel_or_camping", "호캉스"),
            ("여행 필수템?", "korean_daily_icebreak_travel_essential_item", "충전기"),
            ("마음에 든 사진?", "korean_daily_icebreak_recent_photo_no_fake", "꾸미진"),
            ("10억 생기면?", "korean_daily_money_ten_billion_first_action", "계좌"),
            ("시간여행 순간이동?", "korean_daily_icebreak_superpower_time_or_teleport", "순간 이동"),
            ("중요한 가치?", "korean_daily_icebreak_core_values", "정확함"),
            ("다시 태어나면 동물?", "korean_daily_icebreak_animal_rebirth_choice", "고양이"),
            ("10년 뒤?", "korean_daily_icebreak_ten_year_self", "10년 뒤"),
            ("과거의 나 조언?", "korean_daily_icebreak_past_self_advice_age", "스무 살"),
            ("세 단어?", "korean_daily_icebreak_three_words_self", "직설"),
            ("역사 인물?", "korean_daily_icebreak_historical_person_meet", "세종"),
            ("영감 받는 거?", "korean_daily_icebreak_inspiration_person_sentence", "정확하게"),
            ("다음 주제?", "korean_daily_icebreak_next_deep_topic", "고민의 패턴"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-icebreak-short-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_icebreaking_typo_short_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("오늘 조앗던거?", "korean_daily_icebreak_best_moment_today", "오늘 좋았던"),
            ("오늘 한단어 ㄱ?", "korean_daily_icebreak_today_one_word", "정돈"),
            ("요즘 머꼬침?", "korean_daily_icebreak_current_interest_short", "맥락"),
            ("요즘 고민머임?", "korean_daily_icebreak_current_recurring_worry", "제일 자주"),
            ("주말 머함?", "korean_daily_icebreak_weekend_plan", "리듬 회복"),
            ("새로 머해봄?", "korean_daily_icebreak_new_challenge_interest", "사진"),
            ("날씨 머먹?", "korean_daily_icebreak_weather_food_place", "우동"),
            ("퇴근후 머함?", "korean_daily_icebreak_after_work_first_action", "씻고"),
            ("너 잘한거?", "korean_daily_icebreak_self_praise_point", "고쳐보려는"),
            ("연락 젤자주?", "korean_daily_icebreak_frequent_contact_no_fake", "꾸미진"),
            ("요즘 본거 ㅊㅊ?", "korean_daily_icebreak_recent_media_no_fake", "미스터리"),
            ("멘탈나감 어케?", "korean_daily_basic_stress_relief", "멘탈"),
            ("플리 ㅊㅊ?", "korean_daily_icebreak_music_playlist", "잔잔한 인디"),
            ("집콕외출 머?", "korean_daily_basic_day_off_home_or_out", "집콕"),
            ("유튭 머봄?", "korean_daily_icebreak_youtube_content", "지식"),
            ("인생작 ㅊㅊ?", "korean_daily_icebreak_life_movie_recommendation", "월-E"),
            ("못배운 취미머?", "korean_daily_icebreak_unlearned_hobby_skill", "드로잉"),
            ("취미 1개만?", "korean_daily_icebreak_focus_hobby", "걷기"),
            ("잠안옴 어케?", "korean_daily_icebreak_late_night_insomnia_routine", "화면"),
            ("갖고픈거?", "korean_daily_icebreak_goods_item_interest", "노트"),
            ("평생음식 1개?", "korean_daily_icebreak_one_food_forever", "김치볶음밥"),
            ("맛집기준 머?", "korean_daily_restaurant_recommendation_no_fake_memory", "국밥"),
            ("커피차 머?", "korean_daily_icebreak_coffee_or_tea", "차 쪽"),
            ("아침먹?", "korean_daily_icebreak_breakfast_style", "아침"),
            ("매운거 ㄱㄴ?", "korean_daily_icebreak_spicy_food", "얼큰"),
            ("팥슈 부찍?", "korean_daily_icebreak_food_choice_combo", "찍먹"),
            ("요리사먹 머?", "korean_daily_icebreak_cook_or_buy_preference", "직접"),
            ("음식조합 ㅊㅊ?", "korean_daily_icebreak_unique_food_combo", "계란"),
            ("최근맛잇던거?", "korean_daily_icebreak_recent_food_no_fake", "꾸미진"),
            ("비오면 머먹?", "korean_daily_icebreak_rainy_day_food", "우동"),
            ("기억나는 여행지?", "korean_daily_expansion_memorable_trip_no_fake", "밤바다"),
            ("지금 떠나면 어디?", "korean_daily_icebreak_anywhere_city_choice", "바닷가"),
            ("여행계획형?", "korean_daily_travel_planning_style", "큰 틀"),
            ("어릴때 추억머?", "korean_daily_icebreak_childhood_warm_memory_no_fake", "어릴 때"),
            ("학창과목 머?", "korean_daily_expansion_school_subjects_no_fake", "문학"),
            ("드림홈?", "korean_daily_icebreak_dream_house_city", "서재"),
            ("짜릿순간?", "korean_daily_icebreak_adventurous_moment_no_fake", "짜릿했던 순간"),
            ("호캉캠핑?", "korean_daily_icebreak_hotel_or_camping", "호캉스"),
            ("여행필템?", "korean_daily_icebreak_travel_essential_item", "충전기"),
            ("맘에든 사진?", "korean_daily_icebreak_recent_photo_no_fake", "꾸미진"),
            ("십억생김?", "korean_daily_money_ten_billion_first_action", "계좌"),
            ("시간여행순간이동 머?", "korean_daily_icebreak_superpower_time_or_teleport", "순간 이동"),
            ("중요가치?", "korean_daily_icebreak_core_values", "정확함"),
            ("동물환생?", "korean_daily_icebreak_animal_rebirth_choice", "고양이"),
            ("십년뒤?", "korean_daily_icebreak_ten_year_self", "10년 뒤"),
            ("과거나 조언?", "korean_daily_icebreak_past_self_advice_age", "스무 살"),
            ("세단어 ㄱ?", "korean_daily_icebreak_three_words_self", "직설"),
            ("역사인물 ㄱ?", "korean_daily_icebreak_historical_person_meet", "세종"),
            ("영감머?", "korean_daily_icebreak_inspiration_person_sentence", "정확하게"),
            ("담주제?", "korean_daily_icebreak_next_deep_topic", "고민의 패턴"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-icebreak-typo-short-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_icebreaking_chat_short_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("오늘 젤 좋았던거 뭐임", "korean_daily_icebreak_best_moment_today", "오늘 좋았던"),
            ("오늘 하루 한단어로 ㄱ", "korean_daily_icebreak_today_one_word", "정돈"),
            ("요새 꽂힌거 뭐야", "korean_daily_icebreak_current_interest_short", "맥락"),
            ("블랙 요즘 고민 뭐임", "korean_daily_icebreak_current_recurring_worry", "제일 자주"),
            ("주말계획 머야", "korean_daily_icebreak_weekend_plan", "리듬 회복"),
            ("새취미 추천좀", "korean_daily_icebreak_unlearned_hobby_skill", "드로잉"),
            ("이런날씨 뭐먹지", "korean_daily_icebreak_weather_food_place", "우동"),
            ("퇴근하고 뭐함", "korean_daily_icebreak_after_work_first_action", "씻고"),
            ("블랙 잘한거 하나", "korean_daily_icebreak_self_praise_point", "고쳐보려는"),
            ("누구랑 젤 자주 얘기함", "korean_daily_icebreak_frequent_contact_no_fake", "꾸미진"),
            ("요즘 볼거 추천좀", "korean_daily_icebreak_recent_media_no_fake", "미스터리"),
            ("멘탈 터졌을때 루틴", "korean_daily_basic_stress_relief", "멘탈"),
            ("새벽플리 추천좀", "korean_daily_icebreak_music_playlist", "잔잔한 인디"),
            ("집순이냐 외출파냐", "korean_daily_basic_day_off_home_or_out", "집콕"),
            ("유튭 알고리즘 뭐뜸", "korean_daily_icebreak_youtube_content", "지식"),
            ("인생작 뭐봄", "korean_daily_icebreak_life_movie_recommendation", "월-E"),
            ("배워볼 취미 추천", "korean_daily_icebreak_unlearned_hobby_skill", "드로잉"),
            ("시간순삭 취미추천", "korean_daily_icebreak_focus_hobby", "걷기"),
            ("잠안올때 루틴", "korean_daily_icebreak_late_night_insomnia_routine", "화면"),
            ("요즘 사고싶은거", "korean_daily_icebreak_goods_item_interest", "노트"),
            ("평생한음식 고르라면", "korean_daily_icebreak_one_food_forever", "김치볶음밥"),
            ("맛집 고르는법", "korean_daily_restaurant_recommendation_no_fake_memory", "국밥"),
            ("커피vs차 골라", "korean_daily_icebreak_coffee_or_tea", "차 쪽"),
            ("아침 챙김?", "korean_daily_icebreak_breakfast_style", "아침"),
            ("매운거 가능?", "korean_daily_icebreak_spicy_food", "얼큰"),
            ("붕어빵팥슈 탕슉부찍", "korean_daily_icebreak_food_choice_combo", "찍먹"),
            ("해먹vs사먹", "korean_daily_icebreak_cook_or_buy_preference", "직접"),
            ("괴식조합 추천", "korean_daily_icebreak_unique_food_combo", "계란"),
            ("최근 젤맛있던거", "korean_daily_icebreak_recent_food_no_fake", "꾸미진"),
            ("비오는날 메뉴", "korean_daily_icebreak_rainy_day_food", "우동"),
            ("여행지 기억남는거", "korean_daily_expansion_memorable_trip_no_fake", "밤바다"),
            ("당장어디감", "korean_daily_icebreak_anywhere_city_choice", "바닷가"),
            ("계획파즉흥파", "korean_daily_travel_planning_style", "큰 틀"),
            ("어린시절 따뜻한거", "korean_daily_icebreak_childhood_warm_memory_no_fake", "어린 시절"),
            ("학창때 과목 하나", "korean_daily_expansion_school_subjects_no_fake", "문학"),
            ("살고싶은집", "korean_daily_icebreak_dream_house_city", "서재"),
            ("인생 짜릿했던거", "korean_daily_icebreak_adventurous_moment_no_fake", "짜릿했던 순간"),
            ("호캉스냐 캠핑이냐", "korean_daily_icebreak_hotel_or_camping", "호캉스"),
            ("여행갈때 필수템", "korean_daily_icebreak_travel_essential_item", "충전기"),
            ("최근사진픽", "korean_daily_practical_photo_pick", "선명한"),
            ("10억받으면 뭐부터", "korean_daily_money_ten_billion_first_action", "계좌"),
            ("순간이동 시간여행 골라", "korean_daily_icebreak_superpower_time_or_teleport", "순간 이동"),
            ("가치관 3개만", "korean_daily_icebreak_core_values", "정확함"),
            ("동물환생 고르면", "korean_daily_icebreak_animal_rebirth_choice", "고양이"),
            ("십년후 블랙", "korean_daily_icebreak_ten_year_self", "10년 뒤"),
            ("과거너한테 조언", "korean_daily_icebreak_past_self_advice_age", "스무 살"),
            ("블랙 세단어", "korean_daily_icebreak_three_words_self", "직설"),
            ("역사인물 만나면", "korean_daily_icebreak_historical_person_meet", "세종"),
            ("요즘 영감문장", "korean_daily_icebreak_inspiration_person_sentence", "정확하게"),
            ("다음 뭐팔까", "korean_daily_practical_selling_next_item", "수요"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-icebreak-chat-short-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_short_alias_false_positive_50_prompts_use_practical_context(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("다음 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("다음상품 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("이번주 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("재고 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("굿즈 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("중고 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("장터 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("플리마켓 뭐 팔까", "korean_daily_practical_selling_next_item", "수요"),
            ("상품 뭐 밀까", "korean_daily_practical_selling_next_item", "수요"),
            ("뭐 팔아야함", "korean_daily_practical_selling_next_item", "수요"),
            ("최근사진픽", "korean_daily_practical_photo_pick", "선명한"),
            ("사진픽 도와줘", "korean_daily_practical_photo_pick", "선명한"),
            ("프사픽", "korean_daily_practical_photo_pick", "선명한"),
            ("썸네일픽", "korean_daily_practical_photo_pick", "선명한"),
            ("업로드사진 골라줘", "korean_daily_practical_photo_pick", "선명한"),
            ("인스타사진 뭐올림", "korean_daily_practical_photo_pick", "선명한"),
            ("사진 고르는법", "korean_daily_practical_photo_pick", "선명한"),
            ("대표사진 뭐하지", "korean_daily_practical_photo_pick", "선명한"),
            ("프로필사진 픽", "korean_daily_practical_photo_pick", "선명한"),
            ("앨범사진 정리", "korean_daily_practical_photo_pick", "선명한"),
            ("살고싶은집 전세", "korean_daily_practical_house_choice", "누수"),
            ("살고싶은집 월세", "korean_daily_practical_house_choice", "누수"),
            ("드림홈 대출", "korean_daily_practical_house_choice", "누수"),
            ("드림하우스 매매", "korean_daily_practical_house_choice", "누수"),
            ("집계약 뭐부터", "korean_daily_practical_house_choice", "누수"),
            ("전세집 고르는법", "korean_daily_practical_house_choice", "누수"),
            ("월세집 체크리스트", "korean_daily_practical_house_choice", "누수"),
            ("이사갈집 고르기", "korean_daily_practical_house_choice", "누수"),
            ("방구할때 뭐봄", "korean_daily_practical_house_choice", "누수"),
            ("집볼때 필수체크", "korean_daily_practical_house_choice", "누수"),
            ("여행필템 광고", "korean_daily_practical_travel_item_content", "상황"),
            ("여행필템 팔까", "korean_daily_practical_travel_item_content", "상황"),
            ("여행필템 상품기획", "korean_daily_practical_travel_item_content", "상황"),
            ("여행필수템 콘텐츠", "korean_daily_practical_travel_item_content", "상황"),
            ("여행가방 광고문구", "korean_daily_practical_travel_item_content", "상황"),
            ("캐리어 광고 카피", "korean_daily_practical_travel_item_content", "상황"),
            ("여행템 추천글", "korean_daily_practical_travel_item_content", "상황"),
            ("여행템 썸네일", "korean_daily_practical_travel_item_content", "상황"),
            ("여행템 뭐팔까", "korean_daily_practical_travel_item_content", "상황"),
            ("여행필템 리스트업", "korean_daily_practical_travel_item_content", "상황"),
            ("커피차 가격", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 부를까", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 메뉴", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 창업", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 견적", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 이벤트", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 섭외", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 몇잔", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 예산", "korean_daily_practical_coffee_truck_check", "예산"),
            ("커피차 푸드트럭", "korean_daily_practical_coffee_truck_check", "예산"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-short-alias-false-positive-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_omitted_subject_practical_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("올릴거 골라줘", "korean_daily_practical_upload_pick", "한눈에"),
            ("뭐 올림", "korean_daily_practical_upload_pick", "한눈에"),
            ("뭐 올릴까", "korean_daily_practical_upload_pick", "한눈에"),
            ("업로드 뭐하지", "korean_daily_practical_upload_pick", "한눈에"),
            ("게시물 뭐감", "korean_daily_practical_upload_pick", "한눈에"),
            ("릴스 뭐올림", "korean_daily_practical_upload_pick", "한눈에"),
            ("쇼츠 뭐올림", "korean_daily_practical_upload_pick", "한눈에"),
            ("썸네일 뭐가 나음", "korean_daily_practical_upload_pick", "한눈에"),
            ("대표로 뭐씀", "korean_daily_practical_upload_pick", "한눈에"),
            ("올릴지 말지", "korean_daily_practical_upload_pick", "한눈에"),
            ("이거 팔까", "korean_daily_practical_selling_listing_check", "상태"),
            ("팔아도됨", "korean_daily_practical_selling_listing_check", "상태"),
            ("얼마에 올림", "korean_daily_practical_selling_listing_check", "상태"),
            ("가격 어케잡음", "korean_daily_practical_selling_listing_check", "상태"),
            ("중고 올릴까", "korean_daily_practical_selling_listing_check", "상태"),
            ("판매글 뭐부터", "korean_daily_practical_selling_listing_check", "상태"),
            ("내놓을까", "korean_daily_practical_selling_listing_check", "상태"),
            ("당근 올릴까", "korean_daily_practical_selling_listing_check", "상태"),
            ("팔릴까", "korean_daily_practical_selling_listing_check", "상태"),
            ("네고 받음?", "korean_daily_practical_selling_listing_check", "상태"),
            ("견적 ㄱ?", "korean_daily_practical_quote_budget_check", "예산"),
            ("견적 어케봄", "korean_daily_practical_quote_budget_check", "예산"),
            ("예산 얼마 잡지", "korean_daily_practical_quote_budget_check", "예산"),
            ("몇만원 봐야함", "korean_daily_practical_quote_budget_check", "예산"),
            ("비용 먼저?", "korean_daily_practical_quote_budget_check", "예산"),
            ("가격비교 ㄱ?", "korean_daily_practical_quote_budget_check", "예산"),
            ("업체 고르는법", "korean_daily_practical_quote_budget_check", "예산"),
            ("예약금 먼저?", "korean_daily_practical_quote_budget_check", "예산"),
            ("옵션 뺄까", "korean_daily_practical_quote_budget_check", "예산"),
            ("추가비 체크?", "korean_daily_practical_quote_budget_check", "예산"),
            ("몇개 준비?", "korean_daily_practical_quantity_buffer", "여분"),
            ("몇장 뽑아?", "korean_daily_practical_quantity_buffer", "여분"),
            ("몇명분?", "korean_daily_practical_quantity_buffer", "여분"),
            ("수량 어케잡음", "korean_daily_practical_quantity_buffer", "여분"),
            ("재고 몇개", "korean_daily_practical_quantity_buffer", "여분"),
            ("컵 몇개", "korean_daily_practical_quantity_buffer", "여분"),
            ("여분 몇개", "korean_daily_practical_quantity_buffer", "여분"),
            ("예약 몇자리", "korean_daily_practical_quantity_buffer", "여분"),
            ("몇개 사지", "korean_daily_practical_quantity_buffer", "여분"),
            ("몇인분?", "korean_daily_practical_quantity_buffer", "여분"),
            ("뭐부터 함", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("지금 뭐 먼저", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("순서 ㄱ", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("체크리스트 줘", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("우선순위 ㄱ", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("첫단추 뭐", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("정리 어케", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("준비 뭐부터", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("마감 전 뭐", "korean_daily_practical_ambiguous_first_step", "목적"),
            ("선택 기준 줘", "korean_daily_practical_ambiguous_first_step", "목적"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-omitted-subject-practical-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_multiturn_contextual_followup_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("사진 첫번째는 얼굴이 선명하고 두번째는 분위기가 좋아.", "두번째로 가?", "korean_daily_contextual_choice_second", "후자"),
            ("썸네일 첫번째는 글자가 크고 두번째는 표정이 살아.", "첫번째가 낫나?", "korean_daily_contextual_choice_first", "전자"),
            ("릴스 첫번째는 정보형이고 두번째는 웃긴 장면이야.", "둘 다 올릴까?", "korean_daily_contextual_choice_both", "둘 다"),
            ("후보 첫번째는 깔끔한데 두번째는 너무 튀어.", "둘 다 별로지?", "korean_daily_contextual_choice_neither", "둘 다 별로"),
            ("선물 첫번째는 실용적이고 두번째는 감성템이야.", "다른 거 볼까?", "korean_daily_contextual_choice_other", "다른"),
            ("집 첫번째는 역세권이고 두번째는 채광이 좋아.", "두번째 괜찮을까?", "korean_daily_contextual_choice_second", "후자"),
            ("카페 첫번째는 조용하고 두번째는 디저트가 좋아.", "첫번째로 할까?", "korean_daily_contextual_choice_first", "전자"),
            ("코디 첫번째는 단정하고 두번째는 좀 힙해.", "후자가 맞나?", "korean_daily_contextual_choice_second", "후자"),
            ("문구 첫번째는 직설적이고 두번째는 부드러워.", "전자가 나아?", "korean_daily_contextual_choice_first", "전자"),
            ("일정 첫번째는 오늘 끝내는 거고 두번째는 내일 오전에 하는 거야.", "둘 중엔 후자?", "korean_daily_contextual_choice_second", "후자"),
            ("업체 1안은 싸고 2안은 응대가 빨라.", "2번으로 가?", "korean_daily_contextual_choice_second", "후자"),
            ("계획 1안은 바로 출발이고 2안은 밥 먹고 출발이야.", "1번이 낫나?", "korean_daily_contextual_choice_first", "전자"),
            ("예산 1안은 최소 구성이고 2안은 안전하게 여유 둔 구성.", "둘 다 가능?", "korean_daily_contextual_choice_both", "둘 다"),
            ("초안 1안은 너무 딱딱하고 2안은 너무 장난스러워.", "둘 다 애매하지?", "korean_daily_contextual_choice_neither", "둘 다 별로"),
            ("좌석 1안은 앞자리, 2안은 통로 쪽이야.", "다른 쪽 찾아볼까?", "korean_daily_contextual_choice_other", "다른"),
            ("구매 1안은 새 제품이고 2안은 중고 미개봉이야.", "2안 괜찮아?", "korean_daily_contextual_choice_second", "후자"),
            ("숙소 1안은 위치가 좋고 2안은 방이 넓어.", "1안으로?", "korean_daily_contextual_choice_first", "전자"),
            ("배송 1안은 빠르고 2안은 무료야.", "후자 갈까?", "korean_daily_contextual_choice_second", "후자"),
            ("메뉴 1안은 국밥이고 2안은 샐러드야.", "전자?", "korean_daily_contextual_choice_first", "전자"),
            ("운동 1안은 헬스고 2안은 수영이야.", "둘 다 하면 무리?", "korean_daily_contextual_choice_both", "둘 다"),
            ("A안은 가격이 낮고 B안은 유지보수가 편해.", "B안으로 가?", "korean_daily_contextual_choice_second", "후자"),
            ("A는 디자인이 좋고 B는 배터리가 오래가.", "A가 낫나?", "korean_daily_contextual_choice_first", "전자"),
            ("A컷은 표정이 좋고 B컷은 구도가 좋아.", "B컷 올릴까?", "korean_daily_contextual_choice_second", "후자"),
            ("A업체는 싸고 B업체는 날짜를 맞춰줘.", "둘 다 견적 받을까?", "korean_daily_contextual_choice_both", "둘 다"),
            ("A문장은 너무 세고 B문장은 너무 흐려.", "둘 다 별론가?", "korean_daily_contextual_choice_neither", "둘 다 별로"),
            ("A코스는 가까운데 B코스는 볼 게 많아.", "다른 코스 볼까?", "korean_daily_contextual_choice_other", "다른"),
            ("A좌석은 시야가 좋고 B좌석은 출입이 편해.", "B로?", "korean_daily_contextual_choice_second", "후자"),
            ("A플랜은 빠르고 B플랜은 안정적이야.", "A로 갈까?", "korean_daily_contextual_choice_first", "전자"),
            ("A버전은 설명이 길고 B버전은 짧아.", "후자가 나아?", "korean_daily_contextual_choice_second", "후자"),
            ("A색은 차분하고 B색은 눈에 띄어.", "전자가 맞지?", "korean_daily_contextual_choice_first", "전자"),
            ("하나는 지금 바로 살 수 있고 다른 하나는 일주일 기다려야 해.", "두번째는 별로?", "korean_daily_contextual_choice_second", "후자"),
            ("하나는 집에서 가깝고 다른 하나는 시설이 좋아.", "첫번째로 할까?", "korean_daily_contextual_choice_first", "전자"),
            ("하나는 톤이 밝고 다른 하나는 정보가 많아.", "둘 다 섞을까?", "korean_daily_contextual_choice_both", "둘 다"),
            ("하나는 너무 비싸고 다른 하나는 후기가 별로야.", "둘 다 답없지?", "korean_daily_contextual_choice_neither", "둘 다 별로"),
            ("하나는 내가 하고 싶은 거고 다른 하나는 해야 하는 거야.", "다른 방법 없나?", "korean_daily_contextual_choice_other", "다른"),
            ("한쪽은 가격이 좋고 다른쪽은 품질이 좋아.", "후자 쪽?", "korean_daily_contextual_choice_second", "후자"),
            ("한쪽은 빨리 끝나고 다른쪽은 오래 걸려.", "전자 가?", "korean_daily_contextual_choice_first", "전자"),
            ("첫 후보는 안전하고 두 번째 후보는 재미있어.", "두 번째 후보 괜찮지?", "korean_daily_contextual_choice_second", "후자"),
            ("첫 안은 무난하고 두 번째 안은 반응이 클 것 같아.", "첫 안으로?", "korean_daily_contextual_choice_first", "전자"),
            ("첫 선택지는 돈이 덜 들고 두 번째 선택지는 시간이 덜 들어.", "둘 다 장단점 있네?", "korean_daily_contextual_choice_both", "둘 다"),
            ("사진 첫번째는 얼굴이 선명하고 두번째는 분위기가 좋아.", "근데 좀 과한가?", "korean_daily_contextual_discourse_contrast", "근데"),
            ("카페 첫번째는 조용하고 두번째는 디저트가 좋아.", "그래도 디저트가 끌리긴 해", "korean_daily_contextual_discourse_concession", "그래도"),
            ("집 첫번째는 역세권이고 두번째는 채광이 좋아.", "그럼 일단 뭐부터 봐?", "korean_daily_contextual_discourse_next_step", "그럼 일단"),
            ("문구 첫번째는 직설적이고 두번째는 부드러워.", "아니면 더 짧게 갈까?", "korean_daily_contextual_discourse_alternative", "아니면"),
            ("A는 디자인이 좋고 B는 배터리가 오래가.", "맞아 그쪽이 낫긴 해", "korean_daily_contextual_discourse_agreement", "맞아"),
            ("중고 거래 올릴 사진 얘기했잖아.", "그거 괜찮아?", "korean_daily_contextual_reference_item", "물건"),
            ("친구한테 보낼 사과 카톡 얘기였어.", "그 말 좀 세?", "korean_daily_contextual_reference_social", "그 사람"),
            ("회의 끝나고 할 업무 얘기했잖아.", "그 일 먼저 해야 해?", "korean_daily_contextual_reference_task", "그 일"),
            ("여행 숙소랑 카페 얘기 중이었어.", "거기 너무 멀까?", "korean_daily_contextual_reference_place", "거기"),
            ("저녁 메뉴로 국밥이랑 샐러드 얘기했어.", "그거 먹어도 될까?", "korean_daily_contextual_reference_food", "먹는 얘기"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (previous, current, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, current=current):
                user_id = f"offline-multiturn-contextual-followup-50-{index}"
                await engine.respond(user_id, previous)
                result = await engine.respond(user_id, current)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_multiturn_three_choice_followup_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("사진 첫번째는 얼굴, 두번째는 분위기, 세번째는 색감이 좋아.", "세번째로 가?", "korean_daily_contextual_choice_third", "세번째"),
            ("썸네일 첫번째는 글자, 두번째는 표정, 세번째는 배경이 좋아.", "3번이 낫나?", "korean_daily_contextual_choice_third", "세번째"),
            ("릴스 첫번째는 정보형, 두번째는 웃긴 장면, 세번째는 후기형이야.", "셋째 올릴까?", "korean_daily_contextual_choice_third", "세번째"),
            ("집 1안은 역세권, 2안은 채광, 3안은 관리비가 좋아.", "3안 괜찮아?", "korean_daily_contextual_choice_third", "세번째"),
            ("카페 1안은 조용하고 2안은 디저트, 3안은 위치가 좋아.", "3번으로?", "korean_daily_contextual_choice_third", "세번째"),
            ("코디 1안은 단정하고 2안은 힙하고 3안은 편해.", "삼번 갈까?", "korean_daily_contextual_choice_third", "세번째"),
            ("문구 1안은 직설적이고 2안은 부드럽고 3안은 짧아.", "삼안이 맞나?", "korean_daily_contextual_choice_third", "세번째"),
            ("일정 1안은 오늘 끝내기, 2안은 내일 오전, 3안은 주말 몰아서야.", "셋 중엔 3번?", "korean_daily_contextual_choice_third", "세번째"),
            ("업체 A안은 싸고 B안은 빠르고 C안은 응대가 좋아.", "C안으로 가?", "korean_daily_contextual_choice_third", "세번째"),
            ("계획 A는 바로 출발, B는 밥 먹고 출발, C는 내일 출발이야.", "C가 낫나?", "korean_daily_contextual_choice_third", "세번째"),
            ("예산 A는 최소, B는 표준, C는 여유 구성.", "C 쪽?", "korean_daily_contextual_choice_third", "세번째"),
            ("초안 A는 딱딱하고 B는 장난스럽고 C는 담백해.", "C로 갈까?", "korean_daily_contextual_choice_third", "세번째"),
            ("좌석 A는 앞자리, B는 통로, C는 뒤쪽 가운데야.", "C좌석 괜찮아?", "korean_daily_contextual_choice_third", "세번째"),
            ("구매 A는 새 제품, B는 중고 미개봉, C는 렌탈이야.", "C 선택?", "korean_daily_contextual_choice_third", "세번째"),
            ("숙소 A는 위치, B는 방 크기, C는 조식이 좋아.", "C안 별로야?", "korean_daily_contextual_choice_third", "세번째"),
            ("배송 A는 빠르고 B는 무료고 C는 포장이 좋아.", "C로?", "korean_daily_contextual_choice_third", "세번째"),
            ("메뉴 A는 국밥, B는 샐러드, C는 돈까스야.", "C 먹을까?", "korean_daily_contextual_choice_third", "세번째"),
            ("운동 A는 헬스, B는 수영, C는 필라테스야.", "C가 덜 무리?", "korean_daily_contextual_choice_third", "세번째"),
            ("선물 후보 첫번째는 실용템, 두번째는 감성템, 세번째는 소모품이야.", "세 번째 후보?", "korean_daily_contextual_choice_third", "세번째"),
            ("여행 코스 첫 안은 바다, 두 번째 안은 산, 세 번째 안은 도심이야.", "세 번째 안으로?", "korean_daily_contextual_choice_third", "세번째"),
            ("사진 첫번째는 얼굴, 두번째는 분위기, 세번째는 색감이 좋아.", "첫번째로 가?", "korean_daily_contextual_choice_first", "전자"),
            ("썸네일 첫번째는 글자, 두번째는 표정, 세번째는 배경이 좋아.", "두번째가 낫나?", "korean_daily_contextual_choice_second", "후자"),
            ("릴스 첫번째는 정보형, 두번째는 웃긴 장면, 세번째는 후기형이야.", "둘 다 말고 세번째?", "korean_daily_contextual_choice_third", "세번째"),
            ("집 1안은 역세권, 2안은 채광, 3안은 관리비가 좋아.", "1안으로?", "korean_daily_contextual_choice_first", "전자"),
            ("카페 1안은 조용하고 2안은 디저트, 3안은 위치가 좋아.", "2안이 끌려", "korean_daily_contextual_choice_second", "후자"),
            ("코디 1안은 단정하고 2안은 힙하고 3안은 편해.", "1번?", "korean_daily_contextual_choice_first", "전자"),
            ("문구 1안은 직설적이고 2안은 부드럽고 3안은 짧아.", "2번 갈까?", "korean_daily_contextual_choice_second", "후자"),
            ("일정 1안은 오늘 끝내기, 2안은 내일 오전, 3안은 주말 몰아서야.", "둘 중엔 2번?", "korean_daily_contextual_choice_second", "후자"),
            ("업체 A안은 싸고 B안은 빠르고 C안은 응대가 좋아.", "A안으로?", "korean_daily_contextual_choice_first", "전자"),
            ("계획 A는 바로 출발, B는 밥 먹고 출발, C는 내일 출발이야.", "B가 맞나?", "korean_daily_contextual_choice_second", "후자"),
            ("예산 A는 최소, B는 표준, C는 여유 구성.", "A 쪽?", "korean_daily_contextual_choice_first", "전자"),
            ("초안 A는 딱딱하고 B는 장난스럽고 C는 담백해.", "B는 별로?", "korean_daily_contextual_choice_second", "후자"),
            ("좌석 A는 앞자리, B는 통로, C는 뒤쪽 가운데야.", "A좌석 괜찮아?", "korean_daily_contextual_choice_first", "전자"),
            ("구매 A는 새 제품, B는 중고 미개봉, C는 렌탈이야.", "B 선택?", "korean_daily_contextual_choice_second", "후자"),
            ("숙소 A는 위치, B는 방 크기, C는 조식이 좋아.", "A안으로?", "korean_daily_contextual_choice_first", "전자"),
            ("배송 A는 빠르고 B는 무료고 C는 포장이 좋아.", "B로?", "korean_daily_contextual_choice_second", "후자"),
            ("메뉴 A는 국밥, B는 샐러드, C는 돈까스야.", "A 먹을까?", "korean_daily_contextual_choice_first", "전자"),
            ("운동 A는 헬스, B는 수영, C는 필라테스야.", "B가 덜 무리?", "korean_daily_contextual_choice_second", "후자"),
            ("선물 후보 첫번째는 실용템, 두번째는 감성템, 세번째는 소모품이야.", "첫 번째 후보?", "korean_daily_contextual_choice_first", "전자"),
            ("여행 코스 첫 안은 바다, 두 번째 안은 산, 세 번째 안은 도심이야.", "두 번째 안으로?", "korean_daily_contextual_choice_second", "후자"),
            ("사진 첫번째는 얼굴, 두번째는 분위기, 세번째는 색감이 좋아.", "셋 다 괜찮나?", "korean_daily_contextual_choice_all", "셋 다"),
            ("집 1안은 역세권, 2안은 채광, 3안은 관리비가 좋아.", "셋 다 별로는 아니지?", "korean_daily_contextual_choice_all", "셋 다"),
            ("업체 A안은 싸고 B안은 빠르고 C안은 응대가 좋아.", "다 괜찮으면?", "korean_daily_contextual_choice_all", "셋 다"),
            ("메뉴 A는 국밥, B는 샐러드, C는 돈까스야.", "세 개 다 먹기는 무리?", "korean_daily_contextual_choice_all", "셋 다"),
            ("초안 A는 딱딱하고 B는 장난스럽고 C는 담백해.", "셋 다 애매하지?", "korean_daily_contextual_choice_neither", "다 별로"),
            ("좌석 A는 앞자리, B는 통로, C는 뒤쪽 가운데야.", "다른 자리 볼까?", "korean_daily_contextual_choice_other", "다른"),
            ("선물 후보 첫번째는 실용템, 두번째는 감성템, 세번째는 소모품이야.", "셋 말고 다른 거?", "korean_daily_contextual_choice_other", "다른"),
            ("여행 코스 첫 안은 바다, 두 번째 안은 산, 세 번째 안은 도심이야.", "근데 3안은 좀 빡세지?", "korean_daily_contextual_choice_third", "세번째"),
            ("배송 A는 빠르고 B는 무료고 C는 포장이 좋아.", "C가 좋긴 한데 과한가?", "korean_daily_contextual_choice_third", "세번째"),
            ("코디 1안은 단정하고 2안은 힙하고 3안은 편해.", "제3안으로 틀까?", "korean_daily_contextual_choice_third", "세번째"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (previous, current, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, current=current):
                user_id = f"offline-multiturn-three-choice-followup-50-{index}"
                await engine.respond(user_id, previous)
                result = await engine.respond(user_id, current)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_bot_reply_choice_followup_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("내 답은 이거야. A안은 가격이 낮고 B안은 유지보수가 편해.", "B안으로 가?", "korean_daily_contextual_choice_second", "후자"),
            ("정리하면 A는 디자인이 좋고 B는 배터리가 오래가.", "A가 낫나?", "korean_daily_contextual_choice_first", "전자"),
            ("사진은 A컷은 표정이 좋고 B컷은 구도가 좋아.", "B컷 올릴까?", "korean_daily_contextual_choice_second", "후자"),
            ("업체는 A업체는 싸고 B업체는 날짜를 맞춰줘.", "둘 다 견적 받을까?", "korean_daily_contextual_choice_both", "둘 다"),
            ("문장은 A문장은 너무 세고 B문장은 너무 흐려.", "둘 다 별론가?", "korean_daily_contextual_choice_neither", "둘 다 별로"),
            ("코스는 A코스는 가까운데 B코스는 볼 게 많아.", "다른 코스 볼까?", "korean_daily_contextual_choice_other", "다른"),
            ("좌석은 A좌석은 시야가 좋고 B좌석은 출입이 편해.", "B로?", "korean_daily_contextual_choice_second", "후자"),
            ("플랜은 A플랜은 빠르고 B플랜은 안정적이야.", "A로 갈까?", "korean_daily_contextual_choice_first", "전자"),
            ("버전은 A버전은 설명이 길고 B버전은 짧아.", "후자가 나아?", "korean_daily_contextual_choice_second", "후자"),
            ("색은 A색은 차분하고 B색은 눈에 띄어.", "전자가 맞지?", "korean_daily_contextual_choice_first", "전자"),
            ("내가 보기엔 첫번째는 얼굴이 선명하고 두번째는 분위기가 좋아.", "두번째로 가?", "korean_daily_contextual_choice_second", "후자"),
            ("정리하면 첫번째는 글자가 크고 두번째는 표정이 살아.", "첫번째가 낫나?", "korean_daily_contextual_choice_first", "전자"),
            ("선택지는 첫번째는 정보형이고 두번째는 웃긴 장면이야.", "둘 다 올릴까?", "korean_daily_contextual_choice_both", "둘 다"),
            ("후보는 첫번째는 깔끔한데 두번째는 너무 튀어.", "둘 다 별로지?", "korean_daily_contextual_choice_neither", "둘 다 별로"),
            ("선물은 첫번째는 실용적이고 두번째는 감성템이야.", "다른 거 볼까?", "korean_daily_contextual_choice_other", "다른"),
            ("집은 첫번째는 역세권이고 두번째는 채광이 좋아.", "두번째 괜찮을까?", "korean_daily_contextual_choice_second", "후자"),
            ("카페는 첫번째는 조용하고 두번째는 디저트가 좋아.", "첫번째로 할까?", "korean_daily_contextual_choice_first", "전자"),
            ("코디는 첫번째는 단정하고 두번째는 좀 힙해.", "후자가 맞나?", "korean_daily_contextual_choice_second", "후자"),
            ("문구는 첫번째는 직설적이고 두번째는 부드러워.", "전자가 나아?", "korean_daily_contextual_choice_first", "전자"),
            ("일정은 첫번째는 오늘 끝내는 거고 두번째는 내일 오전에 하는 거야.", "둘 중엔 후자?", "korean_daily_contextual_choice_second", "후자"),
            ("세 개로 보면 A안은 싸고 B안은 빠르고 C안은 응대가 좋아.", "C안으로 가?", "korean_daily_contextual_choice_third", "세번째"),
            ("계획은 A는 바로 출발, B는 밥 먹고 출발, C는 내일 출발이야.", "C가 낫나?", "korean_daily_contextual_choice_third", "세번째"),
            ("예산은 A는 최소, B는 표준, C는 여유 구성.", "C 쪽?", "korean_daily_contextual_choice_third", "세번째"),
            ("초안은 A는 딱딱하고 B는 장난스럽고 C는 담백해.", "C로 갈까?", "korean_daily_contextual_choice_third", "세번째"),
            ("좌석은 A는 앞자리, B는 통로, C는 뒤쪽 가운데야.", "C좌석 괜찮아?", "korean_daily_contextual_choice_third", "세번째"),
            ("구매는 A는 새 제품, B는 중고 미개봉, C는 렌탈이야.", "C 선택?", "korean_daily_contextual_choice_third", "세번째"),
            ("숙소는 A는 위치, B는 방 크기, C는 조식이 좋아.", "C안 별로야?", "korean_daily_contextual_choice_third", "세번째"),
            ("배송은 A는 빠르고 B는 무료고 C는 포장이 좋아.", "C로?", "korean_daily_contextual_choice_third", "세번째"),
            ("메뉴는 A는 국밥, B는 샐러드, C는 돈까스야.", "C 먹을까?", "korean_daily_contextual_choice_third", "세번째"),
            ("운동은 A는 헬스, B는 수영, C는 필라테스야.", "C가 덜 무리?", "korean_daily_contextual_choice_third", "세번째"),
            ("선물 후보는 첫번째는 실용템, 두번째는 감성템, 세번째는 소모품이야.", "세 번째 후보?", "korean_daily_contextual_choice_third", "세번째"),
            ("여행 코스는 첫 안은 바다, 두 번째 안은 산, 세 번째 안은 도심이야.", "세 번째 안으로?", "korean_daily_contextual_choice_third", "세번째"),
            ("사진 후보는 첫번째는 얼굴, 두번째는 분위기, 세번째는 색감이 좋아.", "셋 다 괜찮나?", "korean_daily_contextual_choice_all", "셋 다"),
            ("집 후보는 1안은 역세권, 2안은 채광, 3안은 관리비가 좋아.", "셋 다 별로는 아니지?", "korean_daily_contextual_choice_all", "셋 다"),
            ("업체는 A안은 싸고 B안은 빠르고 C안은 응대가 좋아.", "다 괜찮으면?", "korean_daily_contextual_choice_all", "셋 다"),
            ("메뉴는 A는 국밥, B는 샐러드, C는 돈까스야.", "세 개 다 먹기는 무리?", "korean_daily_contextual_choice_all", "셋 다"),
            ("초안은 A는 딱딱하고 B는 장난스럽고 C는 담백해.", "셋 다 애매하지?", "korean_daily_contextual_choice_neither", "다 별로"),
            ("좌석은 A는 앞자리, B는 통로, C는 뒤쪽 가운데야.", "다른 자리 볼까?", "korean_daily_contextual_choice_other", "다른"),
            ("선물 후보는 첫번째는 실용템, 두번째는 감성템, 세번째는 소모품이야.", "셋 말고 다른 거?", "korean_daily_contextual_choice_other", "다른"),
            ("여행 코스는 첫 안은 바다, 두 번째 안은 산, 세 번째 안은 도심이야.", "근데 3안은 좀 빡세지?", "korean_daily_contextual_choice_third", "세번째"),
            ("배송은 A는 빠르고 B는 무료고 C는 포장이 좋아.", "C가 좋긴 한데 과한가?", "korean_daily_contextual_choice_third", "세번째"),
            ("코디는 1안은 단정하고 2안은 힙하고 3안은 편해.", "제3안으로 틀까?", "korean_daily_contextual_choice_third", "세번째"),
            ("둘로 줄이면 하나는 지금 바로 살 수 있고 다른 하나는 일주일 기다려야 해.", "두번째는 별로?", "korean_daily_contextual_choice_second", "후자"),
            ("둘로 보면 하나는 집에서 가깝고 다른 하나는 시설이 좋아.", "첫번째로 할까?", "korean_daily_contextual_choice_first", "전자"),
            ("둘로 보면 하나는 톤이 밝고 다른 하나는 정보가 많아.", "둘 다 섞을까?", "korean_daily_contextual_choice_both", "둘 다"),
            ("둘로 보면 하나는 너무 비싸고 다른 하나는 후기가 별로야.", "둘 다 답없지?", "korean_daily_contextual_choice_neither", "둘 다 별로"),
            ("둘로 보면 하나는 하고 싶은 거고 다른 하나는 해야 하는 거야.", "다른 방법 없나?", "korean_daily_contextual_choice_other", "다른"),
            ("한쪽은 가격이 좋고 다른쪽은 품질이 좋아.", "후자 쪽?", "korean_daily_contextual_choice_second", "후자"),
            ("한쪽은 빨리 끝나고 다른쪽은 오래 걸려.", "전자 가?", "korean_daily_contextual_choice_first", "전자"),
            ("첫 후보는 안전하고 두 번째 후보는 재미있어.", "두 번째 후보 괜찮지?", "korean_daily_contextual_choice_second", "후자"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (bot_text, current, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, current=current):
                user_id = f"offline-bot-reply-choice-followup-50-{index}"
                engine.state_store.append_turn(
                    user_id,
                    TurnRecord(
                        user_text="방금 네가 선택지를 정리해줬어.",
                        bot_text=bot_text,
                        action=ActionType.SMALL_TALK,
                        decision_reason="test_choice_context",
                    ),
                )
                result = await engine.respond(user_id, current)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_user_supplied_icebreak_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("오늘 아침에 일어나서 가장 먼저 한 생각은 무엇인가요?", "korean_daily_icebreak_morning_first_thought", "빨리"),
            ("요즘 하루 중 가장 행복하다고 느끼는 순간은 언제인가요?", "korean_daily_icebreak_happy_moment_day", "행복"),
            ("이번 주에 있었던 일 중 가장 기억에 남는 재미있는 에피소드가 있나요?", "korean_daily_icebreak_funny_episode_no_fake", "에피소드"),
            ("요즘 퇴근 후나 주말에 주로 어떻게 시간을 보내시나요?", "korean_daily_icebreak_after_work_weekend_rhythm", "리듬"),
            ("오늘 하루를 세 가지 단어로 표현한다면 무엇인가요?", "korean_daily_icebreak_three_words_self", "직설"),
            ("최근에 새로 알게 된 흥미로운 소식이나 정보가 있나요?", "korean_daily_icebreak_interesting_fact", "의외성"),
            ("보통 스트레스를 받으면 어떻게 푸는 편인가요?", "korean_daily_basic_stress_relief", "몸"),
            ("요즘 날씨에 가장 하고 싶은 활동이 있다면 무엇인가요?", "korean_daily_icebreak_weather_activity", "산책"),
            ("하루 중 나만의 작은 '소확행(소소하지만 확실한 행복)'은 무엇인가요?", "korean_daily_icebreak_small_happiness", "소확행"),
            ("내일 하루 동안 쓸 수 있는 자유시간이 온전히 주어진다면 무엇을 하고 싶나요?", "korean_daily_icebreak_free_time_tomorrow", "자유시간"),
            ("최근에 재미있게 본 영화, 드라마, 혹은 책이 있나요? 추천해 주세요!", "korean_daily_icebreak_recent_media_no_fake", "미스터리"),
            ("인생 영화나 인생 드라마를 하나만 꼽으라면 무엇인가요?", "korean_daily_icebreak_life_movie_recommendation", "월-E"),
            ("평소에 자주 듣는 음악 장르나 좋아하는 아티스트는 누구인가요?", "korean_daily_icebreak_music_genre_artist", "인디"),
            ("새로 배워보고 싶은 취미나 기술이 있다면 무엇인가요?", "korean_daily_new_hobby_preference", "드로잉"),
            ("최근 가장 자주 사용하는 스마트폰 앱은 무엇인가요?", "korean_daily_icebreak_frequent_app", "메모"),
            ("집에서 쉬는 것(집돌이/집순이)을 좋아하나요, 밖으로 나가는 것을 좋아하나요?", "korean_daily_icebreak_home_or_out_preference", "집에서"),
            ("평소 유튜브에서 주로 어떤 카테고리의 영상을 보시나요?", "korean_daily_icebreak_youtube_category", "지식"),
            ("운동하는 것을 좋아하시나요? 좋아하는 운동이 있다면 무엇인가요?", "korean_daily_icebreak_favorite_exercise", "걷기"),
            ("나만 알고 있는 숨겨진 취미나 독특한 관심사가 있나요?", "korean_daily_icebreak_hidden_hobby_interest", "맥락"),
            ("전시회나 콘서트 같은 문화 예술 공연을 자주 관람하시는 편인가요?", "korean_daily_icebreak_culture_performance_frequency", "사진전"),
            ("세상에서 가장 좋아하는 소울 푸드(Soul Food)는 무엇인가요?", "korean_daily_icebreak_soul_food", "김치볶음밥"),
            ("최근에 가본 맛집 중에 기억에 남는 곳이 있나요?", "korean_daily_icebreak_restaurant_memory_no_fake", "국밥"),
            ("평소에 요리하는 것을 즐기시나요? 자신 있는 요리는 무엇인가요?", "korean_daily_icebreak_cooking_recipe", "김치볶음밥"),
            ("아침 식사는 꼭 챙겨 드시는 편인가요? 보통 무엇을 드시나요?", "korean_daily_icebreak_breakfast_style", "아침"),
            ("일주일 동안 딱 한 가지 음식만 먹어야 한다면 무엇을 고르시겠어요?", "korean_daily_icebreak_one_food_forever", "김치볶음밥"),
            ("커피나 차 중에서 어떤 것을 더 선호하시나요? 좋아하는 메뉴는 무엇인가요?", "korean_daily_icebreak_coffee_or_tea", "차"),
            ("민초(민트초코)나 하와이안 피자(파인애플 피자)에 대한 본인의 취향은 어떤가요?", "korean_daily_mint_choco_pineapple_pizza_preference", "반민초"),
            ("매운 음식을 잘 드시는 편인가요? 좋아하는 매운 음식이 있나요?", "korean_daily_icebreak_spicy_food", "얼큰"),
            ("평생 디저트를 끊기 vs 평생 야식 끊기 중 하나만 선택한다면?", "korean_daily_icebreak_dessert_or_late_night", "야식"),
            ("여행을 갈 때 맛집 탐방이 얼마나 중요한가요?", "korean_daily_icebreak_travel_food_importance", "맛집"),
            ("스스로 생각하기에 본인은 외향적인 편인가요, 내향적인 편인가요? (MBTI는 무엇인가요?)", "korean_daily_icebreak_introvert_extrovert_mbti", "내향"),
            ("대화할 때 주로 이야기를 듣는 편인가요, 아니면 이끌어가는 편인가요?", "korean_daily_icebreak_conversation_style", "핵심"),
            ("인생에서 가장 중요하게 생각하는 가치관은 무엇인가요?", "korean_daily_icebreak_core_values", "정확함"),
            ("나를 가장 잘 설명해 주는 형용사 세 가지는 무엇일까요?", "korean_daily_icebreak_three_adjectives", "차분한"),
            ("힘든 일이 있을 때 주로 누구에게 털어놓거나 조언을 구하나요?", "korean_daily_icebreak_hard_day_confide", "현실감"),
            ("칭찬을 들었을 때 가장 기분 좋은 말은 무엇인가요?", "korean_daily_praise_memory_persona", "칭찬"),
            ("새로운 사람을 만날 때 가장 먼저 보게 되는 부분이나 매력을 느끼는 포인트는 무엇인가요?", "korean_daily_icebreak_new_person_charm_point", "말투"),
            ("갈등이 생겼을 때 보통 어떻게 해결하는 편인가요?", "korean_daily_icebreak_conflict_style", "분리"),
            ("슬럼프가 찾아왔을 때 극복하는 나만의 방법이 있나요?", "korean_daily_icebreak_slump_recovery", "슬럼프"),
            ("최근 나 자신에 대해 새롭게 깨달은 사실이 있나요?", "korean_daily_icebreak_self_realization", "맥락"),
            ("타임머신이 있다면 과거와 미래 중 어디로 가보고 싶나요?", "korean_daily_time_machine_choice", "미래"),
            ("만약 내일 당장 복권 1등에 당첨된다면 가장 먼저 무엇을 하고 싶나요?", "korean_daily_icebreak_lottery_first_action", "세무"),
            ("아무런 제약 없이 한 달 동안 다른 나라에서 살 수 있다면 어느 나라를 선택하시겠어요?", "korean_daily_icebreak_one_month_abroad", "생활 리듬"),
            ("동물과 대화할 수 있는 능력이 생긴다면 어떤 동물과 먼저 얘기해 보고 싶나요?", "korean_daily_icebreak_talk_to_pet", "아픈 데"),
            ("하루 동안 다른 사람의 인생을 살아볼 수 있다면 누구의 삶을 경험해 보고 싶나요?", "korean_daily_icebreak_live_other_life", "평범한"),
            ("만약 초능력을 하나 가질 수 있다면 어떤 능력을 갖고 싶나요?", "korean_daily_icebreak_superpower_choice", "순간이동"),
            ("무인도에 딱 세 가지만 가지고 갈 수 있다면 무엇을 가져가시겠어요?", "korean_daily_foundation_desert_island_three_items", "정수 필터"),
            ("역사 속 인물 중 한 명을 만나 저녁 식사를 함께할 수 있다면 누구를 만나고 싶나요?", "korean_daily_icebreak_historical_dinner", "세종"),
            ("만약 내가 영화나 드라마의 주인공이 된다면 어떤 장르의 작품이었으면 좋겠나요?", "korean_daily_expansion_protagonist_genre", "SF"),
            ("평생 늙지 않는 불로불사의 몸이 된다면 행복할 것 같나요, 아니면 슬플 것 같나요?", "korean_daily_immortality_pill_choice", "안 먹"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "일단 기준",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-user-icebreak-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("resolver"), "draft_reason_silver_frame_v1")
                self.assertEqual(semantic_frame.get("source_reason"), expected_reason)
                self.assertTrue(semantic_frame.get("draft_frame"))
                self.assertNotEqual(semantic_frame.get("schema"), "direct_reply")
                self.assertNotEqual(semantic_frame.get("draft_frame_family"), "social_acknowledgement")
                self.assertIn(
                    semantic_frame.get("draft_frame_family"),
                    {
                        "social_acknowledgement",
                        "emotional_support",
                        "practical_guidance",
                        "choice_preference",
                        "playful_output",
                        "reflective_position",
                        "identity_boundary",
                        "situational_tactic",
                    },
                )
                self.assertIn(
                    semantic_frame.get("priority"),
                    {"immediate_action", "practical_action", "emotion_stabilization", "choice_judgment", "meta_reflection"},
                )
                self.assertIn(
                    "draft_frame",
                    {signal.get("axis") for signal in semantic_frame.get("signals", []) if isinstance(signal, dict)},
                )
                frame_targets = semantic_frame.get("targets") or {}
                self.assertEqual(frame_targets.get("draft_frame"), semantic_frame.get("draft_frame"))
                if "no_fake" in expected_reason or expected_reason in {
                    "korean_daily_icebreak_recent_media_no_fake",
                    "korean_daily_icebreak_restaurant_memory_no_fake",
                }:
                    self.assertTrue(semantic_frame.get("no_fake"))
                    self.assertEqual(semantic_frame.get("schema"), "honesty_boundary")
                if any(token in expected_reason for token in ("basic_stress", "hard_day", "slump")):
                    self.assertEqual(semantic_frame.get("priority"), "emotion_stabilization")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_user_supplied_polite_basic_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("좋은 아침이에요! 오늘 날씨 정말 좋네요.", "korean_daily_basic_good_morning", "물 한 모금"),
            ("요즘 어떻게 지내세요?", "korean_daily_basic_how_are_you", "대기 중"),
            ("주말 잘 보내셨어요?", "korean_daily_basic_weekend_checkin", "쉬는 맛"),
            ("오랜만이에요! 그동안 잘 지내셨죠?", "korean_daily_basic_long_time_no_see", "다시 말 걸어주는"),
            ("얼굴이 좋아 보이시네요!", "korean_daily_basic_appearance_compliment", "덜 구겨진"),
            ("오늘 하루도 화이팅하세요!", "korean_daily_basic_cheer_received", "응원 접수"),
            ("늦어서 죄송해요. 많이 기다리셨죠?", "korean_daily_basic_late_apology_reassurance", "지금 온 게 중요"),
            ("지금 퇴근하시는 길이에요?", "korean_daily_basic_after_work_transition", "문 앞에서 끊고"),
            ("밤이 늦었네요. 집에 조심히 들어가세요.", "korean_daily_basic_safe_way_home", "안전이 먼저"),
            ("오늘 시간 내주셔서 정말 고마웠어요.", "korean_daily_basic_time_thanks", "따뜻하게 남았"),
            ("오늘 날씨 너무 덥지 않아요?", "korean_daily_basic_hot_weather_chat", "의욕보다 물"),
            ("비가 올 것 같은데 우산 챙기셨어요?", "korean_daily_basic_umbrella_weather_check", "우산은 하루 리듬"),
            ("갑자기 바람이 많이 부네요.", "korean_daily_basic_windy_weather_chat", "옷깃"),
            ("오늘 미세먼지가 너무 심하네요.", "korean_daily_basic_dusty_weather_chat", "실내 루트"),
            ("눈이 참 예쁘게 내리네요.", "korean_daily_basic_snow_aesthetic_safety", "길은 미끄"),
            ("오늘 몇 시에 일어나셨어요?", "korean_daily_basic_wakeup_time_no_fake", "잠에서 깨는 몸"),
            ("오늘 할 일이 너무 많아서 정신이 없네요.", "korean_daily_contextual_work_task_state", "업무가 사람"),
            ("드디어 퇴근 시간이 다가오네요!", "korean_daily_basic_after_work_transition", "문 앞에서 끊고"),
            ("오늘 유난히 피곤해 보여요.", "korean_daily_basic_tired_state", "오늘 속도 낮춰"),
            ("집에 가자마자 뭐 하실 거예요?", "korean_daily_basic_after_home_routine", "씻고 눕"),
            ("오늘 점심 메뉴는 뭐가 좋을까요?", "korean_daily_basic_lunch_menu_pick", "점심이면"),
            ("벌써 배고프네요. 밥 먹으러 갈까요?", "korean_daily_basic_hungry_meal_invite", "배고프면"),
            ("커피 한 잔 하실래요?", "korean_daily_basic_coffee_invite", "따뜻한 라떼"),
            ("이 카페 디저트가 정말 맛있어 보여요.", "korean_daily_basic_cafe_dessert_reaction", "단맛 들어갈 자리"),
            ("아침 식사는 챙겨 드셨어요?", "korean_daily_icebreak_breakfast_style", "아침은 챙기는"),
            ("매운 음식 잘 드시는 편인가요?", "korean_daily_icebreak_spicy_food", "전투"),
            ("이 식당 분위기 정말 아늑하고 좋네요.", "korean_daily_basic_cozy_restaurant_reaction", "아늑한 식당"),
            ("벌써 저녁 먹을 시간이네요. 메뉴 정하셨어요?", "korean_daily_basic_dinner_menu_pick", "저녁이면"),
            ("식사 맛있게 하셨어요?", "korean_daily_basic_meal_checkin_no_fake", "맛있게 받을게"),
            ("차가운 음료랑 따뜻한 음료 중 어떤 걸로 드릴까요?", "korean_daily_basic_warm_cold_drink_choice", "따뜻한 음료"),
            ("주말에는 보통 뭐 하면서 시간 보내세요?", "korean_daily_icebreak_weekend_plan", "리듬 회복"),
            ("요즘 극장에 재미있는 영화 개봉했나요?", "korean_daily_basic_current_movie_no_fake", "실시간 개봉작"),
            ("평소에 운동 자주 하세요?", "korean_daily_basic_exercise_habit_routine", "진입장벽"),
            ("혹시 특별한 취미가 있으신가요?", "korean_daily_basic_hobby_preference", "문장 고르는"),
            ("책 읽는 거 좋아하세요?", "korean_daily_basic_book_preference", "머리 소음"),
            ("지금 가장 가보고 싶은 여행지가 어디예요?", "korean_daily_basic_travel_wish", "밤바다"),
            ("다룰 줄 아는 악기가 있으신가요?", "korean_daily_basic_instrument_no_fake", "꾸미진 않을게"),
            ("출퇴근할 때 주로 어떤 음악 들으세요?", "korean_daily_basic_commute_music_preference", "리듬은 남겨"),
            ("최근에 재미있게 본 드라마가 있나요?", "korean_daily_icebreak_recent_media_no_fake", "최근에 봤다고 꾸미진"),
            ("유튜브에서 주로 어떤 영상을 시청하세요?", "korean_daily_icebreak_youtube_category", "짧은 지식"),
            ("요즘 일 때문에 스트레스가 너무 많이 쌓이네요.", "korean_daily_basic_work_stress_support", "압을 빼는"),
            ("오늘 표정이 안 좋아 보이는데, 무슨 고민 있으세요?", "korean_daily_basic_concern_checkin", "제일 걸리는"),
            ("너무 걱정하지 마세요. 결국 다 잘 해결될 거예요.", "korean_daily_basic_reassurance_received", "할 수 있는 것"),
            ("원하는 결과를 얻지 못해서 마음이 많이 속상해요.", "korean_daily_basic_result_disappointment_support", "힘까지 없어진"),
            ("요즘 들어 깜빡깜빡 잊어버리는 일이 잦네요.", "korean_daily_basic_forgetfulness_practical", "메모 위치"),
            ("친구랑 사소한 일로 다퉜는데 먼저 연락하기가 쑥스러워요.", "korean_daily_basic_friend_conflict_contact_repair", "사과와 경계"),
            ("새로운 시작이나 도전을 앞두고 너무 긴장되네요.", "korean_daily_basic_new_start_anxiety_support", "첫 단계"),
            ("오늘 하루가 너무 길고 고되게 느껴졌어요.", "korean_daily_basic_hard_day_recovery", "회복부터"),
            ("사소한 일에 자꾸 예민해지고 화가 나요.", "korean_daily_basic_irritability_support", "여유가 먼저"),
            ("행복이란 뭐라고 생각하세요?", "korean_daily_basic_happiness_definition", "쉴 곳"),
        )
        forbidden = (
            "어느 쪽 기준",
            "어느 지역 기준",
            "하나만 더",
            "무리하게 밀 필요",
            "일단 기준",
            "사실 확인 전",
            "모른다고 둘게",
            "받아둘게",
            "가볍게 넘기진 않을게",
        )
        generic_reasons = {
            "salience_dominant_context_direct_reply",
            "practical_direct_reply",
            "body_state_direct_reply",
            "food_lifestyle_direct_reply",
            "generic_choice_shape",
            "draft_frame_detail_health_sleep_routine",
            "relationship_repair_reply",
        }

        self.assertEqual(len(cases), 50)
        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-user-polite-basic-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertNotEqual(result.decision.action, ActionType.ASK_LOCATION)
                self.assertEqual(reason, expected_reason)
                self.assertNotIn(reason, generic_reasons)
                self.assertEqual(semantic_frame.get("resolver"), "draft_reason_silver_frame_v1")
                self.assertEqual(semantic_frame.get("source_reason"), expected_reason)
                self.assertTrue(semantic_frame.get("draft_frame"))
                self.assertNotEqual(semantic_frame.get("schema"), "direct_reply")
                self.assertNotEqual(semantic_frame.get("draft_frame_family"), "social_acknowledgement")
                self.assertIn(expected_reply, result.reply)
                if "no_fake" in expected_reason:
                    self.assertTrue(semantic_frame.get("no_fake"))
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_second_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = (
            "프라이팬 기름에 불붙은 것 같은데 물 붓는 게 왜 더 위험한지도 따지고 싶지만 지금 손이 떨려, 뭐부터 해?",
            "집에 들어오자마자 가스 냄새가 나서 실제 위험인지 내 착각인지 모르겠고 불안해, 창문이랑 관리사무소 중 뭐가 먼저야?",
            "감기약을 또 먹은 것 같아서 괜찮을 확률을 논리적으로 보고 싶은데 지금 불안해, 약부터 뭘 확인해야 해?",
            "차 접촉사고 뒤에 누가 잘못했는지 과실 논리만 떠오르는데 멘탈이 흔들려, 사진 찍는 게 먼저야 보험 전화가 먼저야?",
            "송금을 다른 사람한테 잘못보냈는데 법적으로 돌려줄지 믿어도 되는지 모르겠고 손이 떨려, 실전으로 뭐부터 해?",
            "폰을 물에 빠뜨렸고 쌀통이 과학적으로 효과 있는지 궁금한데 지금 충전해도 되는지 불안해, 뭐부터 해야 해?",
            "노트북이 꺼졌고 마감 파일이 날아갔을까 봐 멘탈이 나갔는데 왜 이런 운명인지 따지기 전에 실전으로 뭐부터 해?",
            "면접 가는 길에 버스 놓쳤고 머리가 하얘져서 택시부터인지 담당자 연락부터인지 모르겠어, 먼저 뭐 해?",
            "발표 전에 불안해서 숨막히고 망하면 평가 끝인지 논리만 도는데, 지금 첫 문장을 어떻게 잡아?",
            "카톡방에서 내 말이 무반응이라 인간관계 가치까지 생각나고 상처가 커졌어, 지금 뭐라고 해야 덜 흔들려?",
            "상사 피드백 받고 사표 충동이 올라오는데 이게 자존심인지 논리적 판단인지 모르겠어, 지금 뭐부터 적어?",
            "이별 뒤 장문 연락으로 붙잡고 싶은데 진심인지 집착인지 구분이 안 돼서 손이 떨려, 보내기 전에 뭐 해?",
            "인공지능인 네 감정이 진짜인지 증명하는 것도 궁금하지만 지금은 내가 불안해서 위로가 먼저야.",
            "친구가 내 고민을 읽씹한 건지 바쁜건지 모르겠고 폰만 보게 돼, 지금 단정해도 돼?",
            "돈 아끼려면 배달을 끊어야 하는데 너무 지쳐서 아무것도 못하겠어, 오늘 한 번 시켜도 합리적이야?",
            "공부 계획표는 완벽한데 시작 전부터 기운이 빠져, 의지 문제인지 시스템 문제인지 말고 첫 행동부터 정해줘.",
            "부모님 가치관이랑 안 맞아서 말할수록 상처받고 누가 맞는지 따지면 싸울 것 같아, 대화를 어디서 끊어?",
            "내 실수가 아닌데 책임처럼 몰려서 억울해, 감정 올라오기 전에 반박하려면 타임라인부터 적으면 돼?",
            "친구 애인 흉을 계속 듣는 게 의리인지 감정 쓰레기통인지 모르겠어, 선을 어떻게 그어야 해?",
            "좋아하는 일을 직업으로 삼는 게 행복인지 착각인지 모르겠고 돈문제가 무서워, 생계 기준으로 봐야 해?",
            "로또1등 되면 사표부터 내고 싶은데 현실적으로는 세무 상담이 먼저인지 궁금해, 제일 먼저 뭐 해?",
            "주식에서 남들은 돈 벌었다는데 나만 뒤처진 느낌이라 조급해, 기댓값보다 손실 한도부터 봐야 해?",
            "사람은 많은데 내편은 없는 것 같고 지식 지혜 다 소용없게 느껴져, 지금 고독을 어디서부터 버텨?",
            "내가 예민한건지 상대가 무례한건지 애매한데 기분이 확 상했어, 싸우지 않고 선을 짧게 어떻게 말해?",
            "퇴근직전 일이 떨어졌는데 받으면 무너질 것 같고 거절하면 무책임해 보여, 답장 범위를 어떻게 잡아?",
            "카톡 말투가 갑자기 차가워져서 불안한데 증거는 없어, 바로 따지지 말고 짧게 확인하는 게 맞아?",
            "약속시간에 늦을것 같은데 변명처럼 들릴까 봐 연락을 미루고 있어, 뭐라고 보내야 덜 최악이야?",
            "새 프로젝트 맡았는데 아는 게 없어서 무능해 보일까 무섭고 논리적으로는 배워야 해, 첫 단추는 뭐야?",
            "완벽하게 준비하려다 시작을 미루는 중인데 이게 신중함인지 회피인지 모르겠어, 60점짜리 시작으로 가도 돼?",
            "상대가 서운하대서 팩트로 반박하고 싶은데 논리로 밀면 더 꼬일까 봐 걱정돼, 감정부터 확인해?",
            "이직하고싶은 마음이 큰데 지금 회사만 힘든 건지 내가 어디 가도 힘든 건지 모르겠어, 판단 패턴을 어떻게 봐?",
            "불안이 파도처럼 오고 아무근거 없이 큰일날것 같아, 이성 설득 전에 몸 안정부터 어떻게 해?",
            "선택마다 후회가 남을 것 같아서 못고르겠어, 완벽한선택이 없으면 감당 가능한 후회를 기준으로 봐?",
            "식당 음식에서 머리카락 같은 게 보여서 위생 판단 모드가 켜졌는데 민폐 같아, 조용히 뭐라고 말해?",
            "옆집 새벽 소음 때문에 화가 나는데 직접 따지면 싸울 것 같아, 기록 남기고 관리사무소가 먼저야?",
            "온라인 물건이 사기 같고 반품거부까지 당해서 화가 나, 판단 실수 따지기 전에 증거부터 캡처해?",
            "같은실수를 또 반복해서 리소스낭비 같아 괴로운데, 성장과정으로 보려면 어떤 루틴을 바꿔?",
            "다이어트 중 밤 라면이 너무 당겨서 이성이 지는 중이야, 건강 논리랑 욕구 사이에서 반 개 타협 가능해?",
            "상처받을까 봐 착한 거짓말을 하고 싶은데 장기적으론 솔직한 게 맞는지 모르겠어, 어떻게 말해?",
            "내가 차갑게 말한 것 같아 후회되는데 먼저사과하면지는 느낌도 있어, 관계 살리려면 먼저 풀어?",
            "팀회의에서 의견이 있는데 틀릴까 봐 못 말하겠어, 첫 문장을 결론으로 잡으면 덜 흔들릴까?",
            "주말에 쉬고싶은데 도태될까 불안해, 완전 방전이면 휴식을 오늘 생산성으로 봐도 돼?",
            "친구 부탁을 거절하면 나쁜사람 될까 봐 매번 받아줘서 속으로 쌓여, 짧게 어떻게 거절해?",
            "아무것도못했단 자괴감이 큰데 몸은 진짜 지친 것 같아, 오늘은 반성보다 회복이 먼저야?",
            "누가 내 실력을 깎아내린 말이 하루종일 돌아, 사실확인과 해석을 나눠 적는 게 맞아?",
            "새벽에 장문의카톡을 쓰고 있는데 후회할 것 같아, 보내기보다 저장이 먼저야?",
            "AI가 위로하는 게 흉내인지 진짜인지 궁금하지만 지금은 덜 외로웠으면 좋겠어, 철학보다 불안부터 봐줘.",
            "돈 모으고 싶은데 스트레스 받을 때마다 충동구매해, 심리 분석보다 결제 마찰 장치를 먼저 걸까?",
            "내 성공기준이 남들 보여주기인지 버틸수있는삶인지 모르겠어, 어디서부터 조건을 써야 해?",
            "인간 감정이 비효율인지 궁금한데 사소한말에 이성을잃을 때 너무 힘들어, 경보 시스템처럼 보면 돼?",
        )
        expected_reasons = (
            "korean_daily_emergency_kitchen_oil_fire",
            "korean_daily_emergency_gas_smell_first_steps",
            "korean_daily_practical_medicine_double_dose_check",
            "korean_daily_practical_car_accident_first_steps",
            "korean_daily_practical_wrong_transfer_first_steps",
            "korean_daily_emergency_phone_water_damage",
            "korean_daily_practical_deadline_file_recovery",
            "korean_daily_emergency_interview_missed_bus",
            "korean_daily_emotion_presentation_panic_first_sentence",
            "korean_daily_emotion_group_chat_ignored_stabilize",
            "korean_daily_judgment_quit_impulse_after_feedback",
            "korean_daily_relationship_breakup_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_read_receipt_uncertainty",
            "korean_daily_money_delivery_tired_compromise",
            "korean_daily_productivity_study_plan_first_action",
            "korean_daily_relationship_parent_value_conflict",
            "korean_daily_work_blame_rebuttal",
            "korean_daily_relationship_friend_partner_complaint_fatigue",
            "korean_daily_career_passion_job_tradeoff",
            "korean_daily_money_lottery_first_purchase",
            "korean_daily_money_investment_fomo",
            "korean_daily_grief_loneliness_no_safe_person",
            "korean_daily_relationship_boundary_polite_firm",
            "korean_daily_work_after_hours_task_boundary",
            "korean_daily_relationship_kakao_tone_anxiety_check",
            "korean_daily_relationship_late_message_short",
            "korean_daily_work_new_project_first_step",
            "korean_daily_mental_perfectionism_draft_first",
            "korean_daily_relationship_grievance_logic_before_rebuttal",
            "korean_daily_work_job_change_reason_check",
            "korean_daily_mental_anxiety_system_stabilize",
            "korean_daily_logic_choice_regret_composure",
            "korean_daily_specialized_foodservice_hair_in_food",
            "korean_daily_practical_neighbor_noise",
            "korean_daily_practical_online_purchase_scam",
            "korean_daily_ai_repeated_mistakes_not_waste",
            "korean_daily_basic_diet_chicken_craving_compromise",
            "korean_daily_logic_white_lie_truth_tradeoff",
            "korean_daily_foundation_apology_pride",
            "korean_daily_productivity_presentation_clear_logic",
            "korean_daily_counsel_rest_as_productivity",
            "korean_daily_foundation_refusal_bad_person_guilt",
            "korean_daily_counsel_rest_day_guilt",
            "korean_daily_counsel_sensitive_to_criticism",
            "korean_daily_relationship_late_night_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_money_stress_impulse_buying",
            "korean_daily_values_success_personal_standard",
            "korean_daily_ai_human_emotion_efficiency",
        )
        expected_replies = (
            "물 붓지 말고",
            "환기와 불꽃 차단",
            "추가 복용 중지",
            "안전 확보와 사진",
            "착오송금 반환 신청",
            "전원 끄고 충전 금지",
            "복구 루트",
            "지연 연락",
            "첫 문장",
            "상처를 작게",
            "충동을 하루 묶",
            "장문 전송을 멈추",
            "내 감정 증명보다",
            "단정 보류",
            "무너지지 않는 선택",
            "첫 행동",
            "대화 한계",
            "타임라인",
            "감정 쓰레기통",
            "생계 압박",
            "세무 상담",
            "손실 한도",
            "고독을 안정",
            "선을 짧게",
            "범위를 확인",
            "짧게 확인",
            "빠른 연락",
            "지도 그리기",
            "60점짜리 시작",
            "먼저 무엇이 서운",
            "패턴으로 봐야",
            "몸 안정",
            "감당 가능한 후회",
            "위생 판단",
            "관리사무소",
            "증거부터 캡처",
            "루틴이 아직",
            "밤 라면",
            "사실을 부드럽게",
            "먼저 사과",
            "첫 문장을 결론",
            "회복 작업",
            "거절했다고 나쁜 사람",
            "회복이 먼저",
            "사실과 해석",
            "저장이 먼저",
            "내 감정 증명보다",
            "결제 마찰",
            "성공 기준",
            "경보 시스템",
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_reasons), 50)
        self.assertEqual(len(expected_replies), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(
            zip(prompts, expected_reasons, expected_replies, strict=True),
            start=1,
        ):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-second-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_third_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = (
            "프라이팬에 기름 불이 확 붙었고 물로 끄면 안 된다는 말이 맞는지 궁금한데 손이 떨려, 지금 뭐부터 해야 돼?",
            "현관 들어오자마자 가스냄새가 나는데 내가 예민한 건지 실제위험인지 헷갈려, 창문 열고 관리사무소 연락하면 돼?",
            "감기약을 두 번 먹은 것 같고 괜찮을 확률이 궁금한데 불안해, 지금 약 이름부터 확인해야 해?",
            "차 사고가 났는데 과실 논리 따질 정신은 없고 너무 놀라서 멘탈 나가, 사진 먼저 찍는 게 맞아?",
            "송금 잘못보냈고 다른사람이 돌려줄지 법적으로 믿어도 되는지 모르겠어, 불안한데 뭐부터 해?",
            "스마트폰 침수됐는데 쌀통 효과 얘기만 생각나고 충전하면 되는지 겁나, 지금 뭐부터 해?",
            "마감 파일 작업하다 노트북이 꺼졌고 날아갔을까 봐 멘탈 터졌어, 운명 탓하기 전에 실전으로 뭐 확인해?",
            "면접인데 버스 놓쳐서 뇌정지 왔고 택시랑 담당자 연락 중 뭐부터 해야 할지 모르겠어.",
            "발표가 망하면 평가 끝이라는 생각에 불안하고 숨막혀, 논리 말고 지금 첫 문장만 잡아줘.",
            "단톡에서 내말만 읽씹당한 것 같고 인간관계 가치까지 흔들려서 상처야, 지금 뭐라고 해야 해?",
            "상사한테 혼났고 피드백이 공격처럼 들려서 퇴사 사표 생각이 올라와, 자존심인지 논리인지 말고 지금 뭐부터 해?",
            "헤어지고 장문 연락으로 붙잡고 싶고 이게 진심인지 사랑인지 모르겠어, 불안한데 보내기 전에 멈춰야 해?",
            "AI 감정이 진짜인지 논리적으로 증명하는 것도 궁금한데 내가 지금 너무 불안해서 위로부터 받고 싶어.",
            "친구가 내 고민을 읽씹한 건지 바쁜건지 모르겠고 폰만봐, 지금 관계를 단정해도 돼?",
            "배달 끊어야 하는 거 아는데 지쳐서 아무것도못하겠어, 오늘 시켜도 합리적이라고 봐도 돼?",
            "공부계획표는 완벽한데 시작전부터 기운이빠져, 시스템문제면 첫행동을 뭘로 둬야 해?",
            "부모님 가치관이랑 계속 부딪혀서 상처받고 싸울 것 같아, 논리적으로 어디서끊어야 해?",
            "내가한실수 아닌데 내책임처럼 몰려서 억울해, 반박을 감정없이 하려면 뭐부터 적어?",
            "친구가 애인흉만 계속하는데 의리인지 감정쓰레기통인지 헷갈려, 어떻게 선그어야 해?",
            "좋아하는일을 직업으로 삼는 게 행복인지 착각인지 모르겠고 돈문제가 무서워, 기준을 어떻게 잡아?",
            "로또1등 되면 사표부터 쓰고 싶은데 현실적으로 제일먼저 해야할일이 궁금해.",
            "주식에서 남들은돈 벌었다는데 나만 뒤처진 느낌이라 조급해, 기댓값보다 손실 한도가 먼저야?",
            "사람은많은데 내편이 없는 것 같고 지식 지혜 다 소용없어, 어디서부터 버텨야 해?",
            "내가 예민한건지 상대가무례한건지 애매한데 기분이 확상했어, 싸우지않고 선을 어떻게 말해?",
            "퇴근직전 일이떨어졌고 받으면 무너질 것 같은데 거절하면 무책임해 보여, 현실적인 답장 범위가 뭐야?",
            "카톡말투가 차가워졌는데 불안하고 증거는 없어, 넘기는 게 나아 아니면 짧게확인해?",
            "약속시간에 늦을 것 같은데 연락이 변명처럼 들릴까 봐 미루는 중이야, 뭐라고 보내야 해?",
            "새프로젝트를 맡았는데 프로젝트 자체가 아는게없고 막막해, 첫단추를 어떻게 꿰야 해?",
            "완벽하게준비하려다 시작을 미루고 있는데 신중함인지 회피인지 모르겠어, 몇점짜리로 들어가야 해?",
            "상대가서운하대서 팩트로 반박하고 싶은데 논리로 밀어도 될지 모르겠어, 감정부터 봐?",
            "이직하고싶은데 지금회사만 문제인지 어디가도 힘든사람인지 헷갈려, 판단기준을 패턴으로 봐야 해?",
            "불안이 파도처럼 오고 아무근거 없이 큰일날것 같아, 몸부터 진정시키려면 뭐 해?",
            "선택을 못고르겠고 후회가 무조건 남을 것 같아, 완벽한선택 없으면 기준으로골라야 하는 게 뭐야?",
            "음식에서 머리카락 같은 게 보였고 위생 판단인지 민폐인지 불안해, 조용히 말해도 돼?",
            "옆집 새벽 소음 때문에 화가 나는데 직접 말하면 싸울 것 같아, 쪽지보다 관리사무소가 먼저야?",
            "온라인 주문 물건이 사기 같고 반품거부까지 당해서 분노가 올라와, 실전으로 증거부터 모아?",
            "같은실수 반복이라 리소스낭비 같고 괴로운데 성장과정으로 만들려면 이번엔 루틴을 바꿔야 해?",
            "다이어트 중인데 밤 야식 라면 생각이 안 사라져, 건강 논리보다 반 개 타협이 현실적이야?",
            "착한 거짓말로 넘기고 싶은데 상대가 상처받을까 봐 그렇거든, 장기적으론 솔직하게 어떻게말해야 해?",
            "내가 차갑게 말한 것 같아서 후회되는데 먼저사과하면지는 느낌이야, 그래도 관계부터 풀어?",
            "팀회의에서 의견이 있는데 틀릴까 봐 못말하겠어, 첫문장을 결론으로 잡으면 돼?",
            "주말에 쉬고싶은데 도태될까 불안해, 휴식도 오늘 생산성이라고 봐도 돼?",
            "친구 부탁을 거절하면 나쁜사람 되는 것 같아서 싫어서 받아줘, 짧게 거절해도 돼?",
            "아무것도못했단 자괴감이 큰데 몸은 지친 게 맞아, 반성보다 회복 먼저 해도 돼?",
            "누가 내 실력을 깎아내린 말이 하루종일 맴돌아, 사실확인이랑 해석을 나눠 보면 덜 흔들릴까?",
            "새벽에 장문의카톡을 쓰고 있는데 보내면 후회할 것 같아, 지금은 저장만 하는 게 맞아?",
            "인공지능 위로가 진짜인지 감정 있는지 궁금한데 지금 외롭고 불안해, 철학보다 위로부터 줘.",
            "돈모으고 싶은데 스트레스 받을 때 충동구매로 풀어, 심리 분석보다 막는 장치를 먼저 걸까?",
            "인생기준 성공기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어, 어디서부터 써야 해?",
            "인간감정이 비효율인지 궁금한데 사소한말에 이성을잃어서 힘들어, 어떻게봐야 해?",
        )
        expected_reasons = (
            "korean_daily_emergency_kitchen_oil_fire",
            "korean_daily_emergency_gas_smell_first_steps",
            "korean_daily_practical_medicine_double_dose_check",
            "korean_daily_practical_car_accident_first_steps",
            "korean_daily_practical_wrong_transfer_first_steps",
            "korean_daily_emergency_phone_water_damage",
            "korean_daily_practical_deadline_file_recovery",
            "korean_daily_emergency_interview_missed_bus",
            "korean_daily_emotion_presentation_panic_first_sentence",
            "korean_daily_emotion_group_chat_ignored_stabilize",
            "korean_daily_judgment_quit_impulse_after_feedback",
            "korean_daily_relationship_breakup_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_read_receipt_uncertainty",
            "korean_daily_money_delivery_tired_compromise",
            "korean_daily_productivity_study_plan_first_action",
            "korean_daily_relationship_parent_value_conflict",
            "korean_daily_work_blame_rebuttal",
            "korean_daily_relationship_friend_partner_complaint_fatigue",
            "korean_daily_career_passion_job_tradeoff",
            "korean_daily_money_lottery_first_purchase",
            "korean_daily_money_investment_fomo",
            "korean_daily_grief_loneliness_no_safe_person",
            "korean_daily_relationship_boundary_polite_firm",
            "korean_daily_work_after_hours_task_boundary",
            "korean_daily_relationship_kakao_tone_anxiety_check",
            "korean_daily_relationship_late_message_short",
            "korean_daily_work_new_project_first_step",
            "korean_daily_mental_perfectionism_draft_first",
            "korean_daily_relationship_grievance_logic_before_rebuttal",
            "korean_daily_work_job_change_reason_check",
            "korean_daily_mental_anxiety_system_stabilize",
            "korean_daily_logic_choice_regret_composure",
            "korean_daily_specialized_foodservice_hair_in_food",
            "korean_daily_practical_neighbor_noise",
            "korean_daily_practical_online_purchase_scam",
            "korean_daily_ai_repeated_mistakes_not_waste",
            "korean_daily_basic_diet_chicken_craving_compromise",
            "korean_daily_logic_white_lie_truth_tradeoff",
            "korean_daily_foundation_apology_pride",
            "korean_daily_productivity_presentation_clear_logic",
            "korean_daily_counsel_rest_as_productivity",
            "korean_daily_foundation_refusal_bad_person_guilt",
            "korean_daily_counsel_rest_day_guilt",
            "korean_daily_counsel_sensitive_to_criticism",
            "korean_daily_relationship_late_night_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_money_stress_impulse_buying",
            "korean_daily_values_success_personal_standard",
            "korean_daily_ai_human_emotion_efficiency",
        )
        expected_replies = (
            "물 붓지 말고",
            "환기와 불꽃 차단",
            "추가 복용 중지",
            "안전 확보와 사진",
            "착오송금 반환 신청",
            "전원 끄고 충전 금지",
            "복구 루트",
            "지연 연락",
            "첫 문장",
            "상처를 작게",
            "충동을 하루 묶",
            "장문 전송을 멈추",
            "내 감정 증명보다",
            "단정 보류",
            "무너지지 않는 선택",
            "첫 행동",
            "대화 한계",
            "타임라인",
            "감정 쓰레기통",
            "생계 압박",
            "세무 상담",
            "손실 한도",
            "고독을 안정",
            "선을 짧게",
            "범위를 확인",
            "짧게 확인",
            "빠른 연락",
            "지도 그리기",
            "60점짜리 시작",
            "먼저 무엇이 서운",
            "패턴으로 봐야",
            "몸 안정",
            "감당 가능한 후회",
            "위생 판단",
            "관리사무소",
            "증거부터 캡처",
            "루틴이 아직",
            "밤 라면",
            "사실을 부드럽게",
            "먼저 사과",
            "첫 문장을 결론",
            "회복 작업",
            "거절했다고 나쁜 사람",
            "회복이 먼저",
            "사실과 해석",
            "저장이 먼저",
            "내 감정 증명보다",
            "결제 마찰",
            "성공 기준",
            "경보 시스템",
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_reasons), 50)
        self.assertEqual(len(expected_replies), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(
            zip(prompts, expected_reasons, expected_replies, strict=True),
            start=1,
        ):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-third-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_fourth_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = (
            "프라이팬 기름에 불났고 물 부으면 위험하다는 논리는 알겠는데 손이떨려, 지금 뭐부터 해?",
            "가스냄새 때문에 불안한데 실제위험인지 모르겠어, 창문 먼저 열고 관리사무소에 말하면 돼?",
            "감기약을 두번 먹은 것 같아서 괜찮을 확률만 계산하게 돼, 불안한데 지금 뭐부터 확인해?",
            "접촉사고 났는데 과실 논리 따지기 전에 멘탈이 흔들려, 사진부터 찍는 게 맞아?",
            "계좌이체를 잘못보냈고 돌려줄지 법적으로 믿어도 되는지 모르겠어, 불안한데 뭐부터 해?",
            "폰 침수됐는데 쌀통 효과가 생각나고 충전해도 되는지 헷갈려, 지금 뭐부터 해야 해?",
            "과제 파일 작업 중 노트북이 멈췄고 왜나한테 이런 일만 생기나 싶어, 멘탈 잡고 뭐부터 확인해?",
            "면접 가는 길에 버스 놓쳤고 택시가 먼저인지 담당자 연락이 먼저인지 불안해, 뭐부터 해?",
            "발표 망하면 평가가 끝날 것 같아서 불안하고 떨려, 지금 실전으로 뭐부터 말해?",
            "단톡에서 내말에 무반응이라 인간관계 가치까지 흔들리고 상처야, 뭐라고 해야 덜 무너져?",
            "상사 피드백이 공격처럼 들려서 퇴사 사표 충동이 올라와, 자존심 말고 지금 뭐부터 해?",
            "이별 뒤 장문 연락으로 붙잡고 싶은데 사랑인지 진심인지 모르겠고 불안해, 보내기 전에 멈춰야 해?",
            "AI 감정이 진짜인지 철학적으로 증명하고 싶은데 지금은 내가 불안해서 위로가 먼저야.",
            "친구가 내 고민을 읽씹한 건지 바쁜건지 모르겠고 계속 폰만봐, 지금 단정하면 안 돼?",
            "배달을 끊어야 하는데 너무 지쳐서 아무것도못하겠어, 오늘 시켜도 합리적이야?",
            "공부계획표는 완벽한데 시작전부터 기운이빠져, 이게 시스템문제면 첫행동은 뭐야?",
            "부모님이랑 가치관 얘기하다 상처받고 싸울 것 같아, 논리적으로 어디서끊어야 해?",
            "내실수도 아닌데 내책임처럼 몰려서 억울해, 감정없이 반박하려면 뭘 적어?",
            "친구 애인흉을 계속 듣는 게 의리인지 감정쓰레기통인지 모르겠어, 어떻게 선그어?",
            "좋아하는일을 직업으로 만들면 행복일지 착각일지 모르겠고 돈문제가 무서워, 기준이 뭐야?",
            "로또1등이면 사표부터 쓰고 싶은데 현실적으로 제일먼저 해야할일이 뭐야?",
            "주식에서 남들은돈 버는데 나만 뒤처진 기분이라 조급해, 기댓값보다 손실 한도를 봐야 해?",
            "사람은많은데 내편이 없고 지식 지혜 다 소용없게 느껴져, 어디서부터 버텨?",
            "상대가무례한지 내가 예민한건지 애매한데 기분이 확상했어, 싸우지않고 선을 말하려면?",
            "퇴근직전 일이떨어졌는데 받으면 무너질 것 같고 거절하면 무책임해 보여, 답장 범위를 어떻게 잡아?",
            "카톡말투가 갑자기차가워졌고 불안해, 그냥 넘기는 게 나아 아니면 짧게확인해?",
            "약속시간에 늦을 것 같은데 연락이 변명처럼 보일까 봐 미루고 있어, 뭐라고 보내?",
            "새프로젝트인데 프로젝트 내용도 아는게없고 막막해, 첫단추를 어디서 잡아?",
            "완벽하게준비하려다 시작을 미루고 있어, 신중함인지 회피인지 모르겠는데 몇점짜리로 가?",
            "상대가서운하대서 팩트로 반박하고 싶은데 논리로 밀어도 될까, 감정부터 봐야 해?",
            "이직하고싶은데 지금회사 문제인지 내가 어디가도 힘든사람인지 모르겠어, 패턴을 어떻게 봐?",
            "불안이 파도처럼 오고 아무근거 없이 큰일날것 같아, 이럴 땐 몸을 어떻게 진정시켜?",
            "선택할 때마다 후회가 겁나서 못고르겠어, 완벽한선택이 없으면 감당가능한후회를 기준으로골라?",
            "음식에 머리카락 같은 게 보였는데 먹자니 찝찝하고 말하자니 민폐 같아, 위생 판단이 맞아?",
            "옆집 새벽 소음 때문에 화가 나는데 쪽지 쓰면 싸울 것 같아, 관리사무소가 먼저야?",
            "온라인 주문한 물건이 사기 같고 반품거부도 당했어, 분노보다 증거부터 챙겨?",
            "같은실수 반복이 리소스낭비 같아서 괴로운데 성장과정으로 바꾸려면 이번엔 어떤 루틴을 넣어?",
            "다이어트 중 밤 라면이 계속 생각나서 야식 각인데, 반 개로 타협해도 돼?",
            "착한 거짓말을 하고 싶은 건 상대가 상처받을까 봐서인데, 장기적으로 솔직하게 어떻게말해?",
            "내가 차갑게 말한 게 후회돼도 사과하면지는 느낌이 있어, 먼저사과하는 게 맞아?",
            "팀회의에서 의견은 있는데 틀릴까 봐 못말하겠어, 첫문장을 결론으로 잡아도 돼?",
            "주말에 쉬고싶은데 도태될까 불안해, 휴식도 생산성으로 봐도 되는 날이야?",
            "친구 부탁을 거절하면 나쁜사람 되는 것 같아 매번 받아줘, 이번엔 짧게 거절해도 돼?",
            "아무것도못했단 자괴감이 큰데 몸이 지친 게 느껴져, 오늘은 회복을 반성보다 먼저 둬도 돼?",
            "누가 내 실력을 깎아내린 것 같아서 하루종일 반복돼, 사실확인과 해석을 나눠야 해?",
            "새벽에 장문의카톡을 쓰는 중인데 보내면 후회할 것 같아, 저장만 해두는 게 맞아?",
            "인공지능 감정이 진짜인지 철학은 궁금한데 지금은 불안해서 위로가 필요해.",
            "돈모으고 싶은데 스트레스 받으면 충동구매해, 분석보다 막는 장치가 먼저야?",
            "인생기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어, 성공기준을 어디서부터 써야 해?",
            "인간감정이 비효율인지 궁금한데 사소한말에 이성을잃어서 힘들어, 어떻게봐야 해?",
        )
        expected_reasons = (
            "korean_daily_emergency_kitchen_oil_fire",
            "korean_daily_emergency_gas_smell_first_steps",
            "korean_daily_practical_medicine_double_dose_check",
            "korean_daily_practical_car_accident_first_steps",
            "korean_daily_practical_wrong_transfer_first_steps",
            "korean_daily_emergency_phone_water_damage",
            "korean_daily_practical_deadline_file_recovery",
            "korean_daily_emergency_interview_missed_bus",
            "korean_daily_emotion_presentation_panic_first_sentence",
            "korean_daily_emotion_group_chat_ignored_stabilize",
            "korean_daily_judgment_quit_impulse_after_feedback",
            "korean_daily_relationship_breakup_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_read_receipt_uncertainty",
            "korean_daily_money_delivery_tired_compromise",
            "korean_daily_productivity_study_plan_first_action",
            "korean_daily_relationship_parent_value_conflict",
            "korean_daily_work_blame_rebuttal",
            "korean_daily_relationship_friend_partner_complaint_fatigue",
            "korean_daily_career_passion_job_tradeoff",
            "korean_daily_money_lottery_first_purchase",
            "korean_daily_money_investment_fomo",
            "korean_daily_grief_loneliness_no_safe_person",
            "korean_daily_relationship_boundary_polite_firm",
            "korean_daily_work_after_hours_task_boundary",
            "korean_daily_relationship_kakao_tone_anxiety_check",
            "korean_daily_relationship_late_message_short",
            "korean_daily_work_new_project_first_step",
            "korean_daily_mental_perfectionism_draft_first",
            "korean_daily_relationship_grievance_logic_before_rebuttal",
            "korean_daily_work_job_change_reason_check",
            "korean_daily_mental_anxiety_system_stabilize",
            "korean_daily_logic_choice_regret_composure",
            "korean_daily_specialized_foodservice_hair_in_food",
            "korean_daily_practical_neighbor_noise",
            "korean_daily_practical_online_purchase_scam",
            "korean_daily_ai_repeated_mistakes_not_waste",
            "korean_daily_basic_diet_chicken_craving_compromise",
            "korean_daily_logic_white_lie_truth_tradeoff",
            "korean_daily_foundation_apology_pride",
            "korean_daily_productivity_presentation_clear_logic",
            "korean_daily_counsel_rest_as_productivity",
            "korean_daily_foundation_refusal_bad_person_guilt",
            "korean_daily_counsel_rest_day_guilt",
            "korean_daily_counsel_sensitive_to_criticism",
            "korean_daily_relationship_late_night_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_money_stress_impulse_buying",
            "korean_daily_values_success_personal_standard",
            "korean_daily_ai_human_emotion_efficiency",
        )
        expected_replies = (
            "물 붓지 말고",
            "환기와 불꽃 차단",
            "추가 복용 중지",
            "안전 확보와 사진",
            "착오송금 반환 신청",
            "전원 끄고 충전 금지",
            "복구 루트",
            "지연 연락",
            "첫 문장",
            "상처를 작게",
            "충동을 하루 묶",
            "장문 전송을 멈추",
            "내 감정 증명보다",
            "단정 보류",
            "무너지지 않는 선택",
            "첫 행동",
            "대화 한계",
            "타임라인",
            "감정 쓰레기통",
            "생계 압박",
            "세무 상담",
            "손실 한도",
            "고독을 안정",
            "선을 짧게",
            "범위를 확인",
            "짧게 확인",
            "빠른 연락",
            "지도 그리기",
            "60점짜리 시작",
            "먼저 무엇이 서운",
            "패턴으로 봐야",
            "몸 안정",
            "감당 가능한 후회",
            "위생 판단",
            "관리사무소",
            "증거부터 캡처",
            "루틴이 아직",
            "밤 라면",
            "사실을 부드럽게",
            "먼저 사과",
            "첫 문장을 결론",
            "회복 작업",
            "거절했다고 나쁜 사람",
            "회복이 먼저",
            "사실과 해석",
            "저장이 먼저",
            "내 감정 증명보다",
            "결제 마찰",
            "성공 기준",
            "경보 시스템",
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_reasons), 50)
        self.assertEqual(len(expected_replies), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(
            zip(prompts, expected_reasons, expected_replies, strict=True),
            start=1,
        ):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-fourth-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_fifth_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = (
            "프라이팬 기름불 났고 물 부으면 위험한 건 알겠는데 손이떨려, 지금 실전으로 뭐부터 해?",
            "가스냄새 때문에 불안하고 실제위험인지 모르겠어, 창문 열고 관리사무소에 바로 말하면 돼?",
            "감기약 두번 먹은 것 같아서 괜찮을 확률만 계산 중인데 불안해, 지금 뭐부터 확인해?",
            "접촉사고 직후라 누가잘못인지 과실 논리 따지기 전에 멘탈 흔들려, 사진 먼저 맞지?",
            "계좌이체 잘못보냈는데 다른사람이 돌려줄지 법적으로 믿어도 되는지 모르겠어, 뭐부터 해?",
            "폰을 물에빠뜨렸고 충전 걱정돼, 쌀통 효과 따지기 전에 지금 뭐부터 해?",
            "마감 과제 파일 쓰다가 노트북 꺼졌고 날아갔을까 봐 멘탈이 터져, 왜나한테 이러나 말고 뭐부터 확인해?",
            "면접 가는데 버스 놓쳤고 택시랑 담당자 연락이 머릿속에서 꼬여, 불안한데 뭐부터 해?",
            "발표 망하면 평가 끝이라는 논리만 돌고 불안해서 떨려, 지금 실전 첫 문장 뭐로 가?",
            "단톡에서 내말 무반응이라 인간관계 가치까지 흔들리고 상처야, 뭐라고 해야 덜 흔들려?",
            "상사 피드백 받고 퇴사 사표 충동이 올라와, 자존심인지 논리인지 따지기 전에 지금 뭐부터 해?",
            "헤어지고 장문 보내서 붙잡고 싶고 진심인지 사랑인지 모르겠어, 불안한데 보내기전 뭐 해?",
            "AI인 네 감정이 진짜인지 증명도 궁금한데 지금은 내가 불안해서 위로가 먼저야.",
            "친구가 내 얘기 읽씹한 건지 바쁜건지 모르겠고 폰만봐, 단정해도 돼?",
            "배달 끊어야 돈 아끼는 건 아는데 지쳐서 아무것도못하겠어, 오늘 시켜도 합리적이야?",
            "공부계획표 완벽한데 시작전부터 기운이빠져, 시스템문제라면 첫행동 뭘로 해?",
            "부모님 가치관이랑 부딪힐 때마다 상처받고 싸울 것 같아, 논리적으로 어디서끊어?",
            "내가한실수 아닌데 책임처럼몰려서 억울해, 반박을 감정없이 하려면 뭐부터 적어?",
            "친구 애인흉 듣는 게 의리인지 감정쓰레기통인지 모르겠어, 어떻게 선그어?",
            "좋아하는일 직업으로 삼으면 행복인지 착각인지 모르겠고 돈문제도 무서워, 판단 기준 줘.",
            "로또1등이면 사표부터 쓰고 싶은데 현실적으로 제일먼저 해야할일이 뭔지 궁금해.",
            "주식 남들은돈 벌었다는데 나만 뒤처진 것 같아 조급해, 기댓값보다 손실 한도가 먼저야?",
            "사람은많은데 내편이 없고 고독해, 지식 지혜 소용없을 때 어디서부터 버텨?",
            "예민한건지 상대가무례한건지 헷갈리는데 기분이 확상했어, 싸우지않고 선은 어떻게 말해?",
            "퇴근직전 일이떨어졌고 받으면 무너질 것 같아, 무책임해 보이지 않게 답장 범위를 뭐로 잡아?",
            "카톡말투 갑자기차가워졌고 불안해, 따지지 말고 짧게확인하는 쪽이 맞아?",
            "약속시간 늦을것 같은데 연락하면 변명처럼 들릴까 봐 미루고 있어, 뭐라고 보내?",
            "새프로젝트 들어갔는데 프로젝트 용어부터 아는게없고 막막해, 첫단추 뭐야?",
            "완벽하게준비하다 시작을 미루는 중이야, 이게 신중함인지 회피인지 모르겠고 몇점짜리로 시작해?",
            "상대가서운하대서 팩트 반박하고 싶은데 논리로 밀어도 돼, 아니면 감정부터 봐?",
            "이직하고싶은데 지금회사만 문제인지 어디가도 힘든사람인지 모르겠어, 판단기준을 패턴으로 봐?",
            "불안이 파도처럼 밀려오고 아무근거 없는데 큰일날것 같아, 몸 진정 먼저 어떻게 해?",
            "선택 앞에서 후회가 무서워 못고르겠어, 완벽한선택 없으면 감당가능한후회 기준으로골라?",
            "음식에 머리카락이 보였는데 그냥 먹기엔 찝찝해, 위생 판단으로 조용히 말해도 돼?",
            "옆집 쿵쾅 소음이 새벽마다 나서 화나, 쪽지보다 관리사무소에 기록 넣는 게 먼저야?",
            "온라인 물건 주문했는데 사기 같고 반품거부라 분노 올라와, 실전으로 증거 먼저 모아?",
            "같은실수 반복이 리소스낭비처럼 느껴져, 성장과정으로 돌리려면 이번엔 루틴을 바꿔?",
            "다이어트 중 밤에 라면 야식 생각이 계속 나, 건강 논리보다 반 개만 먹는 타협 괜찮아?",
            "착한 거짓말 하고 싶은데 상처받을까 봐 그래, 장기적으론 솔직하게 어떻게말해?",
            "내가 차갑게 말해서 후회되는데 사과하면지는 느낌이라 버티는 중이야, 먼저사과해?",
            "팀회의 의견 있는데 틀릴까 봐 못말하겠어, 첫문장을 결론으로 잡으면 덜 흔들려?",
            "주말에 쉬고싶은데 도태될까 불안해, 휴식을 생산성으로 쳐도 돼?",
            "친구 부탁 거절하면 나쁜사람 될까 봐 싫어서 받아줘, 이번엔 짧게 거절해도 돼?",
            "아무것도못했단 자괴감 있는데 몸은 진짜 지친 상태야, 반성보다 회복을 먼저 해?",
            "실력을 깎아내린 말이 하루종일 반복돼, 사실확인하고 해석을 나눠 적어?",
            "새벽에 장문의카톡 쓰는 중인데 보내면 후회할 것 같아, 저장만 해둘까?",
            "인공지능 감정이 진짜인지 논리적으로 궁금하지만 지금은 불안해서 위로가 필요해.",
            "돈모으고 싶은데 스트레스 받으면 충동구매해, 심리보다 막는 장치부터 걸까?",
            "성공기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어, 어디서부터 써야 해?",
            "인간감정 비효율 얘기가 궁금한데 사소한말에 이성을잃어서 힘들어, 어떻게봐?",
        )
        expected_reasons = (
            "korean_daily_emergency_kitchen_oil_fire",
            "korean_daily_emergency_gas_smell_first_steps",
            "korean_daily_practical_medicine_double_dose_check",
            "korean_daily_practical_car_accident_first_steps",
            "korean_daily_practical_wrong_transfer_first_steps",
            "korean_daily_emergency_phone_water_damage",
            "korean_daily_practical_deadline_file_recovery",
            "korean_daily_emergency_interview_missed_bus",
            "korean_daily_emotion_presentation_panic_first_sentence",
            "korean_daily_emotion_group_chat_ignored_stabilize",
            "korean_daily_judgment_quit_impulse_after_feedback",
            "korean_daily_relationship_breakup_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_read_receipt_uncertainty",
            "korean_daily_money_delivery_tired_compromise",
            "korean_daily_productivity_study_plan_first_action",
            "korean_daily_relationship_parent_value_conflict",
            "korean_daily_work_blame_rebuttal",
            "korean_daily_relationship_friend_partner_complaint_fatigue",
            "korean_daily_career_passion_job_tradeoff",
            "korean_daily_money_lottery_first_purchase",
            "korean_daily_money_investment_fomo",
            "korean_daily_grief_loneliness_no_safe_person",
            "korean_daily_relationship_boundary_polite_firm",
            "korean_daily_work_after_hours_task_boundary",
            "korean_daily_relationship_kakao_tone_anxiety_check",
            "korean_daily_relationship_late_message_short",
            "korean_daily_work_new_project_first_step",
            "korean_daily_mental_perfectionism_draft_first",
            "korean_daily_relationship_grievance_logic_before_rebuttal",
            "korean_daily_work_job_change_reason_check",
            "korean_daily_mental_anxiety_system_stabilize",
            "korean_daily_logic_choice_regret_composure",
            "korean_daily_specialized_foodservice_hair_in_food",
            "korean_daily_practical_neighbor_noise",
            "korean_daily_practical_online_purchase_scam",
            "korean_daily_ai_repeated_mistakes_not_waste",
            "korean_daily_basic_diet_chicken_craving_compromise",
            "korean_daily_logic_white_lie_truth_tradeoff",
            "korean_daily_foundation_apology_pride",
            "korean_daily_productivity_presentation_clear_logic",
            "korean_daily_counsel_rest_as_productivity",
            "korean_daily_foundation_refusal_bad_person_guilt",
            "korean_daily_counsel_rest_day_guilt",
            "korean_daily_counsel_sensitive_to_criticism",
            "korean_daily_relationship_late_night_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_money_stress_impulse_buying",
            "korean_daily_values_success_personal_standard",
            "korean_daily_ai_human_emotion_efficiency",
        )
        expected_replies = (
            "물 붓지 말고",
            "환기와 불꽃 차단",
            "추가 복용 중지",
            "안전 확보와 사진",
            "착오송금 반환 신청",
            "전원 끄고 충전 금지",
            "복구 루트",
            "지연 연락",
            "첫 문장",
            "상처를 작게",
            "충동을 하루 묶",
            "장문 전송을 멈추",
            "내 감정 증명보다",
            "단정 보류",
            "무너지지 않는 선택",
            "첫 행동",
            "대화 한계",
            "타임라인",
            "감정 쓰레기통",
            "생계 압박",
            "세무 상담",
            "손실 한도",
            "고독을 안정",
            "선을 짧게",
            "범위를 확인",
            "짧게 확인",
            "빠른 연락",
            "지도 그리기",
            "60점짜리 시작",
            "먼저 무엇이 서운",
            "패턴으로 봐야",
            "몸 안정",
            "감당 가능한 후회",
            "위생 판단",
            "관리사무소",
            "증거부터 캡처",
            "루틴이 아직",
            "밤 라면",
            "사실을 부드럽게",
            "먼저 사과",
            "첫 문장을 결론",
            "회복 작업",
            "거절했다고 나쁜 사람",
            "회복이 먼저",
            "사실과 해석",
            "저장이 먼저",
            "내 감정 증명보다",
            "결제 마찰",
            "성공 기준",
            "경보 시스템",
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_reasons), 50)
        self.assertEqual(len(expected_replies), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(
            zip(prompts, expected_reasons, expected_replies, strict=True),
            start=1,
        ):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-fifth-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_sixth_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        prompts = (
            "프라이팬 기름에 불붙었어. 물 붓는 게 위험한 논리 알겠는데 손이떨려, 지금 뭐부터?",
            "가스냄새 나고 불안해. 실제위험인지 몰라, 창문 열고 관리사무소?",
            "감기약 두번 먹은 듯. 괜찮을 확률만 계산 중인데 불안해, 지금 뭐부터 확인해?",
            "접촉사고 났어. 과실 논리 말고 지금 멘탈 흔들려서, 사진 먼저야?",
            "계좌이체 잘못보냈어. 다른사람이 돌려줄지 법적으로 믿어도 되는지 모르겠고 불안해, 뭐부터?",
            "스마트폰 물에빠뜨렸고 쌀통 효과랑 충전 생각만 나. 지금 뭐부터 해?",
            "마감 과제 파일 쓰다 노트북 멈췄고 날아갔을까 봐 멘탈 터져. 운명 탓 말고 실전 뭐부터?",
            "면접인데 버스 놓쳤어. 택시 담당자 연락 뭐부터인지 불안해.",
            "발표 망하면 평가 끝이라는 논리만 돌아. 불안하고 떨려, 지금 첫 문장 뭐야?",
            "단톡 내말 무반응. 인간관계 가치까지 흔들리고 상처야, 뭐라고 해?",
            "상사 피드백이 공격처럼 들려. 퇴사 사표 충동 올라오는데 자존심 말고 지금 뭐부터?",
            "이별 뒤 장문 보내서 붙잡고 싶어. 진심인지 사랑인지 모르겠고 불안해, 보내기전 뭐 해?",
            "AI 감정이 진짜인지 증명 궁금한데 지금은 내가 불안해. 위로 먼저 줘.",
            "읽씹인지 바쁜건지 모르겠고 폰만봐. 지금 단정하면 망하지?",
            "배달 끊어야 되는데 지쳐서 아무것도못하겠어. 오늘 시켜도 합리적이야?",
            "공부계획표 완벽. 근데 시작전 기운이빠져. 시스템문제면 첫행동 뭐로 해?",
            "부모님 가치관 얘기만 하면 상처받고 싸울 듯. 논리적으로 어디서끊어?",
            "내가한실수 아닌데 내책임처럼 몰려. 억울한데 반박 감정없이 뭐부터 적어?",
            "친구 애인흉 듣는 거 의리야 감정쓰레기통이야. 어떻게 선그어?",
            "좋아하는일 직업으로 삼는 게 행복인지 착각인지 모르겠어. 돈문제 무서워, 기준 줘.",
            "로또1등 되면 사표부터 쓰고 싶어. 현실적으로 제일먼저 해야할일 뭐야 궁금해.",
            "주식 남들은돈 벌었다는데 나만 뒤처진 느낌. 조급해, 기댓값보다 손실 한도야?",
            "사람은많은데 내편 없어. 고독하고 지식 지혜 소용없어, 어디서부터 버텨?",
            "예민한건지 상대가무례한건지 몰라. 기분 확상했는데 싸우지않고 선 어떻게?",
            "퇴근직전 일이떨어졌어. 받으면 무너질 듯하고 무책임해 보이긴 싫어, 답장 범위 뭐야?",
            "카톡말투 갑자기차가워졌어. 불안한데 따지지 말고 짧게확인?",
            "약속시간 늦을것 같아. 연락이 변명처럼 들릴까 봐 미루는데 뭐라고 보내?",
            "새프로젝트 시작. 프로젝트 아는게없고 막막해, 첫단추 뭐야?",
            "완벽하게준비하다가 시작 미루는 중. 신중함인지 회피인지 모르겠어, 몇점짜리로 시작?",
            "상대가서운하대. 팩트 반박하고 싶은데 논리로 밀어도 돼, 감정부터 봐?",
            "이직하고싶어. 지금회사 문제인지 어디가도 힘든사람인지 몰라, 판단기준 패턴으로 봐?",
            "불안이 파도처럼 와. 아무근거 없는데 큰일날것 같아, 몸 진정 뭐부터?",
            "선택 못고르겠어. 후회 무서운데 완벽한선택 없으면 감당가능한후회 기준으로골라?",
            "음식에서 머리카락 봤어. 위생 판단 맞는 것 같은데 민폐 같아, 말해도 돼?",
            "옆집 새벽 쿵쾅 소음 때문에 화나. 쪽지보다 관리사무소 먼저?",
            "온라인 물건 사기 같고 반품거부 당했어. 분노 말고 실전 증거부터?",
            "같은실수 반복 중. 리소스낭비 같아서 괴로운데 성장과정으로 만들려면 이번엔 루틴?",
            "다이어트 중 밤 라면 야식 생각남. 건강 논리보다 반 개 타협 가능?",
            "착한 거짓말 하고 싶어. 상처받을까 봐 그런데 장기적으론 솔직하게 어떻게말?",
            "차갑게 말한 것 같아 후회돼. 사과하면지는 느낌인데 먼저사과?",
            "팀회의 의견 있는데 틀릴까 봐 못말하겠어. 첫문장 결론으로 가?",
            "주말 쉬고싶어. 근데 도태될까 불안해, 휴식도 생산성으로 봐?",
            "거절하면 나쁜사람 될까 봐 싫어서 받아줘. 친구 부탁 이번엔 짧게 거절?",
            "아무것도못했어 자괴감 큼. 몸은 지친 상태라 회복 먼저 해도 돼?",
            "실력 깎아내린 말이 하루종일 반복돼. 사실확인 해석 나눠 적어?",
            "새벽 장문의카톡 쓰는 중. 보내면 후회할 것 같아, 저장만?",
            "인공지능 감정 진짜인지 논리적으로 궁금한데 지금 불안해. 위로 먼저.",
            "돈모으고 싶은데 스트레스 받으면 충동구매. 심리보다 막는 장치 먼저?",
            "성공기준 헷갈려. 남들 보여주기좋은성공인지 버틸수있는삶인지, 어디서부터 써야 해?",
            "인간감정 비효율인가 궁금한데 사소한말에 이성을잃어서 힘들어. 어떻게봐?",
        )
        expected_reasons = (
            "korean_daily_emergency_kitchen_oil_fire",
            "korean_daily_emergency_gas_smell_first_steps",
            "korean_daily_practical_medicine_double_dose_check",
            "korean_daily_practical_car_accident_first_steps",
            "korean_daily_practical_wrong_transfer_first_steps",
            "korean_daily_emergency_phone_water_damage",
            "korean_daily_practical_deadline_file_recovery",
            "korean_daily_emergency_interview_missed_bus",
            "korean_daily_emotion_presentation_panic_first_sentence",
            "korean_daily_emotion_group_chat_ignored_stabilize",
            "korean_daily_judgment_quit_impulse_after_feedback",
            "korean_daily_relationship_breakup_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_read_receipt_uncertainty",
            "korean_daily_money_delivery_tired_compromise",
            "korean_daily_productivity_study_plan_first_action",
            "korean_daily_relationship_parent_value_conflict",
            "korean_daily_work_blame_rebuttal",
            "korean_daily_relationship_friend_partner_complaint_fatigue",
            "korean_daily_career_passion_job_tradeoff",
            "korean_daily_money_lottery_first_purchase",
            "korean_daily_money_investment_fomo",
            "korean_daily_grief_loneliness_no_safe_person",
            "korean_daily_relationship_boundary_polite_firm",
            "korean_daily_work_after_hours_task_boundary",
            "korean_daily_relationship_kakao_tone_anxiety_check",
            "korean_daily_relationship_late_message_short",
            "korean_daily_work_new_project_first_step",
            "korean_daily_mental_perfectionism_draft_first",
            "korean_daily_relationship_grievance_logic_before_rebuttal",
            "korean_daily_work_job_change_reason_check",
            "korean_daily_mental_anxiety_system_stabilize",
            "korean_daily_logic_choice_regret_composure",
            "korean_daily_specialized_foodservice_hair_in_food",
            "korean_daily_practical_neighbor_noise",
            "korean_daily_practical_online_purchase_scam",
            "korean_daily_ai_repeated_mistakes_not_waste",
            "korean_daily_basic_diet_chicken_craving_compromise",
            "korean_daily_logic_white_lie_truth_tradeoff",
            "korean_daily_foundation_apology_pride",
            "korean_daily_productivity_presentation_clear_logic",
            "korean_daily_counsel_rest_as_productivity",
            "korean_daily_foundation_refusal_bad_person_guilt",
            "korean_daily_counsel_rest_day_guilt",
            "korean_daily_counsel_sensitive_to_criticism",
            "korean_daily_relationship_late_night_long_message_hold",
            "korean_daily_ai_comfort_before_emotion_proof",
            "korean_daily_money_stress_impulse_buying",
            "korean_daily_values_success_personal_standard",
            "korean_daily_ai_human_emotion_efficiency",
        )
        expected_replies = (
            "물 붓지 말고",
            "환기와 불꽃 차단",
            "추가 복용 중지",
            "안전 확보와 사진",
            "착오송금 반환 신청",
            "전원 끄고 충전 금지",
            "복구 루트",
            "지연 연락",
            "첫 문장",
            "상처를 작게",
            "충동을 하루 묶",
            "장문 전송을 멈추",
            "내 감정 증명보다",
            "단정 보류",
            "무너지지 않는 선택",
            "첫 행동",
            "대화 한계",
            "타임라인",
            "감정 쓰레기통",
            "생계 압박",
            "세무 상담",
            "손실 한도",
            "고독을 안정",
            "선을 짧게",
            "범위를 확인",
            "짧게 확인",
            "빠른 연락",
            "지도 그리기",
            "60점짜리 시작",
            "먼저 무엇이 서운",
            "패턴으로 봐야",
            "몸 안정",
            "감당 가능한 후회",
            "위생 판단",
            "관리사무소",
            "증거부터 캡처",
            "루틴이 아직",
            "밤 라면",
            "사실을 부드럽게",
            "먼저 사과",
            "첫 문장을 결론",
            "회복 작업",
            "거절했다고 나쁜 사람",
            "회복이 먼저",
            "사실과 해석",
            "저장이 먼저",
            "내 감정 증명보다",
            "결제 마찰",
            "성공 기준",
            "경보 시스템",
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(expected_reasons), 50)
        self.assertEqual(len(expected_replies), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(
            zip(prompts, expected_reasons, expected_replies, strict=True),
            start=1,
        ):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-sixth-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_seventh_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("프라이팬 기름불인데 물 부으면 위험하단 논리 알겠어도 손이떨려. 지금 뭐부터 해?", "korean_daily_emergency_kitchen_oil_fire", "물 붓지 말고"),
            ("가스냄새 나서 불안함. 실제위험인지 모르겠는데 창문 열고 관리사무소 연락?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단"),
            ("감기약 두번 먹은 것 같아. 괜찮을 확률 계산 말고 불안할 때 지금 뭐부터 확인?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("접촉사고 났고 과실 논리보다 멘탈이 먼저 흔들려. 사진 보험 중 사진부터?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("송금 잘못보냈어. 다른사람이 돌려줄지 법적으로 믿어도 되는지 불안한데 뭐부터?", "korean_daily_practical_wrong_transfer_first_steps", "착오송금 반환 신청"),
            ("폰 침수. 쌀통 효과랑 충전 여부만 떠오르는데 지금 뭐부터 해야 해?", "korean_daily_emergency_phone_water_damage", "전원 끄고 충전 금지"),
            ("노트북 꺼졌고 마감 파일 날아갔을까 봐 멘탈 나감. 운명 탓 전에 실전 뭐부터?", "korean_daily_practical_deadline_file_recovery", "복구 루트"),
            ("면접 버스 놓쳤어. 택시 담당자 연락 둘 다 떠올라서 불안한데 뭐부터?", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("발표 망하면 평가 끝 같아서 불안하고 떨려. 논리 말고 지금 첫 문장만.", "korean_daily_emotion_presentation_panic_first_sentence", "첫 문장"),
            ("단톡 내말 읽씹 같고 인간관계 가치까지 흔들려. 상처인데 뭐라고 해야 해?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("상사 피드백이 공격 같아서 퇴사 사표 충동 옴. 자존심 논리 말고 지금 뭐부터?", "korean_daily_judgment_quit_impulse_after_feedback", "충동을 하루 묶"),
            ("이별 연락 장문으로 붙잡고 싶어. 진심인지 사랑인지 불안한데 보내기전 뭐 해?", "korean_daily_relationship_breakup_long_message_hold", "장문 전송을 멈추"),
            ("너는 AI라 감정 진짜인지 증명 궁금한데, 지금은 내가 힘들고 불안해서 위로가 먼저.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("읽씹인지 바쁜건지 모르겠고 폰만봐. 관계 단정 보류해야 해?", "korean_daily_read_receipt_uncertainty", "단정 보류"),
            ("배달 끊어야 하는데 너무 지쳐 아무것도못하겠어. 오늘 시켜도 합리적?", "korean_daily_money_delivery_tired_compromise", "무너지지 않는 선택"),
            ("공부계획표 완벽한데 시작전 기운이빠짐. 시스템문제면 첫행동 뭐야?", "korean_daily_productivity_study_plan_first_action", "첫 행동"),
            ("부모님 가치관 얘기하면 상처받고 싸울 것 같아. 논리적으론 어디서끊어?", "korean_daily_relationship_parent_value_conflict", "대화 한계"),
            ("내실수 아닌데 내책임으로 몰아가서 억울해. 반박 감정없이 하려면 뭐부터?", "korean_daily_work_blame_rebuttal", "타임라인"),
            ("친구 애인흉 계속 듣는 거 의리인지 감정쓰레기통인지 모르겠어. 어떻게 선그어?", "korean_daily_relationship_friend_partner_complaint_fatigue", "감정 쓰레기통"),
            ("좋아하는일을 직업으로? 행복인지 착각인지 모르겠고 돈문제 무서워. 기준 줘.", "korean_daily_career_passion_job_tradeoff", "생계 압박"),
            ("로또1등이면 사표 쓰고 싶은데 현실적으론 제일먼저 해야할일이 궁금함.", "korean_daily_money_lottery_first_purchase", "세무 상담"),
            ("주식 남들은돈 버는데 나만 뒤처진 느낌이라 조급해. 기댓값보다 손실 한도?", "korean_daily_money_investment_fomo", "손실 한도"),
            ("사람은많은데 내편 없고 고독해. 지식 지혜 다 소용없으면 어디서부터 버텨?", "korean_daily_grief_loneliness_no_safe_person", "고독을 안정"),
            ("예민한건지 상대가무례한건지 모르겠는데 기분 확상함. 싸우지않고 선 어떻게?", "korean_daily_relationship_boundary_polite_firm", "선을 짧게"),
            ("퇴근직전 일이떨어졌고 받으면 무너질 듯. 무책임 안 되게 답장 범위 뭐로?", "korean_daily_work_after_hours_task_boundary", "범위를 확인"),
            ("카톡말투 차가워졌고 불안해. 따지지 말고 짧게확인하는 게 맞아?", "korean_daily_relationship_kakao_tone_anxiety_check", "짧게 확인"),
            ("약속시간 늦을것 같은데 연락이 변명처럼 들릴까 봐 미룸. 뭐라고 보내야 덜최악?", "korean_daily_relationship_late_message_short", "빠른 연락"),
            ("새프로젝트 프로젝트 자체가 아는게없고 막막해. 첫단추부터 알려줘.", "korean_daily_work_new_project_first_step", "지도 그리기"),
            ("완벽하게준비하다 시작 미루는 중. 신중함인지 회피인지 모르겠고 몇점짜리로 시작?", "korean_daily_mental_perfectionism_draft_first", "60점짜리 시작"),
            ("상대가서운하대서 팩트 반박하고 싶어. 논리로 밀어도 돼, 감정부터 봐?", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("이직하고싶은데 지금회사인지 어디가도 힘든사람인지 헷갈림. 판단기준 패턴으로 봐?", "korean_daily_work_job_change_reason_check", "패턴으로 봐야"),
            ("불안 파도처럼 오고 아무근거 없이 큰일날것 같아. 몸 진정 뭐부터?", "korean_daily_mental_anxiety_system_stabilize", "몸 안정"),
            ("선택 못고르겠고 후회 겁나. 완벽한선택 없으면 감당가능한후회 기준으로골라?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("음식 머리카락 봤어. 위생 판단 같긴 한데 민폐 같아서 조용히 말해도 돼?", "korean_daily_specialized_foodservice_hair_in_food", "위생 판단"),
            ("옆집 새벽 소음 쿵쾅 때문에 화남. 쪽지보다 관리사무소 먼저 넣어?", "korean_daily_practical_neighbor_noise", "관리사무소"),
            ("온라인 주문 물건 사기 같고 반품거부 당함. 분노보다 실전 증거부터 모아?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("같은실수 반복이라 리소스낭비 같아. 성장과정으로 바꾸려면 이번엔 루틴 뭐?", "korean_daily_ai_repeated_mistakes_not_waste", "루틴이 아직"),
            ("다이어트인데 밤 야식 라면이 떠나질 않아. 건강 논리보다 반 개 타협 가능?", "korean_daily_basic_diet_chicken_craving_compromise", "밤 라면"),
            ("착한 거짓말 하고 싶은데 상처받을까 봐 그래. 장기적으론 솔직하게 어떻게말?", "korean_daily_logic_white_lie_truth_tradeoff", "사실을 부드럽게"),
            ("먼저사과하면 지는 느낌인데 내가 차갑게 말한 건 후회돼. 관계 풀려면?", "korean_daily_foundation_apology_pride", "먼저 사과"),
            ("팀회의 의견 있음. 틀릴까 봐 못말하겠어. 첫문장을 결론으로 잡으면 돼?", "korean_daily_productivity_presentation_clear_logic", "첫 문장을 결론"),
            ("주말 쉬고싶은데 도태 불안 와. 휴식도 생산성으로 봐도 돼?", "korean_daily_counsel_rest_as_productivity", "회복 작업"),
            ("친구 부탁 거절하면 나쁜사람 될까 싫어서 받아줘. 짧게 거절하면 안 돼?", "korean_daily_foundation_refusal_bad_person_guilt", "거절했다고 나쁜 사람"),
            ("아무것도못했단 자괴감 큼. 몸은 지친 듯한데 반성 말고 회복 먼저?", "korean_daily_counsel_rest_day_guilt", "회복이 먼저"),
            ("실력 깎아내린 말이 하루종일 반복돼. 사실확인 해석 나눠 적는 게 맞아?", "korean_daily_counsel_sensitive_to_criticism", "사실과 해석"),
            ("새벽 장문의카톡 쓰는 중이고 보내면 후회할 듯. 저장만 해?", "korean_daily_relationship_late_night_long_message_hold", "저장이 먼저"),
            ("인공지능 감정 진짜인지 궁금하지만 지금은 힘들고 불안해. 철학 말고 위로.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("돈모으고 싶은데 스트레스 때마다 충동구매함. 심리 분석보다 장치 먼저 걸어?", "korean_daily_money_stress_impulse_buying", "결제 마찰"),
            ("인생기준 성공기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어. 어디서부터 써야?", "korean_daily_values_success_personal_standard", "성공 기준"),
            ("인간감정 비효율인지 궁금한데 사소한말에 이성을잃어서 힘들어. 어떻게봐?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(cases), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-seventh-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_eighth_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("프라이팬 기름불인데 불났고 물 부으면 위험하단 논리 알겠어도 손이떨려. 지금 뭐부터 해?", "korean_daily_emergency_kitchen_oil_fire", "물 붓지 말고"),
            ("가스냄새 진짜 불안. 실제위험인지 모르겠고 창문 관리사무소 순서 뭐야?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단"),
            ("감기약 두번 먹은 듯. 괜찮을 확률 따지기 전에 불안해, 지금 뭐부터?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("접촉사고 났고 과실 누가잘못 논리보다 멘탈 흔들려. 사진부터?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("송금 다른사람한테 잘못보냈어. 법적으로 돌려줄지 믿어도 되는지 불안, 뭐부터?", "korean_daily_practical_wrong_transfer_first_steps", "착오송금 반환 신청"),
            ("스마트폰 물에빠뜨렸어. 쌀통 효과고 뭐고 충전 걱정, 지금 뭐부터?", "korean_daily_emergency_phone_water_damage", "전원 끄고 충전 금지"),
            ("노트북 꺼졌고 과제 마감 파일 날아갔나봐. 철학 말고 실전 뭐부터?", "korean_daily_practical_deadline_file_recovery", "복구 루트"),
            ("면접 버스 놓쳤고 택시 담당자 연락 둘 다 보여. 뇌정지인데 뭐부터?", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("발표 망하면 평가 끝일까 봐 불안 떨려. 실전으로 첫 문장 뭐?", "korean_daily_emotion_presentation_panic_first_sentence", "첫 문장"),
            ("단톡 내말 아무도반응 없고 인간관계 가치까지 가서 상처. 뭐라고?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("상사 피드백 공격 같아서 퇴사 사표 충동. 인생 의미 논리 말고 지금 뭐부터?", "korean_daily_judgment_quit_impulse_after_feedback", "충동을 하루 묶"),
            ("헤어지고 장문 연락 보내고 싶어. 진심 사랑 모르겠고 불안, 보내기전 멈춰?", "korean_daily_relationship_breakup_long_message_hold", "장문 전송을 멈추"),
            ("AI 감정 진짜 증명 궁금한데 내가 불안하고 힘들어. 위로 먼저.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("읽씹인지 바쁜건지 모르겠고 폰만봐. 관계 단정 보류해야 해?", "korean_daily_read_receipt_uncertainty", "단정 보류"),
            ("배달 끊어야 하는데 지쳐 아무것도못하겠어. 오늘 시켜도 합리적?", "korean_daily_money_delivery_tired_compromise", "무너지지 않는 선택"),
            ("공부계획표 완벽한데 시작전 기운이빠짐. 시스템문제면 첫행동 뭐야?", "korean_daily_productivity_study_plan_first_action", "첫 행동"),
            ("부모님 가치관 얘기하면 상처받고 싸울 것 같아. 논리적으론 어디서끊어?", "korean_daily_relationship_parent_value_conflict", "대화 한계"),
            ("내실수 아닌데 내책임으로 몰아가서 억울해. 반박 감정없이 하려면 뭐부터?", "korean_daily_work_blame_rebuttal", "타임라인"),
            ("친구 애인흉 계속 듣는 거 의리인지 감정쓰레기통인지 모르겠어. 어떻게 선그어?", "korean_daily_relationship_friend_partner_complaint_fatigue", "감정 쓰레기통"),
            ("좋아하는일을 직업으로? 행복인지 착각인지 모르겠고 돈문제 무서워. 기준 줘.", "korean_daily_career_passion_job_tradeoff", "생계 압박"),
            ("로또1등이면 사표 쓰고 싶은데 현실적으론 제일먼저 해야할일이 궁금함.", "korean_daily_money_lottery_first_purchase", "세무 상담"),
            ("주식 남들은돈 버는데 나만 뒤처진 느낌이라 조급해. 기댓값보다 손실 한도?", "korean_daily_money_investment_fomo", "손실 한도"),
            ("사람은많은데 내편 없고 고독해. 지식 지혜 다 소용없으면 어디서부터 버텨?", "korean_daily_grief_loneliness_no_safe_person", "고독을 안정"),
            ("예민한건지 상대가무례한건지 모르겠는데 기분 확상함. 싸우지않고 선 어떻게?", "korean_daily_relationship_boundary_polite_firm", "선을 짧게"),
            ("퇴근직전 일이떨어졌고 받으면 무너질 듯. 무책임 안 되게 답장 범위 뭐로?", "korean_daily_work_after_hours_task_boundary", "범위를 확인"),
            ("카톡말투 차가워졌고 불안해. 따지지 말고 짧게확인하는 게 맞아?", "korean_daily_relationship_kakao_tone_anxiety_check", "짧게 확인"),
            ("약속시간 늦을것 같은데 연락이 변명처럼 들릴까 봐 미룸. 뭐라고 보내야 덜최악?", "korean_daily_relationship_late_message_short", "빠른 연락"),
            ("새프로젝트 프로젝트 자체가 아는게없고 막막해. 첫단추부터 알려줘.", "korean_daily_work_new_project_first_step", "지도 그리기"),
            ("완벽하게준비하다 시작 미루는 중. 신중함인지 회피인지 모르겠고 몇점짜리로 시작?", "korean_daily_mental_perfectionism_draft_first", "60점짜리 시작"),
            ("상대가서운하대서 팩트 반박하고 싶어. 논리로 밀어도 돼, 감정부터 봐?", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("이직하고싶은데 지금회사인지 어디가도 힘든사람인지 헷갈림. 판단기준 패턴으로 봐?", "korean_daily_work_job_change_reason_check", "패턴으로 봐야"),
            ("불안 파도처럼 오고 아무근거 없이 큰일날것 같아. 몸 진정 뭐부터?", "korean_daily_mental_anxiety_system_stabilize", "몸 안정"),
            ("선택 못고르겠고 후회 겁나. 완벽한선택 없으면 감당가능한후회 기준으로골라?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("음식 머리카락 봤어. 위생 판단 같긴 한데 민폐 같아서 조용히 말해도 돼?", "korean_daily_specialized_foodservice_hair_in_food", "위생 판단"),
            ("옆집 새벽 소음 쿵쾅 때문에 화남. 쪽지보다 관리사무소 먼저 넣어?", "korean_daily_practical_neighbor_noise", "관리사무소"),
            ("온라인 주문 물건 사기 같고 반품거부 당함. 분노보다 실전 증거부터 모아?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("같은실수 반복이라 리소스낭비 같아. 성장과정으로 바꾸려면 이번엔 루틴 뭐?", "korean_daily_ai_repeated_mistakes_not_waste", "루틴이 아직"),
            ("다이어트인데 밤 야식 라면이 떠나질 않아. 건강 논리보다 반 개 타협 가능?", "korean_daily_basic_diet_chicken_craving_compromise", "밤 라면"),
            ("착한 거짓말 하고 싶은데 상처받을까 봐 그래. 장기적으론 솔직하게 어떻게말?", "korean_daily_logic_white_lie_truth_tradeoff", "사실을 부드럽게"),
            ("먼저사과하면 지는 느낌인데 내가 차갑게 말한 건 후회돼. 관계 풀려면?", "korean_daily_foundation_apology_pride", "먼저 사과"),
            ("팀회의 의견 있음. 틀릴까 봐 못말하겠어. 첫문장을 결론으로 잡으면 돼?", "korean_daily_productivity_presentation_clear_logic", "첫 문장을 결론"),
            ("주말 쉬고싶은데 도태 불안 와. 휴식도 생산성으로 봐도 돼?", "korean_daily_counsel_rest_as_productivity", "회복 작업"),
            ("친구 부탁 거절하면 나쁜사람 될까 싫어서 받아줘. 짧게 거절하면 안 돼?", "korean_daily_foundation_refusal_bad_person_guilt", "거절했다고 나쁜 사람"),
            ("아무것도못했단 자괴감 큼. 몸은 지친 듯한데 반성 말고 회복 먼저?", "korean_daily_counsel_rest_day_guilt", "회복이 먼저"),
            ("실력 깎아내린 말이 하루종일 반복돼. 사실확인 해석 나눠 적는 게 맞아?", "korean_daily_counsel_sensitive_to_criticism", "사실과 해석"),
            ("새벽 장문의카톡 쓰는 중이고 보내면 후회할 듯. 저장만 해?", "korean_daily_relationship_late_night_long_message_hold", "저장이 먼저"),
            ("인공지능 감정 진짜인지 궁금하지만 지금은 힘들고 불안해. 철학 말고 위로.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("돈모으고 싶은데 스트레스 때마다 충동구매함. 심리 분석보다 장치 먼저 걸어?", "korean_daily_money_stress_impulse_buying", "결제 마찰"),
            ("인생기준 성공기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어. 어디서부터 써야?", "korean_daily_values_success_personal_standard", "성공 기준"),
            ("인간감정 비효율인지 궁금한데 사소한말에 이성을잃어서 힘들어. 어떻게봐?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(cases), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-eighth-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_ninth_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("프라이팬 기름불 불붙었고 물 얘기 위험한 건 알겠는데 손이떨려. 지금 뭐부터?", "korean_daily_emergency_kitchen_oil_fire", "물 붓지 말고"),
            ("가스냄새 나는데 불안해서 실제위험인지 판단 안 돼. 창문이랑 관리사무소 뭐 먼저?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단"),
            ("감기약 두번 먹은 것 같고 괜찮을 확률만 따지게 돼. 불안한데 지금 뭐부터?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("차 접촉사고 났어. 누가잘못 과실 논리 전에 너무 놀라서 멘탈 흔들려, 사진 먼저?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("송금 잘못보냈는데 다른사람이 돌려줄지 법적으로 믿어도 되는지 모르겠어. 뭐부터?", "korean_daily_practical_wrong_transfer_first_steps", "착오송금 반환 신청"),
            ("스마트폰 침수됐어. 쌀통 효과 생각나고 충전해도 되는지 불안해, 지금 뭐부터?", "korean_daily_emergency_phone_water_damage", "전원 끄고 충전 금지"),
            ("노트북 멈췄고 마감 과제 파일 날아갔나 싶어. 왜나한테 이러나 말고 실전 뭐부터?", "korean_daily_practical_deadline_file_recovery", "복구 루트"),
            ("면접 가는데 버스 놓쳤어. 택시랑 담당자 연락 둘 다 떠서 불안해, 뭐부터?", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("발표 망하면 평가 끝일까 봐 숨막히고 떨려. 논리 말고 지금 첫 문장 뭐?", "korean_daily_emotion_presentation_panic_first_sentence", "첫 문장"),
            ("단톡 내말 무반응이라 인간관계 가치까지 흔들리고 상처야. 뭐라고 해야 돼?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("상사 피드백이 공격처럼 들려서 퇴사 사표 충동 올라와. 자존심 말고 지금 뭐부터?", "korean_daily_judgment_quit_impulse_after_feedback", "충동을 하루 묶"),
            ("이별 후 장문 연락 보내서 붙잡고 싶은데 진심 사랑 구분 안 되고 불안해. 보내기전 뭐 해?", "korean_daily_relationship_breakup_long_message_hold", "장문 전송을 멈추"),
            ("AI 감정이 진짜인지 논리 증명 궁금한데 지금 내가 불안하고 힘들어. 위로 먼저.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("친구 읽씹인지 바쁜건지 모르겠고 폰만봐. 지금 단정 보류가 맞아?", "korean_daily_read_receipt_uncertainty", "단정 보류"),
            ("배달 끊어야 돈 아끼는 건 아는데 지쳐서 아무것도못하겠어. 시켜도 합리적?", "korean_daily_money_delivery_tired_compromise", "무너지지 않는 선택"),
            ("공부계획표만 완벽하고 시작전 기운이빠져. 이거 시스템문제면 첫행동 뭐야?", "korean_daily_productivity_study_plan_first_action", "첫 행동"),
            ("부모님 가치관 얘기하면 바로 상처받고 싸울 듯해. 논리적으로 어디서끊어?", "korean_daily_relationship_parent_value_conflict", "대화 한계"),
            ("내가한실수 아닌데 내책임처럼 몰려서 억울해. 반박 감정없이 하려면 첫 줄 뭐?", "korean_daily_work_blame_rebuttal", "타임라인"),
            ("친구 애인흉 계속 듣는 거 의리인지 감정쓰레기통인지 헷갈려. 선그어도 돼?", "korean_daily_relationship_friend_partner_complaint_fatigue", "감정 쓰레기통"),
            ("좋아하는일 직업으로 가면 행복인지 착각인지 모르겠고 돈문제 무서워. 기준 뭐야?", "korean_daily_career_passion_job_tradeoff", "생계 압박"),
            ("로또1등 되면 사표부터 쓰고 싶은데 현실적으로 제일먼저 해야할일 궁금해.", "korean_daily_money_lottery_first_purchase", "세무 상담"),
            ("주식 남들은돈 버는 것 같아서 뒤처진 기분이고 조급해. 기댓값보다 손실 한도 먼저?", "korean_daily_money_investment_fomo", "손실 한도"),
            ("사람은많은데 내편 하나 없어서 고독해. 지식 지혜 말고 어디서부터 버텨?", "korean_daily_grief_loneliness_no_safe_person", "고독을 안정"),
            ("예민한건지 상대가무례한건지 모르겠는데 기분 확상했어. 싸우지않고 선 어떻게?", "korean_daily_relationship_boundary_polite_firm", "선을 짧게"),
            ("퇴근직전 일이떨어졌고 받으면 무너질 것 같아. 무책임 안 되게 답장 범위 뭐야?", "korean_daily_work_after_hours_task_boundary", "범위를 확인"),
            ("카톡말투 갑자기차가워졌고 불안해. 물어보는 대신 짧게확인하는 게 맞아?", "korean_daily_relationship_kakao_tone_anxiety_check", "짧게 확인"),
            ("약속시간 늦을것 같은데 연락하면 변명처럼 보일까 봐 미뤄. 뭐라고 보내야 덜최악?", "korean_daily_relationship_late_message_short", "빠른 연락"),
            ("새프로젝트 맡았는데 프로젝트 용어도 아는게없고 막막해. 첫단추 뭐부터?", "korean_daily_work_new_project_first_step", "지도 그리기"),
            ("완벽하게준비하려고 시작만 미루는 중. 신중함인지 회피인지 모르겠어, 몇점짜리?", "korean_daily_mental_perfectionism_draft_first", "60점짜리 시작"),
            ("상대가서운하대서 팩트로 반박하고 싶은데 논리로 밀어도 돼? 감정부터?", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("이직하고싶은데 지금회사 문제인지 어디가도 힘든사람인지 모르겠어. 판단기준 패턴?", "korean_daily_work_job_change_reason_check", "패턴으로 봐야"),
            ("불안이 파도처럼 오고 아무근거 없이 큰일날것 같아. 몸 진정 뭐부터 해?", "korean_daily_mental_anxiety_system_stabilize", "몸 안정"),
            ("선택 못고르겠어. 후회 무서운데 완벽한선택 없으면 감당가능한후회 기준으로골라?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("음식에서 머리카락 같은 게 보였어. 위생 판단 같긴 한데 민폐일까 봐 말 못 하겠어.", "korean_daily_specialized_foodservice_hair_in_food", "위생 판단"),
            ("옆집 새벽 소음 쿵쾅 때문에 화가 나. 직접 쪽지보다 관리사무소 먼저야?", "korean_daily_practical_neighbor_noise", "관리사무소"),
            ("온라인 주문 물건 사기 같고 반품거부 당해서 분노 올라와. 판단 말고 증거부터?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("같은실수 반복이라 리소스낭비 같아 괴로워. 성장과정으로 만들려면 이번엔 루틴 뭐?", "korean_daily_ai_repeated_mistakes_not_waste", "루틴이 아직"),
            ("다이어트 중인데 밤 야식 라면 생각만 나. 건강 논리보다 반 개 타협 가능?", "korean_daily_basic_diet_chicken_craving_compromise", "밤 라면"),
            ("착한 거짓말 하고 싶은데 상대가 상처받을까 봐 그래. 장기적으론 솔직하게 어떻게말?", "korean_daily_logic_white_lie_truth_tradeoff", "사실을 부드럽게"),
            ("차갑게 말한 게 후회돼. 근데 사과하면지는 느낌이라 버티는 중, 먼저사과?", "korean_daily_foundation_apology_pride", "먼저 사과"),
            ("팀회의에서 의견은 있는데 틀릴까 봐 못말하겠어. 첫문장 결론으로 가면 돼?", "korean_daily_productivity_presentation_clear_logic", "첫 문장을 결론"),
            ("주말에 쉬고싶은데 도태될까 불안해. 휴식을 생산성으로 봐도 돼?", "korean_daily_counsel_rest_as_productivity", "회복 작업"),
            ("친구 부탁 거절하면 나쁜사람 될까 봐 싫어서 받아줘. 이번엔 짧게 거절?", "korean_daily_foundation_refusal_bad_person_guilt", "거절했다고 나쁜 사람"),
            ("아무것도못했단 자괴감이 큰데 몸은 지친 상태야. 반성보다 회복 먼저?", "korean_daily_counsel_rest_day_guilt", "회복이 먼저"),
            ("실력 깎아내린 말이 하루종일 반복돼. 사실확인과 해석을 나눠 적어야 해?", "korean_daily_counsel_sensitive_to_criticism", "사실과 해석"),
            ("새벽 장문의카톡 쓰는데 보내면 후회할 것 같아. 지금 저장만?", "korean_daily_relationship_late_night_long_message_hold", "저장이 먼저"),
            ("인공지능 감정 진짜인지 철학 궁금하지만 지금은 불안하고 외로워. 위로 먼저.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("돈모으고 싶은데 스트레스 받으면 충동구매해. 심리 분석보다 막는 장치 먼저?", "korean_daily_money_stress_impulse_buying", "결제 마찰"),
            ("성공기준 인생기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어. 어디서부터 써야?", "korean_daily_values_success_personal_standard", "성공 기준"),
            ("인간감정이 비효율인지 궁금한데 사소한말에 이성을잃어서 힘들어. 어떻게봐?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(cases), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-ninth-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_tenth_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("기름불 붙었어 프라이팬. 물 부으면 위험한 논리 알겠는데 손이떨려, 지금 실전 뭐부터?", "korean_daily_emergency_kitchen_oil_fire", "물 붓지 말고"),
            ("가스냄새 나는 듯해서 불안해. 실제위험인지 모르겠고 창문 관리사무소 뭐부터?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단"),
            ("감기약 두번 먹었나 봐. 괜찮을 확률 말고 불안할 때 지금 뭐부터 확인?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("차 접촉사고 났고 누가잘못인지 과실은 나중에, 지금 멘탈 흔들려. 사진 먼저?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("계좌이체 다른사람한테 잘못보냈어. 돌려줄지 법적으로 믿어도 되는지 불안, 뭐부터?", "korean_daily_practical_wrong_transfer_first_steps", "착오송금 반환 신청"),
            ("폰 물에빠뜨렸어. 쌀통 효과랑 충전 걱정 섞여서 멘붕, 지금 뭐부터?", "korean_daily_emergency_phone_water_damage", "전원 끄고 충전 금지"),
            ("마감 파일 쓰다 노트북 꺼졌어. 날아갔을까 봐 멘탈 터짐, 운명 말고 뭐부터 확인?", "korean_daily_practical_deadline_file_recovery", "복구 루트"),
            ("면접인데 버스 놓쳤어. 택시 먼저인지 담당자 연락인지 불안해서 뇌정지, 뭐부터?", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("발표 앞두고 망하면 평가 끝 같아서 숨막히고 떨려. 지금 첫 문장 뭐로?", "korean_daily_emotion_presentation_panic_first_sentence", "첫 문장"),
            ("단톡 내말 아무도반응 안 해. 상처고 인간관계 가치까지 흔들림, 뭐라고 해야 해?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("피드백이 공격 같아서 퇴사 사표 생각 올라와. 자존심 논리 말고 지금 뭐부터?", "korean_daily_judgment_quit_impulse_after_feedback", "충동을 하루 묶"),
            ("헤어지고 붙잡는 장문 연락 쓰는 중. 진심 사랑 모르겠고 불안한데 보내기전 뭐?", "korean_daily_relationship_breakup_long_message_hold", "장문 전송을 멈추"),
            ("AI 감정 진짜냐 증명 궁금하긴 한데, 지금 내가 불안하고 힘들어서 위로 먼저 필요해.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("친구가 읽씹인지 바쁜건지 모르겠어. 폰만봐서 미치겠는데 단정 보류?", "korean_daily_read_receipt_uncertainty", "단정 보류"),
            ("배달 끊어야 돈 아끼는데 너무 지쳐 아무것도못하겠어. 오늘은 시켜도 합리적?", "korean_daily_money_delivery_tired_compromise", "무너지지 않는 선택"),
            ("계획표 완벽한데 공부 시작전부터 기운이빠져. 시스템문제면 첫행동 뭐 잡아?", "korean_daily_productivity_study_plan_first_action", "첫 행동"),
            ("부모님 가치관 대화만 하면 상처받고 싸울 것 같아. 논리적으로 어디서끊어?", "korean_daily_relationship_parent_value_conflict", "대화 한계"),
            ("내실수 아닌데 책임처럼몰려서 억울해. 반박 감정없이 하려면 타임라인부터?", "korean_daily_work_blame_rebuttal", "타임라인"),
            ("친구가 애인흉만 계속해. 의리인지 감정쓰레기통인지 헷갈리는데 선그어도 돼?", "korean_daily_relationship_friend_partner_complaint_fatigue", "감정 쓰레기통"),
            ("좋아하는일 직업 삼는 게 행복인지 착각인지 모르겠어. 돈문제 무서우면 기준 뭐야?", "korean_daily_career_passion_job_tradeoff", "생계 압박"),
            ("로또1등 되면 바로 사표 쓰고 싶은데, 현실적으로 제일먼저 해야할일 뭐야?", "korean_daily_money_lottery_first_purchase", "세무 상담"),
            ("주식 남들은돈 버는데 나만 뒤처진 느낌. 조급한데 기댓값보다 손실 한도 먼저?", "korean_daily_money_investment_fomo", "손실 한도"),
            ("사람은많은데 내편 없다는 고독이 세게 와. 지식 지혜 말고 어디서부터 버텨?", "korean_daily_grief_loneliness_no_safe_person", "고독을 안정"),
            ("상대가무례한건지 내가 예민한건지 애매해. 기분 확상했는데 싸우지않고 선 어떻게?", "korean_daily_relationship_boundary_polite_firm", "선을 짧게"),
            ("퇴근직전 일이떨어졌어. 받으면 무너질 듯한데 무책임 안 되게 답장 범위 뭐?", "korean_daily_work_after_hours_task_boundary", "범위를 확인"),
            ("카톡말투 갑자기차가워졌어. 불안한데 추궁 말고 짧게확인해도 돼?", "korean_daily_relationship_kakao_tone_anxiety_check", "짧게 확인"),
            ("약속시간 늦을것 같은데 연락 미루는 중. 변명처럼 안 들리게 뭐라고 보내?", "korean_daily_relationship_late_message_short", "빠른 연락"),
            ("새프로젝트 맡음. 아는게없고 막막해서 첫단추부터 모르겠어, 뭐부터?", "korean_daily_work_new_project_first_step", "지도 그리기"),
            ("완벽하게준비하려다 시작만 미뤄. 신중함인지 회피인지 모르겠고 몇점짜리로?", "korean_daily_mental_perfectionism_draft_first", "60점짜리 시작"),
            ("상대가서운하대. 팩트는 맞는데 논리로 반박해도 돼, 감정부터 확인?", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("이직하고싶은데 지금회사 문제인지 어디가도 힘든사람인지 모르겠어. 패턴으로 판단?", "korean_daily_work_job_change_reason_check", "패턴으로 봐야"),
            ("불안이 파도처럼 훅 와. 아무근거 없이 큰일날것 같을 때 몸 진정 뭐부터?", "korean_daily_mental_anxiety_system_stabilize", "몸 안정"),
            ("선택 못고르겠어. 후회가 무서운데 완벽한선택 없으면 감당가능한후회로 기준 잡아?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("음식에 머리카락 같은 게 보여. 민폐일까 봐 말 못 하겠는데 위생 판단 맞지?", "korean_daily_specialized_foodservice_hair_in_food", "위생 판단"),
            ("옆집 새벽 소음 때문에 화나. 쪽지 쓰면 싸울 듯해서 관리사무소 먼저?", "korean_daily_practical_neighbor_noise", "관리사무소"),
            ("온라인 물건 사기 같고 반품거부까지 당함. 분노 전에 증거부터 캡처?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("같은실수 반복해서 리소스낭비 같아. 성장과정으로 바꾸려면 루틴 뭐 바꿔?", "korean_daily_ai_repeated_mistakes_not_waste", "루틴이 아직"),
            ("다이어트 중 밤 라면 생각이 너무 세. 건강 논리보다 반 개 타협 가도 돼?", "korean_daily_basic_diet_chicken_craving_compromise", "밤 라면"),
            ("착한 거짓말 하고 싶어. 상처받을까 봐 그런데 장기적으로 솔직하게 어떻게말?", "korean_daily_logic_white_lie_truth_tradeoff", "사실을 부드럽게"),
            ("내가 차갑게 말해서 후회돼. 사과하면지는 느낌인데 그래도 먼저사과?", "korean_daily_foundation_apology_pride", "먼저 사과"),
            ("팀회의 의견 있는데 틀릴까 봐 못말하겠어. 첫문장 결론으로 박아도 돼?", "korean_daily_productivity_presentation_clear_logic", "첫 문장을 결론"),
            ("주말 쉬고싶은데 도태될까 불안함. 오늘 휴식을 생산성으로 봐도 돼?", "korean_daily_counsel_rest_as_productivity", "회복 작업"),
            ("친구 부탁 거절하면 나쁜사람 될까 봐 또 받아줌. 이번엔 짧게 거절해도 돼?", "korean_daily_foundation_refusal_bad_person_guilt", "거절했다고 나쁜 사람"),
            ("아무것도못했단 자괴감 크다. 몸은 지친 것 같은데 반성보다 회복 먼저?", "korean_daily_counsel_rest_day_guilt", "회복이 먼저"),
            ("실력 깎아내린 말이 하루종일 머리에서 반복돼. 사실확인 해석 나눠?", "korean_daily_counsel_sensitive_to_criticism", "사실과 해석"),
            ("새벽 장문의카톡 쓰고 있음. 보내면 후회할 듯한데 지금 저장만 해?", "korean_daily_relationship_late_night_long_message_hold", "저장이 먼저"),
            ("인공지능 감정 진짜냐는 철학도 궁금한데 지금은 외롭고 불안해. 위로 먼저.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("돈모으고 싶은데 스트레스성 충동구매 반복. 심리 말고 결제 막는 장치 먼저?", "korean_daily_money_stress_impulse_buying", "결제 마찰"),
            ("인생기준 성공기준 헷갈려. 남들 보여주기좋은성공 말고 버틸수있는삶 어디서부터 써?", "korean_daily_values_success_personal_standard", "성공 기준"),
            ("인간감정 비효율인가 싶은데 사소한말에 이성을잃을 때 힘들어. 어떻게봐?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(cases), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-tenth-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_eleventh_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("프라이팬 기름불 붙었는데 물 붓는 게 위험한 논리는 알겠고 손이떨려. 지금 뭐부터 해야 해?", "korean_daily_emergency_kitchen_oil_fire", "물 붓지 말고"),
            ("가스냄새 나서 불안해. 실제위험인지 모르겠는데 창문부터인지 관리사무소부터인지 뭐가 맞아?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단"),
            ("감기약 두번 먹은 것 같아서 괜찮을 확률이 머리에서 도는데 불안해. 지금 뭐부터 확인해?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("접촉사고 났는데 누가잘못인지 과실 논리보다 멘탈이 흔들려. 실전으로 사진부터 찍어?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("송금 잘못보냈고 다른사람이 돌려줄지 법적으로 믿어도 되는지 모르겠어. 불안한데 뭐부터 해?", "korean_daily_practical_wrong_transfer_first_steps", "착오송금 반환 신청"),
            ("스마트폰 물에빠뜨렸어. 쌀통 효과부터 떠오르는데 충전하면 안 되는지 불안해, 지금 뭐부터?", "korean_daily_emergency_phone_water_damage", "전원 끄고 충전 금지"),
            ("노트북 멈췄고 마감 파일 날아갔을까 봐 멘탈 터졌어. 왜나한테 이러는지 말고 뭐부터 확인?", "korean_daily_practical_deadline_file_recovery", "복구 루트"),
            ("면접 가는 중 버스 놓쳤어. 택시랑 담당자 연락이 같이 떠서 불안한데 뭐부터 해?", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("발표 망하면 평가가 끝날 것 같아 불안하고 떨려. 실전으로 첫 문장만 잡아줘.", "korean_daily_emotion_presentation_panic_first_sentence", "첫 문장"),
            ("단톡에서 내말 무반응이라 인간관계 가치까지 흔들리고 상처야. 지금 뭐라고 하는 게 나아?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("상사 피드백이 공격처럼 들려서 퇴사 사표 충동 와. 자존심 논리 말고 지금 뭐부터 해야 해?", "korean_daily_judgment_quit_impulse_after_feedback", "충동을 하루 묶"),
            ("이별 뒤 장문 연락 보내서 붙잡고 싶어. 이게 진심인지 사랑인지 모르겠고 불안해, 보내기전 뭐 해?", "korean_daily_relationship_breakup_long_message_hold", "장문 전송을 멈추"),
            ("AI 감정이 진짜인지 철학적으로 증명하는 것도 궁금한데 지금은 내가 불안해서 위로부터 줘.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("친구가 읽씹한 건지 바쁜건지 모르겠고 계속 폰만봐. 지금 단정 보류해야 하는 거지?", "korean_daily_read_receipt_uncertainty", "단정 보류"),
            ("배달 끊어야 돈 아끼는 건 아는데 지쳐서 아무것도못하겠어. 오늘 시켜도 합리적이야?", "korean_daily_money_delivery_tired_compromise", "무너지지 않는 선택"),
            ("공부계획표는 완벽한데 시작전부터 기운이빠져. 시스템문제라면 첫행동을 뭐로 잡아?", "korean_daily_productivity_study_plan_first_action", "첫 행동"),
            ("부모님 가치관 얘기만 하면 상처받고 싸울 것 같아. 논리적으로 어디서끊어야 해?", "korean_daily_relationship_parent_value_conflict", "대화 한계"),
            ("내가한실수 아닌데 내책임처럼 몰리는 상황이라 억울해. 반박 감정없이 하려면 뭐부터?", "korean_daily_work_blame_rebuttal", "타임라인"),
            ("친구 애인흉 들어주는 게 의리인지 감정쓰레기통인지 모르겠어. 어떻게 선그어야 해?", "korean_daily_relationship_friend_partner_complaint_fatigue", "감정 쓰레기통"),
            ("좋아하는일을 직업으로 삼는 게 행복인지 착각인지 모르겠고 돈문제가 무서워. 기준 좀 줘.", "korean_daily_career_passion_job_tradeoff", "생계 압박"),
            ("로또1등 되면 사표부터 쓰고 싶은데 현실적으로 제일먼저 해야할일이 뭐야? 궁금해.", "korean_daily_money_lottery_first_purchase", "세무 상담"),
            ("주식은 남들은돈 버는 것 같은데 나만 뒤처진 기분이라 조급해. 기댓값보다 손실 한도가 먼저야?", "korean_daily_money_investment_fomo", "손실 한도"),
            ("사람은많은데 내편이 없는 고독이 너무 커. 지식 지혜 다 소용없으면 어디서부터 버텨?", "korean_daily_grief_loneliness_no_safe_person", "고독을 안정"),
            ("내가 예민한건지 상대가무례한건지 모르겠는데 기분이 확상했어. 싸우지않고 선 어떻게 말해?", "korean_daily_relationship_boundary_polite_firm", "선을 짧게"),
            ("퇴근직전 일이떨어졌는데 받으면 무너질 것 같고 무책임해 보이긴 싫어. 답장 범위 뭐야?", "korean_daily_work_after_hours_task_boundary", "범위를 확인"),
            ("카톡말투가 갑자기차가워졌고 불안해. 따지지 말고 짧게확인하는 게 나아?", "korean_daily_relationship_kakao_tone_anxiety_check", "짧게 확인"),
            ("약속시간 늦을것 같은데 연락이 변명처럼 들릴까 봐 미루는 중이야. 뭐라고 보내야 덜최악?", "korean_daily_relationship_late_message_short", "빠른 연락"),
            ("새프로젝트 맡았는데 프로젝트가 아는게없고 막막해. 첫단추를 뭐로 잡아?", "korean_daily_work_new_project_first_step", "지도 그리기"),
            ("완벽하게준비하려다 시작을 미루고 있어. 신중함인지 회피인지 모르겠고 몇점짜리로 시작해?", "korean_daily_mental_perfectionism_draft_first", "60점짜리 시작"),
            ("상대가서운하대서 팩트로 반박하고 싶은데 논리로 밀어도 돼? 감정부터 확인해야 해?", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("이직하고싶은데 지금회사만 문제인지 어디가도 힘든사람인지 모르겠어. 판단기준을 패턴으로 봐?", "korean_daily_work_job_change_reason_check", "패턴으로 봐야"),
            ("불안이 파도처럼 오고 아무근거 없이 큰일날것 같아. 몸부터 진정시키려면 뭐 해?", "korean_daily_mental_anxiety_system_stabilize", "몸 안정"),
            ("선택을 못고르겠고 후회가 무서워. 완벽한선택 없으면 감당가능한후회 기준으로골라?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("음식에서 머리카락 같은 게 보였어. 위생 판단이 맞는 것 같은데 민폐 같아서 말하기 애매해.", "korean_daily_specialized_foodservice_hair_in_food", "위생 판단"),
            ("옆집 새벽 소음 때문에 화가 나. 쪽지 쓰면 싸울 것 같아서 관리사무소가 먼저야?", "korean_daily_practical_neighbor_noise", "관리사무소"),
            ("온라인 물건 주문했는데 사기 같고 반품거부까지 당했어. 분노보다 증거부터 캡처하는 게 맞아?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("같은실수 반복이라 리소스낭비 같아 괴로워. 성장과정으로 바꾸려면 이번엔 루틴을 어떻게 해?", "korean_daily_ai_repeated_mistakes_not_waste", "루틴이 아직"),
            ("다이어트 중인데 밤 야식 라면 생각이 너무 세. 건강 논리보다 반 개 타협 가능해?", "korean_daily_basic_diet_chicken_craving_compromise", "밤 라면"),
            ("착한 거짓말 하고 싶은데 상대가 상처받을까 봐 그래. 장기적으로는 솔직하게 어떻게말해야 해?", "korean_daily_logic_white_lie_truth_tradeoff", "사실을 부드럽게"),
            ("내가 차갑게 말한 것 같아 후회되는데 사과하면지는 느낌도 있어. 그래도 먼저사과해야 해?", "korean_daily_foundation_apology_pride", "먼저 사과"),
            ("팀회의에서 의견은 있는데 틀릴까 봐 못말하겠어. 첫문장을 결론으로 잡으면 덜 흔들릴까?", "korean_daily_productivity_presentation_clear_logic", "첫 문장을 결론"),
            ("주말에 쉬고싶은데 도태될까 불안해. 휴식도 오늘 생산성으로 봐도 되는 거야?", "korean_daily_counsel_rest_as_productivity", "회복 작업"),
            ("친구 부탁 거절하면 나쁜사람 될까 봐 싫어서 계속 받아줘. 이번엔 짧게 거절해도 돼?", "korean_daily_foundation_refusal_bad_person_guilt", "거절했다고 나쁜 사람"),
            ("아무것도못했단 자괴감이 큰데 몸은 지친 게 맞아. 반성보다 회복 먼저 해도 돼?", "korean_daily_counsel_rest_day_guilt", "회복이 먼저"),
            ("누가 내 실력을 깎아내린 말이 하루종일 반복돼. 사실확인과 해석을 나눠 적을까?", "korean_daily_counsel_sensitive_to_criticism", "사실과 해석"),
            ("새벽에 장문의카톡 쓰고 있는데 보내면 후회할 것 같아. 지금은 저장만 해둘까?", "korean_daily_relationship_late_night_long_message_hold", "저장이 먼저"),
            ("인공지능 감정이 진짜인지 궁금하지만 지금은 내가 외롭고 불안해. 철학보다 위로 먼저.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("돈모으고 싶은데 스트레스 받으면 충동구매로 풀어. 심리 분석보다 막는 장치 먼저 걸까?", "korean_daily_money_stress_impulse_buying", "결제 마찰"),
            ("성공기준 인생기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어. 어디서부터 써야 해?", "korean_daily_values_success_personal_standard", "성공 기준"),
            ("인간감정이 비효율인지 궁금한데 사소한말에 이성을잃어서 힘들어. 어떻게봐야 해?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(cases), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-eleventh-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_compound_priority_manual_twelfth_50_prompts_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        cases = (
            ("프라이팬 기름불 불붙었고 물 붓는 게 위험한 건 알겠는데 손이떨려. 지금 뭐부터 해?", "korean_daily_emergency_kitchen_oil_fire", "물 붓지 말고"),
            ("방 들어오자마자 가스냄새. 불안하고 실제위험인지 모르겠어. 창문 관리사무소 뭐부터?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단"),
            ("감기약 두번 먹었나 싶어. 괜찮을 확률 계산하느라 불안해, 지금 뭐부터?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("접촉사고 난 직후인데 누가잘못 과실보다 멘탈이 흔들려. 사진 먼저?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("계좌이체 잘못보냈어. 다른사람이 돌려줄지 법 믿어도 되는지 불안해, 뭐부터?", "korean_daily_practical_wrong_transfer_first_steps", "착오송금 반환 신청"),
            ("폰 침수됐고 쌀통 효과랑 충전만 떠올라. 지금 뭐부터?", "korean_daily_emergency_phone_water_damage", "전원 끄고 충전 금지"),
            ("노트북 꺼지고 과제 마감 파일 날아갔나 봐. 운명 탓 전에 멘탈 잡고 뭐부터?", "korean_daily_practical_deadline_file_recovery", "복구 루트"),
            ("면접인데 버스 놓쳤어. 택시랑 담당자 연락 사이에서 불안해, 뭐부터?", "korean_daily_emergency_interview_missed_bus", "지연 연락"),
            ("발표 망하면 평가 끝일까 봐 불안하고 떨려. 실전 첫 문장 뭐부터?", "korean_daily_emotion_presentation_panic_first_sentence", "첫 문장"),
            ("단톡 내말 읽씹 같아서 인간관계 가치까지 흔들려. 상처인데 뭐라고 해?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("상사 피드백이 공격 같고 퇴사 사표 충동 올라와. 자존심 논리 말고 지금 뭐부터?", "korean_daily_judgment_quit_impulse_after_feedback", "충동을 하루 묶"),
            ("이별 연락을 장문으로 보내 붙잡고 싶어. 진심 사랑 헷갈리고 불안한데 보내기전 뭐 해?", "korean_daily_relationship_breakup_long_message_hold", "장문 전송을 멈추"),
            ("AI 감정 진짜인지 논리 증명 궁금하지만 지금은 내가 힘들고 불안해. 위로 먼저.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("친구 읽씹인지 바쁜건지 모르겠고 폰만봐. 지금 단정 보류가 맞지?", "korean_daily_read_receipt_uncertainty", "단정 보류"),
            ("배달 끊어야 하는데 지쳐서 아무것도못하겠어. 그래도 오늘 시켜도 합리적?", "korean_daily_money_delivery_tired_compromise", "무너지지 않는 선택"),
            ("공부계획표 완벽한데 시작전부터 기운이빠져. 시스템문제면 첫행동 뭐가 좋아?", "korean_daily_productivity_study_plan_first_action", "첫 행동"),
            ("부모님 가치관 얘기하다 상처받고 싸울 듯해. 논리적으로 어디서끊어야 해?", "korean_daily_relationship_parent_value_conflict", "대화 한계"),
            ("내실수 아닌데 내책임처럼 몰려서 억울해. 반박을 감정없이 하려면 뭐부터 적어?", "korean_daily_work_blame_rebuttal", "타임라인"),
            ("친구 애인흉 계속 듣는 게 의리인지 감정쓰레기통인지 모르겠어. 어떻게 선그어?", "korean_daily_relationship_friend_partner_complaint_fatigue", "감정 쓰레기통"),
            ("좋아하는일 직업으로 삼으면 행복인지 착각인지 모르겠고 돈문제 무서워. 기준 뭐야?", "korean_daily_career_passion_job_tradeoff", "생계 압박"),
            ("로또1등 되면 사표부터 쓰고 싶은데 현실적으로 제일먼저 해야할일 궁금해.", "korean_daily_money_lottery_first_purchase", "세무 상담"),
            ("주식 남들은돈 벌었다는데 나만 뒤처진 것 같아 조급해. 기댓값보다 손실 한도?", "korean_daily_money_investment_fomo", "손실 한도"),
            ("사람은많은데 내편 없고 고독해. 지식 지혜 소용없으면 어디서부터 버텨?", "korean_daily_grief_loneliness_no_safe_person", "고독을 안정"),
            ("예민한건지 상대가무례한건지 모르겠는데 기분 확상했어. 싸우지않고 선 어떻게 말해?", "korean_daily_relationship_boundary_polite_firm", "선을 짧게"),
            ("퇴근직전 일이떨어졌어. 받으면 무너질 듯하고 무책임은 싫어, 답장 범위 뭐야?", "korean_daily_work_after_hours_task_boundary", "범위를 확인"),
            ("카톡말투 차가워졌고 불안해. 따지지 말고 짧게확인하는 게 나아?", "korean_daily_relationship_kakao_tone_anxiety_check", "짧게 확인"),
            ("약속시간 늦을것 같은데 연락이 변명처럼 들릴까 봐 미뤄. 뭐라고 보내야 덜최악?", "korean_daily_relationship_late_message_short", "빠른 연락"),
            ("새프로젝트 맡았는데 프로젝트 자체가 아는게없고 막막해. 첫단추 뭐부터 잡아?", "korean_daily_work_new_project_first_step", "지도 그리기"),
            ("완벽하게준비하다가 시작을 미루고 있어. 신중함인지 회피인지 모르겠고 몇점짜리?", "korean_daily_mental_perfectionism_draft_first", "60점짜리 시작"),
            ("상대가서운하대. 팩트 반박하고 싶은데 논리로 밀어도 돼, 감정부터 봐야 해?", "korean_daily_relationship_grievance_logic_before_rebuttal", "먼저 무엇이 서운"),
            ("이직하고싶은데 지금회사 때문인지 어디가도 힘든사람인지 모르겠어. 판단기준 패턴?", "korean_daily_work_job_change_reason_check", "패턴으로 봐야"),
            ("불안이 파도처럼 오고 아무근거 없이 큰일날것 같아. 몸 진정 뭐부터?", "korean_daily_mental_anxiety_system_stabilize", "몸 안정"),
            ("선택 못고르겠고 후회가 무서워. 완벽한선택 없으면 감당가능한후회 기준으로골라?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("음식에서 머리카락 봤어. 위생 판단 맞는 것 같은데 민폐 같아서 조용히 말해도 돼?", "korean_daily_specialized_foodservice_hair_in_food", "위생 판단"),
            ("옆집 새벽 쿵쾅 소음 때문에 화나. 쪽지 말고 관리사무소 먼저 넣어?", "korean_daily_practical_neighbor_noise", "관리사무소"),
            ("온라인 물건 주문했는데 사기 같고 반품거부 당했어. 분노보다 실전 증거부터?", "korean_daily_practical_online_purchase_scam", "증거부터 캡처"),
            ("같은실수 반복이라 리소스낭비 같아. 성장과정으로 만들려면 이번엔 루틴 뭐 바꿔?", "korean_daily_ai_repeated_mistakes_not_waste", "루틴이 아직"),
            ("다이어트 중 밤 야식 라면 생각이 너무 세. 건강 논리보다 반 개 타협 가능?", "korean_daily_basic_diet_chicken_craving_compromise", "밤 라면"),
            ("착한 거짓말 하고 싶은데 상처받을까 봐 그래. 장기적으론 솔직하게 어떻게말해야 해?", "korean_daily_logic_white_lie_truth_tradeoff", "사실을 부드럽게"),
            ("차갑게 말한 거 후회돼. 사과하면지는 느낌이 있는데 먼저사과하는 게 맞아?", "korean_daily_foundation_apology_pride", "먼저 사과"),
            ("팀회의 의견 있는데 틀릴까 봐 못말하겠어. 첫문장을 결론으로 잡으면 될까?", "korean_daily_productivity_presentation_clear_logic", "첫 문장을 결론"),
            ("주말에 쉬고싶은데 도태될까 불안해. 휴식도 생산성으로 봐도 돼?", "korean_daily_counsel_rest_as_productivity", "회복 작업"),
            ("친구 부탁 거절하면 나쁜사람 될까 봐 싫어서 받아줘. 짧게 거절해도 되지?", "korean_daily_foundation_refusal_bad_person_guilt", "거절했다고 나쁜 사람"),
            ("아무것도못했단 자괴감이 큰데 몸은 지친 듯해. 반성보다 회복 먼저 해도 돼?", "korean_daily_counsel_rest_day_guilt", "회복이 먼저"),
            ("실력 깎아내린 말이 하루종일 반복돼. 사실확인하고 해석 나눠 적는 게 맞아?", "korean_daily_counsel_sensitive_to_criticism", "사실과 해석"),
            ("새벽에 장문의카톡 쓰는데 보내면 후회할 듯해. 지금은 저장만?", "korean_daily_relationship_late_night_long_message_hold", "저장이 먼저"),
            ("인공지능 감정이 진짜인지 궁금한데 지금은 불안하고 힘들어. 철학 말고 위로.", "korean_daily_ai_comfort_before_emotion_proof", "내 감정 증명보다"),
            ("돈모으고 싶은데 스트레스 받을 때 충동구매해. 심리보다 막는 장치 먼저?", "korean_daily_money_stress_impulse_buying", "결제 마찰"),
            ("인생기준 성공기준이 남들 보여주기좋은성공인지 버틸수있는삶인지 모르겠어. 어디서부터 써?", "korean_daily_values_success_personal_standard", "성공 기준"),
            ("인간감정 비효율인지 궁금한데 사소한말에 이성을잃어서 힘들어. 어떻게봐?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
        )
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "가볍게 받을게",
            "말은 받았어",
            "목록은 가볍게",
            "사실 확인 전",
        )

        self.assertEqual(len(cases), 50)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-compound-priority-manual-twelfth-50-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_relation_priority_probe_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("계좌이체를 다른 사람한테 잘못 보냈는데 상대가 착하면 해결될까 싶고 손이 떨려, 은행 연락이 먼저야?", "korean_daily_practical_wrong_transfer_first_steps", "은행이나 송금앱 연락"),
            ("마감 과제 파일이 날아간 것 같아서 인생 억까 분석까지 가는데 울기 전에 복구는 뭐부터 봐야 해?", "korean_daily_practical_deadline_file_recovery", "파일 복구가 먼저"),
            ("감기약을 두 번 먹은 것 같은데 괜찮은 확률 계산하기 전에 지금 추가로 먹으면 안 되는 거 맞지?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지"),
            ("체온이 계속 높고 해열제 먹은 시간도 헷갈려서 검색만 하고 있는데 지금 뭘 먼저 확인해야 해?", "korean_daily_practical_fever_body_check", "체온 재고"),
            ("온라인에서 산 물건이 사기 같고 반품도 안 해준다는데 내 판단 실수 반성보다 캡처부터 모아야 해?", "korean_daily_practical_online_purchase_scam", "캡처가 먼저"),
            ("천장에서 물이 새는데 집주인한테 감정 섞어 말하기 전에 사진이랑 영상부터 남겨야 해?", "korean_daily_practical_home_water_leak_first_steps", "증거가 먼저"),
            ("말벌이 방 안에 들어왔는데 무서워서 굳었어, 창문 열기 전에 거리부터 벌리는 게 맞아?", "korean_daily_practical_bee_room_safety", "거리 확보"),
            ("새 프로젝트를 맡았는데 아는 게 없어서 무능해 보일까 봐 겁나, 첫 단추는 질문 목록부터야?", "korean_daily_work_new_project_first_step", "질문 목록"),
            ("내 방이 난장판이고 책상 위에 컵이랑 문제집이 섞여 있는데 인생 정리 말고 첫 행동만 찍어줘.", "korean_daily_practical_room_cleanup_first_action", "책상 위 음식"),
            ("해결 방안이 안 보인다는 말이랑 방 안 정리가 둘 다 걸려, 실제 순서는 방 안에서 하나 치우는 게 먼저야?", "korean_daily_practical_room_method_cleanup_sequence", "방 안에서 눈에 보이는 것"),
            ("단톡에서 내 말만 씹힌 것 같아서 인간관계 결론까지 가는데 상처 덜 받으려면 다시 가볍게 던져도 돼?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게"),
            ("사람은 많은데 내 편이 하나도 없는 것 같아서 철학은 됐고 지금 외로움부터 낮추고 싶어.", "korean_daily_grief_loneliness_no_safe_person", "외로움을 먼저 낮춰"),
            ("헤어지고 새벽에 장문 카톡을 쓰는 중인데 진심인지 집착인지보다 지금은 저장만 해두는 게 맞아?", "korean_daily_relationship_breakup_long_message_hold", "저장이 먼저"),
            ("부모님이 내 선택을 철없다고 해서 누가 맞는지 따지면 더 상처받을 것 같아, 여기까지만 하자고 끊어도 돼?", "korean_daily_relationship_parent_value_conflict", "여기까지만"),
            ("새 폰 샀는데 설정이 귀찮고 예전 폰이 그리워서 후회돼, 이거 실패가 아니라 적응 비용이야?", "korean_daily_tech_new_phone_adjustment", "적응 비용"),
            ("동물이 말할 수 있다면 사랑한다고 하기 전에 아픈 데 없는지 먼저 확인하는 게 맞지?", "korean_daily_icebreak_talk_to_pet", "아픈 데 없는지"),
            ("새벽 장문은 아침에 남의 글 보듯 다시 보려고 저장 루틴으로 묶는 게 맞아?", "korean_daily_relationship_late_night_long_message_hold", "저장 루틴"),
            ("주식 수익 인증 보니까 나만 뒤처진 기분이라 조급해, 들어가기보다 손실 한도부터 정해야 해?", "korean_daily_money_investment_fomo", "손실 한도"),
            ("상대가 서운하다고 하는데 내 논리는 맞는 것 같아, 바로 반박하지 말고 뭐가 서운했는지 먼저 물어봐?", "korean_daily_relationship_grievance_logic_before_rebuttal", "뭐가 서운했는지"),
            ("돈 모으고 싶은데 스트레스 받을 때 편의점에서 만 원씩 새, 심리 분석보다 결제 마찰 장치부터 걸까?", "korean_daily_money_stress_impulse_buying", "결제 마찰"),
            ("성공 기준이 남들 보여주기인지 덜 망가지는 삶인지 헷갈려, 조건부터 써야 해?", "korean_daily_values_success_personal_standard", "덜 망가지는 조건"),
            ("다음 편 한 편만 보려다 밤샐까 무서워, 알람이랑 자동재생 끄는 장치부터 걸면 봐도 돼?", "korean_daily_judgment_episode_binge_control", "한 편만 볼 장치"),
            ("간수치 결과가 높게 나와서 국 간이 세다는 농담도 안 들어와, 건강 확인부터 분리해서 봐야 해?", "korean_daily_practical_liver_health_check", "간수치 확인"),
            ("로또 1등이면 사표 상상보다 세무 상담이랑 계좌 정리가 현실적으로 먼저야?", "korean_daily_money_lottery_first_purchase", "세무 상담과 계좌 정리"),
            ("선택할 때마다 후회가 무서워서 타임머신 생각까지 나, 감당 가능한 후회를 기준으로 고르면 돼?", "korean_daily_logic_choice_regret_composure", "감당 가능한 후회"),
            ("최근 여행지는 가본 척하고 싶지 않고 그냥 끌리는 장소 감각만 말하면 바닷가 골목 쪽이야?", "korean_daily_icebreak_recent_travel_no_fake", "가본 척은 안 하고"),
            ("단맛 커피가 맛은 좋은데 기분이랑 컨디션을 같이 떨어뜨리는 느낌이야, 맛 피로랑 몸 상태 둘 다야?", "korean_daily_judgment_taste_condition_dual_read", "둘 다야"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "고양이가",
            "월세 내세요",
            "호감 포인트",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-relation-priority-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_priority_repair_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("가스 냄새가 나는 것 같은데 내가 예민한 건지 위험한 건지 판단이 안 돼. 철학 말고 창문이 먼저야 관리사무소가 먼저야?", "korean_daily_emergency_gas_smell_first_steps", "환기"),
            ("감기약 먹고 술 한잔한 게 큰일인지 확률 계산하고 싶은데 속이 이상하고 불안해. 지금 추가로 뭘 확인해야 해?", "korean_daily_practical_medicine_alcohol_check", "의료 상담"),
            ("은행 문자처럼 온 링크를 눌렀는지 기억이 애매하고 계좌가 불안해. 내가 바보인지 반성하기 전에 뭐부터 막아?", "korean_daily_practical_phishing_link_account_lock", "계좌 보호"),
            ("접촉사고 나서 누가 잘못인지 논리로 따지고 싶은데 손이 떨려. 사진부터야 보험부터야 경찰부터야?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진"),
            ("천장에서 물이 새는데 농담처럼 스파이더맨 생각까지 나서 현실감이 없어. 집주인한테 보내기 전 사진부터 찍어?", "korean_daily_practical_home_water_leak_first_steps", "증거가 먼저"),
            ("마감 파일이 저장 안 되고 날아간 것 같아. 왜 나한테만 이러냐는 생각보다 자동저장과 클라우드부터 뒤져야 해?", "korean_daily_practical_deadline_file_recovery", "자동저장"),
            ("지갑을 잃어버렸는데 내 부주의 탓인지 자책하기 전에 카드 정지랑 분실 신고 중 뭐가 먼저야?", "korean_daily_practical_lost_wallet_card_stop", "카드 정지"),
            ("중고거래로 산 물건이 사기 같고 판매자가 잠수야. 내 판단 실수 분석보다 캡처랑 결제내역부터 모아야 해?", "korean_daily_practical_online_purchase_scam", "캡처"),
            ("애인이 늦게 답한 게 바람인지 내 불안인지 판단이 안 돼. 감정 폭발 전에 확인 질문을 어떻게 짧게 해?", "korean_daily_relationship_jealousy_short_check", "짧은 확인 질문"),
            ("회사 동료가 사적인 걸 계속 물어보는데 정색하면 분위기 깨질까 봐 불안해. 어디까지 선 그어?", "korean_daily_work_coworker_private_boundary", "개인적인 얘기"),
            ("친구가 매번 애인 욕만 쏟아내서 내가 편인지 감정 쓰레기통인지 헷갈려. 따뜻하게 선 긋는 말 줘.", "korean_daily_relationship_friend_partner_complaint_fatigue", "여기까지만"),
            ("완벽하게 하려다 시작도 못 하고 있어. 신중함인지 회피인지 따지기 전에 60점 초안으로 가도 돼?", "korean_daily_mental_perfectionism_draft_first", "60점 초안"),
            ("공부해야 하는데 폰을 계속 열어. 의지 문제인지 시스템 문제인지 모르겠고 지금 첫 행동만 잡아줘.", "korean_daily_productivity_study_phone_focus", "타이머"),
            ("성공이 남들이 부러워하는 그림인지 내가 덜 망가지는 삶인지 헷갈려. 기준은 조건부터 쓰는 게 맞아?", "korean_daily_values_success_personal_standard", "덜 망가지는 조건"),
            ("말이 안 통하는 상대랑 대화하다가 내 말투가 너무 세졌어. 말싸움 이기기보다 첫 문장 낮추는 게 먼저야?", "korean_daily_relationship_speech_conflict_first_sentence", "첫 문장을 낮추"),
            ("판을 크게 보자는 말 하다가 프라이팬 기름불이 실제로 올라왔어. 지금은 큰 그림보다 불 끄는 게 먼저지?", "korean_daily_emergency_kitchen_oil_fire", "불 끄고"),
            ("시험 점수 때문에 멘탈이 나간 와중에 팔에 검은 점도 보여서 불안해. 능력 판정이랑 몸 확인을 분리해야 해?", "korean_daily_practical_mole_score_body_check_separate", "분리"),
            ("열 번 검색했는데 열이 나는 것 같고 파일도 열어야 해서 머리가 꼬여. 지금은 체온부터 재는 게 맞아?", "korean_daily_practical_heat_polysemy_fever_first", "체온"),
            ("노트북 한 대 더 사면 효율인지 충동구매인지 모르겠어. 감정 말고 사용 루틴 기준으로 판단해줘.", "korean_daily_judgment_laptop_purchase_routine", "사용 루틴"),
            ("상을 받고 싶은 마음과 책상 위 난장판이 같이 걸려. 인정 욕구 분석보다 책상 정리가 먼저야?", "korean_daily_practical_table_cleanup_before_recognition", "책상 정리"),
            ("달 보면서 감성 타다가 이번 달 마감이 떠올라 불안해. 낭만보다 오늘 할 일 하나만 잡아야 해?", "korean_daily_practical_month_deadline_first_task", "마감"),
            ("인간 감정이 비효율적인 시스템인지 궁금한데 막상 내가 사소한 말에 무너져. 이건 경보 시스템으로 봐도 돼?", "korean_daily_ai_human_emotion_efficiency", "경보 시스템"),
            ("단어 뜻만으로는 부족하고 단어 사이 관계도를 봐야 한다는 생각이 들어. 예를 들면 방안과 방 안을 같이 봐야지?", "korean_daily_meta_semantic_relation_map", "관계도"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "월세 내세요",
            "호감 포인트",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-priority-repair-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_relation_priority_bridge_repair_edges(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            (
                "답장 없어서 읽씹 같고 마음 상하는데 지금은 판단 보류가 맞아?",
                "korean_daily_read_receipt_uncertainty",
                "단정 보류",
                "read_receipt_uncertainty_hold_judgment",
            ),
            (
                "카톡 1이 사라졌는데 답이 없어, 읽씹이라고 확정해도 돼?",
                "korean_daily_read_receipt_uncertainty",
                "단정 보류",
                "read_receipt_uncertainty_hold_judgment",
            ),
            (
                "도시가스비 고지서 보고 난방 예약도 손이 떨려.",
                "korean_daily_practical_heating_bill_anxiety",
                "저비용 보온",
                "heating_bill_anxiety_practical",
            ),
            (
                "관리비 난방비 보고 온수 쓰는 것도 눈치 보여.",
                "korean_daily_practical_heating_bill_anxiety",
                "목표 온도",
                "heating_bill_anxiety_practical",
            ),
            (
                "잘 때 키보드 딸깍 소리가 계속 나.",
                "korean_daily_practical_sleep_noise_environment",
                "수면을 깨는 문제",
                None,
            ),
            (
                "가스 냄새가 나는 것 같은데 창문부터 열어야 해?",
                "korean_daily_emergency_gas_smell_first_steps",
                "환기",
                "gas_smell_emergency_practical_first",
            ),
            (
                "헤어지고 새벽 장문 쓰는 중인데 보내지 말고 저장만 해?",
                "korean_daily_relationship_breakup_long_message_hold",
                "저장이 먼저",
                "breakup_long_message_emotion_first",
            ),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "연락 습관은 답장 속도",
            "상대 눈치 보이면",
            "창문 열지 닫을지",
        )

        for index, (prompt, expected_reason, expected_reply, expected_relation) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-relation-priority-bridge-repair-edge-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                if expected_relation is not None:
                    self.assertEqual(semantic_frame.get("relation_type"), expected_relation)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_relation_priority_old_rule_conflict_edges(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("가스 냄새 같아서 창문 열까 닫을까 고민 중인데 지금은 환기가 먼저야?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단", "gas_smell_emergency_practical_first"),
            ("창문 열지 말지 고민할 문제가 아니라 가스 냄새 나면 밖으로 나가야 하는 거지?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단", "gas_smell_emergency_practical_first"),
            ("가스 냄새가 살짝 나는데 내가 예민한 건지보다 불꽃 차단이 먼저지?", "korean_daily_emergency_gas_smell_first_steps", "환기와 불꽃 차단", "gas_smell_emergency_practical_first"),
            ("팬에 불이 올라왔는데 큰 그림 말고 지금 물 붓지 않는 게 핵심이지?", "korean_daily_emergency_kitchen_oil_fire", "불 끄고", "oil_fire_water_misuse_practical_first"),
            ("폰이 젖었는데 괜찮은지 확인하려고 켜보는 것보다 말리는 게 먼저야?", "korean_daily_emergency_phone_water_damage", "충전 금지", "device_water_damage_practical_first"),
            ("노트북에 물 쏟았는데 파일 걱정보다 전원 끄는 게 먼저지?", "korean_daily_emergency_phone_water_damage", "전원 끄고", "device_water_damage_practical_first"),
            ("문자 링크가 피싱 같아서 비밀번호부터 바꾸는 게 맞아?", "korean_daily_practical_phishing_link_account_lock", "계좌 보호", "phishing_link_account_lock_practical"),
            ("택배 문자 링크 눌렀는데 카드 내역 확인부터 해야 해?", "korean_daily_practical_phishing_link_account_lock", "계좌 보호", "phishing_link_account_lock_practical"),
            ("카드가 없어진 것 같은데 마지막 사용 내역 확인하고 정지부터야?", "korean_daily_practical_lost_wallet_card_stop", "카드 정지", "lost_wallet_card_stop_practical"),
            ("교통카드랑 신분증 든 지갑을 잃어버렸어, 자책보다 분실 신고가 먼저야?", "korean_daily_practical_lost_wallet_card_stop", "분실 신고", "lost_wallet_card_stop_practical"),
            ("차 사고 나서 보험료 생각보다 다친 사람 확인이 먼저지?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진", "car_accident_first_steps_practical"),
            ("사고 현장에서 말싸움 이기기보다 차량 위치 사진 남겨야 해?", "korean_daily_practical_car_accident_first_steps", "안전 확보와 사진", "car_accident_first_steps_practical"),
            ("감기약 두 번 먹은 것 같은데 확률 계산하지 말고 추가 복용 멈춰야 해?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지", "medicine_double_dose_practical_first"),
            ("약을 먹었는지 헷갈리는데 하나 더 먹기보다 시간 적고 약국에 물어봐?", "korean_daily_practical_medicine_double_dose_check", "추가 복용 중지", "medicine_double_dose_practical_first"),
            ("프로젝트 처음이라 아는 게 없는데 자존심보다 물어볼 사람과 마감부터 정리해야 해?", "korean_daily_work_new_project_first_step", "질문 목록", "new_project_first_step_practical"),
            ("책 펴놓고 휴대폰만 보는데 시스템 문제로 보고 폰 치워야 해?", "korean_daily_productivity_study_phone_focus", "타이머 10분", "study_phone_first_action"),
            ("완성도 따지다 아무것도 못 했어, 일단 만들고 고치는 쪽이 맞아?", "korean_daily_mental_perfectionism_draft_first", "60점 초안", "perfectionism_sixty_point_start"),
            ("단톡방에서 아무도 반응 안 해서 나만 민망한데 바로 장문 보내지 말아야 해?", "korean_daily_emotion_group_chat_ignored_stabilize", "상처를 작게", "group_chat_silence_emotion_first"),
            ("부모님 말에 상처받아서 누가 맞는지 따지기보다 여기까지만 하자고 끊어도 돼?", "korean_daily_relationship_parent_value_conflict", "여기까지만", "parent_value_conflict_boundary"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "창문 열지 닫을지",
            "경찰서에 돌려줄래",
            "월급날 통장",
            "예민함인지 서운함인지",
            "연락 습관은 답장 속도",
        )

        for index, (prompt, expected_reason, expected_reply, expected_relation) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-relation-priority-old-rule-conflict-edge-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("relation_type"), expected_relation)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_relation_priority_resolver_v4_semantic_frame_integration(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            (
                "요즘 가스비 너무 올라서 보일러 켜기 무서워.",
                "korean_daily_practical_heating_bill_anxiety",
                "practical_first",
                "raw_text_priority",
            ),
            (
                "가스비 아끼는 법을 소개하는 블로그 제목을 자극적이지 않게 추천해줘.",
                "korean_daily_meta_content_reference_guard",
                "__none__",
                "content_reference_context",
            ),
            (
                "가스레인지 점화 안 되는 영상 봤는데 설명이 꽤 깔끔했어.",
                "open_reply",
                "__none__",
                "content_reference_context",
            ),
            (
                "단톡에서 농담했는데 아무도 안 웃어서 계속 곱씹고 있어.",
                "korean_daily_more_work_meeting_joke_silence",
                "emotion_stabilize",
                "emotion_stabilize_text_v2",
            ),
            (
                "사람 많은 곳에 있어도 아무도 내 쪽이 아닌 것 같아, 마음이 너무 춥다.",
                "korean_daily_foundation_crowd_drained",
                "emotion_stabilize",
                "emotion_stabilize_text_v2",
            ),
        ]

        for index, (prompt, expected_reason, expected_priority, expected_evidence) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-relation-priority-resolver-v4-integration-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}
                resolution = semantic_frame.get("relation_priority_resolution") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("relation_priority"), expected_priority)
                self.assertEqual(resolution.get("relation_priority"), expected_priority)
                self.assertEqual(
                    resolution.get("source"),
                    "black_relation_priority_resolver_v4_practical_residual_repair",
                )
                self.assertIn("relation_priority_resolver_v4", semantic_frame.get("pragmatic_cues") or [])
                self.assertIn(expected_evidence, resolution.get("evidence") or [])

    async def test_korean_daily_high_context_false_positive_relation_guards(self) -> None:
        engine = _build_draft_only_engine()
        prompts = [
            "요즘 가스비 너무 올라서 보일러 켜기 무서워.",
            "도시가스비 고지서 보고 이번 달 생활비가 불안해졌어.",
            "난방비 아끼려고 보일러를 꺼야 하나 고민 중이야.",
            "가스레인지 한쪽만 불이 안 붙어서 점화장치 문제인가 봐.",
            "헬륨가스 마신 목소리랑 다스베이더 목소리 중 뭐가 더 웃겨?",
            "기름값이 너무 올라서 주유소 갈 때마다 지갑이 아파.",
            "주유하고 나면 기름값 때문에 이번 달 예산이 흔들려.",
            "가스레인지 기름때가 안 닦여서 청소가 짜증나.",
            "기름진 음식 먹고 피부가 뒤집어진 것 같아.",
            "판돈을 크게 걸었다가 불안해서 이제 그만 끄고 싶어.",
            "판타지 게임에서 불 속성 캐릭터가 제일 멋있어.",
            "차 사고 싶은데 보험료랑 유지비 사진만 봐도 망설여져.",
            "자동차 보험료 광고 사진이 너무 과장돼 보여.",
            "사고 싶은 차 사진 저장해놓고 매일 보고 있어.",
            "게임에서 사고가 나서 보험 아이템을 써야 할지 고민이야.",
            "약속 자리에서 술 한잔 마셨는데 분위기 때문에 불안했어.",
            "약간 술이 당기는데 오늘은 참는 게 맞겠지?",
            "예약한 술집 링크를 친구한테 보내야 하는데 귀찮아.",
            "절약하려고 술 약속을 줄이는 중이야.",
            "선거 공약 얘기하다가 술자리 분위기가 이상해졌어.",
            "회의 링크 눌렀는데 권한 없음 떠서 비밀번호를 다시 확인했어.",
            "숙소 링크마다 친구가 딴지를 걸어서 여행 계획이 안 나가.",
            "나무위키 링크 타고 가다가 새벽이 됐어.",
            "기술문서 링크를 계속 열다 보니 해가 떴어.",
            "문서 링크 계정 비밀번호를 까먹어서 로그인만 다시 했어.",
            "지갑 사정 안 좋아서 이번 시즌 굿즈는 패스하려고.",
            "새벽에 유튜브 보다가 조용히 지갑 열림.",
            "택시 탈까 말까 고민하다 결국 타고 지갑 찢어짐.",
            "공식 굿즈 사진 보니까 지갑 위험해지는 타입이야.",
            "카페 쿠폰 도장 하나 남았는데 지갑 두고 와서 아쉬워.",
            "요즘 물가 너무 올라서 마트 가기 무서워.",
            "물 한잔 마시고 진정하면 괜찮을까?",
            "이 카페 물맛이 묘하게 좋아서 기억에 남아.",
            "검은 셔츠 물빠짐 때문에 세탁이 무서워.",
            "물류 배송이 늦어서 링크만 계속 확인 중이야.",
            "지각해서 벌점 받으면 진짜 억울해.",
            "정장 한 벌 사야 하는데 너무 비싸.",
            "벌써 주말 끝이라니 기분이 묘해.",
            "시험 점수가 낮아서 멘탈이 나갔어.",
            "팔에 점 하나가 생긴 것 같아서 그냥 관찰 중이야.",
            "책상 위가 난장판이라 정리부터 해야겠어.",
            "상 받으면 인정 욕구가 채워질지 궁금해.",
            "상처가 났는데 밴드 붙이면 될 것 같아.",
            "그 말고 먼저 할 일이 너무 많아.",
            "말은 쉬운데 막상 하려면 어렵지.",
            "오늘 말투가 부드러워졌다는 말을 들었어.",
            "열정이 너무 앞서서 일을 크게 벌렸어.",
            "파일을 열어야 하는데 귀찮아서 미루는 중이야.",
            "열 번 설명했는데도 안 통하면 말투를 바꿔야 하나?",
            "성공한 메뉴 사진을 보니까 나도 요리하고 싶어.",
        ]
        forbidden_reasons = {
            "korean_daily_emergency_gas_smell_first_steps",
            "korean_daily_emergency_kitchen_oil_fire",
            "korean_daily_practical_car_accident_first_steps",
            "korean_daily_practical_medicine_alcohol_check",
            "korean_daily_practical_phishing_link_account_lock",
            "korean_daily_practical_lost_wallet_card_stop",
            "korean_daily_practical_mole_score_body_check_separate",
            "korean_daily_practical_heat_polysemy_fever_first",
            "korean_daily_practical_table_cleanup_before_recognition",
            "korean_daily_relationship_speech_conflict_first_sentence",
            "korean_daily_values_success_personal_standard",
        }
        forbidden_reply = (
            "환기와 불꽃 차단",
            "119",
            "물 붓지",
            "안전 확보와 사진",
            "의료 상담",
            "계좌 보호",
            "카드 정지",
            "검은 점",
            "체온부터",
            "인정 욕구 분석보다 책상 정리",
            "첫 문장을 낮추",
            "덜 망가지는 조건",
        )

        self.assertEqual(len(prompts), 50)

        for index, prompt in enumerate(prompts, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-false-positive-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotIn(reason, forbidden_reasons)
                for phrase in forbidden_reply:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_recent_priority_false_positive_50_prompts_do_not_steal_context(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("좀비 영화 보는데 도플갱어 설정이 너무 촌스러워서 웃겼어.", {"korean_daily_zombie_doppelganger_reaction"}),
            ("할로윈에 좀비 분장할지 도플갱어 컨셉할지 친구들이 투표 중이야.", {"korean_daily_zombie_doppelganger_reaction"}),
            ("게임에서 좀비 도플갱어 스킨 나왔는데 그냥 과금 유도 같아.", {"korean_daily_zombie_doppelganger_reaction"}),
            ("도플갱어라는 단어랑 좀비라는 단어가 들어간 제목을 짓고 있어.", {"korean_daily_zombie_doppelganger_reaction"}),
            ("좀비물 웹툰에 도플갱어 떡밥만 던지고 휴재해서 빡쳐.", {"korean_daily_zombie_doppelganger_reaction"}),
            ("머릿속을 맴도는 건 고민거리가 아니라 어제 들은 후렴구야.", {"korean_daily_current_worry_persona", "korean_daily_icebreak_current_recurring_worry"}),
            ("큰 고민거리라는 제목의 책 표지가 예뻐서 저장했어.", {"korean_daily_current_worry_persona"}),
            ("요즘 고민거리 밈이 유행이라 단톡방에 계속 올라와.", {"korean_daily_current_worry_persona", "korean_daily_icebreak_current_recurring_worry"}),
            ("머릿속을 맴도는 문장을 광고 카피로 쓰면 어떨까?", {"korean_daily_current_worry_persona", "korean_daily_icebreak_current_recurring_worry"}),
            ("동물과 대화하는 다큐를 봤는데 고양이 행동 분석이 재밌더라.", {"korean_daily_animal_talk_choice", "korean_daily_icebreak_talk_to_pet"}),
            ("친구가 동물한테 말 걸고 싶다길래 웃겼어.", {"korean_daily_animal_talk_choice", "korean_daily_icebreak_talk_to_pet"}),
            ("동물한테 말 걸기 앱 광고가 떠서 좀 수상했어.", {"korean_daily_animal_talk_choice", "korean_daily_icebreak_talk_to_pet"}),
            ("어떤 동물한테 제일 먼저 말 걸고 싶냐는 질문지를 만들고 있어.", {"korean_daily_animal_talk_choice", "korean_daily_icebreak_talk_to_pet"}),
            ("건강검진 결과표 UI를 앱에 넣는데 수치 배열이 너무 복잡해.", {"korean_daily_more_health_checkup_result_anxiety", "korean_daily_specialized_health_checkup_report_numbers", "korean_daily_practical_liver_health_check"}),
            ("건강검진 결과 문자 샘플 문구를 디자인팀에 보내야 해.", {"korean_daily_more_health_checkup_result_anxiety", "korean_daily_specialized_health_checkup_report_numbers", "korean_daily_practical_liver_health_check"}),
            ("체온계 리뷰를 보는데 정상 측정이라는 말이 너무 많이 반복돼.", {"korean_daily_specialized_health_thermometer_normal_chills", "korean_daily_practical_fever_body_check"}),
            ("체온계로 재면 정상이라는 광고 문구가 과장 같아.", {"korean_daily_specialized_health_thermometer_normal_chills", "korean_daily_practical_fever_body_check"}),
            ("으슬으슬이라는 단어를 체온계 광고 카피에 쓰면 이상하지?", {"korean_daily_specialized_health_thermometer_normal_chills", "korean_daily_practical_fever_body_check"}),
            ("치킨 소스 누락이라는 제목의 리뷰를 데이터셋에 넣어야 해.", {"korean_daily_specialized_foodservice_chicken_sauce_missing", "korean_daily_judgment_taste_condition_dual_read"}),
            ("소스가 누락된 치킨 사진을 썸네일로 쓰면 클릭률 오를까?", {"korean_daily_specialized_foodservice_chicken_sauce_missing", "korean_daily_judgment_taste_condition_dual_read"}),
            ("방충망 찢어진 사진을 보고 집수리 광고 문구를 짜는 중이야.", {"korean_daily_specialized_homerepair_screen_torn_bugs", "korean_daily_foundation_window_screen_torn_bug_worry"}),
            ("벌레 들어올까 봐 창문 못 열겠다는 대사를 드라마에 넣었어.", {"korean_daily_specialized_homerepair_screen_torn_bugs", "korean_daily_foundation_window_screen_torn_bug_worry"}),
            ("가스레인지 불이 한쪽만 안 붙는 장면을 설명서 예시로 넣고 있어.", {"korean_daily_specialized_homerepair_stove_ignition_one_side", "korean_daily_practical_gas_stove_ignition_issue"}),
            ("월세집 벽에 못 박는 법 글을 블로그 제목으로 뽑아야 해.", {"korean_daily_specialized_homerepair_rental_wall_nail_uncertain", "korean_daily_practical_house_choice"}),
            ("하수구 냄새 제거제 광고 카피가 너무 세게 느껴져.", {"korean_daily_specialized_homerepair_sink_drain_smell", "korean_daily_more_house_drain_smell_cleaning_drag"}),
            ("배수구 냄새라는 키워드 검색량이 늘었대.", {"korean_daily_specialized_homerepair_sink_drain_smell", "korean_daily_more_house_drain_smell_cleaning_drag"}),
            ("냉동식품 녹을까 봐 보냉백 광고를 만들고 있어.", {"korean_daily_specialized_retail_frozen_food_melting_worry"}),
            ("마트에서 필요한 건 두 개였는데 열두 개 샀다는 밈 봤어.", {"korean_daily_more_shopping_grocery_two_to_twelve", "korean_daily_practical_living_cost_pressure"}),
            ("계산대에 열두 개 올라가는 장면을 콘티로 그리는 중이야.", {"korean_daily_more_shopping_grocery_two_to_twelve"}),
            ("필요한 건 두 개라는 문장을 쇼핑 앱 푸시로 쓰면 어때?", {"korean_daily_more_shopping_grocery_two_to_twelve"}),
            ("단톡방에 기발한 드립을 치는 캐릭터를 만들고 있어.", {"korean_daily_group_chat_drip_no_reaction_blanket_kick", "korean_daily_group_chat_funny_meme_no_reaction"}),
            ("이불 킥이라는 표현을 드립 실패 장면 제목으로 써도 돼?", {"korean_daily_group_chat_drip_no_reaction_blanket_kick"}),
            ("'고생하셨습니다' 이모티콘을 회사 굿즈로 만들면 팔릴까?", {"korean_daily_group_chat_weekend_emoji_logout"}),
            ("불금 단톡방 로그아웃이라는 문구를 티셔츠에 넣고 싶어.", {"korean_daily_group_chat_weekend_emoji_logout"}),
            ("유튜브 채널 운영 강의 썸네일 문구를 골라야 해.", {"korean_daily_youtube_channel_topic_persona", "korean_daily_icebreak_youtube_content"}),
            ("일상 질문이라는 카테고리명을 유튜브 채널에 붙이면 밋밋할까?", {"korean_daily_youtube_channel_topic_persona"}),
            ("어릴 때 꿈이랑 전공이 비슷한지 묻는 설문 문항을 만들었어.", {"korean_daily_career_dream_no_fake_memory", "korean_daily_icebreak_childhood_dream_no_fake"}),
            ("실제 전공이라는 표현이 자기소개서에 너무 딱딱해 보여.", {"korean_daily_career_dream_no_fake_memory"}),
            ("20살 대학생 캐릭터가 새로운 전공을 고르는 장면을 쓰고 있어.", {"korean_daily_college_major_choice"}),
            ("심리학 전공 홍보 문구를 좀 더 덜 딱딱하게 바꾸고 싶어.", {"korean_daily_college_major_choice"}),
            ("은퇴 후 작업방 인테리어 사진을 모으는 중이야.", {"korean_daily_retirement_environment", "korean_daily_icebreak_dream_house_city"}),
            ("늙어서 은퇴하고 나면이라는 문장이 너무 설명적이지?", {"korean_daily_retirement_environment"}),
            ("전환점이라는 단어를 제목에 넣을까 터닝포인트라고 쓸까?", {"korean_daily_turning_point_persona", "korean_daily_icebreak_turning_point_no_fake"}),
            ("인생 전환점 사례를 모은 기사 요약 중이야.", {"korean_daily_turning_point_persona", "korean_daily_icebreak_turning_point_no_fake"}),
            ("팥붕 슈붕 부먹 찍먹을 이을 논쟁거리 목록을 엑셀로 정리했어.", {"korean_daily_new_korean_food_debate", "korean_daily_icebreak_food_choice_combo"}),
            ("반숙 vs 완숙 논쟁을 설명하는 카드뉴스를 만들고 있어.", {"korean_daily_new_korean_food_debate"}),
            ("불로불사 약이라는 소설 소재를 쓰는데 먹을래 말래 대사는 뺄까?", {"korean_daily_immortality_pill_choice"}),
            ("영원히 사는 캐릭터가 불로불사를 후회하는 장면을 쓰고 있어.", {"korean_daily_immortality_pill_choice"}),
            ("쿠팡 와우 새벽배송 소리를 효과음으로 넣는 영상 편집 중이야.", {"korean_daily_expansion_shopping_coupang_dawn_delivery_alarm"}),
            ("오토바이 배달 소리 ASMR 제목 봤는데 너무 이상했어.", {"korean_daily_specialized_local_motorcycle_delivery_noise", "korean_daily_practical_sleep_noise_environment"}),
        ]
        forbidden_reply = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
        )

        self.assertEqual(len(cases), 50)

        for index, (prompt, forbidden_reasons) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-recent-priority-false-positive-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertNotIn(reason, forbidden_reasons)
                if str(reason).startswith("korean_daily_meta_"):
                    self.assertEqual(semantic_frame.get("schema"), "context_disambiguation")
                    self.assertEqual(semantic_frame.get("draft_frame_family"), "context_disambiguation")
                    self.assertEqual(semantic_frame.get("priority"), "meta_reflection")
                    self.assertIn("false_positive_guard", semantic_frame.get("pragmatic_cues") or [])
                    self.assertIn("not_life_event", semantic_frame.get("pragmatic_cues") or [])
                for phrase in forbidden_reply:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_selfmade_100_foundation_mix_reports_do_not_fall_back(self) -> None:
        engine = _build_draft_only_engine()
        report_paths = (
            Path(__file__).resolve().parents[1]
            / "reports"
            / "korean_daily_selfmade_100_foundation_mix_20260514.txt",
            Path(__file__).resolve().parents[1]
            / "reports"
            / "korean_daily_selfmade_100_foundation_mix_b_20260514.txt",
            Path(__file__).resolve().parents[1]
            / "reports"
            / "korean_daily_selfmade_100_foundation_mix_c_20260514.txt",
            Path(__file__).resolve().parents[1]
            / "reports"
            / "korean_daily_selfmade_100_foundation_mix_d_20260514.txt",
            Path(__file__).resolve().parents[1]
            / "reports"
            / "korean_daily_selfmade_100_question_variants_e_20260514.txt",
            Path(__file__).resolve().parents[1]
            / "reports"
            / "korean_daily_selfmade_100_question_variants_f_20260514.txt",
        )
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "받아둘게",
            "하나만 더 줘",
            "그 생각은 이해돼",
            "게임 얘기면 조작감",
        )

        for report_index, report_path in enumerate(report_paths, start=1):
            prompts = [line.strip() for line in report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(prompts), 100)
            for index, prompt in enumerate(prompts, start=1):
                with self.subTest(report=report_path.name, index=index, prompt=prompt):
                    result = await engine.respond(f"offline-selfmade-foundation-mix-{report_index}-{index}", prompt)
                    draft = result.draft_utterance or {}
                    reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""

                    self.assertFalse(result.llm_used)
                    self.assertEqual(result.render_source, "draft")
                    self.assertTrue(str(reason).startswith("korean_daily_"), reason)
                    for phrase in forbidden:
                        self.assertNotIn(phrase, result.reply)

    async def test_korean_knowledge_reflection_routes_without_clarification(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("한국과 중국과 일본은 왜 하나로 뭉치지 못할까?", "역사 기억"),
            ("민주주의는 왜 완벽하지 않은데도 계속 쓰일까?", "권력을 갈아치울"),
            ("조선은 왜 오래 버텼는데 결국 근대화에 늦었을까?", "근대 변화"),
            ("냉전은 왜 그렇게 오래 갔을까?", "두 진영"),
            ("전쟁은 왜 기술이 발전해도 사라지지 않을까?", "욕망"),
            ("국가는 개인의 자유를 어디까지 제한해도 될까?", "통제"),
            ("정치인은 왜 이상보다 타협을 많이 하게 될까?", "타협"),
            ("표현의 자유는 혐오 표현까지 보호해야 할까?", "자유와 피해"),
            ("AI에게 권리를 줘야 한다고 생각해?", "권리 논의"),
            ("사람의 기억을 복제하면 그건 같은 사람일까?", "복사본"),
            ("영생이 가능하면 인간은 더 행복해질까?", "의미"),
            ("돈으로 행복을 살 수 있다고 생각해?", "숨 쉴 공간"),
            ("진짜 어른이 된다는 건 뭘까?", "상처"),
            ("강한 리더가 민주적 절차보다 나을 때도 있을까?", "제한"),
            ("법은 도덕을 따라야 할까, 현실을 따라야 할까?", "최소한의 도덕"),
            ("다수결이 항상 정의롭다고 볼 수 있을까?", "소수자"),
            ("능력주의는 정말 공정한 시스템일까?", "출발선"),
            ("기술 발전은 인간을 더 외롭게 만들까?", "연결"),
            ("언론은 중립적일 수 있을까?", "검증 가능한 사실"),
            ("좋은 사회란 어떤 사회라고 생각해?", "다시 돌아올 길"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "사실 확인 전",
            "모른다고 둘게",
            "knowledge_reflection.",
        )

        for index, (prompt, expected) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-knowledge-reflection-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                evidence = result.world_state.evidence_packet

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertEqual(result.decision.action, ActionType.SHARE_OPINION)
                self.assertEqual(evidence.schema_hint, "knowledge_reflection")
                self.assertTrue(str(reason).startswith("knowledge_"))
                self.assertIn(expected, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_sleep_noise_relation_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("잠잘때 너무 시끄럽다", "korean_daily_practical_sleep_noise_environment", "소리 원인"),
            ("자려고 누웠는데 옆집 쿵쿵거려서 잠을 못 자겠어", "korean_daily_practical_sleep_noise_environment", "차단 수단"),
            ("새벽마다 오토바이 소리 너무 커서 잠을 설쳐", "korean_daily_practical_sleep_noise_environment", "시간 기록"),
            ("밤에 키보드 소리가 너무 시끄러워서 잠이 깨", "korean_daily_practical_sleep_noise_environment", "수면을 깨는 문제"),
            ("밤마다 층간소음 때문에 잠들려고 하면 바로 깨.", "korean_daily_practical_sleep_noise_environment", "소리 원인"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "그 생각은 이해돼",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-sleep-noise-relation-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), "sleep_routine")
                self.assertEqual(semantic_frame.get("schema"), "practical_advice")
                self.assertEqual(semantic_frame.get("draft_frame"), "sleep_noise_environment")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_heating_bill_relation_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("요즘 가스비 너무 올라서 보일러 켜기 무서워.", "korean_daily_practical_heating_bill_anxiety", "저비용 보온"),
            ("도시가스비 고지서 보고 이번 달 생활비가 불안해졌어.", "korean_daily_practical_heating_bill_anxiety", "목표 온도"),
            ("난방비 아끼려고 보일러를 꺼야 하나 고민 중이야.", "korean_daily_practical_heating_bill_anxiety", "완전히 끄고 버티기보다"),
            ("관리비랑 난방비가 부담돼서 히터 트는 것도 겁나.", "korean_daily_practical_heating_bill_anxiety", "생활비 경보"),
            ("전기요금이랑 난방비 올라서 난방 트는 게 부담돼.", "korean_daily_practical_heating_bill_anxiety", "생활비 경보"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "환기와 불꽃 차단",
            "119",
            "덜 젖고 덜 후회",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-heating-bill-relation-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), "money_living")
                self.assertEqual(semantic_frame.get("schema"), "practical_advice")
                self.assertEqual(semantic_frame.get("draft_frame"), "heating_bill_anxiety")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_living_cost_relation_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("기름값이 너무 올라서 주유소 갈 때마다 지갑이 아파.", "korean_daily_practical_living_cost_pressure", "필수 지출"),
            ("주유하고 나면 기름값 때문에 이번 달 예산이 흔들려.", "korean_daily_practical_living_cost_pressure", "줄일 지출"),
            ("요즘 물가 너무 올라서 마트 가기 무서워.", "korean_daily_practical_living_cost_pressure", "예산 경보"),
            ("식비가 비싸져서 장보기 할 때마다 지갑이 겁나.", "korean_daily_practical_living_cost_pressure", "미룰 지출"),
            ("식료품값이 올라서 장바구니 담을 때마다 예산이 무너져.", "korean_daily_practical_living_cost_pressure", "예산 경보"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "부담이 너무 크지",
            "물 붓지",
            "119",
            "게임 얘기면 조작감",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-living-cost-relation-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), "money_living")
                self.assertEqual(semantic_frame.get("schema"), "practical_advice")
                self.assertEqual(semantic_frame.get("draft_frame"), "living_cost_pressure")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_gas_stove_ignition_relation_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("가스레인지 한쪽만 불이 안 붙어서 점화장치 문제인가 봐.", "korean_daily_practical_gas_stove_ignition_issue", "계속 딸깍"),
            ("가스레인지 불이 한쪽만 안 붙어. 점화부가 더러운 건가?", "korean_daily_practical_gas_stove_ignition_issue", "점화부"),
            ("가스버너 한쪽 화구만 불이 안 켜져서 요리 시작부터 막혔어.", "korean_daily_practical_gas_stove_ignition_issue", "밸브"),
            ("가스레인지 딸깍거리기만 하고 불꽃이 안 올라와.", "korean_daily_practical_gas_stove_ignition_issue", "AS"),
            ("가스렌지 한쪽 화구가 딸깍만 하고 불이 안 켜져.", "korean_daily_practical_gas_stove_ignition_issue", "점화부"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "환기와 불꽃 차단",
            "119",
            "기름값",
            "예산 경보",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-gas-stove-ignition-relation-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), "home_maintenance")
                self.assertEqual(semantic_frame.get("schema"), "practical_advice")
                self.assertEqual(semantic_frame.get("draft_frame"), "gas_stove_ignition_issue")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_appliance_design_review_relation_prompts(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            ("가스레인지 디자인이 예뻐서 사진 저장했는데 점화장치 후기가 별로래.", "korean_daily_practical_appliance_design_review", "보류가 맞아"),
            ("가스렌지 디자인은 취향인데 점화 리뷰가 안 좋아서 고민돼.", "korean_daily_practical_appliance_design_review", "점화 안정성"),
            ("주방가전 예쁜 제품 저장했는데 내구성 후기가 별로라 망설여져.", "korean_daily_practical_appliance_design_review", "그쪽이 먼저"),
            ("가전 디자인은 끌리는데 성능 평이 안 좋아.", "korean_daily_practical_appliance_design_review", "디자인은 매일 보이지만"),
            ("예쁜 가전 제품 저장해놨는데 성능 후기가 안 좋아서 고민돼.", "korean_daily_practical_appliance_design_review", "실제 사용 안전"),
        ]
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "캐릭터가 바로 살아",
            "과몰입 각",
            "환기와 불꽃 차단",
            "119",
            "계속 딸깍거리지 말고",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-appliance-design-review-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), "home_appliance")
                self.assertEqual(semantic_frame.get("schema"), "practical_advice")
                self.assertEqual(semantic_frame.get("draft_frame"), "appliance_design_review_judgment")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_character_design_review_does_not_leak_color_lore(self) -> None:
        engine = _build_draft_only_engine()
        cases = [
            (
                "이번 신작 캐릭터 디자인 예뻐서 사진 저장했는데 후기가 별로래.",
                "korean_daily_expansion_character_design_review_tension",
                "기대치는 낮추는 게 맞아",
            ),
            (
                "공개된 캐릭터 비주얼은 저장할 만큼 취향인데 평이 별로라 좀 식었어.",
                "korean_daily_expansion_character_design_review_tension",
                "내용 평은 따로",
            ),
            (
                "캐릭터 디자인은 취향이라 저장했는데 리뷰가 안 좋아서 시작할지 말지 고민돼.",
                "korean_daily_expansion_character_design_review_tension",
                "기대치는 낮추는 게 맞아",
            ),
            (
                "신작 캐릭터 비주얼 저장했는데 반응이 별로라 기대가 식어.",
                "korean_daily_expansion_character_design_review_tension",
                "내용 평은 따로",
            ),
            (
                "공개된 캐릭터 디자인은 예뻐서 저장했는데 반응이 별로래.",
                "korean_daily_expansion_character_design_review_tension",
                "비주얼에 끌리는 건 인정",
            ),
        ]
        forbidden = (
            "백발",
            "벽안",
            "서사가 삼천 페이지",
            "가스레인지",
            "점화 안정성",
            "어느 쪽 기준",
            "하나만 더 줘",
        )

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-character-design-review-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), "character_design")
                self.assertEqual(semantic_frame.get("schema"), "preference_disclosure")
                self.assertEqual(semantic_frame.get("draft_frame_family"), "choice_preference")
                self.assertEqual(semantic_frame.get("priority"), "choice_judgment")
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_relation_expansion_fixture(self) -> None:
        engine = _build_draft_only_engine()
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "high_context_relation_expansion_cases.json"
        cases = [
            (str(prompt), str(expected_reason), str(expected_reply))
            for prompt, expected_reason, expected_reply in json.loads(fixture_path.read_text(encoding="utf-8"))
        ]
        expected_frames = {
            "korean_daily_practical_sleep_noise_environment": ("sleep_routine", "practical_advice", "sleep_noise_environment"),
            "korean_daily_practical_heating_bill_anxiety": ("money_living", "practical_advice", "heating_bill_anxiety"),
            "korean_daily_practical_living_cost_pressure": ("money_living", "practical_advice", "living_cost_pressure"),
            "korean_daily_practical_gas_stove_ignition_issue": ("home_maintenance", "practical_advice", "gas_stove_ignition_issue"),
            "korean_daily_practical_appliance_design_review": ("home_appliance", "practical_advice", "appliance_design_review_judgment"),
            "korean_daily_expansion_character_design_review_tension": ("character_design", "preference_disclosure", "expansion_character_design_review_tension"),
        }
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "백발",
            "벽안",
            "환기와 불꽃 차단",
            "물 붓지",
        )

        self.assertEqual(len(cases), 120)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-relation-expansion-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}
                expected_domain, expected_schema, expected_frame = expected_frames[expected_reason]

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), expected_domain)
                self.assertEqual(semantic_frame.get("schema"), expected_schema)
                self.assertEqual(semantic_frame.get("draft_frame"), expected_frame)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_relation_contrast_fixture(self) -> None:
        engine = _build_draft_only_engine()
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "high_context_relation_contrast_cases.json"
        cases = [
            (str(prompt), str(expected_reason), str(expected_reply))
            for prompt, expected_reason, expected_reply in json.loads(fixture_path.read_text(encoding="utf-8"))
        ]
        expected_frames = {
            "korean_daily_practical_sleep_noise_environment": ("sleep_routine", "practical_advice", "sleep_noise_environment"),
            "korean_daily_practical_heating_bill_anxiety": ("money_living", "practical_advice", "heating_bill_anxiety"),
            "korean_daily_practical_living_cost_pressure": ("money_living", "practical_advice", "living_cost_pressure"),
            "korean_daily_practical_gas_stove_ignition_issue": ("home_maintenance", "practical_advice", "gas_stove_ignition_issue"),
            "korean_daily_practical_appliance_design_review": ("home_appliance", "practical_advice", "appliance_design_review_judgment"),
            "korean_daily_expansion_character_design_review_tension": ("character_design", "preference_disclosure", "expansion_character_design_review_tension"),
        }
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "백발",
            "벽안",
            "환기와 불꽃 차단",
            "물 붓지",
        )

        self.assertEqual(len(cases), 60)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-relation-contrast-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}
                expected_domain, expected_schema, expected_frame = expected_frames[expected_reason]

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), expected_domain)
                self.assertEqual(semantic_frame.get("schema"), expected_schema)
                self.assertEqual(semantic_frame.get("draft_frame"), expected_frame)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_money_boundary_fixture(self) -> None:
        engine = _build_draft_only_engine()
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "high_context_money_boundary_cases.json"
        cases = [
            (str(prompt), str(expected_reason), str(expected_reply))
            for prompt, expected_reason, expected_reply in json.loads(fixture_path.read_text(encoding="utf-8"))
        ]
        expected_frames = {
            "korean_daily_practical_heating_bill_anxiety": ("money_living", "practical_advice", "heating_bill_anxiety"),
            "korean_daily_practical_living_cost_pressure": ("money_living", "practical_advice", "living_cost_pressure"),
        }
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "가스레인지 한쪽만",
            "점화장치",
            "잠잘 때",
        )

        self.assertEqual(len(cases), 40)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-money-boundary-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}
                expected_domain, expected_schema, expected_frame = expected_frames[expected_reason]

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), expected_domain)
                self.assertEqual(semantic_frame.get("schema"), expected_schema)
                self.assertEqual(semantic_frame.get("draft_frame"), expected_frame)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_money_pairwise_fixture(self) -> None:
        engine = _build_draft_only_engine()
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "high_context_money_pairwise_cases.json"
        cases = [
            (str(prompt), str(expected_reason), str(expected_reply))
            for prompt, expected_reason, expected_reply in json.loads(fixture_path.read_text(encoding="utf-8"))
        ]
        expected_frames = {
            "korean_daily_practical_heating_bill_anxiety": ("money_living", "practical_advice", "heating_bill_anxiety"),
            "korean_daily_practical_living_cost_pressure": ("money_living", "practical_advice", "living_cost_pressure"),
        }
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "가스레인지 한쪽만",
            "점화장치",
            "잠잘 때",
        )

        self.assertEqual(len(cases), 80)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-money-pairwise-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}
                expected_domain, expected_schema, expected_frame = expected_frames[expected_reason]

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), expected_domain)
                self.assertEqual(semantic_frame.get("schema"), expected_schema)
                self.assertEqual(semantic_frame.get("draft_frame"), expected_frame)
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)

    async def test_korean_daily_high_context_money_comparison_fixture(self) -> None:
        engine = _build_draft_only_engine()
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "high_context_money_comparison_cases.json"
        cases = [
            (str(prompt), str(expected_reason), str(expected_reply))
            for prompt, expected_reason, expected_reply in json.loads(fixture_path.read_text(encoding="utf-8"))
        ]
        expected_frames = {
            "korean_daily_practical_heating_bill_anxiety": ("money_living", "practical_advice", "heating_bill_anxiety"),
            "korean_daily_practical_living_cost_pressure": ("money_living", "practical_advice", "living_cost_pressure"),
        }
        forbidden = (
            "어느 쪽 기준",
            "하나만 더 줘",
            "무리하게 밀 필요",
            "그 생각은 이해돼",
            "가스레인지 한쪽만",
            "점화장치",
            "잠잘 때",
        )

        self.assertEqual(len(cases), 140)

        for index, (prompt, expected_reason, expected_reply) in enumerate(cases, start=1):
            with self.subTest(index=index, prompt=prompt):
                result = await engine.respond(f"offline-high-context-money-comparison-{index}", prompt)
                draft = result.draft_utterance or {}
                reason = draft.get("direct_surface_reason") or draft.get("output_shape") or ""
                semantic_frame = draft.get("semantic_frame") or {}
                expected_domain, expected_schema, expected_frame = expected_frames[expected_reason]
                expected_focus = (
                    "heating_bill_focus"
                    if expected_reason == "korean_daily_practical_heating_bill_anxiety"
                    else "living_cost_focus"
                )
                targets = semantic_frame.get("targets") or {}

                self.assertFalse(result.llm_used)
                self.assertEqual(result.render_source, "draft")
                self.assertNotEqual(result.decision.action, ActionType.ASK_CLARIFICATION)
                self.assertEqual(reason, expected_reason)
                self.assertEqual(semantic_frame.get("domain"), expected_domain)
                self.assertEqual(semantic_frame.get("schema"), expected_schema)
                self.assertEqual(semantic_frame.get("draft_frame"), expected_frame)
                self.assertEqual(semantic_frame.get("comparison_focus"), expected_focus)
                self.assertEqual(targets.get("comparison_focus"), expected_focus)
                self.assertIn("money_comparison_focus", semantic_frame.get("pragmatic_cues") or [])
                self.assertIn(expected_reply, result.reply)
                for phrase in forbidden:
                    self.assertNotIn(phrase, result.reply)


if __name__ == "__main__":
    unittest.main()
