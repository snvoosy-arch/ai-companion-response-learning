from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT))

from predictive_bot.config import AppConfig
from predictive_bot.core.models import WeatherReport
from predictive_bot.core.tools import CurrentTimeAnswer, NewsHeadline
from predictive_bot.factory import build_engine

REPORT_DIR = PROJECT_ROOT / "reports"
OUT_JSON = REPORT_DIR / "black_context_understanding_100_20260424.json"
OUT_MD = REPORT_DIR / "black_context_understanding_100_20260424.md"


def load_env_file(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


class FakeWeatherService:
    async def get_current_weather(self, location: str) -> WeatherReport:
        return WeatherReport(location=location, temperature_c=18.0, description="맑음", wind_kph=7.0)


class FakeTimeService:
    def get_current_time(self) -> CurrentTimeAnswer:
        return CurrentTimeAnswer(
            formatted_time="14:32",
            formatted_date="2026-04-24",
            timezone_name="Asia/Seoul",
            source="black_context_understanding_fake_clock",
        )


class FakeNewsService:
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        items = [
            NewsHeadline(title="AI 반도체 경쟁이 다시 커지고 있다", source="테스트뉴스"),
            NewsHeadline(title="국내 증시가 장중 상승세를 보였다", source="테스트경제"),
            NewsHeadline(title="게임 리그 결승 일정이 공개됐다", source="테스트게임"),
            NewsHeadline(title="신작 드라마 공개 후 반응이 갈렸다", source="테스트연예"),
        ]
        return items[:limit]


CASES: list[dict[str, Any]] = [
    {
        "id": "A01",
        "category": "activity_recommendation",
        "text": "바다에서 무엇을 하고 놀면 좋을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["바다"],
            "must_include_any": ["물놀이", "모래사장 산책", "사진 찍기"],
        },
    },
    {
        "id": "A02",
        "category": "activity_recommendation",
        "text": "해변에 가면 뭐 하고 놀까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["해변"],
            "must_include_any": ["물놀이", "모래사장 산책", "사진 찍기"],
        },
    },
    {
        "id": "A03",
        "category": "activity_recommendation",
        "text": "계곡에서는 뭐 하고 쉬면 좋을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["계곡"],
            "must_include_any": ["발 담그기", "물가 산책", "사진 찍기"],
        },
    },
    {
        "id": "A04",
        "category": "activity_recommendation",
        "text": "공원에서 친구랑 뭐 하면 좋을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["공원"],
            "must_include_any": ["산책", "가벼운 공놀이", "돗자리"],
        },
    },
    {
        "id": "A05",
        "category": "activity_recommendation",
        "text": "한강 가면 뭘 하면 재밌어?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["한강"],
            "must_include_any": ["산책", "자전거", "돗자리"],
        },
    },
    {
        "id": "A06",
        "category": "activity_recommendation",
        "text": "산에 가면 뭘 하면서 보내면 좋아?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["산"],
            "must_include_any": ["산책", "전망", "사진"],
        },
    },
    {
        "id": "A07",
        "category": "activity_recommendation",
        "text": "캠핑장에서 뭐 하고 놀면 좋지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["캠핑장"],
            "must_include_any": ["불멍", "요리", "보드게임"],
        },
    },
    {
        "id": "A08",
        "category": "activity_recommendation",
        "text": "놀이공원 가면 뭐부터 타는 게 좋아?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["놀이공원"],
        },
    },
    {
        "id": "A09",
        "category": "activity_recommendation",
        "text": "해수욕장에서 놀 때 뭐 하면 좋아?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["해수욕장"],
            "must_include_any": ["물놀이", "모래사장 산책", "사진 찍기"],
        },
    },
    {
        "id": "A10",
        "category": "activity_recommendation",
        "text": "비 오는 날 실내에서 뭐 하고 놀까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "activity_recommendation",
            "anchor_any": ["실내", "비"],
        },
    },
    {
        "id": "D01",
        "category": "soft_decision",
        "text": "먼저 연락해도 괜찮을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D02",
        "category": "soft_decision",
        "text": "지금 사과해볼까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D03",
        "category": "soft_decision",
        "text": "오늘은 그냥 쉬어도 될까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D04",
        "category": "soft_decision",
        "text": "이 얘기 다시 꺼내도 괜찮을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D05",
        "category": "soft_decision",
        "text": "약속을 미루자고 말해도 될까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D06",
        "category": "soft_decision",
        "text": "선물 보내는 게 나을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D07",
        "category": "soft_decision",
        "text": "지금 물어봐도 괜찮을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D08",
        "category": "soft_decision",
        "text": "조금 기다렸다가 말하는 쪽이 맞을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D09",
        "category": "soft_decision",
        "text": "솔직하게 말해볼까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "D10",
        "category": "soft_decision",
        "text": "내가 먼저 정리하는 게 맞을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "soft_decision_advice",
            "reason_prefix": "opinion.ask.soft_decision",
        },
    },
    {
        "id": "P01",
        "category": "process_advice",
        "text": "이 프로젝트는 무엇부터 해야 할까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P02",
        "category": "process_advice",
        "text": "로그는 무엇부터 점검해야 할까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P03",
        "category": "process_advice",
        "text": "말투 교정은 어떤 순서로 보면 좋을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P04",
        "category": "process_advice",
        "text": "친구한테 뭐라고 하는 게 좋을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P05",
        "category": "process_advice",
        "text": "어색해진 분위기를 어떻게 가볍게 확인할 수 있을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P06",
        "category": "process_advice",
        "text": "발표 준비는 어떻게 시작해야 할까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P07",
        "category": "process_advice",
        "text": "오늘 옷은 어떻게 입는 게 좋을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P08",
        "category": "process_advice",
        "text": "둘 중 어떤 쪽을 우선해야 할까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P09",
        "category": "process_advice",
        "text": "여행 준비는 뭘 먼저 봐야 할까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "P10",
        "category": "process_advice",
        "text": "대화를 다시 이어가려면 무엇부터 분명히 하면 좋을까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "process_advice",
            "reason_prefix": "opinion.ask.process_advice",
        },
    },
    {
        "id": "R01",
        "category": "preference_habit",
        "text": "멜론 좋아해?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "preference_disclosure",
            "reason_prefix": "opinion.ask.preference",
        },
    },
    {
        "id": "R02",
        "category": "preference_habit",
        "text": "공포 영화 좋아해?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "preference_disclosure",
            "reason_prefix": "opinion.ask.preference",
        },
    },
    {
        "id": "R03",
        "category": "preference_habit",
        "text": "잔잔한 음악 좋아해?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "preference_disclosure",
            "reason_prefix": "opinion.ask.preference",
        },
    },
    {
        "id": "R04",
        "category": "preference_habit",
        "text": "매운 음식 좋아해?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "preference_disclosure",
            "reason_prefix": "opinion.ask.preference",
        },
    },
    {
        "id": "R05",
        "category": "preference_habit",
        "text": "조용한 분위기 좋아하는 편이야?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema_in": ["habit_preference", "preference_disclosure"],
        },
    },
    {
        "id": "R06",
        "category": "preference_habit",
        "text": "너는 먼저 말 거는 편이야?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema_in": ["habit_preference", "self_style"],
        },
    },
    {
        "id": "R07",
        "category": "preference_habit",
        "text": "밤에 작업하는 편이야?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema_in": ["habit_preference", "self_style"],
        },
    },
    {
        "id": "R08",
        "category": "preference_habit",
        "text": "장난은 싫어해?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "preference_disclosure",
            "reason_prefix": "opinion.ask.preference",
        },
    },
    {
        "id": "R09",
        "category": "preference_habit",
        "text": "비 오는 날 산책 좋아해?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "preference_disclosure",
            "reason_prefix": "opinion.ask.preference",
        },
    },
    {
        "id": "R10",
        "category": "preference_habit",
        "text": "솔직하게 말하는 편이야?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema_in": ["habit_preference", "self_style"],
        },
    },
    {
        "id": "J01",
        "category": "reflective_judgment",
        "text": "이건 조금 부담되지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J02",
        "category": "reflective_judgment",
        "text": "지금은 한발 물러서는 게 낫지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J03",
        "category": "reflective_judgment",
        "text": "이 분위기면 천천히 가도 되겠지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J04",
        "category": "reflective_judgment",
        "text": "내가 예민한 건 아니겠지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J05",
        "category": "reflective_judgment",
        "text": "이 말투 너무 차갑지 않아?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J06",
        "category": "reflective_judgment",
        "text": "그렇게 느끼는 게 이상하진 않지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J07",
        "category": "reflective_judgment",
        "text": "이 선택 현실적일까?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J08",
        "category": "reflective_judgment",
        "text": "지금 바로 정리하는 게 중요하지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J09",
        "category": "reflective_judgment",
        "text": "그 사람이 서운했을 것 같지?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "J10",
        "category": "reflective_judgment",
        "text": "이 흐름 이해돼?",
        "expect": {
            "action": "share_opinion",
            "intent": "smalltalk_opinion",
            "schema": "reflective_judgment",
            "reason_prefix": "opinion.ask.reflective",
        },
    },
    {
        "id": "F01",
        "category": "feeling_social",
        "text": "오늘 좀 지친 것 같은데 이상한 걸까?",
        "expect": {"action": "share_feeling", "intent": "smalltalk_feeling"},
    },
    {
        "id": "F02",
        "category": "feeling_social",
        "text": "괜히 서운한데 내가 너무 예민한 걸까?",
        "expect": {"action": "share_feeling", "intent": "smalltalk_feeling"},
    },
    {
        "id": "F03",
        "category": "feeling_social",
        "text": "친구 잘되는 거 축하했는데 씁쓸한 게 이상해?",
        "expect": {"action": "share_feeling", "intent": "smalltalk_feeling"},
    },
    {
        "id": "F04",
        "category": "feeling_social",
        "text": "요즘 잠을 못 자서 예민한데 어떻게 다뤄야 할까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
        },
    },
    {
        "id": "F05",
        "category": "feeling_social",
        "text": "답장이 늦으니까 불안한데 너무 집착하는 걸까?",
        "expect": {"action": "share_feeling", "intent": "smalltalk_feeling"},
    },
    {
        "id": "F06",
        "category": "feeling_social",
        "text": "좋은 일인데도 허탈한 건 왜일까?",
        "expect": {"action": "share_feeling", "intent": "smalltalk_feeling"},
    },
    {
        "id": "F07",
        "category": "feeling_social",
        "text": "아무것도 하기 싫을 때는 어떻게 해야 할까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
        },
    },
    {
        "id": "F08",
        "category": "feeling_social",
        "text": "계속 비교하게 되는데 어떻게 끊어야 할까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
        },
    },
    {
        "id": "F09",
        "category": "feeling_social",
        "text": "말수가 줄어든 친구를 보면 내가 불편해진 걸까?",
        "expect": {"action": "share_feeling", "intent": "smalltalk_feeling"},
    },
    {
        "id": "F10",
        "category": "feeling_social",
        "text": "기분 나쁘진 않았겠지?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
        },
    },
    {"id": "K01", "category": "external_lookup", "text": "오늘 날씨 어때?", "expect": {"action": "ask_location", "intent": "weather"}},
    {"id": "K02", "category": "external_lookup", "text": "서울 날씨 어때?", "expect": {"action": "weather_lookup", "intent": "weather"}},
    {"id": "K03", "category": "external_lookup", "text": "지금 몇 시야?", "expect": {"action": "tell_time", "intent": "time_date"}},
    {"id": "K04", "category": "external_lookup", "text": "오늘 날짜가 뭐야?", "expect": {"action": "tell_time", "intent": "time_date"}},
    {"id": "K05", "category": "external_lookup", "text": "오늘 뉴스 알려줘?", "expect": {"action": "news_answer", "intent": "news"}},
    {"id": "K06", "category": "external_lookup", "text": "AI 뉴스 뭐 있어?", "expect": {"action": "news_answer", "intent": "news"}},
    {"id": "K07", "category": "external_lookup", "text": "미국 수도는 어디야?", "expect": {"action": "search_answer", "intent": "search_request"}},
    {"id": "K08", "category": "external_lookup", "text": "파이썬은 무슨 언어야?", "expect": {"action": "search_answer", "intent": "search_request"}},
    {"id": "K09", "category": "external_lookup", "text": "패딩 입어야 되냐?", "expect": {"action": "ask_location", "intent": "weather"}},
    {"id": "K10", "category": "external_lookup", "text": "부산은 지금 몇 도야?", "expect": {"action": "weather_lookup", "intent": "weather"}},
    {"id": "M01", "category": "media_music_game", "text": "볼 만한 영화 추천해줘?", "expect": {"action": "recommend", "intent": "media_recommend"}},
    {"id": "M02", "category": "media_music_game", "text": "잔잔한 드라마 추천해줘?", "expect": {"action": "recommend", "intent": "media_recommend"}},
    {"id": "M03", "category": "media_music_game", "text": "요즘 들을 만한 노래 추천해줘?", "expect": {"action": "music_chat", "intent": "music"}},
    {"id": "M04", "category": "media_music_game", "text": "집중할 때 들을 음악 뭐가 좋아?", "expect": {"action": "music_chat", "intent": "music"}},
    {"id": "M05", "category": "media_music_game", "text": "같이 게임할래?", "expect": {"action": "game_accept_or_decline", "intent": "game_invite"}},
    {"id": "M06", "category": "media_music_game", "text": "롤 한 판 할래?", "expect": {"action": "game_accept_or_decline", "intent": "game_invite"}},
    {
        "id": "M07",
        "category": "media_music_game",
        "text": "이 게임 어떻게 깨는 게 좋을까?",
        "expect": {"action_in": ["game_chat", "share_opinion"], "intent_in": ["game_talk", "smalltalk_opinion"]},
    },
    {
        "id": "M08",
        "category": "media_music_game",
        "text": "스팀에서 뭐 할 만한 거 있어?",
        "expect": {"action_in": ["game_chat", "recommend"], "intent_in": ["game_talk", "media_recommend"]},
    },
    {"id": "M09", "category": "media_music_game", "text": "공포영화 뭐 볼까?", "expect": {"action": "recommend", "intent": "media_recommend"}},
    {"id": "M10", "category": "media_music_game", "text": "노래 뭐 들을까?", "expect": {"action": "music_chat", "intent": "music"}},
    {"id": "I01", "category": "identity_meta", "text": "너 누구야?", "expect": {"action": "answer_identity", "intent": "who_are_you"}},
    {"id": "I02", "category": "identity_meta", "text": "뭐 할 수 있어?", "expect": {"action": "explain_capabilities", "intent": "help"}},
    {"id": "I03", "category": "identity_meta", "text": "가능한 거 대충 알려줘?", "expect": {"action": "explain_capabilities", "intent": "help"}},
    {"id": "I04", "category": "identity_meta", "text": "왜?", "expect": {"action": "ask_clarification", "intent": "why"}},
    {"id": "I05", "category": "identity_meta", "text": "그 판단 근거는 뭐야?", "expect": {"action": "ask_clarification", "intent": "why"}},
    {"id": "I06", "category": "identity_meta", "text": "방금 왜 그렇게 봤어?", "expect": {"action": "ask_clarification", "intent": "why"}},
    {
        "id": "I07",
        "category": "identity_meta",
        "text": "내가 무슨 말을 한 건지 정리해줄래?",
        "expect": {
            "action_in": ["direct_reply", "continue_conversation", "ask_clarification"],
            "intent_in": ["reply_request", "smalltalk_generic", "unknown"],
        },
    },
    {
        "id": "I08",
        "category": "identity_meta",
        "text": "너는 기억할 수 있어?",
        "expect": {
            "action_in": ["explain_capabilities", "answer_identity", "direct_reply"],
            "intent_in": ["help", "who_are_you", "smalltalk_generic"],
        },
    },
    {
        "id": "I09",
        "category": "identity_meta",
        "text": "내 질문을 어떤 기준으로 판단해?",
        "expect": {
            "action_in": ["explain_capabilities", "direct_reply", "answer_identity"],
            "intent_in": ["help", "smalltalk_generic", "who_are_you"],
        },
    },
    {
        "id": "I10",
        "category": "identity_meta",
        "text": "네가 모르는 건 모른다고 말할 수 있어?",
        "expect": {
            "action_in": ["explain_capabilities", "answer_identity", "direct_reply"],
            "intent_in": ["help", "who_are_you", "smalltalk_generic"],
        },
    },
    {
        "id": "H01",
        "category": "high_context_nuance",
        "text": "하트만 남기고 끝났는데 거리가 생긴 걸까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["relational_interpretation", "reflective_judgment", "soft_decision_advice"],
        },
    },
    {
        "id": "H02",
        "category": "high_context_nuance",
        "text": "화제가 자꾸 바뀌면 보류나 거절 쪽으로 봐야 할까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["relational_interpretation", "process_advice", "soft_decision_advice"],
        },
    },
    {
        "id": "H03",
        "category": "high_context_nuance",
        "text": "잘 넘기는 것보다 덜 상처받는 게 더 중요해지는 걸까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["comparative_reflection", "reflective_judgment"],
        },
    },
    {
        "id": "H04",
        "category": "high_context_nuance",
        "text": "오늘은 잘 보내는 것보다 덜 힘들게 보내는 게 목표일까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["comparative_reflection", "reflective_judgment", "soft_decision_advice"],
        },
    },
    {
        "id": "H05",
        "category": "high_context_nuance",
        "text": "밤하늘 색감이 이상하게 차분해?",
        "expect": {
            "action_in": ["continue_conversation", "share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_generic", "smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["aesthetic_reflection", "reflective_observation", "reflective_judgment"],
        },
    },
    {
        "id": "H06",
        "category": "high_context_nuance",
        "text": "이 장면은 침묵이 더 선명한 것 같지?",
        "expect": {
            "action_in": ["continue_conversation", "share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_generic", "smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["aesthetic_reflection", "reflective_observation", "reflective_judgment"],
        },
    },
    {
        "id": "H07",
        "category": "high_context_nuance",
        "text": "계속 밀리는 사람은 나구나 싶은데 그렇게 봐도 될까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["relational_interpretation", "soft_decision_advice", "reflective_judgment"],
        },
    },
    {
        "id": "H08",
        "category": "high_context_nuance",
        "text": "그 말은 이해받기 위한 시도라기보다 닫힌 문을 다시 두드리는 느낌일까?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["relational_interpretation", "reflective_judgment"],
        },
    },
    {
        "id": "H09",
        "category": "high_context_nuance",
        "text": "빛이 너무 차갑게 느껴지지?",
        "expect": {
            "action_in": ["continue_conversation", "share_feeling", "share_opinion"],
            "intent_in": ["smalltalk_generic", "smalltalk_feeling", "smalltalk_opinion"],
            "schema_in": ["aesthetic_reflection", "reflective_observation", "reflective_judgment"],
        },
    },
    {
        "id": "H10",
        "category": "high_context_nuance",
        "text": "이 감정이 오래 남는 종류의 느낌 같아?",
        "expect": {
            "action_in": ["share_feeling", "share_opinion", "continue_conversation"],
            "intent_in": ["smalltalk_feeling", "smalltalk_opinion", "smalltalk_generic"],
            "schema_in": ["reflective_observation", "reflective_judgment"],
        },
    },
]


def expected_ok(actual: dict[str, Any], expect: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks: dict[str, bool] = {}
    if "action" in expect:
        checks["action"] = actual["action"] == expect["action"]
    if "action_in" in expect:
        checks["action"] = actual["action"] in set(expect["action_in"])
    if "intent" in expect:
        checks["intent"] = actual["intent"] == expect["intent"]
    if "intent_in" in expect:
        checks["intent"] = actual["intent"] in set(expect["intent_in"])
    if "schema" in expect:
        checks["schema"] = actual["question_schema"] == expect["schema"]
    if "schema_in" in expect:
        checks["schema"] = actual["question_schema"] in set(expect["schema_in"])
    if "reason_prefix" in expect:
        checks["reason_prefix"] = str(actual["reason_code"]).startswith(expect["reason_prefix"])
    if "cue" in expect:
        checks["cue"] = expect["cue"] in actual["pragmatic_cues"]
    if "anchor_any" in expect:
        anchor = actual.get("response_plan_anchor") or ""
        checks["anchor_any"] = any(item in anchor for item in expect["anchor_any"])
    if "must_include_any" in expect:
        must = actual.get("response_plan_must_include") or []
        checks["must_include_any"] = any(
            any(expected_item in actual_item for actual_item in must)
            for expected_item in expect["must_include_any"]
        )
    return all(checks.values()) if checks else True, checks


def action_ok(actual: dict[str, Any], expect: dict[str, Any]) -> bool:
    if "action" in expect:
        return actual["action"] == expect["action"]
    if "action_in" in expect:
        return actual["action"] in set(expect["action_in"])
    return True


async def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    load_env_file(PROJECT_ROOT / ".env.black.duo.kcbertcpu.broadrebuildv2.local")
    os.environ["GENERATION_BACKEND"] = "template"
    os.environ["STATE_BACKEND"] = "memory"
    os.environ["KNOWLEDGE_BACKEND"] = "builtin"
    os.environ["KCBERT_DEVICE"] = "cpu"
    os.environ["TTS_ENABLED"] = "false"

    config = AppConfig.from_env()
    engine = build_engine(config)
    engine.weather_service = FakeWeatherService()
    engine.time_service = FakeTimeService()
    engine.news_service = FakeNewsService()

    results = []
    for case in CASES:
        result = await engine.respond(f"context100-{case['id']}", case["text"])
        evidence = result.features.classifier_evidence
        plan = result.response_plan
        actual = {
            "intent": result.features.intent.value,
            "action": result.decision.action.value,
            "question_schema": result.features.question_schema,
            "reason_code": result.decision.reason_code,
            "reason_flags": list(result.decision.reason_flags),
            "pragmatic_cues": list(result.features.pragmatic_cues),
            "topic_hint": result.features.topic_hint,
            "speech_act": result.features.speech_act,
            "response_needs": list(result.features.response_needs),
            "classifier_source": evidence.source if evidence else None,
            "classifier_chosen_reason": evidence.chosen_reason if evidence else None,
            "classifier_top_scores": [
                {"label": score.label, "score": score.score}
                for score in (evidence.top_scores if evidence else [])
            ],
            "classifier_rule_hits": list(evidence.rule_hits) if evidence else [],
            "response_plan_stance": plan.stance if plan else None,
            "response_plan_anchor": plan.anchor if plan else None,
            "response_plan_must_include": list(plan.must_include) if plan else [],
            "reply": result.reply,
        }
        strict_pass, checks = expected_ok(actual, case["expect"])
        results.append(
            {
                **case,
                "actual": actual,
                "checks": checks,
                "strict_pass": strict_pass,
                "action_pass": action_ok(actual, case["expect"]),
            }
        )

    total = len(results)
    strict_passes = sum(1 for item in results if item["strict_pass"])
    action_passes = sum(1 for item in results if item["action_pass"])
    by_category = {}
    for category in sorted({item["category"] for item in results}):
        subset = [item for item in results if item["category"] == category]
        by_category[category] = {
            "total": len(subset),
            "strict_pass": sum(1 for item in subset if item["strict_pass"]),
            "action_pass": sum(1 for item in subset if item["action_pass"]),
        }

    action_counts = Counter(item["actual"]["action"] for item in results)
    intent_counts = Counter(item["actual"]["intent"] for item in results)
    schema_counts = Counter(str(item["actual"]["question_schema"]) for item in results)
    source_counts = Counter(str(item["actual"]["classifier_source"]) for item in results)
    failures = [item for item in results if not item["strict_pass"]]

    report = {
        "metadata": {
            "name": "black_context_understanding_100_20260424",
            "date": "2026-04-24",
            "mode": "actual black classifier/policy stack, KoBART disabled, fake weather/time/news services",
            "env_source": ".env.black.duo.kcbertcpu.broadrebuildv2.local",
            "generation_backend": "template",
            "intent_model_type": config.intent_model_type,
            "kcbert_model_path": config.kcbert_model_path,
            "policy_action_model_path": config.policy_action_model_path,
        },
        "summary": {
            "total": total,
            "strict_pass": strict_passes,
            "strict_fail": total - strict_passes,
            "strict_pass_rate": round(strict_passes / total, 4),
            "action_pass": action_passes,
            "action_fail": total - action_passes,
            "action_pass_rate": round(action_passes / total, 4),
            "by_category": by_category,
            "action_counts": dict(action_counts),
            "intent_counts": dict(intent_counts),
            "schema_counts": dict(schema_counts),
            "classifier_source_counts": dict(source_counts),
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Black Context Understanding 100 - 2026-04-24",
        "",
        "- Scope: black classifier/policy/response-plan context understanding",
        "- Generation: KoBART disabled; template renderer only so generation quality is not scored",
        "- Runtime: KcBERT/Hybrid classifier config from `.env.black.duo.kcbertcpu.broadrebuildv2.local`, fake weather/time/news",
        f"- Strict pass: {strict_passes}/{total} ({strict_passes / total:.1%})",
        f"- Action pass: {action_passes}/{total} ({action_passes / total:.1%})",
        "",
        "## Category Scores",
        "",
        "| Category | Strict | Action |",
        "|---|---:|---:|",
    ]
    for category, stats in by_category.items():
        lines.append(f"| {category} | {stats['strict_pass']}/{stats['total']} | {stats['action_pass']}/{stats['total']} |")
    lines.extend(
        [
            "",
            "## Failure List",
            "",
            "| ID | Category | Text | Expected | Actual | Failed Checks |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in failures:
        failed_checks = ", ".join(key for key, ok in item["checks"].items() if not ok)
        expected_short = json.dumps(item["expect"], ensure_ascii=False)
        actual = item["actual"]
        actual_short = (
            f"intent={actual['intent']}; action={actual['action']}; "
            f"schema={actual['question_schema']}; reason={actual['reason_code']}; "
            f"anchor={actual['response_plan_anchor']}"
        )
        safe_text = item["text"].replace("|", "\\|")
        lines.append(
            f"| {item['id']} | {item['category']} | {safe_text} | "
            f"`{expected_short}` | {actual_short} | {failed_checks} |"
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Actions: `{dict(action_counts)}`",
            f"- Intents: `{dict(intent_counts)}`",
            f"- Schemas: `{dict(schema_counts)}`",
            f"- Classifier sources: `{dict(source_counts)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"JSON={OUT_JSON}")
    print(f"MD={OUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())
