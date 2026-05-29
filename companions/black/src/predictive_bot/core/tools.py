from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Protocol
from xml.etree import ElementTree

import httpx
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from predictive_bot.core.models import WeatherReport


class WeatherLookupError(RuntimeError):
    """Raised when weather lookup fails."""


class KnowledgeLookupError(RuntimeError):
    """Raised when a knowledge lookup request cannot be grounded."""


class NewsLookupError(RuntimeError):
    """Raised when a news lookup request cannot be grounded."""


class RecommendationLookupError(RuntimeError):
    """Raised when a recommendation request cannot be satisfied."""


class KnowledgeService(Protocol):
    def answer(self, question: str) -> KnowledgeAnswer:
        """Return a grounded answer or raise KnowledgeLookupError."""


class TimeService(Protocol):
    def get_current_time(self) -> CurrentTimeAnswer:
        """Return the current local time information."""


class NewsService(Protocol):
    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        """Return current headline items or raise NewsLookupError."""


class RecommendationService(Protocol):
    def recommend_media(
        self,
        *,
        query: str,
        preferences: dict[str, str],
        limit: int = 3,
    ) -> RecommendationAnswer:
        """Return media recommendations grounded in a small curated catalog."""

    def recommend_music(
        self,
        *,
        query: str,
        preferences: dict[str, str],
        limit: int = 3,
    ) -> RecommendationAnswer:
        """Return music recommendations grounded in a small curated catalog."""


@dataclass(slots=True)
class KnowledgeAnswer:
    question: str
    query_type: str
    subject: str
    answer: str
    source: str


@dataclass(slots=True)
class CurrentTimeAnswer:
    formatted_time: str
    formatted_date: str
    timezone_name: str
    source: str


@dataclass(slots=True)
class NewsHeadline:
    title: str
    source: str
    link: str | None = None


@dataclass(slots=True)
class RecommendationItem:
    title: str
    detail: str
    tags: tuple[str, ...]


@dataclass(slots=True)
class RecommendationAnswer:
    focus_label: str | None
    items: list[RecommendationItem]
    source: str


_WEATHER_CODE_MAP = {
    0: "맑음",
    1: "대체로 맑음",
    2: "부분적으로 흐림",
    3: "흐림",
    45: "안개",
    48: "서리 안개",
    51: "가벼운 이슬비",
    53: "이슬비",
    55: "강한 이슬비",
    61: "약한 비",
    63: "비",
    65: "강한 비",
    71: "약한 눈",
    73: "눈",
    75: "강한 눈",
    80: "약한 소나기",
    81: "소나기",
    82: "강한 소나기",
    95: "뇌우",
}

_LOCATION_QUERY_ALIASES = {
    "서울": "Seoul",
    "서울시": "Seoul",
    "서울특별시": "Seoul",
    "부산": "Busan",
    "부산시": "Busan",
    "부산광역시": "Busan",
    "인천": "Incheon",
    "인천시": "Incheon",
    "인천광역시": "Incheon",
    "대구": "Daegu",
    "대구시": "Daegu",
    "대구광역시": "Daegu",
    "대전": "Daejeon",
    "대전시": "Daejeon",
    "대전광역시": "Daejeon",
    "광주": "Gwangju",
    "광주시": "Gwangju",
    "광주광역시": "Gwangju",
    "울산": "Ulsan",
    "울산시": "Ulsan",
    "울산광역시": "Ulsan",
    "제주": "Jeju",
    "제주시": "Jeju",
    "제주도": "Jeju",
}


@dataclass(slots=True)
class GeocodedLocation:
    display_name: str
    latitude: float
    longitude: float


_CAPITAL_ALIASES = {
    "대한민국": "한국",
    "남한": "한국",
    "한국": "한국",
    "미국": "미국",
    "미합중국": "미국",
    "일본": "일본",
    "중국": "중국",
    "영국": "영국",
    "잉글랜드": "영국",
    "프랑스": "프랑스",
    "독일": "독일",
    "이탈리아": "이탈리아",
    "스페인": "스페인",
    "포르투갈": "포르투갈",
    "네덜란드": "네덜란드",
    "벨기에": "벨기에",
    "스위스": "스위스",
    "오스트리아": "오스트리아",
    "폴란드": "폴란드",
    "체코": "체코",
    "그리스": "그리스",
    "헝가리": "헝가리",
    "덴마크": "덴마크",
    "스웨덴": "스웨덴",
    "노르웨이": "노르웨이",
    "핀란드": "핀란드",
    "아일랜드": "아일랜드",
    "러시아": "러시아",
    "우크라이나": "우크라이나",
    "튀르키예": "튀르키예",
    "터키": "튀르키예",
    "인도": "인도",
    "파키스탄": "파키스탄",
    "방글라데시": "방글라데시",
    "태국": "태국",
    "베트남": "베트남",
    "인도네시아": "인도네시아",
    "말레이시아": "말레이시아",
    "필리핀": "필리핀",
    "싱가포르": "싱가포르",
    "호주": "호주",
    "뉴질랜드": "뉴질랜드",
    "캐나다": "캐나다",
    "멕시코": "멕시코",
    "브라질": "브라질",
    "아르헨티나": "아르헨티나",
    "칠레": "칠레",
    "콜롬비아": "콜롬비아",
    "페루": "페루",
    "이집트": "이집트",
    "남아공": "남아프리카공화국",
    "남아프리카공화국": "남아프리카공화국",
    "케냐": "케냐",
    "사우디": "사우디아라비아",
    "사우디아라비아": "사우디아라비아",
    "아랍에미리트": "아랍에미리트",
    "이스라엘": "이스라엘",
}


_COUNTRY_CAPITALS = {
    "한국": "서울",
    "미국": "워싱턴 D.C.",
    "일본": "도쿄",
    "중국": "베이징",
    "영국": "런던",
    "프랑스": "파리",
    "독일": "베를린",
    "이탈리아": "로마",
    "스페인": "마드리드",
    "포르투갈": "리스본",
    "네덜란드": "암스테르담",
    "벨기에": "브뤼셀",
    "스위스": "베른",
    "오스트리아": "빈",
    "폴란드": "바르샤바",
    "체코": "프라하",
    "그리스": "아테네",
    "헝가리": "부다페스트",
    "덴마크": "코펜하겐",
    "스웨덴": "스톡홀름",
    "노르웨이": "오슬로",
    "핀란드": "헬싱키",
    "아일랜드": "더블린",
    "러시아": "모스크바",
    "우크라이나": "키이우",
    "튀르키예": "앙카라",
    "인도": "뉴델리",
    "파키스탄": "이슬라마바드",
    "방글라데시": "다카",
    "태국": "방콕",
    "베트남": "하노이",
    "인도네시아": "자카르타",
    "말레이시아": "쿠알라룸푸르",
    "필리핀": "마닐라",
    "싱가포르": "싱가포르",
    "호주": "캔버라",
    "뉴질랜드": "웰링턴",
    "캐나다": "오타와",
    "멕시코": "멕시코시티",
    "브라질": "브라질리아",
    "아르헨티나": "부에노스아이레스",
    "칠레": "산티아고",
    "콜롬비아": "보고타",
    "페루": "리마",
    "이집트": "카이로",
    "남아프리카공화국": "프리토리아",
    "케냐": "나이로비",
    "사우디아라비아": "리야드",
    "아랍에미리트": "아부다비",
    "이스라엘": "예루살렘",
}


_COUNTRY_FLAGS = {
    "한국": "🇰🇷",
    "미국": "🇺🇸",
    "일본": "🇯🇵",
    "중국": "🇨🇳",
    "영국": "🇬🇧",
    "프랑스": "🇫🇷",
    "독일": "🇩🇪",
    "이탈리아": "🇮🇹",
    "스페인": "🇪🇸",
    "포르투갈": "🇵🇹",
    "네덜란드": "🇳🇱",
    "벨기에": "🇧🇪",
    "스위스": "🇨🇭",
    "오스트리아": "🇦🇹",
    "폴란드": "🇵🇱",
    "체코": "🇨🇿",
    "그리스": "🇬🇷",
    "헝가리": "🇭🇺",
    "덴마크": "🇩🇰",
    "스웨덴": "🇸🇪",
    "노르웨이": "🇳🇴",
    "핀란드": "🇫🇮",
    "아일랜드": "🇮🇪",
    "러시아": "🇷🇺",
    "우크라이나": "🇺🇦",
    "튀르키예": "🇹🇷",
    "인도": "🇮🇳",
    "태국": "🇹🇭",
    "베트남": "🇻🇳",
    "인도네시아": "🇮🇩",
    "말레이시아": "🇲🇾",
    "필리핀": "🇵🇭",
    "싱가포르": "🇸🇬",
    "호주": "🇦🇺",
    "뉴질랜드": "🇳🇿",
    "캐나다": "🇨🇦",
    "멕시코": "🇲🇽",
    "브라질": "🇧🇷",
    "아르헨티나": "🇦🇷",
    "칠레": "🇨🇱",
    "콜롬비아": "🇨🇴",
    "페루": "🇵🇪",
    "이집트": "🇪🇬",
    "남아프리카공화국": "🇿🇦",
    "케냐": "🇰🇪",
    "사우디아라비아": "🇸🇦",
    "아랍에미리트": "🇦🇪",
    "이스라엘": "🇮🇱",
}


_COUNTRY_LOCATIONS = {
    "한국": "동아시아에 있어.",
    "미국": "북아메리카에 있어.",
    "일본": "동아시아에 있어.",
    "중국": "동아시아에 있어.",
    "영국": "서유럽 쪽에 있어.",
    "프랑스": "서유럽에 있어.",
    "독일": "중앙유럽에 있어.",
    "이탈리아": "남유럽에 있어.",
    "스페인": "남서유럽에 있어.",
    "포르투갈": "남서유럽에 있어.",
    "네덜란드": "서유럽에 있어.",
    "벨기에": "서유럽에 있어.",
    "스위스": "중앙유럽에 있어.",
    "오스트리아": "중앙유럽에 있어.",
    "폴란드": "중앙유럽에 있어.",
    "체코": "중앙유럽에 있어.",
    "그리스": "남유럽에 있어.",
    "헝가리": "중앙유럽에 있어.",
    "덴마크": "북유럽에 있어.",
    "스웨덴": "북유럽에 있어.",
    "노르웨이": "북유럽에 있어.",
    "핀란드": "북유럽에 있어.",
    "아일랜드": "서유럽에 있어.",
    "러시아": "동유럽과 북아시아에 걸쳐 있어.",
    "우크라이나": "동유럽에 있어.",
    "튀르키예": "서아시아와 동남유럽에 걸쳐 있어.",
    "인도": "남아시아에 있어.",
    "태국": "동남아시아에 있어.",
    "베트남": "동남아시아에 있어.",
    "인도네시아": "동남아시아에 있어.",
    "말레이시아": "동남아시아에 있어.",
    "필리핀": "동남아시아에 있어.",
    "싱가포르": "동남아시아에 있어.",
    "호주": "오세아니아에 있어.",
    "뉴질랜드": "오세아니아에 있어.",
    "캐나다": "북아메리카에 있어.",
    "멕시코": "북아메리카에 있어.",
    "브라질": "남아메리카에 있어.",
    "아르헨티나": "남아메리카에 있어.",
    "칠레": "남아메리카에 있어.",
    "콜롬비아": "남아메리카 북서쪽에 있어.",
    "페루": "남아메리카 서쪽에 있어.",
    "이집트": "북아프리카에 있어.",
    "남아프리카공화국": "남아프리카에 있어.",
    "케냐": "동아프리카에 있어.",
    "사우디아라비아": "서아시아에 있어.",
    "아랍에미리트": "서아시아에 있어.",
    "이스라엘": "서아시아에 있어.",
}


class BasicKnowledgeService:
    """Small deterministic knowledge tool for common factual questions."""

    _capital_pattern = re.compile(r"(?P<country>.+?)의\s*수도는\s*\??$")
    _flag_pattern = re.compile(r"(?P<country>.+?)의\s*국기는\s*\??$")
    _location_pattern = re.compile(r"(?P<country>.+?)의\s*위치는\s*\??$")

    def __init__(self, fallback_service: KnowledgeService | None = None) -> None:
        self.fallback_service = fallback_service

    def answer(self, question: str) -> KnowledgeAnswer:
        normalized_question = question.strip()

        capital_answer = self._lookup_capital(normalized_question)
        if capital_answer is not None:
            subject, answer = capital_answer
            return KnowledgeAnswer(
                question=normalized_question,
                query_type="capital",
                subject=subject,
                answer=answer,
                source="builtin_country_capitals",
            )

        flag_answer = self._lookup_flag(normalized_question)
        if flag_answer is not None:
            subject, answer = flag_answer
            return KnowledgeAnswer(
                question=normalized_question,
                query_type="flag",
                subject=subject,
                answer=answer,
                source="builtin_country_flags",
            )

        location_answer = self._lookup_location(normalized_question)
        if location_answer is not None:
            subject, answer = location_answer
            return KnowledgeAnswer(
                question=normalized_question,
                query_type="location",
                subject=subject,
                answer=answer,
                source="builtin_country_locations",
            )

        if self.fallback_service is not None:
            return self.fallback_service.answer(normalized_question)

        raise KnowledgeLookupError(f"Could not ground question: {question}")

    def _lookup_capital(self, question: str) -> tuple[str, str] | None:
        match = self._capital_pattern.search(question)
        if not match:
            return None

        raw_country = match.group("country").strip(" ?!.")
        if not raw_country:
            return None

        canonical_country = _CAPITAL_ALIASES.get(raw_country, raw_country)
        capital = _COUNTRY_CAPITALS.get(canonical_country)
        if capital is None:
            return None

        return canonical_country, capital

    def _lookup_flag(self, question: str) -> tuple[str, str] | None:
        match = self._flag_pattern.search(question)
        if not match:
            return None

        raw_country = match.group("country").strip(" ?!.")
        canonical_country = _CAPITAL_ALIASES.get(raw_country, raw_country)
        flag = _COUNTRY_FLAGS.get(canonical_country)
        if flag is None:
            return None
        return canonical_country, flag

    def _lookup_location(self, question: str) -> tuple[str, str] | None:
        match = self._location_pattern.search(question)
        if not match:
            return None

        raw_country = match.group("country").strip(" ?!.")
        canonical_country = _CAPITAL_ALIASES.get(raw_country, raw_country)
        location = _COUNTRY_LOCATIONS.get(canonical_country)
        if location is None:
            return None
        return canonical_country, location


class SystemTimeService:
    """Simple local time backend backed by system clock and zoneinfo."""

    def __init__(self, timezone_name: str = "Asia/Seoul") -> None:
        self.timezone_name = timezone_name

    def get_current_time(self) -> CurrentTimeAnswer:
        try:
            zone = ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            zone = datetime.now().astimezone().tzinfo
        now = datetime.now(zone)
        return CurrentTimeAnswer(
            formatted_time=now.strftime("%H:%M"),
            formatted_date=now.strftime("%Y-%m-%d"),
            timezone_name=str(getattr(zone, "key", None) or self.timezone_name),
            source="system_clock",
        )


class GoogleNewsRssService:
    """Lightweight current-news backend using Google News RSS without API keys."""

    _default_feed_url = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"

    def __init__(
        self,
        *,
        feed_url: str | None = None,
        user_agent: str = "predictive-discord-bot/0.1",
        timeout_seconds: float = 8.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.feed_url = feed_url or self._default_feed_url
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def top_headlines(self, *, limit: int = 3) -> list[NewsHeadline]:
        headers = {"User-Agent": self.user_agent}
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                headers=headers,
                follow_redirects=True,
                transport=self.transport,
            ) as client:
                response = client.get(self.feed_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise NewsLookupError(f"News RSS fetch failed: {exc}") from exc

        try:
            root = ElementTree.fromstring(response.text)
        except ElementTree.ParseError as exc:
            raise NewsLookupError(f"News RSS parse failed: {exc}") from exc

        items = root.findall("./channel/item")
        headlines: list[NewsHeadline] = []
        for item in items[:limit]:
            raw_title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip() or None
            title, source = self._split_title_and_source(raw_title)
            if not title:
                continue
            headlines.append(
                NewsHeadline(
                    title=title,
                    source=source or "Google News",
                    link=link,
                )
            )

        if not headlines:
            raise NewsLookupError("No RSS headlines were available")

        return headlines

    @staticmethod
    def _split_title_and_source(raw_title: str) -> tuple[str, str | None]:
        if " - " not in raw_title:
            return raw_title, None
        title, source = raw_title.rsplit(" - ", 1)
        return title.strip(), source.strip() or None


class CuratedRecommendationService:
    """Small deterministic recommendation backend for media and music."""

    _media_catalog: tuple[RecommendationItem, ...] = (
        RecommendationItem("나이브스 아웃", "가볍게 몰입되는 추리물이라 텐션이 좋다.", ("추리", "미스터리", "가벼운")),
        RecommendationItem("슬기로운 의사생활", "잔잔하고 따뜻한 관계 드라마 쪽이다.", ("힐링", "드라마", "잔잔")),
        RecommendationItem("스파이더맨: 뉴 유니버스", "리듬감 좋은 액션 애니다.", ("애니", "액션", "신나는")),
        RecommendationItem("브루클린 나인-나인", "가볍게 보기 좋은 코미디 시리즈다.", ("코미디", "가벼운", "시리즈")),
        RecommendationItem("이상한 변호사 우영우", "부담 적게 보기 좋은 따뜻한 드라마다.", ("힐링", "드라마", "가벼운")),
        RecommendationItem("크래시 랜딩 온 유", "몰입 잘 되는 로맨스 드라마 쪽이다.", ("로맨스", "드라마", "감정")),
        RecommendationItem("듄", "큰 화면 감각이 좋은 묵직한 SF다.", ("sf", "판타지", "무거운")),
        RecommendationItem("곡성", "한국 오컬트 공포라 분위기 몰입감이 세다.", ("공포", "스릴러", "한국", "무거운")),
        RecommendationItem("겟 아웃", "심리적으로 조여오는 공포라 깔끔하게 보기 좋다.", ("공포", "스릴러", "심리")),
        RecommendationItem("콰이어트 플레이스", "긴장감으로 밀어붙이는 생존형 공포다.", ("공포", "스릴러", "서바이벌")),
        RecommendationItem("블레이드 러너 2049", "느리지만 분위기 진한 SF 쪽이다.", ("sf", "느린", "무거운")),
        RecommendationItem("웬즈데이", "다크한 톤인데 보기 쉽게 넘어간다.", ("판타지", "다크", "가벼운")),
    )
    _music_catalog: tuple[RecommendationItem, ...] = (
        RecommendationItem("AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지", "잔잔하게 오래 가는 발라드 쪽이다.", ("잔잔", "발라드", "감성")),
        RecommendationItem("잔나비 - 주저하는 연인들을 위해", "감성 인디 쪽으로 무드가 부드럽다.", ("잔잔", "인디", "감성")),
        RecommendationItem("검정치마 - 기다린 만큼, 더", "새벽에 듣기 좋은 인디 락 결이다.", ("인디", "새벽", "잔잔")),
        RecommendationItem("10CM - 폰서트", "가볍게 듣기 좋은 포근한 톤이다.", ("가벼운", "잔잔", "어쿠스틱")),
        RecommendationItem("DAY6 - 예뻤어", "멜로디 힘 있는 밴드 발라드다.", ("밴드", "발라드", "감정")),
        RecommendationItem("혁오 - Tomboy", "질감 있는 인디 락 쪽이다.", ("인디", "락", "드라이브")),
        RecommendationItem("NewJeans - Ditto", "부드럽게 반복해서 듣기 좋은 팝이다.", ("팝", "잔잔", "가벼운")),
        RecommendationItem("이무진 - 비와 당신", "비 오는 날 무드에 잘 맞는다.", ("비", "잔잔", "감성")),
        RecommendationItem("Epik High - 우산", "비 오는 날 감성 힙합 쪽이다.", ("비", "힙합", "감성")),
        RecommendationItem("NELL - 기억을 걷는 시간", "새벽 감성에 잘 붙는 밴드 사운드다.", ("새벽", "잔잔", "밴드")),
        RecommendationItem("The Chainsmokers - Something Just Like This", "드라이브할 때 무난하게 올리기 좋다.", ("드라이브", "팝", "신나는")),
        RecommendationItem("DPR LIVE - Jasmine", "리듬감 있는 힙합 쪽으로 가볍게 듣기 좋다.", ("힙합", "드라이브", "가벼운")),
    )
    _media_tag_keywords = {
        "공포": ("공포", "호러", "무서운"),
        "스릴러": ("스릴러", "긴장감"),
        "추리": ("추리", "미스터리", "범인", "수사"),
        "힐링": ("힐링", "잔잔", "따뜻", "편한"),
        "로맨스": ("로맨스", "연애", "달달"),
        "sf": ("sf", "sci-fi", "우주", "미래"),
        "판타지": ("판타지", "마법", "다크"),
        "애니": ("애니", "애니메이션", "만화"),
        "액션": ("액션", "시원한", "전투"),
        "코미디": ("코미디", "웃긴", "개그"),
        "가벼운": ("가벼운", "편하게", "킬링타임"),
        "무거운": ("무거운", "진한", "묵직한"),
    }
    _music_tag_keywords = {
        "잔잔": ("잔잔", "조용한", "부드러운", "차분한"),
        "발라드": ("발라드",),
        "인디": ("인디",),
        "락": ("락", "록", "밴드"),
        "팝": ("팝", "pop"),
        "힙합": ("힙합", "랩"),
        "비": ("비", "비오는", "비 올 때"),
        "새벽": ("새벽", "밤", "늦은 밤"),
        "드라이브": ("드라이브", "운전"),
        "신나는": ("신나는", "업템포", "텐션 올리는"),
        "감성": ("감성", "감정", "쓸쓸"),
        "가벼운": ("가벼운", "편한", "무난한"),
    }

    def recommend_media(
        self,
        *,
        query: str,
        preferences: dict[str, str],
        limit: int = 3,
    ) -> RecommendationAnswer:
        focus_label = self._derive_focus_label(
            query=query,
            preferences=preferences,
            positive_key="media_like",
            keyword_map=self._media_tag_keywords,
        )
        items = self._rank_items(
            catalog=self._media_catalog,
            positive_hint=self._combine_hint(query, preferences.get("media_like")),
            negative_hint=preferences.get("media_dislike"),
            keyword_map=self._media_tag_keywords,
            limit=limit,
        )
        if not items:
            raise RecommendationLookupError("No media recommendations were available")
        return RecommendationAnswer(
            focus_label=focus_label,
            items=items,
            source="curated_media_catalog",
        )

    def recommend_music(
        self,
        *,
        query: str,
        preferences: dict[str, str],
        limit: int = 3,
    ) -> RecommendationAnswer:
        focus_label = self._derive_focus_label(
            query=query,
            preferences=preferences,
            positive_key="music_like",
            keyword_map=self._music_tag_keywords,
        )
        items = self._rank_items(
            catalog=self._music_catalog,
            positive_hint=self._combine_hint(query, preferences.get("music_like")),
            negative_hint=preferences.get("music_dislike"),
            keyword_map=self._music_tag_keywords,
            limit=limit,
        )
        if not items:
            raise RecommendationLookupError("No music recommendations were available")
        return RecommendationAnswer(
            focus_label=focus_label,
            items=items,
            source="curated_music_catalog",
        )

    @staticmethod
    def _combine_hint(query: str, preference: str | None) -> str:
        if preference:
            return f"{query} {preference}"
        return query

    @classmethod
    def _derive_focus_label(
        cls,
        *,
        query: str,
        preferences: dict[str, str],
        positive_key: str,
        keyword_map: dict[str, tuple[str, ...]],
    ) -> str | None:
        remembered = preferences.get(positive_key)
        if remembered:
            return remembered
        tags = cls._extract_tags(query, keyword_map)
        if tags:
            return tags[0]
        return None

    @classmethod
    def _rank_items(
        cls,
        *,
        catalog: tuple[RecommendationItem, ...],
        positive_hint: str,
        negative_hint: str | None,
        keyword_map: dict[str, tuple[str, ...]],
        limit: int,
    ) -> list[RecommendationItem]:
        preferred_tags = cls._extract_tags(positive_hint, keyword_map)
        disliked_tags = cls._extract_tags(negative_hint or "", keyword_map)

        scored: list[tuple[int, RecommendationItem]] = []
        for item in catalog:
            score = 0
            item_tags = set(item.tags)
            if preferred_tags:
                score += sum(3 for tag in preferred_tags if tag in item_tags)
            if disliked_tags:
                score -= sum(4 for tag in disliked_tags if tag in item_tags)
            if not preferred_tags and not disliked_tags:
                score += 1
            scored.append((score, item))

        ranked = [
            item
            for score, item in sorted(
                scored,
                key=lambda entry: entry[0],
                reverse=True,
            )
            if score >= 0
        ]
        if not ranked:
            ranked = list(catalog)
        return ranked[:limit]

    @classmethod
    def _extract_tags(
        cls,
        text: str,
        keyword_map: dict[str, tuple[str, ...]],
    ) -> list[str]:
        normalized = text.lower().strip()
        if not normalized:
            return []
        tags: list[str] = []
        for tag, keywords in keyword_map.items():
            if any(keyword in normalized for keyword in keywords):
                tags.append(tag)
        return tags


class WikidataKnowledgeService:
    """External factual backend for searchable common-knowledge questions."""

    _capital_pattern = re.compile(r"(?P<subject>.+?)의\s*수도는\s*\??$")
    _population_pattern = re.compile(r"(?P<subject>.+?)의\s*인구는\s*\??$")
    _area_pattern = re.compile(r"(?P<subject>.+?)의\s*면적은\s*\??$")
    _president_pattern = re.compile(r"(?P<subject>.+?)의\s*대통령은\s*\??$")
    _prime_minister_pattern = re.compile(r"(?P<subject>.+?)의\s*총리는\s*\??$")
    _description_patterns = (
        re.compile(r"(?P<subject>.+?)(?:은|는|이|가)?\s*뭐야\s*\??$"),
        re.compile(r"(?P<subject>.+?)(?:은|는|이|가)?\s*누구야\s*\??$"),
    )

    _search_url = "https://www.wikidata.org/w/api.php"
    _entity_url = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def answer(self, question: str) -> KnowledgeAnswer:
        normalized_question = question.strip()

        capital_subject = self._extract_capital_subject(normalized_question)
        if capital_subject is not None:
            return self._answer_capital(normalized_question, capital_subject)

        population_subject = self._extract_population_subject(normalized_question)
        if population_subject is not None:
            return self._answer_population(normalized_question, population_subject)

        area_subject = self._extract_area_subject(normalized_question)
        if area_subject is not None:
            return self._answer_area(normalized_question, area_subject)

        president_subject = self._extract_president_subject(normalized_question)
        if president_subject is not None:
            return self._answer_president(normalized_question, president_subject)

        prime_minister_subject = self._extract_prime_minister_subject(normalized_question)
        if prime_minister_subject is not None:
            return self._answer_prime_minister(normalized_question, prime_minister_subject)

        description_subject = self._extract_description_subject(normalized_question)
        if description_subject is not None:
            return self._answer_description(normalized_question, description_subject)

        raise KnowledgeLookupError(f"Wikidata backend does not support question: {question}")

    def _answer_capital(self, question: str, subject: str) -> KnowledgeAnswer:
        entity = self._search_entity(subject)
        entity_id = entity["id"]
        claims = self._fetch_entity_claims(entity_id)
        capital_claims = claims.get("P36") or []
        if not capital_claims:
            raise KnowledgeLookupError(f"No capital claim found for {subject}")

        capital_id = capital_claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
        if not capital_id:
            raise KnowledgeLookupError(f"Capital entity missing for {subject}")

        capital_label = self._fetch_entity_label(capital_id)
        subject_label = self._fetch_entity_label(entity_id)
        return KnowledgeAnswer(
            question=question,
            query_type="capital",
            subject=subject_label,
            answer=capital_label,
            source="wikidata_capital",
        )

    def _answer_description(self, question: str, subject: str) -> KnowledgeAnswer:
        entity = self._search_entity(subject)
        entity_id = entity["id"]
        label = self._fetch_entity_label(entity_id)
        description = self._fetch_entity_description(entity_id)
        if not description:
            description = (
                entity.get("description")
                or entity.get("display", {}).get("description", {}).get("value")
            )
        if not description:
            raise KnowledgeLookupError(f"No description found for {subject}")

        return KnowledgeAnswer(
            question=question,
            query_type="description",
            subject=label,
            answer=f"{label}{self._topic_particle(label)} {description}{self._copula_suffix(description)}",
            source="wikidata_description",
        )

    def _answer_population(self, question: str, subject: str) -> KnowledgeAnswer:
        entity = self._search_entity(subject)
        entity_id = entity["id"]
        label = self._fetch_entity_label(entity_id)
        population = self._fetch_quantity_claim(entity_id, "P1082")
        return KnowledgeAnswer(
            question=question,
            query_type="population",
            subject=label,
            answer=f"{label}의 인구는 약 {self._format_number(population)}명으로 기록돼 있어.",
            source="wikidata_population",
        )

    def _answer_area(self, question: str, subject: str) -> KnowledgeAnswer:
        entity = self._search_entity(subject)
        entity_id = entity["id"]
        label = self._fetch_entity_label(entity_id)
        area = self._fetch_quantity_claim(entity_id, "P2046")
        formatted_area = self._format_number(area, decimal_places=1 if not float(area).is_integer() else 0)
        return KnowledgeAnswer(
            question=question,
            query_type="area",
            subject=label,
            answer=f"{label}의 면적은 약 {formatted_area}km²로 볼 수 있어.",
            source="wikidata_area",
        )

    def _answer_president(self, question: str, subject: str) -> KnowledgeAnswer:
        entity = self._search_entity(subject)
        entity_id = entity["id"]
        label = self._fetch_entity_label(entity_id)
        president_id = self._fetch_entity_claim_id(entity_id, "P35")
        president_name = self._fetch_entity_label(president_id)
        return KnowledgeAnswer(
            question=question,
            query_type="head_of_state",
            subject=label,
            answer=f"{label}의 국가원수는 {president_name}야.",
            source="wikidata_head_of_state",
        )

    def _answer_prime_minister(self, question: str, subject: str) -> KnowledgeAnswer:
        entity = self._search_entity(subject)
        entity_id = entity["id"]
        label = self._fetch_entity_label(entity_id)
        prime_minister_id = self._fetch_entity_claim_id(entity_id, "P6")
        prime_minister_name = self._fetch_entity_label(prime_minister_id)
        return KnowledgeAnswer(
            question=question,
            query_type="head_of_government",
            subject=label,
            answer=f"{label}의 정부수반은 {prime_minister_name}야.",
            source="wikidata_head_of_government",
        )

    def _search_entity(self, subject: str) -> dict:
        payload = self._request_json(
            self._search_url,
            params={
                "action": "wbsearchentities",
                "search": subject,
                "language": "ko",
                "format": "json",
                "limit": 1,
            },
        )
        results = payload.get("search") or []
        if not results:
            raise KnowledgeLookupError(f"No Wikidata search result for {subject}")
        return results[0]

    def _fetch_entity_claims(self, entity_id: str) -> dict:
        payload = self._request_json(self._entity_url.format(entity_id=entity_id))
        entity = payload.get("entities", {}).get(entity_id)
        if not entity:
            raise KnowledgeLookupError(f"Missing Wikidata entity payload for {entity_id}")
        return entity.get("claims") or {}

    def _fetch_entity_claim_id(self, entity_id: str, property_id: str) -> str:
        claims = self._fetch_entity_claims(entity_id)
        property_claims = claims.get(property_id) or []
        if not property_claims:
            raise KnowledgeLookupError(f"Missing property {property_id} for {entity_id}")

        best_claim = self._select_best_claim(property_claims, prefer_active=True)
        value = best_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        linked_id = value.get("id")
        if not linked_id:
            raise KnowledgeLookupError(f"Property {property_id} for {entity_id} did not contain entity id")
        return linked_id

    def _fetch_quantity_claim(self, entity_id: str, property_id: str) -> float:
        claims = self._fetch_entity_claims(entity_id)
        property_claims = claims.get(property_id) or []
        if not property_claims:
            raise KnowledgeLookupError(f"Missing property {property_id} for {entity_id}")

        best_claim = self._select_best_claim(property_claims, prefer_active=False)
        value = best_claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        amount = value.get("amount")
        if amount is None:
            raise KnowledgeLookupError(f"Property {property_id} for {entity_id} did not contain quantity")
        return float(str(amount).lstrip("+"))

    def _fetch_entity_label(self, entity_id: str) -> str:
        entity = self._fetch_entity_metadata(entity_id)
        labels = entity.get("labels") or {}
        ko_label = labels.get("ko", {}).get("value")
        en_label = labels.get("en", {}).get("value")
        label = ko_label or en_label
        if not label:
            raise KnowledgeLookupError(f"Missing label for {entity_id}")
        return label

    def _fetch_entity_description(self, entity_id: str) -> str | None:
        entity = self._fetch_entity_metadata(entity_id)
        descriptions = entity.get("descriptions") or {}
        return descriptions.get("ko", {}).get("value") or descriptions.get("en", {}).get("value")

    def _fetch_entity_metadata(self, entity_id: str) -> dict:
        payload = self._request_json(
            self._search_url,
            params={
                "action": "wbgetentities",
                "ids": entity_id,
                "languages": "ko|en",
                "format": "json",
                "props": "labels|descriptions",
            },
        )
        entity = payload.get("entities", {}).get(entity_id)
        if not entity:
            raise KnowledgeLookupError(f"Missing metadata for {entity_id}")
        return entity

    def _request_json(self, url: str, params: dict | None = None) -> dict:
        headers = {"User-Agent": self.user_agent}
        with httpx.Client(
            timeout=self.timeout_seconds,
            headers=headers,
            transport=self.transport,
            follow_redirects=True,
        ) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _best_search_label(entity: dict, *, fallback: str) -> str:
        display_label = entity.get("display", {}).get("label", {}).get("value")
        return display_label or entity.get("label") or fallback

    @classmethod
    def _extract_capital_subject(cls, question: str) -> str | None:
        match = cls._capital_pattern.search(question)
        if not match:
            return None
        return match.group("subject").strip(" ?!.") or None

    @classmethod
    def _extract_population_subject(cls, question: str) -> str | None:
        match = cls._population_pattern.search(question)
        if not match:
            return None
        return match.group("subject").strip(" ?!.") or None

    @classmethod
    def _extract_area_subject(cls, question: str) -> str | None:
        match = cls._area_pattern.search(question)
        if not match:
            return None
        return match.group("subject").strip(" ?!.") or None

    @classmethod
    def _extract_president_subject(cls, question: str) -> str | None:
        match = cls._president_pattern.search(question)
        if not match:
            return None
        return match.group("subject").strip(" ?!.") or None

    @classmethod
    def _extract_prime_minister_subject(cls, question: str) -> str | None:
        match = cls._prime_minister_pattern.search(question)
        if not match:
            return None
        return match.group("subject").strip(" ?!.") or None

    @classmethod
    def _extract_description_subject(cls, question: str) -> str | None:
        for pattern in cls._description_patterns:
            match = pattern.search(question)
            if match:
                subject = match.group("subject").strip(" ?!.")
                if subject and len(subject) >= 2:
                    return subject
        return None

    @staticmethod
    def _topic_particle(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "는"

        code = ord(stripped[-1])
        if 0xAC00 <= code <= 0xD7A3:
            return "은" if (code - 0xAC00) % 28 != 0 else "는"
        return "은"

    @staticmethod
    def _copula_suffix(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "이야."

        code = ord(stripped[-1])
        if 0xAC00 <= code <= 0xD7A3:
            return "이야." if (code - 0xAC00) % 28 != 0 else "야."
        return "야."

    @staticmethod
    def _format_number(value: float, decimal_places: int = 0) -> str:
        if decimal_places <= 0:
            return f"{int(round(value)):,}"
        return f"{value:,.{decimal_places}f}"

    @classmethod
    def _select_best_claim(cls, claims: list[dict], *, prefer_active: bool) -> dict:
        usable_claims = [claim for claim in claims if claim.get("rank") != "deprecated"]
        if not usable_claims:
            raise KnowledgeLookupError("No usable claims available")
        return max(usable_claims, key=lambda claim: cls._claim_sort_key(claim, prefer_active=prefer_active))

    @classmethod
    def _claim_sort_key(cls, claim: dict, *, prefer_active: bool) -> tuple[int, int, str]:
        rank = claim.get("rank")
        rank_weight = 2 if rank == "preferred" else 1
        active_weight = 0
        if prefer_active and not cls._get_qualifier_time(claim, "P582"):
            active_weight = 1
        relevant_time = (
            cls._get_qualifier_time(claim, "P585")
            or cls._get_qualifier_time(claim, "P580")
            or cls._get_qualifier_time(claim, "P582")
            or ""
        )
        return (rank_weight, active_weight, relevant_time)

    @staticmethod
    def _get_qualifier_time(claim: dict, property_id: str) -> str | None:
        qualifiers = claim.get("qualifiers") or {}
        entries = qualifiers.get(property_id) or []
        if not entries:
            return None
        return entries[0].get("datavalue", {}).get("value", {}).get("time")


class OpenMeteoWeatherService:
    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def get_current_weather(self, location: str) -> WeatherReport:
        geocoded = await self._geocode(location)
        weather = await self._fetch_weather(geocoded)
        return WeatherReport(
            location=geocoded.display_name,
            temperature_c=weather["temperature_2m"],
            description=_WEATHER_CODE_MAP.get(weather["weather_code"], "알 수 없음"),
            wind_kph=weather["wind_speed_10m"],
        )

    async def _geocode(self, location: str) -> GeocodedLocation:
        query = self._normalize_location_query(location)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": query, "count": 1, "language": "ko", "format": "json"},
            )
            response.raise_for_status()
            payload = response.json()

        results = payload.get("results") or []
        if not results:
            raise WeatherLookupError(f"Could not geocode location: {location}")

        top = results[0]
        return GeocodedLocation(
            display_name=top.get("name", location),
            latitude=top["latitude"],
            longitude=top["longitude"],
        )

    async def _fetch_weather(self, location: GeocodedLocation) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                },
            )
            response.raise_for_status()
            payload = response.json()

        current = payload.get("current")
        if not current:
            raise WeatherLookupError("Weather API did not return current weather.")
        return current

    @staticmethod
    def _normalize_location_query(location: str) -> str:
        normalized = location.strip()
        return _LOCATION_QUERY_ALIASES.get(normalized, normalized)
