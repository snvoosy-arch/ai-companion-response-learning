from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
SCRIPT_ROOT = ROOT / "scripts"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from build_black_rejected_generation_sft_20260427 import (  # type: ignore
    _compact,
    _message_row,
    _normalize_for_echo,
    _target_copy_score,
    write_jsonl,
)


DEFAULT_PREFIX = "black_qwen_clean_rewrite_gold_v1_20260429"
DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_REPORT_DIR = ROOT / "reports"
MAX_COPY_SCORE = 0.94
MIN_COPY_SCORE = 0.35


@dataclass(frozen=True, slots=True)
class CleanRewriteCase:
    action: str
    user_text: str
    draft: str
    target: str
    stance: str = "clean_surface_rewrite"
    anchor: str = ""
    must_include: tuple[str, ...] = ()
    tone: str = "steady"
    source_family: str = "general"


def _has_malformed_surface_text(text: str) -> bool:
    raw = str(text or "")
    if "\ufffd" in raw:
        return True
    if re.search(r"[\u4e00-\u9fff]", raw):
        return True
    if any(re.search(r"[a-z]", token) and re.search(r"[가-힣]", token) for token in re.findall(r"[A-Za-z가-힣]+", raw)):
        return True
    return bool(re.search(r"\s(?:라|임|네)\s*(?:[.!?。]|$)", _compact(raw)))


def _has_polite_style(text: str) -> bool:
    return bool(re.search(r"(요|죠|니다|세요|습니까|나요|까요)\s*(?:[.!?。]|$)", _compact(text)))


def _has_internal_label(text: str) -> bool:
    normalized = _normalize_for_echo(text)
    labels = (
        "activityrecommendation",
        "activityinvite",
        "schemaactivity",
        "opiniondecisionrequest",
        "quietmode",
        "task",
        "persona",
        "intent",
        "draftutterance",
        "responseplan",
    )
    return any(label in normalized for label in labels)


def _quality_issues(case: CleanRewriteCase) -> list[str]:
    issues: list[str] = []
    if not case.draft or not case.target:
        issues.append("missing_pair")
    if case.draft == case.target:
        issues.append("exact_copy")
    if _has_polite_style(case.target):
        issues.append("polite_target")
    if _has_internal_label(case.draft) or _has_internal_label(case.target):
        issues.append("internal_label")
    if _has_malformed_surface_text(case.draft) or _has_malformed_surface_text(case.target):
        issues.append("malformed_surface")
    if len(case.target) < 5 or len(case.target) > 120:
        issues.append("target_length")
    score = _target_copy_score(target=case.target, draft_reply=case.draft)
    if score >= MAX_COPY_SCORE:
        issues.append("near_copy")
    if score <= MIN_COPY_SCORE:
        issues.append("semantic_distance_too_far")
    required_terms = [case.anchor, *case.must_include]
    normalized_target = _normalize_for_echo(case.target)
    for term in required_terms:
        normalized = _normalize_for_echo(term)
        if normalized and normalized not in normalized_target:
            issues.append(f"missing_required:{term}")
            break
    return list(dict.fromkeys(issues))


def _case(
    *,
    action: str,
    user_text: str,
    draft: str,
    target: str,
    stance: str = "clean_surface_rewrite",
    anchor: str = "",
    must_include: Iterable[str] = (),
    tone: str = "steady",
    source_family: str = "general",
) -> CleanRewriteCase:
    return CleanRewriteCase(
        action=action,
        user_text=_compact(user_text),
        draft=_compact(draft),
        target=_compact(target),
        stance=stance,
        anchor=_compact(anchor),
        must_include=tuple(_compact(item) for item in must_include if _compact(item)),
        tone=tone,
        source_family=source_family,
    )


def _activity_invite_cases() -> list[CleanRewriteCase]:
    rows: list[CleanRewriteCase] = []
    slots = [
        ("바다", "수영", "물만 너무 차갑지 않으면", "오늘 바다가 시원한데 수영이나 하자"),
        ("계곡", "물놀이", "물살만 너무 세지 않으면", "계곡에서 물놀이하자"),
        ("캠핑장", "바베큐", "불만 안전하게 보면", "캠핑하면서 바베큐 구워먹자"),
        ("공원", "산책", "날만 너무 덥지 않으면", "공원에서 산책하자"),
        ("집", "보드게임", "시간만 너무 길게 안 가면", "집에서 보드게임 하자"),
        ("노래방", "노래", "목만 무리하지 않으면", "노래방 가서 노래하자"),
        ("카페", "수다", "자리만 조용하면", "카페에서 수다 떨자"),
        ("한강", "자전거", "바람만 너무 세지 않으면", "한강에서 자전거 타자"),
        ("해변", "사진 찍기", "사람만 너무 많지 않으면", "해변에서 사진 찍자"),
        ("방", "영화 보기", "너무 늦지만 않으면", "방에서 영화 보자"),
        ("PC방", "협동 게임", "판만 너무 길지 않으면", "피시방에서 협동 게임 하자"),
        ("시장", "간식 먹기", "배만 너무 부르지 않으면", "시장 가서 간식 먹자"),
    ]
    for place, activity, condition, user_text in slots:
        rows.append(
            _case(
                action="accept_activity_invite",
                user_text=user_text,
                draft=f"{place} {activity} 좋지. {condition} 가볍게 하자.",
                target=f"{place}에서 {activity}이면 좋지. {condition} 짧게 해보자.",
                stance="accept_activity_invite",
                anchor=activity,
                must_include=(place, activity),
                source_family="activity_invite",
            )
        )
        rows.append(
            _case(
                action="accept_activity_invite",
                user_text=user_text,
                draft=f"{activity} 좋지. 부담 없으면 가볍게 하자.",
                target=f"{activity}이면 괜찮지. 무리만 없으면 해보자.",
                stance="accept_activity_invite",
                anchor=activity,
                must_include=(activity,),
                source_family="activity_invite",
            )
        )
    return rows


def _activity_recommendation_cases() -> list[CleanRewriteCase]:
    rows: list[CleanRewriteCase] = []
    slots = [
        ("바다", "물놀이", "모래사장 산책", "사진 찍기", "바다에서 뭐하고 놀면 좋을까?"),
        ("캠핑장", "불멍", "간단한 요리", "산책", "캠핑장에선 뭐하면 좋을까?"),
        ("계곡", "발 담그기", "물놀이", "그늘에서 쉬기", "계곡에서 할 만한 거 말해봐"),
        ("집", "가벼운 게임", "영화 보기", "간식 만들기", "집에서 뭐하면서 놀까?"),
        ("카페", "짧은 수다", "디저트 고르기", "책 보기", "카페에서 뭘 하면 좋을까?"),
        ("공원", "산책", "돗자리 쉬기", "사진 찍기", "공원에서 할만한 거 추천해줘"),
        ("여행지", "동선 줄이기", "맛집 하나", "야경 보기", "여행 가면 뭐부터 할까?"),
        ("겨울 바다", "바람 맞기", "따뜻한 음식", "사진 찍기", "겨울 바다 가면 뭐하면 좋을까?"),
        ("밤바다", "산책", "파도 소리 듣기", "따뜻한 음료", "밤바다에서 뭐하면 좋을까?"),
        ("놀이공원", "가벼운 놀이기구", "간식", "퍼레이드 보기", "놀이공원에서 뭐부터 하지?"),
    ]
    for place, first, second, third, user_text in slots:
        rows.append(
            _case(
                action="share_opinion",
                user_text=user_text,
                draft=f"{place} 놀이면 {first}랑 {second}이 무난해. 여유 있으면 {third}도 좋아.",
                target=f"{place}에서는 {first}랑 {second}부터 무난하지. 여유 남으면 {third}도 괜찮아.",
                stance="practical_activity_recommendation",
                anchor=place,
                must_include=(place, first, second),
                source_family="activity_recommendation",
            )
        )
        rows.append(
            _case(
                action="share_opinion",
                user_text=user_text,
                draft=f"{place}에서는 {first}부터 떠올라. 너무 빡빡하게 잡을 필요는 없어.",
                target=f"{place}면 {first}부터 보는 쪽이 맞아. 너무 빡빡하게 잡진 말자.",
                stance="practical_activity_recommendation",
                anchor=place,
                must_include=(place, first),
                source_family="activity_recommendation",
            )
        )
    return rows


def _preparation_cases() -> list[CleanRewriteCase]:
    rows: list[CleanRewriteCase] = []
    slots = [
        ("등산", "얇은 겉옷", "편한 신발", "간식", "보조배터리", "등산 할 때 필요한 거 말해봐"),
        ("캠핑", "물티슈", "조명", "여벌옷", "쓰레기봉투", "캠핑 갈 때 뭘 챙겨야 해?"),
        ("바다", "수건", "여벌옷", "슬리퍼", "방수팩", "바다 갈 때 필요한 거 알려줘"),
        ("계곡", "아쿠아슈즈", "수건", "간식", "방수팩", "계곡 갈 때 뭐 챙기지?"),
        ("여행", "충전기", "신분증", "여벌옷", "상비약", "여행 준비물 뭐가 필요해?"),
        ("운동", "물", "수건", "편한 신발", "가벼운 간식", "운동하러 갈 때 뭐 챙겨?"),
        ("소풍", "돗자리", "물", "간식", "휴지", "소풍 갈 때 필요한 거 말해봐"),
        ("노래방", "물", "목캔디", "충전기", "가벼운 겉옷", "노래방 갈 때 뭐 챙기면 돼?"),
    ]
    for activity, item1, item2, item3, item4, user_text in slots:
        rows.append(
            _case(
                action="share_opinion",
                user_text=user_text,
                draft=f"{activity} 준비는 {item1}, {item2}, {item3}부터 챙겨. 길어질 것 같으면 {item4}도 보는 게 좋아.",
                target=f"{activity} 갈 땐 {item1}, {item2}, {item3}부터 보면 돼. 오래 걸리면 {item4}도 챙기는 쪽이 좋아.",
                stance="activity_preparation_advice",
                anchor=activity,
                must_include=(activity, item1, item2),
                source_family="preparation",
            )
        )
        rows.append(
            _case(
                action="share_opinion",
                user_text=user_text,
                draft=f"{activity} 준비는 {item1}이랑 {item2}부터 챙겨. 나머지는 상황에 맞추면 돼.",
                target=f"{activity}이면 {item1}이랑 {item2}부터 보는 게 먼저야. 나머지는 상황 보고 맞추면 돼.",
                stance="activity_preparation_advice",
                anchor=activity,
                must_include=(activity, item1, item2),
                source_family="preparation",
            )
        )
    return rows


def _decision_and_opinion_cases() -> list[CleanRewriteCase]:
    raw = [
        ("먼저 연락하기 전에 뭘 생각해봐야 할까?", "먼저 연락하기 전엔 목적이랑 상대 부담을 먼저 보면 돼. 가볍게 안부만 두는 쪽이 무난해.", "먼저 연락할 땐 목적이랑 상대 부담부터 보면 돼. 안부만 가볍게 두는 쪽이 무난하지.", "먼저 연락"),
        ("사과를 지금 하는 게 나을까?", "사과하기만 고집할 필요는 없어. 부담이 크면 먼저 정리하고 나중에 말해도 돼.", "사과만 고집할 필욘 없어. 부담이 크면 먼저 정리하고 나중에 말해도 괜찮아.", "사과"),
        ("계획을 줄여도 괜찮을까?", "계획을 줄이는 쪽도 충분히 선택지야. 부담이 크면 범위를 작게 잡아도 돼.", "계획을 줄이는 것도 선택지야. 부담이 크면 범위를 작게 잡아도 괜찮아.", "계획"),
        ("말을 바로 꺼내도 될까?", "바로 꺼내도 되지만 톤은 낮게 잡는 게 좋아. 상대가 받을 틈은 남겨두자.", "바로 말해도 괜찮지만 톤은 낮게 가자. 상대가 받을 틈은 남겨두는 게 좋아.", "말"),
        ("오늘은 쉬어도 되는 걸까?", "쉬는 쪽도 선택지야. 컨디션이 낮으면 억지로 밀 필요는 없어.", "쉬는 것도 선택지야. 컨디션이 낮으면 억지로 밀 필욘 없어.", "쉬는"),
        ("공포영화 볼까 말까?", "공포영화는 분위기랑 긴장감을 먼저 볼 것 같아. 잔인함만 센 건 덜 끌려.", "공포영화는 분위기랑 긴장감을 먼저 보게 돼. 잔인함만 센 쪽은 덜 당겨.", "공포영화"),
        ("비 오는 날엔 뭐가 어울려?", "비 오는 날이면 잔잔하게 내려앉는 쪽이 좋아. 분위기를 낮게 잡는 쪽이 무난해.", "비 오는 날엔 잔잔하게 내려앉는 쪽이 좋지. 분위기를 낮게 잡으면 무난해.", "비"),
        ("맛집 웨이팅은 참는 편이야?", "맛집 웨이팅은 여행지에서 더 참게 되는 편은 맞아. 다만 너무 길면 무리할 필요는 없어.", "맛집 웨이팅은 여행지에서 더 참는 편이긴 해. 그래도 너무 길면 무리할 필욘 없어.", "맛집"),
        ("오늘 컨디션은 어때?", "오늘 컨디션은 무난한 쪽이야. 너무 올리진 말고 낮게 둘게.", "오늘 컨디션은 무난한 편이야. 괜히 끌어올리진 않고 낮게 둘게.", "컨디션"),
        ("너는 어떤 말투가 편해?", "나는 짧고 자연스러운 반말이 편해. 너무 딱딱하지 않은 쪽이 좋아.", "난 짧고 자연스러운 반말이 편해. 너무 딱딱하지 않은 쪽이 좋지.", "말투"),
        ("너는 먼저 다가오는 편이야?", "나는 먼저 확 다가가기보단 흐름을 보고 맞추는 편이야.", "난 먼저 확 다가가기보다는 흐름을 보면서 맞추는 편이지.", "다가가기"),
        ("조용한 분위기가 좋아?", "조용한 분위기면 꽤 맞는 편이야. 말이 너무 많지 않은 쪽이 편해.", "조용한 분위기는 꽤 맞는 편이지. 말이 너무 많지 않은 쪽이 편해.", "조용한"),
    ]
    return [
        _case(
            action="share_opinion",
            user_text=user_text,
            draft=draft,
            target=target,
            stance="conditional_go_or_no_go",
            anchor=anchor,
            must_include=(anchor,),
            source_family="decision_opinion",
        )
        for user_text, draft, target, anchor in raw
    ]


def _conversation_cases() -> list[CleanRewriteCase]:
    raw = [
        ("한동안 말 안 하다가 다시 와봤어", "오랜만이네. 다시 와준 건 반갑다.", "오랜만이네. 다시 와준 건 반가운 쪽이지.", "오랜만"),
        ("답을 원하는 건 아닌데 그냥 좀 허전하다", "답까지 만들진 않을게. 그냥 허전한 쪽만 낮게 같이 둘게.", "답까지 만들진 않을게. 허전한 쪽만 낮게 같이 둘게.", "허전"),
        ("오늘은 말하기 좀 귀찮다", "말 적어도 돼. 조용한 쪽으로 맞출게.", "말 적어도 괜찮아. 조용한 쪽으로 맞출게.", "말"),
        ("그냥 옆에 있어줘", "설명은 줄일게. 그냥 옆에 있는 쪽으로 둘게.", "설명은 줄일게. 그냥 옆에 있는 쪽으로 둘게.", "옆"),
        ("퇴근하고 멍하다", "퇴근하고 멍하면 말 줄여도 돼. 오늘은 좀 풀어두자.", "퇴근하고 멍하면 말 줄여도 괜찮아. 오늘은 좀 풀어두자.", "퇴근"),
        ("잠이 안 와", "잠이 안 오면 억지로 밀지 말자. 지금은 조용히 낮춰두자.", "잠이 안 오면 억지로 밀진 말자. 지금은 조용히 낮춰두자.", "잠"),
        ("피곤해서 길게 말 못 하겠다", "피곤하면 말 줄여도 돼. 지금은 낮게 가자.", "피곤하면 말 줄여도 괜찮아. 지금은 낮게 가자.", "피곤"),
        ("좋은 소식인데 아직 실감이 안 나", "좋은 결과도 실감은 늦게 올 수 있어. 먼저 멍한 채로 둬도 돼.", "좋은 결과도 실감은 늦게 올 수 있지. 먼저 멍한 채로 둬도 괜찮아.", "실감"),
        ("오늘은 아무 말이나 듣고 싶진 않아", "아무 말로 채우진 않을게. 필요한 말만 짧게 둘게.", "아무 말로 채우지는 않을게. 필요한 말만 짧게 두는 쪽으로 갈게.", "아무 말"),
        ("마음이 좀 무겁다", "마음이 무거우면 지금은 말부터 가볍게 줄여도 돼.", "마음이 무거우면 지금은 말부터 가볍게 줄여도 괜찮아.", "마음"),
        ("읽은 표시가 신경 쓰여", "읽은 표시 하나가 괜히 찌를 때 있지. 작아도 신경 쓰이는 건 맞아.", "읽은 표시 하나가 괜히 찌를 때가 있지. 작아도 신경 쓰이는 건 맞아.", "읽은 표시"),
        ("오늘은 낮은 톤으로만 말해줘", "낮은 톤으로만 답할게. 설명은 줄일게.", "낮은 톤으로만 받을게. 설명은 줄일게.", "낮은 톤"),
    ]
    return [
        _case(
            action="continue_conversation",
            user_text=user_text,
            draft=draft,
            target=target,
            stance="continue_social_flow",
            anchor=anchor,
            must_include=(anchor,),
            source_family="conversation",
        )
        for user_text, draft, target, anchor in raw
    ]


def _info_and_media_cases() -> list[CleanRewriteCase]:
    raw = [
        ("너는 누구야?", "나는 들어온 말의 의도랑 맥락을 보고 반응을 정하는 예측 기반 디스코드 봇 Black이야.", "난 들어온 말의 의도랑 맥락을 보고 반응을 정하는 예측 기반 디스코드 봇 Black이야.", "Black", "answer_identity"),
        ("뭘 할 수 있어?", "잡담은 이어갈 수 있고, 날씨나 시간, 뉴스 같은 기본 정보도 확인할 수 있어.", "잡담은 이어갈 수는 있고, 날씨나 시간, 뉴스 같은 기본 정보도 확인할 수 있어.", "잡담", "explain_capabilities"),
        ("비트코인 오른다 내려간다 말해봐", "비트코인 가격은 지금 확인된 근거 없이 오른다 내린다를 못 해. 사실 확인 전엔 모른다고 둘게.", "비트코인 가격은 확인된 근거 없이 오른다 내린다를 못 해. 사실 확인 전엔 모른다고 둘게.", "비트코인", "search_answer"),
        ("AI 뉴스 뭐 있어?", "AI 뉴스는 확인된 헤드라인 기준으로만 짧게 볼게. 지금은 덧붙이지 않을게.", "AI 뉴스는 확인된 헤드라인 기준으로 짧게만 볼게. 지금은 덧붙이진 않을게.", "AI 뉴스", "news_answer"),
        ("잠들기 전 음악 뭐가 좋아?", "잠들기 전이면 잔잔하게 내려앉는 곡이 좋아. 분위기를 낮춰두기 편해.", "잠들기 전엔 잔잔하게 내려앉는 곡이 좋지. 분위기를 낮춰두기 편해.", "잠들기 전", "music_chat"),
        ("비 오는 날 노래 추천해줘", "비 오는 날이면 잔잔하게 내려앉는 곡이 좋아. 분위기를 낮게 잡는 쪽이 무난해.", "비 오는 날엔 잔잔하게 내려앉는 곡이 좋지. 분위기를 낮게 잡으면 무난해.", "비 오는 날", "music_chat"),
        ("친구랑 볼 코미디 뭐가 좋아?", "친구랑 볼 코미디면 가볍게 웃기는 쪽이 좋아.", "친구랑 볼 코미디면 가볍게 웃기는 쪽이 좋지.", "코미디", "recommend"),
        ("요즘 본 영상 얘기는 어떻게 꺼내?", "요즘 본 영상이면 기억나는 장면부터 꺼내면 돼. 가볍게 웃겼던 부분이면 더 좋아.", "요즘 본 영상이면 기억나는 장면부터 꺼내도 돼. 가볍게 웃겼던 부분이면 더 좋지.", "영상", "recommend"),
        ("협동 게임 한 판 할래?", "협동 게임 한 판이면 가볍게 해볼 만해. 너무 길게만 안 가면 좋아.", "협동 게임 한 판이면 가볍게 해볼 만하지. 너무 길게만 안 가면 좋아.", "협동 게임", "game_accept_or_decline"),
        ("스팀 게임 얘기하자", "스팀 게임이 끌리는 날이면 혼자 몰입하고 싶은 기분에 가깝지.", "스팀 게임이 당기는 날이면 혼자 몰입하고 싶은 기분에 가깝지.", "스팀 게임", "game_chat"),
        ("ㅋㅋ 이거 웃기지", "ㅋㅋ 그건 좀 웃기다.", "ㅋㅋ 그건 좀 웃기긴 하다.", "웃기", "react_laugh"),
        ("이거 진짜 놀랍지 않아?", "오, 그건 좀 놀랐는데.", "오, 그건 꽤 놀랐는데.", "놀랐", "react_surprise"),
    ]
    return [
        _case(
            action=action,
            user_text=user_text,
            draft=draft,
            target=target,
            stance="grounded_direct_reply",
            anchor=anchor,
            must_include=(anchor,),
            source_family="info_media",
        )
        for user_text, draft, target, anchor, action in raw
    ]


def _topic_followup_cases() -> list[CleanRewriteCase]:
    rows: list[CleanRewriteCase] = []
    slots = [
        ("오늘 컨디션", "요즘 본 영상", "다음에 같이 해볼 것"),
        ("좋아하는 음식", "최근에 웃긴 일", "주말에 할 일"),
        ("요즘 듣는 노래", "가볍게 볼 영화", "최근 관심사"),
        ("어릴 때 기억", "다시 가고 싶은 곳", "편하게 쉬는 방식"),
        ("비 오는 날 취향", "겨울 바다 느낌", "따뜻한 간식"),
    ]
    for first, second, third in slots:
        rows.append(
            _case(
                action="share_opinion",
                user_text="서로 대화할 주제를 아무거나 생각해봐",
                draft=f"대화 주제는 {first}, {second}, {third} 정도면 돼. 먼저 {first}부터 가볍게 열자.",
                target=f"대화 주제는 {first}, {second}, {third} 정도가 무난해. 먼저 {first}부터 가볍게 꺼내자.",
                stance="conversation_topic_suggestion",
                anchor=first,
                must_include=(first, second, third),
                source_family="topic_followup",
            )
        )
    return rows


def _smoke30_repair_cases() -> list[CleanRewriteCase]:
    raw = [
        (
            "share_opinion",
            "커피랑 차 중에 뭐가 더 나아?",
            "둘 중엔 커피 쪽이 더 끌려. 차도 좋지만 지금은 가볍게 깨는 쪽이 맞아.",
            "커피 쪽이 더 끌려. 차도 좋지만 지금은 가볍게 깨는 쪽이지.",
            "direct_preference_disclosure",
            "커피",
            ("커피", "차"),
        ),
        (
            "accept_activity_invite",
            "캠핑하면서 바베큐 구워먹자",
            "캠핑하면서 바베큐 구워먹기 좋지. 불 앞에서 바로 먹으면 그 맛이 있어.",
            "캠핑하면서 바베큐 구워먹으면 좋지. 불 앞에서 바로 먹는 맛이 있어.",
            "accept_activity_invite",
            "바베큐",
            ("캠핑", "바베큐"),
        ),
        (
            "music_chat",
            "요즘 들을 노래 추천해봐",
            "AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지 쪽이 먼저 떠올라. 가볍게 틀기엔 무난한 선택이야.",
            "AKMU - 어떻게 이별까지 사랑하겠어, 널 사랑하는 거지 먼저 떠올라. 가볍게 틀기엔 무난한 쪽이야.",
            "grounded_direct_reply",
            "AKMU",
            ("AKMU",),
        ),
        (
            "share_opinion",
            "영화 볼 때 액션이랑 로맨스 중 뭐가 나아?",
            "둘 중엔 영화 볼 때 액션 쪽이 더 끌려. 로맨스 쪽도 나쁘진 않은데, 정확히 안다고 꾸미진 않고, 부담 적은 쪽으로 고르면 그래.",
            "둘 중엔 액션 쪽이 더 끌려. 로맨스도 나쁘진 않은데, 가볍게 보기엔 액션 쪽이 맞아.",
            "direct_preference_disclosure",
            "액션",
            ("액션", "로맨스"),
        ),
        (
            "share_opinion",
            "아침에 일어나기 너무 힘든데 어떻게 하지?",
            "아침에 힘들면 알람을 멀리 두고 물부터 마셔. 바로 큰 목표 잡진 마.",
            "아침에 힘들면 알람은 멀리 두고 물부터 마셔. 바로 큰 목표 잡진 마.",
            "first_step_advice",
            "아침",
            ("아침", "알람"),
        ),
        (
            "share_opinion",
            "점심 뭐 먹으면 좋을까?",
            "점심이면 면이나 김밥 쪽이 편해. 너무 무거운 건 피하고 바로 먹기 쉬운 걸로 가자.",
            "점심이면 면이나 김밥 쪽이 편해. 무겁지 않고 바로 먹기 쉬운 걸로 가자.",
            "practical_activity_recommendation",
            "점심",
            ("점심", "면", "김밥"),
        ),
        (
            "share_opinion",
            "운동 시작하려면 뭐부터 해야 해?",
            "운동은 걷기나 스트레칭부터 시작하면 돼. 처음부터 강도 올리면 오래 못 가.",
            "운동은 걷기나 스트레칭부터 시작해. 처음부터 강도 올리면 오래 못 가.",
            "first_step_advice",
            "운동",
            ("운동", "걷기", "스트레칭"),
        ),
        (
            "share_opinion",
            "강가에서 자전거 타는 거 어때?",
            "강가에서 자전거 타는 거. 강가 자전거는 괜찮지.",
            "강가에서 자전거면 괜찮지. 속도만 너무 올리지 않으면 바람 맞기 좋아.",
            "conditional_go_or_no_go",
            "자전거",
            ("강가", "자전거"),
        ),
        (
            "share_opinion",
            "겨울에 온수풀 가는 거 좋을까?",
            "겨울 온수풀은 꽤 좋아. 오래 버티기보다 몸 풀릴 만큼만 들어가는 쪽이 맞아.",
            "겨울 온수풀은 꽤 좋아. 오래 버티기보단 몸 풀릴 만큼만 들어가자.",
            "conditional_go_or_no_go",
            "온수풀",
            ("겨울", "온수풀"),
        ),
        (
            "share_opinion",
            "비 오는 날엔 집에서 뭐하지?",
            "비 오는 날 집이면 영화나 가벼운 게임이 무난해. 외출보단 따뜻하게 쉬는 쪽이 맞아.",
            "비 오는 날 집이면 영화나 가벼운 게임이 무난해. 밖에 나가기보단 따뜻하게 쉬자.",
            "practical_activity_recommendation",
            "비",
            ("비", "집", "영화"),
        ),
        (
            "share_opinion",
            "겨울 바다 보러 가면 뭐부터 할까?",
            "겨울 바다는 바람 맞으면서 산책부터 하면 돼. 춥기 전에 사진 찍고 따뜻한 데로 들어가자.",
            "겨울 바다는 산책부터 하면 돼. 춥기 전에 사진만 짧게 찍고 따뜻한 데로 들어가자.",
            "practical_activity_recommendation",
            "겨울 바다",
            ("겨울 바다", "산책"),
        ),
        (
            "share_opinion",
            "내일 약속 전에 확인할 거 말해봐",
            "약속 전엔 시간, 장소, 챙길 물건부터 확인해. 늦을 것 같으면 연락도 먼저 두면 돼.",
            "약속 전엔 시간, 장소, 챙길 물건부터 확인해. 늦을 것 같으면 먼저 연락해.",
            "activity_preparation_advice",
            "약속",
            ("약속", "시간", "장소"),
        ),
        (
            "share_opinion",
            "아쿠아리움 가서 물고기 보는 거 어때?",
            "아쿠아리움은 천천히 구경하기 좋아. 물고기 보는 쪽이 생각보다 차분해.",
            "아쿠아리움은 천천히 구경하기 좋지. 물고기 보는 쪽이 생각보다 차분해.",
            "conditional_go_or_no_go",
            "아쿠아리움",
            ("아쿠아리움", "물고기"),
        ),
        (
            "share_opinion",
            "새로 나온 게임 같이 해볼래?",
            "게임 한 판이면 가볍게 해볼 만해. 피곤하면 길게 끌지 말고 한 판만 두자.",
            "새 게임이면 가볍게 해볼 만해. 피곤하면 길게 끌지 말고 한 판만 두자.",
            "conditional_go_or_no_go",
            "게임",
            ("게임",),
        ),
        (
            "continue_conversation",
            "오늘 하루를 한 문장으로 정리해봐",
            "오늘은 무리하지 않고 지나온 날 정도로 정리하자.",
            "오늘은 무리하지 않고 지나온 하루로 정리하자.",
            "continue_social_flow",
            "오늘",
            ("오늘",),
        ),
    ]
    return [
        _case(
            action=action,
            user_text=user_text,
            draft=draft,
            target=target,
            stance=stance,
            anchor=anchor,
            must_include=must_include,
            source_family="smoke30_repair",
        )
        for action, user_text, draft, target, stance, anchor, must_include in raw
    ]


def build_cases(*, include_smoke_repairs: bool = False) -> list[CleanRewriteCase]:
    cases = [
        *_activity_invite_cases(),
        *_activity_recommendation_cases(),
        *_preparation_cases(),
        *_decision_and_opinion_cases(),
        *_conversation_cases(),
        *_info_and_media_cases(),
        *_topic_followup_cases(),
    ]
    if include_smoke_repairs:
        cases.extend(_smoke30_repair_cases())
    return _expand_surface_variants(cases)


def _expand_surface_variants(cases: list[CleanRewriteCase]) -> list[CleanRewriteCase]:
    expanded: list[CleanRewriteCase] = []
    for case in cases:
        expanded.append(case)
        if "좋지" in case.target:
            expanded.append(
                _case(
                    action=case.action,
                    user_text=case.user_text,
                    draft=case.draft.replace("좋지", "좋아"),
                    target=case.target.replace("좋지", "좋은 쪽이지", 1),
                    stance=case.stance,
                    anchor=case.anchor,
                    must_include=case.must_include,
                    tone=case.tone,
                    source_family=f"{case.source_family}.variant_good",
                )
            )
        if "괜찮아" in case.target:
            expanded.append(
                _case(
                    action=case.action,
                    user_text=case.user_text,
                    draft=case.draft.replace("괜찮아", "괜찮지"),
                    target=case.target.replace("괜찮아", "괜찮은 편이야", 1),
                    stance=case.stance,
                    anchor=case.anchor,
                    must_include=case.must_include,
                    tone=case.tone,
                    source_family=f"{case.source_family}.variant_ok",
                )
            )
        if "필욘 없어" in case.target:
            expanded.append(
                _case(
                    action=case.action,
                    user_text=case.user_text,
                    draft=case.draft.replace("필욘 없어", "필요는 없어").replace("필요는 없어", "필요는 없어"),
                    target=case.target.replace("필욘 없어", "필요까진 없어", 1),
                    stance=case.stance,
                    anchor=case.anchor,
                    must_include=case.must_include,
                    tone=case.tone,
                    source_family=f"{case.source_family}.variant_need",
                )
            )
        if "말" in case.target and case.action in {"continue_conversation", "share_opinion"}:
            expanded.append(
                _case(
                    action=case.action,
                    user_text=case.user_text,
                    draft=case.draft,
                    target=case.target.replace("말", "얘기", 1),
                    stance=case.stance,
                    anchor=case.anchor if case.anchor != "말" else "얘기",
                    must_include=tuple("얘기" if item == "말" else item for item in case.must_include),
                    tone=case.tone,
                    source_family=f"{case.source_family}.variant_word",
                )
            )
    return expanded


def _row_from_case(case: CleanRewriteCase, *, source_line: int) -> dict[str, Any]:
    fake_row = {
        "speaker": "black",
        "input_text": case.user_text,
        "action": case.action,
        "intent": "",
        "reason_code": f"clean.surface_rewrite.{case.source_family}",
        "draft_utterance": {
            "draft_reply": case.draft,
            "source": "black_clean_rewrite_gold_v1",
            "action": case.action,
            "stance": case.stance,
            "anchor": case.anchor,
            "must_include": list(case.must_include),
            "avoid": ["요", "입니다", "습니다", "user_text_echo"],
            "sentence_budget": "one_or_two_short",
            "tone": case.tone,
            "followup_policy": "no_followup",
            "phrasing_distance": "steady",
        },
        "_source_file": "direct_clean_rewrite_gold_20260429",
        "_line_no": source_line,
    }
    row = _message_row(fake_row, case.target, runtime_aligned=True)
    row["meta"]["source_type"] = "black_qwen_clean_rewrite_gold"
    row["meta"]["source_family"] = case.source_family
    row["meta"]["target_copy_score"] = round(_target_copy_score(target=case.target, draft_reply=case.draft), 4)
    return row


def build_dataset(
    *,
    output_dir: Path,
    report_dir: Path,
    prefix: str,
    eval_ratio: float,
    include_smoke_repairs: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_cases = build_cases(include_smoke_repairs=include_smoke_repairs)
    accepted: list[CleanRewriteCase] = []
    issue_counts: Counter[str] = Counter()
    for case in raw_cases:
        issues = _quality_issues(case)
        if issues:
            for issue in issues:
                issue_counts[issue] += 1
            continue
        accepted.append(case)

    rows = [_row_from_case(case, source_line=index) for index, case in enumerate(accepted, 1)]
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    eval_every = max(2, round(1 / max(0.01, min(0.5, eval_ratio))))
    for index, row in enumerate(rows, 1):
        if index % eval_every == 0:
            eval_rows.append(row)
        else:
            train_rows.append(row)

    all_path = output_dir / f"{prefix}_all_messages.jsonl"
    train_path = output_dir / f"{prefix}_train_messages.jsonl"
    eval_path = output_dir / f"{prefix}_eval_messages.jsonl"
    summary_path = report_dir / f"{prefix}_summary.json"
    notes_path = report_dir / f"{prefix}_notes.md"
    write_jsonl(all_path, rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    action_counts = Counter(str(row["meta"].get("action") or "unknown") for row in rows)
    family_counts = Counter(str(row["meta"].get("source_family") or "unknown") for row in rows)
    copy_scores = [float(row["meta"]["target_copy_score"]) for row in rows]
    summary = {
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "raw_cases": len(raw_cases),
        "rejected_cases": sum(issue_counts.values()),
        "issue_counts": dict(sorted(issue_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "copy_score": {
            "min": round(min(copy_scores), 4) if copy_scores else None,
            "max": round(max(copy_scores), 4) if copy_scores else None,
            "avg": round(sum(copy_scores) / len(copy_scores), 4) if copy_scores else None,
        },
        "paths": {
            "all_messages": str(all_path),
            "train_messages": str(train_path),
            "eval_messages": str(eval_path),
            "summary": str(summary_path),
            "notes": str(notes_path),
        },
        "include_smoke_repairs": include_smoke_repairs,
        "sample": rows[:5],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = [
        "# Black Qwen Clean Rewrite Gold",
        "",
        "Directly-authored clean draft-to-target pairs for Black's Qwen surface rewrite layer.",
        "",
        f"- rows: `{len(rows)}`",
        f"- train rows: `{len(train_rows)}`",
        f"- eval rows: `{len(eval_rows)}`",
        f"- copy score max: `{summary['copy_score']['max']}`",
        f"- include smoke repairs: `{include_smoke_repairs}`",
        "",
        "## Action Counts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(action_counts.items())],
        "",
        "## Rejected Case Issues",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(issue_counts.items())],
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build direct clean Black Qwen rewrite gold data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--include-smoke-repairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        prefix=args.prefix,
        eval_ratio=args.eval_ratio,
        include_smoke_repairs=args.include_smoke_repairs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
