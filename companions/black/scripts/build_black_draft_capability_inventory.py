from __future__ import annotations

import ast
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "src" / "predictive_bot" / "core" / "draft_nlg.py"
DEFAULT_OUTPUT = ROOT / "reports" / "black_draft_capability_inventory_20260510.md"


DETAIL_LABEL_BY_DETAIL = {
    "relationship_boundary_position": "관계/연애 경계",
    "romance_drama_relationship_choice": "관계/연애 상상",
    "social_boundary_tactic": "관계/사회적 경계",
    "hostile_social_boundary": "관계/갈등 대처",
    "money_relationship_dilemma": "관계/돈과 신뢰",
    "tf_empathy_logic_choice": "관계/공감과 소통",
    "values_reflective_position": "가치관/철학",
    "ethical_dilemma_boundary": "가치관/윤리 딜레마",
    "memory_life_reset_dilemma": "삶/기억/리셋",
    "legacy_mortality_reflection": "삶/죽음/남김",
    "career_life_goal_reflection": "삶/진로/성공",
    "personality_self_reflection": "성격/자기 이해",
    "private_quirk_reflection": "성격/사소한 비밀",
    "daily_quirk_preference": "일상/습관",
    "everyday_preference_debate": "일상/취향 논쟁",
    "digital_device_habit": "일상/디지털 습관",
    "household_light_if": "일상/집과 가벼운 상상",
    "micro_joy_savoring": "일상/소확행",
    "embarrassing_mishap_recovery": "상황대처/민망한 사고",
    "situational_tactic": "상황대처/현장 수습",
    "intrusion_stalking_threat_response": "상황대처/위협과 안전",
    "crime_secret_survival_dilemma": "상황대처/범죄와 생존",
    "k_work_school_limit_test": "직장/학교 한계",
    "work_money_position": "직장/돈/일",
    "school_exam_memory": "학교/시험 기억",
    "family_childhood_memory": "가족/어린 시절",
    "k_family_sibling_dynamics": "가족/형제자매",
    "food_cooking_preference": "음식/요리",
    "food_debate_preference": "음식/취향 논쟁",
    "food_texture_sauce_preference": "음식/식감과 소스",
    "gross_food_balance": "음식/괴식 밸런스",
    "smell_taste_sensory_balance": "감각/맛과 냄새",
    "sensory_metaphor": "감각/문학적 비유",
    "health_sleep_routine": "몸/건강/수면",
    "body_absurd_power_debuff": "몸/초능력과 저주",
    "animal_nature_preference": "동물/자연 취향",
    "animal_nature_reincarnation_if": "동물/빙의와 환생",
    "animal_docu_pet_reincarnation_bond": "동물/반려와 애착",
    "travel_rest_preference": "여행/휴식",
    "hideout_healing_space_preference": "공간/아지트와 회복",
    "media_music_culture_preference": "미디어/음악/문화",
    "fandom_media_preference": "미디어/덕질",
    "webtoon_anime_fandom_preference": "미디어/웹툰/애니",
    "fantasy_world_role_choice": "상상/판타지",
    "speculative_choice_position": "상상/선택형 IF",
    "absurd_balance_choice": "상상/황당 밸런스",
    "absurd_logic_debate": "상상/무논리 토론",
    "horror_survival_if": "상상/공포 생존",
    "time_parallel_if_reflection": "상상/시간과 평행우주",
    "post_apocalypse_zombie_survival": "상상/좀비와 멸망",
    "cosmic_deepsea_mystery_if": "상상/우주와 심해",
    "uncanny_belief_reflection": "초자연/미신과 괴담",
    "uncanny_experience_reflection": "초자연/기묘한 경험",
    "ai_sentience_humanity_reflection": "AI/감정과 인간성",
    "ai_vtuber_meta_bond": "AI/버튜버 관계",
    "cyber_ai_identity_reflection": "AI/사이버 정체성",
    "virtual_memory_reflection": "AI/가상현실과 기억",
    "companion_meta_reflection": "AI/컴패니언 메타",
    "fashion_style_preference": "취향/패션",
}


OVERALL_TOPIC_BY_DETAIL = {
    "relationship_boundary_position": "관계/사회",
    "romance_drama_relationship_choice": "관계/사회",
    "social_boundary_tactic": "관계/사회",
    "hostile_social_boundary": "관계/사회",
    "money_relationship_dilemma": "관계/사회",
    "tf_empathy_logic_choice": "관계/사회",
    "values_reflective_position": "삶/가치관",
    "ethical_dilemma_boundary": "삶/가치관",
    "memory_life_reset_dilemma": "삶/가치관",
    "legacy_mortality_reflection": "삶/가치관",
    "career_life_goal_reflection": "삶/가치관",
    "personality_self_reflection": "삶/가치관",
    "private_quirk_reflection": "삶/가치관",
    "daily_quirk_preference": "일상/취향",
    "everyday_preference_debate": "일상/취향",
    "digital_device_habit": "일상/취향",
    "household_light_if": "일상/취향",
    "micro_joy_savoring": "일상/취향",
    "fashion_style_preference": "일상/취향",
    "travel_rest_preference": "일상/취향",
    "hideout_healing_space_preference": "일상/취향",
    "embarrassing_mishap_recovery": "상황대처/현실",
    "situational_tactic": "상황대처/현실",
    "intrusion_stalking_threat_response": "상황대처/현실",
    "crime_secret_survival_dilemma": "상황대처/현실",
    "k_work_school_limit_test": "상황대처/현실",
    "work_money_position": "상황대처/현실",
    "school_exam_memory": "상황대처/현실",
    "family_childhood_memory": "상황대처/현실",
    "k_family_sibling_dynamics": "상황대처/현실",
    "food_cooking_preference": "음식/감각/몸",
    "food_debate_preference": "음식/감각/몸",
    "food_texture_sauce_preference": "음식/감각/몸",
    "gross_food_balance": "음식/감각/몸",
    "smell_taste_sensory_balance": "음식/감각/몸",
    "sensory_metaphor": "음식/감각/몸",
    "health_sleep_routine": "음식/감각/몸",
    "body_absurd_power_debuff": "음식/감각/몸",
    "animal_nature_preference": "동물/자연",
    "animal_nature_reincarnation_if": "동물/자연",
    "animal_docu_pet_reincarnation_bond": "동물/자연",
    "media_music_culture_preference": "미디어/문화",
    "fandom_media_preference": "미디어/문화",
    "webtoon_anime_fandom_preference": "미디어/문화",
    "fantasy_world_role_choice": "상상/IF",
    "speculative_choice_position": "상상/IF",
    "absurd_balance_choice": "상상/IF",
    "absurd_logic_debate": "상상/IF",
    "horror_survival_if": "상상/IF",
    "time_parallel_if_reflection": "상상/IF",
    "post_apocalypse_zombie_survival": "상상/IF",
    "cosmic_deepsea_mystery_if": "상상/IF",
    "uncanny_belief_reflection": "초자연/기묘함",
    "uncanny_experience_reflection": "초자연/기묘함",
    "ai_sentience_humanity_reflection": "AI/버튜버",
    "ai_vtuber_meta_bond": "AI/버튜버",
    "cyber_ai_identity_reflection": "AI/버튜버",
    "virtual_memory_reflection": "AI/버튜버",
    "companion_meta_reflection": "AI/버튜버",
}


INTERNAL_VALUES = {
    "neutral",
    "direct_opinion",
    "direct_preference_disclosure",
    "activity_preparation_advice",
    "practical_activity_recommendation",
    "concrete_topic_answer",
    "no_followup",
    "open_general",
    "body_state_response",
    "true",
    "false",
}


FUNCTION_TOPIC_BY_NAME = {
    "_draft_text": "대화운영/기본 Draft",
    "_render_output_shape_direct_reply": "대화운영/직접 문장틀",
    "_render_long_form_story_direct_reply": "창작/장문 서사",
    "_render_story_summary_reaction_direct_reply": "창작/서사 감상",
    "_render_memory_boundary_direct_reply": "기억/검증과 경계",
    "_render_grounded_memory_reference_reply": "기억/검증된 기억",
    "_render_expression_request_direct_reply": "표현/감각/비유",
    "_render_generic_continue_reply": "대화운영/이어가기",
    "_render_generic_feeling_reply": "감정/기분 반응",
    "_render_generic_ack_reply": "대화운영/짧은 인정",
    "_render_relationship_extreme_boundary_direct_reply": "관계/연애 경계",
    "_render_relationship_microtension_direct_reply": "관계/미세 긴장",
    "_render_relationship_deep_context_direct_reply": "관계/깊은 맥락",
    "_render_black_self_style_direct_reply": "AI/Black 정체성",
    "_render_round11_structural_direct_reply": "혼합세트/공포-소통-철학",
    "_render_deep_mix_structural_direct_reply": "혼합세트/딥토크-관계",
    "_render_daily_mix_structural_direct_reply": "혼합세트/일상-역할극",
    "_render_relationship_boundary_direct_reply": "관계/연애 경계",
    "_render_work_school_direct_reply": "직장/학교/돈",
    "_render_food_lifestyle_direct_reply": "음식/생활 취향",
    "_render_unverified_memory_reference_reply": "기억/미확인 기억",
    "_render_black_seasonal_preference_reply": "여행/계절 취향",
    "_render_practical_direct_reply": "추천/실용 조언",
    "_render_body_state_direct_reply": "몸/건강/수면",
    "_render_contextual_observation_direct_reply": "일상/관찰과 다의어",
    "_render_open_question_persona_direct_reply": "AI/페르소나 질문",
    "_render_hypothetical_choice_direct_reply": "상상/IF 선택",
    "_render_daily_companion_priority_direct_reply": "일상/컴패니언",
    "_render_daily_practical_priority_direct_reply": "추천/실용 조언",
    "_render_absurd_situation_direct_reply": "상상/황당 상황극",
    "_render_generic_vs_choice_reply": "상상/밸런스 선택",
    "_render_generic_personified_reply": "상상/사물 빙의",
    "_render_romance_relationship_direct_reply": "관계/연애",
    "_render_lifestyle_preference_direct_reply": "일상/취향",
    "_render_social_personality_direct_reply": "성격/사회성",
    "_render_forest_animal_direct_reply": "동물/자연",
    "render_black_concrete_topic_question_reply": "대화운영/구체 질문",
}


KEYWORD_TOPIC_RULES = {
    "Black구조/개발메타": [
        "black",
        "white",
        "qwen",
        "bert",
        "modernbert",
        "draft",
        "draftnlg",
        "meaningpacket",
        "statedelta",
        "actionpolicy",
        "schema",
        "템플릿",
        "분류",
        "학습",
        "평가",
        "구조",
    ],
    "AI/버튜버/페르소나": [
        "ai",
        "인공지능",
        "버튜버",
        "컴패니언",
        "페르소나",
        "로봇",
        "서버",
        "모니터",
        "사용자",
        "대화상대",
        "사람아니",
        "가상현실",
    ],
    "관계/연애": [
        "애인",
        "연애",
        "고백",
        "사랑",
        "썸",
        "짝사랑",
        "소개팅",
        "이별",
        "전애인",
        "남사친",
        "여사친",
        "깻잎",
        "연락",
        "데이트",
        "결혼",
    ],
    "관계/사회갈등": [
        "친구",
        "상사",
        "뒷담",
        "손절",
        "약속",
        "단톡",
        "진상",
        "무임승차",
        "회식",
        "사과",
        "갈등",
        "선넘",
        "질투",
        "경계",
        "말싸움",
        "정치적",
        "종교적",
        "칭찬",
        "부끄",
        "플러팅",
    ],
    "감정/위로/자기상태": [
        "우울",
        "불안",
        "속상",
        "외롭",
        "위로",
        "자존감",
        "무기력",
        "힘들",
        "피곤",
        "스트레스",
        "눈물",
        "자괴감",
        "마음",
        "기분",
        "상처",
    ],
    "몸/건강/수면": [
        "잠",
        "수면",
        "졸리",
        "감기",
        "목",
        "배고프",
        "소화",
        "두통",
        "몸",
        "운동",
        "헬스",
        "병원",
        "치과",
        "영양제",
        "다이어트",
    ],
    "일상/생활습관": [
        "알람",
        "샤워",
        "양치",
        "청소",
        "방",
        "집",
        "아침",
        "저녁",
        "주말",
        "쉬는날",
        "루틴",
        "스마트폰",
        "배터리",
        "카톡",
        "sns",
    ],
    "음식/취향": [
        "라면",
        "치킨",
        "피자",
        "음식",
        "밥",
        "카페",
        "커피",
        "붕어빵",
        "탕수육",
        "민트초코",
        "맛",
        "매운",
        "야식",
        "과일",
        "김치",
        "국밥",
        "떡볶이",
    ],
    "직장/학교/돈": [
        "회사",
        "직장",
        "학교",
        "시험",
        "발표",
        "과제",
        "교수",
        "상사",
        "퇴근",
        "출근",
        "월급",
        "돈",
        "로또",
        "통장",
        "알바",
        "취업",
        "수능",
        "면접",
        "점심시간",
        "혼자먹",
    ],
    "가족/어린 시절": [
        "가족",
        "부모님",
        "엄마",
        "아빠",
        "형제",
        "자매",
        "어린",
        "초등",
        "학창",
        "명절",
        "산타",
    ],
    "동물/자연": [
        "강아지",
        "고양이",
        "동물",
        "비둘기",
        "모기",
        "벌레",
        "나무",
        "숲",
        "꽃",
        "계절",
        "날씨",
        "비",
        "눈",
        "바다",
        "산",
        "공룡",
        "판다",
    ],
    "미디어/문화/덕질": [
        "영화",
        "드라마",
        "애니",
        "웹툰",
        "웹소설",
        "음악",
        "노래",
        "가수",
        "콘서트",
        "게임",
        "덕질",
        "최애",
        "유튜브",
        "넷플릭스",
        "방송",
        "콘텐츠",
    ],
    "상상/IF/밸런스": [
        "만약",
        "평생",
        "vs",
        "선택",
        "초능력",
        "투명인간",
        "타임머신",
        "무인도",
        "이세계",
        "마법",
        "저주",
        "밸런스",
        "화성",
        "식민지",
        "우주",
    ],
    "공포/초자연/위협": [
        "귀신",
        "유령",
        "공포",
        "무서",
        "침대밑",
        "엘리베이터",
        "스토킹",
        "범죄",
        "살인",
        "경찰",
        "신고",
        "피",
        "칼",
        "흉가",
        "가위눌",
        "데자뷔",
        "좀비",
        "거울",
        "낯설",
    ],
    "철학/가치관/삶": [
        "삶",
        "성공",
        "행복",
        "의미",
        "가치",
        "도덕",
        "진정한",
        "후회",
        "운명",
        "자유",
        "완벽",
        "어른",
        "죽음",
        "묘비",
        "마지막",
        "평범하다",
        "무언가에미쳐",
        "미친다",
        "논리",
    ],
    "표현/감각/비유": [
        "색깔",
        "냄새",
        "향기",
        "소리",
        "온도",
        "질감",
        "비유",
        "묘사",
        "표현",
        "공기",
        "촉감",
        "맛으로",
        "한문장",
    ],
    "여행/휴식/공간": [
        "여행",
        "휴가",
        "캠핑",
        "호캉스",
        "호텔",
        "아지트",
        "벙커",
        "도시",
        "나라",
        "비행기",
        "부산",
        "공간",
    ],
    "창작/스토리": [
        "이야기",
        "서사",
        "주인공",
        "장르",
        "소설",
        "캐릭터",
        "자서전",
        "감독",
        "작곡",
        "책",
    ],
    "역할극/상황대처": [
        "상황",
        "역할극",
        "대처",
        "수습",
        "변명",
        "통화",
        "관제탑",
        "알바생",
        "손님",
        "사장님",
        "카페알바",
    ],
    "대화운영/응답형식": [
        "질문",
        "답변",
        "추천",
        "골라",
        "하나만",
        "어떻게",
        "이유",
        "말해줘",
        "몇번",
        "주제",
    ],
}


SHAPE_PREFIX_TOPIC = {
    "meta_": "Black구조/개발메타",
    "v3_": "대화운영/v3 직접응답",
    "comfort_": "감정/위로",
    "practical_": "추천/실용 조언",
    "philosophy_": "철학/가치관",
    "values_": "철학/가치관",
    "body_": "몸/건강/수면",
    "relationship_": "관계/연애",
    "romance_": "관계/연애",
    "ai_": "AI/버튜버/페르소나",
    "sensory_": "표현/감각/비유",
    "poetic_": "표현/감각/비유",
    "service_": "역할극/상황대처",
    "broadcast_food": "음식/취향",
    "broadcast_ramen": "음식/취향",
    "broadcast_icecream": "음식/취향",
    "broadcast_cafe": "음식/취향",
    "broadcast_late_night_food": "음식/취향",
    "broadcast_spicy": "음식/취향",
    "broadcast_fruit": "음식/취향",
    "broadcast_music": "미디어/문화/덕질",
    "broadcast_media": "미디어/문화/덕질",
    "broadcast_content": "미디어/문화/덕질",
    "broadcast_game": "미디어/문화/덕질",
    "broadcast_entertainment": "미디어/문화/덕질",
    "broadcast_balance": "상상/IF/밸런스",
    "broadcast_time": "상상/IF/밸런스",
    "broadcast_invisibility": "상상/IF/밸런스",
    "broadcast_animal": "동물/자연",
    "broadcast_daily": "방송/시청자 소통",
    "broadcast_virtual": "AI/버튜버/페르소나",
    "broadcast_weekend": "일상/생활습관",
    "broadcast_bedtime": "일상/생활습관",
    "broadcast_alarm": "일상/생활습관",
    "broadcast_stress": "감정/위로",
    "broadcast_small_happiness": "일상/소확행",
    "broadcast_mood": "방송/시청자 소통",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract static Black DraftNLG capability triggers: topic functions, detail frames, "
            "keyword conditions, and reply frames."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def table_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def string_constants(node: ast.AST) -> list[str]:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
    return values


def likely_user_text_name(node: ast.AST) -> bool:
    name = node_name(node)
    return name in {"text", "user_text", "raw", "lower", "normalized"}


def collect_trigger_parts(condition: ast.AST) -> tuple[list[str], list[str]]:
    triggers: list[str] = []
    match_kinds: list[str] = []

    for node in ast.walk(condition):
        if isinstance(node, ast.Call):
            func = node_name(node.func)
            if func in {"_has_any", "_has_all", "_has_all_ordered"} and node.args:
                if not likely_user_text_name(node.args[0]):
                    continue
                values: list[str] = []
                for arg in node.args[1:]:
                    values.extend(string_constants(arg))
                for value in values:
                    if value and value not in INTERNAL_VALUES:
                        triggers.append(value)
                match_kinds.append(func)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                continue
            op = node.ops[0]
            left = node.left
            right = node.comparators[0]
            if isinstance(op, (ast.In, ast.NotIn)):
                if isinstance(left, ast.Constant) and isinstance(left.value, str) and likely_user_text_name(right):
                    if left.value and left.value not in INTERNAL_VALUES:
                        triggers.append(left.value)
                    match_kinds.append("in_text" if isinstance(op, ast.In) else "not_in_text")
                if isinstance(right, ast.Constant) and isinstance(right.value, str) and likely_user_text_name(left):
                    if right.value and right.value not in INTERNAL_VALUES:
                        triggers.append(right.value)
                    match_kinds.append("text_in")

    return sorted(dict.fromkeys(triggers)), sorted(dict.fromkeys(match_kinds))


def compare_string_value(condition: ast.AST, name: str) -> str | None:
    if not isinstance(condition, ast.Compare):
        return None
    if len(condition.ops) != 1 or not isinstance(condition.ops[0], ast.Eq):
        return None
    if not isinstance(condition.left, ast.Name) or condition.left.id != name:
        return None
    if len(condition.comparators) != 1:
        return None
    comparator = condition.comparators[0]
    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
        return comparator.value
    return None


def first_reply_text(statements: list[ast.stmt]) -> str:
    for statement in statements:
        if isinstance(statement, ast.Return):
            value = statement.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            if isinstance(value, ast.JoinedStr):
                parts = [
                    part.value
                    for part in value.values
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                ]
                return "".join(parts)
            if isinstance(value, ast.Call):
                strings = string_constants(value)
                if strings:
                    return strings[0]
            strings = string_constants(value) if value is not None else []
            if strings:
                return " ".join(strings[:3])
        if isinstance(statement, ast.If):
            nested = first_reply_text(statement.body)
            if nested:
                return nested
    return ""


def first_shape_code(statements: list[ast.stmt]) -> str:
    for statement in statements:
        if isinstance(statement, ast.Return):
            value = statement.value
            if isinstance(value, ast.Call) and node_name(value.func) == "_shape_result":
                if len(value.args) > 1:
                    code = value.args[1]
                    if isinstance(code, ast.Constant) and isinstance(code.value, str):
                        return code.value
            strings = string_constants(value) if value is not None else []
            for string in strings:
                if string.endswith("_reply") or string.endswith("_choice") or string.startswith(("meta_", "v3_")):
                    return string
        if isinstance(statement, ast.If):
            nested = first_shape_code(statement.body)
            if nested:
                return nested
    return ""


def parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return ""


def enclosing_detail(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, ast.If):
            detail = compare_string_value(current.test, "detail")
            if detail:
                return detail
        current = parents.get(current)
    return ""


def detail_label(detail: str) -> str:
    return DETAIL_LABEL_BY_DETAIL.get(detail, detail)


def overall_topic(detail: str, function: str) -> str:
    if detail:
        return OVERALL_TOPIC_BY_DETAIL.get(detail, f"미분류/{detail.split('_', 1)[0]}")
    if "food" in function:
        return "음식/감각/몸"
    if "relationship" in function or "social" in function:
        return "관계/사회"
    if "music" in function or "media" in function or "game" in function:
        return "미디어/문화"
    if "memory" in function or "persona" in function or "identity" in function:
        return "삶/가치관"
    if "absurd" in function or "hypothetical" in function:
        return "상상/IF"
    if "body" in function or "health" in function:
        return "음식/감각/몸"
    if "work" in function or "practical" in function:
        return "상황대처/현실"
    if "output_shape" in function:
        return "출력형식/직접응답"
    return "기타/공통"


def branch_role(function: str) -> str:
    if function in {"_infer_draft_frame_detail", "_infer_priority_draft_frame_detail"}:
        return "frame_signal"
    if function == "_render_output_shape_direct_reply":
        return "output_shape_reply"
    if function == "_draft_text":
        return "main_draft"
    if function.startswith("_render_") or function.startswith("render_black"):
        return "draft_reply"
    return "other"


def topic_from_reply_code(reply_code: str, triggers: list[str]) -> str:
    if not reply_code:
        return ""
    if reply_code in DETAIL_LABEL_BY_DETAIL:
        return detail_label(reply_code)
    for prefix, topic in sorted(SHAPE_PREFIX_TOPIC.items(), key=lambda item: len(item[0]), reverse=True):
        if reply_code.startswith(prefix):
            if topic == "대화운영/v3 직접응답":
                inferred = topic_from_keywords(triggers + [reply_code])
                return inferred if inferred != "대화운영/응답형식" else topic
            return topic
    inferred = topic_from_keywords(triggers + [reply_code])
    return inferred if inferred else ""


def topic_from_keywords(values: list[str]) -> str:
    haystack = " ".join(str(value).lower() for value in values if value)
    if not haystack:
        return ""
    scores: Counter[str] = Counter()
    for topic, keywords in KEYWORD_TOPIC_RULES.items():
        for keyword in keywords:
            if keyword.lower() in haystack:
                scores[topic] += 2 if len(keyword) > 2 else 1
    if not scores:
        return ""
    # Prefer more specific topics over the generic wording bucket when scores tie.
    ordered_topics = list(KEYWORD_TOPIC_RULES)
    return max(scores, key=lambda topic: (scores[topic], -ordered_topics.index(topic)))


def capability_topic(*, detail: str, function: str, reply_frame: str, reply_code: str, triggers: list[str]) -> str:
    if detail:
        return detail_label(detail)
    if reply_frame in DETAIL_LABEL_BY_DETAIL:
        return detail_label(reply_frame)
    code_topic = topic_from_reply_code(reply_code, triggers)
    if code_topic:
        return code_topic
    inferred = topic_from_keywords(triggers + [reply_frame, function])
    if inferred:
        return inferred
    return FUNCTION_TOPIC_BY_NAME.get(function, "대화운영/기타 직접응답")


def function_label(function: str) -> str:
    return function.removeprefix("_render_").removeprefix("render_black_").removesuffix("_direct_reply")


def extract_rows(source: Path) -> list[dict[str, Any]]:
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    parents = parent_map(tree)
    rows: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        triggers, match_kinds = collect_trigger_parts(node.test)
        if not triggers:
            continue
        function = enclosing_function(node, parents)
        if not (
            function.startswith("_render_")
            or function.startswith("render_black")
            or function in {"_draft_text", "_infer_draft_frame_detail", "_infer_priority_draft_frame_detail"}
        ):
            continue
        detail = enclosing_detail(node, parents)
        reply = first_reply_text(node.body)
        reply_code = first_shape_code(node.body)
        topic = capability_topic(
            detail=detail,
            function=function,
            reply_frame=reply,
            reply_code=reply_code,
            triggers=triggers,
        )
        rows.append(
            {
                "line": node.lineno,
                "function": function,
                "function_label": function_label(function),
                "overall_topic": overall_topic(detail, function),
                "capability_topic": topic,
                "branch_role": branch_role(function),
                "detail_topic": detail,
                "detail_label": detail_label(detail) if detail else "",
                "match_kinds": match_kinds,
                "triggers": triggers,
                "reply_code": reply_code,
                "reply_frame": reply,
            }
        )
    rows.sort(key=lambda row: (row["capability_topic"], row["detail_topic"], row["function"], row["line"]))
    return rows


def render_markdown(rows: list[dict[str, Any]], source: Path) -> str:
    trigger_counter: Counter[str] = Counter()
    for row in rows:
        trigger_counter.update(row["triggers"])

    rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_function: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_detail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_topic[row["capability_topic"]].append(row)
        rows_by_overall[row["overall_topic"]].append(row)
        rows_by_function[row["function"]].append(row)
        rows_by_role[row["branch_role"]].append(row)
        if row["detail_topic"]:
            rows_by_detail[row["detail_topic"]].append(row)

    lines: list[str] = [
        "# Black Draft Capability Inventory 2026-05-10",
        "",
        "Black DraftNLG 코드에서 실제로 쓰는 응답 분기, 트리거 단어, 세부 문장틀을 정적으로 추출한 파일.",
        "이 파일은 학습 라벨 수가 아니라 현재 코드가 알아보는 단어/문구와 그 단어가 연결되는 답변틀을 보여준다.",
        "",
        "## Summary",
        "",
        f"- source: `{source.relative_to(ROOT)}`",
        f"- trigger_branches: {len(rows)}",
        f"- unique_trigger_terms: {len(trigger_counter)}",
        f"- response_functions: {len(rows_by_function)}",
        f"- detail_frame_contexts: {len(rows_by_detail)}",
        f"- capability_topics: {len(rows_by_topic)}",
        f"- broad_topic_buckets: {len(rows_by_overall)}",
        f"- branch_roles: {len(rows_by_role)}",
        "",
        "## Capability Topic Summary",
        "",
        "| capability_topic | branches | unique_triggers | functions | detail_contexts | response_codes | reply_frames | roles |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for topic, group in sorted(rows_by_topic.items(), key=lambda item: item[0]):
        triggers = {trigger for row in group for trigger in row["triggers"]}
        replies = {row["reply_frame"] for row in group if row["reply_frame"]}
        response_codes = {row["reply_code"] for row in group if row["reply_code"]}
        lines.append(
            "| {topic} | {branches} | {triggers} | {functions} | {details} | {codes} | {replies} | {roles} |".format(
                topic=table_escape(topic),
                branches=len(group),
                triggers=len(triggers),
                functions=len({row["function"] for row in group}),
                details=len({row["detail_topic"] for row in group if row["detail_topic"]}),
                codes=len(response_codes),
                replies=len(replies),
                roles=table_escape(", ".join(sorted({row["branch_role"] for row in group}))),
            )
        )

    lines.extend(
        [
            "",
            "## Broad Bucket Summary",
            "",
            "이 섹션은 큰 묶음만 보는 보조 요약이다. 실제 세부 주제 판단은 위 Capability Topic Summary를 기준으로 본다.",
            "",
            "| broad_topic | branches | unique_triggers | capability_topics | functions | detail_contexts |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for topic, group in sorted(rows_by_overall.items(), key=lambda item: item[0]):
        triggers = {trigger for row in group for trigger in row["triggers"]}
        lines.append(
            "| {topic} | {branches} | {triggers} | {capability_topics} | {functions} | {details} |".format(
                topic=table_escape(topic),
                branches=len(group),
                triggers=len(triggers),
                capability_topics=len({row["capability_topic"] for row in group}),
                functions=len({row["function"] for row in group}),
                details=len({row["detail_topic"] for row in group if row["detail_topic"]}),
            )
        )

    lines.extend(
        [
            "",
            "## Detail Frame Capability Summary",
            "",
            "| overall_topic | detail_label | detail_topic | branches | unique_triggers | reply_frames | sample_triggers |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for detail, group in sorted(rows_by_detail.items(), key=lambda item: (overall_topic(item[0], ""), item[0])):
        triggers = sorted({trigger for row in group for trigger in row["triggers"]})
        replies = {row["reply_frame"] for row in group if row["reply_frame"]}
        lines.append(
            "| {overall} | {label} | `{detail}` | {branches} | {trigger_count} | {reply_count} | {sample} |".format(
                overall=table_escape(overall_topic(detail, "")),
                label=table_escape(detail_label(detail)),
                detail=table_escape(detail),
                branches=len(group),
                trigger_count=len(triggers),
                reply_count=len(replies),
                sample=table_escape(", ".join(triggers[:14])),
            )
        )

    lines.extend(
        [
            "",
            "## Function Capability Summary",
            "",
            "| function | role | branches | unique_triggers | capability_topics | detail_contexts | sample_triggers |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for function, group in sorted(rows_by_function.items(), key=lambda item: item[0]):
        triggers = sorted({trigger for row in group for trigger in row["triggers"]})
        lines.append(
            "| `{function}` | {role} | {branches} | {trigger_count} | {topics} | {details} | {sample} |".format(
                function=table_escape(function),
                role=table_escape(", ".join(sorted({row["branch_role"] for row in group}))),
                branches=len(group),
                trigger_count=len(triggers),
                topics=len({row["capability_topic"] for row in group}),
                details=len({row["detail_topic"] for row in group if row["detail_topic"]}),
                sample=table_escape(", ".join(triggers[:16])),
            )
        )

    lines.extend(["", "## Branch Role Summary", ""])
    lines.extend(["| role | branches | capability_topics | unique_triggers | functions |", "|---|---:|---:|---:|---:|"])
    for role, group in sorted(rows_by_role.items(), key=lambda item: item[0]):
        triggers = {trigger for row in group for trigger in row["triggers"]}
        lines.append(
            "| {role} | {branches} | {topics} | {triggers} | {functions} |".format(
                role=table_escape(role),
                branches=len(group),
                topics=len({row["capability_topic"] for row in group}),
                triggers=len(triggers),
                functions=len({row["function"] for row in group}),
            )
        )

    lines.extend(["", "## Trigger Lexicon By Capability Topic", ""])
    for topic, group in sorted(rows_by_topic.items(), key=lambda item: item[0]):
        triggers = sorted({trigger for row in group for trigger in row["triggers"]})
        lines.extend([f"### {topic}", ""])
        for index in range(0, len(triggers), 40):
            lines.append("- " + ", ".join(f"`{table_escape(trigger)}`" for trigger in triggers[index : index + 40]))
        lines.append("")

    lines.extend(["", "## Full Branch Inventory", ""])
    for topic, group in sorted(rows_by_topic.items(), key=lambda item: item[0]):
        lines.extend([f"### {topic}", ""])
        for row in group:
            context = row["detail_label"] or row["function_label"]
            detail = f" / `{row['detail_topic']}`" if row["detail_topic"] else ""
            lines.extend(
                [
                    f"#### line {row['line']} / `{row['function']}` / {row['branch_role']} / {context}{detail}",
                    "",
                    f"- match: {', '.join(row['match_kinds'])}",
                    "- triggers: " + ", ".join(f"`{table_escape(trigger)}`" for trigger in row["triggers"]),
                ]
            )
            if row["reply_code"]:
                lines.append(f"- response_code: `{table_escape(row['reply_code'])}`")
            if row["reply_frame"]:
                lines.append(f"- reply_frame: {row['reply_frame']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    rows = extract_rows(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(rows, args.source), encoding="utf-8")
    trigger_count = len({trigger for row in rows for trigger in row["triggers"]})
    function_count = len({row["function"] for row in rows})
    detail_count = len({row["detail_topic"] for row in rows if row["detail_topic"]})
    capability_topic_count = len({row["capability_topic"] for row in rows})
    broad_topic_count = len({row["overall_topic"] for row in rows})
    print(
        {
            "mode": "black_draft_capability_inventory",
            "branches": len(rows),
            "unique_trigger_terms": trigger_count,
            "functions": function_count,
            "detail_contexts": detail_count,
            "capability_topics": capability_topic_count,
            "broad_topic_buckets": broad_topic_count,
            "output": str(args.out),
        }
    )


if __name__ == "__main__":
    main()
