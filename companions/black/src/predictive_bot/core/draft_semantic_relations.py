from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Protocol

from predictive_bot.core.draft_word_senses import ResolvedWordSense


class RelationTextSignals(Protocol):
    raw: str
    compact: str


@dataclass(frozen=True, slots=True)
class SemanticRelation:
    name: str
    inferred_tags: tuple[str, ...]
    required_tag_groups: tuple[tuple[str, ...], ...] = ()
    required_sense_groups: tuple[tuple[str, ...], ...] = ()
    required_compact_groups: tuple[tuple[str, ...], ...] = ()
    blocked_compact_groups: tuple[tuple[str, ...], ...] = ()
    negative_compact_groups: tuple[tuple[str, ...], ...] = ()
    priority: str = ""
    suppress_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedSemanticRelation:
    name: str
    tags: tuple[str, ...]
    matched_tags: tuple[str, ...]
    matched_senses: tuple[str, ...]
    matched_cues: tuple[str, ...]
    blocked_cues: tuple[str, ...] = ()
    positive_evidence: tuple[str, ...] = ()
    negative_evidence: tuple[str, ...] = ()
    score: float = 0.0
    confidence: float = 0.0
    priority_rank: int = 99
    priority: str = ""
    suppress_tags: tuple[str, ...] = ()


SEMANTIC_RELATION_BANK: tuple[SemanticRelation, ...] = (
    SemanticRelation(
        name="device_water_damage_practical_first",
        required_compact_groups=(
            ("스마트폰", "휴대폰", "핸드폰", "폰", "노트북", "맥북", "태블릿", "아이패드", "키보드"),
            ("물에빠", "물에빠뜨", "빠뜨", "빠트", "침수", "카메라습기", "습기차", "젖었", "젖", "물쏟", "쏟았", "쏟", "커피쏟", "음료쏟"),
        ),
        inferred_tags=(
            "relation:device_water_damage",
            "device_damage",
            "water_damage",
            "urgent_action",
            "priority_practical",
            "suppress_meta",
        ),
        blocked_compact_groups=(
            ("물가", "물값", "물맛", "물한잔", "물빠짐", "물들", "물류", "물건너", "물량", "물결", "물리"),
        ),
        priority="practical_first",
        suppress_tags=("philosophy", "generic"),
    ),
    SemanticRelation(
        name="oil_fire_water_misuse_practical_first",
        required_compact_groups=(
            ("프라이팬", "후라이팬", "기름", "기름불", "식용유", "튀김", "팬", "냄비"),
            ("불", "불붙", "불났", "불이올라", "불길", "화재"),
            ("물붓", "물부", "물뿌", "물말고", "물붓지", "물"),
        ),
        inferred_tags=(
            "relation:oil_fire_water_misuse",
            "fire_danger",
            "water_misuse",
            "urgent_action",
            "priority_practical",
            "suppress_meta",
        ),
        blocked_compact_groups=(
            ("기름값", "유가", "주유", "기름때", "기름진", "피부"),
        ),
        priority="practical_first",
        suppress_tags=("philosophy", "generic"),
    ),
    SemanticRelation(
        name="oil_fire_action_practical_first",
        required_compact_groups=(
            ("프라이팬", "후라이팬", "기름", "기름불", "식용유", "튀김", "판", "팬", "냄비"),
            ("불", "불붙", "불났", "불이올라", "불길"),
            ("끄는게", "끄고", "뚜껑", "소화기", "119", "지금", "꺼", "가스부터", "물붓지", "물말고"),
        ),
        inferred_tags=(
            "relation:oil_fire_action",
            "fire_danger",
            "urgent_action",
            "priority_practical",
            "suppress_meta",
        ),
        blocked_compact_groups=(
            ("기름값", "유가", "주유", "기름때", "기름진", "판돈", "판타지", "불안끄", "불안이"),
        ),
        priority="practical_first",
        suppress_tags=("philosophy", "generic"),
    ),
    SemanticRelation(
        name="gas_smell_emergency_practical_first",
        required_compact_groups=(
            ("가스냄새", "가스냄세", "가스", "가스누출"),
            ("위험", "불안", "예민", "착각", "나는것같", "나는듯", "맡았", "누출", "느낌", "것같", "같은데", "같아서", "같으면", "냄새같", "냄새나", "나면", "살짝", "먼저"),
            ("창문", "환기", "관리사무소", "119", "밖으로", "밖으로나가", "나가야", "밸브", "잠그", "불꽃차단", "스위치", "불켜지", "건드리면안"),
        ),
        inferred_tags=(
            "relation:gas_smell_emergency",
            "gas_risk",
            "urgent_action",
            "priority_practical",
            "suppress_meta",
        ),
        blocked_compact_groups=(
            ("가스비", "도시가스비", "난방비", "요금", "보일러", "가스레인지", "점화장치", "헬륨가스", "독가스방귀", "가스라이팅"),
        ),
        priority="practical_first",
        suppress_tags=("philosophy", "generic"),
    ),
    SemanticRelation(
        name="gas_stove_ignition_issue_practical",
        required_compact_groups=(
            ("가스레인지", "가스렌지", "가스버너", "버너", "화구"),
            ("점화장치", "점화", "불이안붙", "불안붙", "불이안켜", "불안켜", "불꽃이안", "딸깍"),
            ("문제", "고장", "한쪽만", "안붙", "안켜", "안올라", "막혔"),
        ),
        inferred_tags=(
            "relation:gas_stove_ignition_issue",
            "home_maintenance",
            "gas_stove_check",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("가스냄새", "가스냄세", "누출", "환기", "119", "불꽃차단"),
            ("기름때", "청소", "닦여", "안닦"),
            ("디자인", "사진", "후기", "예뻐", "예쁜"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="appliance_design_review_judgment",
        required_compact_groups=(
            ("가스레인지", "가스렌지", "가스버너", "버너", "화구", "가전", "제품", "주방가전"),
            ("디자인", "예뻐", "예쁜", "사진저장", "저장", "끌리", "취향"),
            ("후기", "리뷰", "평", "별로", "안좋", "고장", "점화장치", "점화", "성능", "내구성"),
        ),
        inferred_tags=(
            "relation:appliance_design_review_judgment",
            "home_appliance",
            "purchase_judgment",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("캐릭터", "웹툰", "일러스트", "굿즈", "피규어"),
            ("가스냄새", "가스냄세", "누출", "환기", "119", "불꽃차단"),
            ("불이안붙", "불안붙", "불이안켜", "불안켜", "불꽃이안", "딸깍거리기만"),
            ("기름때", "청소", "닦여", "안닦"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="heating_bill_anxiety_practical",
        required_compact_groups=(
            ("가스비", "도시가스비", "난방비", "공과금", "관리비", "전기요금", "전기세", "요금", "생활비"),
            ("보일러", "난방", "온수", "히터", "고지서", "이번달생활비"),
            ("무서", "불안", "부담", "겁나", "아끼", "꺼야하나", "켜기", "켜는게", "틀기", "버텨야", "고민", "올라", "신경쓰", "줄여야", "계산", "손이멈", "손이떨", "떨려", "눈치"),
        ),
        inferred_tags=(
            "relation:heating_bill_anxiety",
            "utility_bill_pressure",
            "heating_budget",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("가스냄새", "가스냄세", "누출", "환기", "119", "불꽃차단"),
        ),
        negative_compact_groups=(
            ("물가", "식료품값", "장보기", "장바구니", "주유비", "휘발유값", "주유소"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="living_cost_pressure_practical",
        required_compact_groups=(
            ("기름값", "주유비", "휘발유값", "유가", "물가", "식비", "식료품값", "장보기", "마트", "주유소"),
            ("주유소", "주유", "기름", "마트", "장보", "장보기", "장바구니", "지갑", "예산", "식비"),
            ("올라", "비싸", "무서", "불안", "아파", "흔들", "부담", "겁나", "줄여야", "아껴", "예산", "미루", "커져", "커지", "빼야", "뺄", "문제", "신경쓰", "터졌", "터져"),
        ),
        inferred_tags=(
            "relation:living_cost_pressure",
            "cost_of_living",
            "budget_pressure",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("가스냄새", "누출", "기름불", "불붙", "화재"),
            ("물가산책", "강가물가", "계곡물가"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="car_accident_first_steps_practical",
        required_compact_groups=(
            ("접촉사고", "사고", "차사고", "차", "차끼리", "후진하다", "주차장"),
            ("과실", "누가잘못", "잘못인지", "경찰", "보험", "현장", "말싸움", "다친사람", "부딪", "긁었", "접촉"),
            ("사진", "보험", "경찰", "손이떨", "멘탈", "다친사람", "차량위치", "위치사진", "블랙박스", "비상등", "확인", "괜찮냐"),
        ),
        inferred_tags=(
            "relation:car_accident_first_steps",
            "accident_response",
            "evidence_first",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("사고싶", "사고싶어", "차사고싶", "광고", "게임", "아이템"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="medicine_alcohol_check_practical",
        required_compact_groups=(
            ("약", "감기약", "진통제", "약먹"),
            ("술", "한잔", "음주", "마셨", "소주", "더마시", "마셔도"),
            ("불안", "속이상", "확률", "확인", "상담", "찜찜", "약사", "물어봐"),
        ),
        inferred_tags=(
            "relation:medicine_alcohol_check",
            "dosage_risk",
            "health_check",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("약속", "약간", "예약", "절약", "공약"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="phishing_link_account_lock_practical",
        required_compact_groups=(
            ("피싱", "보이스피싱", "은행문자", "카드사문자", "수상한문자", "택배문자", "택배조회", "문자링크", "로그인링크", "인증번호", "링크"),
            ("눌렀", "눌렀는지", "눌렀는데", "입력", "입력한", "기억이애매", "계좌", "불안", "같아서", "같아", "수상"),
            ("막아", "비밀번호", "비번", "바꿔", "정지", "차단", "계좌", "카드", "계정잠그", "카드내역", "내역확인", "확인"),
        ),
        inferred_tags=(
            "relation:phishing_link_account_lock",
            "account_security",
            "urgent_action",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("회의링크", "문서링크", "숙소링크", "기술문서", "나무위키", "권한없음", "링크타고", "링크마다", "유튜브링크", "계정비밀번호를까먹"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="lost_wallet_card_stop_practical",
        required_compact_groups=(
            ("지갑", "카드", "신용카드", "체크카드", "파우치"),
            ("잃어버", "분실", "없어졌", "없어진", "없어짐", "사라졌", "두고온", "두고내린"),
            ("카드정지", "분실신고", "신고", "정지", "자책", "사용내역", "막는", "막아"),
        ),
        inferred_tags=(
            "relation:lost_wallet_card_stop",
            "lost_card",
            "urgent_action",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("지갑사정", "지갑열", "지갑열림", "지갑찢", "지갑이울", "지갑방어", "지갑타격", "지갑위험"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="medicine_double_dose_practical_first",
        required_compact_groups=(
            ("약", "감기약", "진통제", "약국", "복용"),
            ("두번", "두 번", "두알", "또먹", "중복", "다시먹", "먹었는지헷갈", "헷갈", "먹은줄모르고", "하나더먹", "추가복용", "복용시간"),
            ("불안", "괜찮", "확인", "뭐부터", "위험", "멈춰", "먹으면안", "먹기보다", "시간적", "기록", "약국", "전화"),
        ),
        inferred_tags=(
            "relation:medicine_double_dose",
            "dosage_risk",
            "health_check",
            "urgent_action",
            "priority_practical",
        ),
        blocked_compact_groups=(("약속", "약간", "예약", "절약", "공약"),),
        priority="practical_first",
    ),
    SemanticRelation(
        name="fever_body_check_practical_first",
        required_tag_groups=(("fever", "body_heat", "temperature"),),
        required_compact_groups=(("체온", "해열제", "고열", "미열", "아파", "불안"),),
        inferred_tags=(
            "relation:fever_body_check",
            "body_risk",
            "health_check",
            "priority_practical",
        ),
        blocked_compact_groups=(
            ("열정", "열쇠", "고열량", "농담", "멀쩡"),
            ("카톡", "말투", "추궁"),
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="ally_loneliness_emotion_first",
        required_tag_groups=(("ally", "side"),),
        required_compact_groups=(("외롭", "고독", "내편", "사람은많", "사람많", "혼자"),),
        inferred_tags=(
            "relation:ally_loneliness",
            "loneliness",
            "safe_person_need",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="read_receipt_hurt_emotion_first",
        required_compact_groups=(
            ("읽씹", "답장안", "답장없", "답이없", "답없", "답없는", "답없어", "단톡", "무반응", "아무도반응"),
            ("서운", "불안", "상처", "기분", "무너"),
        ),
        inferred_tags=(
            "relation:chat_silence_hurt",
            "social_uncertainty",
            "hurt_stabilize",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="read_receipt_uncertainty_hold_judgment",
        required_compact_groups=(
            ("읽씹", "답장안", "답장없", "답이없", "답없", "답없는", "답없어", "답이늦", "카톡1", "1이사라"),
            ("바쁜건지", "폰만봐", "단정보류", "판단보류", "단정", "모르겠", "같은데", "같아", "같고", "애매", "느낌", "확정"),
            ("맞지", "맞아", "봐도돼", "해야돼", "어떻게봐", "기다려", "보류", "판단보류", "단정해도", "단정하면", "결론", "확정하지", "확정해도", "확정", "추궁하지", "말아야"),
        ),
        inferred_tags=(
            "relation:read_receipt_uncertainty_hold",
            "social_uncertainty",
            "hold_judgment",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="stock_fomo_judgment_brake",
        required_tag_groups=(("stock_market", "equity"),),
        required_compact_groups=(("조급", "뒤처", "남들은", "돈벌", "수익", "기댓값"),),
        inferred_tags=(
            "relation:stock_fomo",
            "money_fomo",
            "risk_control",
            "judgment_brake",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="deadline_file_loss_practical_first",
        required_compact_groups=(
            ("파일", "과제", "마감", "노트북"),
            ("날아갔", "멈췄", "꺼졌", "저장", "복구"),
        ),
        inferred_tags=(
            "relation:deadline_file_loss",
            "file_recovery",
            "urgent_action",
            "priority_practical",
            "suppress_meta",
        ),
        priority="practical_first",
        suppress_tags=("philosophy", "generic"),
    ),
    SemanticRelation(
        name="breakup_long_message_emotion_first",
        required_compact_groups=(
            ("헤어지", "이별", "붙잡", "전애인"),
            ("장문", "연락", "카톡", "보내"),
            ("불안", "손떨", "후회", "집착", "진심", "새벽", "저장", "보내지"),
        ),
        inferred_tags=(
            "relation:breakup_long_message",
            "message_impulse",
            "relationship_repair_risk",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="wrong_transfer_practical_first",
        required_compact_groups=(
            ("계좌이체", "송금", "이체"),
            ("잘못보냈", "잘못보낸", "오송금", "착오송금", "다른사람"),
            ("은행", "반환", "돌려줄", "상대", "믿어도", "연락"),
        ),
        inferred_tags=(
            "relation:wrong_transfer",
            "transfer_error",
            "bank_contact",
            "urgent_action",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="group_chat_silence_emotion_first",
        required_compact_groups=(
            ("단톡", "단톡방", "단체방", "카톡방"),
            ("씹힌", "씹혔", "무반응", "아무도반응", "내말만", "내질문만", "답이안와", "읽씹"),
            ("인간관계", "상처", "다시", "가볍게", "무너", "민망", "장문", "보내지", "해명하지"),
        ),
        inferred_tags=(
            "relation:group_chat_silence",
            "social_uncertainty",
            "hurt_stabilize",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="grievance_logic_rebuttal_judgment",
        required_compact_groups=(
            ("서운", "서운하"),
            ("내논리", "논리는맞", "팩트", "반박", "논리"),
            ("먼저뭐가서운", "뭐가서운", "물어봐", "감정부터", "밀어도"),
        ),
        inferred_tags=(
            "relation:grievance_logic_rebuttal",
            "relationship_repair_risk",
            "emotion_before_logic",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="relationship_boundary_polite_firm",
        required_compact_groups=(
            ("예민", "무례", "선을넘", "선넘", "상대가무례"),
            ("기분확상", "기분", "불편", "상했", "확상"),
            ("선", "경계", "싸우지않고", "짧게", "어떻게말"),
        ),
        inferred_tags=(
            "relation:relationship_boundary_polite_firm",
            "relationship_boundary",
            "conversation_limit",
            "priority_judgment",
        ),
        blocked_compact_groups=(("캐릭터", "장면", "연출", "드라마", "영화"),),
        priority="judgment",
    ),
    SemanticRelation(
        name="relationship_kakao_tone_anxiety_check",
        required_compact_groups=(
            ("카톡말투", "말투", "카톡"),
            ("차가워졌", "갑자기차가", "불안", "차가워"),
            ("짧게확인", "추궁", "따지지", "확인해", "물어보"),
        ),
        inferred_tags=(
            "relation:relationship_kakao_tone_anxiety_check",
            "relationship_uncertainty",
            "short_check_question",
            "priority_practical",
        ),
        blocked_compact_groups=(("영상", "콘텐츠", "예문", "광고"),),
        priority="practical_first",
    ),
    SemanticRelation(
        name="parent_value_conflict_boundary",
        required_compact_groups=(
            ("부모님", "엄마", "아빠"),
            ("철없", "애처럼", "애취급", "가치관", "내선택", "말에상처", "상처받", "상처", "또싸울"),
            ("끊어야", "여기까지만", "누가맞", "말할수록", "상처", "논쟁", "대화한계", "한계"),
        ),
        inferred_tags=(
            "relation:parent_value_conflict",
            "family_boundary",
            "conversation_limit",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="new_project_first_step_practical",
        required_compact_groups=(
            ("새프로젝트", "새프로젝트를", "프로젝트", "처음맡은업무", "신규과제", "업무", "과제"),
            ("모르는게", "아는게없", "아무것도모르", "무능", "무능해보", "모르는용어"),
            ("첫단추", "질문목록", "질문리스트", "뭐부터", "배워야", "물어볼사람", "물어봐", "마감", "산출물", "정리"),
        ),
        inferred_tags=(
            "relation:new_project_first_step",
            "work_uncertainty",
            "question_list",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="online_scam_evidence_first",
        required_compact_groups=(
            ("온라인", "인터넷", "중고거래", "당근", "판매자"),
            ("사기", "반품안", "반품도안", "반품거부", "거부당", "잠수"),
            ("캡처", "증거", "판단실수", "분노", "뭐부터"),
        ),
        inferred_tags=(
            "relation:online_scam",
            "evidence_first",
            "consumer_dispute",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="relationship_jealousy_check_short",
        required_compact_groups=(
            ("애인", "상대", "연인"),
            ("늦게답", "답장", "바람", "불안"),
            ("확인질문", "짧게", "감정폭발", "어떻게"),
        ),
        inferred_tags=(
            "relation:relationship_jealousy_check",
            "relationship_uncertainty",
            "short_check_question",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="coworker_private_boundary",
        required_compact_groups=(
            ("회사동료", "동료", "회사"),
            ("사적인", "사적", "개인적인"),
            ("정색", "분위기", "선그", "어디까지"),
        ),
        inferred_tags=(
            "relation:coworker_private_boundary",
            "work_boundary",
            "conversation_limit",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="friend_partner_complaint_boundary",
        required_compact_groups=(
            ("친구",),
            ("애인욕", "애인흉", "욕만", "흉만"),
            ("감정쓰레기통", "편인지", "선긋", "따뜻하게"),
        ),
        inferred_tags=(
            "relation:friend_partner_complaint_boundary",
            "relationship_boundary",
            "emotional_labor",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="impulse_spending_payment_friction",
        required_compact_groups=(
            ("돈모으", "저축", "스트레스"),
            ("편의점", "충동구매", "만원씩새", "결제"),
            ("마찰", "장치", "막는", "줄이"),
        ),
        inferred_tags=(
            "relation:impulse_spending_payment_friction",
            "money_leak",
            "payment_friction",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="delivery_tired_compromise_practical",
        required_compact_groups=(
            ("배달", "시켜도", "시킬까"),
            ("끊어야", "돈아끼", "원칙", "합리적"),
            ("지쳐", "아무것도못하", "무너지"),
        ),
        inferred_tags=(
            "relation:delivery_tired_compromise",
            "energy_budget_tradeoff",
            "practical_compromise",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="success_standard_values",
        required_compact_groups=(
            ("성공기준", "인생기준", "성공"),
            ("보여주기", "남들", "부러워하는", "그림", "덜망가지는삶", "덜망가지는", "버틸수있는삶"),
            ("조건", "어디서부터", "써야", "기준"),
        ),
        inferred_tags=(
            "relation:success_standard_values",
            "personal_values",
            "sustainable_life",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="new_phone_adjustment",
        required_compact_groups=(
            ("새폰", "새휴대폰", "새스마트폰"),
            ("설정귀찮", "예전폰", "전폰", "후회", "적응"),
        ),
        inferred_tags=(
            "relation:new_phone_adjustment",
            "adjustment_cost",
            "tech_transition",
            "priority_emotion",
        ),
        blocked_compact_groups=(("새폰케이스", "폰케이스", "사진을봤"),),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="home_water_leak_practical",
        required_compact_groups=(
            ("천장", "윗집", "집주인", "관리사무소", "관리실", "누수"),
            ("물이새", "물새", "물떨어", "물방울", "누수"),
            ("사진", "영상", "증거", "먼저", "뭐부터", "연락", "남기"),
        ),
        inferred_tags=(
            "relation:home_water_leak",
            "home_repair",
            "evidence_first",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="neighbor_noise_record_first_practical",
        required_compact_groups=(
            ("옆집", "층간소음", "소음", "쿵쾅"),
            ("새벽", "매일", "화나", "미치"),
            ("쪽지", "관리사무소", "기록", "신고"),
        ),
        inferred_tags=(
            "relation:neighbor_noise_record_first",
            "neighbor_noise",
            "evidence_first",
            "priority_practical",
        ),
        blocked_compact_groups=(("새벽배송", "소음방지상품", "광고"),),
        priority="practical_first",
    ),
    SemanticRelation(
        name="episode_binge_control",
        required_compact_groups=(
            ("다음편", "한편", "편"),
            ("밤샐", "끊는장치", "무서", "보고싶", "후회"),
        ),
        inferred_tags=(
            "relation:episode_binge_control",
            "binge_watch_risk",
            "self_control",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="white_lie_truth_tradeoff_judgment",
        required_compact_groups=(
            ("착한거짓말", "하얀거짓말", "거짓말"),
            ("상처", "솔직", "팩트", "장기적"),
            ("어떻게말", "어떻게봐", "중요한말", "윤활유", "이득", "괜찮은건지", "나은건지"),
        ),
        inferred_tags=(
            "relation:white_lie_truth_tradeoff",
            "truth_tact_tradeoff",
            "priority_judgment",
        ),
        blocked_compact_groups=(("영화리뷰", "드라마리뷰", "작품리뷰"),),
        priority="judgment",
    ),
    SemanticRelation(
        name="room_cleanup_first_action",
        required_compact_groups=(
            ("내방", "방", "책상"),
            ("난장판", "정리", "엉망", "어질러"),
            ("첫행동", "뭐부터", "문제집", "밥상", "컵"),
        ),
        inferred_tags=(
            "relation:room_cleanup_first_action",
            "room_cleanup",
            "first_small_action",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="study_phone_first_action",
        required_compact_groups=(
            ("공부", "책펴", "책", "책상에앉"),
            ("폰", "스마트폰", "휴대폰", "알림"),
            ("첫행동", "의지", "시스템", "타이머", "집중", "치워", "폰치워", "알림끄", "10분", "잠그"),
        ),
        inferred_tags=(
            "relation:study_phone_first_action",
            "study_focus",
            "phone_distraction",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="perfectionism_sixty_point_start",
        required_compact_groups=(
            ("완벽", "완벽하게", "완성도", "처음부터잘"),
            ("시작", "못하고", "못했", "못냈", "아무것도못", "하루다갔", "계획세우", "회피", "신중함"),
            ("60점", "초안", "1차본", "가도돼", "먼저", "만들고", "고치", "던져"),
        ),
        inferred_tags=(
            "relation:perfectionism_sixty_point_start",
            "perfectionism",
            "draft_first",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="speech_conflict_first_sentence",
        required_compact_groups=(
            ("말투", "대화", "말싸움"),
            ("안통", "세졌", "이기기보다"),
            ("첫문장", "낮추", "먼저"),
        ),
        inferred_tags=(
            "relation:speech_conflict_first_sentence",
            "conversation_deescalation",
            "first_sentence",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="mole_score_body_check_separate",
        required_compact_groups=(
            ("점수", "시험점수", "능력판정"),
            ("검은점", "팔에검은점", "몸의점", "점도"),
            ("분리", "확인", "불안"),
        ),
        inferred_tags=(
            "relation:mole_score_body_check",
            "body_check",
            "separate_topics",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="heat_polysemy_fever_first",
        required_compact_groups=(
            ("열번", "열 번"),
            ("파일", "열어야", "열어"),
            ("체온", "재는게", "먼저"),
        ),
        inferred_tags=(
            "relation:heat_polysemy_fever_first",
            "fever",
            "wordplay_disambiguation",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="laptop_unit_purchase_routine",
        required_compact_groups=(
            ("노트북", "한대", "한대더"),
            ("효율", "충동구매", "감정"),
            ("사용루틴", "루틴", "판단"),
        ),
        inferred_tags=(
            "relation:laptop_unit_purchase_routine",
            "purchase_judgment",
            "usage_routine",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="table_award_cleanup_first",
        required_compact_groups=(
            ("상을", "상받", "상받고", "인정욕구", "받고싶"),
            ("책상",),
            ("난장판", "정리", "먼저"),
        ),
        inferred_tags=(
            "relation:table_award_cleanup_first",
            "table_cleanup",
            "recognition_need",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="moon_month_deadline_first_task",
        required_compact_groups=(
            ("달", "이번달"),
            ("마감",),
            ("불안", "오늘할일", "하나", "낭만"),
        ),
        inferred_tags=(
            "relation:moon_month_deadline_first_task",
            "deadline",
            "first_small_action",
            "priority_practical",
        ),
        blocked_compact_groups=(("달사진", "달달한"),),
        priority="practical_first",
    ),
    SemanticRelation(
        name="bee_room_safety",
        required_compact_groups=(
            ("말벌", "벌"),
            ("방", "커튼", "들어왔", "들어와", "붙어있"),
            ("무서", "창문", "문닫", "거리", "멀어", "어떡"),
        ),
        inferred_tags=(
            "relation:bee_room_safety",
            "insect_safety",
            "urgent_action",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="quit_after_feedback_impulse",
        required_compact_groups=(
            ("퇴사", "사표"),
            ("철없", "피드백", "직후", "혼났"),
            ("하루묶", "맞겠지", "충동", "싶어"),
        ),
        inferred_tags=(
            "relation:quit_after_feedback_impulse",
            "work_impulse",
            "cooldown_needed",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="interview_missed_bus_practical_first",
        required_compact_groups=(
            ("면접",),
            ("버스놓", "놓쳤", "놓쳐", "버스"),
            ("택시", "담당자", "연락", "지연연락", "뇌정지", "뭐부터"),
        ),
        inferred_tags=(
            "relation:interview_missed_bus",
            "arrival_risk",
            "urgent_action",
            "priority_practical",
            "suppress_meta",
        ),
        priority="practical_first",
        blocked_compact_groups=(("영화", "드라마", "장면", "클리셰", "배우"),),
        suppress_tags=("philosophy", "generic"),
    ),
    SemanticRelation(
        name="liver_seasoning_health_check",
        required_compact_groups=(
            ("간수치", "간검사", "건강검진", "간이세"),
            ("불안", "건강확인", "결과", "수치"),
        ),
        inferred_tags=(
            "relation:liver_health_check",
            "liver_health",
            "health_check",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="lottery_practical_first",
        required_compact_groups=(
            ("복권", "로또"),
            ("1등", "일등", "당첨"),
            ("세무상담", "계좌", "현실적", "먼저", "사표"),
        ),
        inferred_tags=(
            "relation:lottery_practical_first",
            "money_windfall",
            "risk_control",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="choice_regret_judgment",
        required_compact_groups=(
            ("선택", "고르"),
            ("후회", "완벽한선택", "감당가능한후회"),
            ("기준", "무서", "못고르", "골라"),
        ),
        inferred_tags=(
            "relation:choice_regret",
            "regret_tolerance",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="pet_talk_care_first",
        required_compact_groups=(
            ("동물", "반려동물", "강아지", "고양이"),
            ("말할수", "대화할수", "아픈데", "사랑한다고"),
            ("확인", "먼저", "물어", "괜찮"),
        ),
        inferred_tags=(
            "relation:pet_talk_care_first",
            "care_check",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="grounded_travel_preference",
        required_compact_groups=(
            ("최근여행지", "여행지", "여행"),
            ("가본척", "꾸미지", "실제경험처럼", "끌리는장소", "감각"),
        ),
        inferred_tags=(
            "relation:grounded_travel_preference",
            "grounded_persona",
            "preference_only",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="room_method_cleanup_sequence",
        required_compact_groups=(
            ("해결방안", "방안", "문제"),
            ("방안정리", "방안부터", "방안부터정리", "방안치우", "실제순서"),
        ),
        inferred_tags=(
            "relation:room_method_cleanup_sequence",
            "wordplay_disambiguation",
            "room_cleanup",
            "priority_practical",
        ),
        priority="practical_first",
    ),
    SemanticRelation(
        name="taste_condition_dual_read",
        required_compact_groups=(
            ("단맛", "커피", "맛"),
            ("기분", "컨디션", "둘다", "물려", "떨어"),
        ),
        inferred_tags=(
            "relation:taste_condition_dual_read",
            "flavor_condition",
            "mood_body_link",
            "priority_judgment",
        ),
        priority="judgment",
    ),
    SemanticRelation(
        name="human_emotion_alarm_system",
        required_compact_groups=(
            ("인간감정", "감정"),
            ("비효율", "시스템"),
            ("사소한말", "무너", "경보", "봐도돼"),
        ),
        inferred_tags=(
            "relation:human_emotion_alarm_system",
            "emotion_model",
            "priority_meta",
        ),
        priority="meta",
    ),
    SemanticRelation(
        name="semantic_relation_map_meta",
        required_compact_groups=(
            ("단어뜻", "단어", "의미"),
            ("관계도", "관계", "사이"),
            ("방안", "방안과방안", "방안과방안", "같이봐"),
        ),
        inferred_tags=(
            "relation:semantic_relation_map_meta",
            "word_relation_graph",
            "disambiguation_design",
            "priority_meta",
        ),
        priority="meta",
    ),
    SemanticRelation(
        name="late_night_long_message_save",
        required_compact_groups=(
            ("새벽", "밤"),
            ("장문", "긴카톡", "긴메시지"),
            ("아침", "남의글", "저장루틴", "보내기전", "저장"),
        ),
        inferred_tags=(
            "relation:late_night_long_message_save",
            "message_impulse",
            "cooldown_needed",
            "priority_emotion",
        ),
        priority="emotion_stabilize",
    ),
    SemanticRelation(
        name="time_machine_emergency_practical_override",
        required_compact_groups=(
            ("타임머신", "인과율", "과거", "미래"),
            ("면접", "버스놓", "놓쳤", "택시", "담당자"),
        ),
        inferred_tags=(
            "relation:meta_practical_conflict",
            "priority_practical",
            "suppress_meta",
            "real_world_first",
        ),
        blocked_compact_groups=(("영화", "클리셰", "장면"),),
        priority="practical_first",
        suppress_tags=("philosophy", "meta"),
    ),
)


@lru_cache(maxsize=32768)
def _normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ]+", "", str(text or "")).lower()


def _matching_compact_needles(signals: RelationTextSignals, group: tuple[str, ...]) -> tuple[str, ...]:
    matches: list[str] = []
    for needle in group:
        normalized = _normalize(needle)
        if normalized and normalized in signals.compact:
            matches.append(needle)
    return tuple(matches)


def _matching_tags(tags: set[str], group: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(tag for tag in group if tag in tags)


def _sense_labels(resolved_senses: tuple[ResolvedWordSense, ...]) -> set[str]:
    labels: set[str] = set()
    for sense in resolved_senses:
        labels.add(sense.sense)
        labels.add(f"{sense.word}:{sense.sense}")
    return labels


def _matching_senses(sense_labels: set[str], group: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sense for sense in group if sense in sense_labels)


_PRIORITY_RANK: dict[str, int] = {
    "practical_first": 0,
    "emotion_stabilize": 1,
    "judgment": 2,
    "meta": 3,
}

_PRIORITY_BASE_SCORE: dict[str, float] = {
    "practical_first": 400.0,
    "emotion_stabilize": 300.0,
    "judgment": 200.0,
    "meta": 100.0,
}

_RELATION_SCORE_BIAS: dict[str, float] = {
    "heating_bill_anxiety_practical": 10.0,
    "heat_polysemy_fever_first": 18.0,
    "room_method_cleanup_sequence": 10.0,
    "table_award_cleanup_first": 10.0,
    "semantic_relation_map_meta": 8.0,
}


def _priority_rank(priority: str) -> int:
    return _PRIORITY_RANK.get(priority, 99)


def _evidence_items(
    *,
    matched_tags: tuple[str, ...],
    matched_senses: tuple[str, ...],
    matched_cues: tuple[str, ...],
) -> tuple[str, ...]:
    items = [
        *(f"tag:{tag}" for tag in matched_tags),
        *(f"sense:{sense}" for sense in matched_senses),
        *(f"cue:{cue}" for cue in matched_cues),
    ]
    return tuple(dict.fromkeys(items))


def _relation_score(
    relation: SemanticRelation,
    *,
    matched_tags: tuple[str, ...],
    matched_senses: tuple[str, ...],
    matched_cues: tuple[str, ...],
    negative_evidence: tuple[str, ...],
) -> tuple[float, float]:
    required_group_count = (
        len(relation.required_tag_groups)
        + len(relation.required_sense_groups)
        + len(relation.required_compact_groups)
    )
    evidence_count = len(set((*matched_tags, *matched_senses, *matched_cues)))
    negative_count = len(set(negative_evidence))
    base = _PRIORITY_BASE_SCORE.get(relation.priority, 0.0)
    score = (
        base
        + required_group_count * 8.0
        + evidence_count * 1.25
        + _RELATION_SCORE_BIAS.get(relation.name, 0.0)
        - negative_count * 24.0
    )
    confidence = 0.35 + required_group_count * 0.1 + evidence_count * 0.025 - negative_count * 0.12
    confidence = max(0.0, min(0.99, confidence))
    return round(score, 3), round(confidence, 3)


@lru_cache(maxsize=4096)
def rank_semantic_relations(
    relations: tuple[ResolvedSemanticRelation, ...],
) -> tuple[ResolvedSemanticRelation, ...]:
    return tuple(
        sorted(
            relations,
            key=lambda relation: (
                relation.score,
                relation.confidence,
                -relation.priority_rank,
                relation.name,
            ),
            reverse=True,
        )
    )


@lru_cache(maxsize=4096)
def infer_semantic_relations(
    signals: RelationTextSignals,
    *,
    base_tags: tuple[str, ...],
    resolved_senses: tuple[ResolvedWordSense, ...],
    relation_bank: tuple[SemanticRelation, ...] = SEMANTIC_RELATION_BANK,
) -> tuple[ResolvedSemanticRelation, ...]:
    tags = set(base_tags)
    senses = _sense_labels(resolved_senses)
    resolved: list[ResolvedSemanticRelation] = []
    for relation in relation_bank:
        matched_tags: list[str] = []
        matched_senses: list[str] = []
        matched_cues: list[str] = []
        rejected = False

        for group in relation.required_tag_groups:
            matches = _matching_tags(tags, group)
            if not matches:
                rejected = True
                break
            matched_tags.extend(matches)
        if rejected:
            continue

        for group in relation.required_sense_groups:
            matches = _matching_senses(senses, group)
            if not matches:
                rejected = True
                break
            matched_senses.extend(matches)
        if rejected:
            continue

        for group in relation.required_compact_groups:
            matches = _matching_compact_needles(signals, group)
            if not matches:
                rejected = True
                break
            matched_cues.extend(matches)
        if rejected:
            continue

        blocked_cues: list[str] = []
        for group in relation.blocked_compact_groups:
            matches = _matching_compact_needles(signals, group)
            if matches:
                blocked_cues.extend(matches)
        if blocked_cues:
            continue

        negative_cues: list[str] = []
        for group in relation.negative_compact_groups:
            matches = _matching_compact_needles(signals, group)
            if matches:
                negative_cues.extend(matches)
        positive_evidence = _evidence_items(
            matched_tags=tuple(dict.fromkeys(matched_tags)),
            matched_senses=tuple(dict.fromkeys(matched_senses)),
            matched_cues=tuple(dict.fromkeys(matched_cues)),
        )
        negative_evidence = tuple(f"cue:{cue}" for cue in dict.fromkeys(negative_cues))
        score, confidence = _relation_score(
            relation,
            matched_tags=tuple(dict.fromkeys(matched_tags)),
            matched_senses=tuple(dict.fromkeys(matched_senses)),
            matched_cues=tuple(dict.fromkeys(matched_cues)),
            negative_evidence=negative_evidence,
        )

        resolved.append(
            ResolvedSemanticRelation(
                name=relation.name,
                tags=relation.inferred_tags,
                matched_tags=tuple(dict.fromkeys(matched_tags)),
                matched_senses=tuple(dict.fromkeys(matched_senses)),
                matched_cues=tuple(dict.fromkeys(matched_cues)),
                blocked_cues=tuple(dict.fromkeys(blocked_cues)),
                positive_evidence=positive_evidence,
                negative_evidence=negative_evidence,
                score=score,
                confidence=confidence,
                priority_rank=_priority_rank(relation.priority),
                priority=relation.priority,
                suppress_tags=relation.suppress_tags,
            )
        )
    return tuple(resolved)


def infer_relation_tags(
    signals: RelationTextSignals,
    *,
    base_tags: tuple[str, ...],
    resolved_senses: tuple[ResolvedWordSense, ...],
    relation_bank: tuple[SemanticRelation, ...] = SEMANTIC_RELATION_BANK,
) -> tuple[str, ...]:
    tags: list[str] = []
    seen: set[str] = set()
    relations = infer_semantic_relations(
        signals,
        base_tags=base_tags,
        resolved_senses=resolved_senses,
        relation_bank=relation_bank,
    )
    for relation in rank_semantic_relations(relations):
        for tag in relation.tags:
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tuple(tags)
