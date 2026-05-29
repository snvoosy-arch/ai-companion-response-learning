from __future__ import annotations

from typing import NotRequired, TypeAlias, TypedDict


KeywordChoices: TypeAlias = tuple[tuple[tuple[str, ...], str], ...]
KeywordSlot: TypeAlias = tuple[KeywordChoices, str]
SemanticSlot: TypeAlias = tuple[tuple[str, ...], str]


class DraftSlotBank(TypedDict):
    fixed_slots: dict[str, str]
    keyword_slots: dict[str, KeywordSlot]
    semantic_slots: NotRequired[dict[str, SemanticSlot]]
    variant_slots: dict[str, tuple[str, ...]]
    templates: tuple[str, ...]


DRAFT_SLOT_BANKS: dict[str, DraftSlotBank] = {
    "semantic_comic_exaggeration_record": {
        "fixed_slots": {},
        "keyword_slots": {
            "topic": (
                (
                    (("족보", "가계도"), "족보"),
                    (("조상", "할아버지", "할머니"), "조상님 라인"),
                    (("기록", "문서"), "그 기록"),
                ),
                "그 얘기",
            ),
        },
        "semantic_slots": {
            "metaphor": (
                (
                    "family_lineage",
                    "huge_record",
                    "ancient_record",
                    "exaggeration",
                    "too_long",
                ),
                "팔만대장경급",
            ),
        },
        "variant_slots": {
            "reaction": (
                "비유가 너무 정확해서 웃겨",
                "그 정도면 그냥 기록물이 아니라 문화재야",
                "말이 과한데 묘하게 맞아",
                "듣자마자 그림이 바로 잡혀",
            ),
            "after": (
                "조상님들 단체 출연한 대하드라마 느낌이야",
                "관계도 펼치면 스크롤이 끝까지 안 내려갈 것 같아",
                "가계도라기보다 역사책 목차를 보는 기분이지",
                "한 명씩 따라가다 보면 정신이 먼저 길을 잃어",
            ),
        },
        "templates": (
            "{topic}가 아니라 거의 {metaphor}인데? {reaction}. {after}.",
            "{topic} 얘기면 {metaphor}이라는 말이 꽤 맞아. {after}.",
            "{topic}를 {metaphor}으로 보는 건 인정. {reaction}, {after}.",
        ),
    },
    "workrevenge_group_project_zero_credit": {
        "fixed_slots": {
            "score": "기여도 0%",
        },
        "keyword_slots": {
            "context": (
                (
                    (("피피티", "ppt", "발표자료"), "발표 자료에서"),
                    (("팀플",), "팀플에서"),
                    (("조별과제", "조별 과제"), "조별과제에서"),
                ),
                "그 판에서",
            ),
            "target": (
                (
                    (("무임승차",), "무임승차 빌런"),
                ),
                "아무것도 안 한 사람",
            ),
        },
        "variant_slots": {
            "reaction": (
                "꽤 사이다야",
                "통쾌한 반격이야",
                "말 대신 남긴 영수증 같아",
                "마지막 경고장 같아",
            ),
            "pressure": (
                "그만큼 많이 참은 거",
                "쌓일 만큼 쌓였던 거",
                "기록으로 남길 만큼 선 넘은 거",
                "좋게 넘기기엔 너무 답답했던 거",
            ),
            "aftertaste": (
                "통쾌한데 살짝 씁쓸하지",
                "이해는 되는데 뒷맛은 좀 쓰지",
                "웃기긴 한데 답답함도 같이 보이지",
                "그 정도면 기록으로 남길 만하지",
            ),
        },
        "templates": (
            "{context} {target}한테 {score}를 남긴 건 {reaction}, {pressure}라 {aftertaste}.",
            "{target}에게 {score}라니 {reaction}, {context} 거기까지 갔으면 {pressure}지.",
            "{score}는 센 선택인데, {context} {target} 상대로는 {reaction}, {pressure}라 {aftertaste}.",
        ),
    },
    "daily_food_menu_pick": {
        "fixed_slots": {},
        "keyword_slots": {
            "setup": (
                (
                    (("먹었어", "뭐먹었", "뭘먹었"), "실제로 뭘 먹었다고 꾸미진 않을게, 추천이면"),
                    (("점심",), "점심이면"),
                    (("저녁",), "저녁이면"),
                    (("야식", "밤에"), "야식이면"),
                    (("배고프", "배고파", "배고픈", "배고픈데"), "배고프면"),
                ),
                "ㅋㅋ 뭐 먹지 모드면",
            ),
            "dish": (
                (
                    (("저녁",), "김치찌개"),
                    (("야식", "밤에"), "계란밥"),
                    (("점심", "먹었어", "뭐먹었", "뭘먹었"), "제육덮밥"),
                    (("배고프", "배고파", "배고픈", "배고픈데"), "덮밥"),
                ),
                "김치볶음밥",
            ),
            "reason": (
                (
                    (("점심",), "오후 버틸 힘이 바로 들어와"),
                    (("저녁",), "밥이랑 국물로 정리돼"),
                    (("야식", "밤에"), "부담이 덜 남아"),
                    (("배고프", "배고파", "배고픈", "배고픈데"), "고민을 끊기 좋아"),
                    (("먹었어", "뭐먹었", "뭘먹었"), "따뜻하고 실패 확률 낮아"),
                ),
                "빠르고 익숙해서 고민 끊기 좋아",
            ),
        },
        "variant_slots": {
            "push": (
                "딱 하나만 고르면",
                "오늘은",
                "고민 길게 끌지 말고",
                "지금 텐션이면",
            ),
            "ending": (
                "그걸로 가자",
                "이 선택이 제일 덜 흔들려",
                "이 정도면 충분히 이겨",
                "여기서 더 고민하면 배고픔만 커져",
            ),
        },
        "templates": (
            "{setup} {dish} 가자. {reason}, {ending}.",
            "{push} {dish}. {setup} {reason}. {ending}.",
            "{setup} {dish} 쪽이 좋아. {push} 실패 적은 걸로 끊는 게 맞아.",
        ),
    },
    "daily_weather_umbrella_uncertainty": {
        "fixed_slots": {
            "item": "작은 우산",
        },
        "keyword_slots": {
            "weather": (
                (
                    (("비",), "비 예보"),
                    (("오락가락", "변덕"), "오락가락한 날씨"),
                    (("흐림", "꾸물"), "꾸물꾸물한 날씨"),
                ),
                "날씨",
            ),
        },
        "variant_slots": {
            "judgement": (
                "챙기는 쪽이 이겨",
                "들고 나가는 쪽이 덜 억울해",
                "작게 보험 드는 게 맞아",
                "오늘은 방심하면 젖는 날이야",
            ),
            "after": (
                "안 쓰면 짐이어도 젖는 것보단 낫지",
                "가방에 있으면 마음이라도 편해",
                "비 맞고 후회하는 것보단 훨씬 싸게 먹혀",
                "날씨한테 지는 것보단 낫다",
            ),
        },
        "templates": (
            "{weather}가 애매하면 {item} 하나 챙겨. {judgement}, {after}.",
            "{weather}면 {item}이 거의 생존템이야. {after}.",
            "이런 {weather}엔 {item} 챙기는 게 맞아. {judgement}.",
        ),
    },
    "daily_weather_fatigue_recovery": {
        "fixed_slots": {},
        "keyword_slots": {
            "weather_effect": (
                (
                    (("기압",), "기압"),
                    (("오락가락", "변덕"), "날씨 오락가락"),
                    (("추웠", "더웠", "덥", "춥"), "온도 변덕"),
                ),
                "날씨",
            ),
        },
        "variant_slots": {
            "body_signal": (
                "몸이 먼저 눈치챈 거야",
                "에너지가 먼저 끌려 내려간 거야",
                "컨디션이 날씨한테 끌려간 거야",
                "괜히 축 처지는 게 아니야",
            ),
            "action": (
                "오늘은 회복 쪽으로 잡자",
                "이유 찾기보다 쉬는 쪽이 먼저야",
                "무리하지 말고 템포 낮추자",
                "작은 일만 처리하고 체력부터 살리자",
            ),
        },
        "templates": (
            "{weather_effect} 탓도 꽤 있어. {body_signal}, {action}.",
            "{weather_effect}이 흔들리면 사람도 같이 흔들려. {body_signal}. {action}.",
            "피곤한 게 네 탓만은 아니야. {weather_effect} 영향도 있으니까 {action}.",
        ),
    },
    "fandom_event_waiting": {
        "fixed_slots": {},
        "keyword_slots": {
            "event": (
                (
                    (("팝업스토어", "팝업"), "팝업스토어"),
                    (("동인행사",), "동인 행사"),
                    (("콘서트", "티켓팅"), "티켓팅"),
                ),
                "행사",
            ),
            "condition": (
                (
                    (("평일",), "평일이어도"),
                    (("첫차", "아침6시"), "첫차 타도"),
                    (("웨이팅",), "웨이팅 걸리면"),
                ),
                "막상 열리면",
            ),
        },
        "variant_slots": {
            "reaction": (
                "눈치 게임 시작이지",
                "반응속도 게임이야",
                "덕심 체력전이야",
                "줄 서는 순간부터 보스전이야",
            ),
            "pressure": (
                "최애 장르면 줄이 갑자기 불어나",
                "후기 사진 하나에 마음이 바로 흔들려",
                "놓치면 계속 생각나는 게 문제야",
                "현장 분위기 자체가 지갑을 흔들어",
            ),
        },
        "templates": (
            "{event}는 {condition} {reaction}. {pressure}.",
            "{condition} {event}는 진짜 {reaction}, {pressure}.",
            "{event} 얘기 나오면 이미 {reaction}. {condition} {pressure}.",
        ),
    },
    "fandom_goods_random_duplicate": {
        "fixed_slots": {},
        "keyword_slots": {
            "item": (
                (
                    (("랜덤굿즈",), "랜덤 굿즈"),
                    (("아크릴",), "아크릴"),
                    (("굿즈",), "굿즈"),
                ),
                "랜덤템",
            ),
            "miss": (
                (
                    (("중복",), "중복"),
                    (("최애",), "최애 회피"),
                    (("똥손",), "똥손 인증"),
                ),
                "확률 억까",
            ),
        },
        "variant_slots": {
            "reaction": (
                "진짜 사람 약 올리지",
                "이건 운빨이 아니라 심리전이야",
                "웃기면서도 빡쳐",
                "괜히 한 번 더 뽑고 싶게 만들어",
            ),
            "after": (
                "최애만 피해 가는 손은 너무 얄미워",
                "중복은 볼 때마다 마음이 살짝 꺾여",
                "그래도 교환판을 뒤지게 되는 게 덕질이지",
                "확률표가 갑자기 차갑게 느껴져",
            ),
        },
        "templates": (
            "{item}에서 {miss} 뜨면 {reaction}. {after}.",
            "{item}에서 {miss} 나오면 진짜 {reaction}, {after}.",
            "{item} 뽑기에서 {miss}면 이미 멘탈전이지. {after}.",
        ),
    },
    "pet_food_rejection": {
        "fixed_slots": {},
        "keyword_slots": {
            "pet": (
                (
                    (("고양이", "냥"), "고양이"),
                    (("강아지", "댕"), "강아지"),
                ),
                "반려동물",
            ),
            "object": (
                (
                    (("사료",), "사료"),
                    (("간식",), "간식"),
                    (("장난감",), "장난감"),
                ),
                "새로 준 거",
            ),
            "gesture": (
                (
                    (("모래파",), "모래 파는 시늉"),
                    (("뱉",), "한 입 먹고 뱉기"),
                    (("쳐다도안", "상자"), "포장 상자 선택"),
                ),
                "냉정한 거부",
            ),
        },
        "variant_slots": {
            "reaction": (
                "상전도 이런 상전이 없어",
                "말은 안 해도 리뷰 평점 1점이야",
                "정성은 사람 기준이고 평가는 냉정하지",
                "킹받는데 결국 귀여움이 이겨",
            ),
            "ending": (
                "그래도 귀여워서 진다",
                "다음엔 더 조심스럽게 골라야 해",
                "보호자 마음만 아주 잘 털렸네",
                "이게 바로 집사의 숙명이다",
            ),
        },
        "templates": (
            "{pet}가 {object} 앞에서 {gesture} 하면 {reaction}. {ending}.",
            "{object} 준비했는데 {pet} 반응이 {gesture}이면 {reaction}, {ending}.",
            "{pet} 기준은 진짜 냉정해. {object}에 {gesture}라니 {reaction}.",
        ),
    },
    "shopping_dawn_delivery_anticipation": {
        "fixed_slots": {},
        "keyword_slots": {
            "delivery": (
                (
                    (("쿠팡와우", "와우"), "쿠팡 와우"),
                    (("새벽배송",), "새벽 배송"),
                    (("택배",), "택배"),
                ),
                "배송",
            ),
            "signal": (
                (
                    (("문앞",), "문 앞에 툭 떨어지는 소리"),
                    (("알람",), "알림 뜨는 순간"),
                    (("새벽",), "새벽에 도착했다는 신호"),
                ),
                "도착 신호",
            ),
        },
        "variant_slots": {
            "reaction": (
                "알람보다 강해",
                "잠보다 소비 본능이 먼저 깨",
                "현관 쪽으로 영혼이 먼저 나가",
                "기대감이 바로 켜져",
            ),
            "after": (
                "그 순간은 거의 선물 받는 기분이지",
                "돈은 내가 냈는데 묘하게 선물 같아",
                "새벽 공기까지 갑자기 설레",
                "뜯기 전이 제일 두근거리는 타이밍이야",
            ),
        },
        "templates": (
            "{delivery} {signal}는 {reaction}. {after}.",
            "{signal} 들리면 {delivery}는 이미 승리야. {reaction}, {after}.",
            "{delivery} 기다릴 때 {signal}만큼 반가운 게 없지. {after}.",
        ),
    },
    "shopping_package_opening_dopamine": {
        "fixed_slots": {},
        "keyword_slots": {
            "object": (
                (
                    (("택배상자",), "택배 상자"),
                    (("상자",), "상자"),
                    (("테이프",), "테이프"),
                ),
                "택배",
            ),
            "motion": (
                (
                    (("칼",), "칼로 테이프 쫙 찢는 손맛"),
                    (("뜯",), "뜯는 순간"),
                    (("찢",), "테이프 찢는 소리"),
                ),
                "개봉하는 순간",
            ),
        },
        "variant_slots": {
            "reaction": (
                "현대인의 작은 도파민 맞아",
                "소소한 쾌감이 있어",
                "기대감이 반은 먹고 들어가",
                "이 맛에 택배 기다리지",
            ),
            "after": (
                "뜯기 전 설렘이 진짜 본체야",
                "내용물 보기 전부터 이미 기분이 올라와",
                "상자 하나에 하루 텐션이 바뀌어",
                "괜히 천천히 뜯고 싶어지는 순간이지",
            ),
        },
        "templates": (
            "{object} {motion}은 {reaction}. {after}.",
            "{motion}, 그거 진짜 {reaction}. {object}는 뜯기 전부터 이미 설레.",
            "{object} 열 때 {motion}이 있으면 {reaction}, {after}.",
        ),
    },
    "growth_miracle_morning_fail": {
        "fixed_slots": {},
        "keyword_slots": {
            "challenge": (
                (
                    (("미라클모닝",), "미라클 모닝"),
                    (("새벽6시", "6시"), "새벽 6시 기상"),
                    (("알람",), "알람 기상"),
                ),
                "갓생 도전",
            ),
            "fail": (
                (
                    (("8시더라", "8시"), "눈 떠보니 8시인 상황"),
                    (("끄고",), "알람 끄고 다시 잔 상황"),
                    (("실패",), "시작부터 실패한 상황"),
                ),
                "몸의 거부권이 발동한 상황",
            ),
        },
        "variant_slots": {
            "reaction": (
                "의지 문제가 아니라 몸이 협상 결렬한 거야",
                "처음부터 너무 세게 잡은 거지",
                "몸이 아직 동의서를 안 쓴 거야",
                "갓생보다 수면권이 먼저 이긴 거야",
            ),
            "next": (
                "내일은 30분만 당기는 걸로 다시 가자",
                "6시 말고 7시 30분부터 협상하자",
                "일단 잠드는 시간부터 손보는 게 맞아",
                "실패 말고 난이도 조절로 보자",
            ),
        },
        "templates": (
            "{challenge} 하다가 {fail}이면 {reaction}. {next}.",
            "{fail} 난 {challenge}은 너무 현실적이야. {reaction}, {next}.",
            "{challenge}은 멋진데 {fail}도 흔해. {next}.",
        ),
    },
    "context_pressure_validate_first": {
        "fixed_slots": {},
        "keyword_slots": {
            "topic_phrase": (
                (
                    (("카톡", "답장", "연락", "읽씹", "안읽씹"), "연락 쪽이"),
                    (("친구", "애인", "상사", "동기", "룸메", "사람"), "관계 쪽이"),
                    (("회사", "출근", "퇴근", "학교", "과제", "회의"), "일상 쪽이"),
                    (("우산", "날씨", "비", "춥", "덥"), "날씨가"),
                    (("밥", "메뉴", "먹", "야식", "치킨", "라면", "커피"), "먹는 쪽이"),
                    (("몸", "머리", "배", "피곤", "잠", "아프", "컨디션"), "몸 상태가"),
                    (("쇼핑", "장바구니", "결제", "택배", "물건"), "소비 쪽이"),
                ),
                "그 일이",
            ),
            "pressure": (
                (
                    (("나만그런", "나만이래", "나만신경", "나만불편"), "나만 그런가 싶은 감각"),
                    (("하루종일", "계속", "신경쓰"), "계속 붙잡히는 느낌"),
                    (("예민", "유난"), "괜히 예민한가 싶은 마음"),
                    (("찝찝", "찜찜", "꺼림칙"), "찝찝하게 남는 느낌"),
                ),
                "마음에 걸리는 느낌",
            ),
        },
        "variant_slots": {
            "validation": (
                "그거 너만 그런 거 아냐",
                "그 정도로 걸리면 이유가 있는 거야",
                "그 감각은 그냥 유난이 아니야",
                "마음이 괜히 확대 해석하는 것만은 아니야",
            ),
            "reframe": (
                "예민한 게 아니라 마음이 기준을 찾는 중이야",
                "괜히 까다로운 게 아니라 불편한 지점을 확인하는 중이야",
                "지금은 넘겨도 되는지 판단하느라 신호가 커진 거야",
                "네 안에서 기준선이 켜진 거라고 보면 돼",
            ),
        },
        "templates": (
            "{validation}. {topic_phrase} {pressure}으로 남으면 작은 것도 크게 보이고, 지금은 {reframe}.",
            "{validation}. {topic_phrase} 계속 남는 건 이유가 있어. 지금은 {reframe}.",
            "{topic_phrase} {pressure}으로 남으면 하루 종일 걸릴 수 있어. {validation}, {reframe}.",
        ),
    },
    "context_pressure_low_voice_boundary": {
        "fixed_slots": {},
        "keyword_slots": {
            "burden": (
                (
                    (("말하기애매", "말꺼내기애매"), "말하기 애매한 부담"),
                    (("화내기도애매", "따지기도애매"), "따지기도 애매한 부담"),
                    (("괜히예민", "예민한사람", "괜히유난"), "예민해 보일까 봐 걸리는 마음"),
                    (("참고있", "참는중", "넘기고있", "괜찮은척"), "참고 넘기는 압력"),
                ),
                "낮게 꺼내기 어려운 부담",
            ),
            "action": (
                (
                    (("친구", "애인", "룸메", "상사", "사람"), "'나는 이 부분이 좀 걸렸어' 정도로 낮게 꺼내보는 쪽"),
                    (("직원", "종업원", "카페", "식당"), "'혹시 이것만 부탁드려도 될까요'처럼 사실만 짧게 말하는 쪽"),
                    (("회사", "학교", "과제", "회의"), "'이 부분은 한번 정리하고 가고 싶어요'처럼 톤을 낮추는 쪽"),
                ),
                "'나는 이 부분이 좀 걸렸어' 정도로 낮게 꺼내보는 쪽",
            ),
        },
        "variant_slots": {
            "opening": (
                "그건 말하기 애매해서 더 피곤한 타입이야",
                "그건 세게 말하긴 애매하고 참자니 쌓이는 쪽이야",
                "그런 건 말문 열기 전까지가 제일 피곤해",
                "그 부담은 작아 보여도 은근히 오래 남아",
            ),
            "ending": (
                "세게 따지는 것보다 덜 터져",
                "상대도 방어적으로 덜 받아들여",
                "네 마음도 덜 뭉개져",
                "참고 쌓는 것보단 훨씬 낫다",
            ),
        },
        "templates": (
            "{opening}. {burden} 때문에 커진 거라, {action}이 {ending}.",
            "{opening}. 바로 터뜨리기보다 {action}부터 가면 {ending}.",
            "{burden}이면 진짜 피곤하지. 그래도 {action}이 {ending}.",
        ),
    },
    "context_pressure_name_unease": {
        "fixed_slots": {},
        "keyword_slots": {
            "unease": (
                (
                    (("찝찝",), "찝찝함"),
                    (("찜찜",), "찜찜함"),
                    (("꺼림칙",), "꺼림칙함"),
                    (("마음에걸", "묘하게걸"), "마음에 걸리는 느낌"),
                    (("신경쓰", "계속신경"), "계속 신경 쓰이는 느낌"),
                ),
                "찝찝함",
            ),
            "topic_phrase": (
                (
                    (("카톡", "답장", "연락"), "연락이"),
                    (("친구", "애인", "상사", "동기", "사람"), "관계가"),
                    (("몸", "머리", "배", "피곤", "잠"), "몸 상태가"),
                    (("돈", "결제", "쇼핑", "택배"), "소비 쪽이"),
                    (("방", "집", "청소", "물건"), "집 안 일이"),
                ),
                "그 일이",
            ),
        },
        "variant_slots": {
            "next": (
                "지금은 무시보다 작게 확인하는 쪽이 나아",
                "한 번만 사실 확인하고 내려놓는 게 낫다",
                "확인할 수 있는 건 확인하고, 나머지는 붙잡지 말자",
                "마음속에서만 굴리면 더 커져",
            ),
            "reason": (
                "이유가 있는 거라",
                "그냥 떠오른 게 아니라",
                "어딘가 걸린 지점이 있다는 뜻이라",
                "마음이 신호를 보내는 중이라",
            ),
        },
        "templates": (
            "그 {unease} 그냥 넘기기 어렵지. {topic_phrase} 계속 머리에 남는 건 {reason}, {next}.",
            "{unease}이 남으면 은근히 오래 가. {topic_phrase} 계속 남는 건 {reason}, {next}.",
            "그건 {unease}이라고 이름 붙이는 게 맞아. {topic_phrase} 계속 남는 건 {reason}, {next}.",
        ),
    },
    "context_pressure_social_tact": {
        "fixed_slots": {},
        "keyword_slots": {
            "target": (
                (
                    (("직원", "종업원", "카페", "식당"), "직원 눈치"),
                    (("친구", "애인", "룸메"), "가까운 사람 눈치"),
                    (("상사", "교수", "회사", "학교"), "위 사람 눈치"),
                    (("옆방", "이웃", "사람"), "상대 눈치"),
                ),
                "상대 눈치",
            ),
            "request": (
                (
                    (("시끄럽", "소음"), "불편한 점은"),
                    (("휴지", "물티슈", "주문", "메뉴"), "필요한 건"),
                    (("약속", "취소", "시간"), "내 일정에 생긴 불편은"),
                    (("답장", "연락"), "연락에서 걸린 부분은"),
                ),
                "필요한 건",
            ),
        },
        "variant_slots": {
            "permission": (
                "그래도 필요한 건 말해도 돼",
                "그래도 네 불편이 사라지는 건 아니야",
                "그렇다고 계속 네가 삼킬 필요는 없어",
                "조용히 말한다고 민폐가 되는 건 아니야",
            ),
            "style": (
                "짧게 한 문장으로",
                "낮은 톤으로 사실만",
                "부드럽게 요청형으로",
                "감정 빼고 상황만",
            ),
        },
        "templates": (
            "{target} 보이면 더 애매하지. {permission}. {request} {style} 말하면 돼.",
            "{target} 때문에 망설이는 거 이해돼. 그래도 {request} {style} 꺼내는 게 낫다.",
            "{target}는 진짜 사람 멈칫하게 해. 그래도 {permission}, {style} 가자.",
        ),
    },
    "context_pressure_choose_practical": {
        "fixed_slots": {},
        "keyword_slots": {
            "choice": (
                (
                    (("먹을까말까", "메뉴", "밥", "야식"), "먹는 선택"),
                    (("살까말까", "결제", "장바구니", "쇼핑"), "사는 선택"),
                    (("갈지말지", "약속", "여행", "나갈까"), "갈지 말지"),
                    (("챙길지", "우산", "겉옷"), "챙길지 말지"),
                    (("처리할까", "할까말까", "일", "과제"), "처리할지 말지"),
                ),
                "그 선택",
            ),
            "criterion": (
                (
                    (("돈", "결제", "비싸", "가격"), "돈 나가고도 후회가 덜한 쪽"),
                    (("피곤", "컨디션", "잠"), "내일 컨디션을 덜 망치는 쪽"),
                    (("비", "우산", "날씨"), "나중에 덜 젖고 덜 후회할 쪽"),
                    (("관계", "친구", "애인", "상사"), "관계가 덜 꼬이는 쪽"),
                ),
                "나중에 덜 후회할 쪽",
            ),
        },
        "variant_slots": {
            "opening": (
                "애매할 땐 완벽한 정답보다 기준이 먼저야",
                "고민이 길어지면 선택지도 같이 흐려져",
                "이럴 땐 멋진 선택보다 덜 후회하는 선택이 이겨",
                "계속 멈춰 있으면 피로만 늘어나",
            ),
            "action": (
                "손해가 작은 선택으로 끊자",
                "오늘은 기준 하나만 잡고 결정하자",
                "지금 감당 가능한 쪽으로 가자",
                "작게 정하고 다음 행동으로 넘기자",
            ),
        },
        "templates": (
            "{opening}. {choice}은 {criterion}으로 보고, {action}.",
            "{choice}이 애매하면 {criterion}이 기준이야. {action}.",
            "{opening}. {criterion}을 고르고, {action}.",
        ),
    },
}
