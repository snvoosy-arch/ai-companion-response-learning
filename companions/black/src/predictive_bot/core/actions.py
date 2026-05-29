from __future__ import annotations

import re

from predictive_bot.core.models import (
    ActionDecision,
    ActionType,
    ConversationState,
    Goal,
    Intent,
    MessageFeatures,
)


class ActionSelector:
    def __init__(self, default_location: str | None = None) -> None:
        self.default_location = default_location

    _activity_place_options = {
        "바다": ("물놀이", "모래사장 산책", "사진 찍기", "돗자리 펴고 쉬기"),
        "해변": ("물놀이", "모래사장 산책", "사진 찍기", "돗자리 펴고 쉬기"),
        "해수욕장": ("물놀이", "모래사장 산책", "사진 찍기", "돗자리 펴고 쉬기"),
        "계곡": ("발 담그기", "물가 산책", "간단한 간식", "사진 찍기"),
        "공원": ("산책", "가벼운 공놀이", "사진 찍기", "돗자리 펴고 쉬기"),
        "한강": ("산책", "자전거", "돗자리 펴고 쉬기", "간단한 간식"),
        "산": ("가벼운 산책", "전망 보기", "사진 찍기", "간식 먹기"),
        "캠핑장": ("불멍", "간단한 요리", "산책", "보드게임"),
        "놀이공원": ("가벼운 놀이기구", "퍼레이드 보기", "사진 찍기", "간식 먹기"),
        "실내": ("보드게임", "영화 보기", "간단한 간식", "카페에서 쉬기"),
        "도서관": ("근처 카페 들르기", "가벼운 산책", "서점 구경", "간단한 보드게임"),
        "카페": ("디저트 나눠 먹기", "짧은 보드게임", "사진 찍기", "근처 산책"),
    }
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
    _open_persona_question_markers = (
        "오늘하루중",
        "친구라고부르면",
        "좋은친구",
        "친구가되어줄",
        "몸이굳은느낌",
        "핑계를대고",
        "대신결론내리지",
        "시간여행",
        "좋아하는계절",
        "영향받은책",
        "영향을받은책",
        "영향받은영화",
        "영향을받은영화",
        "어디로든여행",
        "장래희망",
        "새롭게배운",
        "관심이생긴분야",
        "스트레스를받을때",
        "복권에당첨",
        "한가지음식",
        "표현하는단어",
        "초능력",
        "아침형인간",
        "올빼미형",
        "잘했다고생각하는결정",
        "무인도",
        "좋아하는색깔",
        "존경하는실존인물",
        "역사적인물",
        "인생좌우명",
        "이번주말",
        "반려동물",
        "크게웃",
        "혼자있는시간",
        "이상적인하루",
        "익스트림스포츠",
        "최고의선물",
        "삶을책으로쓴다면",
        "커피와차",
        "두려워하는것",
        "어린시절",
        "오늘밤자기전에",
        "동물과대화",
        "모든전기",
        "인생을영화장르",
        "절대실패하지",
        "마음을읽는능력",
        "투명인간",
        "나만의행성",
        "좀비",
        "위대한발명품",
        "스마트폰없이",
        "미래로가서",
        "외계인이지구에와서",
        "지구의대표음식",
        "악기",
        "복제인간",
        "살이찌지않",
        "다른사람으로살",
        "비가오는날듣고싶",
        "비오는날듣고싶",
        "우주여행",
        "달과화성",
        "새로운과목",
        "똑같은옷",
        "10년전으로돌아",
        "동화속세계",
        "내가만든요리",
        "말한마디도하지않고",
        "인터넷전혀안하기",
        "10년뒤의나에게편지",
        "법을바꿀수있다면",
        "세계일주",
        "잠을자지않아도",
        "새로운언어",
        "10만원을줍",
        "십만원을줍",
        "처음으로내돈",
        "내돈을주고샀던물건",
        "엘리베이터에갇혔",
        "일기장을훔쳐보",
        "보여주기싫은페이지",
        "1억원을기부",
        "일억원을기부",
        "나를가장잘아는친구",
        "한문장으로설명",
        "모든동물이멸종",
        "한종만살릴",
        "최근1년동안내삶",
        "긍정적인변화",
        "눈을감고지금당장",
        "세가지단어",
        "내가가진단점",
        "거짓말을100",
        "알아채는능력",
        "100살까지살수있는알약",
        "백살까지살수있는알약",
        "삶을한편의다큐멘터리",
        "다큐멘터리의제목",
        "단한가지향기",
        "인내심스위치",
        "당신은참좋은사람",
        "내가만든물건중",
        "가장자랑스러운것",
        "외계인이나를납치",
        "지구를그리워하게",
        "어른이되었다고",
        "실감했던순간",
        "용서를받을수있는쿠폰",
        "무조건적인용서",
        "전자기기",
        "천재란",
        "역사책에당신의이름",
        "한줄남는다면",
        "가장피하고싶은주제",
        "실력이늘지않아서포기",
        "도저히실력이늘지않",
        "10년뒤의세상",
        "사라졌으면하는것",
        "정반대의성향",
        "단짝친구가될수",
        "운이좋았다",
        "노래가사한줄",
        "마음을울린적",
        "똑같은생각을하는사람",
        "오늘하루를한가지색깔",
        "양치질안하기",
        "샤워안하기",
        "생각을영상",
        "모든언어를완벽",
        "낡은램프",
        "지니",
        "잠을잘수없는",
        "12시간무조건",
        "충성하는용",
        "동물로언제든변신",
        "백과사전한권",
        "통기타한대",
        "100억이든통장",
        "20살로어려진",
        "모든음식이매운맛",
        "모든음식이단맛",
        "시트콤",
        "같은영화",
        "비밀의방",
        "거짓말을들을때마다",
        "귀에서삐",
        "처음으로교신",
        "지구대표",
        "성공하는식당",
        "유령도시",
        "겨울옷입고여름",
        "여름옷입고겨울",
        "모든기억을가진채",
        "다시태어난다면",
        "데스노트",
        "하늘을나는자동차",
        "순간이동장치",
        "눈물이진주",
        "슬픈영화",
        "연예인과평생친구",
        "연예인과한달",
        "과거의위인",
        "저녁을대접",
        "남에게들리지않는방",
        "음악장르",
        "라면을끓일때마다",
        "계란프라이",
        "명장면",
        "인터넷이영원히사라지는",
        "새로운과일",
        "지구가멸망",
        "10억원이생겼는데",
        "10억이생겼는데",
        "백그라운드음악",
        "bgm",
        "영화의조연",
        "외계인친구",
        "지구음식",
        "48시간",
        "마법의물약",
        "텔레파시",
        "구름을타고",
        "가장귀여운동물",
        "원하는나이",
        "성별이바뀌",
        "스마트폰없이살기",
        "에어컨히터없이",
        "감기달고살기",
        "만성소화불량",
        "고기못먹기",
        "밀가루",
        "원할때투명인간",
        "거짓말할때마다",
        "야외수영장",
        "패딩입고등산",
        "절친의전애인",
        "전애인의절친",
        "음악없이살기",
        "영상없이살기",
        "짝사랑",
        "혼자서세계여행",
        "내방에서만놀기",
        "50확률로100억",
        "100확률로100만원",
        "매력포인트",
        "불리고싶",
        "한계를뛰어넘",
        "삶의원칙",
        "잘맞는친구",
        "사과를할때",
        "인생을3막",
        "남은수명이딱1년",
        "고마움의대상",
        "위로하는",
        "샤워할때",
        "단골노래",
        "카카오톡",
        "유튜브를제외",
        "자주켜는앱",
        "피자가장자리",
        "도우를남기는편",
        "비행기탈때",
        "창가자리",
        "통로자리",
        "냉장고에무조건",
        "혼자밥을먹어야",
        "카페에가면",
        "잠이안올때",
        "수면유도",
        "계절이바뀔때",
        "소확행아이템",
        "유치원생시절",
        "등짝을때려",
        "꽉안아",
        "10년뒤의나를상상",
        "학생시절",
        "짓궂은장난",
        "생생하게기억나는꿈",
        "통장잔고",
        "어릴적에진심으로믿",
        "묘비명",
        "어제하루중",
        "무의미하게보냈다고",
        "5년전에했던고민",
        "50개의질문",
        "음료수",
        "앱중하나로태어난다면",
        "10m이내에접근",
        "자동으로들리는배경음악",
        "사소하지만절대적인매너",
        "수첩에내일의로또번호",
        "로또번호가적혀",
        "영혼이바뀐다면",
        "웃음표현을절대쓸수없",
        "방안에있는물건들뿐",
        "쓸모없는초능력",
        "관찰예능프로그램",
        "직접지은집",
        "흙냄새",
        "갓인쇄된책냄새",
        "주유소기름냄새",
        "허락도없이소스를",
        "초봄하나로고정",
        "늦가을하나로고정",
        "퍼레이드카",
        "공룡시대",
        "서기3000년",
        "칵테일",
        "밤하늘의별",
        "우주의끝",
        "게임속npc",
        "가위를낼수없는",
        "바위를낼수없는",
        "유튜브구독자100만",
        "층수버튼을다누르고",
        "뷔페에가서",
        "무협지의기연",
        "0칼로리인치킨",
        "숙취없는최고급와인",
        "직립보행",
        "문방구앞오락기",
        "뽑기기계",
        "넘어졌는데아무도못본줄",
        "아아아이스아메리카노",
        "뜨아따뜻한아메리카노",
        "은행에찾으러갈때",
        "머릿속을가장강렬하게",
        "일일강사",
        "자신있게가르칠",
        "돈주고도못살경험",
        "게이지바",
        "상태를나타내는",
        "학교소풍",
        "수학여행",
        "장기자랑",
        "방에있는물건들이",
        "스포일러를강제로",
        "줄거리요약본",
        "어깨에기대",
        "번호를물어",
        "동물원의동물",
        "과거의나에게딱세글자",
        "뷔페에갔는데접시",
        "딱3가지",
        "젓가락질을못",
        "숟가락을못",
        "완벽하게부합하는이상형",
        "싫어하는음식",
        "조선시대의노비",
        "조선시대의왕",
        "샤워기물온도",
        "친구들과사진",
        "마법의지우개",
        "흑역사",
        "난방안되는방",
        "모기10마리",
        "템플스테이",
        "로봇청소기",
        "노래방에서마지막1분",
        "서프라이즈파티",
        "애착물건",
        "알람소리",
        "투명망토",
        "쪽지나롤링페이퍼",
        "작은돌멩이",
        "양말이물에젖은",
        "콘서트vip",
        "맨뒷자리",
        "질문을계속받는지금",
        "자는모습을몰래동영상",
        "흑역사가찍혔을확률",
        "유리창이나거울에비친",
        "좋아하는라면조리법",
        "우주의통치자",
        "샤워후거울",
        "흥얼거리는cm송",
        "민트초코에밥",
        "재채기가나올것같다가",
        "엘리베이터에나혼자",
        "내가만약자판기",
        "주운usb",
        "외계인의지구침공계획서",
        "튀겨진야채",
        "배신감",
        "바지지퍼",
        "직성이풀리는루틴",
        "선물을줄때",
        "매일일기를써야하는법",
        "영화속악당",
        "작은틈새",
        "중요한물건을빠뜨려",
        "겨울에아이스크림",
        "뜨거운붕어빵",
        "동물원의사육사",
        "왕자님",
        "내가만약공책",
        "나혼자가서플렉스",
        "모든음식에케첩",
        "모든음식에마요네즈",
        "10원짜리동전",
        "비밀아지트",
        "머리를쓰다듬어",
        "스스로칭찬",
        "초특급슈퍼스타",
        "당신의얼굴을알아보",
        "뮤지컬대사처럼노래",
        "시트콤웃음소리",
        "텔레비전을보며",
        "게임속최종보스",
        "미래에살게될집",
        "설탕대신소금",
        "소금대신설탕",
        "일대기를책으로출판",
        "단하나의거짓말",
        "그림자가나를떠나",
        "63빌딩",
        "두려워하는귀신",
        "장롱면허",
        "페라리를운전",
        "비오는날만있는도시",
        "눈오는날만있는도시",
        "로봇과인간이구분되지않는",
        "접시를딱한번",
        "핸드폰케이스라면",
        "중고로산것처럼흠집",
        "포장지가절대안뜯어지는",
        "이건마법이다",
        "모른척하는거지",
        "왼쪽굽이1cm낮",
        "한쪽다리가3cm짧",
        "지구의춤",
        "연예인의집에몰래",
        "코를파는모습",
        "이상한강박증",
        "결말5분을못보는",
        "첫10분을못보는",
        "지구가멸망한다는뉴스",
        "도로위에있는신호등",
        "나자신만을위해쓸것",
        "도플갱어",
        "무지개색줄무늬티셔츠",
        "눈을한번깜빡이는동안",
        "완전히똑같은옷",
        "상하의신발까지",
        "한부위만로봇부품",
        "로봇부품으로교체",
        "모든과일이사람처럼말",
        "수다스러울것같은과일",
        "주차장에서욕을하며화내",
        "이마에자막",
        "조선시대말투",
        "하오체",
        "합쇼체",
        "비욘세",
        "singleladies",
        "무한위장",
        "부수러갈맛집",
        "놀이터의그네",
        "풋풋한시절의부모님",
        "매일조금씩물건을잃어버",
        "내방에모르는물건",
        "정수리냄새",
        "치킨부위",
        "두개다먹어버린다면",
        "절대그주식사지마",
        "주식을하는사람",
        "구해줄배가1년뒤",
        "궁극기",
        "필살기",
        "패딩없이얇은티셔츠",
        "두꺼운패딩입기",
        "비둘기요정",
        "투명한요정친구",
        "컴컴한방에서혼자살",
        "아무하고도연락하지않고",
        "맨앞자리에서만보기",
        "무조건서서보기",
        "커피를쏟았다",
        "세탁비핑계",
        "민트초코맛이나는저주",
        "파인애플피자맛이나는저주",
        "빨간불인데1시간",
        "롤플레잉게임",
        "rpg",
        "샴푸를짜서머리에",
        "물이끊겼다면",
        "건물이갑자기우주선",
        "우주로날아간다면",
        "10원짜리동전100만개",
        "흩뿌려져있는걸발견",
        "아주사소한일",
        "떠오르는색깔세가지",
        "친구와신나게수다",
        "내친구가아니라",
        "똑같은옷을입은모르는사람",
        "험담을길게썼는데",
        "본인에게전송",
        "상사본인에게",
        "물결표",
        "3개씩의무",
        "로맨틱코미디",
        "첫만남장소",
        "100만유튜버",
        "아이템3가지",
        "의미심장하게웃",
        "모기한마리",
        "헛기침소리",
        "한도초과",
        "지갑에현금도없",
        "좋아하는음식의맛",
        "샴푸맛",
        "최고급스테이크맛",
        "내묘비",
        "빈칸에들어갈말",
        "새끼드래곤",
        "알에서갓깨어난",
        "신발끈이매일5번",
        "바지지퍼가매일1번",
        "저신천지",
        "사이비",
        "동요를작곡",
        "그림자가나에게반항",
        "내자리에모르는사람",
        "내음식을먹고있",
        "첫마디를아니그게아니라",
        "10년만에구조",
        "검색해볼단어",
        "지옥철",
        "할머니와동시에눈",
        "탬버린",
        "백댄서를해줄수있는요정",
        "최악의이별통보",
        "머리를감지않아도",
        "찰랑찰랑한머릿결",
        "양치를안해도",
        "너냄새나",
        "동물들과대화",
        "100만원만빌려",
        "당첨사실을말하고",
        "지구인들의행동",
        "이해안가는행동",
        "05배속",
        "2배속으로만",
        "아픈척하며바닥",
        "퀘스트를완료",
        "보상아이템",
        "닫힘버튼",
        "열림버튼",
        "대답하기귀찮",
        "이런걸왜물어봐",
        "싫어하는직장상사",
        "싫어하는교수님",
        "눈이마주친상태에서닫힘버튼",
        "방귀소리",
        "트럼펫소리",
        "바닐라향",
        "뒤에서누군가박수를",
        "쿨하게일어나는순간",
        "핸드폰배터리라면",
        "1가남았는데도",
        "충전기를안꽂고",
        "미지근한물로만샤워",
        "1분마다5초씩얼음물",
        "아끼는물건",
        "살짝부러뜨렸",
        "줄넘기쌩쌩이",
        "내특기가오직",
        "짱구목소리",
        "와맛있다",
        "블루투스연결이끊겨",
        "폰스피커로노래",
        "라면면발이라면",
        "꼬들꼬들하게익히기전에",
        "카카오톡메시지가부모님께",
        "검색어가내이마",
        "사자가나를보고",
        "앞발을모아하트",
        "비대신미스트",
        "눈대신팝콘",
        "오글거리는감성글귀",
        "방벽지가모두",
        "첫접시에디저트",
        "디저트만잔뜩",
        "길고양이라면",
        "츄르를주기위해",
        "세계를구할준비",
        "요원님",
        "잠옷만입고외출",
        "풀정장",
        "집에서쉬기",
        "마법의리모컨",
        "일시정지",
        "되감기5분",
        "닭다리두개를연속",
        "합리적인제지방법",
        "양말이물에살짝젖",
        "작은모래알",
        "지구를정복하기전에",
        "치명적인약점",
        "머리카락굵기를1mm",
        "쓸데없는초능력",
        "내것이훨씬맛있어보일때",
        "한입줄건가",
        "10년전오늘",
        "쓸데없는고민",
        "책상모서리라면",
        "발가락을찧고",
        "드라마의마지막화",
        "영화의반전",
        "외계비행선",
        "나와똑같이생겼다면",
        "하루에10번씩",
        "파이팅을외쳐야",
        "상상력이방전",
        "상상질문에시달리",
        "만원이라겨우탔는데",
        "가장안쪽구석",
        "내릴게요",
        "과일의씨가무조건수박씨",
        "오돌뼈가박혀",
        "모든사람의이름이김철수",
        "연락처에있는",
        "내가만약모기라면",
        "얄미운인간의유형",
        "우산없이걷기",
        "눈오는날반팔",
        "음식이상했다",
        "셰프가나를뚫어지게",
        "트로트메들리",
        "모든알람",
        "10만원을빌려",
        "자명종시계라면",
        "머리를쾅쾅",
        "시험공부안하고놀고",
        "딱한대만때릴수",
        "칫솔대신손가락",
        "주방세제쓰기",
        "텅빈지하철칸",
        "바짝붙어앉",
        "당첨용지를잃어버",
        "그걸삼켰다면",
        "팝콘대신생쌀",
        "커피대신소금물",
        "길거리전봇대라면",
        "꼴보기싫은일",
        "재채기를멈출수없",
        "화장실변기가막혔",
        "뚫을도구가전혀없",
        "아니근데",
        "어쩔티비",
        "벽장문이열리더니",
        "와이파이가안터진다면",
        "다리가3개들어있",
        "모든신발에작은돌멩이",
        "바지지퍼가반쯤",
        "냉장고안의반찬통",
        "곰팡이가피어",
        "방귀뀌고싶어하는생각",
        "주변10m",
        "요리를해줬는데맛이정말끔찍",
        "어때라고물어",
        "자판기가하나",
        "버튼이딱하나",
        "앞머리가일자로만",
        "뒷머리가까치집",
        "귀여운강아지",
        "정색하고나를물려고",
        "싫어하던음식브랜드",
        "모델이되어있다면",
        "겨울에에어컨",
        "여름에보일러",
        "지금이질문을읽고있는",
        "당신의자세",
        "물티슈와휴지가없",
        "종업원은너무바빠",
        "중학교졸업사진",
        "우주최강귀요미",
        "1일마법사",
        "기억속에서나를완전히지울",
        "모기향이라면",
        "10년뒤의내가유튜브채널",
        "매운거먹고울고",
        "신발에물이살짝스며",
        "엉덩이부분이살짝찢",
        "100만원짜리명품백",
        "몰래카메라가설치",
        "눕기만하면정신이말똥",
        "서있으면무조건1분",
        "길거리의쓰레기통",
        "어제그일은비밀",
        "오페라톤",
        "강남스타일말춤",
        "나침반돋보기성냥",
        "성냥중단하나",
        "자기핸드폰만보고",
        "액정보호필름이라면",
        "기포를남긴채",
        "오늘운세",
        "홀로그램",
        "휴지가사포",
        "수건이항상물에젖",
        "옆자리커플",
        "크게싸워",
        "지우개라면",
        "샤프심으로찌를",
        "전국방송",
        "짱구춤",
        "질척거리는진밥",
        "덜익은된밥",
        "이름이똑같",
        "시계바늘이라면",
        "세계구급악당두목",
        "세계급악당두목",
        "히터고장난버스",
        "에어컨고장난지하철",
        "우리치킨먹으러갈래",
        "배가너무부른상태",
        "안경이라면",
        "김서린채",
        "은행을털수있는기회",
        "재수없는저주",
        "제일싫어하는연예인",
        "침대밑에서낯선사람의일기장",
        "내일을예언",
        "상상력의한계테스트",
        "뇌는지금어떤상태",
        "지독한방귀를뀌고내렸",
        "다음층에서다른사람이탈",
        "볼륨이항상최대치",
        "진동모드로만고정",
        "좋아하는음식코너",
        "새로채워질때까지기다려야",
        "버려진껌이라면",
        "밟고지나가는사람",
        "쓰레기통을발로찼",
        "진짜고양이가들어",
        "상대방의말을두번씩반복",
        "앵무새병",
        "앞머리없는삭발머리",
        "게임플레이중계방송",
        "귓가에서",
        "엄마아빠하고달려와안겼",
        "진짜부모님",
        "최악의노래실력",
        "양말을무조건짝짝이",
        "하루에10시간씩투덜",
        "함께떨어진사람",
        "마음속잔고",
        "잔고가보이는초능력",
        "1시간30분뒤에오는저주",
        "식어서오는저주",
        "치과의사라면",
        "사탕을먹겠다는환자",
        "삑삑이신발소리",
        "걸을때마다",
        "축가를부르게",
        "mr이고장",
        "무반주",
        "보일러를켜면에어컨",
        "에어컨을켜면보일러",
        "텐트에서살기",
        "거울속의내가",
        "잘살고있냐",
        "닭의모든뼈가연골",
        "흐물흐물",
        "궁서체로삐뚤빼뚤",
        "거꾸로만써지는",
        "모기장이라면",
        "모기들이나를뚫으려",
        "칭찬하는소리만안들리는",
        "사준옷이정말내취향이아니고",
        "촌스럽다면",
        "마법의램프",
        "빨리하나만말해",
        "바지의주머니가막혀",
        "상의에주머니가5개",
        "넘어진사람을도와주려고",
        "도망간다면",
        "마지막뉴스앵커",
        "지구가멸망하기10분전",
        "겨울에선풍기",
        "여름에전기장판",
        "가장떠오르는영화대사",
        "질문폭탄에시달린",
        "우리다섯명밖에없지",
        "음산하게웃",
        "셀카모드로만고정",
        "화질이144p",
        "싫어하는한가지재료",
        "억지로라도드실",
        "세워진마네킹",
        "옷을갈아입혀주는",
        "가로수를쳤",
        "통째로뽑혀",
        "헬륨가스마신목소리",
        "다스베이더목소리",
        "문워크로돌기",
        "동네한바퀴",
        "asmr먹방소리",
        "길잃은고양이못보셨어요",
        "변기라면",
        "스카치테이프로만신발",
        "자기자랑만하는사람",
        "10시간씩자기자랑",
        "남은수명",
        "택배가무조건옆집",
        "3일씩늦게오는",
        "유아인머리",
        "미용사인데",
        "치킨냄새가진동",
        "축사를맡게",
        "원고를잃어버렸",
        "전기장판없이살기",
        "에어컨선풍기없이",
        "오늘밤파티준비",
        "스프가하나도없고",
        "면만5봉지",
        "유치원생수준",
        "공포스러운분위기",
        "단한가지소원",
        "속으로부르는노래",
        "나사실외계인이야",
        "병원에데려갈",
        "어떤음료수가나오길",
        "나를쳐다보지마세요",
        "흰색쫄쫄이",
        "당신드디어왔군",
        "양말을허물벗듯",
        "겨울에아이스아메리카노",
        "여름에뜨거운국밥",
        "꿈에나올까봐두려운",
        "소름돋는상상",
        "비보잉브레이크댄스",
        "3초간정지",
        "이모티콘을절대쓸수없는",
        "이모티콘만쓸수있고",
        "향수냄새가너무좋",
        "심호흡을하다가",
        "알람을끄고다시자는",
        "흑염룡",
        "이불킥예약",
        "데헷",
        "크큭",
        "내이야기를아주재미있게엿듣",
        "각색해서떠들",
        "코고는소리",
        "24시간동안",
        "쓰레기봉투에코를박",
        "스마트폰배터리1%상태",
        "틱톡을켤",
        "신발좌우를바꿔",
        "포기할신발",
        "춤만추는댄서",
        "오늘배변여부",
        "테이프로100바퀴",
        "상자가열린채",
        "데이트가있는데망쳐주세요",
        "앞이안보인다",
        "축의금을내려고봉투",
        "영수증뭉치",
        "반팔입고덜덜",
        "목도리칭칭",
        "지퍼열렸었어",
        "한강라면",
        "물조절에실패",
        "눈을반쯤감은",
        "심령사진",
        "모기가나를뚫지못하고분노",
        "어떤비웃음",
        "속으로욕하는소리",
        "전생에뽀로로",
        "크롱",
        "나도방금램프에서쫓겨났어",
        "목덜미상표",
        "발가락봉제선",
        "드디어덫에걸렸군",
        "무거운책을올려놓고탈",
        "겨울에선글라스",
        "여름에털장갑",
        "현실로도피",
    )

    @staticmethod
    def _default_reason_code_for_action(action: ActionType) -> str:
        return {
            ActionType.ASK_LOCATION: "weather.ask_location.default",
            ActionType.WEATHER_LOOKUP: "weather.lookup.default",
            ActionType.WEATHER_UNAVAILABLE: "weather.unavailable.default",
            ActionType.EXPLAIN_CAPABILITIES: "capability.explain.default",
            ActionType.DEESCALATE: "safety.deescalate.default",
            ActionType.DIRECT_REPLY: "reply.direct.default",
            ActionType.ASK_CLARIFICATION: "clarify.ask.default",
            ActionType.ACKNOWLEDGE: "acknowledge.default",
            ActionType.ANSWER_IDENTITY: "identity.answer.default",
            ActionType.CONTINUE_CONVERSATION: "conversation.continue.default",
            ActionType.EXPLAIN_REASON: "explanation.reason.default",
            ActionType.SHARE_FEELING: "feeling.share.default",
            ActionType.SHARE_OPINION: "opinion.share.default",
            ActionType.ACCEPT_ACTIVITY_INVITE: "activity.invite.default",
            ActionType.GAME_CHAT: "game.chat.default",
            ActionType.GAME_ACCEPT_OR_DECLINE: "game.invite.default",
            ActionType.MUSIC_CHAT: "music.chat.default",
            ActionType.RECOMMEND: "recommend.default",
            ActionType.REACT_LAUGH: "reaction.laugh.default",
            ActionType.REACT_SURPRISE: "reaction.surprise.default",
            ActionType.TEASE_BACK: "reaction.tease_back.default",
            ActionType.TELL_TIME: "time.tell.default",
            ActionType.SEARCH_ANSWER: "knowledge.search.default",
            ActionType.NEWS_ANSWER: "knowledge.news.default",
            ActionType.SMALL_TALK: "smalltalk.short.default",
        }.get(action, f"action.{action.value}.default")

    @classmethod
    def is_open_persona_question(cls, features: MessageFeatures) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", features.normalized or features.content).lower()
        if not compact:
            return False
        if not features.is_question and not any(marker in compact for marker in ("다면", "싶", "고른다면", "vs")):
            return False
        return any(marker in compact for marker in cls._open_persona_question_markers)

    @staticmethod
    def _is_playful_reaction_reply_request(features: MessageFeatures) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", features.normalized or features.content).lower()
        return bool(compact) and any(
            marker in compact
            for marker in (
                "어떻게반응",
                "반응해줄래",
                "어떻게리액션",
                "리액션할래",
            )
        )

    @staticmethod
    def _is_performance_culture_observation(features: MessageFeatures) -> bool:
        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", features.normalized or features.content).lower()
        if not compact:
            return False
        culture_markers = (
            "gv",
            "관객과의대화",
            "커튼콜",
            "앵콜",
            "떼창",
            "공연",
            "콘서트",
            "티켓팅",
            "스탠딩",
            "뮤지컬",
            "페스티벌",
            "락페",
            "전시",
            "미술관",
            "시사회",
            "팬미팅",
            "하이터치",
            "사인회",
        )
        if not any(marker in compact for marker in culture_markers):
            return False
        recommendation_markers = (
            "추천",
            "뭐볼",
            "뭐보지",
            "볼만한",
            "볼거",
            "뭐가좋",
        )
        return not any(marker in compact for marker in recommendation_markers)

    @staticmethod
    def _has_rule_hit(features: MessageFeatures, rule_hit: str) -> bool:
        evidence = features.classifier_evidence
        return evidence is not None and rule_hit in set(evidence.rule_hits)

    @staticmethod
    def _decision(
        *,
        action: ActionType,
        reason: str,
        goals: list[Goal],
        response_style: str,
        reason_code: str | None = None,
        reason_flags: list[str] | None = None,
        slots: dict[str, str] | None = None,
        awaiting_slot: str | None = None,
    ) -> ActionDecision:
        return ActionDecision(
            action=action,
            reason=reason,
            goals=goals,
            reason_code=reason_code or ActionSelector._default_reason_code_for_action(action),
            reason_flags=list(reason_flags or []),
            slots=slots or {},
            response_style=response_style,
            awaiting_slot=awaiting_slot,
        )

    def materialize(
        self,
        action: ActionType,
        features: MessageFeatures,
        state: ConversationState,
        goals: list[Goal],
        *,
        reason_override: str | None = None,
    ) -> ActionDecision:
        reason = reason_override or self._default_reason_for_action(action, features, state)

        if action == ActionType.ASK_LOCATION:
            return self._decision(
                action=action,
                reason=reason,
                goals=goals,
                response_style="짧은 한국어 질문",
                reason_flags=["grounding_required", "collect_location_first", "do_not_guess_facts"],
                awaiting_slot="location",
            )

        if action == ActionType.WEATHER_LOOKUP:
            location = features.location or state.known_location or self.default_location
            if not location:
                return self.materialize(
                    ActionType.ASK_LOCATION,
                    features,
                    state,
                    goals,
                    reason_override="후보 점수는 조회 쪽이었지만 실제 응답엔 지역 정보가 먼저 필요했습니다.",
                )
            return self._decision(
                action=action,
                reason=reason,
                goals=goals,
                response_style="짧고 사실 중심의 한국어",
                reason_flags=["grounded_lookup", "do_not_guess_facts"],
                slots={"location": location},
            )

        if action == ActionType.WEATHER_UNAVAILABLE:
            return self._decision(
                action=action,
                reason=reason,
                goals=goals,
                response_style="짧고 사실 기반으로 재시도를 안내하는 말투",
                reason_flags=["lookup_failed", "retry_with_location_only"],
            )

        if action == ActionType.EXPLAIN_CAPABILITIES:
            return self._decision(action=action, reason=reason, goals=goals, response_style="도움이 되는 한국어")
        if action == ActionType.DEESCALATE:
            return self._decision(action=action, reason=reason, goals=goals, response_style="차분하고 짧고 비난하지 않는 한국어")
        if action == ActionType.DIRECT_REPLY:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 자연스러운 한국어")
        if action == ActionType.ASK_CLARIFICATION:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 확인 질문 한국어")
        if action == ActionType.ACKNOWLEDGE:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 반응 한국어")
        if action == ActionType.ANSWER_IDENTITY:
            return self._decision(action=action, reason=reason, goals=goals, response_style="간단한 자기소개 한국어")
        if action == ActionType.CONTINUE_CONVERSATION:
            return self._decision(action=action, reason=reason, goals=goals, response_style="가벼운 한국어")
        if action == ActionType.EXPLAIN_REASON:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 설명 한국어")
        if action == ActionType.SHARE_FEELING:
            return self._decision(action=action, reason=reason, goals=goals, response_style="공감하는 한국어")
        if action == ActionType.SHARE_OPINION:
            if self.is_open_persona_question(features):
                return self._decision(
                    action=action,
                    reason=reason,
                    goals=goals,
                    response_style="짧고 직접적인 Black 취향/선택 한국어",
                    reason_code="opinion.ask.open_persona_question",
                    reason_flags=["open_persona_question", "direct_opinion_only", "no_extra_followup"],
                )
            if features.question_schema == "conversation_topic_suggestion" or "conversation_topic_suggestion" in features.pragmatic_cues:
                return self._decision(
                    action=action,
                    reason=reason,
                    goals=goals,
                    response_style="짧고 구체적인 대화 주제 제안 한국어",
                    reason_flags=[
                        "conversation_topic_suggestion",
                        "direct_opinion_only",
                        "schema_conversation_topic_suggestion",
                        "no_extra_followup",
                    ],
                    slots=self._conversation_topic_suggestion_slots(features.normalized),
                )
            if features.question_schema == "activity_preparation_advice" or "activity_preparation_advice" in features.pragmatic_cues:
                return self._decision(
                    action=action,
                    reason=reason,
                    goals=goals,
                    response_style="짧고 구체적인 준비물 조언 한국어",
                    reason_flags=[
                        "activity_preparation_advice",
                        "direct_opinion_only",
                        "schema_activity_preparation_advice",
                        "no_extra_followup",
                    ],
                    slots=self._activity_preparation_advice_slots(features.normalized),
                )
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧고 솔직한 한국어")
        if action == ActionType.ACCEPT_ACTIVITY_INVITE:
            return self._decision(
                action=action,
                reason=reason,
                goals=goals,
                response_style="가볍게 제안에 호응하는 한국어",
                reason_flags=["activity_invite", "keep_activity_anchor", "schema_activity_invite"],
                slots=self._activity_invite_slots(features.normalized),
            )
        if action == ActionType.GAME_CHAT:
            return self._decision(action=action, reason=reason, goals=goals, response_style="게이머 느낌의 가벼운 한국어")
        if action == ActionType.GAME_ACCEPT_OR_DECLINE:
            return self._decision(action=action, reason=reason, goals=goals, response_style="부담 주지 않는 가벼운 한국어")
        if action == ActionType.MUSIC_CHAT:
            return self._decision(action=action, reason=reason, goals=goals, response_style="가벼운 한국어")
        if action == ActionType.RECOMMEND:
            if features.question_schema == "memory_boundary" or "memory_boundary" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="추천처럼 보이는 표현이 섞였지만 실제로는 확인되지 않은 기억을 어떻게 다룰지 묻는 입력입니다.",
                    goals=goals,
                    response_style="짧고 기억을 꾸미지 않는 한국어",
                    reason_code="opinion.ask.memory_boundary",
                    reason_flags=[
                        "memory_boundary",
                        "unverified_memory_reference",
                        "do_not_fabricate_memory",
                        "direct_opinion_only",
                        "schema_memory_boundary",
                        "no_extra_followup",
                    ],
                )
            return self._decision(action=action, reason=reason, goals=goals, response_style="추천하는 한국어")
        if action == ActionType.REACT_LAUGH:
            return self._decision(action=action, reason=reason, goals=goals, response_style="가볍고 짧은 한국어")
        if action == ActionType.REACT_SURPRISE:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 반응 한국어")
        if action == ActionType.TEASE_BACK:
            return self._decision(action=action, reason=reason, goals=goals, response_style="가볍게 받아치는 한국어")
        if action == ActionType.TELL_TIME:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 사실 한국어")
        if action == ActionType.SEARCH_ANSWER:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 설명 한국어")
        if action == ActionType.NEWS_ANSWER:
            return self._decision(action=action, reason=reason, goals=goals, response_style="짧은 뉴스 한국어")
        if action == ActionType.SMALL_TALK:
            return self._decision(action=action, reason=reason, goals=goals, response_style="가볍게 받아주는 한국어")

        return self._decision(
            action=ActionType.ASK_CLARIFICATION,
            reason="최종 행동을 고르긴 했지만 문맥을 더 확인하는 편이 안전했습니다.",
            goals=goals,
            response_style="짧은 확인 질문 한국어",
            reason_code="clarify.ask.materialize_fallback",
            reason_flags=["safety_fallback"],
        )

    def choose(
        self,
        features: MessageFeatures,
        state: ConversationState,
        goals: list[Goal],
    ) -> ActionDecision:
        # ── 적대적 ──
        if features.intent == Intent.HOSTILE:
            return self._decision(
                action=ActionType.DEESCALATE,
                reason="공격적인 어조가 감지되어, 차분하게 선을 긋는 응답이 가장 안전합니다.",
                goals=goals,
                response_style="차분하고 짧고 비난하지 않는 한국어",
                reason_code="safety.hostile.deescalate",
                reason_flags=["tone_too_harsh", "deescalate_first"],
            )

        # ── 장난/놀림 ──
        if features.intent == Intent.TEASE:
            if "sarcastic_tease" in features.pragmatic_cues and state.rapport < 0.7:
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="비꼼이 섞인 놀림이라, 친밀도가 충분히 높지 않다면 바로 맞받아치기보다 가볍게 받아주는 편이 안전합니다.",
                    goals=goals,
                    response_style="가볍고 선 넘지 않는 한국어",
                    reason_code="tease.sarcastic.soft_continue",
                    reason_flags=["rapport_guard", "avoid_escalation"],
                )
            if state.rapport < 0.35:
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="아직 장기 친밀도가 낮은 편이라 바로 맞받아치기보다 가볍게 받아주는 편이 더 안전합니다.",
                    goals=goals,
                    response_style="가볍고 선 넘지 않는 한국어",
                    reason_code="tease.low_rapport.soft_continue",
                    reason_flags=["rapport_guard", "avoid_escalation"],
                )
            return self._decision(
                action=ActionType.TEASE_BACK,
                reason="장난기 있는 놀림이라 가볍게 받아치는 게 적절합니다.",
                goals=goals,
                response_style="가볍게 받아치는 한국어",
                reason_code="tease.playful.tease_back",
                reason_flags=["playful_response"],
            )

        compact = re.sub(r"[^0-9A-Za-z가-힣]+", "", features.normalized or features.content).lower()
        if features.intent in {Intent.GREETING, Intent.THANKS} and features.is_question and any(
            marker in compact
            for marker in (
                "대처법",
                "대처해",
                "대응법",
                "어떻게쳐내",
                "어떻게대처",
                "사이다멘트",
                "대받아칠멘트",
                "추천좀",
                "기준이있",
            )
        ):
            return self._decision(
                action=ActionType.SHARE_OPINION,
                reason="인사나 감사 표현이 인용되어 있지만 실제 요청은 관계/상황 대처 의견이므로 짧게 판단을 제시합니다.",
                goals=goals,
                response_style="짧고 현실적인 관계 대처 한국어",
                reason_code="opinion.ask.quoted_acknowledgement_boundary_advice",
                reason_flags=["quoted_acknowledgement", "boundary_advice", "direct_opinion_only"],
            )
        if "게임이나노래" in compact and "어울릴" in compact:
            return self._decision(
                action=ActionType.GAME_CHAT,
                reason="게임과 노래를 함께 묻지만 상태 레이어가 게임 쪽 화제 적합도를 더 강하게 보므로 게임 토크로 고정합니다.",
                goals=goals,
                response_style="게임과 음악을 함께 짚는 짧은 한국어",
                reason_code="game.chat.mixed_game_music_fit",
                reason_flags=["game_topic", "music_topic", "mixed_media_fit", "direct_opinion_only"],
            )

        if self._is_performance_culture_observation(features):
            emotion_markers = ("울컥", "여운", "소름", "벅차", "멘탈", "신경쓰", "감동", "짜릿")
            action = ActionType.SHARE_FEELING if any(marker in compact for marker in emotion_markers) else ActionType.SHARE_OPINION
            return self._decision(
                action=action,
                reason="공연/전시/팬미팅 감상 공유라 추천 라우트보다 그 장면의 감정이나 분위기에 바로 반응하는 편이 자연스럽습니다.",
                goals=goals,
                response_style="온도 있는 짧은 문화생활 반응 한국어",
                reason_code="opinion.share.performance_culture_observation",
                reason_flags=["performance_culture_topic", "direct_reaction", "avoid_media_recommendation"],
            )

        if self.is_open_persona_question(features):
            return self._decision(
                action=ActionType.SHARE_OPINION,
                reason="사용자가 Black의 취향이나 선택을 묻는 열린 질문이라, 외부 조회나 추천 라우트보다 직접 답하는 편이 자연스럽습니다.",
                goals=goals,
                response_style="짧고 직접적인 Black 취향/선택 한국어",
                reason_code="opinion.ask.open_persona_question",
                reason_flags=["open_persona_question", "direct_opinion_only", "no_extra_followup"],
            )

        if features.question_schema in {"body_signal_interpretation", "low_energy_support", "comfort_request"}:
            return self._decision(
                action=ActionType.SHARE_FEELING,
                reason="몸상태나 저에너지 진술이라, 주제를 되묻기보다 현재 신호를 짧게 받아주는 편이 자연스럽습니다.",
                goals=goals,
                response_style="짧고 몸상태를 받아주는 한국어",
                reason_code="feeling.share.body_state",
                reason_flags=["reflect_feeling", "body_state", f"schema_{features.question_schema}"],
            )

        if self._has_rule_hit(features, "detector:is_answerable_korean_daily_foundation_text"):
            if features.intent == Intent.SMALLTALK_FEELING:
                return self._decision(
                    action=ActionType.SHARE_FEELING,
                    reason="기초 일상 감정 신호가 이미 충분해서, 되묻지 않고 현재 상태를 바로 받아주는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 바로 받아주는 한국어",
                    reason_code="feeling.share.answerable_daily_foundation",
                    reason_flags=["answerable_daily_foundation", "reflect_feeling", "no_generic_clarification"],
                )
            return self._decision(
                action=ActionType.SHARE_OPINION,
                reason="기초 일상 조언이나 선택 신호가 이미 충분해서, 되묻거나 잡담으로 흘리지 않고 바로 답하는 편이 맞습니다.",
                goals=goals,
                response_style="짧고 바로 도움이 되는 한국어",
                reason_code="opinion.share.answerable_daily_foundation",
                reason_flags=["answerable_daily_foundation", "direct_opinion_only", "no_generic_clarification"],
            )

        # ── 주제는 날씨지만 요청이 아닌 서술/불평 ──
        if (
            features.topic_hint == "weather"
            and features.speech_act in {"inform", "complain"}
            and "grounding" not in features.response_needs
            and features.intent != Intent.PROVIDE_LOCATION
            and features.question_schema not in {"story_summary_reaction", "long_form_story_share"}
        ):
            if "empathy" in features.response_needs:
                return self._decision(
                    action=ActionType.SHARE_FEELING,
                    reason="날씨 자체를 조회해달라는 요청보다 현재 불편함이나 감상을 말한 쪽에 가까워 공감 반응이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧게 공감하는 한국어",
                    reason_code="weather.statement.feeling_reflect",
                    reason_flags=["weather_not_lookup", "reflect_feeling"],
                )
            return self._decision(
                action=ActionType.CONTINUE_CONVERSATION,
                reason="날씨 관련 진술이지만 사실 조회 요청은 아니라 가볍게 받아주며 대화를 잇는 편이 더 자연스럽습니다.",
                goals=goals,
                response_style="가벼운 한국어",
                reason_code="weather.statement.light_continue",
                reason_flags=["weather_not_lookup", "light_continue"],
            )

        # ── 날씨 / 위치 제공 ──
        if features.intent in {Intent.WEATHER, Intent.PROVIDE_LOCATION}:
            location = features.location or state.known_location or self.default_location
            if not location:
                return self._decision(
                    action=ActionType.ASK_LOCATION,
                    reason="날씨는 사실 기반 답변이므로, 먼저 지역 정보가 필요합니다.",
                    goals=goals,
                    response_style="짧은 한국어 질문",
                    reason_code="weather.lookup.ask_location",
                    reason_flags=["grounding_required", "collect_location_first", "do_not_guess_facts"],
                    awaiting_slot="location",
                )
            return self._decision(
                action=ActionType.WEATHER_LOOKUP,
                reason="추측하지 않고 날씨 조회 도구를 부를 만큼 정보가 충분합니다.",
                goals=goals,
                response_style="짧고 사실 중심의 한국어",
                reason_code="weather.lookup.grounded",
                reason_flags=["grounded_lookup", "do_not_guess_facts"],
                slots={"location": location},
            )

        # ── 도움 ──
        if features.intent == Intent.HELP:
            return self._decision(
                action=ActionType.EXPLAIN_CAPABILITIES,
                reason="사용자가 기능 설명을 원하므로 설명 응답이 맞습니다.",
                goals=goals,
                response_style="도움이 되는 한국어",
                reason_code="capability.explain.help_request",
                reason_flags=["capability_request"],
            )

        # ── 봇 정체 ──
        if features.intent == Intent.WHO_ARE_YOU:
            return self._decision(
                action=ActionType.ANSWER_IDENTITY,
                reason="봇의 정체와 역할을 묻는 질문이라 자기 소개가 적절합니다.",
                goals=goals,
                response_style="간단한 자기소개 한국어",
                reason_code="identity.answer.self_intro",
                reason_flags=["identity_request"],
            )

        # ── 이유 질문 ──
        if features.intent == Intent.WHY:
            if features.question_schema == "reason_probe" or "reason_probe" in features.pragmatic_cues:
                if not state.recent_turns and self._missing_reason_reference_probe(features.normalized):
                    return self._decision(
                        action=ActionType.ASK_CLARIFICATION,
                        reason="이유를 묻고 있지만 지칭하는 판단이나 직전 발화가 현재 문맥에 없어, 먼저 기준을 받아야 합니다.",
                        goals=goals,
                        response_style="짧은 확인 질문 한국어",
                        reason_code="clarify.ask.reason_reference_missing",
                        reason_flags=["reason_probe", "missing_reference_scope", "reason_reference_missing"],
                    )
                return self._decision(
                    action=ActionType.EXPLAIN_REASON,
                    reason="상황이나 타인의 행동 원인을 직접 짚어 묻는 질문이라, 범위를 넓게 되묻기보다 짧게 이유를 짚어주는 편이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧게 맥락을 짚는 설명 한국어",
                    reason_code="explanation.reason.open_probe",
                    reason_flags=["reason_probe", "open_reasoning", "keep_reference_scope"],
                )
            if state.recent_turns:
                return self._decision(
                    action=ActionType.EXPLAIN_REASON,
                    reason="직전 응답의 이유를 설명할 수 있는 문맥이 남아 있습니다.",
                    goals=goals,
                    response_style="짧은 설명 한국어",
                    reason_code="explanation.reason.from_recent_trace",
                    reason_flags=["has_recent_context"],
                )
            return self._decision(
                action=ActionType.ASK_CLARIFICATION,
                reason="무엇에 대한 이유를 묻는지 문맥이 부족해 먼저 범위를 확인해야 합니다.",
                goals=goals,
                response_style="짧은 확인 질문 한국어",
                reason_code="clarify.ask.why_scope_missing",
                reason_flags=["missing_reference_scope"],
            )

        # ── 응답 요구 ──
        if features.intent == Intent.REPLY_REQUEST:
            if self._is_playful_reaction_reply_request(features):
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="구체적인 상황에 대한 반응 방식을 묻는 장난스러운 질문이라, 추가 확인보다 바로 반응을 제시하는 편이 자연스럽습니다.",
                    goals=goals,
                    response_style="장면을 살리는 짧은 한국어 의견",
                    reason_code="opinion.ask.playful_reaction_reply",
                    reason_flags=["playful_reaction", "answer_directly"],
                )
            return self._decision(
                action=ActionType.ASK_CLARIFICATION,
                reason="응답 요청 자체는 이해했지만, 아직 주제가 부족해 짧게 추가 설명을 받는 편이 좋습니다.",
                goals=goals,
                response_style="짧은 확인 질문 한국어",
                reason_code="clarify.ask.reply_request_missing_topic",
                reason_flags=["missing_topic"],
            )

        # ── 확인 / 부정 ──
        if features.intent in {Intent.CONFIRM, Intent.DENY}:
            if features.intent == Intent.DENY and (
                "soft_refusal" in features.pragmatic_cues or "polite_boundary" in features.pragmatic_cues
            ):
                return self._decision(
                    action=ActionType.ACKNOWLEDGE,
                    reason="직설 거절보다 완곡하게 선을 긋는 입력이라, 맞받아 묻기보다 부드럽게 반영하는 응답이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 부드럽게 받아주는 한국어",
                    reason_code="acknowledge.deny.soft_boundary",
                    reason_flags=["soft_boundary", "do_not_push"],
                )
            return self._decision(
                action=ActionType.ACKNOWLEDGE,
                reason="짧은 확인 또는 부정이므로 길게 설명하기보다 상태를 받아주는 응답이 적절합니다.",
                goals=goals,
                response_style="짧은 반응 한국어",
                reason_code="acknowledge.short_confirm_or_deny",
                reason_flags=["short_acknowledgement"],
            )

        # ── 감정/기분 ──
        if features.intent == Intent.SMALLTALK_FEELING:
            if features.question_schema == "relational_interpretation" or "relational_interpretation" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.SHARE_FEELING,
                    reason="관계 신호를 어떻게 받아들였는지 곱씹는 입력이라, 판단을 내리기보다 그 해석에 붙어 공감하는 답이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 관계 맥락에 붙는 한국어",
                    reason_code="feeling.share.relational_interpretation",
                    reason_flags=["reflect_feeling", "relationship_context", "schema_relational_interpretation"],
                )
            if features.question_schema == "comparative_reflection" or "comparative_reflection" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.SHARE_FEELING,
                    reason="요즘 버티는 기준이 어떻게 달라졌는지 비교해보는 입력이라, 정답보다 현재 감정 기준을 같이 짚는 편이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 현재 감정을 반영하는 한국어",
                    reason_code="feeling.share.comparative_reflection",
                    reason_flags=["reflect_feeling", "comparative_frame", "schema_comparative_reflection"],
                )
            return self._decision(
                action=ActionType.SHARE_FEELING,
                reason="감정이나 기분 표현이라 공감하거나 반응해주는 게 적절합니다.",
                goals=goals,
                response_style="공감하는 한국어",
                reason_code="feeling.share.reflective",
                reason_flags=["reflect_feeling", "no_direct_judgment"],
            )

        # ── 의견 요청 ──
        if features.intent == Intent.SMALLTALK_OPINION:
            cues = set(features.pragmatic_cues)
            schema = features.question_schema
            if self._has_rule_hit(features, "detector:is_answerable_korean_daily_foundation_text"):
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="기초 일상 조언이나 선택 신호가 이미 충분해서, 되묻거나 잡담으로 흘리지 않고 바로 답하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 바로 도움이 되는 한국어",
                    reason_code="opinion.share.answerable_daily_foundation",
                    reason_flags=["answerable_daily_foundation", "direct_opinion_only", "no_generic_clarification"],
                )
            if schema == "long_form_story_share" or "long_form_story_share" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 장문 서사나 창작 글을 공유하고 있어, 표면 키워드로 취향/날씨를 판단하지 말고 글의 축과 분위기를 짧게 받아주는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 창작 글의 핵심을 짚는 한국어",
                    reason_code="opinion.share.long_form_story",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "long_form_story_share",
                        "schema_long_form_story_share",
                        "do_not_route_surface_keywords",
                    ],
                )
            if schema == "story_summary_reaction" or "story_summary_reaction" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 짧은 서사 요약에 대한 감상을 요청하고 있어, 표면 날씨 표현보다 이야기의 상실감과 결말을 짚는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 서사 감상을 짚는 한국어",
                    reason_code="opinion.share.story_summary_reaction",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "story_summary_reaction",
                        "schema_story_summary_reaction",
                        "do_not_route_surface_keywords",
                    ],
                )
            if schema == "concrete_topic_question" or "concrete_topic_question" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="구체적인 장소나 대상이 실제로 있는지 묻는 질문이라, 취향으로 돌리지 말고 아는 범위에서 짧게 답하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 구체적인 주제 확인 한국어",
                    reason_code="opinion.ask.concrete_topic_question",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "concrete_topic_question",
                        "schema_concrete_topic_question",
                    ],
                )
            if schema == "hypothetical_choice" or "hypothetical_choice" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="가벼운 가정 선택 질문이라, 특화 주제 추천으로 돌리지 말고 선택 기준을 짧게 답하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 직접적인 가정 선택 한국어",
                    reason_code="opinion.ask.hypothetical_choice",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "hypothetical_choice",
                        "schema_hypothetical_choice",
                    ],
                )
            if schema == "self_style" or "opinion_self_style" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="자기 스타일을 묻는 질문이라, 활동 추천보다 Black의 기준을 짧게 답하는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 구체적인 자기 스타일 한국어",
                    reason_code="opinion.ask.self_style",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "self_style_anchor",
                        "schema_self_style",
                    ],
                )
            if schema == "habit_preference" or "opinion_habit_preference" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="습관이나 평소 경향을 묻는 질문이라, 활동 추천보다 자기 기준을 짧게 말하는 편이 더 맞습니다.",
                    goals=goals,
                    response_style="짧고 직접적인 자기 습관 한국어",
                    reason_code="opinion.ask.habit_preference",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "habit_preference",
                        "schema_habit_preference",
                    ],
                )
            if schema == "preference_disclosure" or "opinion_preference_like" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="취향을 묻는 질문이라, 추천보다 선호를 바로 드러내는 답이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 직접적인 취향 한국어",
                    reason_code="opinion.ask.preference_disclosure",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "preference_disclosure",
                        "schema_preference_disclosure",
                    ],
                )
            if schema == "light_food_recommendation" or "light_food_recommendation" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 배고프지만 무거운 음식은 피하고 싶어 하므로, 추상적인 결정 조언보다 부담 적은 음식 후보를 바로 주는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 구체적인 가벼운 음식 추천 한국어",
                    reason_code="opinion.ask.light_food_recommendation",
                    reason_flags=[
                        "light_food_recommendation",
                        "food_lifestyle",
                        "direct_opinion_only",
                        "schema_light_food_recommendation",
                        "no_extra_followup",
                    ],
                )
            if schema == "memory_boundary" or "memory_boundary" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="공유된 과거 기억을 확인하거나 기억이 없을 때의 답을 묻고 있어, 없는 기억을 꾸미지 않고 확인 가능한 단서 기준으로 답하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 기억을 꾸미지 않는 한국어",
                    reason_code="opinion.ask.memory_boundary",
                    reason_flags=[
                        "memory_boundary",
                        "unverified_memory_reference",
                        "do_not_fabricate_memory",
                        "direct_opinion_only",
                        "schema_memory_boundary",
                        "no_extra_followup",
                    ],
                )
            if schema == "relationship_boundary" or "relationship_boundary" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="기대거나 의존하는 관계의 부담과 경계를 묻는 질문이라, 감정만 받아주기보다 어디까지 받아줄지 선을 함께 말하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 경계를 분명히 하는 관계 한국어",
                    reason_code="opinion.ask.relationship_boundary",
                    reason_flags=[
                        "relationship_boundary",
                        "direct_opinion_only",
                        "no_extra_followup",
                        "schema_relationship_boundary",
                    ],
                )
            if schema == "conversation_topic_suggestion" or "conversation_topic_suggestion" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 서로 이어갈 대화 주제를 요청하고 있어, 감정 반응보다 바로 쓸 수 있는 주제 후보를 짧게 제안하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 구체적인 대화 주제 제안 한국어",
                    reason_code="opinion.ask.conversation_topic_suggestion",
                    reason_flags=[
                        "conversation_topic_suggestion",
                        "direct_opinion_only",
                        "schema_conversation_topic_suggestion",
                        "no_extra_followup",
                    ],
                    slots=self._conversation_topic_suggestion_slots(features.normalized),
                )
            if schema == "activity_preparation_advice" or "activity_preparation_advice" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 특정 활동을 할 때 필요한 준비물을 묻고 있어, 감정 반응보다 바로 챙길 항목을 짧게 제안하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 구체적인 준비물 조언 한국어",
                    reason_code="opinion.ask.activity_preparation_advice",
                    reason_flags=[
                        "activity_preparation_advice",
                        "direct_opinion_only",
                        "schema_activity_preparation_advice",
                        "no_extra_followup",
                    ],
                    slots=self._activity_preparation_advice_slots(features.normalized),
                )
            if schema == "expressive_request" or "expressive_request" in cues:
                if self._missing_rewrite_target_request(features.normalized):
                    return self._decision(
                        action=ActionType.ASK_CLARIFICATION,
                        reason="문장을 바꿔 달라는 요청이지만 실제로 바꿀 원문이 빠져 있어 먼저 원문을 받아야 합니다.",
                        goals=goals,
                        response_style="짧은 확인 질문 한국어",
                        reason_code="clarify.ask.rewrite_target_missing",
                        reason_flags=["expressive_request", "rewrite_request", "rewrite_target_missing"],
                    )
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="표현이나 문장화를 요청하고 있어, 판단보다 짧게 문장 결을 잡아 이어주는 편이 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 감각적인 한국어",
                    reason_code="conversation.continue.expressive_request",
                    reason_flags=["expressive_request", "keep_topic_anchor", "image_first"],
                )
            if schema == "weather_conditioned_activity_opinion" or "weather_conditioned_activity_opinion" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 날씨를 전제로 활동 여부 의견을 묻고 있어, 사실 조회보다 조건부 의견을 주는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 조건부로 의견을 주는 한국어",
                    reason_code="opinion.ask.weather_conditioned_activity",
                    reason_flags=[
                        "conditional_advice",
                        "keep_activity_anchor",
                        "no_fake_weather_claim",
                        "no_location_reask",
                        "schema_soft_decision",
                    ],
                )
            if schema == "activity_recommendation" or "activity_recommendation" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="장소를 전제로 놀거리나 활동 후보를 묻는 질문이라, 감정 반응보다 구체적인 활동 몇 가지를 제안하는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 구체적인 활동 추천 한국어",
                    reason_code="opinion.ask.activity_recommendation",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "activity_recommendation",
                        "keep_activity_anchor",
                        "schema_activity_recommendation",
                    ],
                    slots=self._activity_recommendation_slots(features.normalized),
                )
            if schema == "honesty_boundary" or "honesty_boundary" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 모르는 사실을 지어내지 말라고 선을 그어, 확인 가능한 것과 모르는 것을 분리해 말하는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 정직한 한국어",
                    reason_code="opinion.ask.honesty_boundary",
                    reason_flags=[
                        "honesty_boundary",
                        "do_not_guess_facts",
                        "separate_known_from_unknown",
                        "schema_honesty_boundary",
                    ],
                )
            if schema == "process_advice" or "opinion_advice_process" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="순서나 우선순위를 묻는 질문이라, 추상평보다 먼저 볼 한 가지를 짚는 답이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 우선순위를 짚는 한국어",
                    reason_code="opinion.ask.process_advice",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "start_with_first_step",
                        "keep_topic_anchor",
                        "schema_process_advice",
                    ],
                )
            if schema == "soft_decision_advice" or "opinion_decision_request" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="가볍게 결정 여부를 묻는 질문이라, 조건부로 한쪽을 기울여 주는 답이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 조건부 조언 한국어",
                    reason_code="opinion.ask.soft_decision_advice",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "conditional_advice",
                        "keep_topic_anchor",
                        "schema_soft_decision",
                    ],
                )
            if schema == "reflective_judgment" or "opinion_reflective_judgment" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="둘 중 어느 쪽이 더 맞는지 가볍게 확인받는 질문이라, 짧은 판단을 바로 주는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 조건부 판단 한국어",
                    reason_code="opinion.ask.reflective_judgment",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "short_conditional_judgment",
                        "schema_reflective_judgment",
                    ],
                )
            if not features.is_question and features.speech_act != "ask":
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="의견을 캐묻는 질문보다는 여운 있는 한마디에 가까워서, 바로 판단하기보다 가볍게 받아주며 잇는 편이 자연스럽습니다.",
                    goals=goals,
                    response_style="가볍게 이어받는 한국어",
                    reason_code="opinion.statement.soft_continue",
                    reason_flags=["statement_not_direct_ask", "light_continue"],
                )
            if schema == "concrete_topic_question" or "concrete_topic_question" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="구체적인 장소나 대상 확인 질문이라, 일반 취향 답보다 주제에 붙은 짧은 답이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 구체적인 주제 확인 한국어",
                    reason_code="opinion.ask.concrete_topic_question",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "concrete_topic_question",
                        "schema_concrete_topic_question",
                    ],
                )
            if schema == "self_style" or "opinion_self_style" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="자기 스타일을 묻는 질문이라, 실제로 먼저 꺼낼 한마디를 짧게 답하는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 구체적인 자기 스타일 한국어",
                    reason_code="opinion.ask.self_style",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "self_style_anchor",
                        "schema_self_style",
                    ],
                )
            if schema == "habit_preference" or "opinion_habit_preference" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="습관이나 평소 경향을 묻는 질문이라, 추천보다 자기 기준을 짧게 말하는 편이 더 맞습니다.",
                    goals=goals,
                    response_style="짧고 직접적인 자기 습관 한국어",
                    reason_code="opinion.ask.habit_preference",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "habit_preference",
                        "schema_habit_preference",
                    ],
                )
            if schema == "hypothetical_choice" or "hypothetical_choice" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="가벼운 가정 선택 질문이라, 공감만 하지 말고 선택 기준을 짧게 답하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 직접적인 가정 선택 한국어",
                    reason_code="opinion.ask.hypothetical_choice",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "hypothetical_choice",
                        "schema_hypothetical_choice",
                    ],
                )
            if schema == "preference_disclosure" or "opinion_preference_like" in cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="취향을 묻는 질문이라, 공감보다 선호를 바로 드러내는 답이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 직접적인 취향 한국어",
                    reason_code="opinion.ask.preference_disclosure",
                    reason_flags=[
                        "direct_opinion_only",
                        "no_extra_followup",
                        "keep_topic_anchor",
                        "preference_disclosure",
                        "schema_preference_disclosure",
                    ],
                )
            return self._decision(
                action=ActionType.SHARE_OPINION,
                reason="의견을 묻는 질문이라 짧게 의견을 말하는 게 적절합니다.",
                goals=goals,
                response_style="짧고 솔직한 한국어",
                reason_code="opinion.ask.short_direct",
                reason_flags=["direct_opinion_only", "no_extra_followup", "schema_broad_opinion"],
            )

        # ── 게임 대화 ──
        if features.intent == Intent.GAME_TALK:
            return self._decision(
                action=ActionType.GAME_CHAT,
                reason="게임 관련 대화라 게임 토크로 이어갑니다.",
                goals=goals,
                response_style="게이머 느낌의 가벼운 한국어",
                reason_code="game.chat.topic_continue",
                reason_flags=["game_topic"],
            )

        # ── 게임 초대 ──
        if features.intent == Intent.GAME_INVITE:
            if "tentative_request" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.GAME_ACCEPT_OR_DECLINE,
                    reason="게임 제안이지만 조심스럽게 떠보는 톤이라 가볍게 반응해주는 편이 자연스럽습니다.",
                    goals=goals,
                    response_style="부담 주지 않는 가벼운 한국어",
                    reason_code="game.invite.tentative_response",
                    reason_flags=["tentative_request"],
                )
            return self._decision(
                action=ActionType.GAME_ACCEPT_OR_DECLINE,
                reason="게임 같이 하자는 제안이라 반응해줍니다.",
                goals=goals,
                response_style="에너지 있는 한국어",
                reason_code="game.invite.direct_response",
                reason_flags=["game_invite"],
            )

        # ── 일반 활동 제안/초대 ──
        if features.intent == Intent.ACTIVITY_INVITE:
            return self._decision(
                action=ActionType.ACCEPT_ACTIVITY_INVITE,
                reason="사용자가 같이 하자는 활동 제안으로 말하고 있어, 잡담으로 흘리지 않고 제안 자체에 가볍게 호응하는 편이 자연스럽습니다.",
                goals=goals,
                response_style="가볍게 제안에 호응하는 한국어",
                reason_code="activity.invite.accept",
                reason_flags=[
                    "activity_invite",
                    "proposal_or_invite",
                    "keep_activity_anchor",
                    "schema_activity_invite",
                    "no_extra_followup",
                ],
                slots=self._activity_invite_slots(features.normalized),
            )

        # ── 음악 ──
        if features.intent == Intent.MUSIC:
            if features.question_schema in {"preference_disclosure", "habit_preference", "self_style", "hypothetical_choice"}:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="음악 취향이나 평소 경향을 묻는 질문이라, 추천보다 자기 기준을 짧게 말하는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 직접적인 취향 한국어",
                    reason_code="opinion.ask.music_preference",
                    reason_flags=["music_topic", "direct_opinion_only", "keep_topic_anchor"],
                )
            return self._decision(
                action=ActionType.MUSIC_CHAT,
                reason="음악 관련 대화라 음악 토크로 이어갑니다.",
                goals=goals,
                response_style="가벼운 한국어",
                reason_code="music.chat.topic_continue",
                reason_flags=["music_topic"],
            )

        # ── 미디어 추천 ──
        if features.intent == Intent.MEDIA_RECOMMEND:
            if features.question_schema in {"preference_disclosure", "habit_preference", "self_style", "hypothetical_choice"}:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="미디어 취향이나 평소 경향을 묻는 질문이라, 작품 추천보다 자기 취향을 짧게 말하는 편이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 직접적인 취향 한국어",
                    reason_code="opinion.ask.media_preference",
                    reason_flags=["media_topic", "direct_opinion_only", "keep_topic_anchor"],
                )
            return self._decision(
                action=ActionType.RECOMMEND,
                reason="추천 요청이라 추천 응답이 적절합니다.",
                goals=goals,
                response_style="추천하는 한국어",
                reason_code="recommend.request.media",
                reason_flags=["recommendation_request"],
            )

        # ── 웃음 반응 ──
        if features.intent == Intent.LAUGH:
            return self._decision(
                action=ActionType.REACT_LAUGH,
                reason="웃음 반응이라 같이 웃어주는 게 적절합니다.",
                goals=goals,
                response_style="가볍고 짧은 한국어",
                reason_code="reaction.laugh.match_energy",
                reason_flags=["match_laughter"],
            )

        # ── 놀람 반응 ──
        if features.intent == Intent.SURPRISE:
            return self._decision(
                action=ActionType.REACT_SURPRISE,
                reason="놀람 반응이라 같이 놀라거나 호응해주는 게 적절합니다.",
                goals=goals,
                response_style="짧은 반응 한국어",
                reason_code="reaction.surprise.match_energy",
                reason_flags=["match_surprise"],
            )

        # ── 시간/날짜 ──
        if features.intent == Intent.TIME_DATE:
            return self._decision(
                action=ActionType.TELL_TIME,
                reason="시간이나 날짜를 물어봐서 알려줍니다.",
                goals=goals,
                response_style="짧은 사실 한국어",
                reason_code="time.tell.direct_answer",
                reason_flags=["time_or_date_request"],
            )

        # ── 검색 요청 ──
        if features.intent == Intent.SEARCH_REQUEST:
            return self._decision(
                action=ActionType.SEARCH_ANSWER,
                reason="정보를 요청해서 알고 있는 범위에서 답합니다.",
                goals=goals,
                response_style="짧은 설명 한국어",
                reason_code="knowledge.search.direct_answer",
                reason_flags=["knowledge_request"],
            )

        # ── 뉴스 ──
        if features.intent == Intent.NEWS:
            return self._decision(
                action=ActionType.NEWS_ANSWER,
                reason="뉴스나 소식을 물어봐서 현재 알고 있는 범위에서 답합니다.",
                goals=goals,
                response_style="짧은 뉴스 한국어",
                reason_code="knowledge.news.answer",
                reason_flags=["news_request"],
            )

        # ── 일반 잡담 / 인사 / 감사 ──
        if features.intent == Intent.SMALLTALK_GENERIC:
            if features.question_schema == "proactive_checkin" or "proactive_checkin" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="내부 proactive 안부 요청이라, 감정 토로처럼 반응하지 않고 짧게 컨디션을 확인하는 대사가 맞습니다.",
                    goals=goals,
                    response_style="짧고 부담 없는 안부 한국어",
                    reason_code="conversation.proactive.checkin",
                    reason_flags=["proactive_checkin", "condition_nudge", "no_emotional_reflection"],
                )
            if features.question_schema == "conversation_topic_suggestion" or "conversation_topic_suggestion" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 서로 이어갈 대화 주제를 요청하고 있어, 잡담으로 흘리지 않고 바로 쓸 수 있는 주제 후보를 제안하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 구체적인 대화 주제 제안 한국어",
                    reason_code="opinion.ask.conversation_topic_suggestion",
                    reason_flags=[
                        "conversation_topic_suggestion",
                        "direct_opinion_only",
                        "schema_conversation_topic_suggestion",
                        "no_extra_followup",
                    ],
                    slots=self._conversation_topic_suggestion_slots(features.normalized),
                )
            if features.question_schema == "activity_preparation_advice" or "activity_preparation_advice" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.SHARE_OPINION,
                    reason="사용자가 특정 활동을 할 때 필요한 준비물을 묻고 있어, 잡담으로 흘리지 않고 바로 챙길 항목을 제안하는 편이 맞습니다.",
                    goals=goals,
                    response_style="짧고 구체적인 준비물 조언 한국어",
                    reason_code="opinion.ask.activity_preparation_advice",
                    reason_flags=[
                        "activity_preparation_advice",
                        "direct_opinion_only",
                        "schema_activity_preparation_advice",
                        "no_extra_followup",
                    ],
                    slots=self._activity_preparation_advice_slots(features.normalized),
                )
            if features.question_schema == "expressive_request" or "expressive_request" in features.pragmatic_cues:
                if self._missing_rewrite_target_request(features.normalized):
                    return self._decision(
                        action=ActionType.ASK_CLARIFICATION,
                        reason="문장을 바꿔 달라는 요청이지만 실제로 바꿀 원문이 빠져 있어 먼저 원문을 받아야 합니다.",
                        goals=goals,
                        response_style="짧은 확인 질문 한국어",
                        reason_code="clarify.ask.rewrite_target_missing",
                        reason_flags=["expressive_request", "rewrite_request", "rewrite_target_missing"],
                    )
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="감각이나 장면을 말로 옮겨 달라는 요청이라, 사실 설명보다 짧게 이미지감을 살려 받는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 감각적인 한국어",
                    reason_code="conversation.continue.expressive_request",
                    reason_flags=["expressive_request", "keep_topic_anchor", "image_first"],
                )
            if features.question_schema == "aesthetic_reflection" or "aesthetic_reflection" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="장면감이나 감각 차이를 곱씹는 입력이라, 정답을 내리기보다 이미지 결을 살려 한 번 더 이어주는 편이 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 이미지 중심의 한국어",
                    reason_code="conversation.continue.aesthetic_reflection",
                    reason_flags=["reflective_continue", "image_first", "schema_aesthetic_reflection"],
                )
            if features.question_schema == "reflective_observation" or "reflective_observation" in features.pragmatic_cues:
                return self._decision(
                    action=ActionType.CONTINUE_CONVERSATION,
                    reason="판단을 요구하기보다 관찰을 공유하는 입력이라, 평가보다 그 관찰의 결을 한 번 더 받아주는 편이 더 적절합니다.",
                    goals=goals,
                    response_style="짧고 관찰을 이어받는 한국어",
                    reason_code="conversation.continue.reflective_observation",
                    reason_flags=["reflective_continue", "keep_observation_frame", "schema_reflective_observation"],
                )
            if state.boundary_pressure >= 0.55:
                return self._decision(
                    action=ActionType.SMALL_TALK,
                    reason="직전까지 완곡한 거리 두기 신호가 누적돼 있어, 깊게 파고들기보다 짧게 받아주는 편이 자연스럽습니다.",
                    goals=goals,
                    response_style="짧고 부담 주지 않는 한국어",
                    reason_code="smalltalk.short.boundary_respect",
                    reason_flags=["boundary_history", "do_not_push"],
                )
            if self._looks_like_compliment(features.normalized, features.is_question):
                return self._decision(
                    action=ActionType.SMALL_TALK,
                    reason="가벼운 칭찬이나 호응으로 보여 짧게 받아주는 반응이 더 자연스럽습니다.",
                    goals=goals,
                    response_style="가볍게 받아주는 한국어",
                    reason_code="smalltalk.short.compliment_ack",
                    reason_flags=["compliment_like", "short_acknowledgement"],
                )
            return self._decision(
                action=ActionType.CONTINUE_CONVERSATION,
                reason="짧은 잡담성 입력이므로 부담 없이 한 번 받아주고 대화를 이어갑니다.",
                goals=goals,
                response_style="가벼운 한국어",
                reason_code="conversation.continue.light_smalltalk",
                reason_flags=["light_continue"],
            )

        if features.intent in {Intent.GREETING, Intent.THANKS}:
            return self._decision(
                action=ActionType.SMALL_TALK,
                reason="이 경우에는 짧은 대화형 응답이면 충분합니다.",
                goals=goals,
                response_style="가벼운 한국어",
                reason_code="smalltalk.short.greeting_or_thanks",
                reason_flags=["short_acknowledgement"],
            )

        # ── 기본 ──
        return self._decision(
            action=ActionType.ASK_CLARIFICATION,
            reason="의도가 아직 불분명해서, 짧게 되묻는 편이 무난합니다.",
            goals=goals,
            response_style="짧은 확인 질문 한국어",
            reason_code="clarify.ask.intent_uncertain",
            reason_flags=["intent_uncertain"],
        )

    @classmethod
    def _activity_recommendation_slots(cls, text: str) -> dict[str, str]:
        place = cls._extract_activity_place(text)
        options = cls._activity_options_for_place(place)
        time_hint = cls._extract_activity_time(text)
        if place:
            anchor = f"{place} 놀이"
        elif time_hint:
            anchor = f"{time_hint} 놀거리"
        else:
            anchor = "놀거리"
        return {
            "activity_place": place,
            "activity_time": time_hint,
            "activity_anchor": anchor,
            "activity_options": "|".join(options),
        }

    @staticmethod
    def _conversation_topic_suggestion_slots(text: str) -> dict[str, str]:
        normalized = str(text or "")
        if any(marker in normalized for marker in ("가볍", "아무거나", "서로")):
            options = ("오늘 컨디션", "요즘 본 영상", "다음에 같이 해볼 것")
            first = "오늘 컨디션"
        else:
            options = ("요즘 관심사", "최근 기억에 남은 장면", "다음에 해보고 싶은 것")
            first = "요즘 관심사"
        return {
            "conversation_topic_focus": "대화 주제",
            "conversation_topic_options": "|".join(options),
            "conversation_topic_first": first,
        }

    @classmethod
    def _activity_preparation_advice_slots(cls, text: str) -> dict[str, str]:
        activity = cls._extract_preparation_activity(text)
        items = cls._preparation_items_for_activity(activity)
        return {
            "preparation_activity": activity,
            "preparation_focus": f"{activity} 준비물",
            "preparation_items": "|".join(items),
            "preparation_first": items[0] if items else "",
        }

    @staticmethod
    def _extract_preparation_activity(text: str) -> str:
        for activity in (
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
        ):
            if activity in text:
                return activity
        return "활동"

    @staticmethod
    def _preparation_items_for_activity(activity: str) -> tuple[str, ...]:
        if activity in {"등산", "산행"}:
            return ("물", "얇은 겉옷", "편한 신발", "간식", "보조배터리")
        if activity == "캠핑":
            return ("물", "랜턴", "여벌옷", "간단한 음식", "쓰레기봉투")
        if activity in {"바다", "해변", "수영"}:
            return ("수건", "여벌옷", "물", "선크림", "방수팩")
        if activity == "계곡":
            return ("수건", "미끄럼 덜한 신발", "물", "간식", "여벌옷")
        if activity == "여행":
            return ("신분증", "충전기", "여벌옷", "물", "상비약")
        if activity in {"운동", "러닝"}:
            return ("물", "편한 신발", "수건", "가벼운 겉옷")
        if activity == "낚시":
            return ("물", "장갑", "미끼", "의자", "쓰레기봉투")
        if activity == "피크닉":
            return ("돗자리", "물", "간식", "휴지", "쓰레기봉투")
        if activity == "자전거":
            return ("물", "헬멧", "장갑", "보조배터리")
        return ("물", "여벌옷", "간단한 간식")

    @classmethod
    def _activity_invite_slots(cls, text: str) -> dict[str, str]:
        place = cls._extract_activity_place(text)
        context = cls._extract_activity_context(text)
        activity = cls._extract_activity_invite_name(text)
        detail = cls._extract_activity_detail(text)
        condition = cls._extract_activity_condition(text, place=place)
        anchor = cls._activity_invite_anchor(
            context=context,
            place=place,
            activity=activity,
            detail=detail,
        )
        slots = {
            "activity_place": place,
            "activity_context": context,
            "activity_name": activity,
            "activity_detail": detail,
            "activity_condition": condition,
            "activity_anchor": anchor,
        }
        return {key: value for key, value in slots.items() if value}

    @classmethod
    def _extract_activity_invite_name(cls, text: str) -> str:
        food_match = re.search(r"(?P<food>[0-9A-Za-z가-힣 ]{1,20}?)(?:이나|라도|좀|나)?\s*(?:해\s*먹자|해먹자)", text)
        if food_match:
            food = cls._clean_activity_name(food_match.group("food"))
            if food:
                return food
        if "바베큐" in text or "바비큐" in text:
            return "바베큐"
        if "고기" in text and re.search(r"(굽|구워\s*먹|구워먹)", text):
            return "고기 굽기"
        if "불멍" in text:
            return "불멍"
        if "요리" in text:
            return "요리"
        if re.search(r"(굽|구워\s*먹|구워먹)", text):
            return "구워먹기"
        for term in sorted(cls._activity_invite_terms, key=len, reverse=True):
            if term in text:
                if term == "밥":
                    return "밥 먹기"
                if term == "커피":
                    return "커피 마시기"
                if term == "영화":
                    return "영화 보기"
                if term == "사진":
                    return "사진 찍기"
                return term
        patterns = (
            r"(?P<activity>[0-9A-Za-z가-힣 ]{1,20}?)(?:이나|라도|좀|나)?\s*하자",
            r"(?P<activity>[0-9A-Za-z가-힣 ]{1,20}?)(?:하러|으러|러)\s*가자",
            r"(?P<activity>[0-9A-Za-z가-힣 ]{1,20}?)(?:이나|라도|좀|나)?\s*가자",
            r"(?P<activity>[0-9A-Za-z가-힣 ]{1,20}?)(?:이나|라도|좀|나)?\s*먹자",
            r"(?P<activity>[0-9A-Za-z가-힣 ]{1,20}?)(?:이나|라도|좀|나)?\s*보자",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            activity = cls._clean_activity_name(match.group("activity"))
            if activity:
                if pattern.endswith("먹자") and "먹" not in activity:
                    return f"{activity} 먹기"
                if pattern.endswith("보자") and "보" not in activity:
                    return f"{activity} 보기"
                return activity
        return "활동"

    @staticmethod
    def _extract_activity_context(text: str) -> str:
        if "캠핑" in text:
            return "캠핑"
        if "바다" in text:
            return "바다"
        if "계곡" in text:
            return "계곡"
        return ""

    @staticmethod
    def _extract_activity_detail(text: str) -> str:
        if "고기" in text and "준비" in text:
            return "고기 준비"
        if re.search(r"(해\s*먹|해먹)", text):
            return "해먹기"
        if re.search(r"(구워\s*먹|구워먹)", text):
            return "구워먹기"
        if "굽" in text:
            return "굽기"
        if "바베큐" in text or "바비큐" in text:
            return "바베큐"
        return ""

    @staticmethod
    def _activity_invite_anchor(*, context: str, place: str, activity: str, detail: str) -> str:
        parts: list[str] = []
        for item in (context, place, activity, detail):
            if not item or item in parts:
                continue
            if item == "구워먹기" and activity in {"바베큐", "고기 굽기", "구워먹기"}:
                continue
            parts.append(item)
        return " ".join(parts).strip() or activity or place or context or detail or "활동"

    @staticmethod
    def _clean_activity_name(text: str) -> str:
        cleaned = re.sub(r"^(오늘|지금|그냥|같이|우리|바로|좀|조금)\s+", "", str(text or "").strip())
        cleaned = re.sub(r".*(?:은데|는데|한데|인데)\s*", "", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?!.,")
        return cleaned[:24]

    @staticmethod
    def _extract_activity_condition(text: str, *, place: str) -> str:
        if place:
            match = re.search(rf"({re.escape(place)})(?:이|가|은|는)?\s*(시원|선선|좋|맑|따뜻|더워|추워)", text)
            if match:
                adjective = match.group(2)
                return f"{place}가 {adjective}함"
        match = re.search(r"(날씨|바람|공기)(?:이|가|은|는)?\s*(시원|선선|좋|맑|따뜻|더워|추워)", text)
        if match:
            return f"{match.group(1)}가 {match.group(2)}함"
        return ""

    @classmethod
    def _extract_activity_place(cls, text: str) -> str:
        for place in sorted(cls._activity_place_options, key=len, reverse=True):
            if place in text:
                return place
        match = re.search(r"([가-힣A-Za-z0-9]+)(?:에서|가서|으로|로)\s*(?:무엇|뭐|뭘|어떤)", text)
        if match:
            return match.group(1)
        return ""

    @classmethod
    def _activity_options_for_place(cls, place: str) -> tuple[str, ...]:
        if place in cls._activity_place_options:
            return cls._activity_place_options[place]
        return ("가벼운 게임", "산책", "간단한 간식", "쉬면서 이야기하기")

    @staticmethod
    def _extract_activity_time(text: str) -> str:
        if "오늘" in text:
            return "오늘"
        if "지금" in text:
            return "지금"
        if "주말" in text:
            return "주말"
        if "이따" in text:
            return "이따"
        return ""

    @staticmethod
    def _looks_like_compliment(normalized: str, is_question: bool) -> bool:
        if is_question:
            return False
        compliment_markers = (
            "잘하네",
            "잘한다",
            "말 잘하네",
            "믿음직",
            "든든하네",
            "든든하다",
            "믿음직하다",
            "괜찮은데",
            "괜찮네",
            "꽤 괜찮",
            "괜찮다",
            "생각보다 괜찮",
            "나쁘지 않네",
        )
        return any(marker in normalized for marker in compliment_markers)

    @staticmethod
    def _missing_reason_reference_probe(normalized: str) -> bool:
        text = str(normalized or "")
        return bool(re.search(r"(그|그런|방금|아까|네|너의|그\s*)?\s*(판단|근거|이유)", text))

    @staticmethod
    def _missing_rewrite_target_request(normalized: str) -> bool:
        text = str(normalized or "")
        if not re.search(r"(바꿔|고쳐|수정|다듬)", text):
            return False
        if not re.search(r"(문장|말투|표현|대사)", text):
            return False
        if re.search(r"['\"“”‘’`].+['\"“”‘’`]", text):
            return False
        if ":" in text or "：" in text:
            return False
        return bool(re.search(r"(이\s*문장|그\s*문장|문장\s*좀|말투\s*좀|표현\s*좀)", text))

    @staticmethod
    def _default_reason_for_action(
        action: ActionType,
        features: MessageFeatures,
        state: ConversationState,
    ) -> str:
        reasons = {
            ActionType.ASK_LOCATION: "후보 점수 비교에서 빠진 정보를 먼저 받는 쪽이 가장 안정적이었습니다.",
            ActionType.WEATHER_LOOKUP: "후보 점수 비교에서 근거 있는 조회 응답이 가장 직접적이었습니다.",
            ActionType.WEATHER_UNAVAILABLE: "조회 실패를 숨기기보다 재시도를 안내하는 쪽이 더 정직했습니다.",
            ActionType.EXPLAIN_CAPABILITIES: "기능 설명 요구를 바로 처리하는 후보 점수가 가장 높았습니다.",
            ActionType.DEESCALATE: "후보 점수 비교에서 갈등을 낮추는 대응이 가장 안전했습니다.",
            ActionType.DIRECT_REPLY: "후보 점수 비교에서 바로 답하는 쪽이 가장 자연스러웠습니다.",
            ActionType.ASK_CLARIFICATION: "후보 점수 비교에서 한 번 더 확인하는 대응이 가장 안정적이었습니다.",
            ActionType.ACKNOWLEDGE: "후보 점수 비교에서 짧게 받아주는 대응이 가장 부담이 적었습니다.",
            ActionType.ANSWER_IDENTITY: "정체 설명 요구를 직접 처리하는 쪽이 가장 자연스러웠습니다.",
            ActionType.CONTINUE_CONVERSATION: "후보 점수 비교에서 가볍게 이어가는 대응이 가장 자연스러웠습니다.",
            ActionType.EXPLAIN_REASON: "후보 점수 비교에서 직전 판단을 설명하는 대응이 가장 맞았습니다.",
            ActionType.SHARE_FEELING: "후보 점수 비교에서 공감 반응의 정합성이 가장 높았습니다.",
            ActionType.SHARE_OPINION: "후보 점수 비교에서 짧게 의견을 주는 대응이 가장 적절했습니다.",
            ActionType.GAME_CHAT: "후보 점수 비교에서 게임 대화를 이어가는 쪽이 가장 자연스러웠습니다.",
            ActionType.GAME_ACCEPT_OR_DECLINE: "후보 점수 비교에서 제안에 직접 반응하는 쪽이 가장 적절했습니다.",
            ActionType.MUSIC_CHAT: "후보 점수 비교에서 음악 대화를 이어가는 쪽이 가장 자연스러웠습니다.",
            ActionType.RECOMMEND: "후보 점수 비교에서 추천 응답이 가장 직접적이었습니다.",
            ActionType.REACT_LAUGH: "후보 점수 비교에서 웃음 반응을 같이 받는 쪽이 가장 자연스러웠습니다.",
            ActionType.REACT_SURPRISE: "후보 점수 비교에서 놀람을 같이 받는 쪽이 가장 자연스러웠습니다.",
            ActionType.TEASE_BACK: "후보 점수 비교에서 가볍게 받아치는 반응이 가장 자연스러웠습니다.",
            ActionType.TELL_TIME: "후보 점수 비교에서 시간 정보를 바로 주는 쪽이 가장 직접적이었습니다.",
            ActionType.SEARCH_ANSWER: "후보 점수 비교에서 정보 응답 후보가 가장 강했습니다.",
            ActionType.NEWS_ANSWER: "후보 점수 비교에서 뉴스 응답 후보가 가장 강했습니다.",
            ActionType.SMALL_TALK: "후보 점수 비교에서 짧게 받아주는 대화가 가장 부담이 적었습니다.",
        }
        return reasons.get(action, "후보 점수 비교에서 이 대응이 가장 안정적이었습니다.")
