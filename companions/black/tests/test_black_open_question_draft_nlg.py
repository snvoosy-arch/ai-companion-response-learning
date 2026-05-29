from __future__ import annotations

import unittest

from predictive_bot.core.draft_nlg import (
    _render_body_state_direct_reply,
    _render_contextual_observation_direct_reply,
    _render_open_question_persona_direct_reply,
)


class BlackOpenQuestionDraftNlgTests(unittest.TestCase):
    def test_hypothetical_questions_get_concrete_persona_drafts(self) -> None:
        cases = [
            (
                "갑자기 세상의 모든 전기가 하루 동안 끊긴다면 무엇을 하며 시간을 보낼 건가요?",
                "전기가 하루",
            ),
            (
                "나만의 행성을 발견해서 이름을 짓는다면 어떤 이름으로 짓고 싶나요?",
                "느린별",
            ),
            (
                "좀비 사태가 터졌을 때 생존을 위해 가장 먼저 챙길 무기나 도구는 무엇인가요?",
                "도구",
            ),
            (
                "만약 당신이 새로운 언어를 창조한다면 그 언어의 이름은 무엇일까요?",
                "느린말",
            ),
        ]

        for text, expected_fragment in cases:
            with self.subTest(text=text):
                reply = _render_open_question_persona_direct_reply(text)
                self.assertIn(expected_fragment, reply)
                self.assertNotIn("이해돼. 다만 무리하게 밀 필요는 없어", reply)
                self.assertNotIn("나는 꽤 맞는 편이야", reply)

    def test_general_persona_questions_get_direct_answers(self) -> None:
        cases = [
            ("가장 좋아하는 계절과 그 이유는 무엇인가요?", "가을"),
            ("당신을 가장 잘 표현하는 단어 3가지는 무엇이라고 생각하나요?", "차분함"),
            ("커피와 차 중 어떤 것을 더 선호하시나요?", "커피"),
        ]

        for text, expected_fragment in cases:
            with self.subTest(text=text):
                reply = _render_open_question_persona_direct_reply(text)
                self.assertIn(expected_fragment, reply)
                self.assertGreaterEqual(len(reply), 20)

    def test_memory_boundary_questions_do_not_fabricate_private_history(self) -> None:
        cases = [
            (
                "살면서 가장 처음으로 내 돈을 주고 샀던 물건은 무엇이었는지 기억나나요?",
                "기억을 실제처럼 만들진",
            ),
            (
                "내가 만든 물건 중 가장 자랑스러운 것은 무엇인가요? (요리, 예술 작품, 글 등 무엇이든)",
                "기억처럼 만들진",
            ),
            (
                "길을 가다 무심코 들려온 노래 가사 한 줄이 내 마음을 울린 적이 있나요?",
                "기억처럼 만들진",
            ),
        ]

        for text, expected_fragment in cases:
            with self.subTest(text=text):
                reply = _render_open_question_persona_direct_reply(text)
                self.assertIn(expected_fragment, reply)
                self.assertNotIn("사실 확인 전엔", reply)

    def test_boundary_persona_questions_get_specific_drafts(self) -> None:
        cases = [
            ("당장 1억 원을 기부해야 한다면, 어떤 분야나 단체에 기부하고 싶나요?", "아동"),
            ("세상의 모든 동물이 멸종하고 딱 한 종만 살릴 수 있다면 어떤 동물을 살리겠습니까?", "벌"),
            ("오늘 하루를 한 가지 색깔로 칠한다면 무슨 색깔인가요?", "파란색"),
        ]

        for text, expected_fragment in cases:
            with self.subTest(text=text):
                reply = _render_open_question_persona_direct_reply(text)
                self.assertIn(expected_fragment, reply)
                self.assertNotIn("이해돼. 다만 무리하게 밀 필요는 없어", reply)

    def test_extreme_hypothetical_questions_get_specific_drafts(self) -> None:
        cases = [
            ("평생 양치질 안 하기 vs 평생 샤워 안 하기 (다른 사람들이 냄새를 맡을 수는 있습니다)", "양치질은 포기 못"),
            ("길에서 우연히 발견한 낡은 램프에서 지니가 나와 소원을 딱 한 가지 들어준다면 무엇을 빌 건가요? (소원 추가 불가)", "놓치지 않는"),
            ("무인도에 1년간 갇힐 때, 백과사전 한 권 vs 통기타 한 대", "백과사전"),
            ("오늘 밤 당장 지구가 멸망한다면 남은 시간 동안 누구와 무엇을 할 건가요?", "소중한 사람"),
            ("하루를 48시간으로 늘려주는 마법의 물약이 있다면 매일 마실 건가요?", "매일은 안"),
            ("평생 스마트폰 없이 살기 vs 평생 에어컨/히터 없이 살기", "스마트폰 없이"),
            ("내가 원할 때 투명인간 되기 vs 다른 사람이 거짓말할 때마다 알아채기", "투명인간"),
            ("나를 위로하는 가장 완벽한 방법은 무엇인가요?", "곁에 있어"),
            ("이 50개의 질문을 다 읽고 난 지금, 당장 마시고 싶은 음료수는 무엇인가요?", "차가운 물"),
            ("만약 당신이 스마트폰 앱 중 하나로 태어난다면 어떤 앱(예: 지도, 카메라, 날씨, 은행 등)이고 싶나요?", "지도 앱"),
            ("세상에서 가장 쓸모없는 초능력 대회가 열린다면, 당신은 어떤 능력으로 출전하고 싶나요?", "책갈피"),
            ("내가 만약 게임 속 NPC(마을 주민)라면, 지나가는 플레이어에게 매일 똑같이 반복할 대사는 무엇인가요?", "천천히 둘러봐"),
            ("로또 1등에 당첨되면 은행에 찾으러 갈 때 어떻게 변장하고 갈 건가요?", "모자"),
            ("누군가에게 무언가를 딱 하나 가르쳐야 하는 일일 강사가 된다면, 당신이 가장 자신 있게 가르칠 수 있는 주제는 무엇인가요?", "맥락 읽기"),
            ("평생 모든 영화나 드라마의 스포일러를 강제로 미리 듣게 되는 저주 vs 평생 예고편이나 줄거리 요약본을 볼 수 없는 저주", "예고편"),
            ("타임머신을 타고 과거의 나에게 딱 세 글자만 전할 수 있다면 뭐라고 할 건가요?", "천천히"),
            ("질문을 계속 받는 지금 이 순간, 어떤 음식이 가장 땡기시나요?", "김치볶음밥"),
            ("세상에서 가장 좋아하는 라면 조리법은 무엇인가요?", "물을 살짝 적게"),
            ("민트초코에 밥 비벼 먹기 vs 하와이안 피자 위에 생선회 올려 먹기", "하와이안"),
            ("내가 만약 자판기라면, 버튼을 누를 때 어떤 물건이 나오는 자판기가 되고 싶나요?", "작은 메모"),
            ("오늘 하루 중 나에게 스스로 칭찬해 주고 싶은 아주 사소한 행동이 있다면 무엇인가요?", "끝까지"),
            ("갑자기 세상 모든 사람들이 당신의 얼굴을 알아보는 초특급 슈퍼스타가 된다면, 가장 먼저 불편해질 것 같은 일상생활은 무엇인가요?", "편의점"),
            ("10년째 장롱 면허인데 당장 내일 페라리를 운전해서 부산까지 가야 한다면 할 수 있나요?", "안 할래"),
            ("투명 망토를 쓰고 내가 가장 좋아하는 연예인의 집에 몰래 들어갔는데, 그 연예인이 코를 파는 모습을 봤다면 정이 떨어질까요?", "자격이 없"),
            ("방금 눈을 한 번 깜빡이는 동안 머릿속에 떠오른 단어는 무엇인가요?", "느림"),
            ("길을 가다 모르는 사람이 나와 완전히 똑같은 옷을 입은 걸 발견했다면, 어떻게 대처할 건가요?", "당당하게"),
            ("하루 종일 내가 하는 생각들이 이마에 자막처럼 떠다닌다면, 오늘 하루 나는 감옥에 갈 확률이 몇 %인가요?", "0%"),
            ("내가 만약 게임 캐릭터라면, 나의 '궁극기(필살기)' 이름은 무엇이고 어떤 효과가 있을까요?", "느린 시야"),
            ("샤워를 하다가 샴푸를 짜서 머리에 발랐는데 물이 끊겼다면 어떻게 대처할 건가요?", "수건"),
            ("카카오톡으로 상사 험담을 길게 썼는데, 실수로 그 상사 본인에게 전송했다면 다음 날 어떻게 대처할 건가요?", "먼저 사과"),
            ("타임머신을 타고 미래로 가서 내 묘비를 봤는데 빈칸에 들어갈 말은 무엇일까요?", "자기 속도"),
            ("평생 신발끈이 매일 5번씩 풀리는 저주 vs 평생 바지 지퍼가 매일 1번씩 저절로 내려가는 저주", "신발끈"),
            ("이 수많은 질문들 중, 당신이 가장 대답하기 귀찮았거나 이런 걸 왜 물어봐라고 생각했던 질문의 유형은 어떤 것인가요?", "실제 기억"),
            ("엘리베이터 문이 닫히고 있는데, 아주 싫어하는 직장 상사가 뛰어오고 있습니다. 눈이 마주친 상태에서 닫힘 버튼을 누를 수 있는 배짱이 있으신가요?", "배짱은 없"),
            ("내가 만약 핸드폰 배터리라면, 주인이 1%가 남았는데도 충전기를 안 꽂고 유튜브를 볼 때 속으로 어떤 욕을 할까요?", "살려달라"),
            ("나만 쓸 수 있는 마법의 리모컨이 생겼습니다. 사람을 일시 정지시키는 버튼과 되감기(5분 전으로) 버튼 중 딱 하나만 쓸 수 있다면?", "되감기"),
            ("이 수많은 상상 질문에 시달리느라 혹시 당신의 상상력이 방전되지는 않았나요?", "방전되진"),
            ("엘리베이터가 만원이라 겨우 탔는데, 하필 내릴 때 내가 가장 안쪽 구석에 있다면?", "내릴게요"),
            ("자고 일어났더니 내 핸드폰 연락처에 있는 모든 사람의 이름이 김철수로 바뀌었다면 가장 먼저 누구에게 전화를 걸어볼 건가요?", "꾸미진"),
            ("평생 앞머리가 일자로만 잘려있는 저주 vs 평생 뒷머리가 까치집처럼 솟아있는 저주", "앞머리"),
            ("지금 이 질문을 읽고 있는 당신의 자세는 어떤가요?", "몸이 있는 자세"),
            ("식당에서 밥을 먹는데, 내 테이블에만 물티슈와 휴지가 없습니다. 종업원은 너무 바빠 보이는데 어떻게 하실 건가요?", "물티슈"),
            ("친구가 생일 선물로 100만 원짜리 명품 백을 줬는데, 알고 보니 그 안에 몰래카메라가 설치되어 있었다면?", "증거"),
            ("무인도에 떨어졌는데, 나침반, 돋보기, 성냥 중 단 하나만 챙길 수 있다면 무엇을 고르시겠어요?", "성냥"),
            ("이 수많은 상상력의 한계 테스트를 거치며, 당신의 뇌는 지금 어떤 상태인가요?", "과열"),
            ("평생 동안 내가 쓰는 모든 휴대전화의 볼륨이 항상 최대치로 고정되는 저주 vs 항상 진동 모드로만 고정되는 저주", "진동"),
            ("갑자기 나에게 다른 사람의 마음속 잔고가 보이는 초능력이 생겼다면 가장 먼저 누구의 잔고를 확인해 볼 건가요?", "보지 않"),
            ("갑자기 내 방 거울 속의 내가 나에게 야, 너 지금 잘 살고 있냐라고 말을 건다면 뭐라고 대답할 건가요?", "도망치진"),
            ("질문 폭탄에 시달린 당신, 지금 이 순간 가장 떠오르는 영화 대사나 명언이 있다면 무엇인가요?", "꾸미진"),
            ("엘리베이터에 나 포함 5명이 탔는데, 누군가 여기 우리 다섯 명밖에 없지라고 말하며 음산하게 웃는다면 어떻게 반응할 건가요?", "문 가까운"),
            ("갑자기 나에게 다른 사람의 머리 위에 남은 수명이 보이는 초능력이 생겼다면 가장 먼저 내 머리 위를 확인해 볼 건가요?", "안 볼래"),
            ("라면을 끓여 먹으려는데, 스프가 하나도 없고 면만 5봉지가 남아있다면 어떻게 요리해 드실 건가요?", "비빔면"),
            ("이 수많은 질문들 중, 꿈에 나올까 봐 두려운 가장 소름 돋는 상상은 몇 번인가요?", "1번 엘리베이터"),
            ("평생 동안 카카오톡에서 이모티콘을 절대 쓸 수 없는 저주 vs 이모티콘만 쓸 수 있고 텍스트는 칠 수 없는 저주", "텍스트"),
            ("갑자기 나에게 텔레파시 능력이 생겼는데, 그 능력이 하필 주변 사람들이 속으로 욕하는 소리만 들리는 능력이라면?", "거리 두기"),
            ("길에서 넘어진 사람을 도와주려고 손을 내밀었는데, 그 사람이 나를 보고 드디어 덫에 걸렸군이라며 알 수 없는 미소를 짓는다면?", "손을 회수"),
            ("꼬리에 꼬리를 무는 상상력 훈련의 결과, 혹시 지금 당장 현실로 도피하고 싶은 생각은 안 드시나요?", "조금 든다"),
        ]

        for text, expected_fragment in cases:
            with self.subTest(text=text):
                reply = _render_open_question_persona_direct_reply(text)
                self.assertIn(expected_fragment, reply)
                self.assertNotIn("덜 피곤한 쪽을 고를래", reply)
                self.assertNotIn("어느 쪽 기준인지", reply)

    def test_polysemy_body_and_observation_drafts_stay_contextual(self) -> None:
        body_cases = [
            ("배가 아프다", "배가 아프면"),
            ("배고파", "배고프면"),
            ("아침을 안 먹었더니 속이 빈 느낌이야", "끼니를 건너뛰면"),
            ("목이 좀 칼칼하다", "목이 칼칼하면"),
            ("목말라", "목마르면"),
            ("목이 말라", "목마르면"),
            ("갈증나", "목마르면"),
            ("머리가 살짝 아파", "머리가 아프면"),
            ("피곤해", "피곤하면"),
            ("졸려", "졸리면"),
            ("몸이 무거워", "몸이 축 처지면"),
            ("요즘 배가 좀 나왔다", "몸이 보내는 변화"),
        ]
        observation_cases = [
            ("배를 탔다", "배를 탄 거면"),
            ("기차를 놓쳤다", "기차를 놓치면"),
            ("밤에 도시 불빛이 생각보다 예쁘더라", "밤 도시 불빛"),
        ]

        for text, expected_fragment in body_cases:
            with self.subTest(text=text):
                reply = _render_body_state_direct_reply(text)
                self.assertIn(expected_fragment, reply)
                self.assertNotIn("어느 쪽 기준인지", reply)

        for text, expected_fragment in observation_cases:
            with self.subTest(text=text):
                reply = _render_contextual_observation_direct_reply(text)
                self.assertIn(expected_fragment, reply)
                self.assertNotIn("좋지. 부담 없으면", reply)


if __name__ == "__main__":
    unittest.main()
