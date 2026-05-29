from __future__ import annotations

import unittest

from predictive_bot.core.classifier import HeuristicIntentClassifier
from predictive_bot.core.models import ConversationState


class HeuristicIntentClassifierOODTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = HeuristicIntentClassifier()
        self.state = ConversationState(user_id="ood-user")

    def test_classifier_handles_ood_daily_phrases(self) -> None:
        cases = [
            ("하이하이", "greeting"),
            ("와썹", "greeting"),
            ("지금 뭐 하고 있었냐", "smalltalk_generic"),
            ("요새 좀 처진다", "smalltalk_feeling"),
            ("기분이 하루종일 가라앉네", "smalltalk_feeling"),
            ("이쪽이 낫냐 저쪽이 낫냐", "smalltalk_opinion"),
            ("이거 별론가", "smalltalk_opinion"),
            ("너 정체가 뭐냐", "who_are_you"),
            ("무슨 봇인데", "who_are_you"),
            ("가능한 거 대충 읊어봐", "help"),
            ("뭐까지 할 줄 앎", "help"),
            ("왜 또 씹음", "reply_request"),
            ("살아는 있냐", "reply_request"),
            ("ㅇㅋ 알겠음", "confirm"),
            ("그건 아닌 듯", "deny"),
            ("밖에 비 오냐", "weather"),
            ("우산 챙겨야 됨?", "weather"),
            ("패딩 입어야 되냐", "weather"),
            ("이 말 무슨 뜻임", "search_request"),
            ("이 표현 무슨 의미냐", "search_request"),
            ("같이 한 판 할래", "game_invite"),
            ("롤 한 판 ㄱ?", "game_invite"),
            ("무슨 게임 함", "game_talk"),
            ("요즘 무슨 노래 들음", "music"),
            ("플리 뭐 듣냐", "music"),
            ("볼만한 거 있냐", "media_recommend"),
            ("넷플 뭐 보지", "media_recommend"),
            ("ㅋㅋ 바보", "tease"),
            ("ㅋㅋ 어이없네", "laugh"),
            ("헐 뭐야 이거", "surprise"),
            ("진짜 못하네", "hostile"),
        ]

        for text, expected in cases:
            with self.subTest(text=text):
                features = self.classifier.classify(text, self.state)
                self.assertEqual(features.intent.value, expected)


if __name__ == "__main__":
    unittest.main()
