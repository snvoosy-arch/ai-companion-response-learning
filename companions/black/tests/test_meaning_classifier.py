from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from predictive_bot.core.meaning_classifier import (
    MultiHeadMeaningTorchModel,
    decode_bio_slot_spans,
    slot_spans_to_dict,
)


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = nn.Embedding(16, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return SimpleNamespace(last_hidden_state=self.proj(input_ids))


class MultiHeadMeaningTorchModelTests(unittest.TestCase):
    def test_forward_supports_sequence_heads_and_token_slot_head(self) -> None:
        model = MultiHeadMeaningTorchModel(
            _TinyEncoder(),
            head_dims={"coarse_intent": 2, "schema": 3, "speech_act": 2},
            slot_label_count=5,
            dropout=0.0,
        )

        outputs = model(
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
            coarse_intent_labels=torch.tensor([0, 1], dtype=torch.long),
            schema_labels=torch.tensor([2, 1], dtype=torch.long),
            speech_act_labels=torch.tensor([1, 0], dtype=torch.long),
            slot_labels=torch.tensor([[0, 1, 2], [0, 3, -100]], dtype=torch.long),
        )

        self.assertEqual(tuple(outputs["coarse_intent_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["schema_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["speech_act_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["slot_logits"].shape), (2, 3, 5))
        self.assertTrue(torch.isfinite(outputs["loss"]))

    def test_forward_ignores_token_type_ids_for_modernbert_compatible_encoders(self) -> None:
        model = MultiHeadMeaningTorchModel(
            _TinyEncoder(),
            head_dims={"coarse_intent": 2},
            slot_label_count=None,
            dropout=0.0,
        )

        outputs = model(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
            token_type_ids=torch.tensor([[0, 0, 0]], dtype=torch.long),
        )

        self.assertEqual(tuple(outputs["coarse_intent_logits"].shape), (1, 2))


class SlotDecoderTests(unittest.TestCase):
    def test_decode_bio_slot_spans_turns_token_logits_into_slot_dict(self) -> None:
        text = "오늘 바다에서 수영하자"
        offsets = [(0, 0), (0, 2), (3, 5), (8, 10), (10, 12), (0, 0)]
        id2label = {0: "O", 1: "B-time", 2: "B-place", 3: "B-activity", 4: "I-activity"}
        slot_ids = [0, 1, 2, 3, 4, 0]
        logits = torch.full((len(slot_ids), len(id2label)), -5.0)
        for token_index, label_id in enumerate(slot_ids):
            logits[token_index, label_id] = 5.0

        spans = decode_bio_slot_spans(text=text, offsets=offsets, slot_logits=logits, id2label=id2label)

        self.assertEqual(
            [(span.label, span.value) for span in spans],
            [("time", "오늘"), ("place", "바다"), ("activity", "수영하자")],
        )
        self.assertEqual(slot_spans_to_dict(spans), {"time": "오늘", "place": "바다", "activity": "수영하자"})
        self.assertEqual(slot_spans_to_dict(spans, min_confidence=1.1), {})

    def test_slot_spans_to_dict_strips_common_korean_particles(self) -> None:
        text = "오늘은 캠핑장에 가자"
        offsets = [(0, 3), (4, 8)]
        id2label = {0: "B-time", 1: "B-place"}
        logits = torch.full((2, 2), -5.0)
        logits[0, 0] = 5.0
        logits[1, 1] = 5.0
        spans = decode_bio_slot_spans(text=text, offsets=offsets, slot_logits=logits, id2label=id2label)

        self.assertEqual(slot_spans_to_dict(spans), {"time": "오늘", "place": "캠핑장"})

    def test_slot_spans_to_dict_filters_common_runtime_noise(self) -> None:
        spans = [
            SimpleNamespace(label="place", value="오늘", confidence=0.8),
            SimpleNamespace(label="place", value="바다가", confidence=0.8),
            SimpleNamespace(label="activity", value="수영", confidence=0.8),
            SimpleNamespace(label="activity", value="이나", confidence=0.8),
            SimpleNamespace(label="process", value="필요한", confidence=0.8),
            SimpleNamespace(label="process", value="해", confidence=0.8),
            SimpleNamespace(label="topic", value="장비", confidence=0.8),
            SimpleNamespace(label="topic", value="뭐", confidence=0.8),
        ]

        self.assertEqual(
            slot_spans_to_dict(spans),
            {
                "place": "바다",
                "activity": "수영",
                "process": "필요한",
                "topic": "장비",
            },
        )

    def test_slot_spans_to_dict_normalizes_common_runtime_suffixes(self) -> None:
        spans = [
            SimpleNamespace(label="place", value="집에서", confidence=0.8),
            SimpleNamespace(label="activity", value="바베큐 구워", confidence=0.8),
            SimpleNamespace(label="activity", value="텐트 치", confidence=0.8),
            SimpleNamespace(label="activity", value="물놀", confidence=0.8),
            SimpleNamespace(label="activity", value="걷는", confidence=0.3),
            SimpleNamespace(label="condition", value="눈", confidence=0.3),
            SimpleNamespace(label="process", value="확인해야", confidence=0.8),
            SimpleNamespace(label="process", value="저장해", confidence=0.25),
            SimpleNamespace(label="topic", value="지도부터", confidence=0.8),
            SimpleNamespace(label="topic", value="준비할", confidence=0.8),
            SimpleNamespace(label="process", value="알려", confidence=0.8),
            SimpleNamespace(label="people", value="혼자", confidence=0.25),
            SimpleNamespace(label="topic", value="창문은", confidence=0.31),
            SimpleNamespace(label="activity", value="소풍", confidence=0.27),
            SimpleNamespace(label="activity", value="조깅", confidence=0.31),
            SimpleNamespace(label="process", value="가져가도", confidence=0.41),
            SimpleNamespace(label="topic", value="골목은", confidence=0.29),
        ]

        self.assertEqual(
            slot_spans_to_dict(spans, min_confidence=0.35),
            {
                "place": "집",
                "activity": "바베큐|텐트|물놀이|걷|소풍|조깅",
                "condition": "눈",
                "process": "확인|저장|준비|가져가",
                "topic": "지도|창문|골목",
                "people": "혼자",
            },
        )

    def test_slot_spans_to_dict_merges_runtime_fragments(self) -> None:
        spans = [
            SimpleNamespace(label="place", value="캠핑", start=0, end=2, confidence=0.45),
            SimpleNamespace(label="topic", value="장", start=2, end=3, confidence=0.2),
            SimpleNamespace(label="topic", value="불 세", start=10, end=13, confidence=0.56),
            SimpleNamespace(label="topic", value="기", start=13, end=14, confidence=0.39),
            SimpleNamespace(label="topic", value="운동", start=20, end=22, confidence=0.26),
            SimpleNamespace(label="process", value="루", start=23, end=24, confidence=0.13),
            SimpleNamespace(label="choice", value="틴", start=24, end=25, confidence=0.13),
            SimpleNamespace(label="topic", value="스트레", start=30, end=33, confidence=0.51),
            SimpleNamespace(label="topic", value="칭", start=33, end=34, confidence=0.2),
            SimpleNamespace(label="topic", value="돗", start=40, end=41, confidence=0.42),
            SimpleNamespace(label="process", value="자리", start=41, end=43, confidence=0.32),
            SimpleNamespace(label="topic", value="물", start=50, end=51, confidence=0.65),
            SimpleNamespace(label="topic", value="병이", start=51, end=53, confidence=0.31),
            SimpleNamespace(label="activity", value="운동", start=60, end=62, confidence=0.34),
            SimpleNamespace(label="process", value="루틴", start=63, end=65, confidence=0.25),
            SimpleNamespace(label="topic", value="예매", start=70, end=72, confidence=0.41),
            SimpleNamespace(label="topic", value="표", start=72, end=73, confidence=0.30),
            SimpleNamespace(label="topic", value="방수", start=80, end=82, confidence=0.48),
            SimpleNamespace(label="process", value="팩", start=82, end=83, confidence=0.28),
            SimpleNamespace(label="activity", value="줄넘", start=90, end=92, confidence=0.34),
            SimpleNamespace(label="activity", value="기", start=92, end=93, confidence=0.16),
            SimpleNamespace(label="time", value="퇴근", start=100, end=102, confidence=0.42),
            SimpleNamespace(label="time", value="길에", start=102, end=104, confidence=0.43),
            SimpleNamespace(label="time", value="늦은", start=110, end=112, confidence=0.18),
            SimpleNamespace(label="time", value="밤", start=113, end=114, confidence=0.78),
            SimpleNamespace(label="condition", value="비", start=120, end=121, confidence=0.44),
            SimpleNamespace(label="condition", value="바람", start=121, end=123, confidence=0.39),
        ]

        self.assertEqual(
            slot_spans_to_dict(spans, min_confidence=0.35),
            {
                "place": "캠핑장",
                "topic": "불 세기|돗자리|물병|예매표|방수팩",
                "process": "운동 루틴",
                "activity": "스트레칭|줄넘기",
                "condition": "비바람",
                "time": "퇴근길|늦은 밤",
            },
        )

    def test_slot_spans_to_dict_repairs_common_label_slips(self) -> None:
        spans = [
            SimpleNamespace(label="activity", value="도서관", confidence=0.38),
            SimpleNamespace(label="place", value="기차", confidence=0.31),
            SimpleNamespace(label="activity", value="바람", confidence=0.24),
            SimpleNamespace(label="time", value="혼자", confidence=0.55),
            SimpleNamespace(label="process", value="연락", confidence=0.21),
            SimpleNamespace(label="comparison", value="기다릴", confidence=0.22),
            SimpleNamespace(label="topic", value="발표할", confidence=0.28),
            SimpleNamespace(label="topic", value="핫팩", confidence=0.18),
            SimpleNamespace(label="activity", value="낚시", confidence=0.22),
            SimpleNamespace(label="time", value="퇴근길", confidence=0.22),
            SimpleNamespace(label="process", value="충전했는지", confidence=0.22),
            SimpleNamespace(label="place", value="캠핑", confidence=0.41),
            SimpleNamespace(label="place", value="낚시 가기", confidence=0.41),
            SimpleNamespace(label="activity", value="새벽", confidence=0.39),
            SimpleNamespace(label="activity", value="아이들이", confidence=0.39),
        ]

        self.assertEqual(
            slot_spans_to_dict(spans, min_confidence=0.35),
            {
                "place": "도서관|캠핑",
                "activity": "기차|낚시",
                "condition": "바람",
                "people": "혼자|아이들",
                "decision": "연락",
                "comparison": "기다릴",
                "process": "발표|충전",
                "topic": "핫팩",
                "time": "퇴근길|새벽",
            },
        )


if __name__ == "__main__":
    unittest.main()
