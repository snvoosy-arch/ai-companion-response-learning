"""Multi-head meaning classifier for Black's contextual understanding layer."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


NONE_LABEL = "__none__"
HEAD_CONFIG_FILE = "meaning_head_config.json"
HEAD_WEIGHTS_FILE = "meaning_heads.pt"
ENCODER_DIR = "encoder"


@dataclass(slots=True)
class AxisPrediction:
    label: str | None
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SlotSpan:
    label: str
    value: str
    start: int
    end: int
    confidence: float


@dataclass(slots=True)
class MultiHeadMeaningPrediction:
    coarse_intent: AxisPrediction
    schema: AxisPrediction
    speech_act: AxisPrediction
    domain: AxisPrediction | None = None
    extra_axes: dict[str, AxisPrediction] = field(default_factory=dict)
    slots: dict[str, str] = field(default_factory=dict)
    slot_spans: list[SlotSpan] = field(default_factory=list)

    @property
    def intent(self) -> str:
        return str(self.coarse_intent.label or "unknown")

    @property
    def confidence(self) -> float:
        return float(self.coarse_intent.confidence)

    @property
    def scores(self) -> dict[str, float]:
        return dict(self.coarse_intent.scores)


class MultiHeadMeaningTorchModel(nn.Module):
    """Transformer encoder with sequence heads and an optional token slot head."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        head_dims: dict[str, int],
        slot_label_count: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = int(getattr(encoder.config, "hidden_size"))
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_size, dim) for name, dim in head_dims.items()}
        )
        self.slot_head = nn.Linear(hidden_size, slot_label_count) if slot_label_count else None

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        labels = {key: batch.pop(key) for key in tuple(f"{name}_labels" for name in self.heads) if key in batch}
        slot_labels = batch.pop("slot_labels", None)
        # ModernBERT does not consume segment ids. Tokenizers may still emit
        # all-zero token_type_ids through common HF batching paths.
        batch.pop("token_type_ids", None)
        outputs = self.encoder(**batch)
        last_hidden_state = outputs.last_hidden_state
        pooled = self._mean_pool(last_hidden_state, batch.get("attention_mask"))
        pooled = self.dropout(pooled)
        pooled = pooled.to(next(self.heads.parameters()).dtype)
        logits = {name: head(pooled) for name, head in self.heads.items()}

        result: dict[str, torch.Tensor] = {f"{name}_logits": value for name, value in logits.items()}
        if self.slot_head is not None:
            token_features = self.dropout(last_hidden_state).to(next(self.slot_head.parameters()).dtype)
            result["slot_logits"] = self.slot_head(token_features)

        if labels or slot_labels is not None:
            losses: list[torch.Tensor] = []
            loss_fct = nn.CrossEntropyLoss()
            for head in self.heads:
                label_key = f"{head}_labels"
                if label_key in labels:
                    losses.append(loss_fct(logits[head], labels[label_key]))
            if slot_labels is not None and "slot_logits" in result:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                slot_logits = result["slot_logits"]
                losses.append(slot_loss_fct(slot_logits.reshape(-1, slot_logits.shape[-1]), slot_labels.reshape(-1)))
            if losses:
                result["loss"] = sum(losses) / len(losses)
        return result

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom


class MultiHeadMeaningClassifier:
    """Runtime wrapper for a fine-tuned multi-head meaning classifier."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        device: str = "auto",
        max_length: int = 128,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._device = torch.device(self._resolve_device(device))
        self._max_length = max_length

        config_path = self._model_dir / HEAD_CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(f"missing meaning head config: {config_path}")
        self._config = json.loads(config_path.read_text(encoding="utf-8"))
        self._id2label = {
            head: {int(idx): str(label) for idx, label in payload["id2label"].items()}
            for head, payload in self._config["heads"].items()
        }
        slot_payload = self._config.get("slot_labels") if isinstance(self._config.get("slot_labels"), dict) else None
        self._slot_id2label = (
            {int(idx): str(label) for idx, label in slot_payload["id2label"].items()}
            if slot_payload and isinstance(slot_payload.get("id2label"), dict)
            else {}
        )
        metadata = self._config.get("metadata") if isinstance(self._config.get("metadata"), dict) else {}
        self._slot_min_confidence = float(metadata.get("slot_min_confidence", 0.35))

        encoder_dir = self._model_dir / ENCODER_DIR
        encoder_source = encoder_dir if encoder_dir.exists() else self._model_dir
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        encoder = AutoModel.from_pretrained(str(encoder_source))
        self._model = MultiHeadMeaningTorchModel(
            encoder,
            head_dims={head: len(labels) for head, labels in self._id2label.items()},
            slot_label_count=len(self._slot_id2label) or None,
            dropout=0.0,
        )
        weights_path = self._model_dir / HEAD_WEIGHTS_FILE
        state = torch.load(weights_path, map_location="cpu")
        if any(str(key).startswith("encoder.") for key in state):
            self._model.load_state_dict(state, strict=False)
        else:
            state_keys = [str(key) for key in state]
            if any(key.startswith("heads.") or key.startswith("slot_head.") for key in state_keys):
                self._model.load_state_dict(state, strict=False)
            else:
                self._model.heads.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> MultiHeadMeaningPrediction:
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=bool(self._slot_id2label),
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )
        offset_mapping = encoded.pop("offset_mapping", None)
        inputs = encoded.to(self._device)
        outputs = self._model(**inputs)
        slot_spans: list[SlotSpan] = []
        if self._slot_id2label and offset_mapping is not None and "slot_logits" in outputs:
            slot_spans = decode_bio_slot_spans(
                text=text,
                offsets=[tuple(map(int, pair)) for pair in offset_mapping[0].detach().cpu().tolist()],
                slot_logits=outputs["slot_logits"][0],
                id2label=self._slot_id2label,
            )
        coarse_intent = self._axis_prediction(outputs["coarse_intent_logits"][0], "coarse_intent")
        schema = self._axis_prediction(outputs["schema_logits"][0], "schema")
        speech_act = self._axis_prediction(outputs["speech_act_logits"][0], "speech_act")
        domain = (
            self._axis_prediction(outputs["domain_logits"][0], "domain")
            if "domain" in self._id2label and "domain_logits" in outputs
            else None
        )
        core_axes = {"coarse_intent", "domain", "schema", "speech_act"}
        extra_axes = {
            head: self._axis_prediction(outputs[f"{head}_logits"][0], head)
            for head in self._id2label
            if head not in core_axes and f"{head}_logits" in outputs
        }
        coarse_intent, schema, speech_act = _postprocess_axis_predictions(text, coarse_intent, schema, speech_act)
        return MultiHeadMeaningPrediction(
            coarse_intent=coarse_intent,
            schema=schema,
            speech_act=speech_act,
            domain=domain,
            extra_axes=extra_axes,
            slots=slot_spans_to_dict(slot_spans, min_confidence=self._slot_min_confidence),
            slot_spans=slot_spans,
        )

    def _axis_prediction(self, logits: torch.Tensor, head: str) -> AxisPrediction:
        probs = torch.softmax(logits.detach().cpu(), dim=-1)
        top_idx = int(torch.argmax(probs).item())
        id2label = self._id2label[head]
        raw_label = id2label.get(top_idx, "unknown")
        label = None if raw_label == NONE_LABEL else raw_label
        scores = {
            (None if id2label[i] == NONE_LABEL else id2label[i]): float(probs[i].item())
            for i in range(len(probs))
            if float(probs[i].item()) > 0.01
        }
        sorted_scores = dict(sorted(((str(k), v) for k, v in scores.items()), key=lambda x: x[1], reverse=True))
        return AxisPrediction(label=label, confidence=float(probs[top_idx].item()), scores=sorted_scores)

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return normalized


def _parse_bio_label(raw_label: str) -> tuple[str, str | None]:
    label = str(raw_label or "").strip()
    if not label or label in {"O", NONE_LABEL, "PAD", "SPECIAL"}:
        return "O", None
    if "-" in label:
        prefix, name = label.split("-", 1)
        normalized_prefix = prefix.upper()
        if normalized_prefix in {"B", "I"} and name.strip():
            return normalized_prefix, name.strip()
    return "B", label


def _postprocess_axis_predictions(
    text: str,
    coarse_intent: AxisPrediction,
    schema: AxisPrediction,
    speech_act: AxisPrediction,
) -> tuple[AxisPrediction, AxisPrediction, AxisPrediction]:
    normalized = str(text or "").strip()
    if (
        schema.label is None
        and speech_act.label == "react"
        and normalized.startswith(("안녕", "하이", "ㅎㅇ", "반가워"))
    ):
        scores = dict(coarse_intent.scores)
        scores["greeting"] = max(float(scores.get("greeting", 0.0)), 0.99)
        return AxisPrediction(label="greeting", confidence=max(coarse_intent.confidence, 0.99), scores=scores), schema, speech_act
    return coarse_intent, schema, speech_act


def decode_bio_slot_spans(
    *,
    text: str,
    offsets: list[tuple[int, int]],
    slot_logits: torch.Tensor,
    id2label: dict[int, str],
) -> list[SlotSpan]:
    probs = torch.softmax(slot_logits.detach().cpu(), dim=-1)
    predicted_ids = torch.argmax(probs, dim=-1).tolist()
    spans: list[SlotSpan] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    current_scores: list[float] = []

    def close_current() -> None:
        nonlocal current_label, current_start, current_end, current_scores
        if current_label is not None and current_start is not None and current_end is not None:
            value = text[current_start:current_end].strip()
            if value:
                spans.append(
                    SlotSpan(
                        label=current_label,
                        value=value,
                        start=current_start,
                        end=current_end,
                        confidence=sum(current_scores) / max(1, len(current_scores)),
                    )
                )
        current_label = None
        current_start = None
        current_end = None
        current_scores = []

    for token_index, (start, end) in enumerate(offsets[: len(predicted_ids)]):
        if start == end:
            close_current()
            continue
        raw_label = id2label.get(int(predicted_ids[token_index]), "O")
        prefix, label = _parse_bio_label(raw_label)
        confidence = float(probs[token_index, int(predicted_ids[token_index])].item())
        if label is None:
            close_current()
            continue
        if prefix == "B" or current_label != label or current_end is None or start > current_end + 1:
            close_current()
            current_label = label
            current_start = start
            current_end = end
            current_scores = [confidence]
            continue
        current_end = max(current_end, end)
        current_scores.append(confidence)
    close_current()
    return spans


def slot_spans_to_dict(spans: list[SlotSpan], *, min_confidence: float = 0.0) -> dict[str, str]:
    slots: dict[str, str] = {}
    for span in _merge_slot_fragments(spans):
        label = span.label.strip()
        label, value = _normalize_slot_label_and_value(label, span.value.strip())
        if not label or not value or _is_noise_slot_value(label, value):
            continue
        if (
            span.confidence < min_confidence
            and not (min_confidence <= 0.5 and _allow_low_confidence_slot(label, value, span.confidence))
        ):
            continue
        if label not in slots:
            slots[label] = value
            continue
        existing_values = [part.strip() for part in slots[label].split("|") if part.strip()]
        if value not in existing_values:
            slots[label] = "|".join([*existing_values, value])
    return slots


def _merge_slot_fragments(spans: list[SlotSpan]) -> list[SlotSpan]:
    merged: list[SlotSpan] = []
    index = 0
    while index < len(spans):
        span = spans[index]
        next_span = spans[index + 1] if index + 1 < len(spans) else None
        third_span = spans[index + 2] if index + 2 < len(spans) else None

        if (
            next_span is not None
            and _span_value(span) == "캠핑"
            and _span_value(next_span) == "장"
            and _is_adjacent_span(span, next_span)
        ):
            merged.append(_combined_span("place", "캠핑장", [span, next_span]))
            index += 2
            continue

        if (
            next_span is not None
            and _span_value(span) == "불 세"
            and _span_value(next_span) == "기"
            and _is_adjacent_span(span, next_span)
        ):
            merged.append(_combined_span("topic", "불 세기", [span, next_span]))
            index += 2
            continue

        if (
            next_span is not None
            and _span_value(span) == "스트레"
            and _span_value(next_span) == "칭"
            and _is_adjacent_span(span, next_span)
        ):
            merged.append(_combined_span("activity", "스트레칭", [span, next_span]))
            index += 2
            continue

        if (
            next_span is not None
            and third_span is not None
            and _span_value(span) in {"운동", "운동 루"}
            and _span_value(next_span) == "루"
            and _span_value(third_span) == "틴"
            and _is_adjacent_span(span, next_span)
            and _is_adjacent_span(next_span, third_span)
        ):
            merged.append(_combined_span("process", "운동 루틴", [span, next_span, third_span]))
            index += 3
            continue

        if (
            next_span is not None
            and _span_value(span) == "운동 루"
            and _span_value(next_span) == "틴"
            and _is_adjacent_span(span, next_span)
        ):
            merged.append(_combined_span("process", "운동 루틴", [span, next_span]))
            index += 2
            continue

        if (
            next_span is not None
            and _span_value(span) == "돗"
            and _span_value(next_span).startswith("자리")
            and _is_adjacent_span(span, next_span)
        ):
            merged.append(_combined_span("topic", "돗자리", [span, next_span]))
            index += 2
            continue

        if (
            next_span is not None
            and _span_value(span) == "물"
            and _span_value(next_span).startswith("병")
            and _is_adjacent_span(span, next_span)
        ):
            merged.append(_combined_span("topic", "물병", [span, next_span]))
            index += 2
            continue

        fragment_pairs = {
            ("운동", "루틴"): ("process", "운동 루틴"),
            ("학생", "증"): ("topic", "학생증"),
            ("쓰레기", "봉투"): ("topic", "쓰레기봉투"),
            ("메모리", "카드"): ("topic", "메모리카드"),
            ("예매", "표"): ("topic", "예매표"),
            ("방수", "팩"): ("topic", "방수팩"),
            ("예약", "번호"): ("topic", "예약 번호"),
            ("파일", "백업"): ("topic", "파일 백업"),
            ("선물", "포장"): ("topic", "선물 포장"),
            ("문", "단 순서"): ("topic", "문단 순서"),
            ("결정", "사항"): ("topic", "결정사항"),
            ("재료", "손질"): ("topic", "재료 손질"),
            ("하", "체"): ("topic", "하체"),
            ("상", "체"): ("topic", "상체"),
            ("본", "운동"): ("topic", "본운동"),
            ("창", "밖"): ("topic", "창밖"),
            ("노을", "빛"): ("topic", "노을빛"),
            ("줄넘", "기"): ("activity", "줄넘기"),
            ("비", "바람"): ("condition", "비바람"),
            ("퇴근", "길"): ("time", "퇴근길"),
            ("퇴근", "길에"): ("time", "퇴근길"),
            ("늦은", "밤"): ("time", "늦은 밤"),
        }
        if next_span is not None:
            pair_key = (_span_value(span), _span_value(next_span))
            if pair_key in fragment_pairs and _is_adjacent_span(span, next_span):
                merged_label, merged_value = fragment_pairs[pair_key]
                merged.append(_combined_span(merged_label, merged_value, [span, next_span]))
                index += 2
                continue

        merged.append(span)
        index += 1
    return merged


def _span_value(span: SlotSpan) -> str:
    return str(getattr(span, "value", "")).strip()


def _is_adjacent_span(left: SlotSpan, right: SlotSpan) -> bool:
    left_end = getattr(left, "end", None)
    right_start = getattr(right, "start", None)
    return isinstance(left_end, int) and isinstance(right_start, int) and right_start <= left_end + 1


def _combined_span(label: str, value: str, spans: list[SlotSpan]) -> SlotSpan:
    starts = [getattr(span, "start", 0) for span in spans if isinstance(getattr(span, "start", None), int)]
    ends = [getattr(span, "end", 0) for span in spans if isinstance(getattr(span, "end", None), int)]
    scores = [float(getattr(span, "confidence", 0.0)) for span in spans]
    return SlotSpan(
        label=label,
        value=value,
        start=min(starts) if starts else 0,
        end=max(ends) if ends else 0,
        confidence=max(scores) if scores else 0.0,
    )


def _clean_slot_value(value: str) -> str:
    cleaned = value.strip().strip("?!.,")
    if cleaned in {"가져가", "들어가", "나가"}:
        return cleaned
    for suffix in ("에서", "으로", "로", "에게", "한테", "부터", "까지", "은", "는", "이", "가", "을", "를", "에", "도", "만"):
        if len(cleaned) > len(suffix) and cleaned.endswith(suffix):
            return cleaned[: -len(suffix)].strip()
    return cleaned


def _normalize_slot_label_and_value(label: str, value: str) -> tuple[str, str]:
    normalized = _normalize_slot_value(label, _clean_slot_value(value))
    normalized_label = _normalize_slot_label(label, normalized)
    if normalized_label != label:
        normalized = _normalize_slot_value(normalized_label, normalized)
    return normalized_label, normalized


def _normalize_slot_label(label: str, value: str) -> str:
    known_activities = {
        "수영",
        "물놀이",
        "산책",
        "조깅",
        "운전",
        "농구",
        "자전거",
        "공연",
        "사진",
        "운동",
        "바베큐",
        "기차",
        "비행기",
        "영화",
        "전시",
        "공부",
        "피크닉",
        "소풍",
        "낚시",
        "회의",
        "택시",
        "줄넘기",
        "라면",
        "책",
        "요리",
        "피크닉",
        "걷",
        "돌",
    }
    if label == "activity" and value in {
        "바다",
        "계곡",
        "한강",
        "실내",
        "집",
        "카페",
        "공원",
        "도서관",
        "캠핑장",
        "해변",
        "바닷가",
        "해수욕장",
        "영화관",
        "미술관",
        "시장",
        "편의점",
        "놀이공원",
        "피시방",
        "수영장",
        "헬스장",
        "강릉",
        "운동장",
    }:
        return "place"
    if label == "activity" and value in {"비", "눈", "바람", "습한", "더운", "추운", "흐린", "햇빛"}:
        return "condition"
    if label == "place" and (value in known_activities or any(value.startswith(f"{activity} ") or value.startswith(f"{activity}가") for activity in known_activities)):
        return "activity"
    if label == "place" and value in {"기차", "비행기"}:
        return "activity"
    if label == "place" and value in {"낮", "낮에"}:
        return "time"
    if label == "activity" and value in {"오늘", "밤", "새벽", "아침", "점심", "저녁", "주말", "겨울", "낮"}:
        return "time"
    if label == "activity" and value.startswith("아이들"):
        return "people"
    if label == "time" and value in {"혼자", "친구", "둘", "여럿"}:
        return "people"
    if label == "process" and value in {"연락", "나갈"}:
        return "decision"
    if label == "comparison" and value in {"연락", "사과", "약속", "전화"}:
        return "decision"
    if label == "topic" and value in known_activities:
        return "activity"
    if label == "topic" and (value in {"준비", "저장", "확인"} or value.startswith(("준비", "저장", "확인"))):
        return "process"
    if label == "topic" and value in {"스트레칭"}:
        return "activity"
    if label == "topic" and value.startswith("발표"):
        return "process"
    return label


def _normalize_slot_value(label: str, value: str) -> str:
    normalized = value.strip()
    if label == "place":
        known_place_prefixes = (
            "캠핑장",
            "계곡",
            "수영장",
            "한강",
            "해변",
            "도서관",
            "카페",
            "공원",
            "집",
            "시장",
            "피시방",
            "헬스장",
            "편의점",
            "미술관",
            "영화관",
            "야외",
            "옥상",
            "방",
        )
        for prefix in known_place_prefixes:
            if normalized.startswith(prefix):
                return prefix
        for prefix in ("낚시", "캠핑", "택시", "비행기", "기차"):
            if normalized.startswith(prefix):
                return prefix
    if label == "activity":
        for suffix in (" 구워먹", " 구워", " 하기", " 치자", " 치", " 하자", " 보자", " 먹자", " 찍자", "이나"):
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                normalized = normalized[: -len(suffix)].strip()
        for prefix in ("비행기", "기차"):
            if normalized.startswith(prefix):
                return prefix
        if normalized.startswith("줄넘"):
            return "줄넘기"
        if normalized.startswith("산책"):
            return "산책"
        if normalized.startswith("돌"):
            return "돌"
        if normalized == "조":
            return "조깅"
        if normalized == "물놀":
            return "물놀이"
        if normalized == "놀":
            return "놀이"
        if normalized == "걷기":
            return "걷"
        if normalized.endswith("는") and len(normalized) > 2:
            return normalized[:-1].strip()
    if label == "process":
        if normalized.startswith("확인"):
            return "확인"
        if normalized.startswith("저장"):
            return "저장"
        if normalized.startswith("준비"):
            return "준비"
        if normalized.startswith("챙기"):
            return "챙기"
        if normalized.startswith("가져"):
            return "가져가"
        if normalized.startswith("봐"):
            return "봐야"
        if normalized.startswith("정하") or normalized.startswith("정할"):
            return "정할"
        if normalized.startswith("필요하지"):
            return "필요"
        if normalized.startswith("공유"):
            return "공유"
        if normalized.startswith("묶"):
            return "묶"
        if normalized.startswith("씻"):
            return "씻"
        if normalized.startswith("충전"):
            return "충전"
        if normalized.startswith("발표"):
            return "발표"
        if normalized.startswith("사과"):
            return "사과"
        if normalized.startswith("공부"):
            return "공부"
        if normalized.startswith("필요"):
            return normalized
        for suffix in ("까", "야", "해", "할"):
            if len(normalized) > len(suffix) + 1 and normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
    if label == "topic":
        if normalized.startswith("지도"):
            return "지도"
        if normalized == "불 세":
            return "불 세기"
        if normalized == "창문":
            return "창문"
        if normalized == "우산":
            return "우산"
        if normalized == "돗자리":
            return "돗자리"
        if normalized == "물병":
            return "물병"
        if normalized.startswith("골목"):
            return "골목"
        if normalized in {
            "손전등",
            "이어폰",
            "간식",
            "골목",
            "계획",
            "결론",
            "학생증",
            "쓰레기봉투",
            "메모리카드",
            "예약 번호",
            "파일 백업",
            "선물 포장",
            "문단 순서",
            "하체",
            "상체",
            "버스 창문",
            "창밖",
            "예매표",
            "미끼",
            "의자",
            "핫팩",
            "여권",
            "방수팩",
            "재료",
            "이어폰",
            "위치",
            "자료",
            "워밍업",
            "본운동",
            "노을빛",
            "음악",
            "생각",
            "제목",
            "결정사항",
            "재료 손질",
            "옷",
            "충전기",
        }:
            return normalized
    if label == "comparison":
        if normalized.startswith("나가"):
            return "나가"
        if normalized.startswith("잡"):
            return "잡"
        if normalized.startswith("쉴"):
            return "쉴"
        if normalized.endswith("에서") and len(normalized) > 2:
            return normalized[:-2].strip()
    return normalized


def _allow_low_confidence_slot(label: str, value: str, confidence: float) -> bool:
    if confidence < 0.10:
        return False
    if label == "time" and value in {
        "오늘",
        "오늘 밤",
        "밤",
        "새벽",
        "아침",
        "점심",
        "저녁",
        "주말",
        "주말 낮",
        "퇴근",
        "퇴근길",
        "늦은 밤",
        "겨울",
    }:
        return True
    if label == "condition" and value in {
        "비",
        "눈",
        "바람",
        "비바람",
        "폭염",
        "습도",
        "햇살",
        "날",
        "선선한",
        "더운",
        "추운",
        "흐린",
        "습한",
        "햇빛",
    }:
        return True
    if label == "activity" and value in {
        "걷",
        "산책",
        "수영",
        "물놀이",
        "스트레칭",
        "외출",
        "소풍",
        "피크닉",
        "조깅",
        "운전",
        "농구",
        "운동",
        "자전거",
        "공연",
        "사진",
        "기차",
        "비행기",
        "바베큐",
        "캠핑",
        "낚시",
        "회의",
        "택시",
        "줄넘기",
        "라면",
        "책",
        "요리",
        "피크닉",
        "돌",
    }:
        return True
    if label == "place" and (
        value in {"실내", "집", "카페", "공원", "도서관", "캠핑", "캠핑장", "바다", "계곡", "한강", "밖", "야외", "옥상", "방", "시장", "피시방", "헬스장", "편의점", "미술관", "영화관", "수영장", "강릉", "운동장"}
        or len(value) >= 5
    ):
        return True
    if label == "people" and value in {"혼자", "친구", "둘", "여럿"}:
        return True
    if label == "people" and value.startswith("아이들"):
        return True
    if label == "process" and value in {"저장", "준비", "확인", "챙기", "가져가", "사과", "공부", "발표", "운동 루틴", "공유", "묶", "씻", "충전", "필요", "봐야"}:
        return True
    if label == "decision" and value in {"연락", "나갈", "사과"}:
        return True
    if label == "comparison" and value in {"기다릴", "비싼", "싼", "집", "나갈", "문자", "오늘", "내일", "잡", "쉴", "카페", "핵심", "길게"}:
        return True
    if label == "topic" and value in {
        "지도",
        "우산",
        "창문",
        "불 세기",
        "돗자리",
        "물병",
        "손전등",
        "이어폰",
        "간식",
        "골목",
        "계획",
        "결론",
        "학생증",
        "쓰레기봉투",
        "메모리카드",
        "예약 번호",
        "파일 백업",
        "선물 포장",
        "문단 순서",
        "하체",
        "상체",
        "버스 창문",
        "창밖",
        "예매표",
        "미끼",
        "의자",
        "핫팩",
        "여권",
        "방수팩",
        "재료",
        "이어폰",
        "위치",
        "자료",
        "워밍업",
        "본운동",
        "노을빛",
        "음악",
        "생각",
        "제목",
        "결정사항",
        "재료 손질",
        "옷",
        "충전기",
    }:
        return True
    return False


def _is_noise_slot_value(label: str, value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return True
    if normalized in {
        "?",
        "!",
        ".",
        ",",
        "뭐",
        "거",
        "것",
        "게",
        "해",
        "야",
        "돼",
        "되",
        "할",
        "하자",
        "자",
        "때",
        "전",
        "전에",
        "선",
        "에서",
        "하고",
        "면서",
        "이나",
        "말해",
        "봐",
        "좋아",
        "좋을까",
        "알려",
        "줘",
        "면",
        "터",
        "기",
        "오는",
        "그냥",
        "봐도",
    }:
        return True
    if label == "place" and normalized in {
        "오늘",
        "오늘 밤",
        "밤",
        "새벽",
        "아침",
        "점심",
        "저녁",
        "주말",
        "휴일",
        "여름",
        "겨울",
        "봄",
        "가을",
        "시원한",
        "더운",
        "추운",
        "비",
        "눈",
        "바람",
        "전에",
        "하면서",
        "구워",
        "가기",
        "전에",
        "가서",
        "들러",
        "서",
        "하면",
        "오면",
        "보러",
        "많이",
        "불면",
    }:
        return True
    if label == "time" and normalized in {"실내", "밖", "바다", "한강", "캠핑장", "계곡"}:
        return True
    if label == "activity" and normalized in {"데", "이나", "거", "뭐", "시원한", "좋아", "하면", "하고", "먹", "은", "해도", "하자", "부니까"}:
        return True
    if label == "process" and normalized in {"때", "전에", "거", "뭐", "해", "야", "말해", "봐", "하면", "좋아", "알려", "줘", "면", "되지", "오는", "까", "굽", "만한", "있어", "지", "겠지", "해야", "맞아", "좋", "부터", "게", "엔", "전"}:
        return True
    if label == "topic" and normalized in {"뭐", "도착", "전", "전에", "면", "터", "기", "해", "오는", "그냥", "봐도", "랑", "부터", "때", "게", "나아", "보기", "내리는", "하지", "까"}:
        return True
    if label == "comparison" and normalized in {"나", "게", "까", "그냥", "조금", "더", "이번", "주", "할까"}:
        return True
    return False


def save_multihead_meaning_model(
    *,
    output_dir: Path,
    model: MultiHeadMeaningTorchModel,
    tokenizer: Any,
    head_label_maps: dict[str, dict[str, dict[Any, Any]]],
    slot_label_map: dict[str, dict[Any, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = output_dir / ENCODER_DIR
    model.encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(output_dir)
    state: dict[str, torch.Tensor] = {
        f"heads.{key}": value for key, value in model.heads.state_dict().items()
    }
    if model.slot_head is not None:
        state.update({f"slot_head.{key}": value for key, value in model.slot_head.state_dict().items()})
    torch.save(state, output_dir / HEAD_WEIGHTS_FILE)
    payload = {
        "version": 2 if slot_label_map else 1,
        "none_label": NONE_LABEL,
        "heads": head_label_maps,
        "metadata": metadata or {},
    }
    if slot_label_map:
        payload["slot_labels"] = slot_label_map
    (output_dir / HEAD_CONFIG_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
