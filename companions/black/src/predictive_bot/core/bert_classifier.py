"""KcBERT 기반 Intent 분류기 (CPU 추론)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(slots=True)
class IntentPrediction:
    intent: str
    confidence: float
    scores: dict[str, float]


class KcBertIntentClassifier:
    """파인튜닝된 KcBERT 모델로 의도를 분류합니다."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        device: str = "auto",
    ) -> None:
        self._model_dir = Path(model_dir)
        self._device = torch.device(self._resolve_device(device))

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self._model_dir),
        ).to(self._device)
        self._model.eval()

        label_map_path = self._model_dir / "label_map.json"
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        self._id2label: dict[int, str] = {
            int(k): v for k, v in label_map["id2label"].items()
        }

    @torch.no_grad()
    def predict(self, text: str) -> IntentPrediction:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self._device)

        outputs = self._model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())
        intent = self._id2label.get(top_idx, "unknown")

        scores = {
            self._id2label[i]: float(probs[i].item())
            for i in range(len(probs))
            if float(probs[i].item()) > 0.01
        }
        scores = dict(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)
        )

        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            scores=scores,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return normalized
