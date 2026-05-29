from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IntentPrediction:
    intent: str
    confidence: float
    scores: dict[str, float]


class CharNgramCentroidModel:
    def __init__(
        self,
        *,
        idf: dict[str, float],
        examples: list[dict[str, object]],
        min_n: int = 2,
        max_n: int = 4,
        top_features_per_intent: int = 2500,
        top_k: int = 5,
    ) -> None:
        self.idf = idf
        self.examples = examples
        self.min_n = min_n
        self.max_n = max_n
        self.top_features_per_intent = top_features_per_intent
        self.top_k = top_k

    @classmethod
    def train(
        cls,
        rows: list[dict[str, str]],
        *,
        min_n: int = 2,
        max_n: int = 4,
        top_features_per_intent: int = 2500,
        top_k: int = 5,
    ) -> "CharNgramCentroidModel":
        document_counts: list[tuple[str, str, Counter[str]]] = []
        document_frequency: Counter[str] = Counter()

        for row in rows:
            counts = cls._extract_features(row["text"], min_n=min_n, max_n=max_n)
            if not counts:
                continue
            document_counts.append((row["text"], row["intent"], counts))
            document_frequency.update(counts.keys())

        total_docs = len(document_counts)
        if total_docs == 0:
            raise ValueError("학습할 문장이 없습니다.")

        idf = {
            feature: math.log((1 + total_docs) / (1 + df)) + 1.0
            for feature, df in document_frequency.items()
        }

        examples: list[dict[str, object]] = []

        for text, intent, counts in document_counts:
            vector = cls._to_tfidf_vector(counts, idf)
            if not vector:
                continue
            examples.append(
                {
                    "text": text,
                    "intent": intent,
                    "vector": cls._keep_top_features(vector, top_features_per_intent),
                }
            )

        return cls(
            idf=idf,
            examples=examples,
            min_n=min_n,
            max_n=max_n,
            top_features_per_intent=top_features_per_intent,
            top_k=top_k,
        )

    def predict(self, text: str) -> IntentPrediction:
        counts = self._extract_features(text, min_n=self.min_n, max_n=self.max_n)
        vector = self._to_tfidf_vector(counts, self.idf)
        if not vector:
            return IntentPrediction(intent="unknown", confidence=0.0, scores={})

        neighbors: list[tuple[float, dict[str, object]]] = []
        for example in self.examples:
            similarity = self._cosine_similarity(vector, example["vector"])
            if similarity > 0:
                neighbors.append((similarity, example))

        if not neighbors:
            return IntentPrediction(intent="unknown", confidence=0.0, scores={})

        neighbors.sort(key=lambda item: item[0], reverse=True)
        top_neighbors = neighbors[: self.top_k]

        scores: dict[str, float] = defaultdict(float)
        best_similarity_by_intent: dict[str, float] = defaultdict(float)
        for similarity, example in top_neighbors:
            intent = str(example["intent"])
            scores[intent] += similarity
            if similarity > best_similarity_by_intent[intent]:
                best_similarity_by_intent[intent] = similarity

        best_intent, _ = max(scores.items(), key=lambda item: item[1])
        best_score = best_similarity_by_intent[best_intent]
        return IntentPrediction(
            intent=best_intent,
            confidence=max(0.0, best_score),
            scores=dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
        )

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "top_features_per_intent": self.top_features_per_intent,
            "top_k": self.top_k,
            "idf": self.idf,
            "examples": self.examples,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharNgramCentroidModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            idf={key: float(value) for key, value in payload["idf"].items()},
            examples=[
                {
                    "text": example["text"],
                    "intent": example["intent"],
                    "vector": {
                        feature: float(weight)
                        for feature, weight in example["vector"].items()
                    },
                }
                for example in payload["examples"]
            ],
            min_n=int(payload.get("min_n", 2)),
            max_n=int(payload.get("max_n", 4)),
            top_features_per_intent=int(payload.get("top_features_per_intent", 2500)),
            top_k=int(payload.get("top_k", 5)),
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @classmethod
    def _extract_features(
        cls,
        text: str,
        *,
        min_n: int,
        max_n: int,
    ) -> Counter[str]:
        normalized = cls._normalize_text(text)
        compact = normalized.replace(" ", "")
        features: Counter[str] = Counter()

        if not compact:
            return features

        for token in normalized.split():
            features[f"w:{token}"] += 1

        features[f"e:{normalized}"] += 1

        for n in range(min_n, max_n + 1):
            if len(compact) < n:
                continue
            for index in range(len(compact) - n + 1):
                features[f"c{n}:{compact[index:index + n]}"] += 1

        return features

    @staticmethod
    def _to_tfidf_vector(
        counts: Counter[str],
        idf: dict[str, float],
    ) -> dict[str, float]:
        total = sum(counts.values())
        if total <= 0:
            return {}

        weights = {
            feature: (count / total) * idf[feature]
            for feature, count in counts.items()
            if feature in idf
        }
        return CharNgramCentroidModel._normalize_vector(weights)

    @staticmethod
    def _normalize_vector(vector: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        if norm <= 0:
            return {}
        return {feature: weight / norm for feature, weight in vector.items()}

    @staticmethod
    def _keep_top_features(vector: dict[str, float], limit: int) -> dict[str, float]:
        if limit <= 0 or len(vector) <= limit:
            return vector
        items = sorted(vector.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
        return dict(items)

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(feature, 0.0) for feature, weight in left.items())
