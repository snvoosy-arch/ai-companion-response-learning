from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from predictive_bot.core.draft_semantic_words import rank_semantic_word
from predictive_bot.core.draft_slot_bank import DraftSlotBank, KeywordChoices


class SlotTextSignals(Protocol):
    raw: str
    compact: str

    def has_any_compact(self, *needles: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class SlotChoiceTrace:
    slot: str
    value: str
    source: str
    matched_keywords: tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class DraftSlotSelection:
    detail: str
    template: str
    slots: dict[str, str]
    trace: tuple[SlotChoiceTrace, ...]

    def render(self) -> str:
        return self.template.format(**self.slots)


class DraftSlotSelector(Protocol):
    def select(
        self,
        signals: SlotTextSignals,
        *,
        detail: str,
        slot_bank: DraftSlotBank,
    ) -> DraftSlotSelection: ...


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(text or "")).lower()


def _stable_variant(source: str, variants: tuple[str, ...], *, salt: str = "") -> str:
    if not variants:
        return ""
    normalized = _normalize(f"{salt}:{source}")
    index = sum(ord(char) for char in normalized) % len(variants)
    return variants[index]


def _matched_keywords(signals: SlotTextSignals, needles: tuple[str, ...]) -> tuple[str, ...]:
    compact = signals.compact
    return tuple(needle for needle in needles if _normalize(needle) in compact)


def _choose_keyword_slot(
    signals: SlotTextSignals,
    choices: KeywordChoices,
    default: str,
) -> tuple[str, tuple[str, ...], float]:
    for needles, value in choices:
        if signals.has_any_compact(*needles):
            matched = _matched_keywords(signals, needles) or needles[:1]
            return value, matched, 0.9
    return default, (), 0.55


class RuleDraftSlotSelector:
    """Deterministic selector used until the learned slot BERT is ready."""

    def select(
        self,
        signals: SlotTextSignals,
        *,
        detail: str,
        slot_bank: DraftSlotBank,
    ) -> DraftSlotSelection:
        slots: dict[str, str] = dict(slot_bank["fixed_slots"])
        trace: list[SlotChoiceTrace] = [
            SlotChoiceTrace(slot=name, value=value, source="fixed")
            for name, value in slot_bank["fixed_slots"].items()
        ]

        for name, (choices, default) in slot_bank["keyword_slots"].items():
            value, matched, confidence = _choose_keyword_slot(signals, choices, default)
            slots[name] = value
            trace.append(
                SlotChoiceTrace(
                    slot=name,
                    value=value,
                    source="keyword" if matched else "keyword_default",
                    matched_keywords=matched,
                    confidence=confidence,
                )
            )

        for name, (desired_tags, default) in slot_bank.get("semantic_slots", {}).items():
            ranked = rank_semantic_word(
                signals,
                desired_tags=desired_tags,
                default=default,
                context=getattr(signals, "sense_context", None),
            )
            slots[name] = ranked.value
            trace.append(
                SlotChoiceTrace(
                    slot=name,
                    value=ranked.value,
                    source=ranked.source,
                    matched_keywords=(*ranked.matched_tags, *ranked.matched_aliases),
                    confidence=min(0.98, 0.45 + ranked.score / 20),
                )
            )

        for name, variants in slot_bank["variant_slots"].items():
            value = _stable_variant(signals.raw, variants, salt=f"{detail}:{name}")
            slots[name] = value
            trace.append(
                SlotChoiceTrace(
                    slot=name,
                    value=value,
                    source="stable_variant",
                    confidence=0.7,
                )
            )

        template = _stable_variant(signals.raw, slot_bank["templates"], salt=f"{detail}:template")
        trace.append(
            SlotChoiceTrace(
                slot="$template",
                value=template,
                source="stable_template",
                confidence=0.7,
            )
        )
        return DraftSlotSelection(
            detail=detail,
            template=template,
            slots=slots,
            trace=tuple(trace),
        )


class BertDraftSlotSelector:
    """Future adapter for the second BERT that predicts template/slot choices."""

    def __init__(self, fallback: DraftSlotSelector | None = None) -> None:
        self._fallback = fallback or RuleDraftSlotSelector()

    def select(
        self,
        signals: SlotTextSignals,
        *,
        detail: str,
        slot_bank: DraftSlotBank,
    ) -> DraftSlotSelection:
        # The learned selector will return the same DraftSlotSelection shape.
        return self._fallback.select(signals, detail=detail, slot_bank=slot_bank)
