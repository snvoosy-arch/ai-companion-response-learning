from __future__ import annotations

import re
from typing import Iterable


def render_policy_feature_text(
    *,
    input_text: str,
    input_intent: str,
    input_speech_act: str | None,
    input_topic_hint: str | None,
    response_needs: Iterable[str] | None,
    input_sentiment: str,
    conversation_mode: str | None,
    user_emotion: str | None,
    risk_level: str | None,
    unresolved_need: str | None,
    factuality_required: bool | None,
    turn_count_bucket: str | None,
    tension_bucket: str | None,
    rapport_bucket: str | None,
    boundary_history: str | None,
    user_directness_style: str | None,
    last_intent_hint: str | None,
    last_action_hint: str | None,
    constraints: Iterable[str] | None = None,
    evidence: Iterable[str] | None = None,
) -> str:
    parts = [
        f"input={input_text}",
        f"intent={input_intent}",
        f"speech_act={input_speech_act or 'unknown'}",
        f"topic_hint={input_topic_hint or 'none'}",
        f"sentiment={input_sentiment}",
        f"question={_bool_token('?' in input_text)}",
        f"mode={conversation_mode or 'unknown'}",
        f"emotion={user_emotion or 'unknown'}",
        f"risk={risk_level or 'unknown'}",
        f"unresolved={unresolved_need or 'none'}",
        f"factuality={_bool_token(bool(factuality_required))}",
        f"turn_bucket={turn_count_bucket or 'unknown'}",
        f"tension_bucket={tension_bucket or 'unknown'}",
        f"rapport_bucket={rapport_bucket or 'unknown'}",
        f"boundary_history={boundary_history or 'unknown'}",
        f"directness_style={user_directness_style or 'unknown'}",
        f"last_intent={last_intent_hint or 'none'}",
        f"last_action={last_action_hint or 'none'}",
    ]

    cleaned_constraints = [item for item in (constraints or []) if item]
    cleaned_evidence = [item for item in (evidence or []) if item]
    cleaned_response_needs = [item for item in (response_needs or []) if item]

    if cleaned_response_needs:
        parts.append("response_needs=" + ",".join(sorted(cleaned_response_needs)))
    else:
        parts.append("response_needs=none")

    if cleaned_constraints:
        parts.append("constraints=" + ",".join(sorted(cleaned_constraints)))
    else:
        parts.append("constraints=none")

    if cleaned_evidence:
        compact_evidence = [_compact_evidence(item) for item in cleaned_evidence[:4]]
        parts.append("evidence=" + " || ".join(compact_evidence))
    else:
        parts.append("evidence=none")

    return " | ".join(parts)


def build_group_key(*, input_text: str, input_intent: str, selected_action: str) -> str:
    return f"{selected_action}::{input_intent}::{input_text.strip().lower()}"


def is_probably_clean_user_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    # Earlier Windows console runs left rows like "??" and mojibake behind.
    # We keep the filter permissive for real chat, but reject obviously broken rows.
    if "\ufffd" in stripped:
        return False
    if stripped.count("?") >= max(2, len(stripped) // 2):
        return False
    if re.search(r"[ÃÂÐØÞÆ]{2,}", stripped):
        return False

    allowed_chars = sum(1 for ch in stripped if _is_allowed_chat_char(ch))
    ratio = allowed_chars / max(1, len(stripped))
    has_signal = any(_is_signal_char(ch) for ch in stripped)
    return has_signal and ratio >= 0.7


def _bool_token(value: bool) -> str:
    return "yes" if value else "no"


def _compact_evidence(item: str) -> str:
    item = item.strip()
    if not item:
        return "none"
    return item.replace(" ", "_")


def _is_allowed_chat_char(ch: str) -> bool:
    if ch.isascii():
        return ch.isalnum() or ch.isspace() or ch in "?!.,~`'\"():/_-;[]"

    codepoint = ord(ch)
    return (
        0xAC00 <= codepoint <= 0xD7A3
        or 0x3131 <= codepoint <= 0x318E
        or 0x1100 <= codepoint <= 0x11FF
        or ch in "ㅋㅋㅎㅎㅠㅜ"
    )


def _is_signal_char(ch: str) -> bool:
    codepoint = ord(ch)
    return ch.isalnum() or (0xAC00 <= codepoint <= 0xD7A3) or (0x3131 <= codepoint <= 0x318E)
