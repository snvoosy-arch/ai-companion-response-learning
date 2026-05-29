from __future__ import annotations

from dataclasses import dataclass

from .memory_store import ConversationSummary, DurableMemory, StoredMessage

_NEGATIVE_MARKERS = (
    "힘들",
    "불안",
    "우울",
    "무겁",
    "답답",
    "지치",
    "피곤",
    "걱정",
    "속상",
    "허탈",
    "씁쓸",
)
_POSITIVE_MARKERS = (
    "좋",
    "괜찮",
    "나아",
    "편안",
    "기쁘",
    "설레",
    "기대",
    "재밌",
    "즐거",
    "반갑",
)
_RECOVERY_MARKERS = (
    "쉬",
    "잠",
    "회복",
    "정리",
    "괜찮아졌",
    "나아졌",
    "천천히",
)


@dataclass(slots=True)
class WhiteSignalSnapshot:
    tone: str
    negative_hits: int
    positive_hits: int
    recovery_hits: int
    note: str


def build_white_summary_report(
    *,
    summary: ConversationSummary | None,
    memories: list[DurableMemory],
) -> str:
    if summary is None and not memories:
        return "아직 정리된 장기 요약이 없어."

    lines = ["white summary"]
    if summary is not None:
        lines.extend(
            [
                f"- updated_at: {summary.updated_at}",
                f"- summary: {_clip(summary.summary_text, 280)}",
            ]
        )
    else:
        lines.append("- summary: 없음")
    if memories:
        lines.append("- memory:")
        for memory in memories[:3]:
            lines.append(f"  • [{memory.memory_kind}] {_clip(memory.memory_text, 80)}")
    return "\n".join(lines)


def build_white_recall_report(
    *,
    query: str | None,
    summary: ConversationSummary | None,
    memories: list[DurableMemory],
) -> str:
    label = query.strip() if query else "최근 이야기"
    lines = [f"white recall · {label}"]
    if memories:
        for memory in memories[:4]:
            matched = ", ".join(memory.matched_terms[:4]) if memory.matched_terms else "recent-match"
            lines.append(f"- [{memory.memory_kind}] {_clip(memory.memory_text, 90)} ({matched})")
    else:
        lines.append("- 관련 장기기억은 아직 뚜렷하게 안 잡혀.")
    if summary is not None:
        lines.append(f"- summary_hint: {_clip(summary.summary_text, 140)}")
    return "\n".join(lines)


def build_white_dashboard_report(
    *,
    model_name: str,
    web_search_mode: str,
    tts_enabled: bool,
    tts_mode: str,
    runtime_report: str,
    message_count: int,
    user_memory_counts: dict[str, int],
    channel_memory_counts: dict[str, int],
    summary: ConversationSummary | None,
    signals: WhiteSignalSnapshot,
) -> str:
    summary_updated = summary.updated_at if summary is not None else "none"
    return "\n".join(
        [
            "white dashboard",
            f"- model: {model_name}",
            f"- web_search_mode: {web_search_mode}",
            f"- tts: {'on' if tts_enabled else 'off'} ({tts_mode})",
            f"- channel_messages: {message_count}",
            f"- user_memories: {user_memory_counts.get('total', 0)}",
            f"- channel_memories: {channel_memory_counts.get('total', 0)}",
            f"- latest_summary_updated: {summary_updated}",
            f"- recent_tone: {signals.tone}",
            f"- signal_note: {signals.note}",
            f"- runtime: {_clip(runtime_report.replace(chr(10), ' | '), 220)}",
        ]
    )


def build_white_signal_report(
    *,
    recent_messages: list[StoredMessage],
    memories: list[DurableMemory],
) -> str:
    snapshot = analyze_recent_signals(recent_messages)
    lines = [
        "white signals",
        f"- tone: {snapshot.tone}",
        f"- negative_hits: {snapshot.negative_hits}",
        f"- positive_hits: {snapshot.positive_hits}",
        f"- recovery_hits: {snapshot.recovery_hits}",
        f"- note: {snapshot.note}",
    ]
    if memories:
        lines.append("- memory_focus:")
        for memory in memories[:3]:
            lines.append(f"  • [{memory.memory_kind}] {_clip(memory.memory_text, 90)}")
    return "\n".join(lines)


def analyze_recent_signals(messages: list[StoredMessage]) -> WhiteSignalSnapshot:
    negative_hits = 0
    positive_hits = 0
    recovery_hits = 0
    for message in messages:
        lowered = message.content.casefold()
        negative_hits += sum(1 for marker in _NEGATIVE_MARKERS if marker in lowered)
        positive_hits += sum(1 for marker in _POSITIVE_MARKERS if marker in lowered)
        recovery_hits += sum(1 for marker in _RECOVERY_MARKERS if marker in lowered)
    if negative_hits > positive_hits + 1:
        tone = "heavy"
        note = "최근 대화가 무겁거나 지친 쪽으로 기울어 있어."
    elif positive_hits > negative_hits + 1:
        tone = "lighter"
        note = "최근 대화는 비교적 가벼워졌거나 회복 신호가 있어."
    elif recovery_hits >= 2:
        tone = "recovering"
        note = "정리하거나 쉬려는 회복 신호가 보였어."
    else:
        tone = "steady"
        note = "최근 대화 톤이 크게 한쪽으로 쏠리진 않았어."
    return WhiteSignalSnapshot(
        tone=tone,
        negative_hits=negative_hits,
        positive_hits=positive_hits,
        recovery_hits=recovery_hits,
        note=note,
    )


def _clip(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."
