from __future__ import annotations

from predictive_bot.config import AppConfig
from predictive_bot.core.memory import (
    DurableMemoryEntry,
    render_durable_memory_bucket_summary,
    select_relevant_durable_memory_entries,
)
from predictive_bot.core.models import ConversationState, DecisionTrace


def format_black_dashboard(
    *,
    config: AppConfig,
    runtime_report: str,
    state: ConversationState,
    latest_trace: DecisionTrace | None,
) -> str:
    snapshot = latest_trace.world_state_snapshot if latest_trace is not None else {}
    return "\n".join(
        [
            "black dashboard",
            f"- runtime: {'on' if config.runtime_state_enabled else 'off'}",
            f"- tts: {'on' if config.tts_enabled else 'off'} ({config.tts_mode})",
            f"- kobart_input_mode: {config.kobart_input_mode}",
            f"- turn_count: {state.turn_count}",
            f"- recent_turns: {len(state.recent_turns)}",
            f"- durable_memory: {len(state.durable_memory)}",
            f"- last_intent: {state.last_intent.value if state.last_intent else 'none'}",
            f"- last_action: {state.last_action.value if state.last_action else 'none'}",
            f"- emotion: {snapshot.get('user_emotion', 'unknown')}",
            f"- mode: {snapshot.get('conversation_mode', 'unknown')}",
            f"- tension_bucket: {snapshot.get('tension_bucket', 'unknown')}",
            f"- rapport_bucket: {snapshot.get('rapport_bucket', 'unknown')}",
            f"- runtime_report: {_clip(runtime_report.replace(chr(10), ' | '), 220)}",
        ]
    )


def format_black_summary(
    *,
    state: ConversationState,
    latest_trace: DecisionTrace | None,
) -> str:
    snapshot = latest_trace.world_state_snapshot if latest_trace is not None else {}
    lines = [
        "black summary",
        f"- emotion: {snapshot.get('user_emotion', 'unknown')}",
        f"- mode: {snapshot.get('conversation_mode', 'unknown')}",
        f"- memory_summary: {_clip(str(snapshot.get('memory_summary', 'no_recent_memory')), 180)}",
        f"- durable_memory: {render_durable_memory_bucket_summary(state.durable_memory[:6])}",
    ]
    if state.recent_turns:
        lines.append("- recent_turns:")
        for turn in state.recent_turns[-3:]:
            lines.append(f"  • user={_clip(turn.user_text, 42)} | action={turn.action.value}")
    return "\n".join(lines)


def format_black_recall(
    *,
    state: ConversationState,
    query: str,
) -> str:
    memories = select_relevant_durable_memory_entries(
        state.durable_memory,
        query=query,
        limit=4,
        current_turn=state.turn_count,
    )
    lines = [f"black recall · {query}"]
    if not memories:
        lines.append("- 관련 durable memory가 아직 뚜렷하게 없었어.")
        return "\n".join(lines)
    for entry in memories:
        lines.append(f"- [{entry.bucket.value}] {_clip(entry.text, 96)}")
    return "\n".join(lines)


def format_black_state(
    *,
    state: ConversationState,
    latest_trace: DecisionTrace | None,
) -> str:
    snapshot = latest_trace.world_state_snapshot if latest_trace is not None else {}
    inference = latest_trace.state_inference_trace[:4] if latest_trace is not None else []
    lines = [
        "black state",
        f"- user_emotion: {snapshot.get('user_emotion', 'unknown')}",
        f"- conversation_mode: {snapshot.get('conversation_mode', 'unknown')}",
        f"- tension_bucket: {snapshot.get('tension_bucket', 'unknown')}",
        f"- rapport_bucket: {snapshot.get('rapport_bucket', 'unknown')}",
        f"- boundary_history: {snapshot.get('boundary_history', 'unknown')}",
        f"- directness: {snapshot.get('user_directness_style', 'unknown')}",
        f"- durable_memory: {render_durable_memory_bucket_summary(state.durable_memory[:6])}",
    ]
    relevant_open_loops = snapshot.get("relevant_open_loops")
    relevant_stress = snapshot.get("relevant_stress_signals")
    if relevant_open_loops:
        lines.append(f"- open_loops: {_clip(str(relevant_open_loops), 120)}")
    if relevant_stress:
        lines.append(f"- stress_signals: {_clip(str(relevant_stress), 120)}")
    if inference:
        lines.append("- inference:")
        for item in inference:
            reason = item.reasons[0] if item.reasons else "no_reason"
            lines.append(f"  • {item.field}={item.value} | {_clip(reason, 88)}")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."
