from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_ANONYMIZATION_SALT = "predictive-runtime-soak-v1"
DEFAULT_SESSION_GAP_MINUTES = 24 * 60
DEFAULT_MAX_TURNS_PER_SESSION = 10
DEFAULT_MIN_TURNS_PER_SESSION = 1

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)")
_DISCORD_MENTION_RE = re.compile(r"<@!?\d+>|<#\d+>|<@&\d+>")
_LONG_ID_RE = re.compile(r"\b\d{17,20}\b")


@dataclass(slots=True)
class RuntimeLogExportStats:
    source_db: str
    source_table: str
    loaded_rows: int
    exported_sessions: int
    exported_turns: int
    dropped_rows: int
    distinct_source_users: int
    anonymized_users: int
    single_turn_sessions: int
    max_session_turns: int
    category_counts: dict[str, int]
    action_counts: dict[str, int]
    redaction_counts: dict[str, int]


def load_decision_trace_rows(
    db_path: Path,
    *,
    table_name: str = "decision_trace",
) -> list[dict[str, Any]]:
    if table_name != "decision_trace":
        raise ValueError("runtime soak export currently supports decision_trace only")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows: list[dict[str, Any]] = []
    try:
        cursor = conn.execute(
            """
            SELECT
              id,
              user_id,
              input_text,
              input_intent,
              selected_action,
              selected_reason,
              decision_module,
              explanation_mode,
              reason_trace_json,
              evidence_json,
              constraints_json,
              world_state_snapshot_json,
              counterfactuals_json,
              logic_chain_json,
              verification_issues_json,
              output_text,
              created_at
            FROM decision_trace
            ORDER BY user_id ASC, created_at ASC, id ASC
            """
        )
        for row in cursor:
            snapshot = json.loads(row["world_state_snapshot_json"] or "{}")
            rows.append(
                {
                    "row_id": int(row["id"]),
                    "user_id": str(row["user_id"]),
                    "created_at": str(row["created_at"]),
                    "input_text": str(row["input_text"] or ""),
                    "input_intent": str(row["input_intent"] or "unknown"),
                    "selected_action": str(row["selected_action"] or "continue_conversation"),
                    "selected_reason": str(row["selected_reason"] or ""),
                    "decision_module": str(row["decision_module"] or "daily_chat"),
                    "explanation_mode": str(row["explanation_mode"] or "on_request_only"),
                    "reason_codes": [
                        str(item.get("code", ""))
                        for item in json.loads(row["reason_trace_json"] or "[]")
                        if isinstance(item, dict) and item.get("code")
                    ],
                    "logic_rule_ids": [
                        str(item.get("rule_id", ""))
                        for item in json.loads(row["logic_chain_json"] or "[]")
                        if isinstance(item, dict) and item.get("rule_id")
                    ],
                    "counterfactual_actions": [
                        str(item.get("predicted_action", ""))
                        for item in json.loads(row["counterfactuals_json"] or "[]")
                        if isinstance(item, dict) and item.get("predicted_action")
                    ],
                    "constraints": [
                        str(item)
                        for item in json.loads(row["constraints_json"] or "[]")
                        if item is not None
                    ],
                    "evidence": [
                        str(item)
                        for item in json.loads(row["evidence_json"] or "[]")
                        if item is not None
                    ],
                    "verification_issues": [
                        str(item)
                        for item in json.loads(row["verification_issues_json"] or "[]")
                        if item is not None
                    ],
                    "output_text": str(row["output_text"] or ""),
                    "snapshot": snapshot,
                }
            )
    finally:
        conn.close()
    return rows


def build_runtime_soak_sessions(
    rows: list[dict[str, Any]],
    *,
    source_db: str = "",
    anonymization_salt: str = DEFAULT_ANONYMIZATION_SALT,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
    max_turns_per_session: int = DEFAULT_MAX_TURNS_PER_SESSION,
    min_turns_per_session: int = DEFAULT_MIN_TURNS_PER_SESSION,
    redact_text: bool = True,
) -> tuple[list[dict[str, Any]], RuntimeLogExportStats]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["user_id"])].append(row)

    sessions: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    redaction_counts: Counter[str] = Counter()
    exported_turns = 0
    single_turn_sessions = 0
    max_session_turns = 0
    dropped_rows = 0

    for source_user_id, user_rows in sorted(grouped_rows.items(), key=lambda item: item[0]):
        user_alias = _anon_id(source_user_id, anonymization_salt)
        segments = _segment_user_rows(
            user_rows,
            session_gap_minutes=session_gap_minutes,
            max_turns_per_session=max_turns_per_session,
        )
        for segment_index, segment in enumerate(segments, start=1):
            if len(segment) < min_turns_per_session:
                dropped_rows += len(segment)
                continue

            turns: list[dict[str, Any]] = []
            action_list: list[str] = []
            created_values = [str(item["created_at"]) for item in segment]
            session_start = created_values[0]
            session_end = created_values[-1]

            for turn_index, row in enumerate(segment, start=1):
                action = str(row["selected_action"])
                action_list.append(action)
                action_counts[action] += 1

                input_text = str(row["input_text"])
                if redact_text:
                    redacted_text, redactions = redact_runtime_text(input_text)
                    input_text = redacted_text
                    for key, value in redactions.items():
                        redaction_counts[key] += value

                snapshot = dict(row.get("snapshot") or {})
                expect: dict[str, Any] = {
                    "intent": row["input_intent"],
                    "action": action,
                    "speech_act": snapshot.get("speech_act"),
                    "topic_hint": snapshot.get("topic_hint"),
                    "news_topic": snapshot.get("news_topic"),
                    "conversation_mode": snapshot.get("conversation_mode"),
                    "unresolved_need": snapshot.get("unresolved_need"),
                    "risk_level": snapshot.get("risk_level"),
                    "boundary_history": snapshot.get("boundary_history"),
                    "user_directness_style": snapshot.get("user_directness_style"),
                    "rapport_bucket": snapshot.get("rapport_bucket"),
                    "decision_module": row["decision_module"],
                    "explanation_mode": row["explanation_mode"],
                }

                response_needs = _normalize_sequence_field(snapshot.get("response_needs"))
                pragmatic_cues = _normalize_sequence_field(snapshot.get("pragmatic_cues"))
                constraints = _normalize_sequence_field(row.get("constraints"))
                counterfactual_actions = _normalize_sequence_field(row.get("counterfactual_actions"))
                reason_codes = _normalize_sequence_field(row.get("reason_codes"))
                logic_rule_ids = _normalize_sequence_field(row.get("logic_rule_ids"))

                if response_needs:
                    expect["response_needs"] = response_needs
                if pragmatic_cues:
                    expect["pragmatic_cues"] = pragmatic_cues
                if constraints:
                    expect["constraints"] = constraints
                if counterfactual_actions:
                    expect["counterfactual_actions"] = counterfactual_actions[:4]
                if reason_codes:
                    expect["reason_code_prefixes"] = reason_codes[:6]
                if logic_rule_ids:
                    expect["logic_rule_prefixes"] = logic_rule_ids[:6]

                turns.append(
                    {
                        "input": input_text,
                        "expect": expect,
                        "meta": {
                            "source_row_id": row["row_id"],
                            "source_user_alias": user_alias,
                            "source_created_at": row["created_at"],
                            "selected_reason": row["selected_reason"],
                            "verification_issues": list(row.get("verification_issues", [])),
                        },
                    }
                )

            if len(turns) < min_turns_per_session:
                continue

            session_turn_count = len(turns)
            max_session_turns = max(max_session_turns, session_turn_count)
            if session_turn_count == 1:
                single_turn_sessions += 1

            category = _classify_session_category(action_list)
            category_counts[category] += 1
            exported_turns += session_turn_count
            session_id = f"{user_alias}_s{segment_index:02d}"
            sessions.append(
                {
                    "session_id": session_id,
                    "user_id": session_id,
                    "source_user_alias": user_alias,
                    "category": category,
                    "description": (
                        f"Real-log soak session exported from decision_trace "
                        f"({session_turn_count} turns, {session_start} -> {session_end})"
                    ),
                    "source": {
                        "db": source_db,
                        "table": "decision_trace",
                        "anonymized": True,
                        "gap_minutes": session_gap_minutes,
                        "max_turns_per_session": max_turns_per_session,
                        "min_turns_per_session": min_turns_per_session,
                        "turn_count": session_turn_count,
                        "start_at": session_start,
                        "end_at": session_end,
                    },
                    "turns": turns,
                }
            )

    stats = RuntimeLogExportStats(
        source_db=source_db,
        source_table="decision_trace",
        loaded_rows=len(rows),
        exported_sessions=len(sessions),
        exported_turns=exported_turns,
        dropped_rows=dropped_rows,
        distinct_source_users=len(grouped_rows),
        anonymized_users=len(grouped_rows),
        single_turn_sessions=single_turn_sessions,
        max_session_turns=max_session_turns,
        category_counts=dict(sorted(category_counts.items())),
        action_counts=dict(sorted(action_counts.items())),
        redaction_counts=dict(sorted(redaction_counts.items())),
    )
    return sessions, stats


def write_sessions_jsonl(path: Path, sessions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session, ensure_ascii=False) + "\n")


def build_runtime_soak_summary(
    sessions: list[dict[str, Any]],
    stats: RuntimeLogExportStats,
) -> dict[str, Any]:
    turn_counts = [len(session.get("turns", [])) for session in sessions]
    return {
        "source_db": stats.source_db,
        "source_table": stats.source_table,
        "loaded_rows": stats.loaded_rows,
        "exported_sessions": stats.exported_sessions,
        "exported_turns": stats.exported_turns,
        "dropped_rows": stats.dropped_rows,
        "distinct_source_users": stats.distinct_source_users,
        "anonymized_users": stats.anonymized_users,
        "single_turn_sessions": stats.single_turn_sessions,
        "max_session_turns": stats.max_session_turns,
        "min_session_turns": min(turn_counts) if turn_counts else 0,
        "avg_session_turns": round(sum(turn_counts) / max(1, len(turn_counts)), 2),
        "category_counts": stats.category_counts,
        "action_counts": stats.action_counts,
        "redaction_counts": stats.redaction_counts,
        "sample_sessions": [
            {
                "session_id": session["session_id"],
                "category": session["category"],
                "turn_count": len(session.get("turns", [])),
            }
            for session in sessions[:5]
        ],
    }


def snapshot_to_expectation(snapshot: dict[str, Any]) -> dict[str, Any]:
    expect: dict[str, Any] = {}

    exact_fields = (
        "intent",
        "action",
        "speech_act",
        "topic_hint",
        "news_topic",
        "conversation_mode",
        "unresolved_need",
        "risk_level",
        "boundary_history",
        "user_directness_style",
        "rapport_bucket",
        "decision_module",
        "explanation_mode",
        "verification_ok",
    )
    for field in exact_fields:
        if field in snapshot:
            expect[field] = snapshot.get(field)

    subset_fields = (
        "response_needs",
        "pragmatic_cues",
        "constraints",
        "counterfactual_actions",
    )
    for field in subset_fields:
        values = snapshot.get(field)
        if values is None:
            continue
        expect[field] = [str(item) for item in values if item is not None]

    prefix_fields = (
        ("reason_code_prefixes", "reason_codes"),
        ("logic_rule_prefixes", "logic_rule_ids"),
    )
    for expect_field, actual_field in prefix_fields:
        values = snapshot.get(actual_field)
        if values is None:
            continue
        expect[expect_field] = [str(item) for item in values if item is not None]

    return expect


def redact_runtime_text(text: str) -> tuple[str, dict[str, int]]:
    redactions = {
        "url": 0,
        "email": 0,
        "phone": 0,
        "discord_mention": 0,
        "long_id": 0,
    }

    def _sub(pattern: re.Pattern[str], replacement: str, key: str, value: str) -> str:
        nonlocal redactions
        new_value, count = pattern.subn(replacement, value)
        redactions[key] += count
        return new_value

    redacted = text
    redacted = _sub(_URL_RE, "<url>", "url", redacted)
    redacted = _sub(_EMAIL_RE, "<email>", "email", redacted)
    redacted = _sub(_DISCORD_MENTION_RE, "<mention>", "discord_mention", redacted)
    redacted = _sub(_LONG_ID_RE, "<id>", "long_id", redacted)
    redacted = _sub(_PHONE_RE, "<phone>", "phone", redacted)
    redacted = re.sub(r"[ \t]+", " ", redacted).strip()
    return redacted, redactions


def _segment_user_rows(
    rows: list[dict[str, Any]],
    *,
    session_gap_minutes: int,
    max_turns_per_session: int,
) -> list[list[dict[str, Any]]]:
    sorted_rows = sorted(rows, key=lambda row: (_parse_created_at(str(row["created_at"])), int(row["row_id"])))
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_ts: datetime | None = None

    for row in sorted_rows:
        current_ts = _parse_created_at(str(row["created_at"]))
        if current:
            gap = current_ts - previous_ts if previous_ts is not None else timedelta()
            if gap > timedelta(minutes=session_gap_minutes) or len(current) >= max_turns_per_session:
                segments.append(current)
                current = []
        current.append(row)
        previous_ts = current_ts

    if current:
        segments.append(current)
    return segments


def _parse_created_at(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _anon_id(value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return f"anon_{digest[:10]}"


def _normalize_sequence_field(values: Any) -> list[str]:
    filtered = []
    if values is None:
        return filtered
    if isinstance(values, str):
        text = values.strip()
        if not text or text.lower() in {"none", "null", "[]"}:
            return filtered
        if "," in text:
            return [item.strip() for item in text.split(",") if item.strip()]
        return [text]
    if not isinstance(values, (list, tuple, set)):
        return [str(values).strip()] if str(values).strip() else filtered
    for value in values:
        if value is None:
            continue
        item = str(value).strip()
        if item:
            filtered.append(item)
    return filtered


def _classify_session_category(actions: list[str]) -> str:
    if not actions:
        return "uncategorized"

    category_counter: Counter[str] = Counter()
    for action in actions:
        category_counter[_action_category(action)] += 1

    category, count = category_counter.most_common(1)[0]
    if len(category_counter) == 1:
        return category
    if count >= max(2, len(actions) // 2 + 1):
        return category
    return "mixed"


def _action_category(action: str) -> str:
    if action in {"ask_location", "weather_lookup", "tell_time", "search_answer", "news_answer"}:
        return "knowledge"
    if action in {"recommend", "music_chat"}:
        return "media"
    if action == "explain_reason":
        return "explanation"
    if action in {"deescalate"}:
        return "safety"
    if action in {
        "share_feeling",
        "share_opinion",
        "small_talk",
        "acknowledge",
        "continue_conversation",
        "react_laugh",
        "react_surprise",
        "tease_back",
    }:
        return "social"
    if action in {"game_chat", "game_accept_or_decline"}:
        return "game"
    return "mixed"
