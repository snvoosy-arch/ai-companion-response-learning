from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Protocol

from predictive_bot.core.models import (
    ActionType,
    CharacterState,
    ClassifierEvidence,
    ConversationState,
    Counterfactual,
    DecisionTrace,
    DecisionModule,
    ExplanationMode,
    Intent,
    LogicalStep,
    PolicyCandidate,
    ReasonTraceEntry,
    ResponsePlan,
    ScoredLabel,
    StateInferenceEntry,
    TurnRecord,
)
from predictive_bot.core.memory import (
    DurableMemoryBucket,
    DurableMemoryEntry,
    classify_durable_memory_bucket,
    coerce_durable_memory_entry,
    normalize_durable_memory_entries,
)

SQLITE_TIMEOUT_SECONDS = 10
SQLITE_BUSY_TIMEOUT_MS = 10_000
_PERSISTENCE_SENSITIVE_PATTERNS = (
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[redacted:email]"),
    (re.compile(r"\b01[0-9][\s-]?\d{3,4}[\s-]?\d{4}\b"), "[redacted:phone]"),
    (re.compile(r"(?:api|access|refresh)[\s_-]*key\s*[:=]?\s*\S+", re.IGNORECASE), "[redacted:key]"),
    (re.compile(r"(?:token|secret|password|passwd)\s*[:=]?\s*\S+", re.IGNORECASE), "[redacted:secret]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE), "Bearer [redacted]"),
)


def _json_loads_or_default(raw: str | None, default):
    try:
        return json.loads(raw) if raw is not None else default
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _coerce_enum(enum_cls, value, default=None):
    if value in (None, ""):
        return default
    try:
        return enum_cls(value)
    except ValueError:
        return default


def _safe_convert_items(payload, converter):
    if not isinstance(payload, list):
        return []
    converted = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            converted.append(converter(item))
        except (KeyError, TypeError, ValueError):
            continue
    return converted


def _redact_persisted_text(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = value
    for pattern, replacement in _PERSISTENCE_SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _scrub_payload_for_persistence(value):
    if isinstance(value, str):
        return _redact_persisted_text(value)
    if isinstance(value, list):
        return [_scrub_payload_for_persistence(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _scrub_payload_for_persistence(item)
            for key, item in value.items()
        }
    return value


class StateStore(Protocol):
    def get_or_create(self, user_id: str) -> ConversationState: ...

    def append_turn(self, user_id: str, turn: TurnRecord) -> None: ...

    def save_decision_trace(self, trace: DecisionTrace) -> None: ...

    def get_latest_decision_trace(self, user_id: str) -> DecisionTrace | None: ...

    def close(self) -> None: ...


class MemoryStateStore:
    def __init__(self, max_recent_turns: int = 6) -> None:
        self._states: dict[str, ConversationState] = {}
        self._decision_traces: dict[str, list[DecisionTrace]] = {}
        self._max_recent_turns = max_recent_turns

    def get_or_create(self, user_id: str) -> ConversationState:
        state = self._states.get(user_id)
        if state is None:
            state = ConversationState(user_id=user_id)
            self._states[user_id] = state
        return state

    def append_turn(self, user_id: str, turn: TurnRecord) -> None:
        state = self.get_or_create(user_id)
        state.recent_turns.append(turn)
        if len(state.recent_turns) > self._max_recent_turns:
            state.recent_turns = state.recent_turns[-self._max_recent_turns :]

    def save_decision_trace(self, trace: DecisionTrace) -> None:
        state = self.get_or_create(trace.user_id)
        state.last_decision_id = trace.decision_id
        traces = self._decision_traces.setdefault(trace.user_id, [])
        traces.append(trace)
        if len(traces) > self._max_recent_turns:
            self._decision_traces[trace.user_id] = traces[-self._max_recent_turns :]

    def get_latest_decision_trace(self, user_id: str) -> DecisionTrace | None:
        traces = self._decision_traces.get(user_id)
        if not traces:
            return None
        return traces[-1]

    def close(self) -> None:
        self._decision_traces.clear()
        return None


class SQLiteStateStore:
    def __init__(self, db_path: str | Path, max_recent_turns: int = 6) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_recent_turns = max_recent_turns
        self._states: dict[str, ConversationState] = {}
        self._connection: sqlite3.Connection | None = sqlite3.connect(
            self._db_path,
            timeout=SQLITE_TIMEOUT_SECONDS,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._ensure_schema()

    def get_or_create(self, user_id: str) -> ConversationState:
        cached = self._states.get(user_id)
        if cached is not None:
            return cached

        row = self._connection.execute(
            """
            SELECT
              user_id,
              turn_count,
              tension,
              rapport,
              boundary_pressure,
              directness_score,
              last_intent,
              last_action,
              last_decision_id,
              known_location,
              awaiting_slot,
              preference_memory_json,
              durable_memory_json,
              character_state_json,
              recent_turns_json
            FROM conversation_state
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        if row is None:
            state = ConversationState(user_id=user_id)
            self._save_state(state)
        else:
            state = self._row_to_state(row)

        self._states[user_id] = state
        return state

    def append_turn(self, user_id: str, turn: TurnRecord) -> None:
        state = self.get_or_create(user_id)
        state.recent_turns.append(turn)
        if len(state.recent_turns) > self._max_recent_turns:
            state.recent_turns = state.recent_turns[-self._max_recent_turns :]

        self._connection.execute(
            """
            INSERT INTO message_log (
              user_id,
              user_text,
              bot_text,
              action,
              decision_reason
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                _redact_persisted_text(turn.user_text),
                _redact_persisted_text(turn.bot_text),
                turn.action.value,
                _redact_persisted_text(turn.decision_reason),
            ),
        )
        self._save_state(state)

    def save_decision_trace(self, trace: DecisionTrace) -> None:
        state = self.get_or_create(trace.user_id)
        state.last_decision_id = trace.decision_id
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO decision_trace (
                  decision_id,
                  user_id,
                  input_text,
                  input_intent,
                  input_sentiment,
                  selected_action,
                  selected_reason,
                  decision_module,
                  explanation_mode,
                  classifier_evidence_json,
                  reason_trace_json,
                  evidence_json,
                  constraints_json,
                  world_state_snapshot_json,
                  state_inference_trace_json,
                  policy_candidates_json,
                  counterfactuals_json,
                  logic_chain_json,
                  response_plan_json,
                  output_text,
                  llm_used,
                  llm_fallback_reason,
                  verification_issues_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.decision_id,
                    trace.user_id,
                    _redact_persisted_text(trace.input_text),
                    trace.input_intent.value,
                    trace.input_sentiment,
                    trace.selected_action.value,
                    _redact_persisted_text(trace.selected_reason),
                    trace.decision_module.value,
                    trace.explanation_mode.value,
                    json.dumps(_scrub_payload_for_persistence(self._classifier_evidence_to_dict(trace.classifier_evidence)), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence([self._reason_trace_to_dict(item) for item in trace.reason_trace]), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence(trace.evidence), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence(trace.constraints), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence(trace.world_state_snapshot), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence([self._state_inference_to_dict(item) for item in trace.state_inference_trace]), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence([self._policy_candidate_to_dict(item) for item in trace.policy_candidates]), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence([self._counterfactual_to_dict(item) for item in trace.counterfactuals]), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence([self._logical_step_to_dict(item) for item in trace.logic_chain]), ensure_ascii=False),
                    json.dumps(_scrub_payload_for_persistence(self._response_plan_to_dict(trace.response_plan)), ensure_ascii=False),
                    _redact_persisted_text(trace.output_text),
                    1 if trace.llm_used else 0,
                    _redact_persisted_text(trace.llm_fallback_reason),
                    json.dumps(_scrub_payload_for_persistence(trace.verification_issues), ensure_ascii=False),
                ),
            )
        self._save_state(state)

    def get_latest_decision_trace(self, user_id: str) -> DecisionTrace | None:
        row = self._connection.execute(
            """
            SELECT
              decision_id,
              user_id,
              input_text,
              input_intent,
              input_sentiment,
              selected_action,
              selected_reason,
              decision_module,
              explanation_mode,
              classifier_evidence_json,
              reason_trace_json,
              evidence_json,
              constraints_json,
              world_state_snapshot_json,
              state_inference_trace_json,
              policy_candidates_json,
              counterfactuals_json,
              logic_chain_json,
              response_plan_json,
              output_text,
              llm_used,
              llm_fallback_reason,
              verification_issues_json
            FROM decision_trace
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_decision_trace(row)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._states.clear()

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_state (
                  user_id TEXT PRIMARY KEY,
                  turn_count INTEGER NOT NULL,
                  tension REAL NOT NULL,
                  rapport REAL NOT NULL DEFAULT 0.5,
                  boundary_pressure REAL NOT NULL DEFAULT 0.0,
                  directness_score REAL NOT NULL DEFAULT 0.5,
                  last_intent TEXT,
                  last_action TEXT,
                  last_decision_id TEXT,
                  known_location TEXT,
                  awaiting_slot TEXT,
                  preference_memory_json TEXT NOT NULL DEFAULT '{}',
                  durable_memory_json TEXT NOT NULL DEFAULT '[]',
                  character_state_json TEXT NOT NULL DEFAULT '{}',
                  recent_turns_json TEXT NOT NULL,
                  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._ensure_column("conversation_state", "rapport", "REAL NOT NULL DEFAULT 0.5")
            self._ensure_column("conversation_state", "boundary_pressure", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column("conversation_state", "directness_score", "REAL NOT NULL DEFAULT 0.5")
            self._ensure_column("conversation_state", "last_decision_id", "TEXT")
            self._ensure_column("conversation_state", "preference_memory_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column("conversation_state", "durable_memory_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("conversation_state", "character_state_json", "TEXT NOT NULL DEFAULT '{}'")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS message_log (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT NOT NULL,
                  user_text TEXT NOT NULL,
                  bot_text TEXT NOT NULL,
                  action TEXT NOT NULL,
                  decision_reason TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_trace (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  decision_id TEXT NOT NULL UNIQUE,
                  user_id TEXT NOT NULL,
                  input_text TEXT NOT NULL,
                  input_intent TEXT NOT NULL,
                  input_sentiment TEXT NOT NULL,
                  selected_action TEXT NOT NULL,
                  selected_reason TEXT NOT NULL,
                  decision_module TEXT NOT NULL DEFAULT 'daily_chat',
                  explanation_mode TEXT NOT NULL DEFAULT 'on_request_only',
                  classifier_evidence_json TEXT NOT NULL DEFAULT 'null',
                  reason_trace_json TEXT NOT NULL,
                  evidence_json TEXT NOT NULL,
                  constraints_json TEXT NOT NULL,
                  world_state_snapshot_json TEXT NOT NULL,
                  state_inference_trace_json TEXT NOT NULL DEFAULT '[]',
                  policy_candidates_json TEXT NOT NULL DEFAULT '[]',
                  counterfactuals_json TEXT NOT NULL DEFAULT '[]',
                  logic_chain_json TEXT NOT NULL DEFAULT '[]',
                  response_plan_json TEXT NOT NULL DEFAULT 'null',
                  output_text TEXT,
                  llm_used INTEGER NOT NULL DEFAULT 0,
                  llm_fallback_reason TEXT,
                  verification_issues_json TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._ensure_column("decision_trace", "decision_module", "TEXT NOT NULL DEFAULT 'daily_chat'")
            self._ensure_column("decision_trace", "explanation_mode", "TEXT NOT NULL DEFAULT 'on_request_only'")
            self._ensure_column("decision_trace", "classifier_evidence_json", "TEXT NOT NULL DEFAULT 'null'")
            self._ensure_column("decision_trace", "state_inference_trace_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("decision_trace", "policy_candidates_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("decision_trace", "counterfactuals_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("decision_trace", "logic_chain_json", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("decision_trace", "response_plan_json", "TEXT NOT NULL DEFAULT 'null'")
            self._ensure_column("decision_trace", "llm_used", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("decision_trace", "llm_fallback_reason", "TEXT")
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_message_log_user_id_id
                ON message_log (user_id, id DESC)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_decision_trace_user_id_id
                ON decision_trace (user_id, id DESC)
                """
            )

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        rows = self._connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_columns = {row["name"] for row in rows}
        if column_name in existing_columns:
            return
        self._connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )

    def _save_state(self, state: ConversationState) -> None:
        recent_turns_json = json.dumps(
            [self._turn_to_dict(turn) for turn in state.recent_turns],
            ensure_ascii=False,
        )
        preference_memory_json = json.dumps(state.preference_memory, ensure_ascii=False)
        durable_memory_json = json.dumps(
            [self._durable_memory_entry_to_dict(item) for item in state.durable_memory],
            ensure_ascii=False,
        )
        character_state_json = json.dumps(
            self._character_state_to_dict(state.character_state),
            ensure_ascii=False,
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO conversation_state (
                  user_id,
                  turn_count,
                  tension,
                  rapport,
                  boundary_pressure,
                  directness_score,
                  last_intent,
                  last_action,
                  last_decision_id,
                  known_location,
                  awaiting_slot,
                  preference_memory_json,
                  durable_memory_json,
                  character_state_json,
                  recent_turns_json,
                  updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                  turn_count = excluded.turn_count,
                  tension = excluded.tension,
                  rapport = excluded.rapport,
                  boundary_pressure = excluded.boundary_pressure,
                  directness_score = excluded.directness_score,
                  last_intent = excluded.last_intent,
                  last_action = excluded.last_action,
                  last_decision_id = excluded.last_decision_id,
                  known_location = excluded.known_location,
                  awaiting_slot = excluded.awaiting_slot,
                  preference_memory_json = excluded.preference_memory_json,
                  durable_memory_json = excluded.durable_memory_json,
                  character_state_json = excluded.character_state_json,
                  recent_turns_json = excluded.recent_turns_json,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (
                    state.user_id,
                    state.turn_count,
                    state.tension,
                    state.rapport,
                    state.boundary_pressure,
                    state.directness_score,
                    state.last_intent.value if state.last_intent else None,
                    state.last_action.value if state.last_action else None,
                    state.last_decision_id,
                    state.known_location,
                    state.awaiting_slot,
                    preference_memory_json,
                    durable_memory_json,
                    character_state_json,
                    recent_turns_json,
                ),
            )

    def _row_to_state(self, row: sqlite3.Row) -> ConversationState:
        preference_payload = _json_loads_or_default(row["preference_memory_json"], {})
        if not isinstance(preference_payload, dict):
            preference_payload = {}
        recent_turns_payload = _json_loads_or_default(row["recent_turns_json"], [])
        recent_turns: list[TurnRecord] = []
        if isinstance(recent_turns_payload, list):
            for item in recent_turns_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    recent_turns.append(self._dict_to_turn(item))
                except (KeyError, TypeError, ValueError):
                    continue
        return ConversationState(
            user_id=row["user_id"],
            turn_count=int(row["turn_count"]),
            tension=float(row["tension"]),
            rapport=float(row["rapport"] if row["rapport"] is not None else 0.5),
            boundary_pressure=float(row["boundary_pressure"] if row["boundary_pressure"] is not None else 0.0),
            directness_score=float(row["directness_score"] if row["directness_score"] is not None else 0.5),
            last_intent=_coerce_enum(Intent, row["last_intent"], Intent.UNKNOWN if row["last_intent"] else None),
            last_action=_coerce_enum(
                ActionType,
                row["last_action"],
                ActionType.CONTINUE_CONVERSATION if row["last_action"] else None,
            ),
            last_decision_id=row["last_decision_id"],
            known_location=row["known_location"],
            awaiting_slot=row["awaiting_slot"],
            preference_memory={
                str(key): str(value)
                for key, value in preference_payload.items()
            },
            durable_memory=self._row_to_durable_memory(row["durable_memory_json"]),
            recent_turns=recent_turns,
            character_state=self._row_to_character_state(row["character_state_json"]),
        )

    @staticmethod
    def _turn_to_dict(turn: TurnRecord) -> dict[str, str]:
        return {
            "user_text": _redact_persisted_text(turn.user_text) or "",
            "bot_text": _redact_persisted_text(turn.bot_text) or "",
            "action": turn.action.value,
            "decision_reason": _redact_persisted_text(turn.decision_reason) or "",
        }

    @staticmethod
    def _dict_to_turn(payload: dict[str, str]) -> TurnRecord:
        return TurnRecord(
            user_text=payload["user_text"],
            bot_text=payload["bot_text"],
            action=_coerce_enum(ActionType, payload.get("action"), ActionType.CONTINUE_CONVERSATION),
            decision_reason=payload["decision_reason"],
        )

    @staticmethod
    def _durable_memory_entry_to_dict(item: DurableMemoryEntry) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "bucket": item.bucket.value,
            "text": item.text,
            "source": item.source,
        }
        if item.captured_turn is not None:
            payload["captured_turn"] = item.captured_turn
        return payload

    @staticmethod
    def _character_state_to_dict(item: CharacterState) -> dict[str, object]:
        return {
            "mood": item.mood,
            "energy": item.energy,
            "curiosity": item.curiosity,
            "affinity": item.affinity,
            "pressure": item.pressure,
            "engagement": item.engagement,
            "topic_focus": item.topic_focus,
            "recent_topics": list(item.recent_topics),
            "recent_actions": list(item.recent_actions),
        }

    @staticmethod
    def _row_to_character_state(raw_json: str | None) -> CharacterState:
        payload = _json_loads_or_default(raw_json, {})
        if not isinstance(payload, dict):
            return CharacterState()
        return CharacterState(
            mood=str(payload.get("mood") or "relaxed"),
            energy=float(payload.get("energy", 0.6)),
            curiosity=float(payload.get("curiosity", 0.5)),
            affinity=float(payload.get("affinity", 0.5)),
            pressure=float(payload.get("pressure", 0.1)),
            engagement=float(payload.get("engagement", 0.5)),
            topic_focus=str(payload["topic_focus"]) if payload.get("topic_focus") else None,
            recent_topics=[str(item) for item in payload.get("recent_topics", []) if item],
            recent_actions=[str(item) for item in payload.get("recent_actions", []) if item],
        )

    @staticmethod
    def _reason_trace_to_dict(item: ReasonTraceEntry) -> dict[str, str | float | None]:
        return {
            "code": item.code,
            "summary": item.summary,
            "score": item.score,
        }

    @staticmethod
    def _scored_label_to_dict(item: ScoredLabel) -> dict[str, str | float]:
        return {
            "label": item.label,
            "score": item.score,
        }

    @staticmethod
    def _dict_to_scored_label(payload: dict[str, str | float]) -> ScoredLabel:
        return ScoredLabel(
            label=str(payload["label"]),
            score=float(payload["score"]),
        )

    @classmethod
    def _classifier_evidence_to_dict(
        cls,
        evidence: ClassifierEvidence | None,
    ) -> dict[str, object] | None:
        if evidence is None:
            return None
        return {
            "source": evidence.source,
            "chosen_reason": evidence.chosen_reason,
            "rule_hits": list(evidence.rule_hits),
            "top_scores": [cls._scored_label_to_dict(item) for item in evidence.top_scores],
            "override_applied": evidence.override_applied,
            "fallback_source": evidence.fallback_source,
            "fallback_intent": evidence.fallback_intent,
        }

    @classmethod
    def _dict_to_classifier_evidence(cls, payload: dict[str, object] | None) -> ClassifierEvidence | None:
        if payload is None:
            return None
        return ClassifierEvidence(
            source=str(payload["source"]),
            chosen_reason=str(payload["chosen_reason"]),
            rule_hits=[str(item) for item in payload.get("rule_hits", [])],
            top_scores=[
                cls._dict_to_scored_label(item)
                for item in payload.get("top_scores", [])
            ],
            override_applied=bool(payload.get("override_applied", False)),
            fallback_source=str(payload["fallback_source"]) if payload.get("fallback_source") is not None else None,
            fallback_intent=str(payload["fallback_intent"]) if payload.get("fallback_intent") is not None else None,
        )

    @staticmethod
    def _state_inference_to_dict(item: StateInferenceEntry) -> dict[str, object]:
        return {
            "field": item.field,
            "value": item.value,
            "reasons": list(item.reasons),
        }

    @staticmethod
    def _dict_to_state_inference(payload: dict[str, object]) -> StateInferenceEntry:
        return StateInferenceEntry(
            field=str(payload["field"]),
            value=payload.get("value"),
            reasons=[str(item) for item in payload.get("reasons", [])],
        )

    @staticmethod
    def _policy_candidate_to_dict(item: PolicyCandidate) -> dict[str, object]:
        return {
            "action": item.action.value,
            "score": item.score,
            "reason": item.reason,
            "score_breakdown": dict(item.score_breakdown),
        }

    @staticmethod
    def _dict_to_policy_candidate(payload: dict[str, object]) -> PolicyCandidate:
        return PolicyCandidate(
            action=ActionType(str(payload["action"])),
            score=float(payload["score"]),
            reason=str(payload["reason"]),
            score_breakdown={
                str(key): float(value)
                for key, value in dict(payload.get("score_breakdown", {})).items()
            },
        )

    @staticmethod
    def _counterfactual_to_dict(item: Counterfactual) -> dict[str, object]:
        return {
            "condition": item.condition,
            "predicted_action": item.predicted_action.value,
            "explanation": item.explanation,
        }

    @staticmethod
    def _dict_to_counterfactual(payload: dict[str, object]) -> Counterfactual:
        return Counterfactual(
            condition=str(payload["condition"]),
            predicted_action=ActionType(str(payload["predicted_action"])),
            explanation=str(payload["explanation"]),
        )

    @staticmethod
    def _logical_step_to_dict(item: LogicalStep) -> dict[str, object]:
        return {
            "step_type": item.step_type,
            "rule_id": item.rule_id,
            "premise": item.premise,
            "conclusion": item.conclusion,
            "score": item.score,
        }

    @staticmethod
    def _dict_to_logical_step(payload: dict[str, object]) -> LogicalStep:
        score = payload.get("score")
        return LogicalStep(
            step_type=str(payload["step_type"]),
            rule_id=str(payload["rule_id"]),
            premise=str(payload["premise"]),
            conclusion=str(payload["conclusion"]),
            score=float(score) if score is not None else None,
        )

    @staticmethod
    def _response_plan_to_dict(item: ResponsePlan | None) -> dict[str, object] | None:
        if item is None:
            return None
        return item.to_llm_payload()

    @staticmethod
    def _dict_to_response_plan(payload: dict[str, object] | None) -> ResponsePlan | None:
        if not isinstance(payload, dict):
            return None
        action = _coerce_enum(ActionType, payload.get("action"), None)
        if action is None:
            return None
        return ResponsePlan(
            action=action,
            stance=str(payload.get("stance") or "neutral"),
            anchor=str(payload.get("anchor") or ""),
            must_include=[str(item) for item in (payload.get("must_include") or []) if item is not None],
            avoid=[str(item) for item in (payload.get("avoid") or []) if item is not None],
            followup_policy=str(payload.get("followup_policy") or "auto"),
            sentence_budget=str(payload.get("sentence_budget") or "one_or_two_short"),
            tone=str(payload.get("tone") or "steady"),
            notes=[str(item) for item in (payload.get("notes") or []) if item is not None],
        )

    @staticmethod
    def _dict_to_reason_trace(payload: dict[str, str | float | None]) -> ReasonTraceEntry:
        score = payload.get("score")
        return ReasonTraceEntry(
            code=str(payload["code"]),
            summary=str(payload["summary"]),
            score=float(score) if score is not None else None,
        )

    @staticmethod
    def _row_to_durable_memory(raw_json: str | None) -> list[DurableMemoryEntry]:
        payload = _json_loads_or_default(raw_json, [])
        if not isinstance(payload, list):
            return []
        return normalize_durable_memory_entries(payload)

    @staticmethod
    def _classify_durable_memory_entry(text: str) -> DurableMemoryEntry:
        normalized = text.strip()
        return DurableMemoryEntry(
            bucket=classify_durable_memory_bucket(normalized),
            text=normalized,
            source="turn",
            captured_turn=None,
        )

    def _row_to_decision_trace(self, row: sqlite3.Row) -> DecisionTrace:
        classifier_payload = _json_loads_or_default(row["classifier_evidence_json"], None)
        if classifier_payload is not None and not isinstance(classifier_payload, dict):
            classifier_payload = None
        return DecisionTrace(
            decision_id=row["decision_id"],
            user_id=row["user_id"],
            input_text=row["input_text"],
            input_intent=_coerce_enum(Intent, row["input_intent"], Intent.UNKNOWN),
            input_sentiment=row["input_sentiment"],
            selected_action=_coerce_enum(ActionType, row["selected_action"], ActionType.CONTINUE_CONVERSATION),
            selected_reason=row["selected_reason"],
            decision_module=_coerce_enum(DecisionModule, row["decision_module"], DecisionModule.DAILY_CHAT),
            explanation_mode=_coerce_enum(
                ExplanationMode,
                row["explanation_mode"],
                ExplanationMode.ON_REQUEST_ONLY,
            ),
            classifier_evidence=self._dict_to_classifier_evidence(classifier_payload),
            reason_trace=_safe_convert_items(
                _json_loads_or_default(row["reason_trace_json"], []),
                self._dict_to_reason_trace,
            ),
            evidence=_json_loads_or_default(row["evidence_json"], []),
            constraints=_json_loads_or_default(row["constraints_json"], []),
            world_state_snapshot=_json_loads_or_default(row["world_state_snapshot_json"], {}),
            state_inference_trace=_safe_convert_items(
                _json_loads_or_default(row["state_inference_trace_json"], []),
                self._dict_to_state_inference,
            ),
            policy_candidates=_safe_convert_items(
                _json_loads_or_default(row["policy_candidates_json"], []),
                self._dict_to_policy_candidate,
            ),
            counterfactuals=_safe_convert_items(
                _json_loads_or_default(row["counterfactuals_json"], []),
                self._dict_to_counterfactual,
            ),
            logic_chain=_safe_convert_items(
                _json_loads_or_default(row["logic_chain_json"], []),
                self._dict_to_logical_step,
            ),
            response_plan=self._dict_to_response_plan(
                _json_loads_or_default(row["response_plan_json"], None)
            ),
            output_text=row["output_text"],
            llm_used=bool(row["llm_used"]),
            llm_fallback_reason=row["llm_fallback_reason"],
            verification_issues=_json_loads_or_default(row["verification_issues_json"], []),
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
