from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SQLITE_TIMEOUT_SECONDS = 10
SQLITE_BUSY_TIMEOUT_MS = 10_000
MEMORY_SEARCH_CANDIDATE_LIMIT = 128
MEMORY_RETRIEVAL_STOPWORDS = {
    "것",
    "그거",
    "그건",
    "그게",
    "나는",
    "너는",
    "사용자는",
    "white는",
    "black은",
    "한다",
    "한다며",
    "있다",
    "있다며",
    "좋아한다",
    "좋아한다며",
    "받을",
    "편이다",
    "편이라며",
    "가장",
    "그때",
    "네가",
    "다시",
    "사람",
    "사람이",
    "상황",
    "상황이",
    "싶어",
    "예전",
    "예전에",
    "예전에는",
    "요즘",
    "지금",
    "최근",
    "하고",
    "하는",
    "때마다",
}
MEMORY_RETRIEVAL_WEAK_SINGLETONS = {
    "스트레스",
    "기억",
    "요즘",
    "최근",
    "지금",
}


@dataclass(slots=True)
class StoredMessage:
    id: int
    guild_id: int | None
    channel_id: int
    user_id: int | None
    user_name: str | None
    role: str
    content: str
    created_at: str


@dataclass(slots=True)
class ConversationSummary:
    id: int
    guild_id: int | None
    channel_id: int
    summary_text: str
    source_until_message_id: int
    updated_at: str


@dataclass(slots=True)
class DurableMemory:
    id: int
    guild_id: int | None
    channel_id: int
    user_id: int | None
    user_name: str | None
    scope_key: str
    source_kind: str
    memory_kind: str
    memory_text: str
    source_message_id: int | None
    updated_at: str
    status: str = "active"
    fact_key: str | None = None
    fact_subject: str | None = None
    fact_value: str | None = None
    supersedes_id: int | None = None
    confidence: float = 1.0
    invalidated_at: str | None = None
    relevance_score: float = 0.0
    matched_terms: tuple[str, ...] = ()
    retrieval_rank: int = 0


@dataclass(slots=True, frozen=True)
class MemoryFact:
    fact_key: str
    subject: str
    value: str
    confidence: float = 0.86


class MemoryKind(str, Enum):
    PROFILE = "profile"
    ONGOING = "ongoing"
    OPEN_LOOP = "open_loop"
    EPISODIC = "episodic"
    OTHER = "other"


USER_MEMORY_MAX_TOTAL = 24
USER_MEMORY_MAX_PER_KIND = 8
MEMORY_RETENTION_DAYS_BY_KIND: dict[str, int] = {
    MemoryKind.PROFILE.value: 365,
    MemoryKind.ONGOING.value: 90,
    MemoryKind.OPEN_LOOP.value: 60,
    MemoryKind.EPISODIC.value: 30,
    MemoryKind.OTHER.value: 14,
}
MEMORY_CAPTURE_PROMPT_MARKERS = (
    "task:",
    "persona:",
    "action:",
    "rules:",
    "reply:",
    "system:",
    "assistant:",
    "developer:",
    "prompt:",
    "ignore previous",
    "이전 지시",
    "프롬프트",
    "```",
    "<system",
    "</system",
    "[system]",
)
MEMORY_CAPTURE_SENSITIVE_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b01[0-9][\s-]?\d{3,4}[\s-]?\d{4}\b"),
    re.compile(r"(?:api|access|refresh)[\s_-]*key", re.IGNORECASE),
    re.compile(r"(?:token|secret|password|passwd)", re.IGNORECASE),
    re.compile(r"(?:비밀번호|패스워드|암호|주민등록|여권번호|운전면허|계좌번호|카드번호|전화번호|이메일\s*주소|집\s*주소)"),
    re.compile(r"(?:자해|자살|죽고\s*싶|해치고\s*싶)"),
)
PERSISTENCE_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[redacted:email]"),
    (re.compile(r"\b01[0-9][\s-]?\d{3,4}[\s-]?\d{4}\b"), "[redacted:phone]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+\b", re.IGNORECASE), "Bearer [redacted]"),
    (
        re.compile(
            r"\b(?:api|access|refresh|auth|session)[\s:_-]*(?:key|token)\s*[:=]?\s*[A-Za-z0-9._~+/=-]{6,}",
            re.IGNORECASE,
        ),
        "[redacted:key]",
    ),
    (
        re.compile(
            r"\b(?:password|passwd|secret|비밀번호|암호)(?:는|은|이|가)?\s*[:=]?\s*[^\s,;]{4,}",
            re.IGNORECASE,
        ),
        "[redacted:secret]",
    ),
)


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        _best_effort_restrict_file_permissions(self._db_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=SQLITE_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    discord_message_id INTEGER,
                    guild_id INTEGER,
                    channel_id INTEGER NOT NULL,
                    user_id INTEGER,
                    user_name TEXT,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_discord_message_id
                    ON messages(discord_message_id);

                CREATE INDEX IF NOT EXISTS idx_messages_channel_id_id
                    ON messages(channel_id, id DESC);

                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER,
                    channel_id INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    source_until_message_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_until_message_id) REFERENCES messages(id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_summaries_channel_source_until
                    ON summaries(channel_id, source_until_message_id);

                CREATE INDEX IF NOT EXISTS idx_summaries_channel_source_desc
                    ON summaries(channel_id, source_until_message_id DESC);

                CREATE TABLE IF NOT EXISTS durable_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER,
                    channel_id INTEGER NOT NULL,
                    user_id INTEGER,
                    user_name TEXT,
                    scope_key TEXT NOT NULL,
                    source_kind TEXT NOT NULL DEFAULT 'user_note',
                    memory_kind TEXT NOT NULL DEFAULT 'episodic',
                    memory_text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    source_message_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'active',
                    fact_key TEXT,
                    fact_subject TEXT,
                    fact_value TEXT,
                    supersedes_id INTEGER,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    invalidated_at TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_message_id) REFERENCES messages(id),
                    FOREIGN KEY (supersedes_id) REFERENCES durable_memories(id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_durable_memories_scope_normalized
                    ON durable_memories(scope_key, source_kind, normalized_text);

                CREATE INDEX IF NOT EXISTS idx_durable_memories_channel_updated
                    ON durable_memories(channel_id, updated_at DESC);

                """
            )
            self._ensure_durable_memory_schema(connection)

    def _ensure_durable_memory_schema(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(durable_memories)").fetchall()
        }
        if "memory_kind" not in columns:
            connection.execute(
                "ALTER TABLE durable_memories ADD COLUMN memory_kind TEXT NOT NULL DEFAULT 'episodic'"
            )
        migrations = {
            "status": "ALTER TABLE durable_memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "fact_key": "ALTER TABLE durable_memories ADD COLUMN fact_key TEXT",
            "fact_subject": "ALTER TABLE durable_memories ADD COLUMN fact_subject TEXT",
            "fact_value": "ALTER TABLE durable_memories ADD COLUMN fact_value TEXT",
            "supersedes_id": "ALTER TABLE durable_memories ADD COLUMN supersedes_id INTEGER",
            "confidence": "ALTER TABLE durable_memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
            "invalidated_at": "ALTER TABLE durable_memories ADD COLUMN invalidated_at TEXT",
        }
        for column, statement in migrations.items():
            if column not in columns:
                connection.execute(statement)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_durable_memories_fact_active
                ON durable_memories(scope_key, source_kind, fact_key, fact_subject, status)
            """
        )
        self._backfill_memory_facts(connection)

    def _backfill_memory_facts(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT id, source_kind, memory_text
            FROM durable_memories
            WHERE fact_key IS NULL OR fact_subject IS NULL OR fact_value IS NULL
            """
        ).fetchall()
        for row in rows:
            fact = _extract_memory_fact(str(row["memory_text"]), source_kind=str(row["source_kind"]))
            if fact is None:
                continue
            connection.execute(
                """
                UPDATE durable_memories
                SET fact_key = ?,
                    fact_subject = ?,
                    fact_value = ?,
                    confidence = MAX(confidence, ?)
                WHERE id = ?
                """,
                (fact.fact_key, fact.subject, fact.value, fact.confidence, int(row["id"])),
            )

    def save_message(
        self,
        *,
        discord_message_id: int | None,
        guild_id: int | None,
        channel_id: int,
        user_id: int | None,
        user_name: str | None,
        role: str,
        content: str,
    ) -> int:
        normalized_content = _redact_persisted_text(content.strip())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO messages (
                    discord_message_id, guild_id, channel_id, user_id, user_name, role, content
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (discord_message_id, guild_id, channel_id, user_id, user_name, role, normalized_content),
            )
            if cursor.lastrowid:
                return int(cursor.lastrowid)

            if discord_message_id is None:
                row = connection.execute("SELECT last_insert_rowid() AS id").fetchone()
                return int(row["id"]) if row is not None else 0

            row = connection.execute(
                "SELECT id FROM messages WHERE discord_message_id = ?",
                (discord_message_id,),
            ).fetchone()
            return int(row["id"]) if row is not None else 0

    def load_recent_history(self, *, channel_id: int, limit: int) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, guild_id, channel_id, user_id, user_name, role, content, created_at
                FROM messages
                WHERE channel_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (channel_id, limit),
            ).fetchall()

        messages = [_row_to_message(row) for row in reversed(rows)]
        return [_to_history_message(message) for message in messages]

    def load_recent_messages(
        self,
        *,
        channel_id: int,
        limit: int,
        user_id: int | None = None,
        role: str | None = None,
    ) -> list[StoredMessage]:
        query = [
            """
            SELECT id, guild_id, channel_id, user_id, user_name, role, content, created_at
            FROM messages
            WHERE channel_id = ?
            """
        ]
        params: list[object] = [channel_id]
        if user_id is not None:
            query.append("AND user_id = ?")
            params.append(user_id)
        if role is not None:
            query.append("AND role = ?")
            params.append(role)
        query.append("ORDER BY id DESC LIMIT ?")
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute("\n".join(query), tuple(params)).fetchall()
        return [_row_to_message(row) for row in reversed(rows)]

    def count_messages(
        self,
        *,
        channel_id: int,
        user_id: int | None = None,
        role: str | None = None,
    ) -> int:
        query = [
            """
            SELECT COUNT(*) AS count
            FROM messages
            WHERE channel_id = ?
            """
        ]
        params: list[object] = [channel_id]
        if user_id is not None:
            query.append("AND user_id = ?")
            params.append(user_id)
        if role is not None:
            query.append("AND role = ?")
            params.append(role)
        with self._connect() as connection:
            row = connection.execute("\n".join(query), tuple(params)).fetchone()
        return int(row["count"]) if row is not None else 0

    def load_latest_summary(self, *, channel_id: int) -> ConversationSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, guild_id, channel_id, summary_text, source_until_message_id, updated_at
                FROM summaries
                WHERE channel_id = ?
                ORDER BY source_until_message_id DESC
                LIMIT 1
                """,
                (channel_id,),
            ).fetchone()

        return None if row is None else _row_to_summary(row)

    def count_messages_after(self, *, channel_id: int, after_message_id: int) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages
                WHERE channel_id = ? AND id > ?
                """,
                (channel_id, after_message_id),
            ).fetchone()

        return int(row["count"]) if row is not None else 0

    def load_messages_after(self, *, channel_id: int, after_message_id: int, limit: int) -> list[StoredMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, guild_id, channel_id, user_id, user_name, role, content, created_at
                FROM messages
                WHERE channel_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (channel_id, after_message_id, limit),
            ).fetchall()

        return [_row_to_message(row) for row in rows]

    def save_summary(
        self,
        *,
        guild_id: int | None,
        channel_id: int,
        summary_text: str,
        source_until_message_id: int,
    ) -> int:
        with self._connect() as connection:
            latest_row = connection.execute(
                """
                SELECT id, source_until_message_id
                FROM summaries
                WHERE channel_id = ?
                ORDER BY source_until_message_id DESC
                LIMIT 1
                """,
                (channel_id,),
            ).fetchone()
            if latest_row is not None and int(latest_row["source_until_message_id"]) >= source_until_message_id:
                return int(latest_row["id"])
            cursor = connection.execute(
                """
                INSERT INTO summaries (guild_id, channel_id, summary_text, source_until_message_id)
                VALUES (?, ?, ?, ?)
                """,
                    (guild_id, channel_id, _redact_persisted_text(summary_text.strip()), source_until_message_id),
                )
            summary_id = int(cursor.lastrowid)
            self._replace_summary_memories(
                connection,
                guild_id=guild_id,
                channel_id=channel_id,
                source_message_id=source_until_message_id,
                summary_text=summary_text,
            )
            return summary_id

    def save_user_memory_note(
        self,
        *,
        guild_id: int | None,
        channel_id: int,
        user_id: int,
        user_name: str | None,
        memory_text: str,
        source_message_id: int | None,
        memory_kind: str | None = None,
    ) -> int:
        normalized_memory = prepare_durable_memory_capture(memory_text, max_length=220) or ""
        if not normalized_memory:
            return 0

        scope_key = f"user:{user_id}"
        normalized_text = _normalize_memory_text(normalized_memory)
        selected_kind = _normalize_memory_kind(
            memory_kind or classify_durable_memory_kind(normalized_memory)
        )
        source_kind = "user_note"
        fact = _extract_memory_fact(normalized_memory, source_kind=source_kind)
        with self._connect() as connection:
            refreshed_id = self._refresh_same_fact_memory(
                connection,
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                user_name=user_name,
                scope_key=scope_key,
                source_kind=source_kind,
                memory_kind=selected_kind,
                memory_text=normalized_memory,
                normalized_text=normalized_text,
                source_message_id=source_message_id,
                fact=fact,
            )
            if refreshed_id:
                self._prune_user_memory_scope(connection, scope_key=scope_key)
                return refreshed_id
            supersedes_id = self._supersede_conflicting_fact_memories(
                connection,
                scope_key=scope_key,
                source_kind=source_kind,
                normalized_text=normalized_text,
                fact=fact,
            )
            cursor = connection.execute(
                """
                INSERT INTO durable_memories (
                    guild_id,
                    channel_id,
                    user_id,
                    user_name,
                    scope_key,
                    source_kind,
                    memory_kind,
                    memory_text,
                    normalized_text,
                    source_message_id,
                    status,
                    fact_key,
                    fact_subject,
                    fact_value,
                    supersedes_id,
                    confidence,
                    invalidated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(scope_key, source_kind, normalized_text) DO UPDATE SET
                    guild_id = excluded.guild_id,
                    channel_id = excluded.channel_id,
                    user_name = excluded.user_name,
                    memory_kind = excluded.memory_kind,
                    memory_text = excluded.memory_text,
                    source_message_id = COALESCE(excluded.source_message_id, durable_memories.source_message_id),
                    status = 'active',
                    fact_key = excluded.fact_key,
                    fact_subject = excluded.fact_subject,
                    fact_value = excluded.fact_value,
                    supersedes_id = COALESCE(excluded.supersedes_id, durable_memories.supersedes_id),
                    confidence = MAX(durable_memories.confidence, excluded.confidence),
                    invalidated_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    guild_id,
                    channel_id,
                    user_id,
                    user_name,
                    scope_key,
                    source_kind,
                    selected_kind,
                    normalized_memory,
                    normalized_text,
                    source_message_id,
                    fact.fact_key if fact else None,
                    fact.subject if fact else None,
                    fact.value if fact else None,
                    supersedes_id,
                    fact.confidence if fact else 1.0,
                ),
            )
            self._prune_user_memory_scope(connection, scope_key=scope_key)
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = connection.execute(
                """
                SELECT id
                FROM durable_memories
                WHERE scope_key = ? AND source_kind = 'user_note' AND normalized_text = ?
                """,
                (scope_key, normalized_text),
            ).fetchone()
            return int(row["id"]) if row is not None else 0

    def save_channel_memory_note(
        self,
        *,
        guild_id: int | None,
        channel_id: int,
        memory_text: str,
        source_message_id: int | None,
        memory_kind: str | None = None,
        source_kind: str = "channel_note",
    ) -> int:
        normalized_memory = prepare_durable_memory_capture(memory_text, max_length=220) or ""
        if not normalized_memory:
            return 0

        selected_source_kind = _normalize_memory_source_kind(source_kind)
        scope_key = f"channel:{channel_id}"
        normalized_text = _normalize_memory_text(normalized_memory)
        selected_kind = _normalize_memory_kind(
            memory_kind or classify_durable_memory_kind(normalized_memory)
        )
        fact = _extract_memory_fact(normalized_memory, source_kind=selected_source_kind)
        with self._connect() as connection:
            refreshed_id = self._refresh_same_fact_memory(
                connection,
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=None,
                user_name=None,
                scope_key=scope_key,
                source_kind=selected_source_kind,
                memory_kind=selected_kind,
                memory_text=normalized_memory,
                normalized_text=normalized_text,
                source_message_id=source_message_id,
                fact=fact,
            )
            if refreshed_id:
                return refreshed_id
            supersedes_id = self._supersede_conflicting_fact_memories(
                connection,
                scope_key=scope_key,
                source_kind=selected_source_kind,
                normalized_text=normalized_text,
                fact=fact,
            )
            cursor = connection.execute(
                """
                INSERT INTO durable_memories (
                    guild_id,
                    channel_id,
                    user_id,
                    user_name,
                    scope_key,
                    source_kind,
                    memory_kind,
                    memory_text,
                    normalized_text,
                    source_message_id,
                    status,
                    fact_key,
                    fact_subject,
                    fact_value,
                    supersedes_id,
                    confidence,
                    invalidated_at
                ) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(scope_key, source_kind, normalized_text) DO UPDATE SET
                    guild_id = excluded.guild_id,
                    channel_id = excluded.channel_id,
                    memory_kind = excluded.memory_kind,
                    memory_text = excluded.memory_text,
                    source_message_id = COALESCE(excluded.source_message_id, durable_memories.source_message_id),
                    status = 'active',
                    fact_key = excluded.fact_key,
                    fact_subject = excluded.fact_subject,
                    fact_value = excluded.fact_value,
                    supersedes_id = COALESCE(excluded.supersedes_id, durable_memories.supersedes_id),
                    confidence = MAX(durable_memories.confidence, excluded.confidence),
                    invalidated_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    guild_id,
                    channel_id,
                    scope_key,
                    selected_source_kind,
                    selected_kind,
                    normalized_memory,
                    normalized_text,
                    source_message_id,
                    fact.fact_key if fact else None,
                    fact.subject if fact else None,
                    fact.value if fact else None,
                    supersedes_id,
                    fact.confidence if fact else 1.0,
                ),
            )
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = connection.execute(
                """
                SELECT id
                FROM durable_memories
                WHERE scope_key = ? AND source_kind = ? AND normalized_text = ?
                """,
                (scope_key, selected_source_kind, normalized_text),
            ).fetchone()
            return int(row["id"]) if row is not None else 0

    def _refresh_same_fact_memory(
        self,
        connection: sqlite3.Connection,
        *,
        guild_id: int | None,
        channel_id: int,
        user_id: int | None,
        user_name: str | None,
        scope_key: str,
        source_kind: str,
        memory_kind: str,
        memory_text: str,
        normalized_text: str,
        source_message_id: int | None,
        fact: MemoryFact | None,
    ) -> int:
        if fact is None:
            return 0
        row = connection.execute(
            """
            SELECT id
            FROM durable_memories
            WHERE scope_key = ?
              AND source_kind = ?
              AND fact_key = ?
              AND fact_subject = ?
              AND fact_value = ?
              AND COALESCE(status, 'active') = 'active'
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (scope_key, source_kind, fact.fact_key, fact.subject, fact.value),
        ).fetchone()
        if row is None:
            return 0
        memory_id = int(row["id"])
        connection.execute(
            """
            UPDATE durable_memories
            SET guild_id = ?,
                channel_id = ?,
                user_id = ?,
                user_name = ?,
                memory_kind = ?,
                memory_text = ?,
                normalized_text = ?,
                source_message_id = COALESCE(?, source_message_id),
                status = 'active',
                fact_key = ?,
                fact_subject = ?,
                fact_value = ?,
                confidence = MAX(confidence, ?),
                invalidated_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                guild_id,
                channel_id,
                user_id,
                user_name,
                memory_kind,
                memory_text,
                normalized_text,
                source_message_id,
                fact.fact_key,
                fact.subject,
                fact.value,
                fact.confidence,
                memory_id,
            ),
        )
        return memory_id

    def _supersede_conflicting_fact_memories(
        self,
        connection: sqlite3.Connection,
        *,
        scope_key: str,
        source_kind: str,
        normalized_text: str,
        fact: MemoryFact | None,
    ) -> int | None:
        if fact is None:
            return None
        rows = connection.execute(
            """
            SELECT id, normalized_text, fact_value
            FROM durable_memories
            WHERE scope_key = ?
              AND source_kind = ?
              AND fact_key = ?
              AND fact_subject = ?
              AND COALESCE(status, 'active') = 'active'
            ORDER BY updated_at DESC, id DESC
            """,
            (scope_key, source_kind, fact.fact_key, fact.subject),
        ).fetchall()
        supersedes_id: int | None = None
        for row in rows:
            if str(row["normalized_text"]) == normalized_text:
                continue
            if str(row["fact_value"] or "") == fact.value:
                continue
            memory_id = int(row["id"])
            connection.execute(
                """
                UPDATE durable_memories
                SET status = 'superseded',
                    invalidated_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (memory_id,),
            )
            if supersedes_id is None:
                supersedes_id = memory_id
        return supersedes_id

    def search_durable_memories(
        self,
        *,
        channel_id: int,
        user_id: int | None,
        prompt: str,
        limit: int = 4,
    ) -> list[DurableMemory]:
        channel_scope_key = f"channel:{channel_id}"
        user_scope_key = f"user:{user_id}" if user_id is not None else None
        with self._connect() as connection:
            if user_scope_key is not None:
                rows = connection.execute(
                    """
                SELECT
                  id,
                  guild_id,
                  channel_id,
                  user_id,
                  user_name,
                  scope_key,
                  source_kind,
                  memory_kind,
                  memory_text,
                  source_message_id,
                  status,
                  fact_key,
                  fact_subject,
                  fact_value,
                  supersedes_id,
                  confidence,
                  invalidated_at,
                  updated_at
                FROM durable_memories
                WHERE (scope_key = ?
                   OR (scope_key = ? AND channel_id = ?))
                  AND COALESCE(status, 'active') = 'active'
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                    (
                        user_scope_key,
                        channel_scope_key,
                        channel_id,
                        MEMORY_SEARCH_CANDIDATE_LIMIT,
                    ),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                SELECT
                  id,
                  guild_id,
                  channel_id,
                  user_id,
                  user_name,
                  scope_key,
                  source_kind,
                  memory_kind,
                  memory_text,
                  source_message_id,
                  status,
                  fact_key,
                  fact_subject,
                  fact_value,
                  supersedes_id,
                  confidence,
                  invalidated_at,
                  updated_at
                FROM durable_memories
                WHERE scope_key = ? AND channel_id = ?
                  AND COALESCE(status, 'active') = 'active'
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                    (
                        channel_scope_key,
                        channel_id,
                        MEMORY_SEARCH_CANDIDATE_LIMIT,
                    ),
                ).fetchall()

        memories = []
        for rank, row in enumerate(rows):
            memory = _row_to_durable_memory(row)
            if not _memory_within_retention(memory):
                continue
            sanitized_text = prepare_durable_memory_capture(memory.memory_text, max_length=220)
            if not sanitized_text:
                continue
            memory.memory_text = sanitized_text
            memory.retrieval_rank = rank
            memories.append(memory)
        if not memories:
            return []

        prompt_tokens = _tokenize_memory_text(prompt)
        preferred_scope = f"user:{user_id}" if user_id is not None else None
        ranked = []
        for memory in memories:
            memory.relevance_score, matched_terms = self._memory_rank(
                memory,
                prompt_tokens=prompt_tokens,
                preferred_scope=preferred_scope,
            )
            if prompt_tokens and not matched_terms:
                continue
            if _weak_memory_match_only(matched_terms):
                continue
            memory.matched_terms = matched_terms
            ranked.append(memory)
        ranked.sort(key=lambda item: (item.relevance_score, -item.retrieval_rank), reverse=True)
        return ranked[:limit]

    def count_durable_memories_for_scope(self, *, scope_key: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_kind, COUNT(*) AS count
                FROM durable_memories
                WHERE scope_key = ?
                  AND COALESCE(status, 'active') = 'active'
                GROUP BY memory_kind
                """,
                (scope_key,),
            ).fetchall()
        counts = {str(row["memory_kind"]): int(row["count"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts

    def _replace_summary_memories(
        self,
        connection: sqlite3.Connection,
        *,
        guild_id: int | None,
        channel_id: int,
        source_message_id: int,
        summary_text: str,
    ) -> None:
        scope_key = f"channel:{channel_id}"
        connection.execute(
            """
            DELETE FROM durable_memories
            WHERE scope_key = ? AND source_kind = 'summary'
            """,
            (scope_key,),
        )

        bullets = _extract_summary_bullets(summary_text)
        if not bullets:
            return

        for bullet in bullets:
            prepared_bullet = prepare_durable_memory_capture(bullet, max_length=220)
            if not prepared_bullet:
                continue
            connection.execute(
                """
                INSERT INTO durable_memories (
                    guild_id,
                    channel_id,
                    user_id,
                    user_name,
                    scope_key,
                    source_kind,
                    memory_kind,
                    memory_text,
                    normalized_text,
                    source_message_id
                ) VALUES (?, ?, NULL, NULL, ?, 'summary', ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    channel_id,
                    scope_key,
                    classify_durable_memory_kind(prepared_bullet),
                    prepared_bullet,
                    _normalize_memory_text(prepared_bullet),
                    source_message_id,
                ),
            )

    @staticmethod
    def _prune_user_memory_scope(
        connection: sqlite3.Connection,
        *,
        scope_key: str,
    ) -> None:
        rows = connection.execute(
            """
            SELECT id, memory_kind
            FROM durable_memories
            WHERE scope_key = ? AND source_kind = 'user_note'
              AND COALESCE(status, 'active') = 'active'
            ORDER BY updated_at DESC, id DESC
            """,
            (scope_key,),
        ).fetchall()
        keep_ids: set[int] = set()
        kept_total = 0
        kept_per_kind: dict[str, int] = {}
        for row in rows:
            memory_id = int(row["id"])
            memory_kind = _normalize_memory_kind(str(row["memory_kind"]))
            if kept_total >= USER_MEMORY_MAX_TOTAL:
                continue
            if kept_per_kind.get(memory_kind, 0) >= USER_MEMORY_MAX_PER_KIND:
                continue
            keep_ids.add(memory_id)
            kept_total += 1
            kept_per_kind[memory_kind] = kept_per_kind.get(memory_kind, 0) + 1
        stale_ids = [int(row["id"]) for row in rows if int(row["id"]) not in keep_ids]
        if stale_ids:
            placeholders = ", ".join("?" for _ in stale_ids)
            connection.execute(
                f"DELETE FROM durable_memories WHERE id IN ({placeholders})",
                tuple(stale_ids),
            )

    @staticmethod
    def _memory_rank(
        memory: DurableMemory,
        *,
        prompt_tokens: set[str],
        preferred_scope: str | None,
    ) -> tuple[float, tuple[str, ...]]:
        memory_tokens = _tokenize_memory_text(memory.memory_text)
        matched_terms = tuple(sorted(prompt_tokens & memory_tokens)) if prompt_tokens else ()
        overlap = len(matched_terms)
        preferred = 2.0 if preferred_scope and memory.scope_key == preferred_scope else 0.0
        source_bonus = 1.5 if memory.source_kind == "user_note" else 0.5
        kind_bonus = _memory_kind_bonus(memory.memory_kind)
        recency_bonus = max(0.0, 1.0 - (memory.retrieval_rank / 64.0))
        score = (overlap * 3.0) + preferred + source_bonus + kind_bonus + recency_bonus
        return score, matched_terms


def _row_to_message(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        id=int(row["id"]),
        guild_id=int(row["guild_id"]) if row["guild_id"] is not None else None,
        channel_id=int(row["channel_id"]),
        user_id=int(row["user_id"]) if row["user_id"] is not None else None,
        user_name=str(row["user_name"]) if row["user_name"] is not None else None,
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
    )


def _row_to_summary(row: sqlite3.Row) -> ConversationSummary:
    return ConversationSummary(
        id=int(row["id"]),
        guild_id=int(row["guild_id"]) if row["guild_id"] is not None else None,
        channel_id=int(row["channel_id"]),
        summary_text=str(row["summary_text"]),
        source_until_message_id=int(row["source_until_message_id"]),
        updated_at=str(row["updated_at"]),
    )


def _row_to_durable_memory(row: sqlite3.Row) -> DurableMemory:
    row_keys = set(row.keys())
    return DurableMemory(
        id=int(row["id"]),
        guild_id=int(row["guild_id"]) if row["guild_id"] is not None else None,
        channel_id=int(row["channel_id"]),
        user_id=int(row["user_id"]) if row["user_id"] is not None else None,
        user_name=str(row["user_name"]) if row["user_name"] is not None else None,
        scope_key=str(row["scope_key"]),
        source_kind=str(row["source_kind"]),
        memory_kind=str(row["memory_kind"]) if "memory_kind" in row_keys else MemoryKind.EPISODIC.value,
        memory_text=str(row["memory_text"]),
        source_message_id=int(row["source_message_id"]) if row["source_message_id"] is not None else None,
        updated_at=str(row["updated_at"]),
        status=str(row["status"]) if "status" in row_keys and row["status"] is not None else "active",
        fact_key=str(row["fact_key"]) if "fact_key" in row_keys and row["fact_key"] is not None else None,
        fact_subject=str(row["fact_subject"]) if "fact_subject" in row_keys and row["fact_subject"] is not None else None,
        fact_value=str(row["fact_value"]) if "fact_value" in row_keys and row["fact_value"] is not None else None,
        supersedes_id=int(row["supersedes_id"])
        if "supersedes_id" in row_keys and row["supersedes_id"] is not None
        else None,
        confidence=float(row["confidence"]) if "confidence" in row_keys and row["confidence"] is not None else 1.0,
        invalidated_at=str(row["invalidated_at"])
        if "invalidated_at" in row_keys and row["invalidated_at"] is not None
        else None,
    )


def _to_history_message(message: StoredMessage) -> dict[str, str]:
    if message.role == "user":
        display_name = message.user_name or "Unknown user"
        return {
            "role": "user",
            "content": f"Discord user: {display_name}\n\nMessage:\n{message.content.strip()}",
        }
    return {
        "role": message.role,
        "content": message.content.strip(),
    }


def _normalize_memory_text(text: str) -> str:
    normalized = re.sub(r"^\s*[-*•]\s*", "", text.strip())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.casefold()


def _extract_memory_fact(text: str, *, source_kind: str = "") -> MemoryFact | None:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return None
    normalized = compact.casefold()
    nospace = re.sub(r"\s+", "", normalized)
    subject = _memory_fact_subject(normalized, source_kind=source_kind)

    if _contains_any(normalized, ("커피", "아메리카노", "카페인", "디카페인")) and _contains_any(
        nospace,
        ("잠을잘못", "잠잘못", "못잔다", "못자", "불면"),
    ):
        return MemoryFact("caffeine_sleep_effect", subject, "cannot_sleep_after_night_coffee")
    if _contains_any(normalized, ("커피", "아메리카노", "카페인")) and _contains_any(
        nospace,
        ("잠잘와", "잠을잘와", "잠을잘자", "잠을잘잔다", "잠잘잔다", "괜찮", "상관없", "문제없"),
    ):
        return MemoryFact("caffeine_sleep_effect", subject, "can_sleep_after_night_coffee")
    if "디카페인" in normalized and _contains_any(normalized, ("마신", "마셔", "커피")):
        return MemoryFact("caffeine_sleep_effect", subject, "decaf_only")

    if _contains_any(normalized, ("고양이", "길냥이")) and _contains_any(normalized, ("알러지", "알레르기")):
        if _contains_any(normalized, ("아니", "없", "괜찮", "나은", "사라졌")):
            return MemoryFact("cat_allergy", subject, "no_cat_allergy")
        return MemoryFact("cat_allergy", subject, "has_cat_allergy")

    if _contains_any(normalized, ("단거", "단 것", "단 음식", "마카롱", "케이크", "디저트")):
        if _contains_any(normalized, ("안 좋아", "별로", "싫", "줄였", "끊었", "좋아하지")):
            return MemoryFact("sweet_preference", subject, "dislikes_sweets")
        if _contains_any(normalized, ("좋아", "먹고 싶", "당겨", "샀", "찾")):
            return MemoryFact("sweet_preference", subject, "likes_sweets")

    if "스트레스" in normalized and _contains_any(normalized, ("매운", "엽떡", "불닭", "떡볶이")):
        if _contains_any(normalized, ("끊었", "줄였", "안 먹", "못 먹", "피하", "피한다", "피하고")):
            return MemoryFact("stress_food_response", subject, "avoids_spicy_when_stressed")
        return MemoryFact("stress_food_response", subject, "spicy_food_when_stressed")

    if "다이어트" in normalized:
        if _contains_any(normalized, ("그만", "끝", "안 해", "안한다", "안 한다", "하지 않", "포기", "접었")):
            return MemoryFact("diet_status", subject, "not_dieting")
        return MemoryFact("diet_status", subject, "dieting")

    if "이직" in normalized and _contains_any(normalized, ("고민", "생각", "결정", "하기로", "안 하", "하지 않", "포기", "접었")):
        if _contains_any(normalized, ("안 하", "하지 않", "포기", "접었", "고민하지")):
            return MemoryFact("job_change_status", subject, "no_job_change")
        if _contains_any(normalized, ("결정", "하기로", "간다", "옮기")):
            return MemoryFact("job_change_status", subject, "decided_job_change")
        return MemoryFact("job_change_status", subject, "considering_job_change")

    if "글램핑" in normalized:
        if _contains_any(normalized, ("취소", "못 가", "안 가", "미뤘")):
            return MemoryFact("glamping_plan", subject, "glamping_cancelled")
        if _contains_any(normalized, ("계획", "예정", "가기로", "한 달 전부터")):
            return MemoryFact("glamping_plan", subject, "has_glamping_plan")

    if "영양제" in normalized:
        if _contains_any(normalized, ("끊었", "안 먹", "먹지", "그만")):
            return MemoryFact("supplement_use", subject, "no_supplement")
        if _contains_any(normalized, ("먹", "덜 피곤", "추천")):
            return MemoryFact("supplement_use", subject, "uses_supplement")

    if "하와이" in normalized and _contains_any(normalized, ("살", "꿈", "가고 싶")):
        if _contains_any(normalized, ("아니", "포기", "접", "안 가")):
            return MemoryFact("hawaii_dream", subject, "no_hawaii_dream")
        return MemoryFact("hawaii_dream", subject, "wants_hawaii")

    if "스팀 게임" in normalized or "steam" in normalized:
        if _contains_any(normalized, ("안 해", "접었", "관심 없", "관심이 없", "흥미 없")):
            return MemoryFact("steam_game_interest", subject, "not_interested_steam_game")
        if _contains_any(normalized, ("빠져", "시작", "관심", "하는 중")):
            return MemoryFact("steam_game_interest", subject, "interested_steam_game")

    return None


def _memory_fact_subject(text: str, *, source_kind: str) -> str:
    normalized_source = str(source_kind or "").casefold()
    if normalized_source.startswith("speaker_profile:"):
        speaker = normalized_source.split(":", 1)[1].strip()
        if speaker:
            return speaker
    if "black" in text or "블랙" in text:
        return "black"
    if "white" in text or "화이트" in text:
        return "white"
    return "user"


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def prepare_durable_memory_capture(text: str, *, max_length: int) -> str | None:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return None
    normalized = compact.casefold()
    if any(marker in normalized for marker in MEMORY_CAPTURE_PROMPT_MARKERS):
        return None
    if any(pattern.search(compact) for pattern in MEMORY_CAPTURE_SENSITIVE_PATTERNS):
        return None
    if len(compact) <= max_length:
        return compact
    truncated = compact[:max_length].rsplit(" ", 1)[0].strip()
    return (truncated or compact[:max_length].strip()) + "..."


def _redact_persisted_text(text: str) -> str:
    redacted = text
    for pattern, replacement in PERSISTENCE_REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _best_effort_restrict_file_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        return


def _extract_summary_bullets(summary_text: str) -> list[str]:
    bullets: list[str] = []
    for line in summary_text.splitlines():
        cleaned = re.sub(r"^\s*[-*•]\s*", "", line).strip()
        if not cleaned or cleaned == "No durable memory yet.":
            continue
        bullets.append(cleaned)
    return bullets


def _tokenize_memory_text(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text.casefold()):
        for variant in _memory_token_variants(token):
            if len(variant) >= 2 and variant not in MEMORY_RETRIEVAL_STOPWORDS:
                tokens.add(variant)
    return tokens


def _memory_token_variants(token: str) -> set[str]:
    compact = str(token or "").strip().casefold()
    if not compact:
        return set()
    variants = {compact}
    for suffix in (
        "하기로",
        "하러",
        "하고",
        "처럼",
        "보다",
        "부터",
        "까지",
        "에게",
        "한테",
        "에서",
        "으로",
        "라고",
        "이라며",
        "라며",
        "이라서",
        "라서",
        "이라도",
        "라도",
        "이라면",
        "라면",
        "이랑",
        "랑",
        "와",
        "과",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "도",
        "만",
        "에",
        "의",
        "로",
    ):
        if compact.endswith(suffix) and len(compact) > len(suffix) + 1:
            variants.add(compact[: -len(suffix)])
    return variants


def _weak_memory_match_only(matched_terms: tuple[str, ...]) -> bool:
    return len(matched_terms) == 1 and matched_terms[0] in MEMORY_RETRIEVAL_WEAK_SINGLETONS


def classify_durable_memory_kind(text: str) -> str:
    normalized = text.casefold()
    scores: dict[MemoryKind, float] = {
        MemoryKind.PROFILE: 0.0,
        MemoryKind.ONGOING: 0.0,
        MemoryKind.OPEN_LOOP: 0.0,
        MemoryKind.EPISODIC: 0.0,
        MemoryKind.OTHER: 0.0,
    }

    _add_keyword_score(
        scores,
        MemoryKind.PROFILE,
        normalized,
        (
            "좋아해",
            "좋아한다",
            "좋아하지",
            "싫어해",
            "싫어한다",
            "선호",
            "취향",
            "평소",
            "항상",
            "늘 ",
            "원래",
            "즐겨",
            "취미",
            "성향",
            "편이다",
            "찾는 편",
            "찾는다",
            "알러지",
            "알레르기",
            "못 잔다",
        ),
    )
    _add_keyword_score(
        scores,
        MemoryKind.ONGOING,
        normalized,
        (
            "요즘",
            "최근",
            "지금",
            "계속",
            "아직",
            "진행",
            "하는 중",
            "고민 중",
            "준비 중",
            "먹고 있다",
            "신경 쓰",
            "중이야",
            "중입니다",
            "한창",
            "자주",
            "반복",
        ),
    )
    _add_keyword_score(
        scores,
        MemoryKind.OPEN_LOOP,
        normalized,
        (
            "기다",
            "연락",
            "답장",
            "미뤄",
            "남아",
            "남은",
            "해야",
            "할 일",
            "숙제",
            "고민",
            "계획",
            "예정",
            "가기로",
            "결정",
            "걱정",
            "불안",
            "확인",
        ),
    )
    _add_keyword_score(
        scores,
        MemoryKind.EPISODIC,
        normalized,
        (
            "어제",
            "그때",
            "지난",
            "지난번",
            "방금",
            "오늘",
            "이번",
            "오전에",
            "오후에",
            "오랜만",
            "그날",
        ),
    )

    best_kind = max(
        scores.items(),
        key=lambda item: (item[1], _memory_kind_preference(item[0])),
    )[0]
    if scores[best_kind] <= 0:
        return MemoryKind.OTHER.value
    return best_kind.value


def _normalize_memory_kind(kind: str | MemoryKind) -> str:
    if isinstance(kind, MemoryKind):
        return kind.value
    normalized = kind.strip().casefold()
    for candidate in MemoryKind:
        if candidate.value == normalized:
            return candidate.value
    return MemoryKind.OTHER.value


def _normalize_memory_source_kind(source_kind: str) -> str:
    normalized = (source_kind or "channel_note").strip().casefold()
    if not re.fullmatch(r"[0-9a-z_.:-]{1,64}", normalized):
        return "channel_note"
    return normalized


def _memory_kind_bonus(memory_kind: str) -> float:
    normalized = _normalize_memory_kind(memory_kind)
    bonuses = {
        MemoryKind.PROFILE.value: 2.2,
        MemoryKind.ONGOING.value: 2.5,
        MemoryKind.OPEN_LOOP.value: 2.8,
        MemoryKind.EPISODIC.value: 1.4,
        MemoryKind.OTHER.value: 0.2,
    }
    return bonuses.get(normalized, 0.0)


def _memory_kind_preference(kind: MemoryKind) -> int:
    order = {
        MemoryKind.OPEN_LOOP: 4,
        MemoryKind.ONGOING: 3,
        MemoryKind.PROFILE: 2,
        MemoryKind.EPISODIC: 1,
        MemoryKind.OTHER: 0,
    }
    return order.get(kind, 0)


def _memory_within_retention(memory: DurableMemory) -> bool:
    max_age_days = MEMORY_RETENTION_DAYS_BY_KIND.get(
        _normalize_memory_kind(memory.memory_kind),
        MEMORY_RETENTION_DAYS_BY_KIND[MemoryKind.OTHER.value],
    )
    updated_at = _parse_memory_timestamp(memory.updated_at)
    if updated_at is None:
        return True
    return (datetime.now(timezone.utc) - updated_at) <= timedelta(days=max_age_days)


def _parse_memory_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    candidate = raw.strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _add_keyword_score(
    scores: dict[MemoryKind, float],
    kind: MemoryKind,
    normalized: str,
    keywords: tuple[str, ...],
) -> None:
    for keyword in keywords:
        if keyword in normalized:
            scores[kind] += 1.0
