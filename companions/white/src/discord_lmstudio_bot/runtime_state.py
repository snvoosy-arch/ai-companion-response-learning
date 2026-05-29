from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SQLITE_TIMEOUT_SECONDS = 10
SQLITE_BUSY_TIMEOUT_MS = 10_000


STATUS_ONLINE = "online"
STATUS_OFFLINE = "offline"
CLAIM_MODE_AUTO = "auto"
CLAIM_MODE_MANUAL = "manual"


@dataclass(slots=True)
class ClaimResult:
    acquired: bool
    conflict: bool = False
    holder_name: str | None = None
    holder_channel_id: str | None = None
    holder_reason: str | None = None


@dataclass(slots=True)
class RuntimePresence:
    bot_name: str
    project_name: str
    display_name: str | None
    discord_user_id: str | None
    online: bool
    busy: bool
    claim_mode: str | None
    claim_reason: str | None
    current_channel_id: str | None
    current_guild_id: str | None
    current_owner_id: str | None
    current_owner_name: str | None
    last_heartbeat_ts: int
    last_active_ts: int | None
    claim_started_ts: int | None
    claim_expires_ts: int | None


class SharedRuntimeStateStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        _best_effort_restrict_file_permissions(self._db_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path, timeout=SQLITE_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS bot_runtime_state (
                    bot_name TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    discord_user_id TEXT,
                    display_name TEXT,
                    advertised_status TEXT NOT NULL,
                    last_heartbeat_ts INTEGER NOT NULL,
                    last_active_ts INTEGER,
                    current_channel_id TEXT,
                    current_guild_id TEXT,
                    current_owner_id TEXT,
                    current_owner_name TEXT,
                    claim_mode TEXT,
                    claim_reason TEXT,
                    claim_started_ts INTEGER,
                    claim_expires_ts INTEGER,
                    updated_ts INTEGER NOT NULL
                );
                """
            )

    def heartbeat(
        self,
        *,
        bot_name: str,
        project_name: str,
        discord_user_id: str | None,
        display_name: str | None,
        status: str = STATUS_ONLINE,
    ) -> None:
        now_ts = _now_ts()
        with self._connect() as connection:
            self._clear_expired_claims(connection, now_ts)
            self._ensure_row(
                connection,
                bot_name=bot_name,
                project_name=project_name,
                discord_user_id=discord_user_id,
                display_name=display_name,
                now_ts=now_ts,
            )
            connection.execute(
                """
                UPDATE bot_runtime_state
                SET project_name = ?,
                    discord_user_id = ?,
                    display_name = ?,
                    advertised_status = ?,
                    last_heartbeat_ts = ?,
                    updated_ts = ?
                WHERE bot_name = ?
                """,
                (
                    project_name,
                    discord_user_id,
                    display_name,
                    status,
                    now_ts,
                    now_ts,
                    bot_name,
                ),
            )

    def claim(
        self,
        *,
        bot_name: str,
        project_name: str,
        discord_user_id: str | None,
        display_name: str | None,
        owner_id: str | None,
        owner_name: str | None,
        channel_id: str | None,
        guild_id: str | None,
        reason: str,
        ttl_seconds: int,
        mode: str,
    ) -> ClaimResult:
        now_ts = _now_ts()
        ttl_seconds = max(1, int(ttl_seconds))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._clear_expired_claims(connection, now_ts)
            self._ensure_row(
                connection,
                bot_name=bot_name,
                project_name=project_name,
                discord_user_id=discord_user_id,
                display_name=display_name,
                now_ts=now_ts,
            )
            row = connection.execute(
                """
                SELECT current_owner_id, current_owner_name, current_channel_id, claim_mode, claim_reason, claim_expires_ts
                FROM bot_runtime_state
                WHERE bot_name = ?
                """,
                (bot_name,),
            ).fetchone()
            if row is not None and row["claim_expires_ts"] is not None and int(row["claim_expires_ts"]) > now_ts:
                same_owner = owner_id is not None and str(row["current_owner_id"]) == owner_id
                if row["claim_mode"] == CLAIM_MODE_MANUAL and not same_owner:
                    return ClaimResult(
                        acquired=False,
                        conflict=True,
                        holder_name=str(row["current_owner_name"]) if row["current_owner_name"] else None,
                        holder_channel_id=str(row["current_channel_id"]) if row["current_channel_id"] else None,
                        holder_reason=str(row["claim_reason"]) if row["claim_reason"] else None,
                    )

            expires_ts = now_ts + ttl_seconds
            connection.execute(
                """
                UPDATE bot_runtime_state
                SET project_name = ?,
                    discord_user_id = ?,
                    display_name = ?,
                    advertised_status = ?,
                    last_heartbeat_ts = ?,
                    last_active_ts = ?,
                    current_channel_id = ?,
                    current_guild_id = ?,
                    current_owner_id = ?,
                    current_owner_name = ?,
                    claim_mode = ?,
                    claim_reason = ?,
                    claim_started_ts = ?,
                    claim_expires_ts = ?,
                    updated_ts = ?
                WHERE bot_name = ?
                """,
                (
                    project_name,
                    discord_user_id,
                    display_name,
                    STATUS_ONLINE,
                    now_ts,
                    now_ts,
                    channel_id,
                    guild_id,
                    owner_id,
                    owner_name,
                    mode,
                    reason,
                    now_ts,
                    expires_ts,
                    now_ts,
                    bot_name,
                ),
            )
        return ClaimResult(acquired=True)

    def release(self, *, bot_name: str, owner_id: str | None = None, force: bool = False) -> bool:
        now_ts = _now_ts()
        with self._connect() as connection:
            self._clear_expired_claims(connection, now_ts)
            row = connection.execute(
                """
                SELECT current_owner_id, claim_mode
                FROM bot_runtime_state
                WHERE bot_name = ?
                """,
                (bot_name,),
            ).fetchone()
            if row is None:
                return False
            current_owner_id = str(row["current_owner_id"]) if row["current_owner_id"] is not None else None
            if not force and row["claim_mode"] == CLAIM_MODE_MANUAL and owner_id and current_owner_id not in {None, owner_id}:
                return False

            connection.execute(
                """
                UPDATE bot_runtime_state
                SET current_channel_id = NULL,
                    current_guild_id = NULL,
                    current_owner_id = NULL,
                    current_owner_name = NULL,
                    claim_mode = NULL,
                    claim_reason = NULL,
                    claim_started_ts = NULL,
                    claim_expires_ts = NULL,
                    updated_ts = ?
                WHERE bot_name = ?
                """,
                (now_ts, bot_name),
            )
        return True

    def mark_offline(
        self,
        *,
        bot_name: str,
        project_name: str,
        discord_user_id: str | None,
        display_name: str | None,
    ) -> None:
        now_ts = _now_ts()
        with self._connect() as connection:
            self._ensure_row(
                connection,
                bot_name=bot_name,
                project_name=project_name,
                discord_user_id=discord_user_id,
                display_name=display_name,
                now_ts=now_ts,
            )
            connection.execute(
                """
                UPDATE bot_runtime_state
                SET project_name = ?,
                    discord_user_id = ?,
                    display_name = ?,
                    advertised_status = ?,
                    last_heartbeat_ts = ?,
                    updated_ts = ?
                WHERE bot_name = ?
                """,
                (
                    project_name,
                    discord_user_id,
                    display_name,
                    STATUS_OFFLINE,
                    now_ts,
                    now_ts,
                    bot_name,
                ),
            )

    def list_presences(self, *, online_timeout_seconds: float) -> list[RuntimePresence]:
        now_ts = _now_ts()
        with self._connect() as connection:
            self._clear_expired_claims(connection, now_ts)
            rows = connection.execute(
                """
                SELECT
                    bot_name,
                    project_name,
                    discord_user_id,
                    display_name,
                    advertised_status,
                    last_heartbeat_ts,
                    last_active_ts,
                    current_channel_id,
                    current_guild_id,
                    current_owner_id,
                    current_owner_name,
                    claim_mode,
                    claim_reason,
                    claim_started_ts,
                    claim_expires_ts
                FROM bot_runtime_state
                ORDER BY bot_name ASC
                """
            ).fetchall()
        return [_row_to_presence(row, now_ts=now_ts, online_timeout_seconds=online_timeout_seconds) for row in rows]

    def _ensure_row(
        self,
        connection: sqlite3.Connection,
        *,
        bot_name: str,
        project_name: str,
        discord_user_id: str | None,
        display_name: str | None,
        now_ts: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO bot_runtime_state (
                bot_name,
                project_name,
                discord_user_id,
                display_name,
                advertised_status,
                last_heartbeat_ts,
                updated_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bot_name) DO NOTHING
            """,
            (
                bot_name,
                project_name,
                discord_user_id,
                display_name,
                STATUS_OFFLINE,
                now_ts,
                now_ts,
            ),
        )

    def _clear_expired_claims(self, connection: sqlite3.Connection, now_ts: int) -> None:
        connection.execute(
            """
            UPDATE bot_runtime_state
            SET current_channel_id = NULL,
                current_guild_id = NULL,
                current_owner_id = NULL,
                current_owner_name = NULL,
                claim_mode = NULL,
                claim_reason = NULL,
                claim_started_ts = NULL,
                claim_expires_ts = NULL,
                updated_ts = ?
            WHERE claim_expires_ts IS NOT NULL
              AND claim_expires_ts <= ?
            """,
            (now_ts, now_ts),
        )


class RuntimePresenceClient:
    def __init__(
        self,
        *,
        enabled: bool,
        db_path: str | Path,
        bot_name: str,
        project_name: str,
        heartbeat_interval_seconds: float,
        online_timeout_seconds: float,
        auto_claim_ttl_seconds: int,
        manual_claim_ttl_seconds: int,
    ) -> None:
        self.enabled = enabled
        self.bot_name = bot_name
        self.project_name = project_name
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.online_timeout_seconds = online_timeout_seconds
        self.auto_claim_ttl_seconds = auto_claim_ttl_seconds
        self.manual_claim_ttl_seconds = manual_claim_ttl_seconds
        self._store = SharedRuntimeStateStore(db_path) if enabled else None
        self._discord_user_id: str | None = None
        self._display_name: str | None = None

    def activate(self, *, discord_user_id: int | str, display_name: str | None) -> None:
        if not self.enabled or self._store is None:
            return
        self._discord_user_id = str(discord_user_id)
        self._display_name = display_name
        self._store.heartbeat(
            bot_name=self.bot_name,
            project_name=self.project_name,
            discord_user_id=self._discord_user_id,
            display_name=self._display_name,
        )

    def heartbeat(self) -> None:
        if not self.enabled or self._store is None:
            return
        self._store.heartbeat(
            bot_name=self.bot_name,
            project_name=self.project_name,
            discord_user_id=self._discord_user_id,
            display_name=self._display_name,
        )

    def note_auto_activity(
        self,
        *,
        owner_id: int | str | None,
        owner_name: str | None,
        channel_id: int | str | None,
        guild_id: int | str | None,
        reason: str,
    ) -> ClaimResult | None:
        if not self.enabled or self._store is None:
            return None
        return self._store.claim(
            bot_name=self.bot_name,
            project_name=self.project_name,
            discord_user_id=self._discord_user_id,
            display_name=self._display_name,
            owner_id=None if owner_id is None else str(owner_id),
            owner_name=owner_name,
            channel_id=None if channel_id is None else str(channel_id),
            guild_id=None if guild_id is None else str(guild_id),
            reason=reason,
            ttl_seconds=self.auto_claim_ttl_seconds,
            mode=CLAIM_MODE_AUTO,
        )

    def claim_manual(
        self,
        *,
        owner_id: int | str | None,
        owner_name: str | None,
        channel_id: int | str | None,
        guild_id: int | str | None,
        reason: str,
    ) -> ClaimResult | None:
        if not self.enabled or self._store is None:
            return None
        return self._store.claim(
            bot_name=self.bot_name,
            project_name=self.project_name,
            discord_user_id=self._discord_user_id,
            display_name=self._display_name,
            owner_id=None if owner_id is None else str(owner_id),
            owner_name=owner_name,
            channel_id=None if channel_id is None else str(channel_id),
            guild_id=None if guild_id is None else str(guild_id),
            reason=reason,
            ttl_seconds=self.manual_claim_ttl_seconds,
            mode=CLAIM_MODE_MANUAL,
        )

    def release(self, *, owner_id: int | str | None = None, force: bool = False) -> bool:
        if not self.enabled or self._store is None:
            return False
        return self._store.release(
            bot_name=self.bot_name,
            owner_id=None if owner_id is None else str(owner_id),
            force=force,
        )

    def mark_offline(self) -> None:
        if not self.enabled or self._store is None:
            return
        self._store.mark_offline(
            bot_name=self.bot_name,
            project_name=self.project_name,
            discord_user_id=self._discord_user_id,
            display_name=self._display_name,
        )

    def snapshot(self) -> list[RuntimePresence]:
        if not self.enabled or self._store is None:
            return []
        return self._store.list_presences(online_timeout_seconds=self.online_timeout_seconds)

    def format_status_report(self) -> str:
        return format_runtime_presence_report(self.snapshot())


def format_runtime_presence_report(presences: list[RuntimePresence]) -> str:
    if not presences:
        return "공유 runtime 상태가 아직 비어 있어요."

    now_ts = _now_ts()
    lines = ["공유 runtime 상태"]
    for presence in presences:
        online_text = "online" if presence.online else "offline"
        busy_text = "busy" if presence.busy else "idle"
        lines.append(f"- {presence.bot_name} [{online_text}/{busy_text}]")
        details: list[str] = [f"project={presence.project_name}"]
        if presence.display_name:
            details.append(f"display={presence.display_name}")
        details.append(f"heartbeat={_format_age(now_ts - presence.last_heartbeat_ts)} ago")
        if presence.current_owner_name and presence.busy:
            details.append(f"owner={presence.current_owner_name}")
        if presence.current_channel_id and presence.busy:
            details.append(f"channel={presence.current_channel_id}")
        if presence.claim_mode and presence.busy:
            details.append(f"claim={presence.claim_mode}")
        if presence.claim_reason and presence.busy:
            details.append(f"reason={presence.claim_reason}")
        lines.append(f"  {' | '.join(details)}")
    return "\n".join(lines)


def _row_to_presence(
    row: sqlite3.Row,
    *,
    now_ts: int,
    online_timeout_seconds: float,
) -> RuntimePresence:
    last_heartbeat_ts = int(row["last_heartbeat_ts"])
    claim_expires_ts = int(row["claim_expires_ts"]) if row["claim_expires_ts"] is not None else None
    online = str(row["advertised_status"]) == STATUS_ONLINE and last_heartbeat_ts >= int(now_ts - online_timeout_seconds)
    busy = claim_expires_ts is not None and claim_expires_ts > now_ts
    return RuntimePresence(
        bot_name=str(row["bot_name"]),
        project_name=str(row["project_name"]),
        display_name=str(row["display_name"]) if row["display_name"] is not None else None,
        discord_user_id=str(row["discord_user_id"]) if row["discord_user_id"] is not None else None,
        online=online,
        busy=busy,
        claim_mode=str(row["claim_mode"]) if row["claim_mode"] is not None else None,
        claim_reason=str(row["claim_reason"]) if row["claim_reason"] is not None else None,
        current_channel_id=str(row["current_channel_id"]) if row["current_channel_id"] is not None else None,
        current_guild_id=str(row["current_guild_id"]) if row["current_guild_id"] is not None else None,
        current_owner_id=str(row["current_owner_id"]) if row["current_owner_id"] is not None else None,
        current_owner_name=str(row["current_owner_name"]) if row["current_owner_name"] is not None else None,
        last_heartbeat_ts=last_heartbeat_ts,
        last_active_ts=int(row["last_active_ts"]) if row["last_active_ts"] is not None else None,
        claim_started_ts=int(row["claim_started_ts"]) if row["claim_started_ts"] is not None else None,
        claim_expires_ts=claim_expires_ts,
    )


def _format_age(delta_seconds: int) -> str:
    delta_seconds = max(0, int(delta_seconds))
    if delta_seconds < 60:
        return f"{delta_seconds}s"
    if delta_seconds < 3600:
        return f"{delta_seconds // 60}m"
    return f"{delta_seconds // 3600}h"


def _now_ts() -> int:
    return int(time.time())


def _best_effort_restrict_file_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        return
