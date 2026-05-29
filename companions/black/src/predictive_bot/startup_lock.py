from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - exercised on POSIX
    import fcntl


@dataclass(slots=True)
class StartupLock:
    path: Path
    enabled: bool
    acquired: bool = False
    _file: TextIO | None = None

    def release(self) -> None:
        if self._file is None:
            return
        try:
            _unlock_file(self._file)
        finally:
            self._file.close()
            self._file = None
            self.acquired = False

    def __enter__(self) -> "StartupLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


class StartupLockError(RuntimeError):
    pass


def acquire_startup_lock(
    path: str | Path,
    *,
    enabled: bool = True,
    bot_name: str = "predictive-discord-bot",
) -> StartupLock:
    lock_path = Path(path)
    if not enabled:
        return StartupLock(path=lock_path, enabled=False, acquired=False)

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        _lock_file(handle)
        _write_lock_metadata(handle, bot_name=bot_name)
    except Exception as exc:
        try:
            handle.close()
        finally:
            pass
        raise StartupLockError(_format_lock_error(lock_path, bot_name)) from exc

    return StartupLock(path=lock_path, enabled=True, acquired=True, _file=handle)


def _lock_file(handle: TextIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        handle.write(b"\0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: TextIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_lock_metadata(handle: TextIO, *, bot_name: str) -> None:
    handle.seek(0)
    handle.truncate(0)
    payload = (
        f"bot_name={bot_name}\n"
        f"pid={os.getpid()}\n"
        f"started_ts={int(time.time())}\n"
    )
    handle.write(payload.encode("utf-8"))
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def _format_lock_error(path: Path, bot_name: str) -> str:
    return (
        f"{bot_name}이 이미 실행 중인 것 같아. "
        f"startup lock을 잡지 못했어: {path} "
        f"(필요하면 BOT_STARTUP_LOCK_ENABLED=false로 끌 수 있어)"
    )
