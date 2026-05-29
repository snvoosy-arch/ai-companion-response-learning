from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - exercised on POSIX
    import fcntl


class SingletonLockError(RuntimeError):
    pass


@dataclass(slots=True)
class SingletonStartupLock:
    path: str | Path
    enabled: bool
    label: str = "white"
    acquired: bool = False
    _file: BinaryIO | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def acquire(self) -> None:
        if not self.enabled:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            _lock_file(handle)
            _write_lock_metadata(handle, label=self.label)
        except OSError as exc:
            handle.close()
            raise SingletonLockError(_format_lock_error(self.path, self.label)) from exc

        self._file = handle
        self.acquired = True

    def close(self) -> None:
        if self._file is None:
            return
        try:
            _unlock_file(self._file)
        finally:
            self._file.close()
            self._file = None
            self.acquired = False


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        handle.write(b"\0")
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_lock_metadata(handle: BinaryIO, *, label: str) -> None:
    handle.seek(0)
    handle.truncate(0)
    payload = (
        f"bot_name={label}\n"
        f"pid={os.getpid()}\n"
        f"started_ts={int(time.time())}\n"
    ).encode("utf-8")
    handle.write(payload)
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        pass


def _format_lock_error(path: Path, label: str) -> str:
    return (
        f"{label}가 이미 실행 중인 것 같아. "
        f"startup lock을 잡지 못했어: {path} "
        f"(필요하면 BOT_STARTUP_SINGLETON_ENABLED=false로 끌 수 있어)"
    )
